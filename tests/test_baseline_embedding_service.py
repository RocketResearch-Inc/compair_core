from __future__ import annotations

import hashlib
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from compair_core.baseline_embedding.cache import (
    BaselineModelCacheError,
    cleanup_incomplete_fetches,
    fetch_baseline_model,
    snapshot_path,
    verify_baseline_model,
)
from compair_core.baseline_embedding.manifest import (
    BaselineModelArtifact,
    BaselineModelManifest,
    BaselineModelManifestError,
    _load_manifest_bytes,
    load_baseline_model_manifest,
)
from compair_core.baseline_embedding.model_cli import main as model_cli_main
from compair_core.baseline_embedding.service import (
    BaselineEmbeddingRuntime,
    BaselineEmbeddingServiceError,
    _load_fastembed_model,
    create_app,
)
from compair_core.baseline_embedding.service import main as service_main


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _manifest(files: dict[str, bytes], *, dimension: int = 2) -> BaselineModelManifest:
    artifacts = tuple(
        BaselineModelArtifact(path=name, size=len(value), sha256=_sha256(value))
        for name, value in sorted(files.items())
    )
    return BaselineModelManifest(
        schema_version="compair-baseline-model-manifest.v1",
        profile="baseline-v1",
        logical_model="BAAI/bge-small-en-v1.5",
        artifact_repository="qdrant/bge-small-en-v1.5-onnx-Q",
        revision="52398278842ec682c6f32300af41344b1c0b0bb2",
        dimension=dimension,
        dtype="float32",
        normalization="none",
        contract_version="baseline-embedding-http.v1",
        provider="baseline_http_v1",
        runtime_packages=(
            ("fastembed", "0.8.0"),
            ("huggingface-hub", "1.27.0"),
            ("numpy", "2.4.6"),
            ("onnxruntime", "1.28.0"),
            ("tokenizers", "0.23.1"),
        ),
        total_bytes=sum(len(value) for value in files.values()),
        files=artifacts,
        embedding_identity_fingerprint="1" * 64,
        tokenizer_fingerprint="2" * 64,
        model_artifact_fingerprint="3" * 64,
        licenses=(("Apache-2.0", "https://example.invalid/license"),),
        manifest_fingerprint="4" * 64,
    )


def _downloader(files: dict[str, bytes], calls: list[tuple[str, str, str]]):
    def download(
        repository: str, revision: str, filename: str, local_dir: Path
    ) -> Path:
        calls.append((repository, revision, filename))
        path = local_dir / filename
        path.write_bytes(files[filename])
        return path

    return download


def _fetch_fixture(tmp_path: Path, files: dict[str, bytes]):
    manifest = _manifest(files)
    calls: list[tuple[str, str, str]] = []
    result = fetch_baseline_model(
        tmp_path / "cache",
        manifest=manifest,
        downloader=_downloader(files, calls),
    )
    return manifest, calls, result


def test_installed_manifest_freezes_prior_live_artifacts() -> None:
    manifest = load_baseline_model_manifest()

    assert manifest.profile == "baseline-v1"
    assert manifest.logical_model == "BAAI/bge-small-en-v1.5"
    assert manifest.artifact_repository == "qdrant/bge-small-en-v1.5-onnx-Q"
    assert manifest.revision == "52398278842ec682c6f32300af41344b1c0b0bb2"
    assert manifest.dimension == 384
    assert manifest.dtype == "float32"
    assert manifest.normalization == "none"
    assert manifest.contract_version == "baseline-embedding-http.v1"
    assert manifest.runtime_package_versions == {
        "fastembed": "0.8.0",
        "huggingface-hub": "1.27.0",
        "numpy": "2.4.6",
        "onnxruntime": "1.28.0",
        "tokenizers": "0.23.1",
    }
    assert manifest.total_bytes == 67_179_163
    assert {
        artifact.path: (artifact.size, artifact.sha256) for artifact in manifest.files
    } == {
        "config.json": (
            706,
            "13582bcf2effc85b7bf3d3f5532e686bc1c9ce86bb009d10f0ec33cbe92299dd",
        ),
        "model_optimized.onnx": (
            66_465_124,
            "51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431",
        ),
        "special_tokens_map.json": (
            695,
            "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
        ),
        "tokenizer.json": (
            711_396,
            "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        ),
        "tokenizer_config.json": (
            1_242,
            "0b29c7bfc889e53b36d9dd3e686dd4300f6525110eaa98c76a5dafceb2029f53",
        ),
    }
    assert manifest.manifest_fingerprint == (
        "429e20eed22b1fbd1dc3788969be6241e49d8dccf685a7d246d283c5d91d37de"
    )


@pytest.mark.parametrize(
    "raw",
    (
        b'{"profile":"baseline-v1","profile":"shadow"}',
        b'{"files":[{"path":"../model.onnx"}]}',
        b'{"dimension":NaN}',
        b"\xff",
    ),
)
def test_manifest_parser_rejects_duplicate_traversal_nonfinite_and_non_utf8(
    raw: bytes,
) -> None:
    with pytest.raises(BaselineModelManifestError):
        _load_manifest_bytes(raw)


def test_fetch_is_explicit_verified_atomic_offline_and_idempotent(
    tmp_path: Path,
) -> None:
    files = {"a.json": b"alpha", "model.onnx": b"model-bytes"}
    manifest, calls, result = _fetch_fixture(tmp_path, files)
    target = snapshot_path(tmp_path / "cache", manifest=manifest)

    assert result.downloaded is True
    assert target.is_dir()
    assert tuple(path.name for path in target.iterdir()) == manifest.expected_paths
    assert all(path.is_file() and not path.is_symlink() for path in target.iterdir())
    assert all(path.stat().st_mode & 0o077 == 0 for path in target.iterdir())
    assert calls == [
        (manifest.artifact_repository, manifest.revision, artifact.path)
        for artifact in manifest.files
    ]

    replay = fetch_baseline_model(
        tmp_path / "cache",
        manifest=manifest,
        downloader=lambda *_args: pytest.fail("verified replay attempted network"),
    )
    assert replay.downloaded is False
    assert (
        verify_baseline_model(tmp_path / "cache", manifest=manifest).safe_summary()[
            "status"
        ]
        == "verified"
    )


@pytest.mark.parametrize("mutation", ("truncate", "corrupt", "unexpected"))
def test_corrupt_or_partial_publication_fails_closed_without_overwrite(
    tmp_path: Path,
    mutation: str,
) -> None:
    files = {"a.json": b"alpha", "model.onnx": b"model-bytes"}
    manifest, _, _ = _fetch_fixture(tmp_path, files)
    target = snapshot_path(tmp_path / "cache", manifest=manifest)
    if mutation == "truncate":
        (target / "model.onnx").write_bytes(b"model")
    elif mutation == "corrupt":
        (target / "model.onnx").write_bytes(b"wrong-bytes")
    else:
        (target / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(BaselineModelCacheError) as exc_info:
        fetch_baseline_model(
            tmp_path / "cache",
            manifest=manifest,
            downloader=lambda *_args: pytest.fail("invalid target was overwritten"),
        )

    assert exc_info.value.code in {
        "baseline_model_artifact_mismatch",
        "baseline_model_artifact_hash_mismatch",
        "baseline_model_manifest_mismatch",
    }


def test_symlink_cache_and_download_attacks_are_rejected(tmp_path: Path) -> None:
    files = {"a.json": b"alpha", "model.onnx": b"model-bytes"}
    manifest = _manifest(files)
    outside = tmp_path / "outside"
    outside.mkdir()
    cache = tmp_path / "cache"
    cache.symlink_to(outside, target_is_directory=True)

    with pytest.raises(BaselineModelCacheError, match="baseline_model_cache_unsafe"):
        fetch_baseline_model(
            cache, manifest=manifest, downloader=_downloader(files, [])
        )

    safe_cache = tmp_path / "safe-cache"

    def malicious(
        _repository: str,
        _revision: str,
        filename: str,
        local_dir: Path,
    ) -> Path:
        source = tmp_path / f"outside-{filename}"
        source.write_bytes(files[filename])
        target = local_dir / filename
        target.symlink_to(source)
        return target

    with pytest.raises(BaselineModelCacheError, match="baseline_model_download_unsafe"):
        fetch_baseline_model(safe_cache, manifest=manifest, downloader=malicious)
    assert not snapshot_path(safe_cache, manifest=manifest).exists()


def test_offline_verify_rejects_symlinked_cache_or_profile_root(
    tmp_path: Path,
) -> None:
    files = {"a.json": b"alpha", "model.onnx": b"model-bytes"}
    manifest, _, _ = _fetch_fixture(tmp_path, files)
    real_cache = tmp_path / "cache"

    linked_cache = tmp_path / "linked-cache"
    linked_cache.symlink_to(real_cache, target_is_directory=True)
    with pytest.raises(BaselineModelCacheError, match="baseline_model_cache_unsafe"):
        verify_baseline_model(linked_cache, manifest=manifest)

    alternate_root = tmp_path / "alternate-cache"
    alternate_root.mkdir(mode=0o700)
    (alternate_root / manifest.profile).symlink_to(
        real_cache / manifest.profile,
        target_is_directory=True,
    )
    with pytest.raises(BaselineModelCacheError, match="baseline_model_cache_unsafe"):
        verify_baseline_model(alternate_root, manifest=manifest)


def test_interrupted_fetch_is_ineligible_until_explicit_cleanup(tmp_path: Path) -> None:
    files = {"a.json": b"alpha", "model.onnx": b"model-bytes"}
    manifest = _manifest(files)
    calls = 0

    def interrupted(
        _repository: str,
        _revision: str,
        filename: str,
        local_dir: Path,
    ) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        path = local_dir / filename
        path.write_bytes(files[filename])
        return path

    with pytest.raises(KeyboardInterrupt):
        fetch_baseline_model(
            tmp_path / "cache",
            manifest=manifest,
            downloader=interrupted,
        )
    with pytest.raises(BaselineModelCacheError, match="baseline_model_absent"):
        verify_baseline_model(tmp_path / "cache", manifest=manifest)

    assert cleanup_incomplete_fetches(tmp_path / "cache", manifest=manifest) == 1
    assert cleanup_incomplete_fetches(tmp_path / "cache", manifest=manifest) == 0


def test_concurrent_fetches_serialize_and_converge(tmp_path: Path) -> None:
    files = {"a.json": b"alpha", "model.onnx": b"model-bytes"}
    manifest = _manifest(files)
    calls: list[str] = []
    calls_lock = threading.Lock()

    def download(
        _repository: str,
        _revision: str,
        filename: str,
        local_dir: Path,
    ) -> Path:
        with calls_lock:
            calls.append(filename)
        time.sleep(0.02)
        path = local_dir / filename
        path.write_bytes(files[filename])
        return path

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(
            pool.map(
                lambda _value: fetch_baseline_model(
                    tmp_path / "cache",
                    manifest=manifest,
                    downloader=download,
                ),
                range(2),
            )
        )

    assert sorted(result.downloaded for result in results) == [False, True]
    assert calls == list(manifest.expected_paths)
    verify_baseline_model(tmp_path / "cache", manifest=manifest)


def test_fetch_errors_and_cli_output_do_not_disclose_credentials_or_paths(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "hf_secret_model_token"
    cache = tmp_path / secret
    files = {"a.json": b"alpha"}
    manifest = _manifest(files)

    def failing(*_args):
        raise RuntimeError(f"https://user:{secret}@example.invalid/{secret}")

    with pytest.raises(BaselineModelCacheError) as exc_info:
        fetch_baseline_model(cache, manifest=manifest, downloader=failing)
    assert str(exc_info.value) == "baseline_model_fetch_failed"
    assert secret not in repr(exc_info.value)

    exit_code = model_cli_main(("verify", "baseline-v1", "--cache-dir", str(cache)))
    output = capsys.readouterr()
    assert exit_code == 3
    assert secret not in output.out
    assert secret not in output.err


class _FakeModel:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, ...]] = []

    def embed(self, documents, *, batch_size):
        submitted = tuple(documents)
        self.calls.append(submitted)
        assert 1 <= batch_size <= len(submitted)
        return (np.asarray(vector, dtype=np.float32) for vector in self.vectors)


def _ready_runtime(
    tmp_path: Path,
    *,
    vectors: list[list[float]] | None = None,
) -> tuple[BaselineEmbeddingRuntime, _FakeModel]:
    files = {"a.json": b"alpha", "model.onnx": b"model-bytes"}
    manifest, _, _ = _fetch_fixture(tmp_path, files)
    model = _FakeModel(vectors or [[3.0, 4.0], [-0.0, 7.25]])
    runtime = BaselineEmbeddingRuntime(
        cache_root=str(tmp_path / "cache"),
        manifest=manifest,
        model_factory=lambda _verified, _threads: model,
    )
    return runtime, model


def _request_payload(
    runtime: BaselineEmbeddingRuntime, texts: list[str]
) -> dict[str, object]:
    return {**runtime.identity(), "texts": texts}


def test_service_health_and_embeddings_preserve_order_and_float32(
    tmp_path: Path,
) -> None:
    runtime, model = _ready_runtime(tmp_path)
    with TestClient(create_app(runtime)) as client:
        health = client.get("/v1/health")
        response = client.post(
            "/v1/embeddings",
            json=_request_payload(runtime, ["first", "second"]),
        )
        assert client.get("/").status_code == 404
        assert client.get("/openapi.json").status_code == 404

    assert health.status_code == 200
    assert health.json() == {"status": "ok", **runtime.identity()}
    assert response.status_code == 200
    assert response.json()["vectors"] == [[3.0, 4.0], [-0.0, 7.25]]
    assert model.calls == [("first", "second")]


def test_service_absent_model_is_safe_and_does_not_read_text(tmp_path: Path) -> None:
    manifest = _manifest({"a.json": b"alpha"})
    runtime = BaselineEmbeddingRuntime(
        cache_root=str(tmp_path / "missing"),
        manifest=manifest,
        model_factory=lambda *_args: pytest.fail("missing model was loaded"),
    )
    sentinel = "private source sentinel"
    with TestClient(create_app(runtime)) as client:
        health = client.get("/v1/health")
        response = client.post(
            "/v1/embeddings",
            json=_request_payload(runtime, [sentinel]),
        )

    assert health.status_code == 503
    assert health.json() == {
        "status": "unavailable",
        "reason": "baseline_model_absent",
    }
    assert response.status_code == 503
    assert sentinel not in response.text


def test_service_rejects_identity_before_model_processing(tmp_path: Path) -> None:
    runtime, model = _ready_runtime(tmp_path, vectors=[[1.0, 2.0]])
    payload = _request_payload(runtime, ["private identity sentinel"])
    payload["revision"] = "wrong-revision"

    with TestClient(create_app(runtime)) as client:
        response = client.post("/v1/embeddings", json=payload)

    assert response.status_code == 400
    assert response.json() == {"error": "baseline_embedding_revision_mismatch"}
    assert model.calls == []
    assert "private identity sentinel" not in response.text


@pytest.mark.parametrize(
    ("body", "headers", "expected_status"),
    (
        (b"{}", {"content-type": "text/plain"}, 415),
        (b'{"model":"a","model":"b"}', {"content-type": "application/json"}, 400),
        (b'{"dimension":NaN}', {"content-type": "application/json"}, 400),
        (b"\xff", {"content-type": "application/json"}, 400),
    ),
)
def test_service_strict_json_and_content_type(
    tmp_path: Path,
    body: bytes,
    headers: dict[str, str],
    expected_status: int,
) -> None:
    runtime, model = _ready_runtime(tmp_path, vectors=[[1.0, 2.0]])
    with TestClient(create_app(runtime, max_request_bytes=1024)) as client:
        response = client.post("/v1/embeddings", content=body, headers=headers)

    assert response.status_code == expected_status
    assert model.calls == []


def test_service_enforces_text_and_request_limits_without_echo(tmp_path: Path) -> None:
    runtime, model = _ready_runtime(tmp_path, vectors=[[1.0, 2.0]])
    sentinel = "private-oversized-sentinel"
    app = create_app(
        runtime,
        max_request_bytes=1024,
        max_text_items=1,
        max_text_bytes=8,
        max_total_text_bytes=8,
    )
    with TestClient(app) as client:
        oversized_text = client.post(
            "/v1/embeddings",
            json=_request_payload(runtime, [sentinel]),
        )
        too_many = client.post(
            "/v1/embeddings",
            json=_request_payload(runtime, ["a", "b"]),
        )
        request_too_large = client.post(
            "/v1/embeddings",
            content=b"{" + b" " * 1024 + b"}",
            headers={"content-type": "application/json"},
        )

    assert oversized_text.status_code == 413
    assert too_many.status_code == 400
    assert request_too_large.status_code == 413
    assert sentinel not in oversized_text.text
    assert model.calls == []


@pytest.mark.parametrize(
    ("vectors", "reason"),
    (
        ([[1.0]], "baseline_embedding_dimension_mismatch"),
        ([[float("nan"), 0.0]], "baseline_embedding_vector_nonfinite"),
        ([[float("inf"), 0.0]], "baseline_embedding_vector_nonfinite"),
        ([], "baseline_embedding_vector_count_mismatch"),
    ),
)
def test_service_fails_closed_on_invalid_vectors(
    tmp_path: Path,
    vectors: list[list[float]],
    reason: str,
) -> None:
    runtime, _ = _ready_runtime(tmp_path, vectors=vectors)
    with TestClient(create_app(runtime)) as client:
        response = client.post(
            "/v1/embeddings",
            json=_request_payload(runtime, ["safe probe"]),
        )

    assert response.status_code == 503
    assert response.json() == {"error": reason}


def test_service_cli_rejects_non_loopback_bind() -> None:
    with pytest.raises(SystemExit) as exc_info:
        service_main(("--host", "0.0.0.0"))

    assert exc_info.value.code == 2


def test_real_model_loader_rejects_runtime_version_drift_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _, result = _fetch_fixture(tmp_path, {"a.json": b"alpha"})
    expected_versions = manifest.runtime_package_versions

    monkeypatch.setattr(
        "compair_core.baseline_embedding.service.version",
        lambda package: (
            "unexpected-version"
            if package == "onnxruntime"
            else expected_versions[package]
        ),
    )

    with pytest.raises(
        BaselineEmbeddingServiceError,
        match="baseline_embedding_runtime_version_mismatch",
    ):
        _load_fastembed_model(result.model, 1)


def test_cache_permissions_remain_private(tmp_path: Path) -> None:
    files = {"a.json": b"alpha"}
    manifest, _, _ = _fetch_fixture(tmp_path, files)
    cache = tmp_path / "cache"
    target = snapshot_path(cache, manifest=manifest)

    assert stat.S_IMODE(cache.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert stat.S_IMODE((target / "a.json").stat().st_mode) == 0o600
    assert not any(path.is_symlink() for path in cache.rglob("*"))


def test_pyproject_exposes_optional_extra_and_installed_entry_points() -> None:
    import tomllib

    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["optional-dependencies"]["baseline-embedding"] == [
        "fastembed==0.8.0",
        "huggingface-hub==1.27.0",
        "numpy==2.4.6",
        "onnxruntime==1.28.0",
        "tokenizers==0.23.1",
    ]
    assert pyproject["project"]["scripts"]["compair-core-models"] == (
        "compair_core.baseline_embedding.model_cli:main"
    )
    assert pyproject["project"]["scripts"]["compair-core-embedding-service"] == (
        "compair_core.baseline_embedding.service:main"
    )


def test_package_tree_contains_manifest_but_no_model_or_cache_payload() -> None:
    package_root = Path(__file__).resolve().parents[1] / "compair_core"
    service_root = package_root / "baseline_embedding"

    assert (service_root / "baseline-v1.manifest.json").is_file()
    forbidden_suffixes = {".bin", ".onnx", ".safetensors"}
    assert not [
        path
        for path in package_root.rglob("*")
        if path.is_file() and path.suffix.lower() in forbidden_suffixes
    ]
    assert not [
        path
        for path in package_root.rglob("*")
        if path.is_dir() and path.name in {".cache", ".staging", "models"}
    ]
