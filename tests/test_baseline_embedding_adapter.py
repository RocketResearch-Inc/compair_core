from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from pydantic import ValidationError

from compair_core import db as core_db
from compair_core.compair.retrieval.corpus import (
    CorpusFileInput,
    RetrievalBaselineIndexPublication,
    ensure_retrieval_corpus_schema,
)
from compair_core.compair.retrieval.embedding import (
    BASELINE_EMBEDDING_HTTP_CONTRACT,
    BASELINE_EMBEDDING_HTTP_PROVIDER,
    BaselineEmbeddingAdapterError,
    BaselineEmbeddingCapabilityStatus,
    BaselineEmbeddingConfig,
    HTTPBaselineEmbeddingAdapter,
    assess_baseline_embedding,
    build_configured_baseline_index,
    configured_baseline_embedding_adapter,
    create_configured_persistent_baseline_retriever,
)
from compair_core.compair.retrieval.indexing import BaselineIndexBuildError
from compair_core.compair.retrieval.ingestion import (
    CorpusIngestionService,
    CorpusRepositoryInput,
    CorpusSnapshotInput,
)
from compair_core.compair.retrieval.types import (
    RetrievalQueryOrigin,
    RetrievalRequest,
    RetrievalStatus,
)
from compair_core.server.routers.capabilities import (
    capabilities as capabilities_endpoint,
)
from compair_core.server.routers.capabilities import health as health_endpoint
from compair_core.server.settings import Settings

MODEL = "BAAI/bge-small-en-v1.5"
REVISION = "fixture-pinned-revision"
SCOPE_KEY = "group:baseline-http-adapter"


def _settings(**overrides):
    values = {
        "baseline_embedding_provider": "http",
        "baseline_embedding_endpoint": "http://127.0.0.1:9010",
        "baseline_embedding_model": MODEL,
        "baseline_embedding_revision": REVISION,
        "baseline_embedding_dimension": 2,
        "baseline_embedding_timeout_seconds": 2.0,
        "baseline_embedding_batch_size": 2,
        "baseline_embedding_allow_insecure_loopback": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _identity_payload(**overrides) -> dict[str, object]:
    payload: dict[str, object] = {
        "contract_version": BASELINE_EMBEDDING_HTTP_CONTRACT,
        "provider": BASELINE_EMBEDDING_HTTP_PROVIDER,
        "model": MODEL,
        "revision": REVISION,
        "dimension": 2,
    }
    payload.update(overrides)
    return payload


def _client_factory(handler):
    def create_client() -> httpx.Client:
        return httpx.Client(
            transport=httpx.MockTransport(handler),
            timeout=2.0,
        )

    return create_client


def _ready_handler(vector_for_text=None, requests=None):
    vector_for_text = vector_for_text or (lambda text: [0.25, -0.5])
    requests = requests if requests is not None else []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET" and request.url.path.endswith("/v1/health"):
            return httpx.Response(
                200,
                json={"status": "ok", **_identity_payload()},
            )
        assert request.method == "POST"
        assert request.url.path.endswith("/v1/embeddings")
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                **_identity_payload(),
                "vectors": [vector_for_text(text) for text in body["texts"]],
            },
        )

    return handler


def _sessions(tmp_path: Path):
    engine = core_db.create_engine(f"sqlite:///{tmp_path / 'baseline-http.db'}")
    ensure_retrieval_corpus_schema(engine)
    return engine, core_db.sessionmaker(engine, expire_on_commit=False)


def _ingest(SessionMaker):
    files = (
        CorpusFileInput.supported_text(
            repository_id="repo-peer",
            repository_name="peer",
            relative_path="src/alpha.txt",
            content="alpha alpha beta",
        ),
        CorpusFileInput.supported_text(
            repository_id="repo-peer",
            repository_name="peer",
            relative_path="src/beta.txt",
            content="beta gamma",
        ),
    )
    snapshot = CorpusSnapshotInput.create(
        scope_key=SCOPE_KEY,
        generation_version="generation-1",
        changed_repository=CorpusRepositoryInput(
            repository_id="repo-changed",
            repository_name="changed",
            expected_file_count=0,
            repository_revision="changed-revision-1",
            document_id="document-changed",
            document_revision="document-changed-revision-1",
        ),
        sibling_repositories=(
            CorpusRepositoryInput(
                repository_id="repo-peer",
                repository_name="peer",
                expected_file_count=2,
                repository_revision="peer-revision-1",
                document_id="document-peer",
                document_revision="document-peer-revision-1",
            ),
        ),
        files=files,
        producer_id="trusted-baseline-http-test",
        producer_version="1.0",
        snapshot_id="snapshot-baseline-http-1",
        source_revision="changed-revision-1",
        source_manifest_hash=hashlib.sha256(b"baseline-http-source").hexdigest(),
    )
    return CorpusIngestionService(SessionMaker).ingest(snapshot)


def _request(query: str = "alpha change") -> RetrievalRequest:
    return RetrievalRequest(
        request_id="baseline-http-request",
        changed_repository=None,
        repository_roots=(),
        corpus_version="generation-1",
        retrieval_query=query,
        retrieval_query_origin=RetrievalQueryOrigin.EXPLICIT,
        corpus_complete=True,
        corpus_scope_key=SCOPE_KEY,
        changed_repository_id="repo-changed",
    )


def test_baseline_embedding_settings_default_disabled_and_bounded() -> None:
    settings = Settings()

    assert settings.baseline_embedding_provider == "disabled"
    assert configured_baseline_embedding_adapter(settings) is None
    capability = assess_baseline_embedding(settings)
    assert capability.status is BaselineEmbeddingCapabilityStatus.DISABLED

    with pytest.raises(ValidationError):
        Settings(baseline_embedding_timeout_seconds=61)
    with pytest.raises(ValidationError):
        Settings(baseline_embedding_batch_size=0)
    with pytest.raises(ValidationError):
        Settings(baseline_embedding_dimension=0)


@pytest.mark.parametrize(
    ("endpoint", "allow_loopback", "expected_code"),
    (
        (
            "http://embedding.internal:9010",
            True,
            "embedding_remote_plaintext_rejected",
        ),
        (
            "http://127.0.0.1:9010",
            False,
            "embedding_loopback_http_not_enabled",
        ),
        (
            "http://user:secret@127.0.0.1:9010",
            True,
            "embedding_endpoint_invalid",
        ),
        ("ftp://127.0.0.1:9010", True, "embedding_endpoint_invalid"),
    ),
)
def test_transport_policy_rejects_unsafe_endpoints(
    endpoint: str,
    allow_loopback: bool,
    expected_code: str,
) -> None:
    with pytest.raises(BaselineEmbeddingAdapterError) as exc_info:
        BaselineEmbeddingConfig.from_settings(
            _settings(
                baseline_embedding_endpoint=endpoint,
                baseline_embedding_allow_insecure_loopback=allow_loopback,
            )
        )

    assert exc_info.value.code == expected_code
    assert endpoint not in str(exc_info.value)
    assert "secret" not in str(exc_info.value)


def test_https_remote_and_explicit_loopback_are_valid_configurations() -> None:
    remote = BaselineEmbeddingConfig.from_settings(
        _settings(
            baseline_embedding_endpoint="https://embedding.example/vectors",
            baseline_embedding_allow_insecure_loopback=False,
        )
    )
    loopback = BaselineEmbeddingConfig.from_settings(_settings())

    assert remote.enabled is True
    assert loopback.enabled is True


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    (
        ({"baseline_embedding_model": ""}, "embedding_model_missing"),
        ({"baseline_embedding_revision": None}, "embedding_revision_missing"),
    ),
)
def test_enabled_provider_requires_complete_pinned_identity(
    overrides: dict[str, object],
    expected_reason: str,
) -> None:
    capability = assess_baseline_embedding(_settings(**overrides))

    assert capability.status is BaselineEmbeddingCapabilityStatus.UNAVAILABLE
    assert capability.reason == expected_reason


def test_contract_batches_in_order_and_preserves_float32_without_normalizing() -> None:
    requests: list[httpx.Request] = []
    values = {
        "one": [16_777_217.0, 0.1],
        "two": [3.0, 4.0],
        "three": [-0.0, -7.25],
    }
    adapter = HTTPBaselineEmbeddingAdapter(
        BaselineEmbeddingConfig.from_settings(
            _settings(baseline_embedding_batch_size=2)
        ),
        client_factory=_client_factory(
            _ready_handler(lambda text: values[text], requests)
        ),
    )

    vectors = adapter.embed(("one", "two", "three"))

    assert [request.method for request in requests] == ["GET", "POST", "POST"]
    bodies = [json.loads(request.content) for request in requests[1:]]
    assert [body["texts"] for body in bodies] == [["one", "two"], ["three"]]
    assert all(body["model"] == MODEL for body in bodies)
    assert all(body["revision"] == REVISION for body in bodies)
    assert all(body["dimension"] == 2 for body in bodies)
    assert all(vector.dtype == np.dtype("<f4") for vector in vectors)
    assert vectors[0].tobytes() == np.asarray(values["one"], dtype="<f4").tobytes()
    assert vectors[1].tolist() == [3.0, 4.0]
    assert float(np.linalg.norm(vectors[1])) == 5.0
    assert np.signbit(vectors[2][0])


@pytest.mark.parametrize(
    ("health_override", "expected_code"),
    (
        ({"model": "different/model"}, "embedding_model_mismatch"),
        ({"revision": "different-revision"}, "embedding_revision_mismatch"),
        ({"dimension": 3}, "embedding_dimension_mismatch"),
        ({"provider": "different-provider"}, "embedding_provider_mismatch"),
        ({"contract_version": "different-contract"}, "embedding_contract_mismatch"),
    ),
)
def test_health_attestation_identity_mismatch_fails_closed(
    health_override: dict[str, object],
    expected_code: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"status": "ok", **_identity_payload(**health_override)},
        )

    adapter = HTTPBaselineEmbeddingAdapter(
        BaselineEmbeddingConfig.from_settings(_settings()),
        client_factory=_client_factory(handler),
    )

    with pytest.raises(BaselineEmbeddingAdapterError) as exc_info:
        adapter.attest()

    assert exc_info.value.code == expected_code


@pytest.mark.parametrize(
    ("response_change", "expected_code"),
    (
        (lambda payload: payload.update(vectors=[]), "embedding_vector_count_mismatch"),
        (
            lambda payload: payload.update(vectors=[[1.0]]),
            "embedding_dimension_mismatch",
        ),
        (
            lambda payload: payload.update(vectors=[[math.nan, 0.0]]),
            "embedding_vector_nonfinite",
        ),
        (
            lambda payload: payload.update(vectors=[[math.inf, 0.0]]),
            "embedding_vector_nonfinite",
        ),
        (
            lambda payload: payload.update(vectors="not-a-list"),
            "embedding_response_malformed",
        ),
    ),
)
def test_malformed_vector_responses_fail_closed(response_change, expected_code) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"status": "ok", **_identity_payload()})
        payload = {**_identity_payload(), "vectors": [[1.0, 0.0]]}
        response_change(payload)
        # Raw bytes let the client exercise explicit NaN/Inf rejection even
        # though httpx correctly refuses to *produce* non-standard JSON.
        return httpx.Response(
            200,
            content=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    adapter = HTTPBaselineEmbeddingAdapter(
        BaselineEmbeddingConfig.from_settings(_settings()),
        client_factory=_client_factory(handler),
    )

    with pytest.raises(BaselineEmbeddingAdapterError) as exc_info:
        adapter.embed(("private source sentinel",))

    assert exc_info.value.code == expected_code
    assert "private source sentinel" not in str(exc_info.value)


def test_timeout_and_unavailable_status_are_sanitized(caplog) -> None:
    query = "private retrieval query sentinel"

    def timeout_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"status": "ok", **_identity_payload()})
        raise httpx.ReadTimeout("fixture timeout", request=request)

    adapter = HTTPBaselineEmbeddingAdapter(
        BaselineEmbeddingConfig.from_settings(_settings()),
        client_factory=_client_factory(timeout_handler),
    )
    with pytest.raises(BaselineEmbeddingAdapterError) as exc_info:
        adapter.embed((query,))

    assert exc_info.value.code == "embedding_service_timeout"
    assert exc_info.value.__cause__ is None
    assert query not in str(exc_info.value)
    assert query not in caplog.text

    def unavailable_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="private backend detail")

    unavailable = HTTPBaselineEmbeddingAdapter(
        BaselineEmbeddingConfig.from_settings(_settings()),
        client_factory=_client_factory(unavailable_handler),
    )
    with pytest.raises(BaselineEmbeddingAdapterError) as unavailable_exc:
        unavailable.attest()
    assert unavailable_exc.value.code == "embedding_service_unavailable"
    assert "private backend detail" not in str(unavailable_exc.value)


def test_malformed_json_and_embedding_response_identity_fail_closed() -> None:
    def malformed_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"status": "ok", **_identity_payload()})
        return httpx.Response(
            200,
            content=b"{not-json",
            headers={"Content-Type": "application/json"},
        )

    malformed = HTTPBaselineEmbeddingAdapter(
        BaselineEmbeddingConfig.from_settings(_settings()),
        client_factory=_client_factory(malformed_handler),
    )
    with pytest.raises(BaselineEmbeddingAdapterError) as malformed_exc:
        malformed.embed(("source",))
    assert malformed_exc.value.code == "embedding_response_malformed"

    def mismatch_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"status": "ok", **_identity_payload()})
        return httpx.Response(
            200,
            json={
                **_identity_payload(revision="substituted-revision"),
                "vectors": [[1.0, 0.0]],
            },
        )

    mismatch = HTTPBaselineEmbeddingAdapter(
        BaselineEmbeddingConfig.from_settings(_settings()),
        client_factory=_client_factory(mismatch_handler),
    )
    with pytest.raises(BaselineEmbeddingAdapterError) as mismatch_exc:
        mismatch.embed(("source",))
    assert mismatch_exc.value.code == "embedding_revision_mismatch"


def test_http_clients_close_on_success_and_failure() -> None:
    successful_clients: list[httpx.Client] = []

    def success_factory() -> httpx.Client:
        client = httpx.Client(transport=httpx.MockTransport(_ready_handler()))
        successful_clients.append(client)
        return client

    adapter = HTTPBaselineEmbeddingAdapter(
        BaselineEmbeddingConfig.from_settings(_settings()),
        client_factory=success_factory,
    )
    adapter.embed(("source",))
    assert successful_clients and all(client.is_closed for client in successful_clients)

    failing_clients: list[httpx.Client] = []

    def failure_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("fixture timeout", request=request)

    def failure_factory() -> httpx.Client:
        client = httpx.Client(transport=httpx.MockTransport(failure_handler))
        failing_clients.append(client)
        return client

    failing = HTTPBaselineEmbeddingAdapter(
        BaselineEmbeddingConfig.from_settings(_settings()),
        client_factory=failure_factory,
    )
    with pytest.raises(BaselineEmbeddingAdapterError):
        failing.attest()
    assert failing_clients and all(client.is_closed for client in failing_clients)


def test_capability_attests_identity_without_endpoint_credentials_or_text() -> None:
    endpoint = "https://embedding.private.example/service"
    query = "private capability sentinel"
    capability = assess_baseline_embedding(
        _settings(
            baseline_embedding_endpoint=endpoint,
            baseline_embedding_allow_insecure_loopback=False,
        ),
        client_factory=_client_factory(_ready_handler()),
    )

    payload = capability.as_dict()
    assert payload == {
        "status": "ready",
        "reason": "identity_attested",
        "provider_mode": "http",
        "provider": BASELINE_EMBEDDING_HTTP_PROVIDER,
        "contract_version": BASELINE_EMBEDDING_HTTP_CONTRACT,
        "model": MODEL,
        "revision": REVISION,
        "dimension": 2,
        "fingerprint": capability.fingerprint,
        "transport": "https",
    }
    assert endpoint not in repr(payload)
    assert "embedding.private.example" not in repr(payload)
    assert query not in repr(payload)


def test_capabilities_and_health_report_disabled_adapter_without_endpoint() -> None:
    settings = Settings(
        baseline_embedding_endpoint="https://unused.private.example",
    )

    capabilities_payload = capabilities_endpoint(settings)
    health_payload = health_endpoint(settings)

    for payload in (capabilities_payload, health_payload):
        assert payload["baseline_embedding"]["status"] == "disabled"
        assert "unused.private.example" not in repr(payload)


def test_same_production_adapter_identity_builds_and_retrieves(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def vector_for_text(text: str) -> list[float]:
        if text.startswith("Repository file: peer/src/alpha.txt"):
            return [1.0, 0.0]
        if text.startswith("Repository file: peer/src/beta.txt"):
            return [0.0, 1.0]
        return [1.0, 0.0]

    client_factory = _client_factory(_ready_handler(vector_for_text, requests))
    engine, SessionMaker = _sessions(tmp_path)
    generation = _ingest(SessionMaker)
    try:
        build = build_configured_baseline_index(
            SessionMaker,
            settings=_settings(),
            generation_id=generation.generation_id,
            index_version="baseline-http-index-1",
            client_factory=client_factory,
        )
        retriever = create_configured_persistent_baseline_retriever(
            SessionMaker,
            settings=_settings(),
            client_factory=client_factory,
        )

        result = retriever.retrieve(_request())
        disabled = create_configured_persistent_baseline_retriever(
            SessionMaker,
            settings=Settings(),
        ).retrieve(_request())
    finally:
        engine.dispose()

    assert result.status is RetrievalStatus.OK
    assert build.status.value == "compatible"
    assert result.index_id == build.index_id
    assert result.embedding_provider == BASELINE_EMBEDDING_HTTP_PROVIDER
    assert result.embedding_model == MODEL
    assert result.embedding_revision == REVISION
    assert result.embedding_dimension == 2
    assert result.embedding_fingerprint == retriever._provider.fingerprint
    assert [item.relative_path for item in result.evidence] == [
        "src/alpha.txt",
        "src/beta.txt",
    ]
    posts = [request for request in requests if request.method == "POST"]
    assert len(posts) == 2
    assert len(json.loads(posts[0].content)["texts"]) == 2
    assert json.loads(posts[1].content)["texts"] == ["alpha change"]
    assert disabled.status is RetrievalStatus.ERROR
    assert disabled.error.code == "embedding_adapter_unavailable"
    assert disabled.fallback_engine is None


def test_index_build_timeout_does_not_publish(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"status": "ok", **_identity_payload()})
        raise httpx.ReadTimeout("fixture timeout", request=request)

    engine, SessionMaker = _sessions(tmp_path)
    generation = _ingest(SessionMaker)
    try:
        with pytest.raises(BaselineIndexBuildError) as exc_info:
            build_configured_baseline_index(
                SessionMaker,
                settings=_settings(),
                generation_id=generation.generation_id,
                index_version="baseline-http-timeout",
                client_factory=_client_factory(handler),
            )
        with SessionMaker() as session:
            publication = session.query(RetrievalBaselineIndexPublication).one_or_none()
    finally:
        engine.dispose()

    assert exc_info.value.code == "embedding_adapter_failed"
    assert publication is None


def test_query_outage_returns_error_without_query_text(tmp_path: Path) -> None:
    state = {"fail": False}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"status": "ok", **_identity_payload()})
        if state["fail"]:
            raise httpx.ReadTimeout("fixture timeout", request=request)
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                **_identity_payload(),
                "vectors": [[1.0, 0.0] for _ in body["texts"]],
            },
        )

    client_factory = _client_factory(handler)
    engine, SessionMaker = _sessions(tmp_path)
    generation = _ingest(SessionMaker)
    query = "private query outage sentinel"
    try:
        build_configured_baseline_index(
            SessionMaker,
            settings=_settings(),
            generation_id=generation.generation_id,
            index_version="baseline-http-index-1",
            client_factory=client_factory,
        )
        retriever = create_configured_persistent_baseline_retriever(
            SessionMaker,
            settings=_settings(),
            client_factory=client_factory,
        )
        state["fail"] = True

        result = retriever.retrieve(_request(query))
    finally:
        engine.dispose()

    assert result.status is RetrievalStatus.ERROR
    assert result.error.code == "query_embedding_failed"
    assert query not in result.error.message
    assert query not in repr(result)
    assert result.fallback_engine is None


def test_published_index_rejects_different_configured_fingerprint(
    tmp_path: Path,
) -> None:
    first_factory = _client_factory(_ready_handler())
    engine, SessionMaker = _sessions(tmp_path)
    generation = _ingest(SessionMaker)
    try:
        build_configured_baseline_index(
            SessionMaker,
            settings=_settings(),
            generation_id=generation.generation_id,
            index_version="baseline-http-index-1",
            client_factory=first_factory,
        )
        mismatched_settings = _settings(
            baseline_embedding_revision="fixture-pinned-revision-2"
        )
        mismatched_handler = _ready_handler()
        retriever = create_configured_persistent_baseline_retriever(
            SessionMaker,
            settings=mismatched_settings,
            client_factory=_client_factory(mismatched_handler),
        )

        result = retriever.retrieve(_request())
    finally:
        engine.dispose()

    assert result.status is RetrievalStatus.ERROR
    assert result.error.code == "embedding_fingerprint_mismatch"
    assert result.fallback_engine is None
