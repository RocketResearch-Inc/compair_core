"""Frozen model-manifest loading and validation.

The installed manifest is the sole runtime source for the local service model
identity and artifact set. Invalid packaged bytes fail closed during loading.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import PurePosixPath
from typing import Any

MANIFEST_RESOURCE = "baseline-v1.manifest.json"
MANIFEST_SCHEMA_VERSION = "compair-baseline-model-manifest.v1"
PROFILE = "baseline-v1"


class BaselineModelManifestError(RuntimeError):
    """A sanitized failure to load the installed frozen manifest."""


@dataclass(frozen=True, slots=True)
class BaselineModelArtifact:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class BaselineModelManifest:
    schema_version: str
    profile: str
    logical_model: str
    artifact_repository: str
    revision: str
    dimension: int
    dtype: str
    normalization: str
    contract_version: str
    provider: str
    runtime_packages: tuple[tuple[str, str], ...]
    total_bytes: int
    files: tuple[BaselineModelArtifact, ...]
    embedding_identity_fingerprint: str
    tokenizer_fingerprint: str
    model_artifact_fingerprint: str
    licenses: tuple[tuple[str, str], ...]
    manifest_fingerprint: str

    @property
    def expected_paths(self) -> tuple[str, ...]:
        return tuple(artifact.path for artifact in self.files)

    @property
    def runtime_package_versions(self) -> dict[str, str]:
        return dict(self.runtime_packages)

    def safe_summary(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "logical_model": self.logical_model,
            "artifact_repository": self.artifact_repository,
            "revision": self.revision,
            "dimension": self.dimension,
            "dtype": self.dtype,
            "normalization": self.normalization,
            "contract_version": self.contract_version,
            "provider": self.provider,
            "runtime_packages": self.runtime_package_versions,
            "artifact_count": len(self.files),
            "total_bytes": self.total_bytes,
            "embedding_identity_fingerprint": (self.embedding_identity_fingerprint),
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
            "model_artifact_fingerprint": self.model_artifact_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
        }


_TOP_LEVEL_KEYS = {
    "schema_version",
    "profile",
    "logical_model",
    "artifact_repository",
    "revision",
    "dimension",
    "dtype",
    "normalization",
    "contract_version",
    "provider",
    "runtime_packages",
    "total_bytes",
    "files",
    "embedding_identity_fingerprint",
    "tokenizer_fingerprint",
    "model_artifact_fingerprint",
    "licenses",
    "manifest_fingerprint",
}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _required_string(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise BaselineModelManifestError("baseline_model_manifest_invalid")
    return value


def _required_sha256(value: Any) -> str:
    candidate = _required_string(value)
    if len(candidate) != 64 or any(
        character not in "0123456789abcdef" for character in candidate
    ):
        raise BaselineModelManifestError("baseline_model_manifest_invalid")
    return candidate


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise BaselineModelManifestError("baseline_model_manifest_invalid")
        value[key] = item
    return value


def _load_manifest_bytes(raw: bytes) -> BaselineModelManifest:
    try:
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                BaselineModelManifestError("baseline_model_manifest_invalid")
            ),
        )
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise BaselineModelManifestError("baseline_model_manifest_invalid") from exc
    if not isinstance(payload, dict) or set(payload) != _TOP_LEVEL_KEYS:
        raise BaselineModelManifestError("baseline_model_manifest_invalid")

    declared_fingerprint = _required_sha256(payload["manifest_fingerprint"])
    fingerprint_payload = dict(payload)
    del fingerprint_payload["manifest_fingerprint"]
    if _canonical_sha256(fingerprint_payload) != declared_fingerprint:
        raise BaselineModelManifestError("baseline_model_manifest_invalid")

    raw_files = payload["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise BaselineModelManifestError("baseline_model_manifest_invalid")
    artifacts: list[BaselineModelArtifact] = []
    for raw_artifact in raw_files:
        if not isinstance(raw_artifact, dict) or set(raw_artifact) != {
            "path",
            "size",
            "sha256",
        }:
            raise BaselineModelManifestError("baseline_model_manifest_invalid")
        path = _required_string(raw_artifact["path"])
        pure_path = PurePosixPath(path)
        if pure_path.is_absolute() or len(pure_path.parts) != 1 or path in {".", ".."}:
            raise BaselineModelManifestError("baseline_model_manifest_invalid")
        size = raw_artifact["size"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise BaselineModelManifestError("baseline_model_manifest_invalid")
        artifacts.append(
            BaselineModelArtifact(
                path=path,
                size=size,
                sha256=_required_sha256(raw_artifact["sha256"]),
            )
        )
    if tuple(artifact.path for artifact in artifacts) != tuple(
        sorted(artifact.path for artifact in artifacts)
    ) or len({artifact.path for artifact in artifacts}) != len(artifacts):
        raise BaselineModelManifestError("baseline_model_manifest_invalid")

    licenses = payload["licenses"]
    if not isinstance(licenses, list) or not licenses:
        raise BaselineModelManifestError("baseline_model_manifest_invalid")
    parsed_licenses: list[tuple[str, str]] = []
    for license_value in licenses:
        if not isinstance(license_value, dict) or set(license_value) != {"name", "url"}:
            raise BaselineModelManifestError("baseline_model_manifest_invalid")
        parsed_licenses.append(
            (
                _required_string(license_value["name"]),
                _required_string(license_value["url"]),
            )
        )

    runtime_packages = payload["runtime_packages"]
    if not isinstance(runtime_packages, dict) or set(runtime_packages) != {
        "fastembed",
        "huggingface-hub",
        "numpy",
        "onnxruntime",
        "tokenizers",
    }:
        raise BaselineModelManifestError("baseline_model_manifest_invalid")
    parsed_runtime_packages = tuple(
        (name, _required_string(package_version))
        for name, package_version in sorted(runtime_packages.items())
    )

    total_bytes = payload["total_bytes"]
    dimension = payload["dimension"]
    if (
        not isinstance(total_bytes, int)
        or isinstance(total_bytes, bool)
        or total_bytes != sum(artifact.size for artifact in artifacts)
        or not isinstance(dimension, int)
        or isinstance(dimension, bool)
        or dimension < 1
    ):
        raise BaselineModelManifestError("baseline_model_manifest_invalid")

    manifest = BaselineModelManifest(
        schema_version=_required_string(payload["schema_version"]),
        profile=_required_string(payload["profile"]),
        logical_model=_required_string(payload["logical_model"]),
        artifact_repository=_required_string(payload["artifact_repository"]),
        revision=_required_string(payload["revision"]),
        dimension=dimension,
        dtype=_required_string(payload["dtype"]),
        normalization=_required_string(payload["normalization"]),
        contract_version=_required_string(payload["contract_version"]),
        provider=_required_string(payload["provider"]),
        runtime_packages=parsed_runtime_packages,
        total_bytes=total_bytes,
        files=tuple(artifacts),
        embedding_identity_fingerprint=_required_sha256(
            payload["embedding_identity_fingerprint"]
        ),
        tokenizer_fingerprint=_required_sha256(payload["tokenizer_fingerprint"]),
        model_artifact_fingerprint=_required_sha256(
            payload["model_artifact_fingerprint"]
        ),
        licenses=tuple(parsed_licenses),
        manifest_fingerprint=declared_fingerprint,
    )
    if (
        manifest.schema_version != MANIFEST_SCHEMA_VERSION
        or manifest.profile != PROFILE
        or manifest.dtype != "float32"
        or manifest.normalization != "none"
    ):
        raise BaselineModelManifestError("baseline_model_manifest_invalid")

    identity = {
        "contract_version": manifest.contract_version,
        "dimension": manifest.dimension,
        "model": manifest.logical_model,
        "provider": manifest.provider,
        "revision": manifest.revision,
    }
    if _canonical_sha256(identity) != manifest.embedding_identity_fingerprint:
        raise BaselineModelManifestError("baseline_model_manifest_invalid")
    tokenizer_artifacts = [
        raw_artifact
        for raw_artifact in raw_files
        if raw_artifact["path"]
        in {
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        }
    ]
    model_artifacts = [
        raw_artifact
        for raw_artifact in raw_files
        if raw_artifact["path"] in {"config.json", "model_optimized.onnx"}
    ]
    if (
        _canonical_sha256(tokenizer_artifacts) != manifest.tokenizer_fingerprint
        or _canonical_sha256(model_artifacts) != manifest.model_artifact_fingerprint
    ):
        raise BaselineModelManifestError("baseline_model_manifest_invalid")
    return manifest


@lru_cache(maxsize=1)
def load_baseline_model_manifest() -> BaselineModelManifest:
    """Load and cryptographically self-check the installed model manifest."""

    try:
        raw = resources.files(__package__).joinpath(MANIFEST_RESOURCE).read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise BaselineModelManifestError("baseline_model_manifest_unavailable") from exc
    return _load_manifest_bytes(raw)


__all__ = [
    "MANIFEST_RESOURCE",
    "MANIFEST_SCHEMA_VERSION",
    "PROFILE",
    "BaselineModelArtifact",
    "BaselineModelManifest",
    "BaselineModelManifestError",
    "load_baseline_model_manifest",
]
