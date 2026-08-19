"""Privacy-safe attestation of baseline runtime configuration.

This module intentionally lives above :mod:`compair_core.compair`.  Operational
commands can inspect configuration and migration state without importing the
legacy package, whose historical startup contract initializes the database.
Only hashes of database and provider endpoint identities leave this module.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

import rfc8785
from pydantic import SecretStr
from sqlalchemy.engine import URL, make_url

from .baseline_generation.profile import (
    ACCELERATED_GENERATION_TIMEOUT_SECONDS,
    GENERATION_LEASE_COMMIT_MARGIN_SECONDS,
    QUALIFIED_CONTEXT_TOKENS,
    QUALIFIED_OUTPUT_TOKENS,
    required_generation_lease_seconds,
)

RUNTIME_CONFIG_CONTRACT_VERSION = "baseline-runtime-config.v1"
BASELINE_ENGINE_VERSION = "baseline_v1.persistent.v1"
BASELINE_TOKENIZER_VERSION = "baseline_v1_frozen_tokenizer.v1"
BASELINE_DOCUMENT_FORMAT_VERSION = "baseline_v1_whole_file_12000.v1"
BASELINE_INDEX_SCHEMA_VERSION = "baseline-index.v1"
BASELINE_VECTOR_FORMAT = "float32-le.v1"
BASELINE_EMBEDDING_CONTRACT = "baseline-embedding-http.v1"
BASELINE_EMBEDDING_PROVIDER = "baseline_http_v1"
BASELINE_GENERATION_ADAPTER_CONTRACT = "baseline-generation-ollama-http.v1"
BASELINE_GENERATION_OUTPUT_VERSION = "baseline-generation-output.v2"
BASELINE_GENERATION_OUTPUT_SPEC_SHA256 = (
    "e670731777b253f9d5e3984405c2d99871ba26f637a17e6221cc82d97bc8beb1"
)
BASELINE_GENERATION_OUTPUT_SCHEMA_SHA256 = (
    "fc5a85d5d38c18775afe0966987ea74e7e9ac072148822c1be60a199e32cca27"
)
CONTROL_PLANE_V1_SHA256 = (
    "3b45287a54d04cea751e9cc3209c5f0783192de13062e682eadcae40af322650"
)
CONTROL_PLANE_V2_SHA256 = (
    "b278abe007779f05e92509db068f555701c03cba5cf236151e8df231a9b44091"
)
WORKER_CONTRACT_VERSION = "baseline-database-worker.v1"
WORKER_SUPPORTED_JOB_TYPES = (
    "baseline_run",
    "cleanup",
    "corpus_ingestion",
    "index_build",
)
QUERY_PAYLOAD_CONTRACT = "baseline-run-protected-payload.v1"
KEYRING_CONTRACT_VERSION = "baseline-run-keyring.v1"
NOTIFICATIONS_DEFAULT_ENABLED = False

_TOKEN_PATTERN = r"[A-Za-z_][A-Za-z0-9_.:/-]{1,}|[0-9]+"
_STOPWORDS = (
    "and",
    "are",
    "class",
    "const",
    "false",
    "for",
    "from",
    "function",
    "import",
    "into",
    "return",
    "that",
    "the",
    "this",
    "true",
    "with",
)


class RuntimeConfigurationError(RuntimeError):
    """Sanitized invalid runtime configuration."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def canonical_sha256(value: object) -> str:
    """Return SHA-256 over RFC 8785 canonical JSON bytes."""

    try:
        canonical = rfc8785.dumps(value)
    except (TypeError, ValueError, rfc8785.CanonicalizationError):
        raise RuntimeConfigurationError("runtime_configuration_invalid") from None
    return hashlib.sha256(canonical).hexdigest()


def _package_version() -> str:
    try:
        return version("compair-core")
    except PackageNotFoundError:
        # Source checkouts use the release metadata from pyproject.toml.
        return "0.10.4"


def _secret_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if hasattr(value, "get_secret_value"):
        return str(value.get_secret_value())
    return str(value)


@dataclass(frozen=True, slots=True, repr=False)
class KeyringAttestation:
    active_key_id: str | None
    identity_fingerprint: str | None
    key_ids: tuple[str, ...]
    valid: bool
    reason_code: str | None

    def __repr__(self) -> str:
        return "KeyringAttestation(<redacted>)"


def attest_keyring(value: object) -> KeyringAttestation:
    """Validate the protected-query keyring without exposing key material."""

    raw = _secret_value(value)
    if not raw:
        return KeyringAttestation(
            None,
            None,
            (),
            False,
            "run_keyring_unavailable",
        )

    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate")
            result[key] = item
        return result

    try:
        payload = json.loads(
            raw,
            object_pairs_hook=strict_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("nonfinite")
            ),
        )
        if not isinstance(payload, Mapping) or set(payload) != {
            "version",
            "active_key_id",
            "keys",
        }:
            raise ValueError("shape")
        active = payload["active_key_id"]
        entries = payload["keys"]
        if (
            payload["version"] != KEYRING_CONTRACT_VERSION
            or not isinstance(active, str)
            or not active
            or len(active) > 128
            or not isinstance(entries, list)
            or not entries
        ):
            raise ValueError("contract")
        keys: dict[str, bytes] = {}
        for entry in entries:
            if not isinstance(entry, Mapping) or set(entry) != {
                "key_id",
                "key_base64",
            }:
                raise ValueError("entry")
            key_id = entry["key_id"]
            encoded = entry["key_base64"]
            if (
                not isinstance(key_id, str)
                or not key_id
                or len(key_id) > 128
                or key_id in keys
                or not isinstance(encoded, str)
            ):
                raise ValueError("entry")
            decoded = base64.b64decode(encoded, validate=True)
            if len(decoded) != 32:
                raise ValueError("key")
            keys[key_id] = decoded
        if active not in keys:
            raise ValueError("active")
    except (ValueError, TypeError, json.JSONDecodeError):
        return KeyringAttestation(None, None, (), False, "run_keyring_invalid")

    fingerprint = canonical_sha256(
        {
            "version": KEYRING_CONTRACT_VERSION,
            "keys": [
                {
                    "key_id": key_id,
                    "key_sha256": hashlib.sha256(keys[key_id]).hexdigest(),
                }
                for key_id in sorted(keys)
            ],
        }
    )
    return KeyringAttestation(
        active,
        fingerprint,
        tuple(sorted(keys)),
        True,
        None,
    )


def _endpoint_identity(value: object, *, allow_loopback_http: bool) -> dict[str, str]:
    if not isinstance(value, str) or not value or value != value.strip():
        return {
            "identity_sha256": canonical_sha256({"state": "absent"}),
            "transport": "absent",
        }
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return {
            "identity_sha256": canonical_sha256({"state": "invalid"}),
            "transport": "invalid",
        }
    if (
        parsed.scheme not in {"http", "https"}
        or host is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return {
            "identity_sha256": canonical_sha256({"state": "invalid"}),
            "transport": "invalid",
        }
    host_normalized = host.lower().rstrip(".")
    default_port = 80 if parsed.scheme == "http" else 443
    normalized = {
        "scheme": parsed.scheme,
        "host": host_normalized,
        "port": port or default_port,
        "path": parsed.path.rstrip("/") or "/",
    }
    try:
        import ipaddress

        loopback = ipaddress.ip_address(host_normalized).is_loopback
    except ValueError:
        loopback = host_normalized == "localhost"
    if parsed.scheme == "https":
        transport = "verified_https"
    elif loopback and allow_loopback_http:
        transport = "explicit_loopback_http"
    else:
        transport = "prohibited_plaintext"
    return {
        "identity_sha256": canonical_sha256(normalized),
        "transport": transport,
    }


def database_identity(value: URL | str) -> dict[str, str]:
    """Hash a credential-free normalized database identity."""

    try:
        url = value if isinstance(value, URL) else make_url(value)
    except Exception:  # noqa: BLE001 - never reflect a malformed DSN
        raise RuntimeConfigurationError("database_identity_invalid") from None
    backend = url.get_backend_name()
    if backend == "sqlite":
        identity = {
            "backend": "sqlite",
            "database": str(url.database or ""),
        }
        transport = "local_file"
    else:
        identity = {
            "backend": backend,
            "database": str(url.database or ""),
            "host": str(url.host or "").lower().rstrip("."),
            "port": int(url.port or 5432),
        }
        sslmode = str(url.query.get("sslmode", "")).lower()
        transport = (
            "tls_required"
            if sslmode in {"require", "verify-ca", "verify-full"}
            else "transport_unspecified"
        )
    return {
        "backend": backend,
        "identity_sha256": canonical_sha256(identity),
        "transport": transport,
    }


def _embedding_identity(settings: Any) -> dict[str, object]:
    mode = str(getattr(settings, "baseline_embedding_provider", "disabled"))
    model = getattr(settings, "baseline_embedding_model", None)
    revision = getattr(settings, "baseline_embedding_revision", None)
    dimension = int(getattr(settings, "baseline_embedding_dimension", 384))
    adapter_identity = {
        "contract_version": BASELINE_EMBEDDING_CONTRACT,
        "dimension": dimension,
        "model": model,
        "provider": BASELINE_EMBEDDING_PROVIDER,
        "revision": revision,
    }
    identity_fingerprint = canonical_sha256(adapter_identity)
    return {
        **adapter_identity,
        "dtype": "float32",
        "mode": mode,
        "fingerprint": identity_fingerprint,
        "endpoint": _endpoint_identity(
            getattr(settings, "baseline_embedding_endpoint", None),
            allow_loopback_http=bool(
                getattr(settings, "baseline_embedding_allow_insecure_loopback", False)
            ),
        ),
    }


def _generation_identity(settings: Any) -> dict[str, object]:
    identity = {
        "adapter_contract": BASELINE_GENERATION_ADAPTER_CONTRACT,
        "model": getattr(settings, "baseline_generation_model", None),
        "model_digest": getattr(settings, "baseline_generation_model_digest", None),
        "model_version": getattr(settings, "baseline_generation_model_version", None),
        "output_schema_sha256": BASELINE_GENERATION_OUTPUT_SCHEMA_SHA256,
        "output_schema_version": BASELINE_GENERATION_OUTPUT_VERSION,
        "output_spec_sha256": BASELINE_GENERATION_OUTPUT_SPEC_SHA256,
        "provider": str(getattr(settings, "baseline_generation_provider", "disabled")),
    }
    return {
        **identity,
        "fingerprint": canonical_sha256(identity),
        "endpoint": _endpoint_identity(
            getattr(settings, "baseline_generation_endpoint", None),
            allow_loopback_http=bool(
                getattr(settings, "baseline_generation_allow_loopback_http", False)
            ),
        ),
    }


@dataclass(frozen=True, slots=True, repr=False)
class RuntimeConfigurationAttestation:
    contract_version: str
    fingerprint: str
    embedding_identity_fingerprint: str
    generation_identity_fingerprint: str
    keyring_active_key_id: str | None
    keyring_identity_fingerprint: str | None
    database_backend: str
    database_identity_fingerprint: str
    canonical_configuration: Mapping[str, object]

    def __repr__(self) -> str:
        return (
            "RuntimeConfigurationAttestation(contract_version="
            f"{self.contract_version!r}, fingerprint={self.fingerprint!r})"
        )

    def safe_summary(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "fingerprint": self.fingerprint,
            "embedding_identity_fingerprint": self.embedding_identity_fingerprint,
            "generation_identity_fingerprint": self.generation_identity_fingerprint,
            "keyring_active_key_id": self.keyring_active_key_id,
            "keyring_identity_fingerprint": self.keyring_identity_fingerprint,
            "database_backend": self.database_backend,
            "database_identity_fingerprint": self.database_identity_fingerprint,
        }


def build_runtime_configuration(
    settings: Any,
    *,
    database_url: URL | str,
) -> RuntimeConfigurationAttestation:
    """Build the stable cross-process API/worker runtime attestation."""

    database = database_identity(database_url)
    keyring = attest_keyring(getattr(settings, "baseline_run_encryption_keyring", None))
    embedding = _embedding_identity(settings)
    generation = _generation_identity(settings)
    generation_timeout_seconds = float(
        getattr(
            settings,
            "baseline_generation_timeout_seconds",
            ACCELERATED_GENERATION_TIMEOUT_SECONDS,
        )
    )
    try:
        generation_lease_seconds = required_generation_lease_seconds(
            generation_timeout_seconds
        )
    except ValueError:
        raise RuntimeConfigurationError(
            "generation_timeout_lease_incompatible"
        ) from None
    configuration: dict[str, object] = {
        "contract_version": RUNTIME_CONFIG_CONTRACT_VERSION,
        "core_version": _package_version(),
        "retrieval": {
            "engine": str(getattr(settings, "retrieval_engine", "legacy")),
            "baseline_engine_version": BASELINE_ENGINE_VERSION,
            "tokenizer_version": BASELINE_TOKENIZER_VERSION,
            "token_pattern": _TOKEN_PATTERN,
            "stopwords": list(_STOPWORDS),
            "bm25": {"b": 0.75, "k1": 1.5},
            "rrf_k": 60,
            "candidate_limit": 6,
            "evidence_budget": {"characters": 16_000, "items": 4},
            "document_format": BASELINE_DOCUMENT_FORMAT_VERSION,
            "index_schema": BASELINE_INDEX_SCHEMA_VERSION,
            "vector_format": BASELINE_VECTOR_FORMAT,
        },
        "protocols": {
            "control_plane_v1_sha256": CONTROL_PLANE_V1_SHA256,
            "control_plane_v2_sha256": CONTROL_PLANE_V2_SHA256,
            "generation_output_schema_sha256": (
                BASELINE_GENERATION_OUTPUT_SCHEMA_SHA256
            ),
            "generation_output_spec_sha256": (BASELINE_GENERATION_OUTPUT_SPEC_SHA256),
        },
        "embedding": embedding,
        "generation": generation,
        "worker": {
            "contract_version": WORKER_CONTRACT_VERSION,
            "mode": str(getattr(settings, "baseline_worker_mode", "manual")),
            "supported_job_types": list(WORKER_SUPPORTED_JOB_TYPES),
            "heartbeat_ttl_seconds": int(
                getattr(settings, "baseline_worker_heartbeat_ttl_seconds", 30)
            ),
            "heartbeat_interval_seconds": float(
                getattr(settings, "baseline_worker_heartbeat_interval_seconds", 5.0)
            ),
            "poll_interval_seconds": float(
                getattr(settings, "baseline_worker_poll_interval_seconds", 2.0)
            ),
            "cleanup_interval_seconds": int(
                getattr(settings, "baseline_worker_cleanup_interval_seconds", 30)
            ),
            "maximum_attempts": int(
                getattr(settings, "baseline_worker_max_attempts", 5)
            ),
            "maximum_backoff_seconds": float(
                getattr(settings, "baseline_worker_max_backoff_seconds", 30.0)
            ),
            "maximum_pending_per_slot": int(
                getattr(settings, "baseline_worker_max_pending_per_slot", 8)
            ),
        },
        "baseline_runs_enabled": bool(
            getattr(settings, "baseline_runs_enabled", False)
        ),
        "notifications": {
            "enabled": bool(
                getattr(
                    settings,
                    "baseline_notifications_enabled",
                    NOTIFICATIONS_DEFAULT_ENABLED,
                )
            ),
            "default": "disabled",
        },
        "query_payload": {
            "contract_version": QUERY_PAYLOAD_CONTRACT,
            "lifetime_seconds": int(
                getattr(settings, "baseline_run_payload_ttl_seconds", 900)
            ),
            "keyring_active_key_id": keyring.active_key_id,
            "keyring_identity_fingerprint": keyring.identity_fingerprint,
        },
        "database": database,
        "transport": {
            "control_plane_loopback_http": bool(
                getattr(
                    settings,
                    "baseline_control_plane_allow_insecure_loopback",
                    False,
                )
            ),
            "trusted_proxy_allowlist_sha256": canonical_sha256(
                {
                    "allowlist": sorted(
                        item.strip()
                        for item in str(
                            getattr(
                                settings,
                                "baseline_control_plane_trusted_proxy_allowlist",
                                "",
                            )
                        ).split(",")
                        if item.strip()
                    )
                }
            ),
        },
        "limits": {
            "control_request_bytes": 64_000,
            "run_request_bytes": 8_100_000,
            "raw_query_bytes": 8_000_000,
            "embedding_batch_size": int(
                getattr(settings, "baseline_embedding_batch_size", 32)
            ),
            "embedding_timeout_seconds": float(
                getattr(settings, "baseline_embedding_timeout_seconds", 10.0)
            ),
            "generation_request_bytes": int(
                getattr(settings, "baseline_generation_max_request_bytes", 256_000)
            ),
            "generation_response_bytes": int(
                getattr(settings, "baseline_generation_max_response_bytes", 200_000)
            ),
            "generation_timeout_seconds": generation_timeout_seconds,
            "generation_lease_seconds": generation_lease_seconds,
            "generation_lease_commit_margin_seconds": (
                GENERATION_LEASE_COMMIT_MARGIN_SECONDS
            ),
            "generation_context_tokens": int(
                getattr(
                    settings,
                    "baseline_generation_context_tokens",
                    QUALIFIED_CONTEXT_TOKENS,
                )
            ),
            "generation_output_tokens": int(
                getattr(
                    settings,
                    "baseline_generation_output_tokens",
                    QUALIFIED_OUTPUT_TOKENS,
                )
            ),
            "generation_seed": int(getattr(settings, "baseline_generation_seed", 0)),
        },
    }
    frozen = MappingProxyType(configuration)
    fingerprint = canonical_sha256(configuration)
    return RuntimeConfigurationAttestation(
        contract_version=RUNTIME_CONFIG_CONTRACT_VERSION,
        fingerprint=fingerprint,
        embedding_identity_fingerprint=str(embedding["fingerprint"]),
        generation_identity_fingerprint=str(generation["fingerprint"]),
        keyring_active_key_id=keyring.active_key_id,
        keyring_identity_fingerprint=keyring.identity_fingerprint,
        database_backend=database["backend"],
        database_identity_fingerprint=database["identity_sha256"],
        canonical_configuration=frozen,
    )


def validate_runtime_configuration(
    settings: Any,
    *,
    database_url: URL | str,
    require_worker_baseline: bool = False,
) -> RuntimeConfigurationAttestation:
    """Fail closed on unsafe combinations used by installed API/worker commands."""

    attestation = build_runtime_configuration(settings, database_url=database_url)
    configuration = attestation.canonical_configuration
    retrieval_engine = str(getattr(settings, "retrieval_engine", "legacy"))
    if retrieval_engine not in {"legacy", "baseline_v1"}:
        raise RuntimeConfigurationError("retrieval_engine_invalid")

    embedding = configuration["embedding"]
    assert isinstance(embedding, Mapping)
    embedding_mode = str(embedding["mode"])
    if embedding_mode not in {"disabled", "http"}:
        raise RuntimeConfigurationError("embedding_provider_mode_invalid")
    embedding_endpoint = embedding["endpoint"]
    assert isinstance(embedding_endpoint, Mapping)
    if embedding_mode == "http" and (
        not embedding.get("model")
        or not embedding.get("revision")
        or embedding_endpoint.get("transport")
        not in {"verified_https", "explicit_loopback_http"}
    ):
        raise RuntimeConfigurationError("embedding_configuration_invalid")

    generation = configuration["generation"]
    assert isinstance(generation, Mapping)
    generation_mode = str(generation["provider"])
    if generation_mode not in {"disabled", "http", "ollama"}:
        raise RuntimeConfigurationError("generation_provider_mode_invalid")
    generation_endpoint = generation["endpoint"]
    assert isinstance(generation_endpoint, Mapping)
    if generation_mode in {"http", "ollama"} and (
        not generation.get("model")
        or (generation_mode == "ollama" and not generation.get("model_digest"))
        or generation_endpoint.get("transport")
        not in {"verified_https", "explicit_loopback_http"}
    ):
        raise RuntimeConfigurationError("generation_configuration_invalid")

    keyring = attest_keyring(getattr(settings, "baseline_run_encryption_keyring", None))
    baseline_required = require_worker_baseline or bool(
        getattr(settings, "baseline_runs_enabled", False)
    )
    if baseline_required:
        if embedding_mode != "http":
            raise RuntimeConfigurationError("embedding_provider_disabled")
        if generation_mode not in {"http", "ollama"}:
            raise RuntimeConfigurationError("generation_provider_disabled")
        if not keyring.valid:
            raise RuntimeConfigurationError(
                keyring.reason_code or "run_keyring_invalid"
            )
    return attestation


__all__ = [
    "BASELINE_GENERATION_OUTPUT_SCHEMA_SHA256",
    "BASELINE_GENERATION_OUTPUT_SPEC_SHA256",
    "CONTROL_PLANE_V1_SHA256",
    "CONTROL_PLANE_V2_SHA256",
    "RUNTIME_CONFIG_CONTRACT_VERSION",
    "WORKER_CONTRACT_VERSION",
    "WORKER_SUPPORTED_JOB_TYPES",
    "RuntimeConfigurationAttestation",
    "RuntimeConfigurationError",
    "attest_keyring",
    "build_runtime_configuration",
    "canonical_sha256",
    "database_identity",
    "validate_runtime_configuration",
]
