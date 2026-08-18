"""Fail-closed HTTP embeddings used only by persistent ``baseline_v1``.

This module never calls the legacy embedding stack.  It deliberately has no
fallback provider and never logs or includes submitted text in an exception.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlsplit

import httpx
import numpy as np

from .indexing import (
    BaselineEmbeddingIdentity,
    BaselineIndexBuilder,
    BaselineIndexBuildResult,
)
from .persistent import PersistentBaselineV1Retriever

BASELINE_EMBEDDING_HTTP_CONTRACT = "baseline-embedding-http.v1"
BASELINE_EMBEDDING_HTTP_PROVIDER = "baseline_http_v1"
BASELINE_EMBEDDING_HEALTH_PATH = "/v1/health"
BASELINE_EMBEDDING_VECTOR_PATH = "/v1/embeddings"
MIN_BASELINE_EMBEDDING_TIMEOUT_SECONDS = 0.1
MAX_BASELINE_EMBEDDING_TIMEOUT_SECONDS = 60.0
MAX_BASELINE_EMBEDDING_BATCH_SIZE = 256
MAX_BASELINE_EMBEDDING_DIMENSION = 8192

ClientFactory = Callable[[], httpx.Client]

_SAFE_LOCAL_SERVICE_REASONS = {
    "baseline_model_absent",
    "baseline_model_artifact_changed",
    "baseline_model_artifact_hash_mismatch",
    "baseline_model_artifact_mismatch",
    "baseline_model_artifact_unsafe",
    "baseline_model_cache_unsafe",
    "baseline_model_cache_unavailable",
    "baseline_model_manifest_mismatch",
    "baseline_embedding_model_unavailable",
    "baseline_embedding_runtime_unavailable",
    "baseline_embedding_runtime_version_mismatch",
}


class BaselineEmbeddingCapabilityStatus(str, Enum):
    DISABLED = "disabled"
    READY = "ready"
    UNAVAILABLE = "unavailable"


class BaselineEmbeddingAdapterError(RuntimeError):
    """Sanitized machine-readable adapter failure."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class BaselineEmbeddingConfig:
    """Validated baseline-only adapter configuration."""

    provider_mode: str
    endpoint: str | None = field(repr=False)
    model: str | None
    revision: str | None
    dimension: int
    timeout_seconds: float
    batch_size: int
    allow_insecure_loopback: bool

    @property
    def enabled(self) -> bool:
        return self.provider_mode == "http"

    @classmethod
    def from_settings(cls, settings: Any) -> BaselineEmbeddingConfig:
        config = cls(
            provider_mode=str(
                getattr(settings, "baseline_embedding_provider", "disabled")
            ),
            endpoint=getattr(settings, "baseline_embedding_endpoint", None),
            model=getattr(settings, "baseline_embedding_model", None),
            revision=getattr(settings, "baseline_embedding_revision", None),
            dimension=getattr(settings, "baseline_embedding_dimension", 384),
            timeout_seconds=getattr(
                settings,
                "baseline_embedding_timeout_seconds",
                10.0,
            ),
            batch_size=getattr(settings, "baseline_embedding_batch_size", 32),
            allow_insecure_loopback=bool(
                getattr(
                    settings,
                    "baseline_embedding_allow_insecure_loopback",
                    False,
                )
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.provider_mode not in {"disabled", "http"}:
            raise BaselineEmbeddingAdapterError(
                "embedding_provider_mode_invalid",
                "baseline embedding provider mode is invalid",
            )
        if not isinstance(self.dimension, int) or isinstance(self.dimension, bool):
            raise BaselineEmbeddingAdapterError(
                "embedding_dimension_invalid",
                "baseline embedding dimension is invalid",
            )
        if not 1 <= self.dimension <= MAX_BASELINE_EMBEDDING_DIMENSION:
            raise BaselineEmbeddingAdapterError(
                "embedding_dimension_invalid",
                "baseline embedding dimension is outside the supported range",
            )
        if not isinstance(self.batch_size, int) or isinstance(self.batch_size, bool):
            raise BaselineEmbeddingAdapterError(
                "embedding_batch_size_invalid",
                "baseline embedding batch size is invalid",
            )
        if not 1 <= self.batch_size <= MAX_BASELINE_EMBEDDING_BATCH_SIZE:
            raise BaselineEmbeddingAdapterError(
                "embedding_batch_size_invalid",
                "baseline embedding batch size is outside the supported range",
            )
        if not isinstance(self.timeout_seconds, (int, float)) or isinstance(
            self.timeout_seconds,
            bool,
        ):
            raise BaselineEmbeddingAdapterError(
                "embedding_timeout_invalid",
                "baseline embedding timeout is invalid",
            )
        if not (
            MIN_BASELINE_EMBEDDING_TIMEOUT_SECONDS
            <= float(self.timeout_seconds)
            <= MAX_BASELINE_EMBEDDING_TIMEOUT_SECONDS
        ):
            raise BaselineEmbeddingAdapterError(
                "embedding_timeout_invalid",
                "baseline embedding timeout is outside the supported range",
            )
        if not self.enabled:
            return
        _required_identifier(self.model, "embedding_model_missing")
        _required_identifier(self.revision, "embedding_revision_missing")
        _validate_endpoint(
            self.endpoint,
            allow_insecure_loopback=self.allow_insecure_loopback,
        )


@dataclass(frozen=True, slots=True)
class BaselineEmbeddingCapability:
    """Endpoint- and credential-free baseline adapter capability."""

    status: BaselineEmbeddingCapabilityStatus
    reason: str
    provider_mode: str
    provider: str | None
    model: str | None
    revision: str | None
    dimension: int | None
    fingerprint: str | None
    transport: str
    contract_version: str = BASELINE_EMBEDDING_HTTP_CONTRACT

    @property
    def available(self) -> bool:
        return self.status is BaselineEmbeddingCapabilityStatus.READY

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "provider_mode": self.provider_mode,
            "provider": self.provider,
            "contract_version": self.contract_version,
            "model": self.model,
            "revision": self.revision,
            "dimension": self.dimension,
            "fingerprint": self.fingerprint,
            "transport": self.transport,
        }


def _required_identifier(value: str | None, code: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise BaselineEmbeddingAdapterError(
            code,
            "required baseline embedding identity is missing or invalid",
        )
    return value


def _is_loopback_host(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_endpoint(
    endpoint: str | None,
    *,
    allow_insecure_loopback: bool,
) -> str:
    if (
        not isinstance(endpoint, str)
        or not endpoint
        or endpoint != endpoint.strip()
        or any(ord(character) < 32 for character in endpoint)
    ):
        raise BaselineEmbeddingAdapterError(
            "embedding_endpoint_missing",
            "baseline embedding endpoint is missing or invalid",
        )
    try:
        parsed = urlsplit(endpoint)
        host = parsed.hostname
        _ = parsed.port
    except ValueError:
        raise BaselineEmbeddingAdapterError(
            "embedding_endpoint_invalid",
            "baseline embedding endpoint is invalid",
        ) from None
    if (
        parsed.scheme not in {"http", "https"}
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise BaselineEmbeddingAdapterError(
            "embedding_endpoint_invalid",
            "baseline embedding endpoint is invalid",
        )
    if parsed.scheme == "http":
        if not _is_loopback_host(host):
            raise BaselineEmbeddingAdapterError(
                "embedding_remote_plaintext_rejected",
                "plaintext baseline embedding transport is restricted to loopback",
            )
        if not allow_insecure_loopback:
            raise BaselineEmbeddingAdapterError(
                "embedding_loopback_http_not_enabled",
                "loopback HTTP requires the explicit local deployment setting",
            )
    return endpoint.rstrip("/")


def _identity_fingerprint(model: str, revision: str, dimension: int) -> str:
    payload = {
        "contract_version": BASELINE_EMBEDDING_HTTP_CONTRACT,
        "dimension": dimension,
        "model": model,
        "provider": BASELINE_EMBEDDING_HTTP_PROVIDER,
        "revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class HTTPBaselineEmbeddingAdapter:
    """Strict local/HTTPS client for the baseline embedding service contract."""

    provider = BASELINE_EMBEDDING_HTTP_PROVIDER

    def __init__(
        self,
        config: BaselineEmbeddingConfig,
        *,
        client_factory: ClientFactory | None = None,
    ) -> None:
        config.validate()
        if not config.enabled:
            raise BaselineEmbeddingAdapterError(
                "embedding_provider_disabled",
                "baseline embedding provider is disabled",
            )
        assert config.model is not None
        assert config.revision is not None
        self.model = config.model
        self.revision = config.revision
        self.dimension = config.dimension
        self.fingerprint = _identity_fingerprint(
            self.model,
            self.revision,
            self.dimension,
        )
        self._endpoint = _validate_endpoint(
            config.endpoint,
            allow_insecure_loopback=config.allow_insecure_loopback,
        )
        self._batch_size = config.batch_size
        self._timeout_seconds = float(config.timeout_seconds)
        self._client_factory = client_factory or self._new_client

    @property
    def identity(self) -> BaselineEmbeddingIdentity:
        return BaselineEmbeddingIdentity(
            provider=self.provider,
            model=self.model,
            revision=self.revision,
            dimension=self.dimension,
            fingerprint=self.fingerprint,
        )

    def _new_client(self) -> httpx.Client:
        return httpx.Client(
            timeout=httpx.Timeout(self._timeout_seconds),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "compair-core-baseline-embedding/1",
            },
        )

    def _request_json(
        self,
        client: httpx.Client,
        method: str,
        path: str,
        *,
        payload: dict[str, object] | None = None,
    ) -> object:
        try:
            response = client.request(
                method,
                f"{self._endpoint}{path}",
                json=payload,
            )
        except httpx.TimeoutException:
            raise BaselineEmbeddingAdapterError(
                "embedding_service_timeout",
                "baseline embedding service timed out",
            ) from None
        except httpx.HTTPError:
            raise BaselineEmbeddingAdapterError(
                "embedding_service_unavailable",
                "baseline embedding service is unavailable",
            ) from None
        if response.status_code != 200:
            if path == BASELINE_EMBEDDING_HEALTH_PATH:
                try:
                    error_payload = response.json()
                except ValueError:
                    error_payload = None
                if isinstance(error_payload, dict):
                    safe_reason = error_payload.get("reason")
                    if safe_reason in _SAFE_LOCAL_SERVICE_REASONS:
                        raise BaselineEmbeddingAdapterError(
                            str(safe_reason),
                            "baseline embedding service is not ready",
                        )
            raise BaselineEmbeddingAdapterError(
                "embedding_service_unavailable",
                "baseline embedding service returned an unsuccessful status",
            )
        try:
            return response.json()
        except ValueError:
            raise BaselineEmbeddingAdapterError(
                "embedding_response_malformed",
                "baseline embedding service returned malformed JSON",
            ) from None

    def _validate_identity_payload(self, payload: object) -> dict[str, object]:
        if not isinstance(payload, dict):
            raise BaselineEmbeddingAdapterError(
                "embedding_response_malformed",
                "baseline embedding service returned a malformed response",
            )
        if payload.get("contract_version") != BASELINE_EMBEDDING_HTTP_CONTRACT:
            raise BaselineEmbeddingAdapterError(
                "embedding_contract_mismatch",
                "baseline embedding service contract does not match",
            )
        if payload.get("provider") != self.provider:
            raise BaselineEmbeddingAdapterError(
                "embedding_provider_mismatch",
                "baseline embedding service provider does not match",
            )
        if payload.get("model") != self.model:
            raise BaselineEmbeddingAdapterError(
                "embedding_model_mismatch",
                "baseline embedding service model does not match",
            )
        if payload.get("revision") != self.revision:
            raise BaselineEmbeddingAdapterError(
                "embedding_revision_mismatch",
                "baseline embedding service revision does not match",
            )
        dimension = payload.get("dimension")
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension != self.dimension
        ):
            raise BaselineEmbeddingAdapterError(
                "embedding_dimension_mismatch",
                "baseline embedding service dimension does not match",
            )
        return payload

    def _attest(self, client: httpx.Client) -> None:
        payload = self._validate_identity_payload(
            self._request_json(client, "GET", BASELINE_EMBEDDING_HEALTH_PATH)
        )
        if payload.get("status") != "ok":
            raise BaselineEmbeddingAdapterError(
                "embedding_service_not_ready",
                "baseline embedding service is not ready",
            )

    def attest(self) -> BaselineEmbeddingIdentity:
        try:
            with self._client_factory() as client:
                self._attest(client)
        except BaselineEmbeddingAdapterError:
            raise
        except Exception:  # noqa: BLE001 - injected client lifecycle boundary
            raise BaselineEmbeddingAdapterError(
                "embedding_service_unavailable",
                "baseline embedding service is unavailable",
            ) from None
        return self.identity

    def _embedding_payload(self, texts: Sequence[str]) -> dict[str, object]:
        return {
            "contract_version": BASELINE_EMBEDDING_HTTP_CONTRACT,
            "provider": self.provider,
            "model": self.model,
            "revision": self.revision,
            "dimension": self.dimension,
            "texts": list(texts),
        }

    def _vectors_from_payload(
        self,
        payload: object,
        *,
        expected_count: int,
    ) -> tuple[np.ndarray, ...]:
        response = self._validate_identity_payload(payload)
        vectors = response.get("vectors")
        if not isinstance(vectors, list):
            raise BaselineEmbeddingAdapterError(
                "embedding_response_malformed",
                "baseline embedding response has no vector list",
            )
        if len(vectors) != expected_count:
            raise BaselineEmbeddingAdapterError(
                "embedding_vector_count_mismatch",
                "baseline embedding response has the wrong vector count",
            )
        parsed: list[np.ndarray] = []
        for raw_vector in vectors:
            if not isinstance(raw_vector, list):
                raise BaselineEmbeddingAdapterError(
                    "embedding_vector_invalid",
                    "baseline embedding response contains an invalid vector",
                )
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    vector = np.asarray(raw_vector, dtype="<f4")
            except (TypeError, ValueError):
                raise BaselineEmbeddingAdapterError(
                    "embedding_vector_invalid",
                    "baseline embedding response contains an invalid vector",
                ) from None
            if vector.shape != (self.dimension,):
                raise BaselineEmbeddingAdapterError(
                    "embedding_dimension_mismatch",
                    "baseline embedding response vector has the wrong dimension",
                )
            if not np.isfinite(vector).all():
                raise BaselineEmbeddingAdapterError(
                    "embedding_vector_nonfinite",
                    "baseline embedding response contains a non-finite vector",
                )
            parsed.append(vector.copy())
        return tuple(parsed)

    def embed(self, texts: Sequence[str]) -> tuple[np.ndarray, ...]:
        """Embed in order using finite float32 vectors and no normalization."""

        submitted = tuple(texts)
        if any(not isinstance(text, str) for text in submitted):
            raise BaselineEmbeddingAdapterError(
                "embedding_input_invalid",
                "baseline embedding input must contain only text values",
            )
        output: list[np.ndarray] = []
        try:
            with self._client_factory() as client:
                self._attest(client)
                for start in range(0, len(submitted), self._batch_size):
                    batch = submitted[start : start + self._batch_size]
                    payload = self._request_json(
                        client,
                        "POST",
                        BASELINE_EMBEDDING_VECTOR_PATH,
                        payload=self._embedding_payload(batch),
                    )
                    output.extend(
                        self._vectors_from_payload(
                            payload,
                            expected_count=len(batch),
                        )
                    )
        except BaselineEmbeddingAdapterError:
            raise
        except Exception:  # noqa: BLE001 - injected client lifecycle boundary
            raise BaselineEmbeddingAdapterError(
                "embedding_service_unavailable",
                "baseline embedding service is unavailable",
            ) from None
        return tuple(output)


def configured_baseline_embedding_adapter(
    settings: Any,
    *,
    client_factory: ClientFactory | None = None,
) -> HTTPBaselineEmbeddingAdapter | None:
    """Return the configured production adapter, or ``None`` when disabled."""

    config = BaselineEmbeddingConfig.from_settings(settings)
    if not config.enabled:
        return None
    return HTTPBaselineEmbeddingAdapter(config, client_factory=client_factory)


def require_configured_baseline_embedding_adapter(
    settings: Any,
    *,
    client_factory: ClientFactory | None = None,
) -> HTTPBaselineEmbeddingAdapter:
    adapter = configured_baseline_embedding_adapter(
        settings,
        client_factory=client_factory,
    )
    if adapter is None:
        raise BaselineEmbeddingAdapterError(
            "embedding_provider_disabled",
            "baseline embedding provider is disabled",
        )
    return adapter


def build_configured_baseline_index(
    session_factory: Any,
    *,
    settings: Any,
    generation_id: str,
    index_version: str,
    client_factory: ClientFactory | None = None,
    publish_index: Callable[[Any, str], None] | None = None,
) -> BaselineIndexBuildResult:
    """Build using only the separately configured, attested baseline adapter."""

    adapter = require_configured_baseline_embedding_adapter(
        settings,
        client_factory=client_factory,
    )
    adapter.attest()
    return BaselineIndexBuilder(
        session_factory,
        publish_index=publish_index,
    ).build(
        generation_id=generation_id,
        index_version=index_version,
        embedding=adapter.identity,
        provider=adapter,
    )


def create_configured_persistent_baseline_retriever(
    session_factory: Any,
    *,
    settings: Any,
    client_factory: ClientFactory | None = None,
    evidence_filter: Callable[[Any], bool] | None = None,
) -> PersistentBaselineV1Retriever:
    """Create the persistent reader with the same baseline-only identity."""

    adapter = configured_baseline_embedding_adapter(
        settings,
        client_factory=client_factory,
    )
    return PersistentBaselineV1Retriever(
        session_factory,
        adapter,
        evidence_filter=evidence_filter,
    )


def assess_baseline_embedding(
    settings: Any,
    *,
    client_factory: ClientFactory | None = None,
) -> BaselineEmbeddingCapability:
    """Return a sanitized live capability without exposing the endpoint."""

    try:
        config = BaselineEmbeddingConfig.from_settings(settings)
    except BaselineEmbeddingAdapterError as exc:
        return BaselineEmbeddingCapability(
            status=BaselineEmbeddingCapabilityStatus.UNAVAILABLE,
            reason=exc.code,
            provider_mode=str(
                getattr(settings, "baseline_embedding_provider", "disabled")
            ),
            provider=None,
            model=getattr(settings, "baseline_embedding_model", None),
            revision=getattr(settings, "baseline_embedding_revision", None),
            dimension=getattr(settings, "baseline_embedding_dimension", None),
            fingerprint=None,
            transport="unavailable",
        )
    if not config.enabled:
        return BaselineEmbeddingCapability(
            status=BaselineEmbeddingCapabilityStatus.DISABLED,
            reason="embedding_provider_disabled",
            provider_mode=config.provider_mode,
            provider=None,
            model=config.model,
            revision=config.revision,
            dimension=config.dimension,
            fingerprint=None,
            transport="disabled",
        )
    adapter = HTTPBaselineEmbeddingAdapter(
        config,
        client_factory=client_factory,
    )
    try:
        adapter.attest()
    except BaselineEmbeddingAdapterError as exc:
        return BaselineEmbeddingCapability(
            status=BaselineEmbeddingCapabilityStatus.UNAVAILABLE,
            reason=exc.code,
            provider_mode=config.provider_mode,
            provider=adapter.provider,
            model=config.model,
            revision=config.revision,
            dimension=config.dimension,
            fingerprint=adapter.fingerprint,
            transport=(
                "explicit_loopback_http"
                if config.endpoint and urlsplit(config.endpoint).scheme == "http"
                else "https"
            ),
        )
    return BaselineEmbeddingCapability(
        status=BaselineEmbeddingCapabilityStatus.READY,
        reason="identity_attested",
        provider_mode=config.provider_mode,
        provider=adapter.provider,
        model=adapter.model,
        revision=adapter.revision,
        dimension=adapter.dimension,
        fingerprint=adapter.fingerprint,
        transport=(
            "explicit_loopback_http"
            if config.endpoint and urlsplit(config.endpoint).scheme == "http"
            else "https"
        ),
    )
