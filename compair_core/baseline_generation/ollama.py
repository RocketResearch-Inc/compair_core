"""Fail-closed native Ollama provider for document-level baseline generation.

The client never logs request/response bodies and never asks Ollama to pull a
model.  A configured mutable tag is usable only when ``/api/tags`` attests the
exact expected immutable digest before every evidence-bearing request.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any
from urllib.parse import urlsplit

import httpx

from ..compair.retrieval.generation import (
    GENERATION_OUTPUT_SCHEMA_SHA256,
    GENERATION_OUTPUT_SCHEMA_VERSION,
    GENERATION_OUTPUT_SPEC_SHA256,
    MAX_GENERATION_OUTPUT_CHARACTERS,
    BaselineGenerationInput,
    BaselineGenerationProviderError,
    BaselineGenerationService,
)

OLLAMA_GENERATION_ADAPTER_CONTRACT = "baseline-generation-ollama-http.v1"
OLLAMA_PROVIDER = "ollama"
OLLAMA_VERSION_PATH = "/api/version"
OLLAMA_TAGS_PATH = "/api/tags"
OLLAMA_CHAT_PATH = "/api/chat"
MINIMUM_OLLAMA_VERSION = (0, 32, 13)
DEFAULT_MAX_REQUEST_BYTES = 256_000
DEFAULT_MAX_RESPONSE_BYTES = 200_000
MAX_ATTESTATION_RESPONSE_BYTES = 1_000_000
MAX_REFERENCE_COUNT = 4
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_RUNTIME_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:[-+][0-9A-Za-z.-]+)?$")
_READINESS_CODES = frozenset(
    {
        "provider_unconfigured",
        "endpoint_unavailable",
        "insecure_transport",
        "unsupported_runtime",
        "model_absent",
        "digest_mismatch",
        "structured_output_unavailable",
        "ready",
    }
)

ClientFactory = Callable[[], httpx.Client]


def _provider_error(
    code: str, *, retryable: bool = False
) -> BaselineGenerationProviderError:
    return BaselineGenerationProviderError(
        code,
        "baseline Ollama generation is unavailable",
        retryable=retryable,
    )


def _strict_json_loads(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        decoded = raw.decode("utf-8", errors="strict")
        return json.loads(
            decoded,
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("non-finite number")
            ),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise _provider_error("structured_output_unavailable") from None


def _safe_identity(value: object, *, maximum: int = 256) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(ord(character) < 32 for character in value)
    ):
        raise _provider_error("provider_unconfigured")
    return value


def _literal_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_baseline_generation_endpoint(
    endpoint: object,
    *,
    allow_loopback_http: bool,
    require_root_path: bool,
) -> str:
    if (
        not isinstance(endpoint, str)
        or not endpoint
        or endpoint != endpoint.strip()
        or any(ord(character) < 32 for character in endpoint)
    ):
        raise _provider_error("provider_unconfigured")
    try:
        parsed = urlsplit(endpoint)
        host = parsed.hostname
        _ = parsed.port
    except ValueError:
        raise _provider_error("provider_unconfigured") from None
    if (
        parsed.scheme not in {"http", "https"}
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (require_root_path and parsed.path not in {"", "/"})
    ):
        raise _provider_error("insecure_transport")
    if parsed.scheme == "http" and (
        not allow_loopback_http or not _literal_loopback(host)
    ):
        raise _provider_error("insecure_transport")
    return endpoint.rstrip("/")


def _schema_bytes() -> bytes:
    raw = (
        files("compair_core.baseline_generation")
        .joinpath("baseline-generation-output.v2.schema.json")
        .read_bytes()
    )
    if hashlib.sha256(raw).hexdigest() != GENERATION_OUTPUT_SCHEMA_SHA256:
        raise _provider_error("structured_output_unavailable")
    return raw


@dataclass(frozen=True, slots=True)
class OllamaGenerationConfig:
    provider_mode: str
    endpoint: str | None = field(repr=False)
    model: str | None
    expected_digest: str | None
    timeout_seconds: float
    allow_loopback_http: bool
    maximum_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES
    maximum_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    context_tokens: int = 32_768
    output_tokens: int = 1_024
    seed: int = 0

    @classmethod
    def from_settings(cls, settings: Any) -> OllamaGenerationConfig:
        config = cls(
            provider_mode=str(
                getattr(settings, "baseline_generation_provider", "disabled")
            ).lower(),
            endpoint=getattr(settings, "baseline_generation_endpoint", None),
            model=getattr(settings, "baseline_generation_model", None),
            expected_digest=getattr(settings, "baseline_generation_model_digest", None),
            timeout_seconds=float(
                getattr(settings, "baseline_generation_timeout_seconds", 60.0)
            ),
            allow_loopback_http=bool(
                getattr(
                    settings,
                    "baseline_generation_allow_loopback_http",
                    False,
                )
            ),
            maximum_request_bytes=int(
                getattr(
                    settings,
                    "baseline_generation_max_request_bytes",
                    DEFAULT_MAX_REQUEST_BYTES,
                )
            ),
            maximum_response_bytes=int(
                getattr(
                    settings,
                    "baseline_generation_max_response_bytes",
                    DEFAULT_MAX_RESPONSE_BYTES,
                )
            ),
            context_tokens=int(
                getattr(settings, "baseline_generation_context_tokens", 32_768)
            ),
            output_tokens=int(
                getattr(settings, "baseline_generation_output_tokens", 1_024)
            ),
            seed=int(getattr(settings, "baseline_generation_seed", 0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.provider_mode != OLLAMA_PROVIDER:
            raise _provider_error("provider_unconfigured")
        _safe_identity(self.model)
        digest = _safe_identity(self.expected_digest, maximum=71)
        if _DIGEST.fullmatch(digest) is None:
            raise _provider_error("provider_unconfigured")
        validate_baseline_generation_endpoint(
            self.endpoint,
            allow_loopback_http=self.allow_loopback_http,
            require_root_path=True,
        )
        if not 0.1 <= self.timeout_seconds <= 300.0:
            raise _provider_error("provider_unconfigured")
        if not 4_096 <= self.maximum_request_bytes <= 8_000_000:
            raise _provider_error("provider_unconfigured")
        if not 4_096 <= self.maximum_response_bytes <= 1_000_000:
            raise _provider_error("provider_unconfigured")
        if not 2_048 <= self.context_tokens <= 131_072:
            raise _provider_error("provider_unconfigured")
        if not 64 <= self.output_tokens <= 4_096:
            raise _provider_error("provider_unconfigured")
        if self.output_tokens >= self.context_tokens:
            raise _provider_error("provider_unconfigured")
        if not 0 <= self.seed <= 2_147_483_647:
            raise _provider_error("provider_unconfigured")


@dataclass(frozen=True, slots=True)
class OllamaGenerationIdentity:
    provider: str
    adapter_contract: str
    model: str
    digest: str
    runtime_version: str
    output_schema_version: str
    output_spec_sha256: str
    output_schema_sha256: str
    supports_idempotency: bool
    fingerprint: str


@dataclass(frozen=True, slots=True)
class OllamaGenerationReadiness:
    status: str
    ready: bool
    provider: str | None
    model: str | None
    expected_digest: str | None
    runtime_version: str | None
    identity_fingerprint: str | None
    probe_performed: bool
    probe_outcome: str | None
    contract_version: str = OLLAMA_GENERATION_ADAPTER_CONTRACT
    output_schema_version: str = GENERATION_OUTPUT_SCHEMA_VERSION
    output_spec_sha256: str = GENERATION_OUTPUT_SPEC_SHA256
    output_schema_sha256: str = GENERATION_OUTPUT_SCHEMA_SHA256
    supports_idempotency: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": "baseline-generation-readiness.v1",
            "status": self.status,
            "ready": self.ready,
            "provider": self.provider,
            "contract_version": self.contract_version,
            "model": self.model,
            "expected_digest": self.expected_digest,
            "runtime_version": self.runtime_version,
            "identity_fingerprint": self.identity_fingerprint,
            "output_schema_version": self.output_schema_version,
            "output_spec_sha256": self.output_spec_sha256,
            "output_schema_sha256": self.output_schema_sha256,
            "supports_idempotency": self.supports_idempotency,
            "probe_performed": self.probe_performed,
            "probe_outcome": self.probe_outcome,
        }


class OllamaBaselineGenerationProvider:
    """Native nonstreaming ``/api/chat`` baseline provider."""

    provider = OLLAMA_PROVIDER
    supports_idempotency = False

    def __init__(
        self,
        config: OllamaGenerationConfig,
        *,
        client_factory: ClientFactory | None = None,
    ) -> None:
        config.validate()
        assert config.model is not None
        assert config.expected_digest is not None
        self.model = config.model
        self._expected_digest = config.expected_digest
        self._endpoint = validate_baseline_generation_endpoint(
            config.endpoint,
            allow_loopback_http=config.allow_loopback_http,
            require_root_path=True,
        )
        self._timeout = float(config.timeout_seconds)
        self._maximum_request_bytes = config.maximum_request_bytes
        self._maximum_response_bytes = config.maximum_response_bytes
        self._context_tokens = config.context_tokens
        self._output_tokens = config.output_tokens
        self._seed = config.seed
        self._schema_raw = _schema_bytes()
        self._schema = _strict_json_loads(self._schema_raw)
        self._client_factory = client_factory or self._new_client
        self._identity: OllamaGenerationIdentity | None = None

    @property
    def version(self) -> str:
        if self._identity is None:
            raise _provider_error("endpoint_unavailable", retryable=True)
        return (
            f"{self._identity.adapter_contract};runtime="
            f"{self._identity.runtime_version};digest={self._identity.digest};"
            f"spec={GENERATION_OUTPUT_SPEC_SHA256}"
        )

    @property
    def identity(self) -> OllamaGenerationIdentity:
        if self._identity is None:
            raise _provider_error("endpoint_unavailable", retryable=True)
        return self._identity

    def _new_client(self) -> httpx.Client:
        # HTTPX logs full request URLs at INFO and HTTP Core may expose them at
        # DEBUG. Baseline provider endpoints are protected deployment details.
        for logger_name in ("httpx", "httpcore"):
            logger = logging.getLogger(logger_name)
            if logger.getEffectiveLevel() < logging.WARNING:
                logger.setLevel(logging.WARNING)
        connect = min(5.0, self._timeout)
        return httpx.Client(
            timeout=httpx.Timeout(
                self._timeout,
                connect=connect,
                read=self._timeout,
                write=min(10.0, self._timeout),
                pool=connect,
            ),
            follow_redirects=False,
            trust_env=False,
            verify=True,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": "compair-core-baseline-generation/1",
            },
        )

    def _request_bytes(
        self,
        client: httpx.Client,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        maximum_bytes: int,
        chat: bool = False,
    ) -> bytes:
        started = time.monotonic()
        try:
            with client.stream(
                method,
                f"{self._endpoint}{path}",
                content=body,
            ) as response:
                if response.status_code != 200:
                    if chat and response.status_code in {400, 404, 422}:
                        raise _provider_error("structured_output_unavailable")
                    if (
                        response.status_code in {408, 425, 429}
                        or response.status_code >= 500
                    ):
                        raise _provider_error("endpoint_unavailable", retryable=True)
                    raise _provider_error("endpoint_unavailable")
                content = bytearray()
                for block in response.iter_bytes():
                    content.extend(block)
                    if len(content) > maximum_bytes:
                        raise _provider_error("provider_response_too_large")
                    if time.monotonic() - started > self._timeout:
                        raise _provider_error("endpoint_unavailable", retryable=True)
                return bytes(content)
        except BaselineGenerationProviderError:
            raise
        except httpx.TimeoutException:
            raise _provider_error("endpoint_unavailable", retryable=True) from None
        except httpx.HTTPError:
            raise _provider_error("endpoint_unavailable", retryable=True) from None

    def _request_json(
        self,
        client: httpx.Client,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        maximum_bytes: int,
        chat: bool = False,
    ) -> object:
        raw = self._request_bytes(
            client,
            method,
            path,
            body=body,
            maximum_bytes=maximum_bytes,
            chat=chat,
        )
        return _strict_json_loads(raw)

    def _attest_with_client(self, client: httpx.Client) -> OllamaGenerationIdentity:
        version_payload = self._request_json(
            client,
            "GET",
            OLLAMA_VERSION_PATH,
            maximum_bytes=MAX_ATTESTATION_RESPONSE_BYTES,
        )
        if not isinstance(version_payload, dict):
            raise _provider_error("unsupported_runtime")
        runtime = version_payload.get("version")
        if not isinstance(runtime, str):
            raise _provider_error("unsupported_runtime")
        match = _RUNTIME_VERSION.fullmatch(runtime)
        if match is None or tuple(int(value) for value in match.groups()) < (
            MINIMUM_OLLAMA_VERSION
        ):
            raise _provider_error("unsupported_runtime")

        tags_payload = self._request_json(
            client,
            "GET",
            OLLAMA_TAGS_PATH,
            maximum_bytes=MAX_ATTESTATION_RESPONSE_BYTES,
        )
        if not isinstance(tags_payload, dict) or not isinstance(
            tags_payload.get("models"), list
        ):
            raise _provider_error("endpoint_unavailable", retryable=True)
        selected: dict[str, object] | None = None
        for candidate in tags_payload["models"]:
            if not isinstance(candidate, dict):
                continue
            if (
                candidate.get("name") == self.model
                or candidate.get("model") == self.model
            ):
                selected = candidate
                break
        if selected is None:
            raise _provider_error("model_absent")
        digest = selected.get("digest")
        if not isinstance(digest, str):
            raise _provider_error("digest_mismatch")
        attested_digest = digest if digest.startswith("sha256:") else f"sha256:{digest}"
        if attested_digest != self._expected_digest:
            raise _provider_error("digest_mismatch")

        fingerprint_payload = {
            "adapter_contract": OLLAMA_GENERATION_ADAPTER_CONTRACT,
            "digest": attested_digest,
            "model": self.model,
            "output_schema_sha256": GENERATION_OUTPUT_SCHEMA_SHA256,
            "output_spec_sha256": GENERATION_OUTPUT_SPEC_SHA256,
            "provider": OLLAMA_PROVIDER,
            "runtime_version": runtime,
            "supports_idempotency": False,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return OllamaGenerationIdentity(
            provider=OLLAMA_PROVIDER,
            adapter_contract=OLLAMA_GENERATION_ADAPTER_CONTRACT,
            model=self.model,
            digest=attested_digest,
            runtime_version=runtime,
            output_schema_version=GENERATION_OUTPUT_SCHEMA_VERSION,
            output_spec_sha256=GENERATION_OUTPUT_SPEC_SHA256,
            output_schema_sha256=GENERATION_OUTPUT_SCHEMA_SHA256,
            supports_idempotency=False,
            fingerprint=fingerprint,
        )

    def attest(self) -> OllamaGenerationIdentity:
        with self._client_factory() as client:
            identity = self._attest_with_client(client)
        if self._identity is not None and identity != self._identity:
            if identity.digest != self._identity.digest:
                raise _provider_error("digest_mismatch")
            raise _provider_error("unsupported_runtime")
        self._identity = identity
        return identity

    def _chat(
        self,
        *,
        source_text: str,
        evidence: Sequence[str],
        maximum_findings: int,
    ) -> str:
        system = (
            "Review the changed source using only the ordered evidence. Return exactly "
            "one JSON object matching the supplied schema. If there is no concrete "
            "finding, return outcome no_findings with findings as an empty array. If "
            "there are findings, return outcome findings with between one and "
            f"{maximum_findings} nonblank feedback strings. Never use NONE or an empty "
            "string as feedback. Report only concrete correctness or security defects "
            "supported by the evidence. Insufficient context, stylistic preferences, "
            "missing comments or docstrings, and suggested enhancements are not "
            "findings. For example, a function that returns the sum of its two inputs "
            "with matching evidence has no concrete finding and must use no_findings "
            "with an empty findings array. Preserve finding order. Do not return "
            "markdown or extra prose."
        )
        evidence_sections = [
            f"Ordered evidence {ordinal}:\n{renderer_output}"
            for ordinal, renderer_output in enumerate(evidence, start=1)
        ]
        user = (
            "Authoritative changed source document:\n"
            + source_text
            + "\n\n"
            + "\n\n".join(evidence_sections)
        )
        available_input_tokens = self._context_tokens - self._output_tokens
        if len(system.encode("utf-8")) + len(user.encode("utf-8")) > (
            available_input_tokens
        ):
            raise _provider_error("provider_request_too_large")
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": self._schema,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {
                "temperature": 0,
                "seed": self._seed,
                "num_ctx": self._context_tokens,
                "num_predict": self._output_tokens,
            },
        }
        body = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(body) > self._maximum_request_bytes:
            raise _provider_error("provider_request_too_large")
        with self._client_factory() as client:
            response = self._request_json(
                client,
                "POST",
                OLLAMA_CHAT_PATH,
                body=body,
                maximum_bytes=self._maximum_response_bytes,
                chat=True,
            )
        if not isinstance(response, dict):
            raise _provider_error("structured_output_unavailable")
        if response.get("model") != self.model or response.get("done") is not True:
            raise _provider_error("structured_output_unavailable")
        if response.get("done_reason") == "length":
            raise _provider_error("structured_output_unavailable")
        message = response.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise _provider_error("structured_output_unavailable")
        output = message["content"]
        if len(output) > MAX_GENERATION_OUTPUT_CHARACTERS:
            raise _provider_error("provider_response_too_large")
        return output

    def generate(
        self,
        generation_input: BaselineGenerationInput,
        *,
        idempotency_key: str,
    ) -> str:
        del idempotency_key  # Ollama has no channel-side idempotency contract.
        ordinals = [item.ordinal for item in generation_input.evidence]
        if not 1 <= len(ordinals) <= MAX_REFERENCE_COUNT or ordinals != list(
            range(1, len(ordinals) + 1)
        ):
            raise _provider_error("provider_request_invalid")
        self.attest()  # Reattest before sending source/evidence bytes.
        return self._chat(
            source_text=generation_input.source_text,
            evidence=[item.renderer_output for item in generation_input.evidence],
            maximum_findings=len(generation_input.evidence),
        )

    def probe(self) -> str:
        self.attest()
        output = self._chat(
            source_text="Synthetic compatibility document with no private content.",
            evidence=(
                (
                    "Repository file: synthetic/example.txt\n\n"
                    "Synthetic compatibility evidence with no finding."
                ),
            ),
            maximum_findings=1,
        )
        findings, _fingerprint = BaselineGenerationService._parse_output(
            output,
            maximum_findings=1,
        )
        parsed = _strict_json_loads(output.encode("utf-8"))
        if not isinstance(parsed, dict) or parsed.get("outcome") not in {
            "no_findings",
            "findings",
        }:
            raise _provider_error("structured_output_unavailable")
        if parsed["outcome"] == "findings" and not findings:
            raise _provider_error("structured_output_unavailable")
        return str(parsed["outcome"])


def verify_ollama_generation(
    settings: Any,
    *,
    probe: bool = False,
    client_factory: ClientFactory | None = None,
) -> OllamaGenerationReadiness:
    provider: OllamaBaselineGenerationProvider | None = None
    try:
        config = OllamaGenerationConfig.from_settings(settings)
        provider = OllamaBaselineGenerationProvider(
            config,
            client_factory=client_factory,
        )
        identity = provider.attest()
        outcome = provider.probe() if probe else None
        return OllamaGenerationReadiness(
            status="ready",
            ready=True,
            provider=OLLAMA_PROVIDER,
            model=identity.model,
            expected_digest=identity.digest,
            runtime_version=identity.runtime_version,
            identity_fingerprint=identity.fingerprint,
            probe_performed=probe,
            probe_outcome=outcome,
        )
    except (BaselineGenerationProviderError, TypeError, ValueError) as exc:
        code = getattr(exc, "code", "provider_unconfigured")
        status = code if code in _READINESS_CODES else "structured_output_unavailable"
        return OllamaGenerationReadiness(
            status=status,
            ready=False,
            provider=OLLAMA_PROVIDER if provider is not None else None,
            model=provider.model if provider is not None else None,
            expected_digest=(
                provider._expected_digest if provider is not None else None
            ),
            runtime_version=(
                provider._identity.runtime_version
                if provider is not None and provider._identity is not None
                else None
            ),
            identity_fingerprint=(
                provider._identity.fingerprint
                if provider is not None and provider._identity is not None
                else None
            ),
            probe_performed=probe,
            probe_outcome=None,
        )


__all__ = [
    "DEFAULT_MAX_REQUEST_BYTES",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "MINIMUM_OLLAMA_VERSION",
    "OLLAMA_GENERATION_ADAPTER_CONTRACT",
    "OLLAMA_PROVIDER",
    "OllamaBaselineGenerationProvider",
    "OllamaGenerationConfig",
    "OllamaGenerationIdentity",
    "OllamaGenerationReadiness",
    "validate_baseline_generation_endpoint",
    "verify_ollama_generation",
]
