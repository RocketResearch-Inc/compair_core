"""Fail-closed transport policy for explicit retrieval queries.

The query itself is deliberately absent from every type and message in this
module.  The policy only inspects the effective Celery task/application
configuration that will carry it.
"""

from __future__ import annotations

import ipaddress
import ssl
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import parse_qs, urlsplit

REDACTED_TASK_ARGS_REPR = "(<process-document arguments redacted>)"
REDACTED_TASK_KWARGS_REPR = "{<process-document keyword arguments redacted>}"


class RetrievalQueryTransportStatus(str, Enum):
    SAFE = "safe"
    UNAVAILABLE = "unavailable"
    LOCALLY_OVERRIDDEN = "locally_overridden"


@dataclass(frozen=True, slots=True)
class RetrievalQueryTransportCapability:
    status: RetrievalQueryTransportStatus
    reason: str
    transport: str
    broker_scheme: str | None = None
    encrypted: bool = False
    authenticated: bool = False
    tls_verified: bool = False
    task_protocol: int | None = None
    task_arguments: str = "unavailable"
    task_sent_events_enabled: bool = False
    worker_task_events_enabled: bool = False
    result_extended: bool = False
    local_override_enabled: bool = False

    @property
    def available(self) -> bool:
        return self.status in {
            RetrievalQueryTransportStatus.SAFE,
            RetrievalQueryTransportStatus.LOCALLY_OVERRIDDEN,
        }

    def as_dict(self) -> dict[str, object]:
        """Return a credential-free capability/health representation."""

        return {
            "status": self.status.value,
            "reason": self.reason,
            "transport": self.transport,
            "broker_scheme": self.broker_scheme,
            "encrypted": self.encrypted,
            "authenticated": self.authenticated,
            "tls_verified": self.tls_verified,
            "task_protocol": self.task_protocol,
            "task_arguments": self.task_arguments,
            "task_sent_events_enabled": self.task_sent_events_enabled,
            "worker_task_events_enabled": self.worker_task_events_enabled,
            "result_extended": self.result_extended,
            "local_override_enabled": self.local_override_enabled,
        }


class RetrievalQueryTransportPolicyError(RuntimeError):
    """Raised before dispatch when explicit-query transport is unsafe."""

    def __init__(self, capability: RetrievalQueryTransportCapability) -> None:
        self.capability = capability
        super().__init__(
            "explicit retrieval-query transport is unavailable "
            f"({capability.reason})"
        )


def _conf_value(conf: object, name: str, default: Any = None) -> Any:
    if isinstance(conf, Mapping):
        return conf.get(name, default)
    return getattr(conf, name, default)


def _task_app(task: object) -> object | None:
    app = getattr(task, "app", None)
    if app is not None:
        return app
    get_app = getattr(task, "_get_app", None)
    if callable(get_app):
        try:
            return get_app()
        except Exception:  # noqa: BLE001 - an unreadable task app must fail closed
            return None
    return None


def _broker_urls(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(";") if part.strip())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(part).strip() for part in value if str(part).strip())
    return ()


def _is_loopback_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _is_local_only_url(url: str) -> bool:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    if scheme == "memory":
        return True
    if scheme == "redis+socket":
        return True
    return _is_loopback_host(parsed.hostname)


def _ssl_mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _is_cert_required(value: object) -> bool:
    if value == ssl.CERT_REQUIRED:
        return True
    normalized = str(value or "").strip().lower()
    return normalized in {"2", "required", "cert_required", "ssl.cert_required"}


def _tls_verified(url: str, broker_use_ssl: object) -> bool:
    query = parse_qs(urlsplit(url).query)
    for key in ("ssl_cert_reqs", "cert_reqs"):
        if key in query and any(_is_cert_required(item) for item in query[key]):
            return True
    ssl_options = _ssl_mapping(broker_use_ssl)
    return any(
        _is_cert_required(ssl_options.get(key))
        for key in ("ssl_cert_reqs", "cert_reqs")
    )


def _authenticated(
    url: str,
    broker_use_ssl: object,
    configured_password: object = None,
) -> bool:
    parsed = urlsplit(url)
    if parsed.password or configured_password:
        return True
    query = parse_qs(parsed.query)
    if any(value for value in query.get("credential_provider", ())):
        return True
    ssl_options = _ssl_mapping(broker_use_ssl)
    return bool(ssl_options.get("certfile") and ssl_options.get("keyfile"))


def _unavailable(
    reason: str,
    *,
    transport: str,
    broker_scheme: str | None = None,
    encrypted: bool = False,
    authenticated: bool = False,
    tls_verified: bool = False,
    task_protocol: int | None = None,
    task_sent_events_enabled: bool = False,
    worker_task_events_enabled: bool = False,
    result_extended: bool = False,
    local_override_enabled: bool = False,
) -> RetrievalQueryTransportCapability:
    return RetrievalQueryTransportCapability(
        status=RetrievalQueryTransportStatus.UNAVAILABLE,
        reason=reason,
        transport=transport,
        broker_scheme=broker_scheme,
        encrypted=encrypted,
        authenticated=authenticated,
        tls_verified=tls_verified,
        task_protocol=task_protocol,
        task_arguments=("redacted" if task_protocol == 2 else "unavailable"),
        task_sent_events_enabled=task_sent_events_enabled,
        worker_task_events_enabled=worker_task_events_enabled,
        result_extended=result_extended,
        local_override_enabled=local_override_enabled,
    )


def assess_retrieval_query_transport(
    task: object,
    *,
    allow_insecure_local_transport: bool = False,
) -> RetrievalQueryTransportCapability:
    """Assess the effective task transport without opening a connection.

    A secure broker must use authenticated TLS with certificate verification,
    Celery protocol v2 (so representations can be replaced), and non-extended
    result metadata (so the backend never persists task arguments).
    """

    apply_async = getattr(task, "apply_async", None)
    app = _task_app(task)
    if app is None and not callable(apply_async):
        if allow_insecure_local_transport:
            return RetrievalQueryTransportCapability(
                status=RetrievalQueryTransportStatus.LOCALLY_OVERRIDDEN,
                reason="explicit_local_direct_execution",
                transport="direct",
                task_arguments="in_process_only",
                local_override_enabled=True,
            )
        return _unavailable(
            "explicit_local_override_required",
            transport="direct",
            local_override_enabled=False,
        )
    if app is None:
        return _unavailable(
            "celery_configuration_unavailable",
            transport="celery",
            local_override_enabled=allow_insecure_local_transport,
        )

    conf = getattr(app, "conf", None)
    if conf is None:
        return _unavailable(
            "celery_configuration_unavailable",
            transport="celery",
            local_override_enabled=allow_insecure_local_transport,
        )

    try:
        task_protocol = int(_conf_value(conf, "task_protocol", 2))
    except (TypeError, ValueError):
        task_protocol = None
    result_extended = bool(_conf_value(conf, "result_extended", False))
    task_sent_events_enabled = bool(
        _conf_value(conf, "task_send_sent_event", False)
    )
    worker_task_events_enabled = bool(
        _conf_value(conf, "worker_send_task_events", False)
    )
    if task_protocol != 2:
        return _unavailable(
            "celery_task_protocol_v2_required",
            transport="celery",
            task_protocol=task_protocol,
            task_sent_events_enabled=task_sent_events_enabled,
            worker_task_events_enabled=worker_task_events_enabled,
            result_extended=result_extended,
            local_override_enabled=allow_insecure_local_transport,
        )
    if result_extended:
        return _unavailable(
            "celery_result_extended_must_be_disabled",
            transport="celery",
            task_protocol=task_protocol,
            task_sent_events_enabled=task_sent_events_enabled,
            worker_task_events_enabled=worker_task_events_enabled,
            result_extended=True,
            local_override_enabled=allow_insecure_local_transport,
        )
    if not callable(apply_async):
        return _unavailable(
            "celery_redacted_dispatch_unavailable",
            transport="celery",
            task_protocol=task_protocol,
            task_sent_events_enabled=task_sent_events_enabled,
            worker_task_events_enabled=worker_task_events_enabled,
            local_override_enabled=allow_insecure_local_transport,
        )

    if bool(_conf_value(conf, "task_always_eager", False)):
        if allow_insecure_local_transport:
            return RetrievalQueryTransportCapability(
                status=RetrievalQueryTransportStatus.LOCALLY_OVERRIDDEN,
                reason="explicit_local_eager_execution",
                transport="celery_eager",
                task_protocol=task_protocol,
                task_arguments="redacted",
                task_sent_events_enabled=task_sent_events_enabled,
                worker_task_events_enabled=worker_task_events_enabled,
                local_override_enabled=True,
            )
        return _unavailable(
            "explicit_local_override_required",
            transport="celery_eager",
            task_protocol=task_protocol,
            task_sent_events_enabled=task_sent_events_enabled,
            worker_task_events_enabled=worker_task_events_enabled,
        )

    write_url = _conf_value(conf, "broker_write_url")
    urls = _broker_urls(write_url or _conf_value(conf, "broker_url"))
    if not urls:
        return _unavailable(
            "celery_broker_url_unavailable",
            transport="celery",
            task_protocol=task_protocol,
            task_sent_events_enabled=task_sent_events_enabled,
            worker_task_events_enabled=worker_task_events_enabled,
            local_override_enabled=allow_insecure_local_transport,
        )

    schemes = {urlsplit(url).scheme.lower() for url in urls}
    broker_scheme = next(iter(schemes)) if len(schemes) == 1 else "multiple"
    broker_use_ssl = _conf_value(conf, "broker_use_ssl", False)
    broker_password = _conf_value(conf, "broker_password")
    encrypted = all(urlsplit(url).scheme.lower() in {"rediss", "amqps"} for url in urls)
    authenticated = all(
        _authenticated(url, broker_use_ssl, broker_password) for url in urls
    )
    tls_verified = all(_tls_verified(url, broker_use_ssl) for url in urls)

    if encrypted and authenticated and tls_verified:
        return RetrievalQueryTransportCapability(
            status=RetrievalQueryTransportStatus.SAFE,
            reason="authenticated_verified_tls",
            transport="celery_broker",
            broker_scheme=broker_scheme,
            encrypted=True,
            authenticated=True,
            tls_verified=True,
            task_protocol=task_protocol,
            task_arguments="redacted",
            task_sent_events_enabled=task_sent_events_enabled,
            worker_task_events_enabled=worker_task_events_enabled,
            result_extended=False,
            local_override_enabled=allow_insecure_local_transport,
        )

    if allow_insecure_local_transport and all(_is_local_only_url(url) for url in urls):
        return RetrievalQueryTransportCapability(
            status=RetrievalQueryTransportStatus.LOCALLY_OVERRIDDEN,
            reason="explicit_insecure_local_transport_override",
            transport="celery_broker",
            broker_scheme=broker_scheme,
            encrypted=encrypted,
            authenticated=authenticated,
            tls_verified=tls_verified,
            task_protocol=task_protocol,
            task_arguments="redacted",
            task_sent_events_enabled=task_sent_events_enabled,
            worker_task_events_enabled=worker_task_events_enabled,
            result_extended=False,
            local_override_enabled=True,
        )

    reason = (
        "broker_encryption_required"
        if not encrypted
        else "broker_authentication_required"
        if not authenticated
        else "broker_certificate_verification_required"
    )
    return _unavailable(
        reason,
        transport="celery_broker",
        broker_scheme=broker_scheme,
        encrypted=encrypted,
        authenticated=authenticated,
        tls_verified=tls_verified,
        task_protocol=task_protocol,
        task_sent_events_enabled=task_sent_events_enabled,
        worker_task_events_enabled=worker_task_events_enabled,
        result_extended=False,
        local_override_enabled=allow_insecure_local_transport,
    )


def require_retrieval_query_transport(
    task: object,
    *,
    allow_insecure_local_transport: bool = False,
) -> RetrievalQueryTransportCapability:
    capability = assess_retrieval_query_transport(
        task,
        allow_insecure_local_transport=allow_insecure_local_transport,
    )
    if not capability.available:
        raise RetrievalQueryTransportPolicyError(capability)
    return capability
