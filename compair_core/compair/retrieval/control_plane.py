"""Authenticated, staging-only baseline control-plane services.

This module deliberately has no corpus-ingestion, index-build, retrieval,
generation, notification, or task-dispatch dependency.  A successful commit
only seals immutable staging rows.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import secrets
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Any
from uuid import UUID, uuid4

import rfc8785
from sqlalchemy import Engine, func, select, text, update
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

from compair_core.baseline_control_plane_schema import (
    compatible_index_job,
    control_job,
    repository_approval,
    repository_registration,
    snapshot_content_part,
    snapshot_continuation_job,
    snapshot_staging,
)

PROTOCOL_VERSION = "baseline-control-plane.v1"
# SHA-256 of protocol/baseline-control-plane.v1.md. Contract tests pin the file
# to this value; deployed clients declare it in every control request.
PROTOCOL_SHA256 = "3b45287a54d04cea751e9cc3209c5f0783192de13062e682eadcae40af322650"

SNAPSHOT_SCHEMA_VERSION = "baseline-snapshot.v1"
REPOSITORY_ADMIN_SCHEMA_VERSION = "baseline-repository-registration-admin.v1"
REPOSITORY_DESCRIPTOR_VERSION = "repository-identity.v1"
CONTINUATION_SCHEMA_VERSION = "baseline-snapshot-continuation.v1"
STAGING_LIFETIME = timedelta(hours=24)
DEFAULT_LEASE_LIFETIME = timedelta(minutes=5)

MAX_SIBLING_REPOSITORIES = 128
MAX_FILE_RECORDS = 50_000
MAX_FILE_BYTES = 200_000
MAX_SUPPORTED_CONTENT_BYTES = 512_000_000
MAX_MANIFEST_REQUEST_BYTES = 32_000_000
MAX_CONTENT_PART_REQUEST_BYTES = 8_000_000
MAX_CONTROL_REQUEST_BYTES = 64_000
MAX_CONTENT_PART_BYTES = 1_000_000
MAX_CONTENT_PART_ITEMS = 1_000
MAX_CONTENT_PARTS = 512

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]*$")
_REPOSITORY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REPOSITORY_AUTHORITY = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SNAPSHOT_ID = re.compile(r"^bsnap_[0-9a-f]{64}$")
_EXCLUDED_COMPONENTS = frozenset({".git", ".compair", "build", "dist", "node_modules"})
_JOB_STATES = frozenset(
    {
        "queued",
        "running",
        "succeeded",
        "retryable_failed",
        "terminal_failed",
        "cancelled",
    }
)


class DuplicateJSONKeyError(ValueError):
    """Raised before canonicalization when a JSON object repeats a key."""


class ControlPlaneError(RuntimeError):
    """A typed error containing only safe, stable metadata."""

    def __init__(
        self,
        code: str,
        *,
        status_code: int = 422,
        stage: str = "snapshot",
        retryable: bool = False,
    ) -> None:
        self.code = code
        self.status_code = status_code
        self.stage = stage
        self.retryable = retryable
        super().__init__(code)

    def to_dict(self, request_id: str | None = None) -> dict[str, object]:
        response = {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "message_type": "error",
            "request_id": request_id,
            "http_status": self.status_code,
            "stage": self.stage,
            "retryable": self.retryable,
            "code": self.code,
        }
        return response


class ControlWriteStage(str, Enum):
    REGISTRATION = "registration"
    JOB = "job"
    STAGING = "staging"
    PART = "part"
    COMMIT = "commit"
    CONTINUATION = "continuation"


class ControlTransportStatus(str, Enum):
    SAFE = "safe"
    UNAVAILABLE = "unavailable"
    LOCAL_OVERRIDE = "local_override"


@dataclass(frozen=True, slots=True)
class ControlTransportCapability:
    status: ControlTransportStatus
    reason: str
    encrypted: bool
    local_override_enabled: bool

    @property
    def available(self) -> bool:
        return self.status in {
            ControlTransportStatus.SAFE,
            ControlTransportStatus.LOCAL_OVERRIDE,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "encrypted": self.encrypted,
            "local_override_enabled": self.local_override_enabled,
        }


@dataclass(frozen=True, slots=True)
class ParsedJSONBody:
    value: dict[str, Any]
    body_sha256: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class ValidatedSnapshot:
    group_id: str
    snapshot_id: str
    manifest_hash: str
    canonical_manifest: bytes
    changed_repository_id: str
    source_document_id: str
    expected_repository_count: int
    expected_file_count: int
    expected_supported_file_count: int
    expected_supported_content_bytes: int
    expected_parts: tuple[tuple[int, ...], ...]
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LeaseReceipt:
    job_id: str
    lease_token: str
    lease_expires_at: datetime
    attempt_count: int


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError("duplicate_json_key")
        result[key] = value
    return result


def _reject_nonfinite(_value: str) -> None:
    raise ValueError("nonfinite_json_number")


def decode_json_object(raw: bytes) -> ParsedJSONBody:
    """Decode strict UTF-8 JSON and reject duplicate keys/non-finite numbers."""

    digest = hashlib.sha256(raw).hexdigest()
    try:
        decoded = raw.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ControlPlaneError("invalid_contract", status_code=400) from exc
    if not isinstance(value, dict):
        raise ControlPlaneError("invalid_contract", status_code=422)
    return ParsedJSONBody(value=value, body_sha256=digest, byte_size=len(raw))


def canonicalize(value: Any) -> bytes:
    """Return RFC 8785 JSON bytes using the pinned implementation."""

    try:
        return rfc8785.dumps(value)
    except (rfc8785.CanonicalizationError, TypeError, ValueError) as exc:
        raise ControlPlaneError("invalid_contract", status_code=422) from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonicalize(value)).hexdigest()


def _is_loopback(value: str | None) -> bool:
    if not value:
        return False
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _trusted_proxy_networks(
    trusted_proxy_allowlist: str,
) -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] | None:
    """Parse an explicit IP/CIDR allowlist; ``None`` means invalid config."""

    if not trusted_proxy_allowlist.strip():
        return ()
    entries = trusted_proxy_allowlist.split(",")
    if any(not entry.strip() for entry in entries):
        return None
    try:
        return tuple(
            ipaddress.ip_network(entry.strip(), strict=False) for entry in entries
        )
    except ValueError:
        return None


def _peer_is_trusted_proxy(
    peer_host: str | None,
    networks: Sequence[ipaddress.IPv4Network | ipaddress.IPv6Network],
) -> bool:
    """Match an immediate peer IP against explicit IP/CIDR entries only."""

    if not peer_host or not networks:
        return False
    try:
        peer = ipaddress.ip_address(peer_host)
    except ValueError:
        return False
    return any(
        peer.version == network.version and peer in network for network in networks
    )


def _trusted_forwarded_proto(
    *,
    forwarded_values: Sequence[str],
    x_forwarded_proto_values: Sequence[str],
) -> str | None:
    """Return one unambiguous trusted-proxy scheme attestation."""

    if len(forwarded_values) > 1 or len(x_forwarded_proto_values) > 1:
        return None
    attestations: list[str] = []
    for header in forwarded_values:
        elements = [element.strip() for element in header.split(",")]
        if len(elements) != 1 or not elements[0]:
            return None
        proto_values = []
        for parameter in elements[0].split(";"):
            key, separator, value = parameter.strip().partition("=")
            if separator and key.strip().lower() == "proto":
                normalized = value.strip().strip('"').lower()
                if normalized not in {"http", "https"}:
                    return None
                proto_values.append(normalized)
        if len(proto_values) != 1:
            return None
        attestations.extend(proto_values)

    for header in x_forwarded_proto_values:
        values = [value.strip().lower() for value in header.split(",")]
        if len(values) != 1 or values[0] not in {"http", "https"}:
            return None
        attestations.extend(values)

    if not attestations or any(value != attestations[0] for value in attestations):
        return None
    return attestations[0]


def assess_control_transport(
    *,
    connection_scheme: str,
    peer_host: str | None,
    allow_insecure_loopback: bool,
    trusted_proxy_allowlist: str = "",
    forwarded_values: Sequence[str] = (),
    x_forwarded_proto_values: Sequence[str] = (),
    proxy_headers_present: bool = False,
) -> ControlTransportCapability:
    """Assess transport from the connection peer, never advertised host/client data.

    A trusted immediate proxy may attest the original scheme. Forwarded client
    addresses are deliberately irrelevant to both HTTPS and the local override.
    """

    scheme = connection_scheme.lower()
    proxy_networks = _trusted_proxy_networks(trusted_proxy_allowlist)
    if scheme == "https":
        return ControlTransportCapability(
            status=ControlTransportStatus.SAFE,
            reason="https",
            encrypted=True,
            local_override_enabled=allow_insecure_loopback,
        )
    if proxy_networks is None:
        return ControlTransportCapability(
            status=ControlTransportStatus.UNAVAILABLE,
            reason="authenticated_https_required",
            encrypted=False,
            local_override_enabled=allow_insecure_loopback,
        )
    peer_is_proxy = _peer_is_trusted_proxy(peer_host, proxy_networks)
    if peer_is_proxy:
        attested_scheme = _trusted_forwarded_proto(
            forwarded_values=forwarded_values,
            x_forwarded_proto_values=x_forwarded_proto_values,
        )
        if attested_scheme == "https":
            return ControlTransportCapability(
                status=ControlTransportStatus.SAFE,
                reason="https",
                encrypted=True,
                local_override_enabled=allow_insecure_loopback,
            )
    elif (
        allow_insecure_loopback
        and scheme == "http"
        and _is_loopback(peer_host)
        and not proxy_headers_present
    ):
        return ControlTransportCapability(
            status=ControlTransportStatus.LOCAL_OVERRIDE,
            reason="explicit_loopback_http_override",
            encrypted=False,
            local_override_enabled=True,
        )
    return ControlTransportCapability(
        status=ControlTransportStatus.UNAVAILABLE,
        reason="authenticated_https_required",
        encrypted=False,
        local_override_enabled=allow_insecure_loopback,
    )


def require_control_transport(capability: ControlTransportCapability) -> None:
    if not capability.available:
        raise ControlPlaneError(
            "transport_unavailable",
            status_code=503,
            stage="transport",
            retryable=True,
        )


def _exact_keys(value: Mapping[str, Any], required: set[str]) -> None:
    if set(value) != required:
        raise ControlPlaneError("invalid_contract")


def _string(
    value: Any,
    *,
    minimum: int = 1,
    maximum: int,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if not isinstance(value, str) or not minimum <= len(value) <= maximum:
        raise ControlPlaneError("invalid_contract")
    if value != value.strip() or (pattern is not None and not pattern.fullmatch(value)):
        raise ControlPlaneError("invalid_contract")
    return value


def _integer(value: Any, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ControlPlaneError("invalid_contract")
    if not minimum <= value <= maximum:
        raise ControlPlaneError("limit_exceeded", status_code=413)
    return value


def _uuid(value: Any) -> str:
    candidate = _string(value, maximum=36)
    try:
        parsed = UUID(candidate)
    except ValueError as exc:
        raise ControlPlaneError("invalid_contract") from exc
    if str(parsed) != candidate.lower():
        raise ControlPlaneError("invalid_contract")
    return candidate.lower()


def _protocol_request(
    payload: Mapping[str, Any],
    *,
    message_type: str,
    required: set[str],
) -> tuple[str, str]:
    _exact_keys(
        payload,
        required
        | {
            "protocol_version",
            "protocol_sha256",
            "message_type",
            "request_id",
            "group_id",
        },
    )
    if (
        payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("protocol_sha256") != PROTOCOL_SHA256
        or payload.get("message_type") != message_type
    ):
        raise ControlPlaneError("protocol_mismatch", status_code=409)
    request_id = _uuid(payload.get("request_id"))
    group_id = _string(payload.get("group_id"), maximum=64, pattern=_IDENTIFIER)
    return request_id, group_id


def _repository_admin_request(
    payload: Mapping[str, Any],
    *,
    message_type: str,
    required: set[str],
) -> tuple[str, str]:
    _exact_keys(
        payload,
        required
        | {
            "schema_version",
            "message_type",
            "request_id",
            "group_id",
        },
    )
    if (
        payload.get("schema_version") != REPOSITORY_ADMIN_SCHEMA_VERSION
        or payload.get("message_type") != message_type
    ):
        raise ControlPlaneError("invalid_contract", status_code=422)
    return (
        _uuid(payload.get("request_id")),
        _string(payload.get("group_id"), maximum=64, pattern=_IDENTIFIER),
    )


def _continuation_status_request(
    payload: Mapping[str, Any],
) -> tuple[str, str, str | None, str | None]:
    _exact_keys(
        payload,
        {
            "schema_version",
            "message_type",
            "request_id",
            "group_id",
            "staging_job_id",
            "continuation_job_id",
        },
    )
    if (
        payload.get("schema_version") != CONTINUATION_SCHEMA_VERSION
        or payload.get("message_type") != "continuation_job_status_request"
    ):
        raise ControlPlaneError("invalid_contract")
    staging_value = payload["staging_job_id"]
    continuation_value = payload["continuation_job_id"]
    if (staging_value is None) == (continuation_value is None):
        raise ControlPlaneError("invalid_contract")
    return (
        _uuid(payload["request_id"]),
        _string(payload["group_id"], maximum=64, pattern=_IDENTIFIER),
        None if staging_value is None else _uuid(staging_value),
        None if continuation_value is None else _uuid(continuation_value),
    )


def _relative_path(value: Any) -> str:
    candidate = _string(value, maximum=4096)
    if (
        not unicodedata.is_normalized("NFC", candidate)
        or "\\" in candidate
        or "\x00" in candidate
        or "//" in candidate
        or candidate.endswith("/")
        or re.match(r"^[A-Za-z]:", candidate)
    ):
        raise ControlPlaneError("invalid_contract")
    path = PurePosixPath(candidate)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ControlPlaneError("invalid_contract")
    if str(path) != candidate:
        raise ControlPlaneError("invalid_contract")
    return candidate


def _git_revision(value: Any) -> str:
    return _string(value, maximum=64, pattern=_GIT_REVISION)


def _sha256(value: Any) -> str:
    return _string(value, minimum=64, maximum=64, pattern=_HEX_64)


def _validate_changed_repository(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ControlPlaneError("invalid_contract")
    required = {
        "repository_id",
        "repository_name",
        "repository_revision",
        "role",
        "base_revision",
        "head_revision",
        "source_document_id",
        "expected_file_count",
    }
    _exact_keys(value, required)
    repository_id = _string(value["repository_id"], maximum=128, pattern=_IDENTIFIER)
    repository_name = _string(
        value["repository_name"], maximum=128, pattern=_REPOSITORY_NAME
    )
    repository_revision = _git_revision(value["repository_revision"])
    base_revision = _git_revision(value["base_revision"])
    head_revision = _git_revision(value["head_revision"])
    if (
        value["role"] != "changed"
        or value["expected_file_count"] != 0
        or repository_revision != head_revision
        or base_revision == head_revision
    ):
        raise ControlPlaneError("invalid_contract")
    return {
        "repository_id": repository_id,
        "repository_name": repository_name,
        "repository_revision": repository_revision,
        "role": "changed",
        "base_revision": base_revision,
        "head_revision": head_revision,
        "source_document_id": _uuid(value["source_document_id"]),
        "expected_file_count": 0,
    }


def _validate_sibling_repository(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ControlPlaneError("invalid_contract")
    required = {
        "repository_id",
        "repository_name",
        "repository_revision",
        "role",
        "expected_file_count",
    }
    _exact_keys(value, required)
    if value["role"] != "sibling":
        raise ControlPlaneError("invalid_contract")
    return {
        "repository_id": _string(
            value["repository_id"], maximum=128, pattern=_IDENTIFIER
        ),
        "repository_name": _string(
            value["repository_name"], maximum=128, pattern=_REPOSITORY_NAME
        ),
        "repository_revision": _git_revision(value["repository_revision"]),
        "role": "sibling",
        "expected_file_count": _integer(
            value["expected_file_count"], minimum=0, maximum=MAX_FILE_RECORDS
        ),
    }


def _validate_file_record(
    value: Any,
    *,
    sibling_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ControlPlaneError("invalid_contract")
    required = {
        "ordinal",
        "repository_id",
        "repository_name",
        "repository_revision",
        "relative_path",
        "git_mode",
        "git_object_id",
        "file_state",
        "skip_reason",
        "byte_size",
        "content_sha256",
        "content_required",
    }
    _exact_keys(value, required)
    repository_id = _string(value["repository_id"], maximum=128, pattern=_IDENTIFIER)
    sibling = sibling_by_id.get(repository_id)
    if sibling is None:
        raise ControlPlaneError("repository_not_authorized", status_code=404)
    repository_name = _string(
        value["repository_name"], maximum=128, pattern=_REPOSITORY_NAME
    )
    revision = _git_revision(value["repository_revision"])
    if (
        repository_name != sibling["repository_name"]
        or revision != sibling["repository_revision"]
    ):
        raise ControlPlaneError("invalid_contract")
    relative_path = _relative_path(value["relative_path"])
    mode = value["git_mode"]
    if mode not in {"100644", "100755", "120000", "160000"}:
        raise ControlPlaneError("invalid_contract")
    state = value["file_state"]
    reason = value["skip_reason"]
    content_required = value["content_required"]
    if not isinstance(content_required, bool):
        raise ControlPlaneError("invalid_contract")
    byte_size = _integer(
        value["byte_size"], minimum=0, maximum=MAX_SUPPORTED_CONTENT_BYTES
    )
    content_hash = value["content_sha256"]
    if content_hash is not None:
        content_hash = _sha256(content_hash)
    components = set(PurePosixPath(relative_path).parts)
    if mode == "120000":
        expected = ("symlink_rejected", "symlink", False)
    elif mode == "160000":
        expected = ("excluded", "unsupported_file_type", False)
    elif components & _EXCLUDED_COMPONENTS:
        expected = ("excluded", "excluded_directory", False)
    elif byte_size > MAX_FILE_BYTES:
        expected = ("oversized", "oversized", False)
    elif state == "unsupported_utf8":
        expected = ("unsupported_utf8", "non_utf8", False)
    elif state == "unreadable":
        expected = ("unreadable", "unreadable", False)
    else:
        expected = ("supported", None, True)
    if (state, reason, content_required) != expected:
        raise ControlPlaneError("invalid_contract")
    if state == "supported" and content_hash is None:
        raise ControlPlaneError("invalid_contract")
    return {
        "ordinal": _integer(value["ordinal"], minimum=1, maximum=MAX_FILE_RECORDS),
        "repository_id": repository_id,
        "repository_name": repository_name,
        "repository_revision": revision,
        "relative_path": relative_path,
        "git_mode": mode,
        "git_object_id": _git_revision(value["git_object_id"]),
        "file_state": state,
        "skip_reason": reason,
        "byte_size": byte_size,
        "content_sha256": content_hash,
        "content_required": content_required,
    }


def validate_snapshot_manifest(
    value: Any, *, expected_group_id: str
) -> ValidatedSnapshot:
    if not isinstance(value, dict):
        raise ControlPlaneError("invalid_contract")
    required = {
        "schema_version",
        "group_id",
        "changed_repository",
        "sibling_repositories",
        "files",
        "repository_count",
        "total_file_count",
        "supported_file_count",
        "supported_content_bytes",
        "canonical_manifest_hash",
        "snapshot_id",
    }
    _exact_keys(value, required)
    if value["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
        raise ControlPlaneError("invalid_contract")
    group_id = _string(value["group_id"], maximum=64, pattern=_IDENTIFIER)
    if group_id != expected_group_id:
        raise ControlPlaneError("not_found_or_forbidden", status_code=404)
    changed = _validate_changed_repository(value["changed_repository"])
    siblings_value = value["sibling_repositories"]
    if not isinstance(siblings_value, list) or not (
        1 <= len(siblings_value) <= MAX_SIBLING_REPOSITORIES
    ):
        raise ControlPlaneError("limit_exceeded", status_code=413)
    siblings = [_validate_sibling_repository(item) for item in siblings_value]
    if siblings != sorted(
        siblings, key=lambda item: (item["repository_name"], item["repository_id"])
    ):
        raise ControlPlaneError("invalid_contract")
    sibling_ids = [item["repository_id"] for item in siblings]
    sibling_names = [item["repository_name"] for item in siblings]
    if (
        len(sibling_ids) != len(set(sibling_ids))
        or len(sibling_names) != len(set(sibling_names))
        or changed["repository_id"] in sibling_ids
    ):
        raise ControlPlaneError("invalid_contract")
    sibling_by_id = {item["repository_id"]: item for item in siblings}
    files_value = value["files"]
    if not isinstance(files_value, list) or len(files_value) > MAX_FILE_RECORDS:
        raise ControlPlaneError("limit_exceeded", status_code=413)
    files = [
        _validate_file_record(item, sibling_by_id=sibling_by_id) for item in files_value
    ]
    expected_order = sorted(
        files,
        key=lambda item: (
            item["repository_name"],
            item["relative_path"],
            item["repository_id"],
        ),
    )
    if files != expected_order or [item["ordinal"] for item in files] != list(
        range(1, len(files) + 1)
    ):
        raise ControlPlaneError("invalid_contract")
    pairs = [(item["repository_id"], item["relative_path"]) for item in files]
    if len(pairs) != len(set(pairs)):
        raise ControlPlaneError("invalid_contract")
    per_repository = {
        repository_id: sum(
            1 for item in files if item["repository_id"] == repository_id
        )
        for repository_id in sibling_ids
    }
    if any(
        per_repository[item["repository_id"]] != item["expected_file_count"]
        for item in siblings
    ):
        raise ControlPlaneError("invalid_contract")
    supported = [item for item in files if item["content_required"]]
    supported_bytes = sum(item["byte_size"] for item in supported)
    if (
        value["repository_count"] != len(siblings)
        or value["total_file_count"] != len(files)
        or value["supported_file_count"] != len(supported)
        or value["supported_content_bytes"] != supported_bytes
        or supported_bytes > MAX_SUPPORTED_CONTENT_BYTES
    ):
        raise ControlPlaneError("invalid_contract")
    canonical_value = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "changed_repository": changed,
        "sibling_repositories": siblings,
        "files": files,
    }
    canonical_manifest = canonicalize(canonical_value)
    manifest_hash = hashlib.sha256(canonical_manifest).hexdigest()
    snapshot_id = _string(value["snapshot_id"], maximum=72, pattern=_SNAPSHOT_ID)
    if (
        _sha256(value["canonical_manifest_hash"]) != manifest_hash
        or snapshot_id != f"bsnap_{manifest_hash}"
    ):
        raise ControlPlaneError("manifest_hash_mismatch", status_code=409)
    parts: list[list[int]] = []
    for item in supported:
        if (
            not parts
            or len(parts[-1]) >= MAX_CONTENT_PART_ITEMS
            or sum(files[ordinal - 1]["byte_size"] for ordinal in parts[-1])
            + item["byte_size"]
            > MAX_CONTENT_PART_BYTES
        ):
            parts.append([])
        parts[-1].append(item["ordinal"])
    if len(parts) > MAX_CONTENT_PARTS:
        raise ControlPlaneError("limit_exceeded", status_code=413)
    normalized_manifest = dict(value)
    normalized_manifest["changed_repository"] = changed
    normalized_manifest["sibling_repositories"] = siblings
    normalized_manifest["files"] = files
    return ValidatedSnapshot(
        group_id=group_id,
        snapshot_id=snapshot_id,
        manifest_hash=manifest_hash,
        canonical_manifest=canonical_manifest,
        changed_repository_id=changed["repository_id"],
        source_document_id=changed["source_document_id"],
        expected_repository_count=len(siblings),
        expected_file_count=len(files),
        expected_supported_file_count=len(supported),
        expected_supported_content_bytes=supported_bytes,
        expected_parts=tuple(tuple(part) for part in parts),
        manifest=normalized_manifest,
    )


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class BaselineControlPlaneService:
    """Transactional staging service over the migration-owned tables."""

    def __init__(
        self,
        engine: Engine,
        *,
        clock: Callable[[], datetime] = _utcnow,
        stage_hook: Callable[[ControlWriteStage], None] | None = None,
    ) -> None:
        self.engine = engine
        self.clock = clock
        self.stage_hook = stage_hook

    def _stage(self, stage: ControlWriteStage) -> None:
        if self.stage_hook is not None:
            self.stage_hook(stage)

    def authorize_group(self, *, caller_user_id: str, group_id: str) -> None:
        with self.engine.connect() as connection:
            self._authorize_group(connection, user_id=caller_user_id, group_id=group_id)

    @staticmethod
    def _authorize_group(
        connection: Connection, *, user_id: str, group_id: str
    ) -> None:
        authorized = connection.execute(
            text(
                "SELECT 1 FROM user_to_group "
                "WHERE user_id = :user_id AND group_id = :group_id"
            ),
            {"user_id": user_id, "group_id": group_id},
        ).first()
        if authorized is None:
            raise ControlPlaneError("not_found_or_forbidden", status_code=404)

    @classmethod
    def _authorize_group_admin(
        cls,
        connection: Connection,
        *,
        user_id: str,
        group_id: str,
    ) -> None:
        cls._authorize_group(connection, user_id=user_id, group_id=group_id)
        authorized = connection.execute(
            text(
                "SELECT 1 FROM administrator AS administrator_record "
                "JOIN admin_to_group AS administrator_scope "
                "ON administrator_scope.admin_id = administrator_record.admin_id "
                "WHERE administrator_record.user_id = :user_id "
                "AND administrator_scope.group_id = :group_id"
            ),
            {"user_id": user_id, "group_id": group_id},
        ).first()
        if authorized is None:
            raise ControlPlaneError("not_found_or_forbidden", status_code=404)

    @staticmethod
    def _repository_registration_response(
        *,
        request_id: str,
        group_id: str,
        registration_id: str,
        descriptor_hash: str,
        state: str,
        replayed: bool,
    ) -> dict[str, object]:
        return {
            "schema_version": REPOSITORY_ADMIN_SCHEMA_VERSION,
            "message_type": "repository_registration",
            "request_id": request_id,
            "group_id": group_id,
            "registration_id": registration_id,
            "identity_descriptor_hash": descriptor_hash,
            "state": state,
            "replayed": replayed,
        }

    def register_repository(
        self,
        payload: Mapping[str, Any],
        *,
        caller_user_id: str,
    ) -> dict[str, object]:
        request_id, group_id = _repository_admin_request(
            payload,
            message_type="repository_registration_create",
            required={"identity_descriptor", "source_document_id"},
        )
        descriptor = payload["identity_descriptor"]
        if not isinstance(descriptor, Mapping):
            raise ControlPlaneError("invalid_contract")
        _exact_keys(descriptor, {"version", "authority", "repository_uid"})
        if descriptor.get("version") != REPOSITORY_DESCRIPTOR_VERSION:
            raise ControlPlaneError("invalid_contract")
        authority = _string(
            descriptor.get("authority"),
            maximum=253,
            pattern=_REPOSITORY_AUTHORITY,
        )
        repository_uid = _string(
            descriptor.get("repository_uid"),
            maximum=256,
            pattern=_IDENTIFIER,
        )
        source_document_value = payload["source_document_id"]
        source_document_id = (
            None if source_document_value is None else _uuid(source_document_value)
        )
        normalized_descriptor = {
            "version": REPOSITORY_DESCRIPTOR_VERSION,
            "authority": authority,
            "repository_uid": repository_uid,
        }
        descriptor_hash = canonical_sha256(normalized_descriptor)
        try:
            with self.engine.begin() as connection:
                self._authorize_group_admin(
                    connection,
                    user_id=caller_user_id,
                    group_id=group_id,
                )
                if source_document_id is not None:
                    source_scope = connection.execute(
                        text(
                            "SELECT 1 FROM document_to_group "
                            "WHERE document_id = :document_id AND group_id = :group_id"
                        ),
                        {"document_id": source_document_id, "group_id": group_id},
                    ).first()
                    if source_scope is None:
                        raise ControlPlaneError(
                            "source_not_authorized", status_code=404
                        )
                existing = (
                    connection.execute(
                        select(
                            repository_approval,
                            repository_registration.c.source_document_id,
                        )
                        .select_from(
                            repository_approval.join(
                                repository_registration,
                                (
                                    repository_registration.c.registration_id
                                    == repository_approval.c.registration_id
                                )
                                & (
                                    repository_registration.c.group_id
                                    == repository_approval.c.group_id
                                ),
                            )
                        )
                        .where(
                            repository_approval.c.group_id == group_id,
                            repository_approval.c.repository_authority == authority,
                            repository_approval.c.repository_uid == repository_uid,
                        )
                    )
                    .mappings()
                    .first()
                )
                if existing is not None:
                    if existing["source_document_id"] != source_document_id:
                        raise ControlPlaneError(
                            "repository_registration_conflict", status_code=409
                        )
                    return self._repository_registration_response(
                        request_id=request_id,
                        group_id=group_id,
                        registration_id=str(existing["registration_id"]),
                        descriptor_hash=str(existing["descriptor_hash"]),
                        state=str(existing["state"]),
                        replayed=True,
                    )
                now = self.clock()
                registration_id = str(uuid4())
                connection.execute(
                    repository_registration.insert().values(
                        registration_id=registration_id,
                        group_id=group_id,
                        repository_id=registration_id,
                        repository_name=registration_id,
                        source_document_id=source_document_id,
                        created_by_user_id=caller_user_id,
                        enabled=True,
                        created_at=now,
                        updated_at=now,
                    )
                )
                connection.execute(
                    repository_approval.insert().values(
                        registration_id=registration_id,
                        group_id=group_id,
                        descriptor_version=REPOSITORY_DESCRIPTOR_VERSION,
                        repository_authority=authority,
                        repository_uid=repository_uid,
                        descriptor_hash=descriptor_hash,
                        state="active",
                        approved_by_user_id=caller_user_id,
                        disabled_by_user_id=None,
                        created_at=now,
                        updated_at=now,
                        disabled_at=None,
                    )
                )
                self._stage(ControlWriteStage.REGISTRATION)
        except IntegrityError:
            with self.engine.connect() as connection:
                self._authorize_group_admin(
                    connection,
                    user_id=caller_user_id,
                    group_id=group_id,
                )
                existing = (
                    connection.execute(
                        select(
                            repository_approval,
                            repository_registration.c.source_document_id,
                        )
                        .select_from(
                            repository_approval.join(
                                repository_registration,
                                (
                                    repository_registration.c.registration_id
                                    == repository_approval.c.registration_id
                                )
                                & (
                                    repository_registration.c.group_id
                                    == repository_approval.c.group_id
                                ),
                            )
                        )
                        .where(
                            repository_approval.c.group_id == group_id,
                            repository_approval.c.repository_authority == authority,
                            repository_approval.c.repository_uid == repository_uid,
                        )
                    )
                    .mappings()
                    .first()
                )
                if existing is None:
                    raise ControlPlaneError(
                        "concurrent_conflict", status_code=409, retryable=True
                    )
                if existing["source_document_id"] != source_document_id:
                    raise ControlPlaneError(
                        "repository_registration_conflict", status_code=409
                    )
                return self._repository_registration_response(
                    request_id=request_id,
                    group_id=group_id,
                    registration_id=str(existing["registration_id"]),
                    descriptor_hash=str(existing["descriptor_hash"]),
                    state=str(existing["state"]),
                    replayed=True,
                )
        return self._repository_registration_response(
            request_id=request_id,
            group_id=group_id,
            registration_id=registration_id,
            descriptor_hash=descriptor_hash,
            state="active",
            replayed=False,
        )

    def set_repository_registration_state(
        self,
        payload: Mapping[str, Any],
        *,
        caller_user_id: str,
    ) -> dict[str, object]:
        request_id, group_id = _repository_admin_request(
            payload,
            message_type="repository_registration_state",
            required={"registration_id", "active"},
        )
        registration_id = _uuid(payload["registration_id"])
        active = payload["active"]
        if not isinstance(active, bool):
            raise ControlPlaneError("invalid_contract")
        with self.engine.begin() as connection:
            self._authorize_group_admin(
                connection,
                user_id=caller_user_id,
                group_id=group_id,
            )
            approval = (
                connection.execute(
                    select(repository_approval).where(
                        repository_approval.c.registration_id == registration_id,
                        repository_approval.c.group_id == group_id,
                    )
                )
                .mappings()
                .first()
            )
            if approval is None:
                raise ControlPlaneError("not_found_or_forbidden", status_code=404)
            state = "active" if active else "disabled"
            replayed = approval["state"] == state
            if not replayed:
                now = self.clock()
                connection.execute(
                    update(repository_approval)
                    .where(
                        repository_approval.c.registration_id == registration_id,
                        repository_approval.c.group_id == group_id,
                    )
                    .values(
                        state=state,
                        disabled_by_user_id=None if active else caller_user_id,
                        disabled_at=None if active else now,
                        updated_at=now,
                    )
                )
                connection.execute(
                    update(repository_registration)
                    .where(
                        repository_registration.c.registration_id == registration_id,
                        repository_registration.c.group_id == group_id,
                    )
                    .values(enabled=active, updated_at=now)
                )
                self._stage(ControlWriteStage.REGISTRATION)
            return self._repository_registration_response(
                request_id=request_id,
                group_id=group_id,
                registration_id=registration_id,
                descriptor_hash=str(approval["descriptor_hash"]),
                state=state,
                replayed=replayed,
            )

    @staticmethod
    def _authorize_snapshot_repositories(
        connection: Connection,
        *,
        group_id: str,
        snapshot: ValidatedSnapshot,
    ) -> None:
        repositories = [snapshot.manifest["changed_repository"]] + list(
            snapshot.manifest["sibling_repositories"]
        )
        repository_ids = [item["repository_id"] for item in repositories]
        rows = connection.execute(
            select(
                repository_registration.c.registration_id,
                repository_registration.c.repository_id,
                repository_registration.c.repository_name,
                repository_registration.c.source_document_id,
                repository_approval.c.descriptor_version,
                repository_approval.c.repository_authority,
                repository_approval.c.repository_uid,
                repository_approval.c.descriptor_hash,
            )
            .select_from(
                repository_registration.join(
                    repository_approval,
                    (
                        repository_approval.c.registration_id
                        == repository_registration.c.registration_id
                    )
                    & (
                        repository_approval.c.group_id
                        == repository_registration.c.group_id
                    ),
                )
            )
            .where(
                repository_registration.c.group_id == group_id,
                repository_registration.c.registration_id.in_(repository_ids),
                repository_registration.c.enabled.is_(True),
                repository_approval.c.state == "active",
            )
        ).mappings()
        by_id = {str(row["registration_id"]): row for row in rows}
        if set(by_id) != set(repository_ids):
            raise ControlPlaneError("repository_not_authorized", status_code=404)
        for registration_id, row in by_id.items():
            descriptor = {
                "version": str(row["descriptor_version"]),
                "authority": str(row["repository_authority"]),
                "repository_uid": str(row["repository_uid"]),
            }
            if (
                row["descriptor_version"] != REPOSITORY_DESCRIPTOR_VERSION
                or row["repository_id"] != registration_id
                or row["repository_name"] != registration_id
                or canonical_sha256(descriptor) != row["descriptor_hash"]
            ):
                raise ControlPlaneError("repository_not_authorized", status_code=404)
        changed_row = by_id[snapshot.changed_repository_id]
        if changed_row["source_document_id"] != snapshot.source_document_id:
            raise ControlPlaneError("source_not_authorized", status_code=404)
        document_scope = connection.execute(
            text(
                "SELECT 1 FROM document_to_group "
                "WHERE document_id = :document_id AND group_id = :group_id"
            ),
            {
                "document_id": snapshot.source_document_id,
                "group_id": group_id,
            },
        ).first()
        if document_scope is None:
            raise ControlPlaneError("source_not_authorized", status_code=404)

    @staticmethod
    def _snapshot_from_staging(row: Mapping[str, Any]) -> ValidatedSnapshot:
        try:
            manifest = json.loads(str(row["canonical_manifest_json"]))
        except json.JSONDecodeError as exc:  # pragma: no cover - DB corruption
            raise ControlPlaneError("staging_incompatible", status_code=409) from exc
        envelope = {
            "schema_version": manifest["schema_version"],
            "group_id": row["group_id"],
            "changed_repository": manifest["changed_repository"],
            "sibling_repositories": manifest["sibling_repositories"],
            "files": manifest["files"],
            "repository_count": row["expected_repository_count"],
            "total_file_count": row["expected_file_count"],
            "supported_file_count": row["expected_supported_file_count"],
            "supported_content_bytes": row["expected_supported_content_bytes"],
            "canonical_manifest_hash": row["canonical_manifest_hash"],
            "snapshot_id": row["snapshot_id"],
        }
        return validate_snapshot_manifest(
            envelope, expected_group_id=str(row["group_id"])
        )

    def begin_snapshot(
        self,
        payload: Mapping[str, Any],
        *,
        caller_user_id: str,
    ) -> dict[str, object]:
        request_id, group_id = _protocol_request(
            payload,
            message_type="snapshot_begin",
            required={"idempotency_key", "snapshot"},
        )
        idempotency_key = _string(
            payload["idempotency_key"],
            minimum=32,
            maximum=128,
            pattern=_IDENTIFIER,
        )
        snapshot = validate_snapshot_manifest(
            payload["snapshot"], expected_group_id=group_id
        )
        intent = {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "message_type": "snapshot_begin",
            "group_id": group_id,
            "snapshot": payload["snapshot"],
        }
        intent_hash = canonical_sha256(intent)
        try:
            return self._begin_transaction(
                request_id=request_id,
                group_id=group_id,
                idempotency_key=idempotency_key,
                intent_hash=intent_hash,
                snapshot=snapshot,
                caller_user_id=caller_user_id,
            )
        except IntegrityError:
            return self._load_begin_replay(
                request_id=request_id,
                group_id=group_id,
                idempotency_key=idempotency_key,
                intent_hash=intent_hash,
                snapshot=snapshot,
                caller_user_id=caller_user_id,
            )

    def _begin_transaction(
        self,
        *,
        request_id: str,
        group_id: str,
        idempotency_key: str,
        intent_hash: str,
        snapshot: ValidatedSnapshot,
        caller_user_id: str,
    ) -> dict[str, object]:
        with self.engine.begin() as connection:
            self._authorize_group(connection, user_id=caller_user_id, group_id=group_id)
            self._authorize_snapshot_repositories(
                connection, group_id=group_id, snapshot=snapshot
            )
            existing = (
                connection.execute(
                    select(control_job).where(
                        control_job.c.group_id == group_id,
                        control_job.c.operation == "snapshot_ingest",
                        control_job.c.idempotency_key == idempotency_key,
                    )
                )
                .mappings()
                .first()
            )
            if existing is not None:
                return self._replay_begin(
                    connection,
                    request_id=request_id,
                    existing=existing,
                    intent_hash=intent_hash,
                    snapshot=snapshot,
                    caller_user_id=caller_user_id,
                )
            if (
                connection.execute(
                    select(snapshot_staging.c.staging_id).where(
                        snapshot_staging.c.group_id == group_id,
                        snapshot_staging.c.snapshot_id == snapshot.snapshot_id,
                    )
                ).first()
                is not None
            ):
                raise ControlPlaneError("idempotency_conflict", status_code=409)
            now = self.clock()
            job_id = str(uuid4())
            staging_id = str(uuid4())
            connection.execute(
                control_job.insert().values(
                    job_id=job_id,
                    group_id=group_id,
                    request_id=request_id,
                    operation="snapshot_ingest",
                    idempotency_key=idempotency_key,
                    intent_hash=intent_hash,
                    protocol_version=PROTOCOL_VERSION,
                    protocol_sha256=PROTOCOL_SHA256,
                    state="queued",
                    attempt_count=0,
                    lease_token=None,
                    lease_expires_at=None,
                    progress_completed=0,
                    progress_total=len(snapshot.expected_parts),
                    result_snapshot_id=None,
                    error_code=None,
                    error_fingerprint=None,
                    created_at=now,
                    updated_at=now,
                    finished_at=None,
                )
            )
            self._stage(ControlWriteStage.JOB)
            connection.execute(
                snapshot_staging.insert().values(
                    staging_id=staging_id,
                    group_id=group_id,
                    job_id=job_id,
                    snapshot_id=snapshot.snapshot_id,
                    status="open",
                    manifest_schema_version=SNAPSHOT_SCHEMA_VERSION,
                    canonical_manifest_hash=snapshot.manifest_hash,
                    canonical_manifest_json=snapshot.canonical_manifest.decode("utf-8"),
                    changed_repository_id=snapshot.changed_repository_id,
                    source_document_id=snapshot.source_document_id,
                    expected_repository_count=snapshot.expected_repository_count,
                    expected_file_count=snapshot.expected_file_count,
                    expected_supported_file_count=snapshot.expected_supported_file_count,
                    expected_supported_content_bytes=(
                        snapshot.expected_supported_content_bytes
                    ),
                    expected_part_count=len(snapshot.expected_parts),
                    received_part_count=0,
                    received_file_count=0,
                    received_content_bytes=0,
                    content_manifest_hash=None,
                    expires_at=now + STAGING_LIFETIME,
                    created_at=now,
                    updated_at=now,
                    sealed_at=None,
                )
            )
            self._stage(ControlWriteStage.STAGING)
        return self._job_accepted(
            request_id=request_id,
            group_id=group_id,
            job_id=job_id,
            replayed=False,
        )

    def _load_begin_replay(
        self,
        *,
        request_id: str,
        group_id: str,
        idempotency_key: str,
        intent_hash: str,
        snapshot: ValidatedSnapshot,
        caller_user_id: str,
    ) -> dict[str, object]:
        with self.engine.connect() as connection:
            self._authorize_group(connection, user_id=caller_user_id, group_id=group_id)
            self._authorize_snapshot_repositories(
                connection, group_id=group_id, snapshot=snapshot
            )
            existing = (
                connection.execute(
                    select(control_job).where(
                        control_job.c.group_id == group_id,
                        control_job.c.operation == "snapshot_ingest",
                        control_job.c.idempotency_key == idempotency_key,
                    )
                )
                .mappings()
                .first()
            )
            if existing is None:
                raise ControlPlaneError(
                    "concurrent_conflict", status_code=409, retryable=True
                )
            return self._replay_begin(
                connection,
                request_id=request_id,
                existing=existing,
                intent_hash=intent_hash,
                snapshot=snapshot,
                caller_user_id=caller_user_id,
            )

    def _replay_begin(
        self,
        connection: Connection,
        *,
        request_id: str,
        existing: Mapping[str, Any],
        intent_hash: str,
        snapshot: ValidatedSnapshot,
        caller_user_id: str,
    ) -> dict[str, object]:
        if existing["intent_hash"] != intent_hash:
            raise ControlPlaneError("idempotency_conflict", status_code=409)
        staging = (
            connection.execute(
                select(snapshot_staging).where(
                    snapshot_staging.c.job_id == existing["job_id"],
                    snapshot_staging.c.group_id == existing["group_id"],
                )
            )
            .mappings()
            .one()
        )
        persisted = self._snapshot_from_staging(staging)
        if persisted.manifest_hash != snapshot.manifest_hash:
            raise ControlPlaneError("idempotency_conflict", status_code=409)
        self._authorize_group(
            connection,
            user_id=caller_user_id,
            group_id=str(existing["group_id"]),
        )
        self._authorize_snapshot_repositories(
            connection,
            group_id=str(existing["group_id"]),
            snapshot=persisted,
        )
        return self._job_accepted(
            request_id=request_id,
            group_id=str(existing["group_id"]),
            job_id=str(existing["job_id"]),
            replayed=True,
        )

    @staticmethod
    def _job_accepted(
        *, request_id: str, group_id: str, job_id: str, replayed: bool
    ) -> dict[str, object]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "message_type": "job_accepted",
            "request_id": request_id,
            "group_id": group_id,
            "job_id": job_id,
            "operation": "snapshot_ingest",
            "state": "queued",
            "replayed": replayed,
        }

    @staticmethod
    def _content_file_map(snapshot: ValidatedSnapshot) -> dict[int, Mapping[str, Any]]:
        return {
            int(item["ordinal"]): item
            for item in snapshot.manifest["files"]
            if item["content_required"]
        }

    def stage_content_part(
        self,
        payload: Mapping[str, Any],
        *,
        caller_user_id: str,
        request_body_sha256: str,
        path_job_id: str,
    ) -> dict[str, object]:
        request_id, group_id = _protocol_request(
            payload,
            message_type="snapshot_content_part",
            required={
                "job_id",
                "snapshot_id",
                "part_ordinal",
                "part_sha256",
                "content_items",
            },
        )
        job_id = _uuid(payload["job_id"])
        if job_id != path_job_id:
            raise ControlPlaneError("not_found_or_forbidden", status_code=404)
        try:
            return self._stage_content_part_transaction(
                payload=payload,
                request_id=request_id,
                group_id=group_id,
                job_id=job_id,
                caller_user_id=caller_user_id,
                request_body_sha256=request_body_sha256,
            )
        except IntegrityError:
            return self._load_part_replay(
                payload=payload,
                request_id=request_id,
                group_id=group_id,
                job_id=job_id,
                caller_user_id=caller_user_id,
                request_body_sha256=request_body_sha256,
            )

    def _load_open_staging(
        self,
        connection: Connection,
        *,
        group_id: str,
        job_id: str,
    ) -> Mapping[str, Any]:
        row = (
            connection.execute(
                select(snapshot_staging).where(
                    snapshot_staging.c.group_id == group_id,
                    snapshot_staging.c.job_id == job_id,
                )
            )
            .mappings()
            .first()
        )
        if row is None:
            raise ControlPlaneError("not_found_or_forbidden", status_code=404)
        return row

    def _validated_part(
        self,
        payload: Mapping[str, Any],
        *,
        snapshot: ValidatedSnapshot,
    ) -> tuple[int, str, bytes, int, int]:
        if payload["snapshot_id"] != snapshot.snapshot_id:
            raise ControlPlaneError("stale_snapshot", status_code=409)
        ordinal = _integer(
            payload["part_ordinal"], minimum=1, maximum=MAX_CONTENT_PARTS
        )
        if ordinal > len(snapshot.expected_parts):
            raise ControlPlaneError("invalid_contract")
        items = payload["content_items"]
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_CONTENT_PART_ITEMS:
            raise ControlPlaneError("limit_exceeded", status_code=413)
        expected_ordinals = snapshot.expected_parts[ordinal - 1]
        file_map = self._content_file_map(snapshot)
        normalized: list[dict[str, Any]] = []
        content_bytes = 0
        for expected_ordinal, item in zip(expected_ordinals, items, strict=False):
            if not isinstance(item, dict):
                raise ControlPlaneError("invalid_contract")
            _exact_keys(
                item,
                {"file_ordinal", "byte_size", "content_sha256", "content_utf8"},
            )
            file_ordinal = _integer(
                item["file_ordinal"], minimum=1, maximum=MAX_FILE_RECORDS
            )
            if file_ordinal != expected_ordinal:
                raise ControlPlaneError("invalid_contract")
            file_record = file_map[file_ordinal]
            content = item["content_utf8"]
            if not isinstance(content, str):
                raise ControlPlaneError("invalid_contract")
            encoded = content.encode("utf-8")
            byte_size = _integer(item["byte_size"], minimum=0, maximum=MAX_FILE_BYTES)
            content_hash = _sha256(item["content_sha256"])
            if (
                len(encoded) != byte_size
                or byte_size != file_record["byte_size"]
                or hashlib.sha256(encoded).hexdigest() != content_hash
                or content_hash != file_record["content_sha256"]
            ):
                raise ControlPlaneError("content_hash_mismatch", status_code=409)
            content_bytes += byte_size
            normalized.append(
                {
                    "file_ordinal": file_ordinal,
                    "byte_size": byte_size,
                    "content_sha256": content_hash,
                    "content_utf8": content,
                }
            )
        if (
            len(items) != len(expected_ordinals)
            or content_bytes > MAX_CONTENT_PART_BYTES
        ):
            raise ControlPlaneError("limit_exceeded", status_code=413)
        canonical_items = canonicalize(normalized)
        part_hash = hashlib.sha256(canonical_items).hexdigest()
        if _sha256(payload["part_sha256"]) != part_hash:
            raise ControlPlaneError("content_hash_mismatch", status_code=409)
        return ordinal, part_hash, canonical_items, len(normalized), content_bytes

    def _stage_content_part_transaction(
        self,
        *,
        payload: Mapping[str, Any],
        request_id: str,
        group_id: str,
        job_id: str,
        caller_user_id: str,
        request_body_sha256: str,
    ) -> dict[str, object]:
        expired = False
        with self.engine.begin() as connection:
            self._authorize_group(connection, user_id=caller_user_id, group_id=group_id)
            staging = self._load_open_staging(
                connection, group_id=group_id, job_id=job_id
            )
            snapshot = self._snapshot_from_staging(staging)
            self._authorize_snapshot_repositories(
                connection, group_id=group_id, snapshot=snapshot
            )
            now = self.clock()
            if staging["status"] == "open" and _aware(staging["expires_at"]) <= now:
                self._expire_one(connection, staging=staging, now=now)
                expired = True
            elif staging["status"] != "open":
                raise ControlPlaneError("staging_not_open", status_code=409)
            else:
                ordinal, part_hash, canonical_items, item_count, content_bytes = (
                    self._validated_part(payload, snapshot=snapshot)
                )
                existing = (
                    connection.execute(
                        select(snapshot_content_part).where(
                            snapshot_content_part.c.staging_id == staging["staging_id"],
                            snapshot_content_part.c.part_ordinal == ordinal,
                        )
                    )
                    .mappings()
                    .first()
                )
                if existing is not None:
                    if (
                        existing["part_sha256"] != part_hash
                        or existing["request_body_sha256"] != request_body_sha256
                        or existing["canonical_content_items_json"]
                        != canonical_items.decode("utf-8")
                    ):
                        raise ControlPlaneError("part_conflict", status_code=409)
                    return self._status_from_rows(
                        request_id=request_id,
                        job=connection.execute(
                            select(control_job).where(control_job.c.job_id == job_id)
                        )
                        .mappings()
                        .one(),
                        staging=staging,
                        replayed=True,
                    )
                connection.execute(
                    snapshot_content_part.insert().values(
                        part_id=str(uuid4()),
                        staging_id=staging["staging_id"],
                        group_id=group_id,
                        part_ordinal=ordinal,
                        part_sha256=part_hash,
                        request_body_sha256=request_body_sha256,
                        item_count=item_count,
                        content_bytes=content_bytes,
                        canonical_content_items_json=canonical_items.decode("utf-8"),
                        created_at=now,
                    )
                )
                self._stage(ControlWriteStage.PART)
                totals = connection.execute(
                    select(
                        func.count(snapshot_content_part.c.part_id),
                        func.coalesce(func.sum(snapshot_content_part.c.item_count), 0),
                        func.coalesce(
                            func.sum(snapshot_content_part.c.content_bytes), 0
                        ),
                    ).where(snapshot_content_part.c.staging_id == staging["staging_id"])
                ).one()
                received_parts, received_files, received_bytes = map(int, totals)
                connection.execute(
                    update(snapshot_staging)
                    .where(snapshot_staging.c.staging_id == staging["staging_id"])
                    .values(
                        received_part_count=received_parts,
                        received_file_count=received_files,
                        received_content_bytes=received_bytes,
                        expires_at=now + STAGING_LIFETIME,
                        updated_at=now,
                    )
                )
                connection.execute(
                    update(control_job)
                    .where(control_job.c.job_id == job_id)
                    .values(
                        progress_completed=received_parts,
                        updated_at=now,
                    )
                )
                staging = dict(staging)
                staging.update(
                    {
                        "received_part_count": received_parts,
                        "received_file_count": received_files,
                        "received_content_bytes": received_bytes,
                        "expires_at": now + STAGING_LIFETIME,
                        "updated_at": now,
                    }
                )
                job = (
                    connection.execute(
                        select(control_job).where(control_job.c.job_id == job_id)
                    )
                    .mappings()
                    .one()
                )
                result = self._status_from_rows(
                    request_id=request_id,
                    job=job,
                    staging=staging,
                    replayed=False,
                )
        if expired:
            raise ControlPlaneError("staging_expired", status_code=409)
        return result

    def _load_part_replay(
        self,
        *,
        payload: Mapping[str, Any],
        request_id: str,
        group_id: str,
        job_id: str,
        caller_user_id: str,
        request_body_sha256: str,
    ) -> dict[str, object]:
        with self.engine.connect() as connection:
            self._authorize_group(connection, user_id=caller_user_id, group_id=group_id)
            staging = self._load_open_staging(
                connection, group_id=group_id, job_id=job_id
            )
            snapshot = self._snapshot_from_staging(staging)
            self._authorize_snapshot_repositories(
                connection, group_id=group_id, snapshot=snapshot
            )
            ordinal, part_hash, canonical_items, _count, _bytes = self._validated_part(
                payload, snapshot=snapshot
            )
            existing = (
                connection.execute(
                    select(snapshot_content_part).where(
                        snapshot_content_part.c.staging_id == staging["staging_id"],
                        snapshot_content_part.c.part_ordinal == ordinal,
                    )
                )
                .mappings()
                .first()
            )
            if existing is None:
                raise ControlPlaneError(
                    "concurrent_conflict", status_code=409, retryable=True
                )
            if (
                existing["part_sha256"] != part_hash
                or existing["request_body_sha256"] != request_body_sha256
                or existing["canonical_content_items_json"]
                != canonical_items.decode("utf-8")
            ):
                raise ControlPlaneError("part_conflict", status_code=409)
            job = (
                connection.execute(
                    select(control_job).where(control_job.c.job_id == job_id)
                )
                .mappings()
                .one()
            )
            return self._status_from_rows(
                request_id=request_id,
                job=job,
                staging=staging,
                replayed=True,
            )

    @staticmethod
    def _repository_set_hash(snapshot: ValidatedSnapshot) -> str:
        repository_ids = sorted(
            [snapshot.manifest["changed_repository"]["repository_id"]]
            + [
                repository["repository_id"]
                for repository in snapshot.manifest["sibling_repositories"]
            ]
        )
        return canonical_sha256(repository_ids)

    def _ensure_continuation_job(
        self,
        connection: Connection,
        *,
        request_id: str,
        caller_user_id: str,
        staging_job: Mapping[str, Any],
        staging: Mapping[str, Any],
        snapshot: ValidatedSnapshot,
        content_manifest_hash: str,
        now: datetime,
    ) -> Mapping[str, Any]:
        group_id = str(staging["group_id"])
        repository_set_hash = self._repository_set_hash(snapshot)
        intent = {
            "contract_version": CONTINUATION_SCHEMA_VERSION,
            "group_id": group_id,
            "staging_id": str(staging["staging_id"]),
            "snapshot_id": snapshot.snapshot_id,
            "canonical_manifest_hash": snapshot.manifest_hash,
            "content_manifest_hash": content_manifest_hash,
            "repository_set_hash": repository_set_hash,
            "expected_repository_count": snapshot.expected_repository_count,
            "expected_file_count": snapshot.expected_file_count,
            "expected_supported_file_count": snapshot.expected_supported_file_count,
            "expected_supported_content_bytes": (
                snapshot.expected_supported_content_bytes
            ),
            "expected_part_count": len(snapshot.expected_parts),
        }
        sealed_intent_hash = canonical_sha256(intent)
        idempotency_key = str(staging_job["idempotency_key"])
        existing = (
            connection.execute(
                select(snapshot_continuation_job).where(
                    snapshot_continuation_job.c.group_id == group_id,
                    snapshot_continuation_job.c.idempotency_key == idempotency_key,
                )
            )
            .mappings()
            .first()
        )
        if existing is not None:
            if (
                existing["sealed_intent_hash"] != sealed_intent_hash
                or existing["staging_id"] != staging["staging_id"]
            ):
                raise ControlPlaneError("continuation_conflict", status_code=409)
            return existing
        continuation_job_id = str(uuid4())
        connection.execute(
            snapshot_continuation_job.insert().values(
                continuation_job_id=continuation_job_id,
                group_id=group_id,
                staging_id=staging["staging_id"],
                request_id=request_id,
                created_by_user_id=caller_user_id,
                contract_version=CONTINUATION_SCHEMA_VERSION,
                idempotency_key=idempotency_key,
                sealed_intent_hash=sealed_intent_hash,
                snapshot_id=snapshot.snapshot_id,
                canonical_manifest_hash=snapshot.manifest_hash,
                content_manifest_hash=content_manifest_hash,
                repository_set_hash=repository_set_hash,
                expected_repository_count=snapshot.expected_repository_count,
                expected_file_count=snapshot.expected_file_count,
                expected_supported_file_count=snapshot.expected_supported_file_count,
                expected_supported_content_bytes=(
                    snapshot.expected_supported_content_bytes
                ),
                expected_part_count=len(snapshot.expected_parts),
                state="queued",
                attempt_count=0,
                lease_token=None,
                lease_expires_at=None,
                error_code=None,
                error_fingerprint=None,
                created_at=now,
                updated_at=now,
                finished_at=None,
            )
        )
        self._stage(ControlWriteStage.CONTINUATION)
        return (
            connection.execute(
                select(snapshot_continuation_job).where(
                    snapshot_continuation_job.c.continuation_job_id
                    == continuation_job_id
                )
            )
            .mappings()
            .one()
        )

    def commit_snapshot(
        self,
        payload: Mapping[str, Any],
        *,
        caller_user_id: str,
        path_job_id: str,
    ) -> dict[str, object]:
        request_id, group_id = _protocol_request(
            payload,
            message_type="snapshot_commit",
            required={"job_id", "snapshot_id", "parts", "content_manifest_hash"},
        )
        job_id = _uuid(payload["job_id"])
        if job_id != path_job_id:
            raise ControlPlaneError("not_found_or_forbidden", status_code=404)
        expired = False
        with self.engine.begin() as connection:
            self._authorize_group(connection, user_id=caller_user_id, group_id=group_id)
            staging = self._load_open_staging(
                connection, group_id=group_id, job_id=job_id
            )
            snapshot = self._snapshot_from_staging(staging)
            self._authorize_snapshot_repositories(
                connection, group_id=group_id, snapshot=snapshot
            )
            now = self.clock()
            if staging["status"] == "open" and _aware(staging["expires_at"]) <= now:
                self._expire_one(connection, staging=staging, now=now)
                expired = True
            else:
                parts_value = payload["parts"]
                if (
                    not isinstance(parts_value, list)
                    or len(parts_value) > MAX_CONTENT_PARTS
                ):
                    raise ControlPlaneError("limit_exceeded", status_code=413)
                normalized_parts: list[dict[str, Any]] = []
                for index, item in enumerate(parts_value, start=1):
                    if not isinstance(item, dict):
                        raise ControlPlaneError("invalid_contract")
                    _exact_keys(item, {"part_ordinal", "part_sha256"})
                    ordinal = _integer(
                        item["part_ordinal"], minimum=1, maximum=MAX_CONTENT_PARTS
                    )
                    if ordinal != index:
                        raise ControlPlaneError("invalid_contract")
                    normalized_parts.append(
                        {
                            "part_ordinal": ordinal,
                            "part_sha256": _sha256(item["part_sha256"]),
                        }
                    )
                manifest_hash = canonical_sha256(normalized_parts)
                declared_manifest_hash = _sha256(payload["content_manifest_hash"])
                stored_parts = (
                    connection.execute(
                        select(snapshot_content_part)
                        .where(
                            snapshot_content_part.c.staging_id == staging["staging_id"]
                        )
                        .order_by(snapshot_content_part.c.part_ordinal)
                    )
                    .mappings()
                    .all()
                )
                stored_descriptors = [
                    {
                        "part_ordinal": int(item["part_ordinal"]),
                        "part_sha256": str(item["part_sha256"]),
                    }
                    for item in stored_parts
                ]
                if staging["status"] == "sealed":
                    if (
                        declared_manifest_hash != staging["content_manifest_hash"]
                        or normalized_parts != stored_descriptors
                    ):
                        raise ControlPlaneError("commit_conflict", status_code=409)
                    job = (
                        connection.execute(
                            select(control_job).where(control_job.c.job_id == job_id)
                        )
                        .mappings()
                        .one()
                    )
                    self._ensure_continuation_job(
                        connection,
                        request_id=request_id,
                        caller_user_id=caller_user_id,
                        staging_job=job,
                        staging=staging,
                        snapshot=snapshot,
                        content_manifest_hash=declared_manifest_hash,
                        now=now,
                    )
                    return self._status_from_rows(
                        request_id=request_id,
                        job=job,
                        staging=staging,
                        replayed=True,
                    )
                if staging["status"] != "open":
                    raise ControlPlaneError("staging_not_open", status_code=409)
                if (
                    payload["snapshot_id"] != snapshot.snapshot_id
                    or len(stored_parts) != len(snapshot.expected_parts)
                    or normalized_parts != stored_descriptors
                    or manifest_hash != declared_manifest_hash
                    or int(staging["received_file_count"])
                    != snapshot.expected_supported_file_count
                    or int(staging["received_content_bytes"])
                    != snapshot.expected_supported_content_bytes
                ):
                    raise ControlPlaneError("incomplete_staging", status_code=409)
                connection.execute(
                    update(snapshot_staging)
                    .where(snapshot_staging.c.staging_id == staging["staging_id"])
                    .values(
                        status="sealed",
                        content_manifest_hash=manifest_hash,
                        updated_at=now,
                        sealed_at=now,
                    )
                )
                staging_job = (
                    connection.execute(
                        select(control_job).where(control_job.c.job_id == job_id)
                    )
                    .mappings()
                    .one()
                )
                self._ensure_continuation_job(
                    connection,
                    request_id=request_id,
                    caller_user_id=caller_user_id,
                    staging_job=staging_job,
                    staging=staging,
                    snapshot=snapshot,
                    content_manifest_hash=manifest_hash,
                    now=now,
                )
                connection.execute(
                    update(control_job)
                    .where(control_job.c.job_id == job_id)
                    .values(
                        state="succeeded",
                        lease_token=None,
                        lease_expires_at=None,
                        progress_completed=len(stored_parts),
                        result_snapshot_id=snapshot.snapshot_id,
                        error_code=None,
                        error_fingerprint=None,
                        updated_at=now,
                        finished_at=now,
                    )
                )
                self._stage(ControlWriteStage.COMMIT)
                staging = dict(staging)
                staging.update(
                    {
                        "status": "sealed",
                        "content_manifest_hash": manifest_hash,
                        "updated_at": now,
                        "sealed_at": now,
                    }
                )
                job = (
                    connection.execute(
                        select(control_job).where(control_job.c.job_id == job_id)
                    )
                    .mappings()
                    .one()
                )
                result = self._status_from_rows(
                    request_id=request_id,
                    job=job,
                    staging=staging,
                    replayed=False,
                )
        if expired:
            raise ControlPlaneError("staging_expired", status_code=409)
        return result

    @staticmethod
    def _expire_one(
        connection: Connection, *, staging: Mapping[str, Any], now: datetime
    ) -> None:
        connection.execute(
            update(snapshot_staging)
            .where(snapshot_staging.c.staging_id == staging["staging_id"])
            .values(status="expired", updated_at=now)
        )
        connection.execute(
            update(control_job)
            .where(control_job.c.job_id == staging["job_id"])
            .values(
                state="terminal_failed",
                lease_token=None,
                lease_expires_at=None,
                error_code="staging_expired",
                error_fingerprint=hashlib.sha256(b"staging_expired").hexdigest(),
                updated_at=now,
                finished_at=now,
            )
        )

    def expire_staging_sessions(self) -> int:
        now = self.clock()
        count = 0
        with self.engine.begin() as connection:
            rows = (
                connection.execute(
                    select(snapshot_staging).where(
                        snapshot_staging.c.status == "open",
                        snapshot_staging.c.expires_at <= now,
                    )
                )
                .mappings()
                .all()
            )
            for row in rows:
                staging_job = (
                    connection.execute(
                        select(
                            control_job.c.state,
                            control_job.c.lease_expires_at,
                        ).where(control_job.c.job_id == row["job_id"])
                    )
                    .mappings()
                    .one()
                )
                if (
                    staging_job["state"] == "running"
                    and staging_job["lease_expires_at"] is not None
                    and _aware(staging_job["lease_expires_at"]) > now
                ):
                    continue
                self._expire_one(connection, staging=row, now=now)
                count += 1
        return count

    def job_status(
        self,
        payload: Mapping[str, Any],
        *,
        caller_user_id: str,
    ) -> dict[str, object]:
        request_id, group_id = _protocol_request(
            payload,
            message_type="job_status_request",
            required={"job_id"},
        )
        job_id = _uuid(payload["job_id"])
        with self.engine.begin() as connection:
            self._authorize_group(connection, user_id=caller_user_id, group_id=group_id)
            job = (
                connection.execute(
                    select(control_job).where(
                        control_job.c.group_id == group_id,
                        control_job.c.job_id == job_id,
                    )
                )
                .mappings()
                .first()
            )
            if job is None:
                raise ControlPlaneError("not_found_or_forbidden", status_code=404)
            staging = (
                connection.execute(
                    select(snapshot_staging).where(
                        snapshot_staging.c.group_id == group_id,
                        snapshot_staging.c.job_id == job_id,
                    )
                )
                .mappings()
                .first()
            )
            index_extension = None
            if job["operation"] == "index_build":
                index_extension = (
                    connection.execute(
                        select(compatible_index_job).where(
                            compatible_index_job.c.group_id == group_id,
                            compatible_index_job.c.job_id == job_id,
                        )
                    )
                    .mappings()
                    .first()
                )
                if index_extension is None:
                    raise ControlPlaneError("job_state_incompatible", status_code=409)
            now = self.clock()
            if (
                staging is not None
                and staging["status"] == "open"
                and _aware(staging["expires_at"]) <= now
            ):
                self._expire_one(connection, staging=staging, now=now)
                job = (
                    connection.execute(
                        select(control_job).where(
                            control_job.c.group_id == group_id,
                            control_job.c.job_id == job_id,
                        )
                    )
                    .mappings()
                    .one()
                )
                staging = (
                    connection.execute(
                        select(snapshot_staging).where(
                            snapshot_staging.c.group_id == group_id,
                            snapshot_staging.c.job_id == job_id,
                        )
                    )
                    .mappings()
                    .one()
                )
            return self._status_from_rows(
                request_id=request_id,
                job=job,
                staging=staging,
                index_extension=index_extension,
                replayed=False,
            )

    @staticmethod
    def _timestamp(value: datetime | None) -> str | None:
        if value is None:
            return None
        return _aware(value).isoformat().replace("+00:00", "Z")

    def _status_from_rows(
        self,
        *,
        request_id: str,
        job: Mapping[str, Any],
        staging: Mapping[str, Any] | None,
        replayed: bool,
        index_extension: Mapping[str, Any] | None = None,
    ) -> dict[str, object]:
        state = str(job["state"])
        if state not in _JOB_STATES:  # pragma: no cover - database corruption
            raise ControlPlaneError("job_state_incompatible", status_code=409)
        result: dict[str, object] | None = None
        if job["result_snapshot_id"]:
            result = {
                "snapshot_id": str(job["result_snapshot_id"]),
                "staging_state": str(staging["status"]) if staging else None,
                "corpus_eligible": False,
                "index_eligible": False,
            }
        elif index_extension is not None:
            has_index_result = index_extension["result_index_id"] is not None
            if (state == "succeeded") != has_index_result:
                # Never expose a publication from an inconsistent or partially
                # committed job. Normal writes cannot reach this state because
                # publication/result/success share the builder transaction.
                raise ControlPlaneError("job_state_incompatible", status_code=409)
            if has_index_result:
                result = {
                    "corpus_generation_id": str(index_extension["generation_id"]),
                    "index_publication_id": str(index_extension["result_index_id"]),
                }
        response: dict[str, object] = {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "message_type": "job_status",
            "request_id": request_id,
            "group_id": str(job["group_id"]),
            "job_id": str(job["job_id"]),
            "operation": str(job["operation"]),
            "state": state,
            "attempt": int(job["attempt_count"]),
            "created_at": self._timestamp(job["created_at"]),
            "updated_at": self._timestamp(job["updated_at"]),
            "progress": {
                "completed": int(job["progress_completed"]),
                "total": int(job["progress_total"]),
            },
            "result": result,
            "error_code": str(job["error_code"]) if job["error_code"] else None,
            "staging": (
                {
                    "state": str(staging["status"]),
                    "received_parts": int(staging["received_part_count"]),
                    "expected_parts": int(staging["expected_part_count"]),
                    "expires_at": self._timestamp(staging["expires_at"]),
                    "corpus_eligible": False,
                    "index_eligible": False,
                }
                if staging is not None
                else None
            ),
            "replayed": replayed,
        }
        return response

    def _continuation_summary(
        self, continuation: Mapping[str, Any]
    ) -> dict[str, object]:
        succeeded = continuation["state"] == "succeeded"
        return {
            "job_id": str(continuation["continuation_job_id"]),
            "operation": "sealed_snapshot_continue",
            "state": str(continuation["state"]),
            "attempt": int(continuation["attempt_count"]),
            "created_at": self._timestamp(continuation["created_at"]),
            "updated_at": self._timestamp(continuation["updated_at"]),
            "error_code": (
                str(continuation["error_code"]) if continuation["error_code"] else None
            ),
            "corpus_ingestion_complete": succeeded,
            "corpus_eligible": succeeded,
            "index_eligible": False,
            "baseline_eligible": False,
        }

    def continuation_status(
        self,
        payload: Mapping[str, Any],
        *,
        caller_user_id: str,
    ) -> dict[str, object]:
        request_id, group_id, staging_job_id, continuation_job_id = (
            _continuation_status_request(payload)
        )
        with self.engine.connect() as connection:
            self._authorize_group(connection, user_id=caller_user_id, group_id=group_id)
            if staging_job_id is not None:
                staging = (
                    connection.execute(
                        select(snapshot_staging).where(
                            snapshot_staging.c.group_id == group_id,
                            snapshot_staging.c.job_id == staging_job_id,
                        )
                    )
                    .mappings()
                    .first()
                )
                if staging is None:
                    raise ControlPlaneError("not_found_or_forbidden", status_code=404)
                continuation = (
                    connection.execute(
                        select(snapshot_continuation_job).where(
                            snapshot_continuation_job.c.group_id == group_id,
                            snapshot_continuation_job.c.staging_id
                            == staging["staging_id"],
                        )
                    )
                    .mappings()
                    .first()
                )
            else:
                continuation = (
                    connection.execute(
                        select(snapshot_continuation_job).where(
                            snapshot_continuation_job.c.group_id == group_id,
                            snapshot_continuation_job.c.continuation_job_id
                            == continuation_job_id,
                        )
                    )
                    .mappings()
                    .first()
                )
                staging = (
                    connection.execute(
                        select(snapshot_staging).where(
                            snapshot_staging.c.group_id == group_id,
                            snapshot_staging.c.staging_id
                            == (
                                continuation["staging_id"]
                                if continuation is not None
                                else ""
                            ),
                        )
                    )
                    .mappings()
                    .first()
                )
            if continuation is None or staging is None:
                raise ControlPlaneError("not_found_or_forbidden", status_code=404)
            return self._continuation_status_from_rows(
                request_id=request_id,
                continuation=continuation,
                staging=staging,
                replayed=False,
            )

    def _continuation_status_from_rows(
        self,
        *,
        request_id: str,
        continuation: Mapping[str, Any],
        staging: Mapping[str, Any],
        replayed: bool,
    ) -> dict[str, object]:
        state = str(continuation["state"])
        if state not in _JOB_STATES:  # pragma: no cover - database corruption
            raise ControlPlaneError("job_state_incompatible", status_code=409)
        succeeded = state == "succeeded"
        result: dict[str, object] = {
            "snapshot_id": str(continuation["snapshot_id"]),
            "staging_state": str(staging["status"]),
            "corpus_ingestion_complete": succeeded,
            "corpus_eligible": succeeded,
            "index_eligible": False,
            "baseline_eligible": False,
            "index_state": "incomplete" if succeeded else "unavailable",
        }
        if succeeded:
            result.update(
                {
                    "corpus_id": str(continuation["result_corpus_id"]),
                    "corpus_generation_id": str(continuation["result_generation_id"]),
                    "corpus_generation_version": str(
                        continuation["result_generation_version"]
                    ),
                    "corpus_manifest_hash": str(continuation["result_manifest_hash"]),
                    "corpus_provenance_fingerprint": str(
                        continuation["result_provenance_fingerprint"]
                    ),
                    "worker_contract_version": str(
                        continuation["result_worker_contract_version"]
                    ),
                }
            )
        return {
            "schema_version": CONTINUATION_SCHEMA_VERSION,
            "message_type": "continuation_job_status",
            "request_id": request_id,
            "group_id": str(continuation["group_id"]),
            "staging_job_id": str(staging["job_id"]),
            "job_id": str(continuation["continuation_job_id"]),
            "operation": "sealed_snapshot_continue",
            "state": state,
            "attempt": int(continuation["attempt_count"]),
            "created_at": self._timestamp(continuation["created_at"]),
            "updated_at": self._timestamp(continuation["updated_at"]),
            "progress": {"completed": 1 if succeeded else 0, "total": 1},
            "result": result,
            "error_code": (
                str(continuation["error_code"]) if continuation["error_code"] else None
            ),
            "staging": {
                "state": str(staging["status"]),
                "received_parts": int(staging["received_part_count"]),
                "expected_parts": int(staging["expected_part_count"]),
                "expires_at": self._timestamp(staging["expires_at"]),
                "corpus_eligible": False,
                "index_eligible": False,
            },
            "continuation": self._continuation_summary(continuation),
            "replayed": replayed,
        }

    def _validate_continuation_claim(
        self,
        connection: Connection,
        *,
        continuation: Mapping[str, Any],
        caller_user_id: str,
    ) -> Mapping[str, Any]:
        group_id = str(continuation["group_id"])
        if continuation["created_by_user_id"] != caller_user_id:
            raise ControlPlaneError("not_found_or_forbidden", status_code=404)
        staging = (
            connection.execute(
                select(snapshot_staging).where(
                    snapshot_staging.c.group_id == group_id,
                    snapshot_staging.c.staging_id == continuation["staging_id"],
                )
            )
            .mappings()
            .first()
        )
        if staging is None or staging["status"] != "sealed":
            raise ControlPlaneError("staging_not_sealed", status_code=409)
        snapshot = self._snapshot_from_staging(staging)
        self._authorize_snapshot_repositories(
            connection, group_id=group_id, snapshot=snapshot
        )
        descriptors = (
            connection.execute(
                select(
                    snapshot_content_part.c.part_ordinal,
                    snapshot_content_part.c.part_sha256,
                    snapshot_content_part.c.item_count,
                    snapshot_content_part.c.content_bytes,
                )
                .where(snapshot_content_part.c.staging_id == staging["staging_id"])
                .order_by(snapshot_content_part.c.part_ordinal)
            )
            .mappings()
            .all()
        )
        content_manifest_hash = canonical_sha256(
            [
                {
                    "part_ordinal": int(item["part_ordinal"]),
                    "part_sha256": str(item["part_sha256"]),
                }
                for item in descriptors
            ]
        )
        expected_values = (
            snapshot.snapshot_id,
            snapshot.manifest_hash,
            str(staging["content_manifest_hash"]),
            self._repository_set_hash(snapshot),
            snapshot.expected_repository_count,
            snapshot.expected_file_count,
            snapshot.expected_supported_file_count,
            snapshot.expected_supported_content_bytes,
            len(snapshot.expected_parts),
        )
        continuation_values = (
            str(continuation["snapshot_id"]),
            str(continuation["canonical_manifest_hash"]),
            str(continuation["content_manifest_hash"]),
            str(continuation["repository_set_hash"]),
            int(continuation["expected_repository_count"]),
            int(continuation["expected_file_count"]),
            int(continuation["expected_supported_file_count"]),
            int(continuation["expected_supported_content_bytes"]),
            int(continuation["expected_part_count"]),
        )
        sealed_intent_hash = canonical_sha256(
            {
                "contract_version": CONTINUATION_SCHEMA_VERSION,
                "group_id": group_id,
                "staging_id": str(staging["staging_id"]),
                "snapshot_id": snapshot.snapshot_id,
                "canonical_manifest_hash": snapshot.manifest_hash,
                "content_manifest_hash": str(staging["content_manifest_hash"]),
                "repository_set_hash": self._repository_set_hash(snapshot),
                "expected_repository_count": snapshot.expected_repository_count,
                "expected_file_count": snapshot.expected_file_count,
                "expected_supported_file_count": (
                    snapshot.expected_supported_file_count
                ),
                "expected_supported_content_bytes": (
                    snapshot.expected_supported_content_bytes
                ),
                "expected_part_count": len(snapshot.expected_parts),
            }
        )
        if (
            continuation["contract_version"] != CONTINUATION_SCHEMA_VERSION
            or expected_values != continuation_values
            or content_manifest_hash != continuation["content_manifest_hash"]
            or len(descriptors) != len(snapshot.expected_parts)
            or sealed_intent_hash != continuation["sealed_intent_hash"]
        ):
            raise ControlPlaneError("sealed_snapshot_incompatible", status_code=409)
        if (
            sum(int(item["item_count"]) for item in descriptors)
            != snapshot.expected_supported_file_count
            or sum(int(item["content_bytes"]) for item in descriptors)
            != snapshot.expected_supported_content_bytes
        ):
            raise ControlPlaneError("sealed_snapshot_incompatible", status_code=409)
        return staging

    def claim_continuation_job(
        self,
        *,
        caller_user_id: str,
        group_id: str,
        job_id: str,
        lifetime: timedelta = DEFAULT_LEASE_LIFETIME,
    ) -> LeaseReceipt:
        now = self.clock()
        token = secrets.token_urlsafe(32)
        with self.engine.begin() as connection:
            self._authorize_group(connection, user_id=caller_user_id, group_id=group_id)
            continuation = (
                connection.execute(
                    select(snapshot_continuation_job)
                    .where(
                        snapshot_continuation_job.c.group_id == group_id,
                        snapshot_continuation_job.c.continuation_job_id == job_id,
                    )
                    .with_for_update()
                )
                .mappings()
                .first()
            )
            if continuation is None:
                raise ControlPlaneError("not_found_or_forbidden", status_code=404)
            self._validate_continuation_claim(
                connection,
                continuation=continuation,
                caller_user_id=caller_user_id,
            )
            lease_expired = (
                continuation["state"] == "running"
                and continuation["lease_expires_at"] is not None
                and _aware(continuation["lease_expires_at"]) <= now
            )
            if (
                continuation["state"] not in {"queued", "retryable_failed"}
                and not lease_expired
            ):
                raise ControlPlaneError(
                    "job_lease_unavailable", status_code=409, retryable=True
                )
            attempt = int(continuation["attempt_count"]) + 1
            expires = now + lifetime
            claimed = connection.execute(
                update(snapshot_continuation_job)
                .where(
                    snapshot_continuation_job.c.group_id == group_id,
                    snapshot_continuation_job.c.continuation_job_id == job_id,
                    (
                        snapshot_continuation_job.c.state.in_(
                            {"queued", "retryable_failed"}
                        )
                        | (
                            (snapshot_continuation_job.c.state == "running")
                            & (snapshot_continuation_job.c.lease_expires_at <= now)
                        )
                    ),
                )
                .values(
                    state="running",
                    attempt_count=attempt,
                    lease_token=token,
                    lease_expires_at=expires,
                    error_code=None,
                    error_fingerprint=None,
                    updated_at=now,
                )
            )
            if claimed.rowcount != 1:
                raise ControlPlaneError(
                    "job_lease_unavailable", status_code=409, retryable=True
                )
        return LeaseReceipt(job_id, token, expires, attempt)

    def record_continuation_failure(
        self,
        *,
        caller_user_id: str,
        group_id: str,
        job_id: str,
        lease_token: str,
        error_code: str,
        retryable: bool,
    ) -> None:
        token = _string(lease_token, maximum=128)
        safe_error = _string(error_code, maximum=128, pattern=_IDENTIFIER)
        now = self.clock()
        state = "retryable_failed" if retryable else "terminal_failed"
        with self.engine.begin() as connection:
            self._authorize_group(connection, user_id=caller_user_id, group_id=group_id)
            continuation = (
                connection.execute(
                    select(snapshot_continuation_job).where(
                        snapshot_continuation_job.c.group_id == group_id,
                        snapshot_continuation_job.c.continuation_job_id == job_id,
                    )
                )
                .mappings()
                .first()
            )
            if (
                continuation is None
                or continuation["created_by_user_id"] != caller_user_id
            ):
                raise ControlPlaneError("not_found_or_forbidden", status_code=404)
            updated = connection.execute(
                update(snapshot_continuation_job)
                .where(
                    snapshot_continuation_job.c.group_id == group_id,
                    snapshot_continuation_job.c.continuation_job_id == job_id,
                    snapshot_continuation_job.c.state == "running",
                    snapshot_continuation_job.c.lease_token == token,
                )
                .values(
                    state=state,
                    lease_token=None,
                    lease_expires_at=None,
                    error_code=safe_error,
                    error_fingerprint=hashlib.sha256(
                        safe_error.encode("utf-8")
                    ).hexdigest(),
                    updated_at=now,
                    finished_at=None if retryable else now,
                )
            )
            if updated.rowcount != 1:
                raise ControlPlaneError(
                    "job_lease_unavailable", status_code=409, retryable=True
                )

    def acquire_job_lease(
        self,
        *,
        caller_user_id: str,
        group_id: str,
        job_id: str,
        lifetime: timedelta = DEFAULT_LEASE_LIFETIME,
    ) -> LeaseReceipt:
        now = self.clock()
        token = secrets.token_urlsafe(32)
        with self.engine.begin() as connection:
            self._authorize_group(connection, user_id=caller_user_id, group_id=group_id)
            job = (
                connection.execute(
                    select(control_job)
                    .where(
                        control_job.c.group_id == group_id,
                        control_job.c.job_id == job_id,
                    )
                    .with_for_update()
                )
                .mappings()
                .first()
            )
            if job is None:
                raise ControlPlaneError("not_found_or_forbidden", status_code=404)
            lease_expired = (
                job["state"] == "running"
                and job["lease_expires_at"] is not None
                and _aware(job["lease_expires_at"]) <= now
            )
            if job["state"] not in {"queued", "retryable_failed"} and not lease_expired:
                raise ControlPlaneError(
                    "job_lease_unavailable", status_code=409, retryable=True
                )
            attempt = int(job["attempt_count"]) + 1
            expires = now + lifetime
            connection.execute(
                update(control_job)
                .where(
                    control_job.c.group_id == group_id,
                    control_job.c.job_id == job_id,
                )
                .values(
                    state="running",
                    attempt_count=attempt,
                    lease_token=token,
                    lease_expires_at=expires,
                    error_code=None,
                    error_fingerprint=None,
                    updated_at=now,
                )
            )
        return LeaseReceipt(job_id, token, expires, attempt)


# The frozen baseline-control-plane.v1 schema permits only ``unavailable`` for
# index_build. Keep admission and advertisement on one source of truth until a
# future versioned protocol can truthfully advertise authenticated submission.
INDEX_BUILD_CAPABILITY_STATUS = "unavailable"


def index_build_submission_available() -> bool:
    return INDEX_BUILD_CAPABILITY_STATUS == "safe"


def capabilities_response(
    *,
    request_id: str,
    group_id: str,
    transport: ControlTransportCapability,
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "message_type": "capabilities",
        "request_id": request_id,
        "group_id": group_id,
        "operations": {
            "snapshot_staging": "safe" if transport.available else "unavailable",
            "corpus_ingestion": "unavailable",
            "index_build": INDEX_BUILD_CAPABILITY_STATUS,
            "baseline_run": "unavailable",
        },
        "transport": transport.to_dict(),
        "request_body_logging": False,
        "staging_is_corpus_eligible": False,
        "staging_is_index_eligible": False,
        "limits": {
            "sibling_repositories": MAX_SIBLING_REPOSITORIES,
            "file_records": MAX_FILE_RECORDS,
            "file_bytes": MAX_FILE_BYTES,
            "supported_content_bytes": MAX_SUPPORTED_CONTENT_BYTES,
            "manifest_request_bytes": MAX_MANIFEST_REQUEST_BYTES,
            "content_part_request_bytes": MAX_CONTENT_PART_REQUEST_BYTES,
            "content_part_bytes": MAX_CONTENT_PART_BYTES,
            "content_part_items": MAX_CONTENT_PART_ITEMS,
            "content_parts": MAX_CONTENT_PARTS,
            "control_request_bytes": MAX_CONTROL_REQUEST_BYTES,
            "staging_lifetime_seconds": int(STAGING_LIFETIME.total_seconds()),
        },
    }


def validate_capabilities_request(
    payload: Mapping[str, Any],
) -> tuple[str, str]:
    return _protocol_request(
        payload,
        message_type="capabilities_request",
        required=set(),
    )


__all__ = [
    "INDEX_BUILD_CAPABILITY_STATUS",
    "MAX_CONTENT_PARTS",
    "MAX_CONTENT_PART_BYTES",
    "MAX_CONTENT_PART_ITEMS",
    "MAX_CONTENT_PART_REQUEST_BYTES",
    "MAX_CONTROL_REQUEST_BYTES",
    "MAX_MANIFEST_REQUEST_BYTES",
    "PROTOCOL_SHA256",
    "PROTOCOL_VERSION",
    "BaselineControlPlaneService",
    "ControlPlaneError",
    "ControlTransportCapability",
    "ControlTransportStatus",
    "ControlWriteStage",
    "DuplicateJSONKeyError",
    "LeaseReceipt",
    "ParsedJSONBody",
    "assess_control_transport",
    "canonical_sha256",
    "canonicalize",
    "capabilities_response",
    "decode_json_object",
    "index_build_submission_available",
    "require_control_transport",
    "validate_capabilities_request",
    "validate_snapshot_manifest",
]
