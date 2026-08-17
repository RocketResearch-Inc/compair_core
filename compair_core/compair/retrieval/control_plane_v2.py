"""Frozen v2 capability and compatible-index HTTP contract helpers.

This module is deliberately limited to the already-existing compatible-index
continuation.  It owns no job, corpus, index, retrieval-run, or query storage.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import inspect, select
from sqlalchemy.exc import SQLAlchemyError

from compair_core.baseline_control_plane_schema import (
    COMPATIBLE_INDEX_JOB_TABLE,
    CONTROL_JOB_TABLE,
    SNAPSHOT_CONTINUATION_JOB_TABLE,
)
from compair_core.schema_migrations import (
    CORE_SCHEMA_MIGRATIONS,
    MIGRATION_TABLE_NAME,
    schema_migration_table,
)

from .baseline import BASELINE_TOKENIZER_VERSION
from .control_plane import canonicalize
from .embedding import (
    BASELINE_EMBEDDING_HTTP_CONTRACT,
    BASELINE_EMBEDDING_HTTP_PROVIDER,
)
from .index_continuation import (
    BASELINE_EMBEDDING_DTYPE,
    PINNED_BASELINE_DIMENSION,
    PINNED_BASELINE_MODEL,
    BaselineCompatibleIndexJobService,
    IndexJobError,
)
from .indexing import (
    BASELINE_INDEX_SCHEMA_VERSION,
    BaselineEmbeddingIdentity,
    baseline_engine_config_fingerprint,
)

PROTOCOL_V2_VERSION = "baseline-control-plane.v2"
PROTOCOL_V2_SHA256 = "b278abe007779f05e92509db068f555701c03cba5cf236151e8df231a9b44091"
PROTOCOL_V1_VERSION = "baseline-control-plane.v1"
PROTOCOL_V1_SHA256 = "3b45287a54d04cea751e9cc3209c5f0783192de13062e682eadcae40af322650"

MAX_V2_CONTROL_REQUEST_BYTES = 64_000
MAX_V2_RUN_REQUEST_BYTES = 8_100_000
MAX_V2_RAW_QUERY_BYTES = 8_000_000
MAX_V2_SELECTED_EVIDENCE_CHARACTERS = 16_000
_INDEX_MIGRATION_ID = "0008_baseline_compatible_index_job_v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INDEX_TABLES = frozenset(
    {
        MIGRATION_TABLE_NAME,
        CONTROL_JOB_TABLE,
        SNAPSHOT_CONTINUATION_JOB_TABLE,
        COMPATIBLE_INDEX_JOB_TABLE,
        "retrieval_corpus",
        "retrieval_corpus_generation",
        "retrieval_corpus_ingestion",
        "retrieval_corpus_file",
        "retrieval_index_state",
        "retrieval_baseline_index_build",
        "retrieval_baseline_index_document",
        "retrieval_baseline_index_term",
        "retrieval_baseline_index_vector",
        "retrieval_baseline_index_publication",
    }
)
_SAFE_REASON_CODES = frozenset(
    {
        "authorization_revoked",
        "capability_unavailable",
        "corpus_incompatible",
        "embedding_identity_mismatch",
        "embedding_unavailable",
        "generation_blocked",
        "generation_malformed",
        "generation_terminal_failure",
        "idempotency_conflict",
        "index_build_failed",
        "index_publication_stale",
        "index_vector_invalid",
        "internal_failure",
        "job_cancelled",
        "job_not_found_or_forbidden",
        "limit_exceeded",
        "protocol_mismatch",
        "repository_not_authorized",
        "retrieval_error",
        "retrieval_insufficient",
        "source_not_authorized",
        "transport_unavailable",
        "worker_unavailable",
    }
)


class V2ControlPlaneError(RuntimeError):
    """Frozen, non-reflective v2 error representation."""

    def __init__(
        self,
        code: str,
        *,
        status_code: int,
        stage: str,
        retryable: bool = False,
    ) -> None:
        self.code = code if code in _SAFE_REASON_CODES else "internal_failure"
        self.status_code = status_code
        self.stage = stage
        self.retryable = retryable
        super().__init__(self.code)

    def as_dict(self, request_id: str | None) -> dict[str, object]:
        return {
            "protocol_version": PROTOCOL_V2_VERSION,
            "protocol_sha256": PROTOCOL_V2_SHA256,
            "message_type": "error",
            "request_id": request_id,
            "http_status": self.status_code,
            "stage": self.stage,
            "retryable": self.retryable,
            "code": self.code,
        }


@dataclass(frozen=True, slots=True)
class V2IndexBuildSubmission:
    request_id: str
    group_id: str
    idempotency_key: str
    continuation_id: str
    generation_id: str
    corpus_manifest_hash: str
    ingestion_provenance_fingerprint: str
    index_format_version: str
    tokenizer_version: str
    retrieval_config_fingerprint: str
    embedding_contract_version: str
    embedding: BaselineEmbeddingIdentity


@dataclass(frozen=True, slots=True)
class V2IndexCapability:
    readiness: str
    reason_code: str | None
    identity: BaselineEmbeddingIdentity
    dispatch: str = "manual"

    @property
    def ready(self) -> bool:
        return self.readiness == "ready"


@dataclass(frozen=True, slots=True)
class V2RunCapability:
    """Frozen baseline-run operation projection.

    ``available`` distinguishes the default-off endpoint from an explicitly
    enabled deployment whose runtime prerequisites are temporarily not ready.
    """

    available: bool
    readiness: str
    reason_code: str | None
    dispatch: str = "manual"

    @property
    def ready(self) -> bool:
        return self.available and self.readiness == "ready"

    def as_dict(self) -> dict[str, object]:
        if not self.available:
            return {
                "submission": "unavailable",
                "endpoint": "unavailable",
                "dispatch": "unavailable",
                "readiness": "unavailable",
                "reason_code": "capability_unavailable",
            }
        return {
            "submission": "safe",
            "endpoint": "authenticated_post",
            "dispatch": self.dispatch,
            "readiness": self.readiness,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True, slots=True)
class V2IndexPublication:
    index_publication_id: str
    corpus_generation_id: str
    corpus_manifest_hash: str
    index_format_version: str
    tokenizer_version: str
    retrieval_config_fingerprint: str
    embedding_fingerprint: str
    index_fingerprint: str


@dataclass(frozen=True, slots=True, repr=False)
class V2RetrievalQuery:
    representation: str
    origin: str
    encoding: str
    base_revision: str
    head_revision: str
    byte_size: int
    sha256: str
    text: str

    def __repr__(self) -> str:
        return "V2RetrievalQuery(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class V2RunSubmission:
    request_id: str
    group_id: str
    idempotency_key: str
    source_document_id: str
    changed_repository_registration_id: str
    index_publication: V2IndexPublication
    retrieval_query: V2RetrievalQuery

    def __repr__(self) -> str:
        return "V2RunSubmission(<redacted>)"


def _exact_keys(value: Mapping[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise V2ControlPlaneError(
            "protocol_mismatch", status_code=422, stage="protocol"
        )


def _uuid(value: Any) -> str:
    if not isinstance(value, str):
        raise V2ControlPlaneError(
            "protocol_mismatch", status_code=422, stage="protocol"
        )
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError):
        raise V2ControlPlaneError(
            "protocol_mismatch", status_code=422, stage="protocol"
        ) from None
    if str(parsed) != value.lower():
        raise V2ControlPlaneError(
            "protocol_mismatch", status_code=422, stage="protocol"
        )
    return str(parsed)


def _identifier(value: Any, *, minimum: int = 1, maximum: int = 128) -> str:
    if (
        not isinstance(value, str)
        or not minimum <= len(value) <= maximum
        or _SAFE_ID.fullmatch(value) is None
    ):
        raise V2ControlPlaneError(
            "protocol_mismatch", status_code=422, stage="protocol"
        )
    return value


def _text(value: Any, *, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= maximum
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise V2ControlPlaneError(
            "protocol_mismatch", status_code=422, stage="protocol"
        )
    return value


def _sha256(value: Any) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise V2ControlPlaneError(
            "protocol_mismatch", status_code=422, stage="protocol"
        )
    return value


def _protocol(payload: Mapping[str, Any], message_type: str) -> None:
    if (
        payload.get("protocol_version") != PROTOCOL_V2_VERSION
        or payload.get("protocol_sha256") != PROTOCOL_V2_SHA256
        or payload.get("message_type") != message_type
    ):
        raise V2ControlPlaneError(
            "protocol_mismatch", status_code=409, stage="protocol"
        )


def parse_capabilities_request(payload: Mapping[str, Any]) -> tuple[str, str]:
    _exact_keys(
        payload,
        {
            "protocol_version",
            "protocol_sha256",
            "message_type",
            "request_id",
            "group_id",
        },
    )
    _protocol(payload, "capabilities_request")
    canonicalize(payload)
    return _uuid(payload["request_id"]), _identifier(payload["group_id"], maximum=64)


def parse_index_status_request(payload: Mapping[str, Any]) -> tuple[str, str, str]:
    _exact_keys(
        payload,
        {
            "protocol_version",
            "protocol_sha256",
            "message_type",
            "request_id",
            "group_id",
            "job_id",
            "operation",
        },
    )
    _protocol(payload, "job_status_request")
    canonicalize(payload)
    if payload["operation"] != "index_build":
        raise V2ControlPlaneError(
            "protocol_mismatch", status_code=422, stage="protocol"
        )
    return (
        _uuid(payload["request_id"]),
        _identifier(payload["group_id"], maximum=64),
        _uuid(payload["job_id"]),
    )


def parse_run_status_request(payload: Mapping[str, Any]) -> tuple[str, str, str]:
    _exact_keys(
        payload,
        {
            "protocol_version",
            "protocol_sha256",
            "message_type",
            "request_id",
            "group_id",
            "job_id",
            "operation",
        },
    )
    _protocol(payload, "job_status_request")
    canonicalize(payload)
    if payload["operation"] != "baseline_run":
        raise V2ControlPlaneError(
            "protocol_mismatch", status_code=422, stage="protocol"
        )
    return (
        _uuid(payload["request_id"]),
        _identifier(payload["group_id"], maximum=64),
        _uuid(payload["job_id"]),
    )


def parse_index_build_submission(
    payload: Mapping[str, Any],
) -> V2IndexBuildSubmission:
    _exact_keys(
        payload,
        {
            "protocol_version",
            "protocol_sha256",
            "message_type",
            "request_id",
            "group_id",
            "idempotency_key",
            "ingestion_continuation_id",
            "corpus_generation_id",
            "corpus_manifest_hash",
            "ingestion_provenance_fingerprint",
            "index_intent",
        },
    )
    _protocol(payload, "index_build_submit")
    canonicalize(payload)
    intent = payload["index_intent"]
    if not isinstance(intent, Mapping):
        raise V2ControlPlaneError(
            "protocol_mismatch", status_code=422, stage="protocol"
        )
    _exact_keys(
        intent,
        {
            "index_format_version",
            "tokenizer_version",
            "retrieval_config_fingerprint",
            "embedding",
        },
    )
    embedding = intent["embedding"]
    if not isinstance(embedding, Mapping):
        raise V2ControlPlaneError(
            "protocol_mismatch", status_code=422, stage="protocol"
        )
    _exact_keys(
        embedding,
        {
            "contract_version",
            "provider",
            "model",
            "revision",
            "dimension",
            "dtype",
            "fingerprint",
        },
    )
    if (
        intent["index_format_version"] != BASELINE_INDEX_SCHEMA_VERSION
        or intent["tokenizer_version"] != BASELINE_TOKENIZER_VERSION
        or embedding["contract_version"] != BASELINE_EMBEDDING_HTTP_CONTRACT
        or embedding["provider"] != BASELINE_EMBEDDING_HTTP_PROVIDER
        or embedding["model"] != PINNED_BASELINE_MODEL
        or embedding["dimension"] != PINNED_BASELINE_DIMENSION
        or isinstance(embedding["dimension"], bool)
        or embedding["dtype"] != BASELINE_EMBEDDING_DTYPE
    ):
        raise V2ControlPlaneError(
            "protocol_mismatch", status_code=422, stage="protocol"
        )
    identity = BaselineEmbeddingIdentity(
        provider=BASELINE_EMBEDDING_HTTP_PROVIDER,
        model=PINNED_BASELINE_MODEL,
        revision=_text(embedding["revision"], maximum=128),
        dimension=PINNED_BASELINE_DIMENSION,
        fingerprint=_sha256(embedding["fingerprint"]),
    )
    return V2IndexBuildSubmission(
        request_id=_uuid(payload["request_id"]),
        group_id=_identifier(payload["group_id"], maximum=64),
        idempotency_key=_identifier(
            payload["idempotency_key"], minimum=32, maximum=128
        ),
        continuation_id=_uuid(payload["ingestion_continuation_id"]),
        generation_id=_uuid(payload["corpus_generation_id"]),
        corpus_manifest_hash=_sha256(payload["corpus_manifest_hash"]),
        ingestion_provenance_fingerprint=_sha256(
            payload["ingestion_provenance_fingerprint"]
        ),
        index_format_version=BASELINE_INDEX_SCHEMA_VERSION,
        tokenizer_version=BASELINE_TOKENIZER_VERSION,
        retrieval_config_fingerprint=_sha256(intent["retrieval_config_fingerprint"]),
        embedding_contract_version=BASELINE_EMBEDDING_HTTP_CONTRACT,
        embedding=identity,
    )


def parse_run_submission(payload: Mapping[str, Any]) -> V2RunSubmission:
    """Validate the frozen v2 run message without exposing it through HTTP."""

    _exact_keys(
        payload,
        {
            "protocol_version",
            "protocol_sha256",
            "message_type",
            "request_id",
            "group_id",
            "idempotency_key",
            "source_document_id",
            "changed_repository_registration_id",
            "index_publication",
            "retrieval_query",
        },
    )
    _protocol(payload, "run_submit")
    publication = payload["index_publication"]
    query = payload["retrieval_query"]
    if not isinstance(publication, Mapping) or not isinstance(query, Mapping):
        raise V2ControlPlaneError(
            "protocol_mismatch", status_code=422, stage="protocol"
        )
    _exact_keys(
        publication,
        {
            "index_publication_id",
            "corpus_generation_id",
            "corpus_manifest_hash",
            "index_format_version",
            "tokenizer_version",
            "retrieval_config_fingerprint",
            "embedding_fingerprint",
            "index_fingerprint",
        },
    )
    _exact_keys(
        query,
        {
            "representation",
            "origin",
            "encoding",
            "base_revision",
            "head_revision",
            "byte_size",
            "sha256",
            "text",
        },
    )
    if (
        publication["index_format_version"] != BASELINE_INDEX_SCHEMA_VERSION
        or publication["tokenizer_version"] != BASELINE_TOKENIZER_VERSION
        or query["representation"] != "raw_git_diff_v1"
        or query["origin"] != "explicit"
        or query["encoding"] != "utf-8"
    ):
        raise V2ControlPlaneError(
            "protocol_mismatch", status_code=422, stage="protocol"
        )
    raw_query = query["text"]
    if not isinstance(raw_query, str) or not raw_query.strip():
        raise V2ControlPlaneError(
            "protocol_mismatch", status_code=422, stage="protocol"
        )
    try:
        query_bytes = raw_query.encode("utf-8")
    except UnicodeEncodeError:
        raise V2ControlPlaneError(
            "protocol_mismatch", status_code=422, stage="protocol"
        ) from None
    byte_size = query["byte_size"]
    if (
        isinstance(byte_size, bool)
        or not isinstance(byte_size, int)
        or len(raw_query) > MAX_V2_RAW_QUERY_BYTES
        or len(query_bytes) > MAX_V2_RAW_QUERY_BYTES
        or byte_size != len(query_bytes)
        or query["sha256"] != hashlib.sha256(query_bytes).hexdigest()
    ):
        raise V2ControlPlaneError(
            "limit_exceeded"
            if len(query_bytes) > MAX_V2_RAW_QUERY_BYTES
            else "protocol_mismatch",
            status_code=413 if len(query_bytes) > MAX_V2_RAW_QUERY_BYTES else 422,
            stage="protocol",
        )
    revision_pattern = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    base_revision = query["base_revision"]
    head_revision = query["head_revision"]
    if (
        not isinstance(base_revision, str)
        or revision_pattern.fullmatch(base_revision) is None
        or not isinstance(head_revision, str)
        or revision_pattern.fullmatch(head_revision) is None
    ):
        raise V2ControlPlaneError(
            "protocol_mismatch", status_code=422, stage="protocol"
        )
    canonicalize(payload)
    return V2RunSubmission(
        request_id=_uuid(payload["request_id"]),
        group_id=_identifier(payload["group_id"], maximum=64),
        idempotency_key=_identifier(
            payload["idempotency_key"], minimum=32, maximum=128
        ),
        source_document_id=_uuid(payload["source_document_id"]),
        changed_repository_registration_id=_uuid(
            payload["changed_repository_registration_id"]
        ),
        index_publication=V2IndexPublication(
            index_publication_id=_uuid(publication["index_publication_id"]),
            corpus_generation_id=_uuid(publication["corpus_generation_id"]),
            corpus_manifest_hash=_sha256(publication["corpus_manifest_hash"]),
            index_format_version=BASELINE_INDEX_SCHEMA_VERSION,
            tokenizer_version=BASELINE_TOKENIZER_VERSION,
            retrieval_config_fingerprint=_sha256(
                publication["retrieval_config_fingerprint"]
            ),
            embedding_fingerprint=_sha256(publication["embedding_fingerprint"]),
            index_fingerprint=_sha256(publication["index_fingerprint"]),
        ),
        retrieval_query=V2RetrievalQuery(
            representation="raw_git_diff_v1",
            origin="explicit",
            encoding="utf-8",
            base_revision=base_revision,
            head_revision=head_revision,
            byte_size=byte_size,
            sha256=_sha256(query["sha256"]),
            text=raw_query,
        ),
    )


def _fallback_identity() -> BaselineEmbeddingIdentity:
    return BaselineEmbeddingIdentity(
        provider=BASELINE_EMBEDDING_HTTP_PROVIDER,
        model=PINNED_BASELINE_MODEL,
        revision="unconfigured",
        dimension=PINNED_BASELINE_DIMENSION,
        fingerprint=hashlib.sha256(b"baseline-embedding-unconfigured").hexdigest(),
    )


def assess_index_build_capability(
    service: BaselineCompatibleIndexJobService,
) -> V2IndexCapability:
    """Read-only readiness check; it never creates or repairs schema."""

    identity = _fallback_identity()
    try:
        table_names = set(inspect(service.engine).get_table_names())
        if not _INDEX_TABLES <= table_names:
            return V2IndexCapability("not_ready", "capability_unavailable", identity)
        migration = next(
            item
            for item in CORE_SCHEMA_MIGRATIONS
            if item.migration_id == _INDEX_MIGRATION_ID
        )
        with service.engine.connect() as connection:
            row = (
                connection.execute(
                    select(schema_migration_table).where(
                        schema_migration_table.c.migration_id == _INDEX_MIGRATION_ID
                    )
                )
                .mappings()
                .first()
            )
            if (
                row is None
                or row["state"] != "applied"
                or row["checksum"] != migration.checksum
            ):
                return V2IndexCapability(
                    "not_ready", "capability_unavailable", identity
                )
            connection.exec_driver_sql("SELECT 1").scalar_one()
    except (SQLAlchemyError, StopIteration):
        return V2IndexCapability("not_ready", "capability_unavailable", identity)

    try:
        identity = service.attest_configured_identity()
    except IndexJobError as exc:
        reason = (
            "embedding_identity_mismatch"
            if "identity" in exc.code or "dimension" in exc.code
            else "embedding_unavailable"
        )
        return V2IndexCapability("not_ready", reason, identity)
    return V2IndexCapability("ready", None, identity)


def unavailable_index_build_capability() -> V2IndexCapability:
    """Return the frozen safe state when the service itself cannot be built."""

    return V2IndexCapability(
        "not_ready", "capability_unavailable", _fallback_identity()
    )


def unavailable_run_capability() -> V2RunCapability:
    """Return the frozen default-off run capability."""

    return V2RunCapability(False, "unavailable", "capability_unavailable")


def not_ready_run_capability(
    reason_code: str, *, dispatch: str = "manual"
) -> V2RunCapability:
    """Return an enabled but fail-closed manual run capability."""

    reason = (
        reason_code if reason_code in _SAFE_REASON_CODES else "capability_unavailable"
    )
    return V2RunCapability(True, "not_ready", reason, dispatch)


def ready_run_capability(*, dispatch: str = "manual") -> V2RunCapability:
    """Return the only capability that permits a durable run submission."""

    return V2RunCapability(True, "ready", None, dispatch)


def _index_intent(identity: BaselineEmbeddingIdentity) -> dict[str, object]:
    return {
        "index_format_version": BASELINE_INDEX_SCHEMA_VERSION,
        "tokenizer_version": BASELINE_TOKENIZER_VERSION,
        "retrieval_config_fingerprint": baseline_engine_config_fingerprint(identity),
        "embedding": {
            "contract_version": BASELINE_EMBEDDING_HTTP_CONTRACT,
            "provider": identity.provider,
            "model": identity.model,
            "revision": identity.revision,
            "dimension": identity.dimension,
            "dtype": BASELINE_EMBEDDING_DTYPE,
            "fingerprint": identity.fingerprint,
        },
    }


def capabilities_response(
    *,
    request_id: str,
    group_id: str,
    capability: V2IndexCapability,
    run_capability: V2RunCapability | None = None,
) -> dict[str, object]:
    run_operation = (run_capability or unavailable_run_capability()).as_dict()
    return {
        "protocol_version": PROTOCOL_V2_VERSION,
        "protocol_sha256": PROTOCOL_V2_SHA256,
        "message_type": "capabilities",
        "request_id": request_id,
        "group_id": group_id,
        "supported_protocols": [
            {
                "version": PROTOCOL_V1_VERSION,
                "sha256": PROTOCOL_V1_SHA256,
                "role": "staging_only",
            },
            {
                "version": PROTOCOL_V2_VERSION,
                "sha256": PROTOCOL_V2_SHA256,
                "role": "index_and_run_submission",
            },
        ],
        "operations": {
            "index_build": {
                "submission": "safe",
                "endpoint": "authenticated_post",
                "dispatch": capability.dispatch,
                "readiness": capability.readiness,
                "reason_code": capability.reason_code,
            },
            "baseline_run": run_operation,
        },
        "limits": {
            "control_request_bytes": MAX_V2_CONTROL_REQUEST_BYTES,
            "run_request_bytes": MAX_V2_RUN_REQUEST_BYTES,
            "raw_query_bytes": MAX_V2_RAW_QUERY_BYTES,
            "idempotency_key_min_characters": 32,
            "idempotency_key_max_characters": 128,
            "selected_evidence_items": 4,
            "selected_evidence_characters": MAX_V2_SELECTED_EVIDENCE_CHARACTERS,
            "feedback_items": 4,
            "terminal_status_retention_days": 30,
        },
        "required_index_identity": _index_intent(capability.identity),
        "transport": {
            "remote": "verified_https_required",
            "loopback_http": "explicit_actual_peer_exception",
            "json_media_type": "application/json",
            "encoding": "utf-8",
        },
    }


def accepted_response(
    *, submission: V2IndexBuildSubmission, accepted: Mapping[str, Any]
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_V2_VERSION,
        "protocol_sha256": PROTOCOL_V2_SHA256,
        "message_type": "job_accepted",
        "request_id": submission.request_id,
        "group_id": submission.group_id,
        "job_id": str(accepted["job_id"]),
        "operation": "index_build",
        # Acceptance acknowledges the durable queue identity. Existing state is
        # obtained through the status endpoint on a replay.
        "state": "queued",
        "replayed": bool(accepted["replayed"]),
        "processing_run_id": None,
    }


def _timestamp(value: datetime) -> str:
    aware = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return aware.isoformat().replace("+00:00", "Z")


def status_response(
    *, request_id: str, snapshot: Mapping[str, Any]
) -> dict[str, object]:
    state = str(snapshot["state"])
    terminal = state in {"succeeded", "terminal_failed", "cancelled"}
    exit_classification = {
        "succeeded": "success",
        "terminal_failed": "failed",
        "cancelled": "cancelled",
    }.get(state, "pending")
    reason_code = None
    if state == "cancelled":
        reason_code = "job_cancelled"
    elif state in {"retryable_failed", "terminal_failed"}:
        reason_code = safe_index_error_code(str(snapshot["error_code"] or ""))
    identity = BaselineEmbeddingIdentity(
        provider=str(snapshot["embedding_provider"]),
        model=str(snapshot["embedding_model"]),
        revision=str(snapshot["embedding_revision"]),
        dimension=int(snapshot["embedding_dimension"]),
        fingerprint=str(snapshot["embedding_fingerprint"]),
    )
    result = snapshot.get("result")
    count = int(snapshot.get("document_count") or 0)
    return {
        "protocol_version": PROTOCOL_V2_VERSION,
        "protocol_sha256": PROTOCOL_V2_SHA256,
        "message_type": "job_status",
        "request_id": request_id,
        "group_id": str(snapshot["group_id"]),
        "job_id": str(snapshot["job_id"]),
        "operation": "index_build",
        "state": state,
        "terminal": terminal,
        "exit_classification": exit_classification,
        "attempt": int(snapshot["attempt_count"]),
        "created_at": _timestamp(snapshot["created_at"]),
        "updated_at": _timestamp(snapshot["updated_at"]),
        "ingestion_continuation_id": str(snapshot["continuation_job_id"]),
        "corpus_generation_id": str(snapshot["generation_id"]),
        "corpus_manifest_hash": str(snapshot["corpus_manifest_hash"]),
        "index_intent": _index_intent(identity),
        "progress": {"document_count": count, "vector_count": count},
        "result": result,
        "reason_code": reason_code,
        "replayed": False,
    }


def safe_index_error_code(code: str) -> str:
    if code in _SAFE_REASON_CODES:
        return code
    if code in {"worker_cancelled", "job_cancelled"}:
        return "job_cancelled"
    if code in {"not_found_or_forbidden", "job_not_found_or_forbidden"}:
        return "job_not_found_or_forbidden"
    if code in {"index_build_conflict", "idempotency_conflict"}:
        return "idempotency_conflict"
    if code == "repository_not_authorized":
        return "repository_not_authorized"
    if code in {"source_not_authorized", "authorization_revoked"}:
        return code
    if code in {
        "corpus_generation_stale",
        "corpus_generation_mismatch",
        "index_publication_mismatch",
    }:
        return "index_publication_stale"
    if code.startswith(("corpus_", "index_state_")) or code in {
        "ingestion_provenance_mismatch",
        "ingestion_continuation_not_succeeded",
    }:
        return "corpus_incompatible"
    if code == "index_intent_incompatible":
        return "embedding_identity_mismatch"
    if "identity" in code or "fingerprint" in code or "dimension" in code:
        return "embedding_identity_mismatch"
    if code.startswith("embedding_service_") or code in {
        "embedding_adapter_unavailable",
        "embedding_provider_disabled",
    }:
        return "embedding_unavailable"
    if "vector" in code or code.endswith("nonfinite"):
        return "index_vector_invalid"
    return "index_build_failed"


def from_index_job_error(error: IndexJobError, *, stage: str) -> V2ControlPlaneError:
    code = safe_index_error_code(error.code)
    status = error.status_code
    if code == "job_not_found_or_forbidden":
        status = 404
    elif code == "idempotency_conflict":
        status = 409
    elif error.retryable:
        status = 503
    if status not in {400, 401, 403, 404, 409, 413, 422, 429, 503}:
        status = 503
    return V2ControlPlaneError(
        code,
        status_code=status,
        stage=stage,
        retryable=error.retryable,
    )


__all__ = [
    "MAX_V2_CONTROL_REQUEST_BYTES",
    "PROTOCOL_V2_SHA256",
    "PROTOCOL_V2_VERSION",
    "V2ControlPlaneError",
    "V2IndexBuildSubmission",
    "V2IndexCapability",
    "V2IndexPublication",
    "V2RetrievalQuery",
    "V2RunCapability",
    "V2RunSubmission",
    "accepted_response",
    "assess_index_build_capability",
    "capabilities_response",
    "from_index_job_error",
    "not_ready_run_capability",
    "parse_capabilities_request",
    "parse_index_build_submission",
    "parse_index_status_request",
    "parse_run_status_request",
    "parse_run_submission",
    "ready_run_capability",
    "safe_index_error_code",
    "status_response",
    "unavailable_index_build_capability",
    "unavailable_run_capability",
]
