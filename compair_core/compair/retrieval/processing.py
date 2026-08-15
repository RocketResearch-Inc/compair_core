"""Trace-safe processing identity and baseline integration outcomes."""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from enum import Enum

from .types import RetrievalQueryProvenance, RetrievalResult

BASELINE_PROCESSING_OUTCOME_SCHEMA_VERSION = "baseline-processing-outcome.v2"
BASELINE_DOCUMENT_PROCESSING_SCHEMA_VERSION = "baseline-document-processing.v2"
PROCESSING_RUN_KEY_BYTES = 32
MAX_BASELINE_GROUP_ID_LENGTH = 36


class ProcessingRunIdentityError(ValueError):
    """Raised when a processing-run identity is missing or malformed."""


class BaselineProcessingStatus(str, Enum):
    REFERENCES_PERSISTED = "references_persisted"
    INSUFFICIENT = "insufficient"
    ERROR = "error"


def new_processing_run_key() -> str:
    """Return a caller-owned opaque parent identity with 256 bits of entropy."""

    return secrets.token_urlsafe(PROCESSING_RUN_KEY_BYTES)


def validate_processing_run_key(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
        or any(ord(character) < 33 or ord(character) > 126 for character in value)
    ):
        raise ProcessingRunIdentityError(
            "processing_run_key must be an opaque canonical task argument"
        )
    return value


def validate_baseline_group_id(value: object) -> str:
    """Return one canonical explicit authorization scope identifier."""

    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > MAX_BASELINE_GROUP_ID_LENGTH
        or any(ord(character) < 33 or ord(character) > 126 for character in value)
    ):
        raise ProcessingRunIdentityError(
            "baseline_v1 requires an explicit canonical group_id"
        )
    return value


def processing_run_trace_id(
    processing_run_key: str,
    group_id: str | None = None,
) -> str:
    """Hash the parent key together with its explicit group intent."""

    validated = validate_processing_run_key(processing_run_key)
    group_marker = (
        validate_baseline_group_id(group_id)
        if group_id is not None
        else "<explicit-group-absent>"
    )
    return hashlib.sha256(
        b"baseline-processing-intent.v2\x00"
        + group_marker.encode("utf-8")
        + b"\x00"
        + validated.encode("utf-8")
    ).hexdigest()


def derive_baseline_persistence_idempotency_key(
    processing_run_key: str,
    group_id: str,
    source_chunk_id: str,
) -> str:
    """Derive a stable group/source-specific opaque key from one parent run."""

    parent = validate_processing_run_key(processing_run_key)
    group = validate_baseline_group_id(group_id)
    if (
        not isinstance(source_chunk_id, str)
        or not source_chunk_id
        or source_chunk_id != source_chunk_id.strip()
        or len(source_chunk_id) > 128
        or any(ord(character) < 32 for character in source_chunk_id)
    ):
        raise ProcessingRunIdentityError("source_chunk_id is invalid")
    return hmac.new(
        parent.encode("utf-8"),
        b"baseline_v1\x00"
        + group.encode("utf-8")
        + b"\x00"
        + source_chunk_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class BaselineProcessingOutcome:
    """Versioned retrieval/persistence outcome; generation is always bypassed."""

    status: BaselineProcessingStatus
    retrieval_status: str
    request_id: str
    source_chunk_id: str
    group_id: str | None
    parent_run_trace_id: str
    selected_reference_count: int
    persistence_run_id: str | None
    idempotent_replay: bool
    error_code: str | None
    query_provenance: RetrievalQueryProvenance
    corpus_id: str | None = None
    corpus_manifest_hash: str | None = None
    index_id: str | None = None
    index_fingerprint: str | None = None
    config_fingerprint: str | None = None
    embedding_fingerprint: str | None = None
    engine: str = "baseline_v1"
    generation_bypassed: bool = True
    schema_version: str = BASELINE_PROCESSING_OUTCOME_SCHEMA_VERSION

    @classmethod
    def from_result(
        cls,
        result: RetrievalResult,
        *,
        source_chunk_id: str,
        group_id: str,
        parent_run_trace_id: str,
        status: BaselineProcessingStatus,
        selected_reference_count: int = 0,
        persistence_run_id: str | None = None,
        idempotent_replay: bool = False,
        error_code: str | None = None,
    ) -> BaselineProcessingOutcome:
        if result.query_provenance is None:
            raise ValueError("baseline processing result lacks query provenance")
        return cls(
            status=status,
            retrieval_status=result.status.value,
            request_id=result.request_id,
            source_chunk_id=source_chunk_id,
            group_id=validate_baseline_group_id(group_id),
            parent_run_trace_id=parent_run_trace_id,
            selected_reference_count=selected_reference_count,
            persistence_run_id=persistence_run_id,
            idempotent_replay=idempotent_replay,
            error_code=error_code,
            query_provenance=result.query_provenance,
            corpus_id=result.corpus_id,
            corpus_manifest_hash=result.corpus_manifest_hash,
            index_id=result.index_id,
            index_fingerprint=result.index_fingerprint,
            config_fingerprint=result.config_fingerprint,
            embedding_fingerprint=result.embedding_fingerprint,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "retrieval_status": self.retrieval_status,
            "engine": self.engine,
            "generation_bypassed": self.generation_bypassed,
            "request_id": self.request_id,
            "source_chunk_id": self.source_chunk_id,
            "group_id": self.group_id,
            "parent_run_trace_id": self.parent_run_trace_id,
            "selected_reference_count": self.selected_reference_count,
            "persistence_run_id": self.persistence_run_id,
            "idempotent_replay": self.idempotent_replay,
            "error_code": self.error_code,
            "corpus_id": self.corpus_id,
            "corpus_manifest_hash": self.corpus_manifest_hash,
            "index_id": self.index_id,
            "index_fingerprint": self.index_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "embedding_fingerprint": self.embedding_fingerprint,
            **self.query_provenance.trace_fields(),
        }


def baseline_document_processing_outcome(
    outcomes: list[BaselineProcessingOutcome],
    *,
    group_id: str | None,
    parent_run_trace_id: str,
    error_code: str | None = None,
    query_provenance: RetrievalQueryProvenance | None = None,
) -> dict[str, object]:
    """Serialize safe per-chunk outcomes for the task result backend."""

    if error_code is not None or any(
        outcome.status is BaselineProcessingStatus.ERROR for outcome in outcomes
    ):
        document_status = BaselineProcessingStatus.ERROR
    elif any(
        outcome.status is BaselineProcessingStatus.INSUFFICIENT for outcome in outcomes
    ) or not outcomes:
        document_status = BaselineProcessingStatus.INSUFFICIENT
    else:
        document_status = BaselineProcessingStatus.REFERENCES_PERSISTED

    serialized: dict[str, object] = {
        "schema_version": BASELINE_DOCUMENT_PROCESSING_SCHEMA_VERSION,
        "engine": "baseline_v1",
        "generation_bypassed": True,
        "group_id": group_id,
        "parent_run_trace_id": parent_run_trace_id,
        "status": document_status.value,
        "error_code": error_code,
        "outcomes": [outcome.as_dict() for outcome in outcomes],
    }
    if query_provenance is not None:
        serialized.update(query_provenance.trace_fields())
    return serialized


__all__ = [
    "BASELINE_DOCUMENT_PROCESSING_SCHEMA_VERSION",
    "BASELINE_PROCESSING_OUTCOME_SCHEMA_VERSION",
    "BaselineProcessingOutcome",
    "BaselineProcessingStatus",
    "ProcessingRunIdentityError",
    "baseline_document_processing_outcome",
    "derive_baseline_persistence_idempotency_key",
    "new_processing_run_key",
    "processing_run_trace_id",
    "validate_baseline_group_id",
    "validate_processing_run_key",
]
