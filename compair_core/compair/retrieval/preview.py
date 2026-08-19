"""Authorized, read-only preview of completed document-level baseline jobs.

The control job is the only authoritative run identity. This module reads safe
provenance plus authorized Feedback text; it never reads selected evidence
content, renderer output, the protected query payload, or provider bodies, and
it never mutates delivery state.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from ...baseline_control_plane_schema import (
    BASELINE_CONTROL_GENERATION_CONTRACT_VERSION,
)
from ...baseline_evidence_schema import (
    SOURCE_SCOPE_CONTROL_DOCUMENT,
    SOURCE_SCOPE_LEGACY_CHUNK,
    SOURCE_SCOPE_VERSION,
)
from .control_document_scope import (
    ControlDocumentCorpusScopeError,
    control_document_corpus_identity,
)
from .generation import (
    GENERATION_OUTPUT_SCHEMA_SHA256,
    GENERATION_OUTPUT_SCHEMA_VERSION,
)
from .notification_outbox import (
    BASELINE_NOTIFICATION_CHANNEL,
    BaselineNotificationOutboxError,
    load_authorized_baseline_notification_digest,
)

BASELINE_PREVIEW_SCHEMA_VERSION = "baseline-preview.v1"
BASELINE_PREVIEW_MAX_REQUEST_BYTES = 4_096
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


class BaselinePreviewError(RuntimeError):
    """Safe preview failure which never includes private row data."""

    def __init__(
        self, code: str, message: str = "baseline preview is unavailable"
    ) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class BaselinePreviewCommand:
    caller_user_id: str
    request_id: str
    group_id: str
    job_id: str | None = None
    digest_id: str | None = None


@dataclass(frozen=True, slots=True)
class BaselinePreviewFeedback:
    ordinal: int
    feedback_id: str
    feedback: str


@dataclass(frozen=True, slots=True)
class BaselinePreviewControlJob:
    job_id: str
    state: str
    completed_at: str
    generation_invoked: bool
    feedback_count: int
    notification_outbox_count: int


@dataclass(frozen=True, slots=True)
class BaselinePreviewRetrieval:
    persisted_run_id: str
    status: str
    evidence_count: int
    reference_count: int


@dataclass(frozen=True, slots=True)
class BaselinePreviewDigest:
    digest_id: str
    state: str
    channel: str
    finding_count: int
    finding_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class BaselinePreviewSource:
    group_id: str
    document_id: str
    source_scope: str
    chunk_id: str | None


@dataclass(frozen=True, slots=True)
class BaselinePreviewQueryProvenance:
    sha256: str
    length: int
    origin: str


@dataclass(frozen=True, slots=True)
class BaselinePreviewRetrievalProvenance:
    engine: str
    version: str
    result_schema_version: str


@dataclass(frozen=True, slots=True)
class BaselinePreviewCorpusProvenance:
    generation_id: str
    generation_version: str
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class BaselinePreviewIndexProvenance:
    publication_id: str
    publication_fingerprint: str
    index_id: str
    version: str
    schema_version: str
    fingerprint: str
    config_fingerprint: str


@dataclass(frozen=True, slots=True)
class BaselinePreviewEmbeddingProvenance:
    provider: str
    model: str
    revision: str
    dimension: int
    fingerprint: str


@dataclass(frozen=True, slots=True)
class BaselinePreviewGenerationProvenance:
    provider: str
    model: str
    version: str
    input_fingerprint: str
    output_fingerprint: str


@dataclass(frozen=True, slots=True)
class BaselinePreviewProvenance:
    retrieval: BaselinePreviewRetrievalProvenance
    query: BaselinePreviewQueryProvenance
    corpus: BaselinePreviewCorpusProvenance
    index: BaselinePreviewIndexProvenance
    embedding: BaselinePreviewEmbeddingProvenance
    generation: BaselinePreviewGenerationProvenance


@dataclass(frozen=True, slots=True)
class BaselinePreview:
    request_id: str
    control_job: BaselinePreviewControlJob
    retrieval: BaselinePreviewRetrieval
    source: BaselinePreviewSource
    feedback: tuple[BaselinePreviewFeedback, ...]
    digest: BaselinePreviewDigest | None
    provenance: BaselinePreviewProvenance
    schema_version: str = BASELINE_PREVIEW_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["feedback"] = list(payload["feedback"])
        return payload


def _required_uuid(name: str, value: object) -> str:
    if not isinstance(value, str) or value != value.strip():
        raise BaselinePreviewError(
            "baseline_preview_request_invalid", f"{name} is invalid"
        )
    try:
        return str(UUID(value))
    except (ValueError, AttributeError) as exc:
        raise BaselinePreviewError(
            "baseline_preview_request_invalid", f"{name} is invalid"
        ) from exc


def parse_baseline_preview_request(
    payload: dict[str, Any], *, caller_user_id: str
) -> BaselinePreviewCommand:
    """Validate the exact post-parse request object without compatibility aliases."""

    if not isinstance(payload, dict):
        raise BaselinePreviewError("baseline_preview_request_invalid")
    allowed = {"schema_version", "request_id", "group_id", "job_id", "digest_id"}
    if not set(payload) <= allowed or payload.get("schema_version") != (
        BASELINE_PREVIEW_SCHEMA_VERSION
    ):
        raise BaselinePreviewError("baseline_preview_request_invalid")
    job_id = payload.get("job_id")
    digest_id = payload.get("digest_id")
    if (job_id is None) == (digest_id is None):
        raise BaselinePreviewError("baseline_preview_request_invalid")
    expected = {"schema_version", "request_id", "group_id"}
    expected.add("job_id" if job_id is not None else "digest_id")
    if set(payload) != expected:
        raise BaselinePreviewError("baseline_preview_request_invalid")
    return BaselinePreviewCommand(
        caller_user_id=_required_uuid("caller_user_id", caller_user_id),
        request_id=_required_uuid("request_id", payload["request_id"]),
        group_id=_required_uuid("group_id", payload["group_id"]),
        job_id=(_required_uuid("job_id", job_id) if job_id is not None else None),
        digest_id=(
            _required_uuid("digest_id", digest_id) if digest_id is not None else None
        ),
    )


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and _HEX_64.fullmatch(value) is not None


def _iso8601(value: object) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise BaselinePreviewError("baseline_preview_unavailable") from exc
    else:
        raise BaselinePreviewError("baseline_preview_unavailable")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class BaselinePreviewService:
    """Load one terminal control job after fresh durable authorization checks."""

    def __init__(self, session_factory) -> None:
        self._session_factory = session_factory

    def load(self, command: BaselinePreviewCommand) -> BaselinePreview:
        caller_user_id = _required_uuid("caller_user_id", command.caller_user_id)
        request_id = _required_uuid("request_id", command.request_id)
        group_id = _required_uuid("group_id", command.group_id)
        if (command.job_id is None) == (command.digest_id is None):
            raise BaselinePreviewError("baseline_preview_request_invalid")
        job_id = (
            _required_uuid("job_id", command.job_id)
            if command.job_id is not None
            else None
        )
        digest_id = (
            _required_uuid("digest_id", command.digest_id)
            if command.digest_id is not None
            else None
        )

        with self._session_factory() as session:
            row = self._resolve_row(
                session,
                caller_user_id=caller_user_id,
                group_id=group_id,
                job_id=job_id,
                digest_id=digest_id,
            )
            evidence_rows = self._load_evidence_reference_manifest(
                session, str(row["persisted_run_id"])
            )
            feedback_rows = self._load_feedback(session, str(row["persisted_run_id"]))
            outbox_rows = self._load_outbox(session, str(row["persisted_run_id"]))
            self._validate_durable_result(
                row,
                evidence_rows=evidence_rows,
                feedback_rows=feedback_rows,
                outbox_rows=outbox_rows,
                caller_user_id=caller_user_id,
                selected_digest_id=digest_id,
                session=session,
            )
            final = self._resolve_row(
                session,
                caller_user_id=caller_user_id,
                group_id=group_id,
                job_id=str(row["job_id"]),
                digest_id=None,
            )
            if str(final["persisted_run_id"]) != str(row["persisted_run_id"]):
                raise BaselinePreviewError("baseline_preview_unavailable")
            return self._serialize(
                request_id=request_id,
                row=row,
                feedback_rows=feedback_rows,
                outbox_rows=outbox_rows,
            )

    @staticmethod
    def _resolve_row(
        session: Session,
        *,
        caller_user_id: str,
        group_id: str,
        job_id: str | None,
        digest_id: str | None,
    ) -> dict[str, object]:
        selector_sql = (
            "j.job_id = :selector"
            if job_id is not None
            else "EXISTS (SELECT 1 FROM baseline_notification_outbox selected_o "
            "WHERE selected_o.outbox_id = :selector "
            "AND selected_o.run_id = j.persisted_run_id "
            "AND selected_o.group_id = j.group_id "
            "AND selected_o.recipient_user_id = :caller_user_id "
            "AND selected_o.channel = :channel)"
        )
        row = (
            session.execute(
                text(
                    "SELECT j.job_id, j.group_id, j.submitted_by_user_id, "
                    "j.source_document_id AS job_source_document_id, j.state AS job_state, "
                    "j.changed_repository_registration_id, j.corpus_id AS job_corpus_id, "
                    "j.corpus_generation_id "
                    "AS job_corpus_generation_id, j.index_publication_id, "
                    "j.corpus_manifest_hash AS job_corpus_manifest_hash, "
                    "index_job.corpus_manifest_hash AS submission_manifest_hash, "
                    "index_job.corpus_file_manifest_hash AS publication_file_manifest_hash, "
                    "j.retrieval_config_fingerprint AS job_config_fingerprint, "
                    "j.embedding_fingerprint AS job_embedding_fingerprint, "
                    "j.index_fingerprint AS job_index_fingerprint, "
                    "j.query_sha256 AS job_query_sha256, "
                    "j.query_length AS job_query_length, "
                    "j.query_origin AS job_query_origin, "
                    "j.retrieval_result_fingerprint, j.evidence_count, "
                    "j.reference_count, j.feedback_count, j.generation_invoked, "
                    "j.notification_outbox_count, j.persisted_run_id, "
                    "j.generation_contract_version, j.generation_provider "
                    "AS job_generation_provider, j.generation_model "
                    "AS job_generation_model, j.generation_model_version "
                    "AS job_generation_model_version, j.generation_provider_fingerprint, "
                    "j.generation_output_schema_version, "
                    "j.generation_output_schema_sha256, j.generation_input_fingerprint "
                    "AS job_generation_input_fingerprint, "
                    "j.generation_output_fingerprint AS job_generation_output_fingerprint, "
                    "j.generation_completed_at, j.finished_at, "
                    "r.source_scope_version, r.source_scope, r.source_chunk_id, "
                    "r.source_document_id, r.corpus_id AS run_corpus_id, "
                    "r.corpus_scope_key, r.retrieval_status, r.result_schema_version, "
                    "r.engine, r.engine_version, r.config_fingerprint, "
                    "r.query_sha256, r.query_length, r.query_origin, "
                    "r.corpus_generation_id, r.corpus_generation_version, "
                    "r.corpus_manifest_hash, r.index_publication_fingerprint, "
                    "r.index_id, r.index_version, r.index_schema_version, "
                    "r.index_fingerprint, r.embedding_provider, r.embedding_model, "
                    "r.embedding_revision, r.embedding_dimension, "
                    "r.embedding_fingerprint, r.selected_count, r.generation_state, "
                    "r.generation_provider, r.generation_model, "
                    "r.generation_model_version, r.generation_input_fingerprint, "
                    "r.generation_output_fingerprint, r.generation_completed_at "
                    "AS run_generation_completed_at, corpus.scope_key "
                    "AS stored_corpus_scope_key "
                    "FROM baseline_control_run_job j "
                    "JOIN baseline_retrieval_run r ON r.run_id = j.persisted_run_id "
                    "AND r.group_id = j.group_id "
                    "JOIN baseline_compatible_index_job index_job ON "
                    "index_job.job_id = j.index_job_id "
                    "AND index_job.group_id = j.group_id "
                    "AND index_job.corpus_id = j.corpus_id "
                    "AND index_job.generation_id = j.corpus_generation_id "
                    "AND index_job.result_index_id = j.index_publication_id "
                    "JOIN baseline_control_job index_control ON "
                    "index_control.job_id = index_job.job_id "
                    "AND index_control.group_id = index_job.group_id "
                    "AND index_control.operation = 'index_build' "
                    "AND index_control.state = 'succeeded' "
                    "JOIN retrieval_corpus corpus ON corpus.corpus_id = j.corpus_id "
                    'JOIN "user" u ON u.user_id = j.submitted_by_user_id '
                    "JOIN user_to_group utg ON utg.user_id = :caller_user_id "
                    "AND utg.group_id = j.group_id "
                    "JOIN document d ON d.document_id = j.source_document_id "
                    "JOIN document_to_group dtg ON dtg.document_id = d.document_id "
                    "AND dtg.group_id = j.group_id "
                    "JOIN baseline_control_repository_registration registration "
                    "ON registration.registration_id = "
                    "j.changed_repository_registration_id "
                    "AND registration.group_id = j.group_id "
                    "AND registration.enabled = true "
                    "AND corpus.changed_repository_id = registration.registration_id "
                    "AND corpus.source_document_id = j.source_document_id "
                    "JOIN baseline_control_repository_approval approval "
                    "ON approval.registration_id = registration.registration_id "
                    "AND approval.group_id = registration.group_id "
                    "AND approval.state = 'active' "
                    f"WHERE {selector_sql} AND j.group_id = :group_id "
                    "AND j.submitted_by_user_id = :caller_user_id "
                    "AND j.state = 'feedback_persisted' "
                    "AND j.generation_invoked = true "
                    "AND j.source_document_id = r.source_document_id "
                    "AND r.generation_state = 'succeeded' "
                    "AND r.source_scope_version = :source_scope_version "
                    "AND ((r.source_scope = :control_scope "
                    "AND r.source_chunk_id IS NULL) OR "
                    "(r.source_scope = :legacy_scope AND r.source_chunk_id IS NOT NULL "
                    "AND EXISTS (SELECT 1 FROM chunk c WHERE "
                    "c.chunk_id = r.source_chunk_id "
                    "AND c.document_id = r.source_document_id))) "
                    "AND NOT EXISTS (SELECT 1 FROM baseline_control_run_payload p "
                    "WHERE p.job_id = j.job_id AND p.group_id = j.group_id)"
                ),
                {
                    "selector": job_id if job_id is not None else digest_id,
                    "group_id": group_id,
                    "caller_user_id": caller_user_id,
                    "channel": BASELINE_NOTIFICATION_CHANNEL,
                    "source_scope_version": SOURCE_SCOPE_VERSION,
                    "control_scope": SOURCE_SCOPE_CONTROL_DOCUMENT,
                    "legacy_scope": SOURCE_SCOPE_LEGACY_CHUNK,
                },
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise BaselinePreviewError("baseline_preview_unavailable")
        return dict(row)

    @staticmethod
    def _load_evidence_reference_manifest(session: Session, run_id: str):
        return (
            session.execute(
                text(
                    "SELECT s.ordinal, s.selected_evidence_id, r.reference_id, "
                    "r.source_chunk_id AS reference_source_chunk_id, "
                    "r.reference_chunk_id, r.reference_document_id, "
                    "r.reference_note_id, r.reference_type "
                    "FROM baseline_selected_evidence s "
                    "LEFT JOIN reference r ON "
                    "r.baseline_selected_evidence_id = s.selected_evidence_id "
                    "WHERE s.run_id = :run_id "
                    "ORDER BY s.ordinal ASC, r.reference_id ASC"
                ),
                {"run_id": run_id},
            )
            .mappings()
            .all()
        )

    @staticmethod
    def _load_feedback(session: Session, run_id: str):
        return (
            session.execute(
                text(
                    "SELECT feedback_id, feedback, baseline_finding_ordinal, "
                    "generation_provider, generation_model, generation_model_version, "
                    "generation_input_fingerprint, generation_output_fingerprint "
                    "FROM feedback WHERE baseline_retrieval_run_id = :run_id "
                    "ORDER BY baseline_finding_ordinal ASC, feedback_id ASC"
                ),
                {"run_id": run_id},
            )
            .mappings()
            .all()
        )

    @staticmethod
    def _load_outbox(session: Session, run_id: str):
        return (
            session.execute(
                text(
                    "SELECT outbox_id, run_id, group_id, recipient_user_id, channel, "
                    "finding_count, finding_manifest_hash, state "
                    "FROM baseline_notification_outbox WHERE run_id = :run_id "
                    "ORDER BY outbox_id"
                ),
                {"run_id": run_id},
            )
            .mappings()
            .all()
        )

    @staticmethod
    def _validate_durable_result(
        row: dict[str, object],
        *,
        evidence_rows,
        feedback_rows,
        outbox_rows,
        caller_user_id: str,
        selected_digest_id: str | None,
        session: Session,
    ) -> None:
        evidence_count = int(row["evidence_count"])
        reference_count = int(row["reference_count"])
        feedback_count = int(row["feedback_count"])
        outbox_count = int(row["notification_outbox_count"])
        source_scope = str(row["source_scope"])
        source_chunk_id = row["source_chunk_id"]
        expected_source_chunk = (
            str(source_chunk_id)
            if source_scope == SOURCE_SCOPE_LEGACY_CHUNK and source_chunk_id is not None
            else None
        )
        run_generation_identity = (
            row["generation_provider"],
            row["generation_model"],
            row["generation_model_version"],
            row["generation_input_fingerprint"],
            row["generation_output_fingerprint"],
        )
        job_generation_identity = (
            row["job_generation_provider"],
            row["job_generation_model"],
            row["job_generation_model_version"],
            row["job_generation_input_fingerprint"],
            row["job_generation_output_fingerprint"],
        )
        try:
            control_corpus_identity = control_document_corpus_identity(
                group_id=str(row["group_id"]),
                changed_repository_registration_id=str(
                    row["changed_repository_registration_id"]
                ),
                source_document_id=str(row["source_document_id"]),
            )
            control_corpus_matches = control_corpus_identity.matches_stored_corpus(
                scope_key=str(row["stored_corpus_scope_key"]),
                changed_repository_id=str(row["changed_repository_registration_id"]),
                source_document_id=str(row["source_document_id"]),
            )
        except ControlDocumentCorpusScopeError:
            control_corpus_matches = False
        invalid = (
            row["job_state"] != "feedback_persisted"
            or row["retrieval_status"] != "ok"
            or row["generation_state"] != "succeeded"
            or not bool(row["generation_invoked"])
            or not 1 <= evidence_count <= 4
            or evidence_count != reference_count
            or evidence_count != int(row["selected_count"])
            or not 0 <= feedback_count <= 4
            or outbox_count not in {0, 1}
            or row["job_source_document_id"] != row["source_document_id"]
            or row["job_corpus_id"] != row["run_corpus_id"]
            or row["corpus_scope_key"] != row["stored_corpus_scope_key"]
            or not control_corpus_matches
            or row["job_corpus_generation_id"] != row["corpus_generation_id"]
            or row["job_corpus_manifest_hash"] != row["submission_manifest_hash"]
            or row["corpus_manifest_hash"] != row["publication_file_manifest_hash"]
            or row["job_config_fingerprint"] != row["config_fingerprint"]
            or row["job_embedding_fingerprint"] != row["embedding_fingerprint"]
            or row["job_index_fingerprint"] != row["index_fingerprint"]
            or row["job_query_sha256"] != row["query_sha256"]
            or int(row["job_query_length"]) != int(row["query_length"])
            or row["job_query_origin"] != row["query_origin"]
            or row["index_publication_id"] != row["index_id"]
            or run_generation_identity != job_generation_identity
            or row["generation_contract_version"]
            != BASELINE_CONTROL_GENERATION_CONTRACT_VERSION
            or row["generation_output_schema_version"]
            != GENERATION_OUTPUT_SCHEMA_VERSION
            or row["generation_output_schema_sha256"] != GENERATION_OUTPUT_SCHEMA_SHA256
            or row["generation_completed_at"] is None
            or row["run_generation_completed_at"] is None
            or _iso8601(row["generation_completed_at"])
            != _iso8601(row["run_generation_completed_at"])
            or row["finished_at"] is None
            or not _is_hash(row["retrieval_result_fingerprint"])
            or not _is_hash(row["generation_provider_fingerprint"])
            or not _is_hash(row["job_generation_input_fingerprint"])
            or not _is_hash(row["job_generation_output_fingerprint"])
            or not _is_hash(row["query_sha256"])
            or not _is_hash(row["corpus_manifest_hash"])
            or not _is_hash(row["index_publication_fingerprint"])
            or not _is_hash(row["config_fingerprint"])
            or not _is_hash(row["index_fingerprint"])
            or not _is_hash(row["embedding_fingerprint"])
            or row["engine"] != "baseline_v1"
            or not isinstance(row["engine_version"], str)
            or not row["engine_version"]
            or not isinstance(row["result_schema_version"], str)
            or not row["result_schema_version"]
            or row["query_origin"] != "explicit"
            or int(row["query_length"]) <= 0
            or int(row["embedding_dimension"]) <= 0
        )
        if invalid:
            raise BaselinePreviewError("baseline_preview_unavailable")

        if len(evidence_rows) != evidence_count:
            raise BaselinePreviewError("baseline_preview_unavailable")
        for expected_ordinal, evidence in enumerate(evidence_rows, start=1):
            if (
                int(evidence["ordinal"]) != expected_ordinal
                or evidence["reference_id"] is None
                or evidence["reference_type"] != "baseline_file"
                or evidence["reference_source_chunk_id"] != expected_source_chunk
                or evidence["reference_chunk_id"] is not None
                or evidence["reference_document_id"] is not None
                or evidence["reference_note_id"] is not None
            ):
                raise BaselinePreviewError("baseline_preview_unavailable")

        if len(feedback_rows) != feedback_count:
            raise BaselinePreviewError("baseline_preview_unavailable")
        for expected_ordinal, feedback in enumerate(feedback_rows, start=1):
            feedback_identity = (
                feedback["generation_provider"],
                feedback["generation_model"],
                feedback["generation_model_version"],
                feedback["generation_input_fingerprint"],
                feedback["generation_output_fingerprint"],
            )
            value = feedback["feedback"]
            if (
                int(feedback["baseline_finding_ordinal"]) != expected_ordinal
                or not isinstance(value, str)
                or not value.strip()
                or value.strip().upper() == "NONE"
                or feedback_identity != run_generation_identity
            ):
                raise BaselinePreviewError("baseline_preview_unavailable")

        if feedback_count == 0:
            if outbox_count != 0 or outbox_rows or selected_digest_id is not None:
                raise BaselinePreviewError("baseline_preview_unavailable")
            return
        if outbox_count != 1 or len(outbox_rows) != 1:
            raise BaselinePreviewError("baseline_preview_unavailable")
        outbox = outbox_rows[0]
        if (
            outbox["recipient_user_id"] != caller_user_id
            or outbox["group_id"] != row["group_id"]
            or outbox["channel"] != BASELINE_NOTIFICATION_CHANNEL
            or int(outbox["finding_count"]) != feedback_count
            or not _is_hash(outbox["finding_manifest_hash"])
            or (
                selected_digest_id is not None
                and outbox["outbox_id"] != selected_digest_id
            )
        ):
            raise BaselinePreviewError("baseline_preview_unavailable")
        try:
            digest = load_authorized_baseline_notification_digest(
                session,
                outbox_id=str(outbox["outbox_id"]),
                recipient_user_id=caller_user_id,
                group_id=str(row["group_id"]),
            )
        except BaselineNotificationOutboxError as exc:
            raise BaselinePreviewError("baseline_preview_unavailable") from exc
        expected_manifest = tuple(
            (int(item["baseline_finding_ordinal"]), str(item["feedback_id"]))
            for item in feedback_rows
        )
        actual_manifest = tuple(
            (item.ordinal, item.feedback_id) for item in digest.findings
        )
        if actual_manifest != expected_manifest:
            raise BaselinePreviewError("baseline_preview_unavailable")

    @staticmethod
    def _serialize(
        *,
        request_id: str,
        row: dict[str, object],
        feedback_rows,
        outbox_rows,
    ) -> BaselinePreview:
        digest = None
        if outbox_rows:
            outbox = outbox_rows[0]
            digest = BaselinePreviewDigest(
                digest_id=str(outbox["outbox_id"]),
                state=str(outbox["state"]),
                channel=str(outbox["channel"]),
                finding_count=int(outbox["finding_count"]),
                finding_manifest_sha256=str(outbox["finding_manifest_hash"]),
            )
        return BaselinePreview(
            request_id=request_id,
            control_job=BaselinePreviewControlJob(
                job_id=str(row["job_id"]),
                state=str(row["job_state"]),
                completed_at=_iso8601(row["generation_completed_at"]),
                generation_invoked=bool(row["generation_invoked"]),
                feedback_count=int(row["feedback_count"]),
                notification_outbox_count=int(row["notification_outbox_count"]),
            ),
            retrieval=BaselinePreviewRetrieval(
                persisted_run_id=str(row["persisted_run_id"]),
                status=str(row["retrieval_status"]),
                evidence_count=int(row["evidence_count"]),
                reference_count=int(row["reference_count"]),
            ),
            source=BaselinePreviewSource(
                group_id=str(row["group_id"]),
                document_id=str(row["source_document_id"]),
                source_scope=str(row["source_scope"]),
                chunk_id=(
                    str(row["source_chunk_id"])
                    if row["source_chunk_id"] is not None
                    else None
                ),
            ),
            feedback=tuple(
                BaselinePreviewFeedback(
                    ordinal=int(item["baseline_finding_ordinal"]),
                    feedback_id=str(item["feedback_id"]),
                    feedback=str(item["feedback"]),
                )
                for item in feedback_rows
            ),
            digest=digest,
            provenance=BaselinePreviewProvenance(
                retrieval=BaselinePreviewRetrievalProvenance(
                    engine=str(row["engine"]),
                    version=str(row["engine_version"]),
                    result_schema_version=str(row["result_schema_version"]),
                ),
                query=BaselinePreviewQueryProvenance(
                    sha256=str(row["query_sha256"]),
                    length=int(row["query_length"]),
                    origin=str(row["query_origin"]),
                ),
                corpus=BaselinePreviewCorpusProvenance(
                    generation_id=str(row["corpus_generation_id"]),
                    generation_version=str(row["corpus_generation_version"]),
                    manifest_sha256=str(row["corpus_manifest_hash"]),
                ),
                index=BaselinePreviewIndexProvenance(
                    publication_id=str(row["index_publication_id"]),
                    publication_fingerprint=str(row["index_publication_fingerprint"]),
                    index_id=str(row["index_id"]),
                    version=str(row["index_version"]),
                    schema_version=str(row["index_schema_version"]),
                    fingerprint=str(row["index_fingerprint"]),
                    config_fingerprint=str(row["config_fingerprint"]),
                ),
                embedding=BaselinePreviewEmbeddingProvenance(
                    provider=str(row["embedding_provider"]),
                    model=str(row["embedding_model"]),
                    revision=str(row["embedding_revision"]),
                    dimension=int(row["embedding_dimension"]),
                    fingerprint=str(row["embedding_fingerprint"]),
                ),
                generation=BaselinePreviewGenerationProvenance(
                    provider=str(row["generation_provider"]),
                    model=str(row["generation_model"]),
                    version=str(row["generation_model_version"]),
                    input_fingerprint=str(row["generation_input_fingerprint"]),
                    output_fingerprint=str(row["generation_output_fingerprint"]),
                ),
            ),
        )


__all__ = [
    "BASELINE_PREVIEW_MAX_REQUEST_BYTES",
    "BASELINE_PREVIEW_SCHEMA_VERSION",
    "BaselinePreview",
    "BaselinePreviewCommand",
    "BaselinePreviewError",
    "BaselinePreviewService",
    "parse_baseline_preview_request",
]
