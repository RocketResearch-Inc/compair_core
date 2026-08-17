"""Transactional persistence for successful persistent ``baseline_v1`` evidence.

This module is deliberately not called by the API, task, CLI, retrieval, or
generation paths.  It is the Phase 2B2G write boundary over the schema created
by the forward-only baseline-evidence migrations.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from typing import Any
from uuid import uuid4

from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...baseline_control_plane_schema import (
    BASELINE_RUN_WORKER_CONTRACT_VERSION,
    BASELINE_RUN_WORKER_SERVICE_ID,
    baseline_run_payload,
    compatible_index_job,
    control_job,
    repository_approval,
    repository_registration,
)
from ...baseline_evidence_schema import (
    BRIDGE_SCHEMA_VERSION,
    PROVENANCE_SCHEMA_VERSION,
    RENDERER_VERSION,
    SOURCE_SCOPE_CONTROL_DOCUMENT,
    SOURCE_SCOPE_LEGACY_CHUNK,
    SOURCE_SCOPE_VERSION,
    baseline_evidence_artifact,
    baseline_retrieval_run,
    baseline_selected_evidence,
)
from .baseline import (
    MAX_EVIDENCE_CHARACTERS,
    MAX_EVIDENCE_ITEMS,
    RANKING_CONTENT_CHARACTERS,
    RETRIEVAL_LIMIT,
    baseline_ranking_document,
    normalize_retrieved_candidates,
    reciprocal_rank_fusion,
)
from .corpus import (
    BaselineIndexBuildStatus,
    CorpusFileState,
    CorpusGenerationStatus,
    CorpusIngestionStatus,
    IndexStateStatus,
    RetrievalBaselineIndexBuild,
    RetrievalBaselineIndexDocument,
    RetrievalBaselineIndexPublication,
    RetrievalCorpus,
    RetrievalCorpusFile,
    RetrievalCorpusGeneration,
    RetrievalCorpusIngestion,
    RetrievalIndexState,
)
from .indexing import BaselineIndexLifecycle
from .persistent import (
    PERSISTENT_BASELINE_ENGINE_VERSION,
    published_index_fingerprint,
)
from .types import (
    RESULT_SCHEMA_VERSION,
    RetrievalCandidate,
    RetrievalEvidence,
    RetrievalQueryOrigin,
    RetrievalResult,
    RetrievalStatus,
)

AUTHORIZATION_SCOPE_VERSION = "baseline-group-authorization.v1"
QUERY_KIND = "raw_git_diff_v1"
REFERENCE_TYPE = "baseline_file"


class PersistenceWriteStage(str, Enum):
    """Flush boundaries exposed only for deterministic rollback testing."""

    RUN = "run"
    ARTIFACTS = "artifacts"
    SELECTED_EVIDENCE = "selected_evidence"
    REFERENCES = "references"
    CONTROL_RELATIONSHIP = "control_relationship"
    PROTECTED_PAYLOAD = "protected_payload"


@dataclass(frozen=True, slots=True)
class LegacyChunkSource:
    """Authoritative chunk/document identity for the existing Core path."""

    document_id: str
    chunk_id: str


@dataclass(frozen=True, slots=True)
class ControlDocumentSource:
    """Authoritative document and leased control-job identity."""

    document_id: str
    control_job_id: str
    lease_token: str


BaselinePersistenceSource = LegacyChunkSource | ControlDocumentSource


@dataclass(frozen=True, slots=True)
class BaselineEvidencePersistenceCommand:
    """Authoritative, group-scoped intent to persist one retrieval result."""

    group_id: str
    source: BaselinePersistenceSource
    idempotency_key: str
    retrieval_result: RetrievalResult
    caller_user_id: str | None = None


@dataclass(frozen=True, slots=True)
class BaselineEvidencePersistenceReceipt:
    """Stable identifiers returned after commit or an identical replay."""

    run_id: str
    group_id: str
    idempotency_key: str
    selected_evidence_ids: tuple[str, ...]
    reference_ids: tuple[str, ...]
    replayed: bool


class BaselineEvidencePersistenceError(RuntimeError):
    """Fail-closed error with a safe machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class _SelectedIntent:
    ordinal: int
    evidence: RetrievalEvidence
    candidate: RetrievalCandidate
    document: RetrievalBaselineIndexDocument
    source: RetrievalCorpusFile
    artifact_key: str
    artifact_values: Mapping[str, object]
    selected_values: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _ValidatedIntent:
    command: BaselineEvidencePersistenceCommand
    run_values: Mapping[str, object]
    selected: tuple[_SelectedIntent, ...]
    control_job: Mapping[str, object] | None


StageHook = Callable[[PersistenceWriteStage], None]


def render_baseline_evidence(
    repository: str,
    relative_path: str,
    selected_content: str,
) -> str:
    """Frozen renderer consumed later by generation without reformatting."""

    return f"Repository file: {repository}/{relative_path}\n\n{selected_content}"


def _digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_text(value: str) -> str:
    return _digest_bytes(value.encode("utf-8"))


def _digest_json(value: object) -> str:
    return _digest_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    )


def retrieval_result_fingerprint(result: RetrievalResult) -> str:
    """Hash one result without retaining query, evidence, or path text."""

    query = result.query_provenance
    return _digest_json(
        {
            "schema_version": result.schema_version,
            "request_id": result.request_id,
            "status": result.status.value,
            "engine": result.engine,
            "engine_version": result.engine_version,
            "corpus_version": result.corpus_version,
            "corpus_id": result.corpus_id,
            "corpus_manifest_hash": result.corpus_manifest_hash,
            "corpus_scope_key": result.corpus_scope_key,
            "index_id": result.index_id,
            "index_version": result.index_version,
            "index_schema_version": result.index_schema_version,
            "index_fingerprint": result.index_fingerprint,
            "config_fingerprint": result.config_fingerprint,
            "embedding_provider": result.embedding_provider,
            "embedding_model": result.embedding_model,
            "embedding_revision": result.embedding_revision,
            "embedding_dimension": result.embedding_dimension,
            "embedding_fingerprint": result.embedding_fingerprint,
            "query": (
                None
                if query is None
                else {
                    "sha256": query.sha256,
                    "length": query.length,
                    "origin": query.origin.value,
                }
            ),
            "counts": {
                "candidate": result.candidate_count,
                "retrieved": result.retrieved_count,
                "filtered": result.filtered_count,
                "duplicate": result.duplicate_count,
                "refill": result.refill_count,
                "evidence_characters": result.evidence_characters,
                "underfilled": result.underfilled,
            },
            "candidates": [
                {
                    "document_id": item.document_id,
                    "content_hash": item.content_hash,
                    "byte_size": item.byte_size,
                    "bm25_score": item.bm25_score,
                    "bm25_rank": item.bm25_rank,
                    "dense_score": item.dense_score,
                    "dense_rank": item.dense_rank,
                    "rrf_score": item.rrf_score,
                    "fused_rank": item.fused_rank,
                }
                for item in result.candidates
            ],
            "evidence": [
                {
                    "document_id": item.document_id,
                    "content_hash": item.content_hash,
                    "fused_rank": item.fused_rank,
                    "render_truncated": item.render_truncated,
                    "bm25_score": item.bm25_score,
                    "bm25_rank": item.bm25_rank,
                    "dense_score": item.dense_score,
                    "dense_rank": item.dense_rank,
                    "rrf_score": item.rrf_score,
                }
                for item in result.evidence
            ],
            "error_code": result.error.code if result.error is not None else None,
            "fallback_engine": result.fallback_engine,
        }
    )


def _as_utc(value: object) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _identifier(value: object, label: str, max_length: int) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise BaselineEvidencePersistenceError(
            "invalid_command", f"{label} must be a canonical non-empty identifier"
        )
    if len(value) > max_length or any(ord(character) < 32 for character in value):
        raise BaselineEvidencePersistenceError(
            "invalid_command", f"{label} is not a valid identifier"
        )
    return value


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BaselineEvidencePersistenceError(
            "malformed_result", f"{label} is not a SHA-256 digest"
        )
    return value


def _normalized_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
        or any(ord(character) < 32 for character in value)
    ):
        raise BaselineEvidencePersistenceError(
            "malformed_result", "selected evidence contains an invalid relative path"
        )
    return value


def _finite_score(value: object, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BaselineEvidencePersistenceError(
            "malformed_result", f"{label} must be numeric"
        )
    converted = float(value)
    if not math.isfinite(converted) or (nonnegative and converted < 0.0):
        raise BaselineEvidencePersistenceError(
            "malformed_result", f"{label} is outside the frozen score contract"
        )
    return converted


def _positive_rank(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BaselineEvidencePersistenceError(
            "malformed_result", f"{label} must be a positive integer"
        )
    return value


def _scope_key(group_id: str) -> str:
    return f"group:{group_id}"


def _artifact_key(
    *,
    generation_id: str,
    index_fingerprint: str,
    document: RetrievalBaselineIndexDocument,
    source: RetrievalCorpusFile,
) -> str:
    return _digest_json(
        {
            "bridge_schema_version": BRIDGE_SCHEMA_VERSION,
            "corpus_generation_id": generation_id,
            "index_document_id": document.index_document_id,
            "index_fingerprint": index_fingerprint,
            "repository_id": source.repository_id,
            "repository_name": source.repository_name,
            "relative_path": source.relative_path,
            "whole_file_content_hash": source.content_hash,
        }
    )


_RUN_INTENT_COLUMNS = tuple(
    column.name
    for column in baseline_retrieval_run.columns
    if column.name
    not in {
        "run_id",
        "created_at",
        "generation_state",
        "generation_error_code",
        "generation_attempt_count",
        "generation_lease_expires_at",
        "generation_completed_at",
    }
)
_ARTIFACT_INTENT_COLUMNS = tuple(
    column.name
    for column in baseline_evidence_artifact.columns
    if column.name not in {"artifact_id", "created_at"}
)
_SELECTED_INTENT_COLUMNS = tuple(
    column.name
    for column in baseline_selected_evidence.columns
    if column.name
    not in {"selected_evidence_id", "run_id", "artifact_id", "created_at"}
)


def _row_matches(
    row: Mapping[str, object],
    expected: Mapping[str, object],
    columns: Sequence[str],
) -> bool:
    return all(row[column] == expected[column] for column in columns)


class BaselineEvidencePersistenceService:
    """Validate and atomically persist a frozen persistent-baseline result."""

    def __init__(
        self,
        session_factory: Any,
        *,
        stage_hook: StageHook | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._stage_hook = stage_hook

    def persist(
        self,
        command: BaselineEvidencePersistenceCommand,
    ) -> BaselineEvidencePersistenceReceipt:
        """Persist one intent, or return its fully validated prior receipt."""

        with self._session_factory() as session:
            try:
                self._begin_write_transaction(session)
                intent = self._validate(session, command)
                existing = (
                    session.execute(
                        select(baseline_retrieval_run).where(
                            baseline_retrieval_run.c.group_id == command.group_id,
                            baseline_retrieval_run.c.idempotency_key
                            == command.idempotency_key,
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                if existing is not None:
                    receipt = self._replay_receipt(session, intent, existing)
                    session.commit()
                    return receipt

                receipt = self._write(session, intent)
                session.commit()
                return receipt
            except BaselineEvidencePersistenceError:
                session.rollback()
                raise
            except IntegrityError as exc:
                session.rollback()
                raise BaselineEvidencePersistenceError(
                    "persistence_conflict",
                    "baseline evidence persistence violated a durable constraint",
                ) from exc
            except Exception:
                session.rollback()
                raise

    @staticmethod
    def _begin_write_transaction(session: Session) -> None:
        if session.get_bind().dialect.name == "sqlite":
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
        else:
            session.begin()

    def _validate(
        self,
        session: Session,
        command: BaselineEvidencePersistenceCommand,
    ) -> _ValidatedIntent:
        group_id = _identifier(command.group_id, "group_id", 36)
        source = command.source
        control_job: Mapping[str, object] | None = None
        if isinstance(source, LegacyChunkSource):
            source_scope = SOURCE_SCOPE_LEGACY_CHUNK
            chunk_id = _identifier(source.chunk_id, "source_chunk_id", 36)
            document_id = _identifier(source.document_id, "source_document_id", 36)
            control_job_id = None
        elif isinstance(source, ControlDocumentSource):
            source_scope = SOURCE_SCOPE_CONTROL_DOCUMENT
            chunk_id = None
            document_id = _identifier(source.document_id, "source_document_id", 36)
            control_job_id = _identifier(source.control_job_id, "control_job_id", 36)
            _identifier(source.lease_token, "control_job_lease_token", 128)
        else:
            raise BaselineEvidencePersistenceError(
                "invalid_command", "source must use a supported typed descriptor"
            )
        idempotency_key = _identifier(command.idempotency_key, "idempotency_key", 256)
        caller_user_id = (
            _identifier(command.caller_user_id, "caller_user_id", 36)
            if command.caller_user_id is not None
            else None
        )
        result = command.retrieval_result
        if not isinstance(result, RetrievalResult):
            raise BaselineEvidencePersistenceError(
                "malformed_result", "retrieval result has an unsupported type"
            )

        if source_scope == SOURCE_SCOPE_LEGACY_CHUNK:
            assert chunk_id is not None
            self._lock_and_authorize_legacy_source(
                session,
                group_id=group_id,
                chunk_id=chunk_id,
                document_id=document_id,
                caller_user_id=caller_user_id,
            )
        else:
            assert isinstance(source, ControlDocumentSource)
            control_job = self._lock_and_authorize_control_source(
                session,
                group_id=group_id,
                document_id=document_id,
                control_job_id=control_job_id or "",
                lease_token=source.lease_token,
                caller_user_id=caller_user_id,
            )
        corpus, generation, _ingestion, publication, build = self._lock_publication(
            session,
            group_id=group_id,
            source_document_id=document_id,
            result=result,
        )
        self._validate_result_header(result, corpus, generation, build)
        if control_job is not None:
            self._validate_control_job_intent(
                session,
                control_job,
                result=result,
                corpus=corpus,
                generation=generation,
                build=build,
            )

        documents = tuple(
            session.scalars(
                select(RetrievalBaselineIndexDocument)
                .where(RetrievalBaselineIndexDocument.index_id == build.index_id)
                .order_by(RetrievalBaselineIndexDocument.ordinal)
            )
        )
        sources_by_id = {
            row.file_id: row
            for row in session.scalars(
                select(RetrievalCorpusFile).where(
                    RetrievalCorpusFile.generation_id == generation.generation_id
                )
            )
        }
        documents_by_id = {
            document.index_document_id: document for document in documents
        }
        candidates_by_id = self._validate_candidates(
            result,
            documents=documents,
            sources_by_id=sources_by_id,
        )
        expected_evidence, filtered, duplicates, refills = (
            normalize_retrieved_candidates(result.candidates[:RETRIEVAL_LIMIT])
        )
        if tuple(expected_evidence) != result.evidence:
            raise BaselineEvidencePersistenceError(
                "evidence_contract_mismatch",
                "selected evidence does not match the frozen top-six selection contract",
            )
        if (
            result.retrieved_count != min(RETRIEVAL_LIMIT, len(documents))
            or result.filtered_count != filtered
            or result.duplicate_count != duplicates
            or result.refill_count != refills
            or result.evidence_characters
            != sum(len(item.content) for item in result.evidence)
            or result.underfilled != (len(result.evidence) < MAX_EVIDENCE_ITEMS)
        ):
            raise BaselineEvidencePersistenceError(
                "result_counter_mismatch",
                "retrieval result counters do not match frozen selection",
            )

        publication_fingerprint = published_index_fingerprint(build)
        assert publication.published_at is not None
        assert generation.manifest_hash is not None
        query = result.query_provenance
        assert query is not None and query.sha256 is not None
        if idempotency_key == query.sha256:
            raise BaselineEvidencePersistenceError(
                "invalid_command",
                "idempotency_key must be an opaque caller intent, not the query hash",
            )
        authorization_values: dict[str, object] = {
            "group_id": group_id,
            "source_chunk_id": chunk_id,
            "source_document_id": document_id,
            "version": AUTHORIZATION_SCOPE_VERSION,
        }
        if control_job_id is not None:
            authorization_values["control_job_id"] = control_job_id
            authorization_values["source_scope"] = source_scope
        authorization_hash = _digest_json(authorization_values)
        run_id = str(uuid4())
        run_values: dict[str, object] = {
            "run_id": run_id,
            "group_id": group_id,
            "source_chunk_id": chunk_id,
            "source_document_id": document_id,
            "source_scope_version": SOURCE_SCOPE_VERSION,
            "source_scope": source_scope,
            "idempotency_key": idempotency_key,
            "bridge_schema_version": BRIDGE_SCHEMA_VERSION,
            "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
            "renderer_version": RENDERER_VERSION,
            "request_id": result.request_id,
            "result_schema_version": result.schema_version,
            "retrieval_status": result.status.value,
            "engine": result.engine,
            "engine_version": result.engine_version,
            "config_fingerprint": result.config_fingerprint,
            "query_kind": QUERY_KIND,
            "query_sha256": query.sha256,
            "query_length": query.length,
            "query_origin": query.origin.value,
            "corpus_scope_key": corpus.scope_key,
            "corpus_id": corpus.corpus_id,
            "corpus_generation_id": generation.generation_id,
            "corpus_generation_version": generation.generation_version,
            "corpus_manifest_hash": generation.manifest_hash,
            "index_publication_fingerprint": publication_fingerprint,
            "index_published_at": publication.published_at,
            "index_id": build.index_id,
            "index_version": build.index_version,
            "index_schema_version": build.index_schema_version,
            "index_fingerprint": publication_fingerprint,
            "embedding_provider": build.embedding_provider,
            "embedding_model": build.embedding_model,
            "embedding_revision": build.embedding_revision,
            "embedding_dimension": build.embedding_dimension,
            "embedding_fingerprint": build.embedding_fingerprint,
            "authorization_scope_version": AUTHORIZATION_SCOPE_VERSION,
            "authorization_scope_hash": authorization_hash,
            "candidate_count": result.candidate_count,
            "retrieved_count": result.retrieved_count,
            "filtered_count": result.filtered_count,
            "duplicate_count": result.duplicate_count,
            "refill_count": result.refill_count,
            "selected_count": len(result.evidence),
            "evidence_character_count": result.evidence_characters,
            "underfilled": result.underfilled,
            "generation_state": "pending",
            "generation_attempt_count": 0,
        }

        selected: list[_SelectedIntent] = []
        for ordinal, evidence in enumerate(result.evidence, start=1):
            document = documents_by_id.get(evidence.document_id or "")
            candidate = candidates_by_id.get(evidence.document_id or "")
            if document is None or candidate is None:
                raise BaselineEvidencePersistenceError(
                    "selected_document_absent",
                    "selected evidence does not identify a published index document",
                )
            source = sources_by_id.get(document.corpus_file_id)
            if source is None or source.content is None:
                raise BaselineEvidencePersistenceError(
                    "selected_source_absent",
                    "selected evidence source file is unavailable",
                )
            artifact_key = _artifact_key(
                generation_id=generation.generation_id,
                index_fingerprint=publication_fingerprint,
                document=document,
                source=source,
            )
            artifact_source_document_id = self._scoped_optional_document(
                session,
                group_id=group_id,
                document_id=source.document_id,
            )
            artifact_values = {
                "group_id": group_id,
                "artifact_key": artifact_key,
                "bridge_schema_version": BRIDGE_SCHEMA_VERSION,
                "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
                "repository_id": source.repository_id,
                "repository_name": source.repository_name,
                "relative_path": source.relative_path,
                "corpus_id": corpus.corpus_id,
                "corpus_file_id": source.file_id,
                "corpus_generation_id": generation.generation_id,
                "corpus_generation_version": generation.generation_version,
                "corpus_manifest_hash": generation.manifest_hash,
                "index_publication_fingerprint": publication_fingerprint,
                "index_published_at": publication.published_at,
                "index_id": build.index_id,
                "index_document_id": document.index_document_id,
                "index_fingerprint": publication_fingerprint,
                "indexed_document_hash": document.indexed_document_hash,
                "source_document_id": artifact_source_document_id,
                "source_snapshot_id": source.source_snapshot_id,
                "complete_content": source.content,
                "whole_file_content_hash": source.content_hash,
                "byte_size": source.byte_size,
                "character_count": len(source.content),
            }
            renderer_output = render_baseline_evidence(
                evidence.repository,
                evidence.relative_path,
                evidence.content,
            )
            selected_values = {
                "group_id": group_id,
                "ordinal": ordinal,
                "fused_rank": evidence.fused_rank,
                "selected_content": evidence.content,
                "selected_content_hash": _digest_text(evidence.content),
                "selected_character_count": len(evidence.content),
                "ranking_truncated": len(source.content) > RANKING_CONTENT_CHARACTERS,
                "budget_truncated": evidence.render_truncated,
                "bm25_score": evidence.bm25_score,
                "bm25_rank": evidence.bm25_rank,
                "dense_score": evidence.dense_score,
                "dense_rank": evidence.dense_rank,
                "rrf_score": evidence.rrf_score,
                "renderer_version": RENDERER_VERSION,
                "renderer_output": renderer_output,
                "renderer_output_hash": _digest_text(renderer_output),
                "renderer_output_character_count": len(renderer_output),
            }
            selected.append(
                _SelectedIntent(
                    ordinal=ordinal,
                    evidence=evidence,
                    candidate=candidate,
                    document=document,
                    source=source,
                    artifact_key=artifact_key,
                    artifact_values=artifact_values,
                    selected_values=selected_values,
                )
            )
        return _ValidatedIntent(
            command=command,
            run_values=run_values,
            selected=tuple(selected),
            control_job=control_job,
        )

    @staticmethod
    def _lock_and_authorize_legacy_source(
        session: Session,
        *,
        group_id: str,
        chunk_id: str,
        document_id: str,
        caller_user_id: str | None,
    ) -> None:
        # Deliberately use stable Core tables rather than ORM relationships.
        # This boundary only needs an authorization fact, and avoiding mapper
        # configuration also keeps it isolated from application model routing.
        suffix = (
            " FOR UPDATE OF c, d, dtg, g"
            if (session.get_bind().dialect.name == "postgresql")
            else ""
        )
        row = session.execute(
            text(
                "SELECT c.chunk_id FROM chunk c JOIN document d "
                "ON d.document_id = c.document_id "
                "JOIN document_to_group dtg ON dtg.document_id = d.document_id "
                'JOIN "group" g ON g.group_id = dtg.group_id '
                "WHERE c.chunk_id = :chunk_id AND c.document_id = :document_id "
                "AND dtg.group_id = :group_id" + suffix
            ),
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "group_id": group_id,
            },
        ).one_or_none()
        if row is None:
            raise BaselineEvidencePersistenceError(
                "source_unauthorized",
                "source chunk/document is absent or unauthorized for the group",
            )
        if caller_user_id is not None:
            membership = session.execute(
                text(
                    'SELECT utg.user_id FROM user_to_group utg JOIN "user" u '
                    "ON u.user_id = utg.user_id "
                    "WHERE utg.user_id = :caller_user_id "
                    "AND utg.group_id = :group_id"
                    + (
                        " FOR UPDATE OF utg, u"
                        if session.get_bind().dialect.name == "postgresql"
                        else ""
                    )
                ),
                {"caller_user_id": caller_user_id, "group_id": group_id},
            ).one_or_none()
            if membership is None:
                raise BaselineEvidencePersistenceError(
                    "caller_unauthorized",
                    "caller is absent or unauthorized for the selected group",
                )

    @staticmethod
    def _lock_and_authorize_control_source(
        session: Session,
        *,
        group_id: str,
        document_id: str,
        control_job_id: str,
        lease_token: str,
        caller_user_id: str | None,
    ) -> Mapping[str, object]:
        sql = (
            "SELECT j.* FROM baseline_control_run_job j "
            "JOIN document d ON d.document_id = j.source_document_id "
            "JOIN document_to_group dtg ON dtg.document_id = d.document_id "
            "AND dtg.group_id = j.group_id "
            "JOIN user_to_group submitter_group ON "
            "submitter_group.user_id = j.submitted_by_user_id "
            "AND submitter_group.group_id = j.group_id "
            "WHERE j.job_id = :job_id AND j.group_id = :group_id "
            "AND j.source_document_id = :document_id"
        )
        if session.get_bind().dialect.name == "postgresql":
            sql += " FOR UPDATE OF j, d, dtg, submitter_group"
        row = (
            session.execute(
                text(sql),
                {
                    "job_id": control_job_id,
                    "group_id": group_id,
                    "document_id": document_id,
                },
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise BaselineEvidencePersistenceError(
                "control_source_unauthorized",
                "control job source is absent or unauthorized for the group",
            )
        values = dict(row)
        if caller_user_id is not None:
            if values.get("submitted_by_user_id") != caller_user_id:
                raise BaselineEvidencePersistenceError(
                    "caller_unauthorized",
                    "caller is not authoritative for the selected control job",
                )
            caller_authorized = session.execute(
                text(
                    "SELECT 1 FROM user_to_group WHERE user_id = :user_id "
                    "AND group_id = :group_id"
                ),
                {"user_id": caller_user_id, "group_id": group_id},
            ).scalar_one_or_none()
            if caller_authorized is None:
                raise BaselineEvidencePersistenceError(
                    "caller_unauthorized",
                    "caller is absent or unauthorized for the selected group",
                )
        state = str(values["state"])
        persisted_run_id = values.get("persisted_run_id")
        if (
            values.get("worker_service_id") != BASELINE_RUN_WORKER_SERVICE_ID
            or values.get("worker_contract_version")
            != BASELINE_RUN_WORKER_CONTRACT_VERSION
            or values.get("started_at") is None
        ):
            raise BaselineEvidencePersistenceError(
                "control_job_worker_invalid",
                "control job is not owned by the required internal worker contract",
            )
        if state == "running":
            expiry = _as_utc(values.get("lease_expires_at"))
            if (
                persisted_run_id is not None
                or values.get("retrieval_result_fingerprint") is not None
                or values.get("lease_token") != lease_token
                or expiry is None
                or expiry <= datetime.now(timezone.utc)
            ):
                raise BaselineEvidencePersistenceError(
                    "control_job_lease_invalid",
                    "control job does not hold the required active persistence lease",
                )
        elif state in {"references_persisted", "feedback_persisted"}:
            if persisted_run_id is None:
                raise BaselineEvidencePersistenceError(
                    "control_job_state_invalid",
                    "control job durable state has no retrieval-run relationship",
                )
        else:
            raise BaselineEvidencePersistenceError(
                "control_job_state_invalid",
                "control job state does not permit evidence persistence",
            )
        return values

    @staticmethod
    def _validate_control_job_intent(
        session: Session,
        job: Mapping[str, object],
        *,
        result: RetrievalResult,
        corpus: RetrievalCorpus,
        generation: RetrievalCorpusGeneration,
        build: RetrievalBaselineIndexBuild,
    ) -> None:
        query = result.query_provenance
        assert query is not None
        exact = (
            (job.get("corpus_id"), corpus.corpus_id),
            (job.get("corpus_generation_id"), generation.generation_id),
            (job.get("index_publication_id"), build.index_id),
            (job.get("index_format_version"), build.index_schema_version),
            (job.get("tokenizer_version"), build.tokenizer_version),
            (job.get("retrieval_config_fingerprint"), result.config_fingerprint),
            (job.get("embedding_fingerprint"), result.embedding_fingerprint),
            (job.get("index_fingerprint"), result.index_fingerprint),
            (job.get("query_sha256"), query.sha256),
            (job.get("query_length"), query.length),
            (job.get("query_origin"), query.origin.value),
        )
        if any(actual != expected for actual, expected in exact):
            raise BaselineEvidencePersistenceError(
                "control_job_intent_mismatch",
                "control job does not match the successful retrieval intent",
            )
        extension_statement = (
            select(compatible_index_job.c.job_id)
            .join(
                control_job,
                control_job.c.job_id == compatible_index_job.c.job_id,
            )
            .where(
                compatible_index_job.c.job_id == job.get("index_job_id"),
                compatible_index_job.c.group_id == job.get("group_id"),
                compatible_index_job.c.corpus_id == corpus.corpus_id,
                compatible_index_job.c.generation_id == generation.generation_id,
                compatible_index_job.c.result_index_id == build.index_id,
                compatible_index_job.c.corpus_manifest_hash
                == job.get("corpus_manifest_hash"),
                compatible_index_job.c.corpus_file_manifest_hash
                == generation.manifest_hash,
                compatible_index_job.c.index_format_version
                == build.index_schema_version,
                compatible_index_job.c.tokenizer_version == build.tokenizer_version,
                compatible_index_job.c.retrieval_config_fingerprint
                == result.config_fingerprint,
                compatible_index_job.c.embedding_fingerprint
                == result.embedding_fingerprint,
                control_job.c.group_id == job.get("group_id"),
                control_job.c.operation == "index_build",
                control_job.c.state == "succeeded",
            )
        )
        approval_statement = (
            select(repository_registration.c.registration_id)
            .join(
                repository_approval,
                (
                    repository_approval.c.registration_id
                    == repository_registration.c.registration_id
                )
                & (
                    repository_approval.c.group_id == repository_registration.c.group_id
                ),
            )
            .where(
                repository_registration.c.registration_id
                == job.get("changed_repository_registration_id"),
                repository_registration.c.group_id == job.get("group_id"),
                repository_registration.c.source_document_id
                == job.get("source_document_id"),
                repository_registration.c.enabled.is_(True),
                repository_approval.c.state == "active",
                corpus.changed_repository_id
                == job.get("changed_repository_registration_id"),
            )
        )
        if session.get_bind().dialect.name == "postgresql":
            extension_statement = extension_statement.with_for_update()
            approval_statement = approval_statement.with_for_update()
        if (
            session.execute(extension_statement).first() is None
            or session.execute(approval_statement).first() is None
        ):
            raise BaselineEvidencePersistenceError(
                "control_job_intent_mismatch",
                "control job publication or repository authorization is no longer current",
            )
        expected_result_fingerprint = retrieval_result_fingerprint(result)
        stored_result_fingerprint = job.get("retrieval_result_fingerprint")
        if stored_result_fingerprint not in {None, expected_result_fingerprint}:
            raise BaselineEvidencePersistenceError(
                "control_job_result_mismatch",
                "control job is bound to a different retrieval result",
            )

    @staticmethod
    def _lock_publication(
        session: Session,
        *,
        group_id: str,
        source_document_id: str,
        result: RetrievalResult,
    ) -> tuple[
        RetrievalCorpus,
        RetrievalCorpusGeneration,
        RetrievalCorpusIngestion,
        RetrievalBaselineIndexPublication,
        RetrievalBaselineIndexBuild,
    ]:
        corpus_statement = select(RetrievalCorpus).where(
            RetrievalCorpus.scope_key == _scope_key(group_id)
        )
        if session.get_bind().dialect.name == "postgresql":
            corpus_statement = corpus_statement.with_for_update()
        corpus = session.scalar(corpus_statement)
        if corpus is None:
            raise BaselineEvidencePersistenceError(
                "active_corpus_absent", "group has no active trusted corpus"
            )
        if corpus.source_document_id != source_document_id:
            raise BaselineEvidencePersistenceError(
                "corpus_source_mismatch",
                "active corpus is not authoritative for the source document",
            )
        if corpus.active_generation_id is None:
            raise BaselineEvidencePersistenceError(
                "active_corpus_absent", "group has no active trusted generation"
            )
        generation_statement = select(RetrievalCorpusGeneration).where(
            RetrievalCorpusGeneration.generation_id == corpus.active_generation_id
        )
        ingestion_statement = select(RetrievalCorpusIngestion).where(
            RetrievalCorpusIngestion.generation_id == corpus.active_generation_id
        )
        if session.get_bind().dialect.name == "postgresql":
            generation_statement = generation_statement.with_for_update()
            ingestion_statement = ingestion_statement.with_for_update()
        generation = session.scalar(generation_statement)
        ingestion = session.scalar(ingestion_statement)
        if (
            generation is None
            or generation.status != CorpusGenerationStatus.ACTIVE.value
            or ingestion is None
            or ingestion.status != CorpusIngestionStatus.ACTIVE.value
        ):
            raise BaselineEvidencePersistenceError(
                "active_corpus_incomplete", "active trusted corpus is incomplete"
            )
        publication_statement = select(RetrievalBaselineIndexPublication).where(
            RetrievalBaselineIndexPublication.corpus_id == corpus.corpus_id
        )
        if session.get_bind().dialect.name == "postgresql":
            publication_statement = publication_statement.with_for_update()
        publication = session.scalar(publication_statement)
        if publication is None or publication.index_id is None:
            raise BaselineEvidencePersistenceError(
                "compatible_publication_absent",
                "active corpus has no compatible index publication",
            )
        build_statement = select(RetrievalBaselineIndexBuild).where(
            RetrievalBaselineIndexBuild.index_id == publication.index_id
        )
        if session.get_bind().dialect.name == "postgresql":
            build_statement = build_statement.with_for_update()
        build = session.scalar(build_statement)
        if (
            build is None
            or build.status != BaselineIndexBuildStatus.COMPATIBLE.value
            or build.generation_id != generation.generation_id
        ):
            raise BaselineEvidencePersistenceError(
                "publication_stale", "published baseline index is not current"
            )
        index_state_statement = select(RetrievalIndexState).where(
            RetrievalIndexState.generation_id == generation.generation_id
        )
        if session.get_bind().dialect.name == "postgresql":
            index_state_statement = index_state_statement.with_for_update()
        index_state = session.scalar(index_state_statement)
        if (
            index_state is None
            or index_state.status != IndexStateStatus.COMPATIBLE.value
        ):
            raise BaselineEvidencePersistenceError(
                "publication_incompatible",
                "published baseline index state is incompatible",
            )
        validation_error = BaselineIndexLifecycle.validation_error(
            session, build.index_id
        )
        if validation_error is not None:
            raise BaselineEvidencePersistenceError(
                "publication_incompatible",
                "published baseline index failed frozen compatibility validation",
            )
        return corpus, generation, ingestion, publication, build

    @staticmethod
    def _validate_result_header(
        result: RetrievalResult,
        corpus: RetrievalCorpus,
        generation: RetrievalCorpusGeneration,
        build: RetrievalBaselineIndexBuild,
    ) -> None:
        if (
            result.schema_version != RESULT_SCHEMA_VERSION
            or result.status is not RetrievalStatus.OK
            or result.engine != "baseline_v1"
            or result.engine_version != PERSISTENT_BASELINE_ENGINE_VERSION
            or result.error is not None
            or result.fallback_engine is not None
        ):
            raise BaselineEvidencePersistenceError(
                "unsupported_result",
                "only a successful persistent baseline_v1 result is accepted",
            )
        if not 1 <= len(result.evidence) <= MAX_EVIDENCE_ITEMS:
            raise BaselineEvidencePersistenceError(
                "malformed_result",
                "result must contain one to four ordered evidence items",
            )
        if result.candidate_count != len(result.candidates):
            raise BaselineEvidencePersistenceError(
                "result_counter_mismatch",
                "candidate count does not match result candidates",
            )
        if not result.request_id or len(result.request_id) > 128:
            raise BaselineEvidencePersistenceError(
                "malformed_result", "request identifier is invalid"
            )
        publication_fingerprint = published_index_fingerprint(build)
        exact = (
            (result.corpus_scope_key, corpus.scope_key),
            (result.corpus_id, corpus.corpus_id),
            (result.corpus_version, generation.generation_version),
            (result.corpus_manifest_hash, generation.manifest_hash),
            (result.index_id, build.index_id),
            (result.index_version, build.index_version),
            (result.index_schema_version, build.index_schema_version),
            (result.index_fingerprint, publication_fingerprint),
            (result.config_fingerprint, build.engine_config_fingerprint),
            (result.embedding_provider, build.embedding_provider),
            (result.embedding_model, build.embedding_model),
            (result.embedding_revision, build.embedding_revision),
            (result.embedding_dimension, build.embedding_dimension),
            (result.embedding_fingerprint, build.embedding_fingerprint),
        )
        if any(actual != expected for actual, expected in exact):
            raise BaselineEvidencePersistenceError(
                "result_publication_mismatch",
                "retrieval result fingerprints do not match the current publication",
            )
        for value, label in (
            (result.config_fingerprint, "config fingerprint"),
            (result.corpus_manifest_hash, "corpus manifest hash"),
            (result.index_fingerprint, "index fingerprint"),
            (result.embedding_fingerprint, "embedding fingerprint"),
        ):
            _sha256(value, label)
        query = result.query_provenance
        if (
            query is None
            or query.origin is not RetrievalQueryOrigin.EXPLICIT
            or query.length <= 0
            or query.sha256 is None
        ):
            raise BaselineEvidencePersistenceError(
                "query_provenance_invalid",
                "successful baseline evidence requires explicit trace-safe query provenance",
            )
        _sha256(query.sha256, "query hash")
        if (
            result.evidence_characters <= 0
            or result.evidence_characters > MAX_EVIDENCE_CHARACTERS
        ):
            raise BaselineEvidencePersistenceError(
                "malformed_result", "evidence character count exceeds the frozen budget"
            )
        fused_ranks = [item.fused_rank for item in result.evidence]
        if fused_ranks != sorted(fused_ranks) or len(set(fused_ranks)) != len(
            fused_ranks
        ):
            raise BaselineEvidencePersistenceError(
                "evidence_order_invalid",
                "evidence order is not explicit fused-rank order",
            )

    @staticmethod
    def _validate_candidates(
        result: RetrievalResult,
        *,
        documents: Sequence[RetrievalBaselineIndexDocument],
        sources_by_id: Mapping[str, RetrievalCorpusFile],
    ) -> dict[str, RetrievalCandidate]:
        if len(result.candidates) != len(documents):
            raise BaselineEvidencePersistenceError(
                "candidate_coverage_mismatch",
                "result does not cover the complete published index",
            )
        documents_by_id = {row.index_document_id: row for row in documents}
        candidates_by_id: dict[str, RetrievalCandidate] = {}
        for candidate in result.candidates:
            document_id = candidate.document_id
            if not document_id or document_id in candidates_by_id:
                raise BaselineEvidencePersistenceError(
                    "candidate_identity_invalid",
                    "candidate index document identity is absent or duplicated",
                )
            document = documents_by_id.get(document_id)
            source = (
                sources_by_id.get(document.corpus_file_id)
                if document is not None
                else None
            )
            if document is None or source is None or source.content is None:
                raise BaselineEvidencePersistenceError(
                    "candidate_source_absent",
                    "candidate does not map to immutable published content",
                )
            raw = source.content.encode("utf-8")
            ranking_text = baseline_ranking_document(
                source.repository_name, source.relative_path, source.content
            )
            if (
                source.file_state != CorpusFileState.SUPPORTED.value
                or len(raw) != source.byte_size
                or _digest_bytes(raw) != source.content_hash
                or document.repository_id != source.repository_id
                or document.repository_name != source.repository_name
                or document.relative_path != source.relative_path
                or document.source_content_hash != source.content_hash
                or document.ranking_text != ranking_text
                or document.indexed_document_hash != _digest_text(ranking_text)
                or candidate.repository != source.repository_name
                or candidate.relative_path != source.relative_path
                or candidate.content != source.content[:RANKING_CONTENT_CHARACTERS]
                or candidate.content_hash != source.content_hash
                or candidate.byte_size != source.byte_size
            ):
                raise BaselineEvidencePersistenceError(
                    "candidate_content_mismatch",
                    "candidate differs from its trusted indexed corpus file",
                )
            _normalized_path(candidate.relative_path)
            _positive_rank(candidate.bm25_rank, "BM25 rank")
            _positive_rank(candidate.dense_rank, "dense rank")
            _positive_rank(candidate.fused_rank, "fused rank")
            _finite_score(candidate.bm25_score, "BM25 score", nonnegative=True)
            _finite_score(candidate.dense_score, "dense score")
            _finite_score(candidate.rrf_score, "RRF score", nonnegative=True)
            candidates_by_id[document_id] = candidate

        count = len(documents)
        expected_ranks = list(range(1, count + 1))
        if (
            sorted(item.bm25_rank for item in result.candidates) != expected_ranks
            or sorted(item.dense_rank for item in result.candidates) != expected_ranks
            or [item.fused_rank for item in result.candidates] != expected_ranks
        ):
            raise BaselineEvidencePersistenceError(
                "candidate_rank_invalid",
                "candidate ranks are not complete deterministic permutations",
            )
        tie_key = lambda item: (item.path, item.document_id or "")
        if list(result.candidates) != sorted(
            result.candidates, key=lambda item: (-item.rrf_score, *tie_key(item))
        ):
            raise BaselineEvidencePersistenceError(
                "candidate_order_invalid",
                "candidate fused order violates deterministic tie-breaking",
            )
        if list(result.candidates) != sorted(
            result.candidates, key=lambda item: item.fused_rank
        ):
            raise BaselineEvidencePersistenceError(
                "candidate_order_invalid", "candidate tuple does not follow fused rank"
            )
        bm25_order = sorted(
            result.candidates,
            key=lambda item: (-item.bm25_score, *tie_key(item)),
        )
        dense_order = sorted(
            result.candidates,
            key=lambda item: (-item.dense_score, *tie_key(item)),
        )
        if any(item.bm25_rank != rank for rank, item in enumerate(bm25_order, start=1)):
            raise BaselineEvidencePersistenceError(
                "candidate_rank_invalid",
                "BM25 scores and ranks violate deterministic ordering",
            )
        if any(
            item.dense_rank != rank for rank, item in enumerate(dense_order, start=1)
        ):
            raise BaselineEvidencePersistenceError(
                "candidate_rank_invalid",
                "dense scores and ranks violate deterministic ordering",
            )
        for candidate in result.candidates:
            expected_rrf = reciprocal_rank_fusion(
                [candidate.bm25_rank], [candidate.dense_rank]
            )[0]
            if candidate.rrf_score != expected_rrf:
                raise BaselineEvidencePersistenceError(
                    "candidate_rrf_mismatch",
                    "candidate RRF score violates the frozen contract",
                )
        return candidates_by_id

    @staticmethod
    def _scoped_optional_document(
        session: Session,
        *,
        group_id: str,
        document_id: str | None,
    ) -> str | None:
        if document_id is None:
            return None
        return (
            document_id
            if session.execute(
                text(
                    "SELECT document_id FROM document_to_group "
                    "WHERE document_id = :document_id AND group_id = :group_id"
                ),
                {"document_id": document_id, "group_id": group_id},
            ).one_or_none()
            is not None
            else None
        )

    def _write(
        self,
        session: Session,
        intent: _ValidatedIntent,
    ) -> BaselineEvidencePersistenceReceipt:
        run_values = dict(intent.run_values)
        run_id = str(run_values["run_id"])
        session.execute(baseline_retrieval_run.insert().values(**run_values))
        session.flush()
        self._after_stage(PersistenceWriteStage.RUN)

        artifact_ids: dict[str, str] = {}
        for item in intent.selected:
            existing = (
                session.execute(
                    select(baseline_evidence_artifact).where(
                        baseline_evidence_artifact.c.group_id
                        == intent.command.group_id,
                        baseline_evidence_artifact.c.artifact_key == item.artifact_key,
                    )
                )
                .mappings()
                .one_or_none()
            )
            if existing is not None:
                if not _row_matches(
                    existing, item.artifact_values, _ARTIFACT_INTENT_COLUMNS
                ):
                    raise BaselineEvidencePersistenceError(
                        "artifact_conflict",
                        "group artifact key identifies different immutable evidence",
                    )
                artifact_ids[item.artifact_key] = str(existing["artifact_id"])
                continue
            artifact_id = str(uuid4())
            session.execute(
                baseline_evidence_artifact.insert().values(
                    artifact_id=artifact_id, **item.artifact_values
                )
            )
            artifact_ids[item.artifact_key] = artifact_id
        session.flush()
        self._after_stage(PersistenceWriteStage.ARTIFACTS)

        selected_ids: list[str] = []
        for item in intent.selected:
            selected_id = str(uuid4())
            session.execute(
                baseline_selected_evidence.insert().values(
                    selected_evidence_id=selected_id,
                    run_id=run_id,
                    artifact_id=artifact_ids[item.artifact_key],
                    **item.selected_values,
                )
            )
            selected_ids.append(selected_id)
        session.flush()
        self._after_stage(PersistenceWriteStage.SELECTED_EVIDENCE)

        reference_ids: list[str] = []
        for selected_id in selected_ids:
            reference_id = str(uuid4())
            session.execute(
                text(
                    "INSERT INTO reference "
                    "(reference_id, source_chunk_id, reference_chunk_id, "
                    "reference_document_id, reference_note_id, reference_type, "
                    "baseline_selected_evidence_id) "
                    "VALUES (:reference_id, :source_chunk_id, NULL, NULL, NULL, "
                    ":reference_type, :selected_id)"
                ),
                {
                    "reference_id": reference_id,
                    "source_chunk_id": intent.run_values["source_chunk_id"],
                    "reference_type": REFERENCE_TYPE,
                    "selected_id": selected_id,
                },
            )
            reference_ids.append(reference_id)
        session.flush()
        self._after_stage(PersistenceWriteStage.REFERENCES)
        if intent.control_job is not None:
            updated = session.execute(
                text(
                    "UPDATE baseline_control_run_job SET "
                    "persisted_run_id = :run_id, state = 'references_persisted', "
                    "evidence_count = :count, reference_count = :count, "
                    "retrieval_result_fingerprint = :result_fingerprint, "
                    "lease_token = NULL, lease_expires_at = NULL, "
                    "reason_code = NULL, failure_stage = NULL, "
                    "updated_at = CURRENT_TIMESTAMP "
                    "WHERE job_id = :job_id AND group_id = :group_id "
                    "AND state = 'running' AND persisted_run_id IS NULL "
                    "AND retrieval_result_fingerprint IS NULL "
                    "AND lease_token = :lease_token "
                    "AND lease_expires_at > CURRENT_TIMESTAMP"
                ),
                {
                    "run_id": run_id,
                    "count": len(selected_ids),
                    "result_fingerprint": retrieval_result_fingerprint(
                        intent.command.retrieval_result
                    ),
                    "job_id": intent.control_job["job_id"],
                    "group_id": intent.command.group_id,
                    "lease_token": intent.control_job["lease_token"],
                },
            )
            if updated.rowcount != 1:
                raise BaselineEvidencePersistenceError(
                    "control_job_attachment_conflict",
                    "control job changed before retrieval evidence attachment",
                )
            session.flush()
            self._after_stage(PersistenceWriteStage.CONTROL_RELATIONSHIP)
            erased = session.execute(
                baseline_run_payload.delete().where(
                    baseline_run_payload.c.job_id == intent.control_job["job_id"],
                    baseline_run_payload.c.group_id == intent.command.group_id,
                )
            )
            if erased.rowcount != 1:
                raise BaselineEvidencePersistenceError(
                    "control_job_payload_absent",
                    "control job protected payload is unavailable for atomic erasure",
                )
            session.flush()
            self._after_stage(PersistenceWriteStage.PROTECTED_PAYLOAD)
        return BaselineEvidencePersistenceReceipt(
            run_id=run_id,
            group_id=intent.command.group_id,
            idempotency_key=intent.command.idempotency_key,
            selected_evidence_ids=tuple(selected_ids),
            reference_ids=tuple(reference_ids),
            replayed=False,
        )

    def _replay_receipt(
        self,
        session: Session,
        intent: _ValidatedIntent,
        run: Mapping[str, object],
    ) -> BaselineEvidencePersistenceReceipt:
        if not _row_matches(run, intent.run_values, _RUN_INTENT_COLUMNS):
            raise BaselineEvidencePersistenceError(
                "idempotency_conflict",
                "idempotency key is already bound to a different intent",
            )
        run_id = str(run["run_id"])
        if intent.control_job is not None and (
            intent.control_job.get("persisted_run_id") != run_id
            or intent.control_job.get("state")
            not in {"references_persisted", "feedback_persisted"}
        ):
            raise BaselineEvidencePersistenceError(
                "control_job_attachment_conflict",
                "control job is attached to a different retrieval run",
            )
        rows = (
            session.execute(
                select(
                    baseline_selected_evidence,
                    baseline_evidence_artifact,
                )
                .join(
                    baseline_evidence_artifact,
                    baseline_evidence_artifact.c.artifact_id
                    == baseline_selected_evidence.c.artifact_id,
                )
                .where(baseline_selected_evidence.c.run_id == run_id)
                .order_by(baseline_selected_evidence.c.ordinal)
            )
            .mappings()
            .all()
        )
        if len(rows) != len(intent.selected):
            raise BaselineEvidencePersistenceError(
                "idempotency_conflict",
                "persisted idempotent run has different evidence cardinality",
            )
        selected_ids: list[str] = []
        for row, expected in zip(rows, intent.selected):
            if not _row_matches(
                row, expected.artifact_values, _ARTIFACT_INTENT_COLUMNS
            ) or not _row_matches(
                row, expected.selected_values, _SELECTED_INTENT_COLUMNS
            ):
                raise BaselineEvidencePersistenceError(
                    "idempotency_conflict",
                    "persisted idempotent evidence differs from the current intent",
                )
            selected_ids.append(str(row["selected_evidence_id"]))
        references = (
            session.execute(
                text(
                    "SELECT r.reference_id, r.source_chunk_id, "
                    "r.reference_chunk_id, r.reference_document_id, r.reference_note_id, "
                    "r.reference_type, r.baseline_selected_evidence_id "
                    "FROM reference r JOIN baseline_selected_evidence s "
                    "ON s.selected_evidence_id = r.baseline_selected_evidence_id "
                    "WHERE s.run_id = :run_id ORDER BY s.ordinal"
                ),
                {"run_id": run_id},
            )
            .mappings()
            .all()
        )
        if (
            len(references) != len(selected_ids)
            or [row["baseline_selected_evidence_id"] for row in references]
            != selected_ids
            or any(
                row["source_chunk_id"] != intent.run_values["source_chunk_id"]
                or row["reference_chunk_id"] is not None
                or row["reference_document_id"] is not None
                or row["reference_note_id"] is not None
                or row["reference_type"] != REFERENCE_TYPE
                for row in references
            )
        ):
            raise BaselineEvidencePersistenceError(
                "idempotency_conflict",
                "persisted idempotent References differ from the current intent",
            )
        return BaselineEvidencePersistenceReceipt(
            run_id=run_id,
            group_id=intent.command.group_id,
            idempotency_key=intent.command.idempotency_key,
            selected_evidence_ids=tuple(selected_ids),
            reference_ids=tuple(str(row["reference_id"]) for row in references),
            replayed=True,
        )

    def _after_stage(self, stage: PersistenceWriteStage) -> None:
        if self._stage_hook is not None:
            self._stage_hook(stage)


__all__ = [
    "AUTHORIZATION_SCOPE_VERSION",
    "REFERENCE_TYPE",
    "BaselineEvidencePersistenceCommand",
    "BaselineEvidencePersistenceError",
    "BaselineEvidencePersistenceReceipt",
    "BaselineEvidencePersistenceService",
    "ControlDocumentSource",
    "LegacyChunkSource",
    "PersistenceWriteStage",
    "render_baseline_evidence",
    "retrieval_result_fingerprint",
]
