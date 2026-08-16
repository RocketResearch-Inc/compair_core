"""Ordered, leased generation over persisted ``baseline_v1`` evidence.

The service deliberately has no dependency on legacy ``Chunk`` references or
the legacy generation helper.  A provider receives the immutable renderer
outputs exactly as stored by the baseline evidence bridge.  Database leases
and Feedback uniqueness make database effects retry-safe; an external model
call can still be repeated after a worker crash unless that provider honors
the stable idempotency key supplied here.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol
from uuid import uuid4

import requests
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from ...baseline_evidence_schema import RENDERER_VERSION
from .corpus import (
    BaselineIndexBuildStatus,
    CorpusGenerationStatus,
    IndexStateStatus,
    RetrievalBaselineIndexBuild,
    RetrievalBaselineIndexPublication,
    RetrievalCorpus,
    RetrievalCorpusGeneration,
    RetrievalIndexState,
)
from .evidence_persistence import render_baseline_evidence
from .notification_outbox import (
    baseline_notifications_enabled,
    schedule_baseline_notification,
)
from .persistent import published_index_fingerprint

GENERATION_CONTRACT_VERSION = "baseline-generation-input.v1"
GENERATION_STATE_VERSION = "baseline-generation-state.v1"
GENERATION_FINDING_SEPARATOR = "<<<FINDING>>>"
DEFAULT_LEASE_SECONDS = 300
MAX_GENERATION_OUTPUT_CHARACTERS = 100_000


class BaselineGenerationState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    RETRYABLE_FAILED = "retryable_failed"
    TERMINAL_FAILED = "terminal_failed"
    BLOCKED = "blocked"


class GenerationWriteStage(str, Enum):
    FEEDBACK = "feedback"
    STATE = "state"
    OUTBOX = "outbox"


@dataclass(frozen=True, slots=True)
class BaselineGenerationEvidence:
    ordinal: int
    fused_rank: int
    bm25_score: float
    bm25_rank: int
    dense_score: float
    dense_rank: int
    rrf_score: float
    selected_evidence_id: str
    artifact_id: str
    repository_id: str
    repository_name: str
    relative_path: str
    renderer_version: str
    renderer_output: str
    renderer_output_hash: str
    selected_content_hash: str
    whole_file_content_hash: str
    corpus_generation_id: str
    index_id: str
    index_document_id: str
    index_fingerprint: str


@dataclass(frozen=True, slots=True)
class BaselineGenerationInput:
    run_id: str
    group_id: str
    source_chunk_id: str
    source_document_id: str
    source_text: str
    corpus_generation_id: str
    corpus_manifest_hash: str
    index_id: str
    index_fingerprint: str
    query_sha256: str
    evidence: tuple[BaselineGenerationEvidence, ...]
    input_fingerprint: str
    contract_version: str = GENERATION_CONTRACT_VERSION


@dataclass(frozen=True, slots=True)
class BaselineGenerationCommand:
    run_id: str
    group_id: str
    caller_user_id: str


@dataclass(frozen=True, slots=True)
class BaselineGenerationReceipt:
    run_id: str
    group_id: str
    state: BaselineGenerationState
    attempt_count: int
    input_fingerprint: str | None
    output_fingerprint: str | None
    feedback_ids: tuple[str, ...]
    error_code: str | None = None
    replayed: bool = False


class BaselineGenerationProvider(Protocol):
    provider: str
    model: str
    version: str
    supports_idempotency: bool

    def generate(
        self,
        generation_input: BaselineGenerationInput,
        *,
        idempotency_key: str,
    ) -> str: ...


class BaselineGenerationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class BaselineGenerationBusyError(BaselineGenerationError):
    pass


class BaselineGenerationProviderError(BaselineGenerationError):
    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        super().__init__(code, message)
        self.retryable = retryable


StageHook = Callable[[GenerationWriteStage], None]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_identifier(value: object, label: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(ord(character) < 32 for character in value)
    ):
        raise BaselineGenerationError("invalid_command", f"{label} is invalid")
    return value


def _safe_provider_identity(value: object, label: str, maximum: int) -> str:
    return _safe_identifier(value, label, maximum)


def _safe_error_code(value: object, fallback: str) -> str:
    candidate = str(value or "").strip().lower()
    candidate = re.sub(r"[^a-z0-9_.-]+", "_", candidate).strip("_.-")
    return (candidate or fallback)[:128]


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fingerprint_parts(parts: Sequence[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    digest.update(GENERATION_CONTRACT_VERSION.encode("ascii"))
    for label, value in parts:
        raw_label = label.encode("utf-8")
        raw_value = value.encode("utf-8")
        digest.update(len(raw_label).to_bytes(4, "big"))
        digest.update(raw_label)
        digest.update(len(raw_value).to_bytes(8, "big"))
        digest.update(raw_value)
    return digest.hexdigest()


def _begin_write(session: Session) -> None:
    if session.get_bind().dialect.name == "sqlite":
        session.connection().exec_driver_sql("BEGIN IMMEDIATE")
    else:
        session.begin()


def _run_statement(run_id: str, group_id: str, *, lock: bool, dialect: str):
    sql = (
        "SELECT * FROM baseline_retrieval_run "
        "WHERE run_id = :run_id AND group_id = :group_id"
    )
    if lock and dialect == "postgresql":
        sql += " FOR UPDATE"
    return text(sql), {"run_id": run_id, "group_id": group_id}


class BaselineGenerationInputAdapter:
    """Load immutable evidence in its single authoritative ordinal order."""

    def load(
        self,
        session: Session,
        *,
        run: dict[str, object],
        source_text: str,
    ) -> BaselineGenerationInput:
        rows = (
            session.execute(
                text(
                    "SELECT s.ordinal, s.fused_rank, s.bm25_score, s.bm25_rank, "
                    "s.dense_score, s.dense_rank, s.rrf_score, "
                    "s.selected_evidence_id, s.artifact_id, "
                    "s.renderer_version, s.renderer_output, s.renderer_output_hash, "
                    "s.renderer_output_character_count, s.selected_content, "
                    "s.selected_character_count, s.selected_content_hash, "
                    "a.repository_id, a.repository_name, a.relative_path, "
                    "a.artifact_key, a.complete_content, a.byte_size, a.character_count, "
                    "a.whole_file_content_hash, a.corpus_generation_id, a.index_id, "
                    "a.index_document_id, a.index_fingerprint "
                    "FROM baseline_selected_evidence s "
                    "JOIN baseline_evidence_artifact a ON a.artifact_id = s.artifact_id "
                    "AND a.group_id = s.group_id "
                    "WHERE s.run_id = :run_id AND s.group_id = :group_id "
                    "ORDER BY s.ordinal ASC"
                ),
                {"run_id": run["run_id"], "group_id": run["group_id"]},
            )
            .mappings()
            .all()
        )
        selected_count = int(run["selected_count"])
        if not 1 <= selected_count <= 4 or len(rows) != selected_count:
            raise BaselineGenerationError(
                "generation_evidence_invalid",
                "persisted baseline evidence cardinality is invalid",
            )
        evidence: list[BaselineGenerationEvidence] = []
        parts: list[tuple[str, str]] = [
            ("run_id", str(run["run_id"])),
            ("group_id", str(run["group_id"])),
            ("source_chunk_id", str(run["source_chunk_id"])),
            ("source_document_id", str(run["source_document_id"])),
            ("source_text", source_text),
            ("corpus_generation_id", str(run["corpus_generation_id"])),
            ("corpus_manifest_hash", str(run["corpus_manifest_hash"])),
            ("index_id", str(run["index_id"])),
            ("index_fingerprint", str(run["index_fingerprint"])),
            ("query_sha256", str(run["query_sha256"])),
        ]
        for expected_ordinal, row in enumerate(rows, start=1):
            renderer_output = str(row["renderer_output"])
            selected_content = str(row["selected_content"])
            complete_content = str(row["complete_content"])
            ranks = (
                int(row["fused_rank"]),
                int(row["bm25_rank"]),
                int(row["dense_rank"]),
            )
            scores = (
                float(row["bm25_score"]),
                float(row["dense_score"]),
                float(row["rrf_score"]),
            )
            expected_renderer = render_baseline_evidence(
                str(row["repository_name"]),
                str(row["relative_path"]),
                selected_content,
            )
            if (
                int(row["ordinal"]) != expected_ordinal
                or any(rank <= 0 for rank in ranks)
                or not all(math.isfinite(score) for score in scores)
                or row["renderer_version"] != RENDERER_VERSION
                or len(renderer_output) != int(row["renderer_output_character_count"])
                or _sha256_text(renderer_output) != row["renderer_output_hash"]
                or renderer_output != expected_renderer
                or len(selected_content) != int(row["selected_character_count"])
                or _sha256_text(selected_content) != row["selected_content_hash"]
                or len(complete_content) != int(row["character_count"])
                or len(complete_content.encode("utf-8")) != int(row["byte_size"])
                or _sha256_text(complete_content) != row["whole_file_content_hash"]
                or len(str(row["artifact_key"])) != 64
                or row["corpus_generation_id"] != run["corpus_generation_id"]
                or row["index_id"] != run["index_id"]
                or row["index_fingerprint"] != run["index_fingerprint"]
            ):
                raise BaselineGenerationError(
                    "generation_evidence_invalid",
                    "persisted renderer output failed its frozen contract",
                )
            item = BaselineGenerationEvidence(
                ordinal=expected_ordinal,
                fused_rank=int(row["fused_rank"]),
                bm25_score=float(row["bm25_score"]),
                bm25_rank=int(row["bm25_rank"]),
                dense_score=float(row["dense_score"]),
                dense_rank=int(row["dense_rank"]),
                rrf_score=float(row["rrf_score"]),
                selected_evidence_id=str(row["selected_evidence_id"]),
                artifact_id=str(row["artifact_id"]),
                repository_id=str(row["repository_id"]),
                repository_name=str(row["repository_name"]),
                relative_path=str(row["relative_path"]),
                renderer_version=str(row["renderer_version"]),
                renderer_output=renderer_output,
                renderer_output_hash=str(row["renderer_output_hash"]),
                selected_content_hash=str(row["selected_content_hash"]),
                whole_file_content_hash=str(row["whole_file_content_hash"]),
                corpus_generation_id=str(row["corpus_generation_id"]),
                index_id=str(row["index_id"]),
                index_document_id=str(row["index_document_id"]),
                index_fingerprint=str(row["index_fingerprint"]),
            )
            evidence.append(item)
            prefix = f"evidence[{expected_ordinal}]"
            parts.extend(
                (
                    (f"{prefix}.selected_evidence_id", item.selected_evidence_id),
                    (f"{prefix}.fused_rank", str(item.fused_rank)),
                    (f"{prefix}.bm25_score", item.bm25_score.hex()),
                    (f"{prefix}.bm25_rank", str(item.bm25_rank)),
                    (f"{prefix}.dense_score", item.dense_score.hex()),
                    (f"{prefix}.dense_rank", str(item.dense_rank)),
                    (f"{prefix}.rrf_score", item.rrf_score.hex()),
                    (f"{prefix}.repository_id", item.repository_id),
                    (f"{prefix}.repository_name", item.repository_name),
                    (f"{prefix}.relative_path", item.relative_path),
                    (f"{prefix}.renderer_version", item.renderer_version),
                    (f"{prefix}.renderer_output", item.renderer_output),
                    (f"{prefix}.renderer_output_hash", item.renderer_output_hash),
                    (f"{prefix}.selected_content_hash", item.selected_content_hash),
                    (f"{prefix}.whole_file_content_hash", item.whole_file_content_hash),
                    (f"{prefix}.index_document_id", item.index_document_id),
                )
            )
        fingerprint = _fingerprint_parts(parts)
        return BaselineGenerationInput(
            run_id=str(run["run_id"]),
            group_id=str(run["group_id"]),
            source_chunk_id=str(run["source_chunk_id"]),
            source_document_id=str(run["source_document_id"]),
            source_text=source_text,
            corpus_generation_id=str(run["corpus_generation_id"]),
            corpus_manifest_hash=str(run["corpus_manifest_hash"]),
            index_id=str(run["index_id"]),
            index_fingerprint=str(run["index_fingerprint"]),
            query_sha256=str(run["query_sha256"]),
            evidence=tuple(evidence),
            input_fingerprint=fingerprint,
        )


class ReviewerBaselineGenerationProvider:
    """Identity-preserving adapter over Core's configured generation client.

    It intentionally does not call ``get_feedback`` or ``_local_references``.
    The structured local/HTTP payload contains each stored renderer string as
    one unchanged list element; the OpenAI prompt embeds those strings without
    clipping or normalization.
    """

    supports_idempotency = False

    def __init__(self, reviewer: Any) -> None:
        self._reviewer = reviewer
        self.provider = _safe_provider_identity(
            getattr(reviewer, "provider", ""), "generation provider", 128
        )
        if self.provider == "openai":
            model = getattr(reviewer, "openai_model", "")
        else:
            model = os.getenv("COMPAIR_BASELINE_GENERATION_MODEL") or getattr(
                reviewer, "model", ""
            )
        self.model = _safe_provider_identity(model, "generation model", 256)
        self.version = _safe_provider_identity(
            os.getenv("COMPAIR_BASELINE_GENERATION_MODEL_VERSION") or self.model,
            "generation model version",
            256,
        )

    def generate(
        self,
        generation_input: BaselineGenerationInput,
        *,
        idempotency_key: str,
    ) -> str:
        evidence = [item.renderer_output for item in generation_input.evidence]
        if self.provider in {"local", "http"}:
            endpoint = (
                getattr(self._reviewer, "endpoint", None)
                if self.provider == "local"
                else getattr(self._reviewer, "custom_endpoint", None)
            )
            if not endpoint:
                raise BaselineGenerationProviderError(
                    "provider_unavailable",
                    "baseline generation endpoint is unavailable",
                    retryable=True,
                )
            payload = {
                "contract_version": GENERATION_CONTRACT_VERSION,
                "document": generation_input.source_text,
                "references": evidence,
                "finding_separator": GENERATION_FINDING_SEPARATOR,
                "idempotency_key": idempotency_key,
            }
            try:
                response = requests.post(
                    endpoint,
                    json=payload,
                    timeout=float(os.getenv("COMPAIR_BASELINE_GENERATION_TIMEOUT", "30")),
                )
                response.raise_for_status()
                body = response.json()
                output = body.get("feedback") or body.get("text")
            except Exception as exc:  # no provider details or payload in the error
                raise BaselineGenerationProviderError(
                    "provider_unavailable",
                    "baseline generation provider request failed",
                    retryable=True,
                ) from exc
            if not isinstance(output, str):
                raise BaselineGenerationProviderError(
                    "provider_malformed_output",
                    "baseline generation provider returned malformed output",
                    retryable=False,
                )
            return output

        if self.provider == "openai":
            client = getattr(self._reviewer, "_openai_client", None)
            if client is None or not hasattr(client, "responses"):
                raise BaselineGenerationProviderError(
                    "provider_unavailable",
                    "configured baseline generation client is unavailable",
                    retryable=True,
                )
            prompt = (
                "Changed source:\n"
                + generation_input.source_text
                + "\n\nOrdered baseline evidence follows. Preserve its order and return "
                + f"findings separated by {GENERATION_FINDING_SEPARATOR}.\n\n"
                + "\n\n".join(evidence)
            )
            try:
                response = client.responses.create(
                    model=self.model,
                    input=prompt,
                )
                output = getattr(response, "output_text", None)
            except Exception as exc:
                raise BaselineGenerationProviderError(
                    "provider_unavailable",
                    "baseline generation provider request failed",
                    retryable=True,
                ) from exc
            if not isinstance(output, str):
                raise BaselineGenerationProviderError(
                    "provider_malformed_output",
                    "baseline generation provider returned malformed output",
                    retryable=False,
                )
            return output

        raise BaselineGenerationProviderError(
            "provider_unsupported",
            "configured provider has no identity-preserving baseline contract",
            retryable=False,
        )


class BaselineGenerationService:
    """Lease, call, and atomically persist baseline findings."""

    def __init__(
        self,
        session_factory: Any,
        *,
        lease_seconds: int = DEFAULT_LEASE_SECONDS,
        stage_hook: StageHook | None = None,
        clock: Callable[[], datetime] = _utcnow,
        notifications_enabled: bool | None = None,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self._session_factory = session_factory
        self._lease_seconds = lease_seconds
        self._stage_hook = stage_hook
        self._clock = clock
        self._notifications_enabled = (
            baseline_notifications_enabled()
            if notifications_enabled is None
            else notifications_enabled
        )
        self._input_adapter = BaselineGenerationInputAdapter()

    def generate(
        self,
        command: BaselineGenerationCommand,
        provider: BaselineGenerationProvider,
    ) -> BaselineGenerationReceipt:
        run_id = _safe_identifier(command.run_id, "run_id", 36)
        group_id = _safe_identifier(command.group_id, "group_id", 36)
        caller_user_id = _safe_identifier(
            command.caller_user_id, "caller_user_id", 36
        )
        provider_name = _safe_provider_identity(provider.provider, "provider", 128)
        model = _safe_provider_identity(provider.model, "model", 256)
        version = _safe_provider_identity(provider.version, "version", 256)
        command = BaselineGenerationCommand(run_id, group_id, caller_user_id)

        lease = self._acquire_lease(
            command,
            provider_name=provider_name,
            model=model,
            version=version,
        )
        if isinstance(lease, BaselineGenerationReceipt):
            return lease
        lease_token, generation_input, attempt_count = lease
        provider_key = _sha256_text(
            f"{GENERATION_STATE_VERSION}\x00{run_id}\x00{generation_input.input_fingerprint}"
        )
        try:
            output = provider.generate(
                generation_input,
                idempotency_key=provider_key,
            )
        except BaselineGenerationProviderError as exc:
            return self._record_failure(
                command,
                lease_token=lease_token,
                attempt_count=attempt_count,
                input_fingerprint=generation_input.input_fingerprint,
                code=exc.code,
                retryable=exc.retryable,
                error_class="provider",
            )
        except Exception:
            return self._record_failure(
                command,
                lease_token=lease_token,
                attempt_count=attempt_count,
                input_fingerprint=generation_input.input_fingerprint,
                code="provider_unavailable",
                retryable=True,
                error_class="provider",
            )

        try:
            findings, output_fingerprint = self._parse_output(
                output,
                maximum_findings=len(generation_input.evidence),
            )
        except BaselineGenerationError as exc:
            return self._record_failure(
                command,
                lease_token=lease_token,
                attempt_count=attempt_count,
                input_fingerprint=generation_input.input_fingerprint,
                code=exc.code,
                retryable=False,
                error_class="output",
            )
        try:
            return self._commit_feedback(
                command,
                lease_token=lease_token,
                expected_input=generation_input,
                findings=findings,
                output_fingerprint=output_fingerprint,
                provider_name=provider_name,
                model=model,
                version=version,
            )
        except BaselineGenerationBusyError:
            raise
        except Exception as exc:
            self._record_failure(
                command,
                lease_token=lease_token,
                attempt_count=attempt_count,
                input_fingerprint=generation_input.input_fingerprint,
                code="database_commit_failed",
                retryable=True,
                error_class="database",
            )
            if isinstance(exc, BaselineGenerationError):
                raise
            raise BaselineGenerationError(
                "database_commit_failed",
                "baseline Feedback transaction failed",
            ) from exc

    def _acquire_lease(
        self,
        command: BaselineGenerationCommand,
        *,
        provider_name: str,
        model: str,
        version: str,
    ) -> tuple[str, BaselineGenerationInput, int] | BaselineGenerationReceipt:
        with self._session_factory() as session:
            try:
                _begin_write(session)
                run = self._load_run(session, command, lock=True)
                state = BaselineGenerationState(str(run["generation_state"]))
                if state is BaselineGenerationState.SUCCEEDED:
                    receipt = self._receipt(session, run, replayed=True)
                    session.commit()
                    return receipt
                if state in {
                    BaselineGenerationState.TERMINAL_FAILED,
                    BaselineGenerationState.BLOCKED,
                }:
                    receipt = self._receipt(session, run, replayed=True)
                    session.commit()
                    return receipt
                now = self._clock()
                if state is BaselineGenerationState.RUNNING:
                    expiry = _as_utc(run["generation_lease_expires_at"])
                    if expiry is not None and expiry > now:
                        raise BaselineGenerationBusyError(
                            "generation_lease_active",
                            "baseline generation is already leased",
                        )
                validation = self._validate_authorization_and_provenance(
                    session, command, run
                )
                if validation[0] is not None:
                    self._block(session, command.run_id, validation[0])
                    updated = self._load_run(session, command, lock=False)
                    receipt = self._receipt(session, updated)
                    session.commit()
                    return receipt
                source_text = validation[1]
                assert source_text is not None
                try:
                    generation_input = self._input_adapter.load(
                        session,
                        run=run,
                        source_text=source_text,
                    )
                except BaselineGenerationError as exc:
                    self._block(session, command.run_id, exc.code)
                    updated = self._load_run(session, command, lock=False)
                    receipt = self._receipt(session, updated)
                    session.commit()
                    return receipt
                lease_token = uuid4().hex
                attempt_count = int(run["generation_attempt_count"]) + 1
                expires = now + timedelta(seconds=self._lease_seconds)
                session.execute(
                    text(
                        "UPDATE baseline_retrieval_run SET generation_state = 'running', "
                        "generation_lease_token = :lease_token, "
                        "generation_lease_expires_at = :expires, "
                        "generation_started_at = :started, "
                        "generation_attempt_count = :attempt_count, "
                        "generation_input_fingerprint = :input_fingerprint, "
                        "generation_provider = :provider, generation_model = :model, "
                        "generation_model_version = :version, "
                        "generation_output_fingerprint = NULL, "
                        "generation_error_code = NULL, "
                        "generation_error_fingerprint = NULL, "
                        "generation_completed_at = NULL, "
                        "generation_updated_at = :updated "
                        "WHERE run_id = :run_id"
                    ),
                    {
                        "lease_token": lease_token,
                        "expires": expires,
                        "started": now,
                        "attempt_count": attempt_count,
                        "input_fingerprint": generation_input.input_fingerprint,
                        "provider": provider_name,
                        "model": model,
                        "version": version,
                        "updated": now,
                        "run_id": command.run_id,
                    },
                )
                session.commit()
                return lease_token, generation_input, attempt_count
            except Exception:
                session.rollback()
                raise

    def _commit_feedback(
        self,
        command: BaselineGenerationCommand,
        *,
        lease_token: str,
        expected_input: BaselineGenerationInput,
        findings: tuple[str, ...],
        output_fingerprint: str,
        provider_name: str,
        model: str,
        version: str,
    ) -> BaselineGenerationReceipt:
        with self._session_factory() as session:
            try:
                _begin_write(session)
                run = self._load_run(session, command, lock=True)
                if run["generation_state"] != BaselineGenerationState.RUNNING.value:
                    if run["generation_state"] == BaselineGenerationState.SUCCEEDED.value:
                        receipt = self._receipt(session, run, replayed=True)
                        session.commit()
                        return receipt
                    raise BaselineGenerationBusyError(
                        "generation_lease_lost", "baseline generation lease is no longer active"
                    )
                if run["generation_lease_token"] != lease_token:
                    raise BaselineGenerationBusyError(
                        "generation_lease_lost", "baseline generation lease token changed"
                    )
                expiry = _as_utc(run["generation_lease_expires_at"])
                if expiry is None or expiry <= self._clock():
                    session.execute(
                        text(
                            "UPDATE baseline_retrieval_run SET "
                            "generation_state = 'retryable_failed', "
                            "generation_lease_token = NULL, "
                            "generation_lease_expires_at = NULL, "
                            "generation_error_code = 'generation_lease_expired', "
                            "generation_error_fingerprint = :fingerprint, "
                            "generation_updated_at = :updated WHERE run_id = :run_id"
                        ),
                        {
                            "fingerprint": _sha256_text("lease:generation_lease_expired"),
                            "updated": self._clock(),
                            "run_id": command.run_id,
                        },
                    )
                    updated = self._load_run(session, command, lock=False)
                    receipt = self._receipt(session, updated)
                    session.commit()
                    return receipt
                validation = self._validate_authorization_and_provenance(
                    session, command, run
                )
                if validation[0] is not None:
                    self._block(session, command.run_id, validation[0])
                    updated = self._load_run(session, command, lock=False)
                    receipt = self._receipt(session, updated)
                    session.commit()
                    return receipt
                source_text = validation[1]
                assert source_text is not None
                try:
                    current_input = self._input_adapter.load(
                        session, run=run, source_text=source_text
                    )
                except BaselineGenerationError as exc:
                    self._block(session, command.run_id, exc.code)
                    updated = self._load_run(session, command, lock=False)
                    receipt = self._receipt(session, updated)
                    session.commit()
                    return receipt
                if current_input.input_fingerprint != expected_input.input_fingerprint:
                    self._block(session, command.run_id, "generation_input_changed")
                    updated = self._load_run(session, command, lock=False)
                    receipt = self._receipt(session, updated)
                    session.commit()
                    return receipt
                existing = self._feedback_rows(session, command.run_id)
                if existing:
                    raise BaselineGenerationError(
                        "feedback_state_conflict",
                        "baseline Feedback exists before a succeeded transition",
                    )
                feedback_ids: list[str] = []
                for ordinal, finding in enumerate(findings, start=1):
                    feedback_id = str(uuid4())
                    session.execute(
                        text(
                            "INSERT INTO feedback "
                            "(feedback_id, source_chunk_id, feedback, model, "
                            "timestamp, is_hidden, "
                            "baseline_retrieval_run_id, baseline_finding_ordinal, "
                            "generation_provider, generation_model, "
                            "generation_model_version, generation_input_fingerprint, "
                            "generation_output_fingerprint) VALUES "
                            "(:feedback_id, :source_chunk_id, :feedback, :model, "
                            ":timestamp, false, "
                            ":run_id, :ordinal, :provider, :generation_model, :version, "
                            ":input_fingerprint, :output_fingerprint)"
                        ),
                        {
                            "feedback_id": feedback_id,
                            "source_chunk_id": current_input.source_chunk_id,
                            "feedback": finding,
                            "model": model,
                            "timestamp": self._clock(),
                            "run_id": command.run_id,
                            "ordinal": ordinal,
                            "provider": provider_name,
                            "generation_model": model,
                            "version": version,
                            "input_fingerprint": current_input.input_fingerprint,
                            "output_fingerprint": output_fingerprint,
                        },
                    )
                    feedback_ids.append(feedback_id)
                session.flush()
                self._after_stage(GenerationWriteStage.FEEDBACK)
                completed = self._clock()
                session.execute(
                    text(
                        "UPDATE baseline_retrieval_run SET generation_state = 'succeeded', "
                        "generation_lease_token = NULL, generation_lease_expires_at = NULL, "
                        "generation_output_fingerprint = :output_fingerprint, "
                        "generation_error_code = NULL, generation_error_fingerprint = NULL, "
                        "generation_completed_at = :completed, "
                        "generation_updated_at = :completed "
                        "WHERE run_id = :run_id AND generation_lease_token = :lease_token"
                    ),
                    {
                        "output_fingerprint": output_fingerprint,
                        "completed": completed,
                        "run_id": command.run_id,
                        "lease_token": lease_token,
                    },
                )
                session.flush()
                self._after_stage(GenerationWriteStage.STATE)
                schedule_baseline_notification(
                    session,
                    run_id=command.run_id,
                    group_id=command.group_id,
                    recipient_user_id=command.caller_user_id,
                    feedback_ids=feedback_ids,
                    enabled=self._notifications_enabled,
                    now=completed,
                )
                session.flush()
                self._after_stage(GenerationWriteStage.OUTBOX)
                session.commit()
                return BaselineGenerationReceipt(
                    run_id=command.run_id,
                    group_id=command.group_id,
                    state=BaselineGenerationState.SUCCEEDED,
                    attempt_count=int(run["generation_attempt_count"]),
                    input_fingerprint=current_input.input_fingerprint,
                    output_fingerprint=output_fingerprint,
                    feedback_ids=tuple(feedback_ids),
                )
            except Exception:
                session.rollback()
                raise

    def _record_failure(
        self,
        command: BaselineGenerationCommand,
        *,
        lease_token: str,
        attempt_count: int,
        input_fingerprint: str,
        code: str,
        retryable: bool,
        error_class: str,
    ) -> BaselineGenerationReceipt:
        safe_code = _safe_error_code(code, "generation_failed")
        state = (
            BaselineGenerationState.RETRYABLE_FAILED
            if retryable
            else BaselineGenerationState.TERMINAL_FAILED
        )
        with self._session_factory() as session:
            try:
                _begin_write(session)
                run = self._load_run(session, command, lock=True)
                if run["generation_state"] == BaselineGenerationState.SUCCEEDED.value:
                    receipt = self._receipt(session, run, replayed=True)
                    session.commit()
                    return receipt
                if (
                    run["generation_state"] != BaselineGenerationState.RUNNING.value
                    or run["generation_lease_token"] != lease_token
                ):
                    receipt = self._receipt(session, run, replayed=True)
                    session.commit()
                    return receipt
                session.execute(
                    text(
                        "UPDATE baseline_retrieval_run SET generation_state = :state, "
                        "generation_lease_token = NULL, generation_lease_expires_at = NULL, "
                        "generation_error_code = :code, "
                        "generation_error_fingerprint = :fingerprint, "
                        "generation_updated_at = :updated WHERE run_id = :run_id"
                    ),
                    {
                        "state": state.value,
                        "code": safe_code,
                        "fingerprint": _sha256_text(f"{error_class}:{safe_code}"),
                        "updated": self._clock(),
                        "run_id": command.run_id,
                    },
                )
                session.commit()
                return BaselineGenerationReceipt(
                    run_id=command.run_id,
                    group_id=command.group_id,
                    state=state,
                    attempt_count=attempt_count,
                    input_fingerprint=input_fingerprint,
                    output_fingerprint=None,
                    feedback_ids=(),
                    error_code=safe_code,
                )
            except Exception:
                session.rollback()
                raise

    def _validate_authorization_and_provenance(
        self,
        session: Session,
        command: BaselineGenerationCommand,
        run: dict[str, object],
    ) -> tuple[str | None, str | None]:
        if run["source_chunk_id"] is None or run["source_document_id"] is None:
            return "generation_source_deleted", None
        source = (
            session.execute(
                text(
                    "SELECT c.content FROM chunk c "
                    "JOIN document d ON d.document_id = c.document_id "
                    "JOIN document_to_group dtg ON dtg.document_id = d.document_id "
                    "JOIN user_to_group utg ON utg.group_id = dtg.group_id "
                    "JOIN \"user\" u ON u.user_id = utg.user_id "
                    "WHERE c.chunk_id = :chunk_id AND c.document_id = :document_id "
                    "AND dtg.group_id = :group_id AND utg.user_id = :caller_user_id"
                ),
                {
                    "chunk_id": run["source_chunk_id"],
                    "document_id": run["source_document_id"],
                    "group_id": command.group_id,
                    "caller_user_id": command.caller_user_id,
                },
            )
            .mappings()
            .one_or_none()
        )
        if source is None:
            return "generation_authorization_revoked", None
        expected_scope = f"group:{command.group_id}"
        if run["corpus_scope_key"] != expected_scope:
            return "generation_scope_mismatch", None
        corpus_statement = select(RetrievalCorpus).where(
            RetrievalCorpus.scope_key == expected_scope
        )
        if session.get_bind().dialect.name == "postgresql":
            corpus_statement = corpus_statement.with_for_update()
        corpus = session.scalar(corpus_statement)
        if (
            corpus is None
            or corpus.corpus_id != run["corpus_id"]
            or corpus.active_generation_id != run["corpus_generation_id"]
            or (
                corpus.source_document_id is not None
                and corpus.source_document_id != run["source_document_id"]
            )
        ):
            return "generation_corpus_stale", None
        generation = session.get(
            RetrievalCorpusGeneration, corpus.active_generation_id
        )
        if (
            generation is None
            or generation.status != CorpusGenerationStatus.ACTIVE.value
            or generation.manifest_hash != run["corpus_manifest_hash"]
            or generation.generation_version != run["corpus_generation_version"]
        ):
            return "generation_corpus_stale", None
        publication = session.get(RetrievalBaselineIndexPublication, corpus.corpus_id)
        if publication is None or publication.index_id != run["index_id"]:
            return "generation_publication_stale", None
        build = session.get(RetrievalBaselineIndexBuild, publication.index_id)
        if (
            build is None
            or build.status != BaselineIndexBuildStatus.COMPATIBLE.value
            or build.generation_id != generation.generation_id
        ):
            return "generation_publication_stale", None
        index_state = session.get(RetrievalIndexState, generation.generation_id)
        publication_fingerprint = published_index_fingerprint(build)
        exact = (
            (build.index_version, run["index_version"]),
            (build.index_schema_version, run["index_schema_version"]),
            (build.engine_config_fingerprint, run["config_fingerprint"]),
            (build.embedding_provider, run["embedding_provider"]),
            (build.embedding_model, run["embedding_model"]),
            (build.embedding_revision, run["embedding_revision"]),
            (build.embedding_dimension, run["embedding_dimension"]),
            (build.embedding_fingerprint, run["embedding_fingerprint"]),
            (publication_fingerprint, run["index_fingerprint"]),
            (publication_fingerprint, run["index_publication_fingerprint"]),
        )
        if (
            index_state is None
            or index_state.status != IndexStateStatus.COMPATIBLE.value
            or any(actual != expected for actual, expected in exact)
        ):
            return "generation_publication_incompatible", None
        return None, str(source["content"])

    def _load_run(
        self,
        session: Session,
        command: BaselineGenerationCommand,
        *,
        lock: bool,
    ) -> dict[str, object]:
        statement, params = _run_statement(
            command.run_id,
            command.group_id,
            lock=lock,
            dialect=session.get_bind().dialect.name,
        )
        row = session.execute(statement, params).mappings().one_or_none()
        if row is None:
            raise BaselineGenerationError(
                "generation_run_absent", "baseline retrieval run is unavailable"
            )
        return dict(row)

    def _block(self, session: Session, run_id: str, code: str) -> None:
        safe_code = _safe_error_code(code, "generation_blocked")
        session.execute(
            text(
                "UPDATE baseline_retrieval_run SET generation_state = 'blocked', "
                "generation_lease_token = NULL, generation_lease_expires_at = NULL, "
                "generation_error_code = :code, "
                "generation_error_fingerprint = :fingerprint, "
                "generation_updated_at = :updated WHERE run_id = :run_id"
            ),
            {
                "code": safe_code,
                "fingerprint": _sha256_text(f"blocked:{safe_code}"),
                "updated": self._clock(),
                "run_id": run_id,
            },
        )

    @staticmethod
    def _parse_output(
        output: object,
        *,
        maximum_findings: int,
    ) -> tuple[tuple[str, ...], str]:
        if not isinstance(output, str) or len(output) > MAX_GENERATION_OUTPUT_CHARACTERS:
            raise BaselineGenerationError(
                "provider_malformed_output", "baseline provider output is invalid"
            )
        output_fingerprint = _sha256_text(output)
        stripped = output.strip()
        if not stripped or stripped.upper() == "NONE":
            raise BaselineGenerationError(
                "provider_empty_output", "baseline provider produced no finding"
            )
        findings = tuple(
            part.strip()
            for part in re.split(
                rf"\s*{re.escape(GENERATION_FINDING_SEPARATOR)}\s*", stripped
            )
            if part.strip()
        )
        if not findings or len(findings) > maximum_findings:
            raise BaselineGenerationError(
                "provider_malformed_output", "baseline provider finding count is invalid"
            )
        return findings, output_fingerprint

    def _feedback_rows(self, session: Session, run_id: str):
        return (
            session.execute(
                text(
                    "SELECT feedback_id, baseline_finding_ordinal, "
                    "generation_input_fingerprint, generation_output_fingerprint "
                    "FROM feedback WHERE baseline_retrieval_run_id = :run_id "
                    "ORDER BY baseline_finding_ordinal"
                ),
                {"run_id": run_id},
            )
            .mappings()
            .all()
        )

    def _receipt(
        self,
        session: Session,
        run: dict[str, object],
        *,
        replayed: bool = False,
    ) -> BaselineGenerationReceipt:
        rows = self._feedback_rows(session, str(run["run_id"]))
        return BaselineGenerationReceipt(
            run_id=str(run["run_id"]),
            group_id=str(run["group_id"]),
            state=BaselineGenerationState(str(run["generation_state"])),
            attempt_count=int(run["generation_attempt_count"]),
            input_fingerprint=(
                str(run["generation_input_fingerprint"])
                if run["generation_input_fingerprint"] is not None
                else None
            ),
            output_fingerprint=(
                str(run["generation_output_fingerprint"])
                if run["generation_output_fingerprint"] is not None
                else None
            ),
            feedback_ids=tuple(str(row["feedback_id"]) for row in rows),
            error_code=(
                str(run["generation_error_code"])
                if run["generation_error_code"] is not None
                else None
            ),
            replayed=replayed,
        )

    def _after_stage(self, stage: GenerationWriteStage) -> None:
        if self._stage_hook is not None:
            self._stage_hook(stage)


__all__ = [
    "GENERATION_CONTRACT_VERSION",
    "GENERATION_FINDING_SEPARATOR",
    "GENERATION_STATE_VERSION",
    "BaselineGenerationBusyError",
    "BaselineGenerationCommand",
    "BaselineGenerationError",
    "BaselineGenerationEvidence",
    "BaselineGenerationInput",
    "BaselineGenerationInputAdapter",
    "BaselineGenerationProvider",
    "BaselineGenerationProviderError",
    "BaselineGenerationReceipt",
    "BaselineGenerationService",
    "BaselineGenerationState",
    "GenerationWriteStage",
    "ReviewerBaselineGenerationProvider",
]
