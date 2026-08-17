"""Lease-safe internal executor for one document-level baseline retrieval.

The only dispatch input is an opaque ``baseline_control_run_job`` identifier.
This module stops at ``references_persisted`` and has no generation,
notification, preview, public API, legacy-retrieval, or chunk-fan-out path.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol
from uuid import UUID, uuid4

from sqlalchemy import delete, select, text, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ...baseline_control_plane_schema import (
    BASELINE_RUN_WORKER_CONTRACT_VERSION,
    BASELINE_RUN_WORKER_SERVICE_ID,
    baseline_run_job,
    baseline_run_payload,
)
from ...baseline_evidence_schema import (
    SOURCE_SCOPE_CONTROL_DOCUMENT,
    SOURCE_SCOPE_VERSION,
    baseline_retrieval_run,
    baseline_selected_evidence,
)
from .baseline import MAX_EVIDENCE_CHARACTERS, MAX_EVIDENCE_ITEMS, RETRIEVAL_LIMIT
from .control_plane_v2 import PROTOCOL_V2_SHA256, PROTOCOL_V2_VERSION
from .corpus import RetrievalCorpusGeneration
from .embedding import create_configured_persistent_baseline_retriever
from .evidence_persistence import (
    BaselineEvidencePersistenceCommand,
    BaselineEvidencePersistenceError,
    BaselineEvidencePersistenceReceipt,
    BaselineEvidencePersistenceService,
    ControlDocumentSource,
    retrieval_result_fingerprint,
)
from .persistent import PERSISTENT_BASELINE_ENGINE_VERSION
from .run_jobs import (
    BaselineRunJobError,
    BaselineRunJobService,
    BaselineRunKeyring,
    ProtectedRunPayload,
    keyring_from_settings,
)
from .types import (
    RESULT_SCHEMA_VERSION,
    RetrievalQueryOrigin,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStatus,
)

RUN_PERSISTENCE_IDENTITY_VERSION = "baseline-document-persistence.v1"
DEFAULT_RUN_LEASE_LIFETIME = timedelta(minutes=5)
MIN_RUN_LEASE_LIFETIME = timedelta(seconds=30)
MAX_RUN_LEASE_LIFETIME = timedelta(minutes=30)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]*$")
_HEX = frozenset("0123456789abcdef")
_CLAIMABLE_STATES = frozenset({"queued", "retryable_failed"})
_PAYLOAD_ERASING_STATES = frozenset(
    {"insufficient", "terminal_failed", "blocked", "cancelled"}
)
_RETRYABLE_RESULT_CODES = frozenset(
    {
        "embedding_adapter_unavailable",
        "embedding_provider_unavailable",
        "embedding_request_failed",
        "embedding_service_timeout",
        "embedding_service_unavailable",
        "persistent_index_read_failed",
        "query_embedding_failed",
    }
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: object) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _uuid(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise BaselineRunExecutorError(f"{label}_invalid", retryable=False)
    try:
        parsed = UUID(value)
    except (AttributeError, ValueError):
        raise BaselineRunExecutorError(f"{label}_invalid", retryable=False) from None
    if str(parsed) != value.lower():
        raise BaselineRunExecutorError(f"{label}_invalid", retryable=False)
    return str(parsed)


def _safe_code(value: object, fallback: str = "internal_failure") -> str:
    if (
        isinstance(value, str)
        and 1 <= len(value) <= 128
        and _SAFE_ID.fullmatch(value) is not None
    ):
        return value
    return fallback


class BaselineRunRetriever(Protocol):
    def retrieve(self, request: RetrievalRequest) -> RetrievalResult: ...


class BaselineRunExecutorStage(str, Enum):
    CLAIMED = "claimed"
    PAYLOAD_DECRYPTED = "payload_decrypted"
    BEFORE_RETRIEVAL = "before_retrieval"
    AFTER_RETRIEVAL = "after_retrieval"
    BEFORE_EVIDENCE_PERSISTENCE = "before_evidence_persistence"
    AFTER_EVIDENCE_COMMIT = "after_evidence_commit"
    BEFORE_NON_OK_COMMIT = "before_non_ok_commit"


class BaselineRunExecutorError(RuntimeError):
    """Sanitized worker failure containing no protected inputs."""

    def __init__(
        self,
        code: str,
        *,
        retryable: bool,
        state: str | None = None,
    ) -> None:
        self.code = _safe_code(code)
        self.retryable = retryable
        self.state = state
        super().__init__(self.code)


@dataclass(frozen=True, slots=True)
class InternalBaselineRunWorkerIdentity:
    instance_id: str
    service_id: str = BASELINE_RUN_WORKER_SERVICE_ID
    contract_version: str = BASELINE_RUN_WORKER_CONTRACT_VERSION

    @classmethod
    def create(
        cls, instance_id: str | None = None
    ) -> InternalBaselineRunWorkerIdentity:
        return cls(instance_id=instance_id or f"worker-{uuid4()}")

    def validate(self) -> None:
        if (
            self.service_id != BASELINE_RUN_WORKER_SERVICE_ID
            or self.contract_version != BASELINE_RUN_WORKER_CONTRACT_VERSION
            or not 1 <= len(self.instance_id) <= 128
            or _SAFE_ID.fullmatch(self.instance_id) is None
        ):
            raise BaselineRunExecutorError("worker_identity_invalid", retryable=False)


@dataclass(frozen=True, slots=True, repr=False)
class BaselineRunLease:
    job_id: str
    lease_token: str
    lease_expires_at: datetime
    attempt_count: int

    def __repr__(self) -> str:
        return (
            "BaselineRunLease(job_id=<opaque>, lease_token=<redacted>, "
            f"attempt_count={self.attempt_count})"
        )


@dataclass(frozen=True, slots=True)
class BaselineRunExecutionOutcome:
    job_id: str
    state: str
    attempt_count: int
    retrieval_result_fingerprint: str | None
    persisted_run_id: str | None
    selected_evidence_ids: tuple[str, ...]
    reference_ids: tuple[str, ...]
    evidence_count: int
    reference_count: int
    replayed: bool


@dataclass(frozen=True, slots=True)
class _ExecutionContext:
    job: Mapping[str, Any]
    generation_version: str
    generation_manifest_hash: str


def derive_document_persistence_identity(
    parent_secret: bytes,
    *,
    intent_hash: str,
) -> str:
    """Derive one opaque document-level persistence identity."""

    if not isinstance(parent_secret, bytes) or len(parent_secret) != 32:
        raise BaselineRunExecutorError("parent_identity_invalid", retryable=False)
    if (
        not isinstance(intent_hash, str)
        or len(intent_hash) != 64
        or any(character not in _HEX for character in intent_hash)
    ):
        raise BaselineRunExecutorError("run_intent_invalid", retryable=False)
    return hmac.new(
        parent_secret,
        f"{RUN_PERSISTENCE_IDENTITY_VERSION}\x00{intent_hash}".encode("ascii"),
        hashlib.sha256,
    ).hexdigest()


class BaselineDocumentRunExecutor:
    """Internal one-query executor that stops at durable References."""

    def __init__(
        self,
        engine: Engine,
        *,
        identity: InternalBaselineRunWorkerIdentity,
        keyring: BaselineRunKeyring,
        retriever_factory: Callable[[], BaselineRunRetriever],
        clock: Callable[[], datetime] = _utcnow,
        token_factory: Callable[[int], str] | None = None,
        stage_hook: Callable[[BaselineRunExecutorStage], None] | None = None,
        persistence_factory: Callable[[], BaselineEvidencePersistenceService]
        | None = None,
    ) -> None:
        identity.validate()
        self.engine = engine
        self.sessions = sessionmaker(engine, expire_on_commit=False)
        self.identity = identity
        self.keyring = keyring
        self.retriever_factory = retriever_factory
        self.clock = clock
        self.token_factory = token_factory or (lambda size: secrets.token_urlsafe(size))
        self.stage_hook = stage_hook
        self.jobs = BaselineRunJobService(engine, keyring, clock=clock)
        self.persistence_factory = persistence_factory or (
            lambda: BaselineEvidencePersistenceService(self.sessions)
        )

    @classmethod
    def from_settings(
        cls,
        engine: Engine,
        settings: Any,
        *,
        identity: InternalBaselineRunWorkerIdentity,
        clock: Callable[[], datetime] = _utcnow,
    ) -> BaselineDocumentRunExecutor:
        return cls(
            engine,
            identity=identity,
            keyring=keyring_from_settings(settings),
            retriever_factory=lambda: create_configured_persistent_baseline_retriever(
                sessionmaker(engine, expire_on_commit=False),
                settings=settings,
            ),
            clock=clock,
        )

    def _stage(self, stage: BaselineRunExecutorStage) -> None:
        if self.stage_hook is not None:
            self.stage_hook(stage)

    @staticmethod
    def _begin(session: Session) -> None:
        if session.get_bind().dialect.name == "sqlite":
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
        else:
            session.begin()

    @staticmethod
    def _lock(statement: Any, session: Session) -> Any:
        return (
            statement.with_for_update()
            if session.get_bind().dialect.name == "postgresql"
            else statement
        )

    def _locked_job(self, session: Session, job_id: str) -> Mapping[str, Any]:
        row = (
            session.execute(
                self._lock(
                    select(baseline_run_job).where(baseline_run_job.c.job_id == job_id),
                    session,
                )
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise BaselineRunExecutorError("job_not_found", retryable=False)
        return dict(row)

    def _validate_frozen_job(self, job: Mapping[str, Any]) -> None:
        if (
            job.get("contract_version") != "baseline-run-job.v1"
            or job.get("protocol_version") != PROTOCOL_V2_VERSION
            or job.get("protocol_sha256") != PROTOCOL_V2_SHA256
            or job.get("query_representation") != "raw_git_diff_v1"
            or job.get("query_encoding") != "utf-8"
            or job.get("query_origin") != "explicit"
            or not isinstance(job.get("query_sha256"), str)
            or len(str(job["query_sha256"])) != 64
            or int(job.get("query_length") or 0) <= 0
            or int(job.get("query_byte_length") or 0) <= 0
            or job.get("source_document_id") is None
            or job.get("submitted_by_user_id") is None
        ):
            raise BaselineRunExecutorError("job_contract_incompatible", retryable=False)

    def _reauthorize(
        self,
        session: Session,
        job: Mapping[str, Any],
    ) -> tuple[str, str]:
        self._validate_frozen_job(job)
        try:
            authorized = self.jobs._authorize_publication(
                session,
                caller_user_id=str(job["submitted_by_user_id"]),
                group_id=str(job["group_id"]),
                source_document_id=str(job["source_document_id"]),
                changed_registration_id=str(job["changed_repository_registration_id"]),
                publication=self.jobs._publication_from_row(job),
                lock=True,
            )
        except BaselineRunJobError as exc:
            raise BaselineRunExecutorError(
                _safe_code(exc.code, "authorization_revoked"),
                retryable=False,
                state="blocked",
            ) from None
        if (
            authorized.index_job_id != job["index_job_id"]
            or authorized.corpus_id != job["corpus_id"]
        ):
            raise BaselineRunExecutorError(
                "index_publication_stale", retryable=False, state="blocked"
            )
        generation = session.get(
            RetrievalCorpusGeneration, str(job["corpus_generation_id"])
        )
        if (
            generation is None
            or not isinstance(generation.manifest_hash, str)
            or len(generation.manifest_hash) != 64
        ):
            raise BaselineRunExecutorError(
                "corpus_incompatible", retryable=False, state="blocked"
            )
        return generation.generation_version, generation.manifest_hash

    def _payload(
        self,
        session: Session,
        job: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        payload = (
            session.execute(
                self._lock(
                    select(baseline_run_payload).where(
                        baseline_run_payload.c.job_id == job["job_id"],
                        baseline_run_payload.c.group_id == job["group_id"],
                    ),
                    session,
                )
            )
            .mappings()
            .one_or_none()
        )
        if payload is None:
            raise BaselineRunExecutorError(
                "payload_unavailable", retryable=False, state="blocked"
            )
        expiry = _aware(payload["expires_at"])
        if expiry is None or expiry <= self.clock():
            raise BaselineRunExecutorError(
                "payload_expired", retryable=False, state="blocked"
            )
        try:
            self.keyring.decryption_key(str(payload["key_id"]))
        except BaselineRunJobError:
            raise BaselineRunExecutorError(
                "payload_authentication_failed", retryable=False, state="blocked"
            ) from None
        return dict(payload)

    def _erase_payload(self, session: Session, job: Mapping[str, Any]) -> None:
        session.execute(
            delete(baseline_run_payload).where(
                baseline_run_payload.c.job_id == job["job_id"],
                baseline_run_payload.c.group_id == job["group_id"],
            )
        )

    def _block_locked(
        self,
        session: Session,
        job: Mapping[str, Any],
        *,
        code: str,
        require_token: str | None = None,
    ) -> None:
        now = self.clock()
        conditions = [baseline_run_job.c.job_id == job["job_id"]]
        if require_token is not None:
            conditions.extend(
                [
                    baseline_run_job.c.state == "running",
                    baseline_run_job.c.lease_token == require_token,
                ]
            )
        changed = session.execute(
            update(baseline_run_job)
            .where(*conditions)
            .values(
                state="blocked",
                lease_token=None,
                lease_expires_at=None,
                reason_code=_safe_code(code),
                failure_stage="retrieval",
                updated_at=now,
                finished_at=now,
            )
        )
        if changed.rowcount != 1:
            raise BaselineRunExecutorError("job_lease_unavailable", retryable=True)
        self._erase_payload(session, job)

    def claim(
        self,
        job_id: str,
        *,
        lifetime: timedelta = DEFAULT_RUN_LEASE_LIFETIME,
    ) -> BaselineRunLease:
        job_id = _uuid(job_id, "job_id")
        if not MIN_RUN_LEASE_LIFETIME <= lifetime <= MAX_RUN_LEASE_LIFETIME:
            raise BaselineRunExecutorError("lease_lifetime_invalid", retryable=False)
        terminal_error: BaselineRunExecutorError | None = None
        receipt: BaselineRunLease | None = None
        with self.sessions() as session:
            try:
                self._begin(session)
                job = self._locked_job(session, job_id)
                now = self.clock()
                if job["state"] == "references_persisted":
                    raise BaselineRunExecutorError(
                        "job_already_completed", retryable=False
                    )
                expired_running = (
                    job["state"] == "running"
                    and _aware(job["lease_expires_at"]) is not None
                    and _aware(job["lease_expires_at"]) <= now
                )
                if job["state"] not in _CLAIMABLE_STATES and not expired_running:
                    raise BaselineRunExecutorError(
                        "job_lease_unavailable", retryable=True
                    )
                try:
                    self._reauthorize(session, job)
                    payload = self._payload(session, job)
                except BaselineRunExecutorError as exc:
                    self._block_locked(session, job, code=exc.code)
                    terminal_error = exc
                if terminal_error is None:
                    attempt = int(job["attempt_count"]) + 1
                    token = self.token_factory(32)
                    if (
                        not isinstance(token, str)
                        or not 32 <= len(token) <= 128
                        or any(ord(character) < 33 for character in token)
                    ):
                        raise BaselineRunExecutorError(
                            "lease_token_invalid", retryable=False
                        )
                    payload_expiry = _aware(payload["expires_at"])
                    assert payload_expiry is not None
                    expires_at = min(now + lifetime, payload_expiry)
                    changed = session.execute(
                        update(baseline_run_job)
                        .where(
                            baseline_run_job.c.job_id == job_id,
                            (
                                baseline_run_job.c.state.in_(_CLAIMABLE_STATES)
                                | (
                                    (baseline_run_job.c.state == "running")
                                    & (baseline_run_job.c.lease_expires_at <= now)
                                )
                            ),
                        )
                        .values(
                            state="running",
                            attempt_count=attempt,
                            lease_token=token,
                            lease_expires_at=expires_at,
                            worker_service_id=self.identity.service_id,
                            worker_contract_version=self.identity.contract_version,
                            started_at=job["started_at"] or now,
                            reason_code=None,
                            failure_stage=None,
                            updated_at=now,
                            finished_at=None,
                        )
                    )
                    if changed.rowcount != 1:
                        raise BaselineRunExecutorError(
                            "job_lease_unavailable", retryable=True
                        )
                    receipt = BaselineRunLease(job_id, token, expires_at, attempt)
                session.commit()
            except Exception:
                session.rollback()
                raise
        if terminal_error is not None:
            raise terminal_error
        assert receipt is not None
        self._stage(BaselineRunExecutorStage.CLAIMED)
        return receipt

    def renew(
        self,
        job_id: str,
        lease_token: str,
        *,
        lifetime: timedelta = DEFAULT_RUN_LEASE_LIFETIME,
    ) -> datetime:
        job_id = _uuid(job_id, "job_id")
        if not MIN_RUN_LEASE_LIFETIME <= lifetime <= MAX_RUN_LEASE_LIFETIME:
            raise BaselineRunExecutorError("lease_lifetime_invalid", retryable=False)
        with self.sessions() as session:
            try:
                self._begin(session)
                job = self._locked_job(session, job_id)
                now = self.clock()
                if (
                    job["state"] != "running"
                    or job["lease_token"] != lease_token
                    or _aware(job["lease_expires_at"]) is None
                    or _aware(job["lease_expires_at"]) <= now
                ):
                    raise BaselineRunExecutorError(
                        "job_lease_unavailable", retryable=True
                    )
                try:
                    self._reauthorize(session, job)
                    payload = self._payload(session, job)
                except BaselineRunExecutorError as exc:
                    self._block_locked(
                        session, job, code=exc.code, require_token=lease_token
                    )
                    session.commit()
                    raise
                payload_expiry = _aware(payload["expires_at"])
                assert payload_expiry is not None
                expiry = min(now + lifetime, payload_expiry)
                changed = session.execute(
                    update(baseline_run_job)
                    .where(
                        baseline_run_job.c.job_id == job_id,
                        baseline_run_job.c.state == "running",
                        baseline_run_job.c.lease_token == lease_token,
                        baseline_run_job.c.lease_expires_at > now,
                    )
                    .values(lease_expires_at=expiry, updated_at=now)
                )
                if changed.rowcount != 1:
                    raise BaselineRunExecutorError(
                        "job_lease_unavailable", retryable=True
                    )
                session.commit()
                return expiry
            except Exception:
                if session.in_transaction():
                    session.rollback()
                raise

    def _open_payload(
        self,
        job_id: str,
        lease_token: str,
    ) -> tuple[_ExecutionContext, ProtectedRunPayload]:
        failure: BaselineRunExecutorError | None = None
        opened: ProtectedRunPayload | None = None
        context: _ExecutionContext | None = None
        with self.sessions() as session:
            try:
                self._begin(session)
                job = self._locked_job(session, job_id)
                now = self.clock()
                if (
                    job["state"] != "running"
                    or job["lease_token"] != lease_token
                    or _aware(job["lease_expires_at"]) is None
                    or _aware(job["lease_expires_at"]) <= now
                ):
                    raise BaselineRunExecutorError(
                        "job_lease_unavailable", retryable=True
                    )
                try:
                    generation_version, generation_manifest_hash = self._reauthorize(
                        session, job
                    )
                    payload = self._payload(session, job)
                    opened = self.jobs.cipher.decrypt(job=job, payload=payload)
                except (BaselineRunJobError, BaselineRunExecutorError) as exc:
                    code = (
                        exc.code
                        if isinstance(
                            exc, (BaselineRunJobError, BaselineRunExecutorError)
                        )
                        else "payload_authentication_failed"
                    )
                    safe = (
                        "payload_authentication_failed"
                        if code
                        in {
                            "run_payload_authentication_failed",
                            "run_payload_key_unavailable",
                        }
                        else code
                    )
                    self._block_locked(
                        session, job, code=safe, require_token=lease_token
                    )
                    failure = BaselineRunExecutorError(
                        safe, retryable=False, state="blocked"
                    )
                if failure is None:
                    context = _ExecutionContext(
                        job=dict(job),
                        generation_version=generation_version,
                        generation_manifest_hash=generation_manifest_hash,
                    )
                session.commit()
            except Exception:
                session.rollback()
                raise
        if failure is not None:
            raise failure
        assert opened is not None and context is not None
        self._stage(BaselineRunExecutorStage.PAYLOAD_DECRYPTED)
        return context, opened

    @staticmethod
    def _request(
        context: _ExecutionContext, payload: ProtectedRunPayload
    ) -> RetrievalRequest:
        job = context.job
        return RetrievalRequest(
            request_id=str(job["processing_run_id"]),
            changed_repository=None,
            repository_roots=(),
            corpus_version=context.generation_version,
            retrieval_query=payload.retrieval_query,
            retrieval_query_origin=RetrievalQueryOrigin.EXPLICIT,
            query_kind="raw_git_diff_v1",
            corpus_complete=True,
            corpus_scope_key=f"group:{job['group_id']}",
            changed_repository_id=str(job["changed_repository_registration_id"]),
            group_id=str(job["group_id"]),
            source_document_id=str(job["source_document_id"]),
        )

    @staticmethod
    def _validate_result(
        context: _ExecutionContext,
        result: RetrievalResult,
    ) -> None:
        if not isinstance(result, RetrievalResult):
            raise BaselineRunExecutorError(
                "retrieval_result_incompatible", retryable=False
            )
        job = context.job
        query = result.query_provenance
        publication_bound = result.corpus_id is not None
        if (
            result.schema_version != RESULT_SCHEMA_VERSION
            or result.request_id != job["processing_run_id"]
            or result.engine != "baseline_v1"
            or result.engine_version != PERSISTENT_BASELINE_ENGINE_VERSION
            or result.fallback_engine is not None
            or result.corpus_version != context.generation_version
            or query is None
            or query.origin is not RetrievalQueryOrigin.EXPLICIT
            or query.sha256 != job["query_sha256"]
            or query.length != job["query_length"]
            or result.status is RetrievalStatus.OK
            and result.candidate_count != len(result.candidates)
            or result.status is not RetrievalStatus.OK
            and bool(result.candidates)
            and result.candidate_count != len(result.candidates)
            or result.evidence_characters
            != sum(len(item.content) for item in result.evidence)
            or result.evidence_characters > MAX_EVIDENCE_CHARACTERS
            or len(result.evidence) > MAX_EVIDENCE_ITEMS
            or result.retrieved_count > RETRIEVAL_LIMIT
            or result.status is RetrievalStatus.OK
            and (result.error is not None or not result.evidence)
            or result.status is not RetrievalStatus.OK
            and result.error is None
            or result.status is not RetrievalStatus.OK
            and bool(result.evidence)
            or not publication_bound
            and (
                result.status is RetrievalStatus.OK
                or bool(result.candidates)
                or bool(result.evidence)
                or result.candidate_count != 0
                or result.retrieved_count != 0
            )
        ):
            raise BaselineRunExecutorError(
                "retrieval_result_incompatible", retryable=False
            )
        if publication_bound and (
            result.config_fingerprint != job["retrieval_config_fingerprint"]
            or result.corpus_id != job["corpus_id"]
            or result.corpus_manifest_hash != context.generation_manifest_hash
            or result.corpus_scope_key != f"group:{job['group_id']}"
            or result.index_id != job["index_publication_id"]
            or result.index_schema_version != job["index_format_version"]
            or result.index_fingerprint != job["index_fingerprint"]
            or result.embedding_fingerprint != job["embedding_fingerprint"]
        ):
            raise BaselineRunExecutorError(
                "retrieval_result_incompatible", retryable=False
            )
        scores = [
            score
            for item in result.candidates
            for score in (item.bm25_score, item.dense_score, item.rrf_score)
        ]
        if any(not math.isfinite(score) for score in scores):
            raise BaselineRunExecutorError(
                "retrieval_result_incompatible", retryable=False
            )

    def _complete_non_ok(
        self,
        *,
        job_id: str,
        lease_token: str,
        result: RetrievalResult,
    ) -> BaselineRunExecutionOutcome:
        self._stage(BaselineRunExecutorStage.BEFORE_NON_OK_COMMIT)
        fingerprint = retrieval_result_fingerprint(result)
        state = "insufficient"
        reason = "retrieval_insufficient"
        with self.sessions() as session:
            try:
                self._begin(session)
                job = self._locked_job(session, job_id)
                now = self.clock()
                if (
                    job["state"] != "running"
                    or job["lease_token"] != lease_token
                    or _aware(job["lease_expires_at"]) is None
                    or _aware(job["lease_expires_at"]) <= now
                    or job["persisted_run_id"] is not None
                    or int(job["evidence_count"]) != 0
                    or int(job["reference_count"]) != 0
                ):
                    raise BaselineRunExecutorError(
                        "job_lease_unavailable", retryable=True
                    )
                try:
                    self._reauthorize(session, job)
                    self._payload(session, job)
                except BaselineRunExecutorError as exc:
                    self._block_locked(
                        session, job, code=exc.code, require_token=lease_token
                    )
                    session.commit()
                    raise
                changed = session.execute(
                    update(baseline_run_job)
                    .where(
                        baseline_run_job.c.job_id == job_id,
                        baseline_run_job.c.state == "running",
                        baseline_run_job.c.lease_token == lease_token,
                        baseline_run_job.c.lease_expires_at > now,
                        baseline_run_job.c.persisted_run_id.is_(None),
                    )
                    .values(
                        state=state,
                        lease_token=None,
                        lease_expires_at=None,
                        retrieval_result_fingerprint=fingerprint,
                        reason_code=reason,
                        failure_stage="retrieval",
                        updated_at=now,
                        finished_at=now,
                    )
                )
                if changed.rowcount != 1:
                    raise BaselineRunExecutorError(
                        "job_lease_unavailable", retryable=True
                    )
                self._erase_payload(session, job)
                session.commit()
            except Exception:
                if session.in_transaction():
                    session.rollback()
                raise
        return BaselineRunExecutionOutcome(
            job_id=job_id,
            state=state,
            attempt_count=int(job["attempt_count"]),
            retrieval_result_fingerprint=fingerprint,
            persisted_run_id=None,
            selected_evidence_ids=(),
            reference_ids=(),
            evidence_count=0,
            reference_count=0,
            replayed=False,
        )

    def _record_failure(
        self,
        *,
        job_id: str,
        lease_token: str,
        code: str,
        retryable: bool,
        state: str | None = None,
        result_fingerprint: str | None = None,
    ) -> bool:
        safe = _safe_code(code)
        with self.sessions() as session:
            try:
                self._begin(session)
                job = self._locked_job(session, job_id)
                if job["state"] == "references_persisted":
                    session.commit()
                    return False
                now = self.clock()
                if (
                    job["state"] != "running"
                    or job["lease_token"] != lease_token
                    or _aware(job["lease_expires_at"]) is None
                    or _aware(job["lease_expires_at"]) <= now
                ):
                    session.commit()
                    return False
                target = state or (
                    "retryable_failed" if retryable else "terminal_failed"
                )
                if target == "retryable_failed":
                    try:
                        self._reauthorize(session, job)
                        self._payload(session, job)
                    except BaselineRunExecutorError as exc:
                        target = "blocked"
                        safe = exc.code
                values: dict[str, object] = {
                    "state": target,
                    "lease_token": None,
                    "lease_expires_at": None,
                    "reason_code": safe,
                    "failure_stage": "retrieval",
                    "updated_at": now,
                    "finished_at": None if target == "retryable_failed" else now,
                }
                if result_fingerprint is not None and target != "retryable_failed":
                    values["retrieval_result_fingerprint"] = result_fingerprint
                changed = session.execute(
                    update(baseline_run_job)
                    .where(
                        baseline_run_job.c.job_id == job_id,
                        baseline_run_job.c.state == "running",
                        baseline_run_job.c.lease_token == lease_token,
                        baseline_run_job.c.lease_expires_at > now,
                    )
                    .values(**values)
                )
                if changed.rowcount != 1:
                    session.commit()
                    return False
                if target in _PAYLOAD_ERASING_STATES:
                    self._erase_payload(session, job)
                session.commit()
                return True
            except Exception:
                if session.in_transaction():
                    session.rollback()
                raise

    def cancel(self, job_id: str, lease_token: str | None = None) -> None:
        job_id = _uuid(job_id, "job_id")
        if lease_token is None:
            lease_token = self.claim(job_id).lease_token
        if not self._record_failure(
            job_id=job_id,
            lease_token=lease_token,
            code="job_cancelled",
            retryable=False,
            state="cancelled",
        ):
            raise BaselineRunExecutorError("job_lease_unavailable", retryable=True)

    def _completed_outcome(self, job_id: str) -> BaselineRunExecutionOutcome | None:
        with self.sessions() as session:
            try:
                self._begin(session)
                job = self._locked_job(session, job_id)
                if job["state"] != "references_persisted":
                    if job["persisted_run_id"] is not None:
                        raise BaselineRunExecutorError(
                            "job_state_incompatible", retryable=False
                        )
                    session.commit()
                    return None
                run_id = job["persisted_run_id"]
                fingerprint = job["retrieval_result_fingerprint"]
                run = (
                    session.execute(
                        select(baseline_retrieval_run).where(
                            baseline_retrieval_run.c.run_id == run_id,
                            baseline_retrieval_run.c.group_id == job["group_id"],
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                selected = (
                    session.execute(
                        select(
                            baseline_selected_evidence.c.selected_evidence_id,
                        )
                        .where(baseline_selected_evidence.c.run_id == run_id)
                        .order_by(baseline_selected_evidence.c.ordinal)
                    )
                    .scalars()
                    .all()
                )
                references = (
                    session.execute(
                        text(
                            "SELECT r.reference_id FROM reference r JOIN "
                            "baseline_selected_evidence s ON "
                            "s.selected_evidence_id = r.baseline_selected_evidence_id "
                            "WHERE s.run_id = :run_id ORDER BY s.ordinal"
                        ),
                        {"run_id": run_id},
                    )
                    .scalars()
                    .all()
                )
                if (
                    run is None
                    or run["source_scope_version"] != SOURCE_SCOPE_VERSION
                    or run["source_scope"] != SOURCE_SCOPE_CONTROL_DOCUMENT
                    or run["source_chunk_id"] is not None
                    or (
                        run["source_document_id"] is not None
                        and job["source_document_id"] is not None
                        and run["source_document_id"] != job["source_document_id"]
                    )
                    or not isinstance(fingerprint, str)
                    or len(fingerprint) != 64
                    or not 1 <= len(selected) <= MAX_EVIDENCE_ITEMS
                    or len(references) != len(selected)
                    or int(job["evidence_count"]) != len(selected)
                    or int(job["reference_count"]) != len(references)
                ):
                    raise BaselineRunExecutorError(
                        "job_state_incompatible", retryable=False
                    )
                self._erase_payload(session, job)
                session.execute(
                    update(baseline_run_job)
                    .where(baseline_run_job.c.job_id == job_id)
                    .values(lease_token=None, lease_expires_at=None)
                )
                session.commit()
                return BaselineRunExecutionOutcome(
                    job_id=job_id,
                    state="references_persisted",
                    attempt_count=int(job["attempt_count"]),
                    retrieval_result_fingerprint=fingerprint,
                    persisted_run_id=str(run_id),
                    selected_evidence_ids=tuple(str(item) for item in selected),
                    reference_ids=tuple(str(item) for item in references),
                    evidence_count=len(selected),
                    reference_count=len(references),
                    replayed=True,
                )
            except Exception:
                if session.in_transaction():
                    session.rollback()
                raise

    def execute(
        self,
        job_id: str,
        *,
        lifetime: timedelta = DEFAULT_RUN_LEASE_LIFETIME,
    ) -> BaselineRunExecutionOutcome:
        job_id = _uuid(job_id, "job_id")
        completed = self._completed_outcome(job_id)
        if completed is not None:
            return completed
        lease = self.claim(job_id, lifetime=lifetime)
        try:
            context, protected = self._open_payload(job_id, lease.lease_token)
            request = self._request(context, protected)
            self._stage(BaselineRunExecutorStage.BEFORE_RETRIEVAL)
            self.renew(job_id, lease.lease_token, lifetime=lifetime)
            try:
                result = self.retriever_factory().retrieve(request)
            except Exception:  # noqa: BLE001 - provider/database execution boundary
                self._record_failure(
                    job_id=job_id,
                    lease_token=lease.lease_token,
                    code="retrieval_error",
                    retryable=True,
                )
                raise BaselineRunExecutorError(
                    "retrieval_error", retryable=True, state="retryable_failed"
                ) from None
            self._stage(BaselineRunExecutorStage.AFTER_RETRIEVAL)
            try:
                self._validate_result(context, result)
            except BaselineRunExecutorError as exc:
                self._record_failure(
                    job_id=job_id,
                    lease_token=lease.lease_token,
                    code=exc.code,
                    retryable=False,
                    state="terminal_failed",
                )
                raise

            fingerprint = retrieval_result_fingerprint(result)
            if result.status is RetrievalStatus.INSUFFICIENT:
                return self._complete_non_ok(
                    job_id=job_id,
                    lease_token=lease.lease_token,
                    result=result,
                )
            if result.status is RetrievalStatus.ERROR:
                assert result.error is not None
                retryable = result.error.code in _RETRYABLE_RESULT_CODES
                target = "retryable_failed" if retryable else "blocked"
                self._record_failure(
                    job_id=job_id,
                    lease_token=lease.lease_token,
                    code=result.error.code,
                    retryable=retryable,
                    state=target,
                    result_fingerprint=None if retryable else fingerprint,
                )
                raise BaselineRunExecutorError(
                    result.error.code, retryable=retryable, state=target
                )

            self._stage(BaselineRunExecutorStage.BEFORE_EVIDENCE_PERSISTENCE)
            command = BaselineEvidencePersistenceCommand(
                group_id=str(context.job["group_id"]),
                source=ControlDocumentSource(
                    document_id=str(context.job["source_document_id"]),
                    control_job_id=job_id,
                    lease_token=lease.lease_token,
                ),
                idempotency_key=derive_document_persistence_identity(
                    protected.parent_processing_secret,
                    intent_hash=str(context.job["intent_hash"]),
                ),
                retrieval_result=result,
                caller_user_id=str(context.job["submitted_by_user_id"]),
            )
            try:
                receipt = self.persistence_factory().persist(command)
            except BaselineEvidencePersistenceError as exc:
                retryable = exc.code in {
                    "control_job_lease_invalid",
                    "persistence_conflict",
                }
                state = None if retryable else "blocked"
                self._record_failure(
                    job_id=job_id,
                    lease_token=lease.lease_token,
                    code=exc.code,
                    retryable=retryable,
                    state=state,
                )
                raise BaselineRunExecutorError(
                    exc.code, retryable=retryable, state=state
                ) from None
            except Exception:  # noqa: BLE001 - transactional persistence boundary
                self._record_failure(
                    job_id=job_id,
                    lease_token=lease.lease_token,
                    code="persistence_failed",
                    retryable=True,
                )
                raise BaselineRunExecutorError(
                    "persistence_failed", retryable=True, state="retryable_failed"
                ) from None
            self._stage(BaselineRunExecutorStage.AFTER_EVIDENCE_COMMIT)
            return self._outcome_from_receipt(receipt, fingerprint)
        except BaselineRunExecutorError:  # noqa: TRY203 - explicit worker boundary
            raise

    def _outcome_from_receipt(
        self,
        receipt: BaselineEvidencePersistenceReceipt,
        fingerprint: str,
    ) -> BaselineRunExecutionOutcome:
        with self.sessions() as session:
            job = (
                session.execute(
                    select(baseline_run_job).where(
                        baseline_run_job.c.persisted_run_id == receipt.run_id
                    )
                )
                .mappings()
                .one_or_none()
            )
        if job is None or job["state"] != "references_persisted":
            raise BaselineRunExecutorError("job_state_incompatible", retryable=False)
        return BaselineRunExecutionOutcome(
            job_id=str(job["job_id"]),
            state="references_persisted",
            attempt_count=int(job["attempt_count"]),
            retrieval_result_fingerprint=fingerprint,
            persisted_run_id=receipt.run_id,
            selected_evidence_ids=receipt.selected_evidence_ids,
            reference_ids=receipt.reference_ids,
            evidence_count=len(receipt.selected_evidence_ids),
            reference_count=len(receipt.reference_ids),
            replayed=receipt.replayed,
        )


__all__ = [
    "BASELINE_RUN_WORKER_CONTRACT_VERSION",
    "BASELINE_RUN_WORKER_SERVICE_ID",
    "DEFAULT_RUN_LEASE_LIFETIME",
    "RUN_PERSISTENCE_IDENTITY_VERSION",
    "BaselineDocumentRunExecutor",
    "BaselineRunExecutionOutcome",
    "BaselineRunExecutorError",
    "BaselineRunExecutorStage",
    "BaselineRunLease",
    "InternalBaselineRunWorkerIdentity",
    "derive_document_persistence_identity",
]
