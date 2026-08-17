"""Durable database-backed scheduler for existing baseline job services.

The database job rows are the dispatch records.  This module selects only
opaque identifiers, closes the scheduler transaction, and then delegates
claiming and execution to the existing lease-owning workers.  It never accepts
or reconstructs protected input at the scheduling boundary.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from time import monotonic
from typing import Any, Protocol
from uuid import UUID, uuid4

from sqlalchemy import Engine, delete, func, inspect, or_, select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from ...baseline_control_plane_schema import (
    BASELINE_DATABASE_WORKER_CONTRACT_VERSION,
    BASELINE_WORKER_INSTANCE_TABLE,
    baseline_run_job,
    baseline_run_payload,
    baseline_worker_instance,
    control_job,
    snapshot_continuation_job,
)
from ...baseline_evidence_schema import baseline_retrieval_run
from ...schema_migrations import (
    CORE_SCHEMA_MIGRATIONS,
    schema_migration_table,
)
from .continuation_worker import (
    BaselineContinuationWorker,
    ContinuationWorkerError,
    InternalContinuationWorkerIdentity,
)
from .index_continuation import (
    BaselineCompatibleIndexJobService,
    IndexJobError,
    InternalIndexWorkerIdentity,
)
from .run_operator import (
    BaselineManualRunOperator,
    BaselineRunRuntimeError,
)

DATABASE_WORKER_MIGRATION_ID = "0013_baseline_database_worker_v1"
DATABASE_WORKER_CONTRACT_VERSION = BASELINE_DATABASE_WORKER_CONTRACT_VERSION
DATABASE_WORKER_SUPPORTED_JOB_TYPES = (
    "corpus_ingestion",
    "index_build",
    "baseline_run",
    "cleanup",
)
RUN_URGENCY_WINDOW = timedelta(seconds=120)
MAXIMUM_URGENT_RUN_BURST = 3
_RETRYABLE_STATES = frozenset({"retryable_failed"})
_LOGGER = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _uuid(value: str, label: str) -> str:
    try:
        parsed = UUID(value)
    except (AttributeError, ValueError):
        raise DatabaseWorkerError(f"{label}_invalid") from None
    if str(parsed) != value.lower():
        raise DatabaseWorkerError(f"{label}_invalid")
    return str(parsed)


class DatabaseWorkerError(RuntimeError):
    """A sanitized worker-process failure."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class DatabaseWorkerCandidate:
    job_type: str
    job_id: str
    group_id: str
    state: str
    attempt_count: int
    created_at: datetime
    updated_at: datetime
    payload_expires_at: datetime | None = None

    def __repr__(self) -> str:
        return (
            "DatabaseWorkerCandidate(job_type="
            f"{self.job_type!r}, job_id=<opaque>, state={self.state!r}, "
            f"attempt_count={self.attempt_count})"
        )


@dataclass(frozen=True, slots=True)
class DatabaseWorkerDispatchResult:
    job_type: str
    job_id: str
    state: str
    attempt_count: int
    reason_code: str | None = None


@dataclass(frozen=True, slots=True)
class DatabaseWorkerReadiness:
    ready: bool
    reason_code: str | None
    healthy_workers: int
    total_capacity: int
    active_count: int
    pending_count: int
    maximum_pending: int


class JobDispatcher(Protocol):
    def dispatch(
        self, candidate: DatabaseWorkerCandidate
    ) -> DatabaseWorkerDispatchResult: ...

    def exhaust(
        self, candidate: DatabaseWorkerCandidate
    ) -> DatabaseWorkerDispatchResult: ...


class DatabaseWorkerRegistry:
    """Register heartbeat-only worker state and assess automatic readiness."""

    def __init__(
        self,
        engine: Engine,
        *,
        heartbeat_ttl: timedelta,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        if heartbeat_ttl < timedelta(seconds=5) or heartbeat_ttl > timedelta(minutes=5):
            raise DatabaseWorkerError("worker_configuration_invalid")
        self.engine = engine
        self.heartbeat_ttl = heartbeat_ttl
        self.clock = clock

    @staticmethod
    def _support_column(job_type: str):
        columns = {
            "corpus_ingestion": baseline_worker_instance.c.supports_corpus_ingestion,
            "index_build": baseline_worker_instance.c.supports_index_build,
            "baseline_run": baseline_worker_instance.c.supports_baseline_run,
            "cleanup": baseline_worker_instance.c.supports_cleanup,
        }
        try:
            return columns[job_type]
        except KeyError:
            raise DatabaseWorkerError("worker_job_type_invalid") from None

    def register(
        self,
        worker_instance_id: str,
        *,
        concurrency_limit: int = 1,
    ) -> None:
        worker_instance_id = _uuid(worker_instance_id, "worker_instance")
        if not 1 <= concurrency_limit <= 64:
            raise DatabaseWorkerError("worker_capacity_invalid")
        now = self.clock()
        cutoff = now - self.heartbeat_ttl

        def write(connection: Any) -> None:
            connection.execute(
                delete(baseline_worker_instance).where(
                    baseline_worker_instance.c.last_heartbeat_at < cutoff
                )
            )
            if connection.dialect.name == "sqlite":
                other = connection.execute(
                    select(baseline_worker_instance.c.worker_instance_id).where(
                        baseline_worker_instance.c.worker_instance_id
                        != worker_instance_id,
                        baseline_worker_instance.c.last_heartbeat_at >= cutoff,
                        or_(
                            baseline_worker_instance.c.draining.is_(False),
                            baseline_worker_instance.c.active_count > 0,
                        ),
                    )
                ).first()
                if other is not None:
                    raise DatabaseWorkerError("worker_capacity_unavailable")
            existing = connection.execute(
                select(baseline_worker_instance.c.worker_instance_id).where(
                    baseline_worker_instance.c.worker_instance_id == worker_instance_id
                )
            ).first()
            values = {
                "worker_contract_version": DATABASE_WORKER_CONTRACT_VERSION,
                "supports_corpus_ingestion": True,
                "supports_index_build": True,
                "supports_baseline_run": True,
                "supports_cleanup": True,
                "started_at": now,
                "last_heartbeat_at": now,
                "draining": False,
                "concurrency_limit": concurrency_limit,
                "active_count": 0,
            }
            if existing is None:
                connection.execute(
                    baseline_worker_instance.insert().values(
                        worker_instance_id=worker_instance_id,
                        **values,
                    )
                )
            else:
                connection.execute(
                    update(baseline_worker_instance)
                    .where(
                        baseline_worker_instance.c.worker_instance_id
                        == worker_instance_id
                    )
                    .values(**values)
                )

        if self.engine.dialect.name == "sqlite":
            connection = self.engine.connect()
            try:
                connection.exec_driver_sql("BEGIN IMMEDIATE")
                write(connection)
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()
        else:
            try:
                with self.engine.begin() as connection:
                    write(connection)
            except IntegrityError:
                raise DatabaseWorkerError(
                    "worker_registration_conflict", retryable=True
                ) from None

    def heartbeat(
        self,
        worker_instance_id: str,
        *,
        active_count: int,
        draining: bool,
    ) -> None:
        worker_instance_id = _uuid(worker_instance_id, "worker_instance")
        if active_count not in {0, 1}:
            raise DatabaseWorkerError("worker_capacity_invalid")
        with self.engine.begin() as connection:
            changed = connection.execute(
                update(baseline_worker_instance)
                .where(
                    baseline_worker_instance.c.worker_instance_id == worker_instance_id,
                    baseline_worker_instance.c.worker_contract_version
                    == DATABASE_WORKER_CONTRACT_VERSION,
                )
                .values(
                    last_heartbeat_at=self.clock(),
                    active_count=active_count,
                    draining=draining,
                )
            )
            if changed.rowcount != 1:
                raise DatabaseWorkerError("worker_registration_missing")

    def cleanup_expired(self) -> int:
        cutoff = self.clock() - self.heartbeat_ttl
        with self.engine.begin() as connection:
            removed = connection.execute(
                delete(baseline_worker_instance).where(
                    baseline_worker_instance.c.last_heartbeat_at < cutoff
                )
            )
            return int(removed.rowcount or 0)

    def _schema_ready(self) -> bool:
        try:
            if BASELINE_WORKER_INSTANCE_TABLE not in set(
                inspect(self.engine).get_table_names()
            ):
                return False
            migration = next(
                item
                for item in CORE_SCHEMA_MIGRATIONS
                if item.migration_id == DATABASE_WORKER_MIGRATION_ID
            )
            with self.engine.connect() as connection:
                row = (
                    connection.execute(
                        select(schema_migration_table).where(
                            schema_migration_table.c.migration_id
                            == DATABASE_WORKER_MIGRATION_ID
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                if (
                    row is None
                    or row["state"] != "applied"
                    or row["checksum"] != migration.checksum
                ):
                    return False
                migration.validate(connection)
                connection.exec_driver_sql("SELECT 1").scalar_one()
        except Exception:  # noqa: BLE001 - schema readiness is non-reflective
            return False
        return True

    def readiness(
        self,
        *,
        required_job_types: Iterable[str],
        pending_count: int,
        maximum_pending_per_slot: int,
    ) -> DatabaseWorkerReadiness:
        if not self._schema_ready():
            return DatabaseWorkerReadiness(
                False, "capability_unavailable", 0, 0, 0, pending_count, 0
            )
        if maximum_pending_per_slot < 1:
            return DatabaseWorkerReadiness(
                False, "capability_unavailable", 0, 0, 0, pending_count, 0
            )
        cutoff = self.clock() - self.heartbeat_ttl
        conditions = [
            baseline_worker_instance.c.worker_contract_version
            == DATABASE_WORKER_CONTRACT_VERSION,
            baseline_worker_instance.c.last_heartbeat_at >= cutoff,
            baseline_worker_instance.c.draining.is_(False),
        ]
        for job_type in required_job_types:
            conditions.append(self._support_column(job_type).is_(True))
        try:
            with self.engine.connect() as connection:
                rows = (
                    connection.execute(
                        select(baseline_worker_instance).where(*conditions)
                    )
                    .mappings()
                    .all()
                )
        except SQLAlchemyError:
            rows = []
        total = sum(int(row["concurrency_limit"]) for row in rows)
        active = sum(int(row["active_count"]) for row in rows)
        maximum = total * maximum_pending_per_slot
        ready = total > 0 and pending_count < maximum
        return DatabaseWorkerReadiness(
            ready=ready,
            reason_code=None if ready else "worker_unavailable",
            healthy_workers=len(rows),
            total_capacity=total,
            active_count=active,
            pending_count=pending_count,
            maximum_pending=maximum,
        )


class DatabaseJobScheduler:
    """Short-transaction, deadline-aware and starvation-resistant selector."""

    def __init__(
        self,
        engine: Engine,
        *,
        poll_interval_seconds: float,
        maximum_backoff_seconds: float,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.engine = engine
        self.poll_interval_seconds = poll_interval_seconds
        self.maximum_backoff_seconds = maximum_backoff_seconds
        self.clock = clock
        self._urgent_run_streak = 0

    @staticmethod
    def _claimable(table: Any, now: datetime):
        return table.c.state.in_({"queued", "retryable_failed"}) | (
            (table.c.state == "running")
            & (table.c.lease_expires_at.is_not(None))
            & (table.c.lease_expires_at <= now)
        )

    def _backoff_ready(self, candidate: DatabaseWorkerCandidate, now: datetime) -> bool:
        if candidate.state not in _RETRYABLE_STATES:
            return True
        exponent = max(0, min(candidate.attempt_count - 1, 16))
        delay = min(
            self.poll_interval_seconds * (2**exponent),
            self.maximum_backoff_seconds,
        )
        return _aware(candidate.updated_at) + timedelta(seconds=delay) <= now

    def _ingestion_candidates(self, connection: Any, now: datetime):
        statement = (
            select(snapshot_continuation_job)
            .where(self._claimable(snapshot_continuation_job, now))
            .order_by(
                snapshot_continuation_job.c.created_at,
                snapshot_continuation_job.c.continuation_job_id,
            )
            .limit(32)
        )
        if connection.dialect.name == "postgresql":
            statement = statement.with_for_update(skip_locked=True)
        for row in connection.execute(statement).mappings():
            yield DatabaseWorkerCandidate(
                "corpus_ingestion",
                str(row["continuation_job_id"]),
                str(row["group_id"]),
                str(row["state"]),
                int(row["attempt_count"]),
                _aware(row["created_at"]),
                _aware(row["updated_at"]),
            )

    def _index_candidates(self, connection: Any, now: datetime):
        statement = (
            select(control_job)
            .where(
                control_job.c.operation == "index_build",
                self._claimable(control_job, now),
            )
            .order_by(control_job.c.created_at, control_job.c.job_id)
            .limit(32)
        )
        if connection.dialect.name == "postgresql":
            statement = statement.with_for_update(skip_locked=True)
        for row in connection.execute(statement).mappings():
            yield DatabaseWorkerCandidate(
                "index_build",
                str(row["job_id"]),
                str(row["group_id"]),
                str(row["state"]),
                int(row["attempt_count"]),
                _aware(row["created_at"]),
                _aware(row["updated_at"]),
            )

    def _run_candidates(self, connection: Any, now: datetime):
        claimable = self._claimable(baseline_run_job, now) | (
            baseline_run_job.c.state == "references_persisted"
        )
        statement = (
            select(baseline_run_job)
            .where(claimable)
            .order_by(baseline_run_job.c.created_at, baseline_run_job.c.job_id)
            .limit(32)
        )
        if connection.dialect.name == "postgresql":
            statement = statement.with_for_update(skip_locked=True)
        for row in connection.execute(statement).mappings():
            attempt = max(
                int(row["attempt_count"]),
                int(row["generation_attempt_count"]),
            )
            yield DatabaseWorkerCandidate(
                "baseline_run",
                str(row["job_id"]),
                str(row["group_id"]),
                str(row["state"]),
                attempt,
                _aware(row["created_at"]),
                _aware(row["updated_at"]),
                _aware(row["payload_expires_at"]),
            )

    def select(self) -> DatabaseWorkerCandidate | None:
        now = self.clock()
        candidates: list[DatabaseWorkerCandidate] = []
        with self.engine.begin() as connection:
            for source in (
                self._ingestion_candidates,
                self._index_candidates,
                self._run_candidates,
            ):
                first = next(
                    (
                        item
                        for item in source(connection, now)
                        if self._backoff_ready(item, now)
                    ),
                    None,
                )
                if first is not None:
                    candidates.append(first)
        if not candidates:
            return None
        urgent = [
            item
            for item in candidates
            if item.job_type == "baseline_run"
            and item.state != "references_persisted"
            and item.payload_expires_at is not None
            and item.payload_expires_at <= now + RUN_URGENCY_WINDOW
        ]
        order = {"corpus_ingestion": 0, "index_build": 1, "baseline_run": 2}
        non_run = [item for item in candidates if item.job_type != "baseline_run"]
        if urgent and (
            self._urgent_run_streak < MAXIMUM_URGENT_RUN_BURST or not non_run
        ):
            self._urgent_run_streak += 1
            return min(
                urgent,
                key=lambda item: (item.payload_expires_at, item.job_id),
            )
        pool = non_run or candidates
        self._urgent_run_streak = 0
        return min(
            pool,
            key=lambda item: (
                item.created_at,
                order[item.job_type],
                item.job_id,
            ),
        )

    def pending_count(self) -> int:
        with self.engine.connect() as connection:
            ingestion = connection.scalar(
                select(func.count())
                .select_from(snapshot_continuation_job)
                .where(
                    snapshot_continuation_job.c.state.in_(
                        {"queued", "running", "retryable_failed"}
                    )
                )
            )
            index = connection.scalar(
                select(func.count())
                .select_from(control_job)
                .where(
                    control_job.c.operation == "index_build",
                    control_job.c.state.in_({"queued", "running", "retryable_failed"}),
                )
            )
            runs = connection.scalar(
                select(func.count())
                .select_from(baseline_run_job)
                .where(
                    baseline_run_job.c.state.in_(
                        {
                            "queued",
                            "running",
                            "references_persisted",
                            "retryable_failed",
                        }
                    ),
                )
            )
        return int(ingestion or 0) + int(index or 0) + int(runs or 0)


class ExistingServiceDispatcher:
    """Opaque-ID adapter over the three existing lease-owning services."""

    def __init__(
        self,
        engine: Engine,
        *,
        worker_instance_id: str,
        continuation_worker: BaselineContinuationWorker,
        index_service: BaselineCompatibleIndexJobService,
        run_operator: BaselineManualRunOperator,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        self.engine = engine
        self.worker_instance_id = _uuid(worker_instance_id, "worker_instance")
        self.continuation_worker = continuation_worker
        self.index_service = index_service
        self.run_operator = run_operator
        self.clock = clock
        self.continuation_identity = InternalContinuationWorkerIdentity.create(
            self.worker_instance_id
        )
        self.index_identity = InternalIndexWorkerIdentity.create(
            self.worker_instance_id
        )

    def dispatch(
        self, candidate: DatabaseWorkerCandidate
    ) -> DatabaseWorkerDispatchResult:
        if candidate.job_type == "corpus_ingestion":
            result = self.continuation_worker.execute(
                identity=self.continuation_identity,
                group_id=candidate.group_id,
                continuation_job_id=candidate.job_id,
            )
            return DatabaseWorkerDispatchResult(
                candidate.job_type,
                candidate.job_id,
                result.state,
                result.attempt_count,
            )
        if candidate.job_type == "index_build":
            result = self.index_service.execute(
                identity=self.index_identity,
                group_id=candidate.group_id,
                job_id=candidate.job_id,
            )
            return DatabaseWorkerDispatchResult(
                candidate.job_type,
                candidate.job_id,
                result.state,
                result.attempt_count,
            )
        if candidate.job_type == "baseline_run":
            result = self.run_operator.process(candidate.job_id)
            return DatabaseWorkerDispatchResult(
                candidate.job_type,
                candidate.job_id,
                result.state,
                candidate.attempt_count + (0 if result.replayed else 1),
                result.reason_code,
            )
        raise DatabaseWorkerError("worker_job_type_invalid")

    def exhaust(
        self, candidate: DatabaseWorkerCandidate
    ) -> DatabaseWorkerDispatchResult:
        """Move an already retryable job to its existing terminal state."""

        if candidate.state != "retryable_failed":
            return self.dispatch(candidate)
        now = self.clock()
        reason = "worker_unavailable"
        fingerprint = hashlib.sha256(b"worker:attempts_exhausted").hexdigest()
        with self.engine.begin() as connection:
            if candidate.job_type == "corpus_ingestion":
                changed = connection.execute(
                    update(snapshot_continuation_job)
                    .where(
                        snapshot_continuation_job.c.continuation_job_id
                        == candidate.job_id,
                        snapshot_continuation_job.c.group_id == candidate.group_id,
                        snapshot_continuation_job.c.state == "retryable_failed",
                    )
                    .values(
                        state="terminal_failed",
                        error_code=reason,
                        error_fingerprint=fingerprint,
                        updated_at=now,
                        finished_at=now,
                    )
                )
            elif candidate.job_type == "index_build":
                changed = connection.execute(
                    update(control_job)
                    .where(
                        control_job.c.job_id == candidate.job_id,
                        control_job.c.group_id == candidate.group_id,
                        control_job.c.operation == "index_build",
                        control_job.c.state == "retryable_failed",
                    )
                    .values(
                        state="terminal_failed",
                        error_code=reason,
                        error_fingerprint=fingerprint,
                        updated_at=now,
                        finished_at=now,
                    )
                )
            elif candidate.job_type == "baseline_run":
                row = (
                    connection.execute(
                        select(baseline_run_job).where(
                            baseline_run_job.c.job_id == candidate.job_id,
                            baseline_run_job.c.group_id == candidate.group_id,
                            baseline_run_job.c.state == "retryable_failed",
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                if row is None:
                    changed = None
                else:
                    if row["persisted_run_id"] is not None:
                        connection.execute(
                            update(baseline_retrieval_run)
                            .where(
                                baseline_retrieval_run.c.run_id
                                == row["persisted_run_id"],
                                baseline_retrieval_run.c.generation_state
                                == "retryable_failed",
                            )
                            .values(
                                generation_state="terminal_failed",
                                generation_error_code=reason,
                            )
                        )
                    connection.execute(
                        delete(baseline_run_payload).where(
                            baseline_run_payload.c.job_id == candidate.job_id,
                            baseline_run_payload.c.group_id == candidate.group_id,
                        )
                    )
                    changed = connection.execute(
                        update(baseline_run_job)
                        .where(
                            baseline_run_job.c.job_id == candidate.job_id,
                            baseline_run_job.c.group_id == candidate.group_id,
                            baseline_run_job.c.state == "retryable_failed",
                        )
                        .values(
                            state="terminal_failed",
                            reason_code=reason,
                            failure_stage="dispatch",
                            updated_at=now,
                            finished_at=now,
                        )
                    )
            else:
                raise DatabaseWorkerError("worker_job_type_invalid")
            if changed is None or changed.rowcount != 1:
                raise DatabaseWorkerError("job_lease_unavailable", retryable=True)
        return DatabaseWorkerDispatchResult(
            candidate.job_type,
            candidate.job_id,
            "terminal_failed",
            candidate.attempt_count,
            reason,
        )


class BaselineDatabaseWorker:
    """One sequential worker process with managed heartbeat and cleanup."""

    def __init__(
        self,
        *,
        registry: DatabaseWorkerRegistry,
        scheduler: DatabaseJobScheduler,
        dispatcher: JobDispatcher,
        cleanup_callbacks: Iterable[Callable[[], int]],
        worker_instance_id: str | None = None,
        heartbeat_interval_seconds: float = 5.0,
        cleanup_interval_seconds: float = 30.0,
        maximum_attempts: int = 5,
        poll_interval_seconds: float = 2.0,
        maximum_backoff_seconds: float = 30.0,
        clock_monotonic: Callable[[], float] = monotonic,
    ) -> None:
        self.registry = registry
        self.scheduler = scheduler
        self.dispatcher = dispatcher
        self.cleanup_callbacks = tuple(cleanup_callbacks)
        self.worker_instance_id = _uuid(
            worker_instance_id or str(uuid4()), "worker_instance"
        )
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.maximum_attempts = maximum_attempts
        self.poll_interval_seconds = poll_interval_seconds
        self.maximum_backoff_seconds = maximum_backoff_seconds
        self.clock_monotonic = clock_monotonic
        self._state_lock = threading.Lock()
        self._active_count = 0
        self._draining = False
        self._registered = False
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._next_cleanup = 0.0

    def _snapshot_state(self) -> tuple[int, bool]:
        with self._state_lock:
            return self._active_count, self._draining

    def _set_state(
        self, *, active_count: int | None = None, draining: bool | None = None
    ) -> None:
        with self._state_lock:
            if active_count is not None:
                self._active_count = active_count
            if draining is not None:
                self._draining = draining

    def start(self) -> None:
        if self._registered:
            return
        self.registry.register(self.worker_instance_id, concurrency_limit=1)
        self._registered = True
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="compair-baseline-heartbeat",
            daemon=False,
        )
        self._heartbeat_thread.start()
        self._safe_log("started")

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(self.heartbeat_interval_seconds):
            active, draining = self._snapshot_state()
            try:
                self.registry.heartbeat(
                    self.worker_instance_id,
                    active_count=active,
                    draining=draining,
                )
            except DatabaseWorkerError:
                _LOGGER.error(
                    "baseline_worker event=heartbeat_failed worker_id=%s reason=%s",
                    self.worker_instance_id,
                    "worker_unavailable",
                )

    def begin_draining(self) -> None:
        if not self._registered:
            return
        self._set_state(draining=True)
        active, _ = self._snapshot_state()
        self.registry.heartbeat(
            self.worker_instance_id,
            active_count=active,
            draining=True,
        )
        self._safe_log("draining")

    def request_draining(self) -> None:
        """Signal-safe in-memory drain request; heartbeat performs the write."""

        self._set_state(draining=True)

    def close(self) -> None:
        if not self._registered:
            return
        self.begin_draining()
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(
                timeout=max(1.0, self.heartbeat_interval_seconds * 2)
            )
        active, _ = self._snapshot_state()
        self.registry.heartbeat(
            self.worker_instance_id,
            active_count=active,
            draining=True,
        )
        self._registered = False
        self._safe_log("stopped")

    def _safe_log(
        self,
        event: str,
        *,
        candidate: DatabaseWorkerCandidate | None = None,
        result: DatabaseWorkerDispatchResult | None = None,
        elapsed_ms: int | None = None,
    ) -> None:
        _LOGGER.info(
            "baseline_worker event=%s worker_id=%s job_id=%s job_type=%s "
            "state=%s attempt=%s elapsed_ms=%s reason=%s",
            event,
            self.worker_instance_id,
            candidate.job_id if candidate is not None else "none",
            candidate.job_type if candidate is not None else "none",
            result.state if result is not None else "none",
            result.attempt_count if result is not None else 0,
            elapsed_ms if elapsed_ms is not None else 0,
            result.reason_code if result is not None else "none",
        )

    def _cleanup_if_due(self, *, force: bool = False) -> int:
        now = self.clock_monotonic()
        if not force and now < self._next_cleanup:
            return 0
        total = 0
        for callback in self.cleanup_callbacks:
            try:
                total += int(callback())
            except Exception:  # noqa: BLE001 - cleanup failure stays sanitized
                _LOGGER.error(
                    "baseline_worker event=cleanup_failed worker_id=%s reason=%s",
                    self.worker_instance_id,
                    "worker_unavailable",
                )
        total += self.registry.cleanup_expired()
        self._next_cleanup = now + self.cleanup_interval_seconds
        return total

    def run_once(self) -> DatabaseWorkerDispatchResult | None:
        if not self._registered:
            raise DatabaseWorkerError("worker_registration_missing")
        _, draining = self._snapshot_state()
        if draining:
            return None
        self._cleanup_if_due()
        candidate = self.scheduler.select()
        if candidate is None:
            self.registry.heartbeat(
                self.worker_instance_id, active_count=0, draining=False
            )
            return None
        self._set_state(active_count=1)
        self.registry.heartbeat(self.worker_instance_id, active_count=1, draining=False)
        started = self.clock_monotonic()
        try:
            if (
                candidate.state == "retryable_failed"
                and candidate.attempt_count >= self.maximum_attempts
            ):
                result = self.dispatcher.exhaust(candidate)
            else:
                result = self.dispatcher.dispatch(candidate)
            self._safe_log(
                "processed",
                candidate=candidate,
                result=result,
                elapsed_ms=int((self.clock_monotonic() - started) * 1000),
            )
            return result
        except (ContinuationWorkerError, IndexJobError, BaselineRunRuntimeError) as exc:
            result = DatabaseWorkerDispatchResult(
                candidate.job_type,
                candidate.job_id,
                "retryable_failed" if getattr(exc, "retryable", False) else "failed",
                candidate.attempt_count + 1,
                getattr(exc, "code", "internal_failure"),
            )
            self._safe_log(
                "failed",
                candidate=candidate,
                result=result,
                elapsed_ms=int((self.clock_monotonic() - started) * 1000),
            )
            return result
        except Exception:  # noqa: BLE001 - scheduler boundary is non-reflective
            result = DatabaseWorkerDispatchResult(
                candidate.job_type,
                candidate.job_id,
                "retryable_failed",
                candidate.attempt_count + 1,
                "internal_failure",
            )
            self._safe_log(
                "failed",
                candidate=candidate,
                result=result,
                elapsed_ms=int((self.clock_monotonic() - started) * 1000),
            )
            return result
        finally:
            self._set_state(active_count=0)
            _, draining = self._snapshot_state()
            self.registry.heartbeat(
                self.worker_instance_id,
                active_count=0,
                draining=draining,
            )

    def poll(self, stop_event: threading.Event) -> None:
        failure_streak = 0
        while not stop_event.is_set():
            _, draining = self._snapshot_state()
            if draining:
                break
            result = self.run_once()
            if result is None:
                failure_streak = 0
                stop_event.wait(self.poll_interval_seconds)
                continue
            if result.state in {"retryable_failed", "failed"}:
                failure_streak = min(failure_streak + 1, 16)
                delay = min(
                    self.poll_interval_seconds * (2**failure_streak),
                    self.maximum_backoff_seconds,
                )
                stop_event.wait(delay)
            else:
                failure_streak = 0


def assess_database_worker_readiness(
    engine: Engine,
    settings: Any,
    *,
    required_job_types: Iterable[str],
) -> DatabaseWorkerReadiness:
    """Return safe automatic-dispatch readiness without constructing a worker."""

    try:
        registry = DatabaseWorkerRegistry(
            engine,
            heartbeat_ttl=timedelta(
                seconds=int(settings.baseline_worker_heartbeat_ttl_seconds)
            ),
        )
        scheduler = DatabaseJobScheduler(
            engine,
            poll_interval_seconds=float(settings.baseline_worker_poll_interval_seconds),
            maximum_backoff_seconds=float(settings.baseline_worker_max_backoff_seconds),
        )
        return registry.readiness(
            required_job_types=required_job_types,
            pending_count=scheduler.pending_count(),
            maximum_pending_per_slot=int(settings.baseline_worker_max_pending_per_slot),
        )
    except Exception:  # noqa: BLE001 - capability is intentionally non-reflective
        return DatabaseWorkerReadiness(False, "capability_unavailable", 0, 0, 0, 0, 0)


__all__ = [
    "DATABASE_WORKER_CONTRACT_VERSION",
    "DATABASE_WORKER_MIGRATION_ID",
    "DATABASE_WORKER_SUPPORTED_JOB_TYPES",
    "BaselineDatabaseWorker",
    "DatabaseJobScheduler",
    "DatabaseWorkerCandidate",
    "DatabaseWorkerDispatchResult",
    "DatabaseWorkerError",
    "DatabaseWorkerReadiness",
    "DatabaseWorkerRegistry",
    "ExistingServiceDispatcher",
    "assess_database_worker_readiness",
]
