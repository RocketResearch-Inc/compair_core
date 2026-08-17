from __future__ import annotations

import hashlib
import inspect as pyinspect
import logging
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import sessionmaker
from test_baseline_control_generation import _structured
from test_baseline_control_plane import (
    ControlEnvironment,
    _continuation_worker,
    _stage_worker_snapshot,
)
from test_baseline_control_plane import (
    environment as _environment_fixture,  # noqa: F401
)
from test_baseline_generation import CapturingProvider, RawOutputProvider
from test_baseline_index_continuation import (
    FixtureAdapter,
)
from test_baseline_index_continuation import (
    _payload as _index_payload,
)
from test_baseline_index_continuation import (
    _service as _index_service,
)
from test_baseline_run_api import _manual_operator
from test_baseline_run_jobs import RAW_QUERY, _keyring

from compair_core import worker as worker_module
from compair_core.baseline_control_plane_schema import (
    baseline_run_job,
    baseline_run_payload,
    baseline_worker_instance,
    compatible_index_job,
)
from compair_core.compair.retrieval import database_worker as database_worker_module
from compair_core.compair.retrieval.control_plane_v2 import (
    PROTOCOL_V2_SHA256,
    PROTOCOL_V2_VERSION,
    parse_run_submission,
)
from compair_core.compair.retrieval.corpus import RetrievalBaselineIndexBuild
from compair_core.compair.retrieval.database_worker import (
    DATABASE_WORKER_CONTRACT_VERSION,
    DATABASE_WORKER_MIGRATION_ID,
    MAXIMUM_URGENT_RUN_BURST,
    BaselineDatabaseWorker,
    DatabaseJobScheduler,
    DatabaseWorkerCandidate,
    DatabaseWorkerDispatchResult,
    DatabaseWorkerError,
    DatabaseWorkerRegistry,
    ExistingServiceDispatcher,
    assess_database_worker_readiness,
)
from compair_core.compair.retrieval.persistent import published_index_fingerprint
from compair_core.compair.retrieval.run_jobs import BaselineRunJobService
from compair_core.schema_migrations import read_schema_migration_state
from compair_core.server.settings import Settings
from compair_core.worker import main as worker_main


@pytest.fixture
def environment(request: pytest.FixtureRequest):
    return request.getfixturevalue("_environment_fixture")


class FixedScheduler:
    def __init__(self, candidates=()) -> None:
        self.candidates = list(candidates)
        self.calls = 0

    def select(self):
        self.calls += 1
        return self.candidates.pop(0) if self.candidates else None

    def pending_count(self) -> int:
        return len(self.candidates)


class RecordingDispatcher:
    def __init__(self, *, state: str = "succeeded") -> None:
        self.state = state
        self.candidates: list[DatabaseWorkerCandidate] = []
        self.exhausted: list[DatabaseWorkerCandidate] = []

    def dispatch(self, candidate: DatabaseWorkerCandidate):
        self.candidates.append(candidate)
        return DatabaseWorkerDispatchResult(
            candidate.job_type,
            candidate.job_id,
            self.state,
            candidate.attempt_count + 1,
        )

    def exhaust(self, candidate: DatabaseWorkerCandidate):
        self.exhausted.append(candidate)
        return DatabaseWorkerDispatchResult(
            candidate.job_type,
            candidate.job_id,
            "terminal_failed",
            candidate.attempt_count,
            "worker_unavailable",
        )


def _candidate(*, state: str = "queued", attempt: int = 0):
    now = datetime.now(timezone.utc)
    return DatabaseWorkerCandidate(
        "baseline_run",
        str(uuid4()),
        str(uuid4()),
        state,
        attempt,
        now,
        now,
        now + timedelta(minutes=5),
    )


def _typed_candidate(job_type: str, created_at: datetime, *, expires=None):
    return DatabaseWorkerCandidate(
        job_type,
        str(uuid4()),
        str(uuid4()),
        "queued",
        0,
        created_at,
        created_at,
        expires,
    )


def _registry(environment: ControlEnvironment, *, ttl: int = 30, clock=None):
    options = {}
    if clock is not None:
        options["clock"] = clock
    return DatabaseWorkerRegistry(
        environment.engine,
        heartbeat_ttl=timedelta(seconds=ttl),
        **options,
    )


def _worker(
    environment: ControlEnvironment,
    *,
    scheduler,
    dispatcher,
    cleanup=(),
    maximum_attempts: int = 5,
):
    return BaselineDatabaseWorker(
        registry=_registry(environment),
        scheduler=scheduler,
        dispatcher=dispatcher,
        cleanup_callbacks=cleanup,
        heartbeat_interval_seconds=0.05,
        cleanup_interval_seconds=1,
        maximum_attempts=maximum_attempts,
        poll_interval_seconds=0.01,
        maximum_backoff_seconds=1,
    )


def test_worker_settings_default_manual_and_entrypoint_surface() -> None:
    settings = Settings()
    assert settings.baseline_worker_mode == "manual"
    assert settings.baseline_worker_max_pending_per_slot == 8
    signature = pyinspect.signature(worker_main)
    assert "argv" in signature.parameters
    assert "worker_factory" in signature.parameters
    parser_actions = worker_module._parser()._actions
    lifecycle_options = {
        option
        for action in parser_actions
        for option in action.option_strings
        if option != "-h" and option != "--help"
    }
    assert lifecycle_options == {"--once", "--poll"}
    source = pyinspect.getsource(worker_module) + pyinspect.getsource(
        database_worker_module
    )
    assert "import celery" not in source.lower()
    assert "import redis" not in source.lower()


def test_migration_0013_owns_privacy_safe_heartbeat_table(
    environment: ControlEnvironment,
) -> None:
    state = read_schema_migration_state(environment.engine)
    assert state[-1].migration_id == DATABASE_WORKER_MIGRATION_ID
    assert state[-1].state == "applied"
    columns = {
        column["name"]
        for column in inspect(environment.engine).get_columns(
            "baseline_database_worker_instance"
        )
    }
    assert columns == {
        "worker_instance_id",
        "worker_contract_version",
        "supports_corpus_ingestion",
        "supports_index_build",
        "supports_baseline_run",
        "supports_cleanup",
        "started_at",
        "last_heartbeat_at",
        "draining",
        "concurrency_limit",
        "active_count",
    }
    assert (
        not {
            "hostname",
            "path",
            "endpoint",
            "credential",
            "lease_token",
            "payload",
            "query",
        }
        & columns
    )


def test_sqlite_single_worker_heartbeat_readiness_expiry_and_restart(
    environment: ControlEnvironment,
) -> None:
    now = [datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)]
    registry = _registry(environment, ttl=10, clock=lambda: now[0])
    first = str(uuid4())
    second = str(uuid4())
    registry.register(first)
    ready = registry.readiness(
        required_job_types=("baseline_run", "cleanup"),
        pending_count=0,
        maximum_pending_per_slot=8,
    )
    assert ready.ready and ready.total_capacity == 1
    with pytest.raises(DatabaseWorkerError, match="worker_capacity_unavailable"):
        registry.register(second)

    registry.heartbeat(first, active_count=0, draining=True)
    registry.register(second)
    with environment.engine.connect() as connection:
        row = (
            connection.execute(
                select(baseline_worker_instance).where(
                    baseline_worker_instance.c.worker_instance_id == second
                )
            )
            .mappings()
            .one()
        )
    assert row["worker_contract_version"] == DATABASE_WORKER_CONTRACT_VERSION

    now[0] += timedelta(seconds=11)
    expired = registry.readiness(
        required_job_types=("baseline_run",),
        pending_count=0,
        maximum_pending_per_slot=8,
    )
    assert not expired.ready and expired.reason_code == "worker_unavailable"
    assert registry.cleanup_expired() >= 1


def test_bounded_backpressure_is_fail_closed(
    environment: ControlEnvironment,
) -> None:
    registry = _registry(environment)
    registry.register(str(uuid4()))
    full = registry.readiness(
        required_job_types=("baseline_run", "cleanup"),
        pending_count=1,
        maximum_pending_per_slot=1,
    )
    assert not full.ready
    assert full.reason_code == "worker_unavailable"
    assert full.maximum_pending == 1


def test_scheduler_is_oldest_first_across_lanes_with_bounded_run_urgency(
    environment: ControlEnvironment,
) -> None:
    now = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)
    ingestion = _typed_candidate("corpus_ingestion", now - timedelta(minutes=3))
    index = _typed_candidate("index_build", now - timedelta(minutes=2))
    run = _typed_candidate(
        "baseline_run",
        now - timedelta(minutes=1),
        expires=now + timedelta(minutes=10),
    )

    class LaneScheduler(DatabaseJobScheduler):
        def _ingestion_candidates(self, _connection, _now):
            yield ingestion

        def _index_candidates(self, _connection, _now):
            yield index

        def _run_candidates(self, _connection, _now):
            yield run

    scheduler = LaneScheduler(
        environment.engine,
        poll_interval_seconds=1,
        maximum_backoff_seconds=30,
        clock=lambda: now,
    )
    assert scheduler.select() == ingestion

    urgent = DatabaseWorkerCandidate(
        run.job_type,
        run.job_id,
        run.group_id,
        run.state,
        run.attempt_count,
        run.created_at,
        run.updated_at,
        now + timedelta(seconds=30),
    )
    run = urgent
    assert scheduler.select() == urgent
    for _ in range(MAXIMUM_URGENT_RUN_BURST - 1):
        assert scheduler.select() == urgent
    assert scheduler.select() == ingestion


def test_scheduler_defers_retryable_jobs_with_bounded_exponential_backoff(
    environment: ControlEnvironment,
) -> None:
    now = [datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)]
    retryable = DatabaseWorkerCandidate(
        "corpus_ingestion",
        str(uuid4()),
        str(uuid4()),
        "retryable_failed",
        3,
        now[0] - timedelta(minutes=1),
        now[0],
    )

    class RetryScheduler(DatabaseJobScheduler):
        def _ingestion_candidates(self, _connection, _now):
            yield retryable

        def _index_candidates(self, _connection, _now):
            return iter(())

        def _run_candidates(self, _connection, _now):
            return iter(())

    scheduler = RetryScheduler(
        environment.engine,
        poll_interval_seconds=1,
        maximum_backoff_seconds=3,
        clock=lambda: now[0],
    )
    assert scheduler.select() is None
    now[0] += timedelta(seconds=3)
    assert scheduler.select() == retryable


def test_once_uses_opaque_candidate_invokes_cleanup_and_exits_draining(
    environment: ControlEnvironment,
    caplog: pytest.LogCaptureFixture,
) -> None:
    candidate = _candidate()
    dispatcher = RecordingDispatcher()
    cleaned: list[bool] = []
    worker = _worker(
        environment,
        scheduler=FixedScheduler([candidate]),
        dispatcher=dispatcher,
        cleanup=(lambda: cleaned.append(True) or 1,),
    )
    caplog.set_level(logging.INFO)
    worker.start()
    try:
        result = worker.run_once()
    finally:
        worker.close()
    assert result is not None and result.state == "succeeded"
    assert dispatcher.candidates == [candidate]
    assert cleaned == [True]
    with environment.engine.connect() as connection:
        heartbeat = (
            connection.execute(
                select(baseline_worker_instance).where(
                    baseline_worker_instance.c.worker_instance_id
                    == worker.worker_instance_id
                )
            )
            .mappings()
            .one()
        )
    assert heartbeat["draining"] is True
    assert heartbeat["active_count"] == 0
    assert candidate.job_id in caplog.text
    for forbidden in (
        "retrieval_query",
        "ciphertext",
        "nonce",
        "lease_token",
        "feedback text",
    ):
        assert forbidden not in caplog.text


def test_poll_graceful_shutdown_and_attempt_exhaustion(
    environment: ControlEnvironment,
) -> None:
    candidate = _candidate(state="retryable_failed", attempt=2)
    dispatcher = RecordingDispatcher()
    scheduler = FixedScheduler([candidate])
    worker = _worker(
        environment,
        scheduler=scheduler,
        dispatcher=dispatcher,
        maximum_attempts=2,
    )
    stop = threading.Event()
    worker.start()
    thread = threading.Thread(target=worker.poll, args=(stop,))
    thread.start()
    try:
        for _ in range(200):
            if dispatcher.exhausted:
                break
            stop.wait(0.005)
        worker.request_draining()
        stop.set()
        thread.join(timeout=2)
    finally:
        worker.close()
    assert not thread.is_alive()
    assert dispatcher.exhausted == [candidate]


def test_entrypoint_once_and_manual_mode_failure(
    environment: ControlEnvironment,
) -> None:
    idle = _worker(
        environment,
        scheduler=FixedScheduler(),
        dispatcher=RecordingDispatcher(),
    )
    assert worker_main(["--once"], worker_factory=lambda: idle) == 0

    def unavailable():
        raise DatabaseWorkerError("worker_mode_manual")

    assert worker_main(["--once"], worker_factory=unavailable) == 2


def test_readiness_helper_reports_missing_then_recent_worker(
    environment: ControlEnvironment,
) -> None:
    settings = SimpleNamespace(
        baseline_worker_heartbeat_ttl_seconds=30,
        baseline_worker_poll_interval_seconds=1.0,
        baseline_worker_max_backoff_seconds=10.0,
        baseline_worker_max_pending_per_slot=8,
    )
    missing = assess_database_worker_readiness(
        environment.engine,
        settings,
        required_job_types=("index_build",),
    )
    assert not missing.ready
    _registry(environment).register(str(uuid4()))
    ready = assess_database_worker_readiness(
        environment.engine,
        settings,
        required_job_types=("index_build",),
    )
    assert ready.ready


def _real_worker(
    environment: ControlEnvironment,
    provider,
):
    index_service = _index_service(environment, FixtureAdapter())
    operator, recording = _manual_operator(environment, provider)
    registry = _registry(environment)
    scheduler = DatabaseJobScheduler(
        environment.engine,
        poll_interval_seconds=0.01,
        maximum_backoff_seconds=1.0,
    )
    worker_instance_id = str(uuid4())
    worker = BaselineDatabaseWorker(
        registry=registry,
        scheduler=scheduler,
        dispatcher=ExistingServiceDispatcher(
            environment.engine,
            worker_instance_id=worker_instance_id,
            continuation_worker=_continuation_worker(environment),
            index_service=index_service,
            run_operator=operator,
        ),
        cleanup_callbacks=(
            environment.service.expire_staging_sessions,
            lambda: 0,
        ),
        worker_instance_id=worker_instance_id,
        heartbeat_interval_seconds=0.05,
        cleanup_interval_seconds=1,
        maximum_attempts=5,
        poll_interval_seconds=0.01,
        maximum_backoff_seconds=1,
    )
    return worker, index_service, recording


def _run_submission_for_index(
    environment: ControlEnvironment,
    *,
    index_job_id: str,
    idempotency_key: str,
):
    sessions = sessionmaker(environment.engine, expire_on_commit=False)
    with sessions() as session:
        extension = (
            session.execute(
                select(compatible_index_job).where(
                    compatible_index_job.c.job_id == index_job_id
                )
            )
            .mappings()
            .one()
        )
        build = session.get(
            RetrievalBaselineIndexBuild,
            str(extension["result_index_id"]),
        )
        assert build is not None
        fingerprint = published_index_fingerprint(build)
    encoded = RAW_QUERY.encode("utf-8")
    return {
        "protocol_version": PROTOCOL_V2_VERSION,
        "protocol_sha256": PROTOCOL_V2_SHA256,
        "message_type": "run_submit",
        "request_id": str(uuid4()),
        "group_id": environment.group_id,
        "idempotency_key": idempotency_key,
        "source_document_id": environment.source_document_id,
        "changed_repository_registration_id": environment.changed_repository_id,
        "index_publication": {
            "index_publication_id": str(extension["result_index_id"]),
            "corpus_generation_id": str(extension["generation_id"]),
            "corpus_manifest_hash": str(extension["corpus_manifest_hash"]),
            "index_format_version": str(extension["index_format_version"]),
            "tokenizer_version": str(extension["tokenizer_version"]),
            "retrieval_config_fingerprint": str(
                extension["retrieval_config_fingerprint"]
            ),
            "embedding_fingerprint": str(extension["embedding_fingerprint"]),
            "index_fingerprint": fingerprint,
        },
        "retrieval_query": {
            "representation": "raw_git_diff_v1",
            "origin": "explicit",
            "encoding": "utf-8",
            "base_revision": "1" * 40,
            "head_revision": "2" * 40,
            "byte_size": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "text": RAW_QUERY,
        },
    }


@pytest.mark.parametrize("findings", [("automatic finding",), ()])
def test_automatic_ingestion_index_run_positive_and_zero_finding_completion(
    environment: ControlEnvironment,
    findings: tuple[str, ...],
) -> None:
    provider = (
        CapturingProvider(*findings)
        if findings
        else RawOutputProvider(_structured("no_findings", []))
    )
    worker, index_service, recording = _real_worker(environment, provider)
    worker.start()
    try:
        continuation_id = _stage_worker_snapshot(
            environment,
            content="automatic benign corpus\n",
            idempotency_key=(
                "opaque-automatic-ingestion-positive-0001"
                if findings
                else "opaque-automatic-ingestion-zero-0000001"
            ),
        )
        ingestion = worker.run_once()
        assert ingestion is not None
        assert ingestion.job_type == "corpus_ingestion"
        assert ingestion.job_id == continuation_id
        assert ingestion.state == "succeeded"

        index_payload = _index_payload(
            environment,
            idempotency_key=(
                "opaque-automatic-index-positive-000001"
                if findings
                else "opaque-automatic-index-zero-000000001"
            ),
        )
        accepted_index = index_service.submit(
            index_payload,
            caller_user_id=environment.user_id,
        )
        indexed = worker.run_once()
        assert indexed is not None
        assert indexed.job_type == "index_build"
        assert indexed.job_id == accepted_index["job_id"]
        assert indexed.state == "succeeded"

        run_payload = _run_submission_for_index(
            environment,
            index_job_id=str(accepted_index["job_id"]),
            idempotency_key=(
                "opaque-automatic-run-positive-0000001"
                if findings
                else "opaque-automatic-run-zero-0000000001"
            ),
        )
        run_service = BaselineRunJobService(
            environment.engine,
            _keyring(),
        )
        accepted_run = run_service.submit(
            parse_run_submission(run_payload),
            caller_user_id=environment.user_id,
        )
        completed = worker.run_once()
        assert completed is not None
        assert completed.job_type == "baseline_run"
        assert completed.job_id == accepted_run["job_id"]
        assert completed.state == "feedback_persisted"
    finally:
        worker.close()

    assert len(recording.requests) == 1
    assert recording.requests[0].retrieval_query == RAW_QUERY
    assert recording.results[0].fallback_engine is None
    with environment.engine.connect() as connection:
        job = (
            connection.execute(
                select(baseline_run_job).where(
                    baseline_run_job.c.job_id == accepted_run["job_id"]
                )
            )
            .mappings()
            .one()
        )
        payload_count = connection.execute(
            select(baseline_run_payload.c.job_id).where(
                baseline_run_payload.c.job_id == accepted_run["job_id"]
            )
        ).first()
        notification_count = connection.execute(
            text("SELECT count(*) FROM notification_event")
        ).scalar_one()
    assert job["state"] == "feedback_persisted"
    assert job["feedback_count"] == len(findings)
    assert job["generation_invoked"] is True
    assert payload_count is None
    assert notification_count == 0
    assert job["notification_outbox_count"] == (1 if findings else 0)
