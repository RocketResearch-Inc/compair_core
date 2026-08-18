"""Real PostgreSQL database-worker scheduling and completion coverage.

Run with::

    COMPAIR_TEST_POSTGRES_URL=postgresql+psycopg2://... \
      pytest -q tests/test_baseline_database_worker_postgres.py
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import select, update
from test_baseline_control_generation import _structured
from test_baseline_control_plane import _stage_worker_snapshot
from test_baseline_control_plane_postgres import (
    postgres_control_environment as _postgres_control_environment_fixture,  # noqa: F401
)
from test_baseline_database_worker import (
    _index_payload,
    _real_worker,
    _run_submission_for_index,
)
from test_baseline_generation import CapturingProvider, RawOutputProvider
from test_baseline_run_jobs import _keyring

from compair_core.baseline_control_plane_schema import (
    baseline_run_job,
    baseline_worker_attestation,
    baseline_worker_instance,
    snapshot_continuation_job,
)
from compair_core.compair.retrieval.control_plane_v2 import parse_run_submission
from compair_core.compair.retrieval.database_worker import (
    DatabaseJobScheduler,
    DatabaseWorkerAttestation,
    DatabaseWorkerError,
    DatabaseWorkerRegistry,
)
from compair_core.compair.retrieval.run_jobs import BaselineRunJobService
from compair_core.doctor import run_doctor
from compair_core.runtime_config import build_runtime_configuration
from compair_core.server.settings import Settings


@pytest.fixture
def postgres_control_environment(request: pytest.FixtureRequest):
    return request.getfixturevalue("_postgres_control_environment_fixture")


def test_postgres_runtime_attestation_requires_exact_match_before_work(
    postgres_control_environment,
) -> None:
    environment = postgres_control_environment
    runtime = build_runtime_configuration(
        SimpleNamespace(),
        database_url=environment.engine.url,
    )
    registry = DatabaseWorkerRegistry(
        environment.engine,
        heartbeat_ttl=timedelta(seconds=30),
        attestation=DatabaseWorkerAttestation.from_runtime(runtime),
    )
    worker_id = str(uuid4())
    registry.register(worker_id)
    ready = registry.readiness(
        required_job_types=("baseline_run", "cleanup"),
        pending_count=0,
        maximum_pending_per_slot=8,
        required_runtime_config_fingerprint=runtime.fingerprint,
    )
    assert ready.ready
    assert ready.mismatched_workers == 0

    with environment.engine.begin() as connection:
        connection.execute(
            update(baseline_worker_attestation)
            .where(baseline_worker_attestation.c.worker_instance_id == worker_id)
            .values(runtime_config_fingerprint="0" * 64)
        )
    mismatch = registry.readiness(
        required_job_types=("baseline_run", "cleanup"),
        pending_count=0,
        maximum_pending_per_slot=8,
        required_runtime_config_fingerprint=runtime.fingerprint,
    )
    assert not mismatch.ready
    assert mismatch.mismatched_workers == 1
    with pytest.raises(DatabaseWorkerError, match="worker_configuration_mismatch"):
        registry.heartbeat(worker_id, active_count=0, draining=False)


def test_postgres_doctor_reports_matching_safe_database_worker(
    postgres_control_environment,
) -> None:
    environment = postgres_control_environment
    settings = Settings(baseline_worker_mode="database")
    runtime = build_runtime_configuration(settings, database_url=environment.engine.url)
    registry = DatabaseWorkerRegistry(
        environment.engine,
        heartbeat_ttl=timedelta(seconds=30),
        attestation=DatabaseWorkerAttestation.from_runtime(runtime),
    )
    registry.register(str(uuid4()))
    result = run_doctor(settings=settings, engine=environment.engine)
    assert result.component("database").details["backend"] == "postgresql"
    assert result.component("migrations").reason_code == "migrations_current"
    assert result.component("worker").reason_code == "worker_ready"
    rendered = str(result.as_dict())
    assert "postgresql://" not in rendered
    assert "127.0.0.1" not in rendered


def test_postgres_multiple_workers_skip_locked_and_single_service_claim(
    postgres_control_environment,
) -> None:
    environment = postgres_control_environment
    first_id = _stage_worker_snapshot(
        environment,
        content="first postgres worker corpus\n",
        idempotency_key="opaque-postgres-worker-first-snapshot-0001",
    )
    second_id = _stage_worker_snapshot(
        environment,
        content="second postgres worker corpus\n",
        idempotency_key="opaque-postgres-worker-second-snapshot-0002",
    )
    scheduler = DatabaseJobScheduler(
        environment.engine,
        poll_interval_seconds=0.01,
        maximum_backoff_seconds=1,
    )
    with environment.engine.connect() as locking:
        transaction = locking.begin()
        oldest = (
            locking.execute(
                select(snapshot_continuation_job.c.continuation_job_id)
                .where(
                    snapshot_continuation_job.c.continuation_job_id.in_(
                        {first_id, second_id}
                    )
                )
                .order_by(
                    snapshot_continuation_job.c.created_at,
                    snapshot_continuation_job.c.continuation_job_id,
                )
                .limit(1)
                .with_for_update()
            )
            .scalars()
            .one()
        )
        selected = scheduler.select()
        transaction.rollback()
    assert selected is not None
    assert selected.job_type == "corpus_ingestion"
    assert selected.job_id in {first_id, second_id} - {str(oldest)}

    first_worker, _, _ = _real_worker(
        environment,
        CapturingProvider("unused postgres worker finding"),
    )
    second_worker, _, _ = _real_worker(
        environment,
        CapturingProvider("unused postgres worker finding"),
    )
    first_worker.start()
    second_worker.start()
    barrier = threading.Barrier(2)

    def execute(worker):
        barrier.wait()
        return worker.run_once()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(execute, (first_worker, second_worker)))
        first_worker.run_once()
    finally:
        first_worker.close()
        second_worker.close()

    assert any(item is not None for item in outcomes)
    with environment.engine.connect() as connection:
        rows = connection.execute(
            select(
                snapshot_continuation_job.c.state,
                snapshot_continuation_job.c.attempt_count,
            ).where(
                snapshot_continuation_job.c.continuation_job_id.in_(
                    {first_id, second_id}
                )
            )
        ).all()
        workers = connection.execute(select(baseline_worker_instance)).all()
    assert sum(state == "succeeded" for state, _attempts in rows) == 2
    assert all(attempts == 1 for _state, attempts in rows)
    assert len(workers) == 2


@pytest.mark.parametrize("findings", [("postgres automatic finding",), ()])
def test_postgres_automatic_ingestion_index_run_positive_and_zero_completion(
    postgres_control_environment,
    findings: tuple[str, ...],
) -> None:
    environment = postgres_control_environment
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
            content="benign postgres automatic corpus\n",
            idempotency_key=(
                "opaque-postgres-auto-ingest-positive-0001"
                if findings
                else "opaque-postgres-auto-ingest-zero-0000001"
            ),
        )
        ingestion = worker.run_once()
        assert ingestion is not None
        assert ingestion.job_id == continuation_id
        assert ingestion.state == "succeeded"

        index_payload = _index_payload(
            environment,
            idempotency_key=(
                "opaque-postgres-auto-index-positive-00001"
                if findings
                else "opaque-postgres-auto-index-zero-0000001"
            ),
        )
        accepted_index = index_service.submit(
            index_payload,
            caller_user_id=environment.user_id,
        )
        indexed = worker.run_once()
        assert indexed is not None
        assert indexed.job_id == accepted_index["job_id"]
        assert indexed.state == "succeeded"

        run_payload = _run_submission_for_index(
            environment,
            index_job_id=str(accepted_index["job_id"]),
            idempotency_key=(
                "opaque-postgres-auto-run-positive-000001"
                if findings
                else "opaque-postgres-auto-run-zero-000000001"
            ),
        )
        accepted_run = BaselineRunJobService(
            environment.engine,
            _keyring(),
        ).submit(
            parse_run_submission(run_payload),
            caller_user_id=environment.user_id,
        )
        completed = worker.run_once()
        assert completed is not None
        assert completed.job_id == accepted_run["job_id"]
        assert completed.state == "feedback_persisted"
    finally:
        worker.close()

    assert len(recording.requests) == 1
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
    assert job["feedback_count"] == len(findings)
    assert job["generation_invoked"] is True
    assert job["notification_outbox_count"] == (1 if findings else 0)
