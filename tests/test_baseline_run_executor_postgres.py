"""Real PostgreSQL coverage for the internal document-level run executor.

Run with::

    COMPAIR_TEST_POSTGRES_URL=postgresql+psycopg2://... \
      pytest -q tests/test_baseline_run_executor_postgres.py
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.orm import sessionmaker
from test_baseline_control_plane_postgres import (
    postgres_control_environment as _postgres_control_environment_fixture,  # noqa: F401
)
from test_baseline_index_continuation import FixtureAdapter
from test_baseline_run_jobs import _keyring, _run_payload, _service

from compair_core.baseline_control_plane_schema import (
    baseline_run_job,
    baseline_run_payload,
)
from compair_core.compair.retrieval.control_plane_v2 import parse_run_submission
from compair_core.compair.retrieval.evidence_persistence import (
    BaselineEvidencePersistenceService,
    PersistenceWriteStage,
)
from compair_core.compair.retrieval.persistent import PersistentBaselineV1Retriever
from compair_core.compair.retrieval.run_executor import (
    BaselineDocumentRunExecutor,
    BaselineRunExecutorError,
    InternalBaselineRunWorkerIdentity,
)
from compair_core.schema_migrations import read_schema_migration_state


@pytest.fixture
def postgres_control_environment(request: pytest.FixtureRequest):
    return request.getfixturevalue("_postgres_control_environment_fixture")


def _submit(environment, suffix: str) -> str:
    accepted = _service(environment).submit(
        parse_run_submission(
            _run_payload(
                environment,
                idempotency_key=f"opaque-postgres-executor-{suffix}-intent-000001",
            )
        ),
        caller_user_id=environment.user_id,
    )
    return str(accepted["job_id"])


def _retriever(environment):
    return PersistentBaselineV1Retriever(
        sessionmaker(environment.engine, expire_on_commit=False), FixtureAdapter()
    )


def _executor(environment, instance_id: str, **kwargs):
    return BaselineDocumentRunExecutor(
        environment.engine,
        identity=InternalBaselineRunWorkerIdentity.create(instance_id),
        keyring=_keyring(),
        retriever_factory=lambda: _retriever(environment),
        **kwargs,
    )


def test_postgres_concurrent_execution_restart_and_exactly_one_effect_set(
    postgres_control_environment,
) -> None:
    environment = postgres_control_environment
    assert read_schema_migration_state(environment.engine)[-1].migration_id == (
        "0013_baseline_database_worker_v1"
    )
    job_id = _submit(environment, "concurrent")
    barrier = threading.Barrier(2)

    def execute(ordinal: int):
        barrier.wait()
        try:
            return _executor(environment, f"postgres-runner-{ordinal}").execute(job_id)
        except BaselineRunExecutorError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        attempts = tuple(pool.map(execute, range(2)))
    outcomes = tuple(
        attempt for attempt in attempts if not isinstance(attempt, Exception)
    )
    errors = tuple(attempt for attempt in attempts if isinstance(attempt, Exception))
    assert outcomes
    assert all(outcome.state == "references_persisted" for outcome in outcomes)
    assert all(error.code == "job_lease_unavailable" for error in errors)

    replay = _executor(environment, "postgres-restart").execute(job_id)
    assert replay.state == "references_persisted"
    assert replay.replayed is True
    with environment.engine.connect() as connection:
        job = (
            connection.execute(
                select(baseline_run_job).where(baseline_run_job.c.job_id == job_id)
            )
            .mappings()
            .one()
        )
        assert job["persisted_run_id"] == replay.persisted_run_id
        assert (
            connection.execute(
                select(func.count())
                .select_from(baseline_run_payload)
                .where(baseline_run_payload.c.job_id == job_id)
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(
                text(
                    "SELECT count(*) FROM baseline_retrieval_run WHERE run_id = :run_id"
                ),
                {"run_id": replay.persisted_run_id},
            ).scalar_one()
            == 1
        )
        assert (
            connection.execute(
                text(
                    "SELECT count(*) FROM reference r JOIN baseline_selected_evidence s "
                    "ON s.selected_evidence_id = r.baseline_selected_evidence_id "
                    "WHERE s.run_id = :run_id"
                ),
                {"run_id": replay.persisted_run_id},
            ).scalar_one()
            == replay.reference_count
        )


def test_postgres_effect_transaction_rollback_retains_payload_for_retry(
    postgres_control_environment,
) -> None:
    environment = postgres_control_environment
    job_id = _submit(environment, "rollback")

    def persistence_factory():
        def fail(stage: PersistenceWriteStage) -> None:
            if stage is PersistenceWriteStage.PROTECTED_PAYLOAD:
                raise RuntimeError("injected postgres evidence transaction rollback")

        return BaselineEvidencePersistenceService(
            sessionmaker(environment.engine, expire_on_commit=False), stage_hook=fail
        )

    with pytest.raises(BaselineRunExecutorError) as caught:
        _executor(
            environment,
            "postgres-rollback-worker",
            persistence_factory=persistence_factory,
        ).execute(job_id)
    assert caught.value.state == "retryable_failed"
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT count(*) FROM baseline_retrieval_run")
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(
                select(func.count())
                .select_from(baseline_run_payload)
                .where(baseline_run_payload.c.job_id == job_id)
            ).scalar_one()
            == 1
        )
    recovered = _executor(environment, "postgres-recovery-worker").execute(job_id)
    assert recovered.state == "references_persisted"
    assert recovered.attempt_count == 2
