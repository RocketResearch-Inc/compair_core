"""Real PostgreSQL coverage for coordinated document-control generation.

Run with::

    COMPAIR_TEST_POSTGRES_URL=postgresql+psycopg2://... \
      pytest -q tests/test_baseline_control_generation_postgres.py
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from sqlalchemy import text
from test_baseline_control_generation import (
    CoordinatedLeaseInspectingProvider,
    _persist_control,
    _state,
    _structured,
)
from test_baseline_generation import CapturingProvider, RawOutputProvider
from test_baseline_generation_postgres import (
    postgres_generation_environment as _postgres_generation_environment_fixture,  # noqa: F401
)

from compair_core.baseline_generation.profile import (
    CPU_GENERATION_TIMEOUT_SECONDS,
    required_generation_lease_seconds,
)
from compair_core.compair.retrieval.generation import (
    BaselineGenerationBusyError,
    BaselineGenerationError,
    BaselineGenerationService,
    GenerationWriteStage,
)
from compair_core.schema_migrations import read_schema_migration_state


@pytest.fixture
def postgres_generation_environment(request: pytest.FixtureRequest):
    return request.getfixturevalue("_postgres_generation_environment_fixture")


def test_postgres_control_positive_and_restart(
    postgres_generation_environment,
) -> None:
    environment = postgres_generation_environment
    assert read_schema_migration_state(environment.engine)[-1].migration_id == (
        "0014_baseline_worker_runtime_attestation_v1"
    )
    job_id, _caller, persisted = _persist_control(environment)
    provider = CapturingProvider("postgres first", "postgres second")
    first = BaselineGenerationService(environment.sessions).generate_control(
        job_id, provider
    )
    assert first.state == "feedback_persisted"
    environment.engine.dispose()
    replay = BaselineGenerationService(environment.sessions).generate_control(
        job_id, provider
    )
    assert replay.replayed is True
    assert replay.feedback_ids == first.feedback_ids
    assert len(provider.inputs) == 1
    job, run, feedback, outbox, notifications = _state(
        environment, job_id, persisted.run_id
    )
    assert job["feedback_count"] == 2
    assert run["generation_state"] == "succeeded"
    assert [row["baseline_finding_ordinal"] for row in feedback] == [1, 2]
    assert len(outbox) == 1
    assert notifications == 0


def test_postgres_control_and_generation_leases_cover_cpu_timeout(
    postgres_generation_environment,
) -> None:
    environment = postgres_generation_environment
    job_id, _caller, persisted = _persist_control(environment)
    provider = CoordinatedLeaseInspectingProvider(
        environment,
        job_id,
        persisted.run_id,
    )
    receipt = BaselineGenerationService(
        environment.sessions,
        lease_seconds=required_generation_lease_seconds(CPU_GENERATION_TIMEOUT_SECONDS),
        provider_timeout_seconds=CPU_GENERATION_TIMEOUT_SECONDS,
    ).generate_control(job_id, provider)
    assert receipt.state == "feedback_persisted"
    assert provider.control_remaining is not None
    assert provider.generation_remaining is not None
    assert provider.control_remaining > timedelta(seconds=350)
    assert provider.generation_remaining > timedelta(seconds=350)


def test_postgres_control_zero_findings(postgres_generation_environment) -> None:
    environment = postgres_generation_environment
    job_id, _caller, persisted = _persist_control(environment)
    zero = BaselineGenerationService(environment.sessions).generate_control(
        job_id, RawOutputProvider(_structured("no_findings", []))
    )
    assert zero.state == "feedback_persisted"
    job, run, feedback, outbox, _notifications = _state(
        environment, job_id, persisted.run_id
    )
    assert job["feedback_count"] == 0
    assert job["notification_outbox_count"] == 0
    assert run["generation_state"] == "succeeded"
    assert feedback == []
    assert outbox == []


@pytest.mark.parametrize("selected_stage", list(GenerationWriteStage))
def test_postgres_control_atomic_rollback_and_retry(
    postgres_generation_environment, selected_stage: GenerationWriteStage
) -> None:
    environment = postgres_generation_environment
    job_id, _caller, persisted = _persist_control(environment)

    def fail(stage: GenerationWriteStage) -> None:
        if stage is selected_stage:
            raise RuntimeError("injected PostgreSQL transaction failure")

    with pytest.raises(BaselineGenerationError) as error:
        BaselineGenerationService(
            environment.sessions, stage_hook=fail
        ).generate_control(job_id, CapturingProvider("rolled back"))
    assert error.value.code == "database_commit_failed"
    job, run, feedback, outbox, _notifications = _state(
        environment, job_id, persisted.run_id
    )
    assert job["state"] == "retryable_failed"
    assert run["generation_state"] == "retryable_failed"
    assert feedback == []
    assert outbox == []
    recovered = BaselineGenerationService(environment.sessions).generate_control(
        job_id, CapturingProvider("retried")
    )
    assert recovered.state == "feedback_persisted"


def test_postgres_control_concurrent_workers(
    postgres_generation_environment,
) -> None:
    environment = postgres_generation_environment
    job_id, _caller, persisted = _persist_control(environment)
    started = threading.Event()
    release = threading.Event()

    class BlockingProvider(CapturingProvider):
        def generate(self, generation_input, *, idempotency_key: str) -> str:
            self.inputs.append(generation_input)
            self.idempotency_keys.append(idempotency_key)
            started.set()
            assert release.wait(timeout=15)
            return self.output

    provider = BlockingProvider("single postgres finding")
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(
            BaselineGenerationService(environment.sessions).generate_control,
            job_id,
            provider,
        )
        assert started.wait(timeout=15)
        with pytest.raises(BaselineGenerationBusyError):
            BaselineGenerationService(environment.sessions).generate_control(
                job_id, CapturingProvider("duplicate")
            )
        release.set()
        assert future.result(timeout=20).state == "feedback_persisted"
    assert len(provider.inputs) == 1
    _job, _run, feedback, _outbox, _notifications = _state(
        environment, job_id, persisted.run_id
    )
    assert len(feedback) == 1


def test_postgres_control_authorization_revocation(
    postgres_generation_environment,
) -> None:
    environment = postgres_generation_environment
    revoked_job, caller, revoked_persisted = _persist_control(environment)

    class RevokingProvider(CapturingProvider):
        def generate(self, generation_input, *, idempotency_key: str) -> str:
            output = super().generate(generation_input, idempotency_key=idempotency_key)
            with environment.engine.begin() as connection:
                connection.execute(
                    text(
                        "DELETE FROM user_to_group WHERE user_id = :user_id "
                        "AND group_id = :group_id"
                    ),
                    {"user_id": caller, "group_id": environment.group_id},
                )
            return output

    blocked = BaselineGenerationService(environment.sessions).generate_control(
        revoked_job, RevokingProvider("must roll back")
    )
    assert blocked.state == "blocked"
    _job, run, feedback, outbox, _notifications = _state(
        environment, revoked_job, revoked_persisted.run_id
    )
    assert run["generation_state"] == "blocked"
    assert feedback == []
    assert outbox == []
