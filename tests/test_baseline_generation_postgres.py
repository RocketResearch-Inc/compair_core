"""Real PostgreSQL coverage for the baseline generation lease/Feedback bridge.

Run with::

    COMPAIR_TEST_POSTGRES_URL=postgresql+psycopg2://... \
      pytest -q tests/test_baseline_generation_postgres.py
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import text
from test_baseline_evidence_persistence import make_persistence_environment
from test_baseline_generation import (
    CapturingProvider,
    _feedback_rows,
    _persist,
)

from compair_core import db as core_db
from compair_core.baseline_generation.profile import (
    CPU_GENERATION_TIMEOUT_SECONDS,
    required_generation_lease_seconds,
)
from compair_core.compair.retrieval.generation import (
    BaselineGenerationBusyError,
    BaselineGenerationError,
    BaselineGenerationService,
    BaselineGenerationState,
    GenerationWriteStage,
)

POSTGRES_URL = os.getenv("COMPAIR_TEST_POSTGRES_URL")


@pytest.fixture
def postgres_generation_environment():
    if not POSTGRES_URL:
        pytest.skip(
            "set COMPAIR_TEST_POSTGRES_URL for real PostgreSQL generation tests"
        )
    schema_name = f"baseline_generation_{uuid4().hex}"
    admin_engine = core_db.create_engine(POSTGRES_URL, pool_pre_ping=True)
    if admin_engine.dialect.name != "postgresql":
        pytest.fail("COMPAIR_TEST_POSTGRES_URL must select PostgreSQL")
    with admin_engine.begin() as connection:
        connection.exec_driver_sql(f'CREATE SCHEMA "{schema_name}"')
    scoped_engine = core_db.create_engine(
        POSTGRES_URL,
        pool_pre_ping=True,
        connect_args={"options": f"-csearch_path={schema_name}"},
    )
    try:
        yield make_persistence_environment(scoped_engine)
    finally:
        scoped_engine.dispose()
        with admin_engine.begin() as connection:
            connection.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
        admin_engine.dispose()


def test_postgres_ordered_success_replay_restart_and_no_notification(
    postgres_generation_environment,
) -> None:
    environment = postgres_generation_environment
    persisted, command = _persist(environment, "postgres-success")
    provider = CapturingProvider("postgres finding")

    first = BaselineGenerationService(environment.sessions).generate(command, provider)
    replay = BaselineGenerationService(environment.sessions).generate(command, provider)

    assert first.state is BaselineGenerationState.SUCCEEDED
    assert replay.replayed is True
    assert replay.feedback_ids == first.feedback_ids
    assert [item.ordinal for item in provider.inputs[0].evidence] == [1, 2, 3, 4]
    with environment.engine.connect() as connection:
        stored_outputs = (
            connection.execute(
                text(
                    "SELECT renderer_output FROM baseline_selected_evidence "
                    "WHERE run_id = :run_id ORDER BY ordinal"
                ),
                {"run_id": persisted.run_id},
            )
            .scalars()
            .all()
        )
        assert (
            connection.execute(
                text("SELECT count(*) FROM notification_event")
            ).scalar_one()
            == 0
        )
    assert [
        item.renderer_output for item in provider.inputs[0].evidence
    ] == stored_outputs
    assert len(_feedback_rows(environment, persisted.run_id)) == 1

    environment.engine.dispose()
    restarted = BaselineGenerationService(environment.sessions).generate(
        command, provider
    )
    assert restarted.replayed is True
    assert restarted.feedback_ids == first.feedback_ids


def test_postgres_cpu_timeout_lease_covers_provider_and_commit_margin(
    postgres_generation_environment,
) -> None:
    environment = postgres_generation_environment
    persisted, command = _persist(environment, "postgres-cpu-timeout")

    class InspectingProvider(CapturingProvider):
        remaining: timedelta | None = None

        def generate(self, generation_input, *, idempotency_key: str) -> str:
            with environment.engine.connect() as connection:
                expiry = connection.execute(
                    text(
                        "SELECT generation_lease_expires_at "
                        "FROM baseline_retrieval_run WHERE run_id = :run_id"
                    ),
                    {"run_id": persisted.run_id},
                ).scalar_one()
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            self.remaining = expiry - datetime.now(timezone.utc)
            return super().generate(
                generation_input,
                idempotency_key=idempotency_key,
            )

    provider = InspectingProvider("postgres lease-safe finding")
    service = BaselineGenerationService(
        environment.sessions,
        lease_seconds=required_generation_lease_seconds(CPU_GENERATION_TIMEOUT_SECONDS),
        provider_timeout_seconds=CPU_GENERATION_TIMEOUT_SECONDS,
    )
    receipt = service.generate(command, provider)
    assert receipt.state is BaselineGenerationState.SUCCEEDED
    assert provider.remaining is not None
    assert provider.remaining > timedelta(seconds=350)


def test_postgres_feedback_rollback_is_retryable(
    postgres_generation_environment,
) -> None:
    environment = postgres_generation_environment
    persisted, command = _persist(environment, "postgres-rollback")

    def fail(stage: GenerationWriteStage) -> None:
        if stage is GenerationWriteStage.STATE:
            raise RuntimeError("injected PostgreSQL transaction failure")

    with pytest.raises(BaselineGenerationError) as error:
        BaselineGenerationService(environment.sessions, stage_hook=fail).generate(
            command, CapturingProvider()
        )
    assert error.value.code == "database_commit_failed"
    assert _feedback_rows(environment, persisted.run_id) == []

    recovered = BaselineGenerationService(environment.sessions).generate(
        command, CapturingProvider("postgres retry")
    )
    assert recovered.state is BaselineGenerationState.SUCCEEDED
    assert recovered.attempt_count == 2
    assert len(_feedback_rows(environment, persisted.run_id)) == 1


def test_postgres_concurrent_attempts_observe_row_lease(
    postgres_generation_environment,
) -> None:
    environment = postgres_generation_environment
    persisted, command = _persist(environment, "postgres-concurrent")
    started = threading.Event()
    release = threading.Event()

    class BlockingProvider(CapturingProvider):
        def generate(self, generation_input, *, idempotency_key: str) -> str:
            self.inputs.append(generation_input)
            started.set()
            assert release.wait(timeout=15)
            return self.output

    provider = BlockingProvider("postgres concurrent")
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(
            BaselineGenerationService(environment.sessions).generate,
            command,
            provider,
        )
        assert started.wait(timeout=15)
        with pytest.raises(BaselineGenerationBusyError):
            BaselineGenerationService(environment.sessions).generate(
                command, CapturingProvider()
            )
        release.set()
        receipt = future.result(timeout=20)
    assert receipt.state is BaselineGenerationState.SUCCEEDED
    assert len(_feedback_rows(environment, persisted.run_id)) == 1


def test_postgres_authorization_revocation_before_commit_blocks_feedback(
    postgres_generation_environment,
) -> None:
    environment = postgres_generation_environment
    persisted, command = _persist(environment, "postgres-authorization")

    class RevokingProvider(CapturingProvider):
        def generate(self, generation_input, *, idempotency_key: str) -> str:
            output = super().generate(generation_input, idempotency_key=idempotency_key)
            with environment.engine.begin() as connection:
                connection.execute(
                    text(
                        "DELETE FROM user_to_group WHERE user_id = :user_id "
                        "AND group_id = :group_id"
                    ),
                    {
                        "user_id": command.caller_user_id,
                        "group_id": command.group_id,
                    },
                )
            return output

    receipt = BaselineGenerationService(environment.sessions).generate(
        command, RevokingProvider()
    )
    assert receipt.state is BaselineGenerationState.BLOCKED
    assert receipt.error_code == "generation_authorization_revoked"
    assert _feedback_rows(environment, persisted.run_id) == []
