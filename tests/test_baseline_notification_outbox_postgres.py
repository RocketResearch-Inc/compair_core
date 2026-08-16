"""Real PostgreSQL coverage for the baseline notification outbox.

Run with::

    COMPAIR_TEST_POSTGRES_URL=postgresql+psycopg2://... \
      pytest -q tests/test_baseline_notification_outbox_postgres.py
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from sqlalchemy import text
from test_baseline_evidence_persistence import make_persistence_environment
from test_baseline_generation import CapturingProvider, _persist

from compair_core import db as core_db
from compair_core.compair.retrieval.generation import (
    BaselineGenerationError,
    BaselineGenerationService,
    BaselineGenerationState,
    GenerationWriteStage,
)
from compair_core.compair.retrieval.notification_outbox import (
    BASELINE_NOTIFICATION_CHANNEL,
    BaselineNotificationOutboxDispatcher,
    BaselineNotificationState,
)

POSTGRES_URL = os.getenv("COMPAIR_TEST_POSTGRES_URL")


@pytest.fixture
def postgres_notification_environment():
    if not POSTGRES_URL:
        pytest.skip("set COMPAIR_TEST_POSTGRES_URL for real PostgreSQL outbox tests")
    schema_name = f"baseline_notification_{uuid4().hex}"
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


class PostgresSink:
    channel = BASELINE_NOTIFICATION_CHANNEL
    supports_idempotency = False

    def __init__(self) -> None:
        self.deliveries = []

    def deliver(self, digest, *, idempotency_key: str) -> None:
        self.deliveries.append((digest, idempotency_key))


def _generate(environment, key: str, output: str):
    persisted, command = _persist(environment, key)
    receipt = BaselineGenerationService(
        environment.sessions, notifications_enabled=True
    ).generate(command, CapturingProvider(output))
    assert receipt.state is BaselineGenerationState.SUCCEEDED
    return persisted, command, receipt


def test_postgres_success_replay_dispatch_privacy_and_restart(
    postgres_notification_environment,
) -> None:
    environment = postgres_notification_environment
    persisted, command, generated = _generate(
        environment, "postgres-notify-success", "postgres benign finding"
    )
    replay = BaselineGenerationService(
        environment.sessions, notifications_enabled=True
    ).generate(command, CapturingProvider("must not run"))
    assert replay.replayed is True

    with environment.engine.connect() as connection:
        row = (
            connection.execute(text("SELECT * FROM baseline_notification_outbox"))
            .mappings()
            .one()
        )
        assert row["state"] == "pending"
        assert row["finding_count"] == 1
        serialized = json.dumps(dict(row), default=str)
        assert "alpha persistence query" not in serialized
        assert "postgres benign finding" not in serialized
        assert (
            connection.execute(
                text("SELECT count(*) FROM baseline_notification_outbox")
            ).scalar_one()
            == 1
        )

    sink = PostgresSink()
    delivered = BaselineNotificationOutboxDispatcher(
        environment.sessions, enabled=True
    ).dispatch_one(sink)
    assert delivered is not None
    assert delivered.state is BaselineNotificationState.DELIVERED
    digest, idempotency_key = sink.deliveries[0]
    assert digest.run_id == persisted.run_id
    assert digest.recipient_user_id == command.caller_user_id
    assert digest.findings[0].feedback_id == generated.feedback_ids[0]
    assert idempotency_key == digest.digest_key

    environment.engine.dispose()
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT state FROM baseline_notification_outbox")
            ).scalar_one()
            == "delivered"
        )


def test_postgres_skip_locked_dispatchers_deliver_distinct_rows_once(
    postgres_notification_environment,
) -> None:
    environment = postgres_notification_environment
    _generate(environment, "postgres-notify-one", "first postgres finding")
    _generate(environment, "postgres-notify-two", "second postgres finding")
    sinks = (PostgresSink(), PostgresSink())

    def dispatch(sink):
        return BaselineNotificationOutboxDispatcher(
            environment.sessions, enabled=True
        ).dispatch_one(sink)

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(dispatch, sinks))
    assert all(receipt is not None for receipt in receipts)
    assert {receipt.state for receipt in receipts if receipt is not None} == {
        BaselineNotificationState.DELIVERED
    }
    delivered_ids = {
        sink.deliveries[0][0].outbox_id for sink in sinks if sink.deliveries
    }
    assert len(delivered_ids) == 2
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                text(
                    "SELECT count(*) FROM baseline_notification_outbox "
                    "WHERE state = 'delivered'"
                )
            ).scalar_one()
            == 2
        )


def test_postgres_outbox_stage_failure_rolls_back_feedback_state_and_digest(
    postgres_notification_environment,
) -> None:
    environment = postgres_notification_environment
    persisted, command = _persist(environment, "postgres-notify-rollback")

    def fail(stage: GenerationWriteStage) -> None:
        if stage is GenerationWriteStage.OUTBOX:
            raise RuntimeError("private injected database detail")

    with pytest.raises(BaselineGenerationError) as error:
        BaselineGenerationService(
            environment.sessions,
            notifications_enabled=True,
            stage_hook=fail,
        ).generate(command, CapturingProvider("rolled back finding"))
    assert error.value.code == "database_commit_failed"
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                text(
                    "SELECT count(*) FROM feedback "
                    "WHERE baseline_retrieval_run_id = :run_id"
                ),
                {"run_id": persisted.run_id},
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(
                text(
                    "SELECT count(*) FROM baseline_notification_outbox "
                    "WHERE run_id = :run_id"
                ),
                {"run_id": persisted.run_id},
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(
                text(
                    "SELECT generation_state FROM baseline_retrieval_run "
                    "WHERE run_id = :run_id"
                ),
                {"run_id": persisted.run_id},
            ).scalar_one()
            == "retryable_failed"
        )


def test_postgres_revoked_recipient_is_suppressed_before_delivery(
    postgres_notification_environment,
) -> None:
    environment = postgres_notification_environment
    _persisted, command, _generated = _generate(
        environment, "postgres-notify-reauth", "postgres authorization finding"
    )
    with environment.engine.begin() as connection:
        connection.execute(
            text(
                "DELETE FROM user_to_group WHERE user_id = :user_id "
                "AND group_id = :group_id"
            ),
            {"user_id": command.caller_user_id, "group_id": command.group_id},
        )
    sink = PostgresSink()
    receipt = BaselineNotificationOutboxDispatcher(
        environment.sessions, enabled=True
    ).dispatch_one(sink)
    assert receipt is not None
    assert receipt.state is BaselineNotificationState.SUPPRESSED
    assert sink.deliveries == []
