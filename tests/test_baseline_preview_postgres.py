"""Real PostgreSQL read coverage for ``baseline-preview.v1``.

Run with::

    COMPAIR_TEST_POSTGRES_URL=postgresql+psycopg2://... \\
      pytest -q tests/test_baseline_preview_postgres.py
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy import text
from test_baseline_control_generation import _persist_control, _structured
from test_baseline_evidence_persistence import make_persistence_environment
from test_baseline_generation import CapturingProvider, RawOutputProvider

from compair_core import db as core_db
from compair_core.compair.retrieval.generation import BaselineGenerationService
from compair_core.compair.retrieval.preview import (
    BaselinePreviewCommand,
    BaselinePreviewError,
    BaselinePreviewService,
)

POSTGRES_URL = os.getenv("COMPAIR_TEST_POSTGRES_URL")


@pytest.fixture
def postgres_preview_environment():
    if not POSTGRES_URL:
        pytest.skip("set COMPAIR_TEST_POSTGRES_URL for real PostgreSQL preview tests")
    schema_name = f"baseline_preview_{uuid4().hex}"
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


def _command(environment, job_id: str, caller: str, *, digest_id: str | None = None):
    return BaselinePreviewCommand(
        caller_user_id=caller,
        request_id=str(uuid4()),
        group_id=environment.group_id,
        job_id=None if digest_id else job_id,
        digest_id=digest_id,
    )


def test_postgres_positive_job_digest_restart_and_revocation(
    postgres_preview_environment,
) -> None:
    environment = postgres_preview_environment
    job_id, caller, persisted = _persist_control(environment)
    generated = BaselineGenerationService(
        environment.sessions,
        notifications_enabled=False,
    ).generate_control(
        job_id,
        CapturingProvider("postgres first", "postgres second"),
    )

    service = BaselinePreviewService(environment.sessions)
    preview = service.load(_command(environment, job_id, caller))
    assert [item.ordinal for item in preview.feedback] == [1, 2]
    assert [item.feedback_id for item in preview.feedback] == list(
        generated.feedback_ids
    )
    assert preview.source.source_scope == "control_document"
    assert preview.source.chunk_id is None
    assert preview.digest is not None
    assert preview.digest.state == "suppressed"

    by_digest = service.load(
        _command(
            environment,
            job_id,
            caller,
            digest_id=preview.digest.digest_id,
        )
    )
    assert by_digest.control_job == preview.control_job
    assert by_digest.retrieval.persisted_run_id == persisted.run_id

    environment.engine.dispose()
    restarted = BaselinePreviewService(environment.sessions).load(
        _command(environment, job_id, caller)
    )
    assert restarted.control_job == preview.control_job
    assert restarted.feedback == preview.feedback

    with environment.engine.begin() as connection:
        connection.execute(
            text(
                "DELETE FROM user_to_group WHERE user_id = :user_id "
                "AND group_id = :group_id"
            ),
            {"user_id": caller, "group_id": environment.group_id},
        )
    with pytest.raises(BaselinePreviewError) as error:
        service.load(_command(environment, job_id, caller))
    assert error.value.code == "baseline_preview_unavailable"


def test_postgres_zero_findings_is_readable_without_digest(
    postgres_preview_environment,
) -> None:
    environment = postgres_preview_environment
    job_id, caller, persisted = _persist_control(environment)
    receipt = BaselineGenerationService(
        environment.sessions,
        notifications_enabled=False,
    ).generate_control(
        job_id,
        RawOutputProvider(_structured("no_findings", [])),
    )
    assert receipt.feedback_ids == ()

    preview = BaselinePreviewService(environment.sessions).load(
        _command(environment, job_id, caller)
    )
    assert preview.retrieval.persisted_run_id == persisted.run_id
    assert preview.feedback == ()
    assert preview.digest is None
    assert preview.control_job.feedback_count == 0
    assert preview.control_job.notification_outbox_count == 0
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                text("SELECT count(*) FROM baseline_notification_outbox")
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(
                text("SELECT count(*) FROM notification_event")
            ).scalar_one()
            == 0
        )
