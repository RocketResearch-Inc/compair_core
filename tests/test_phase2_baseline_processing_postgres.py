"""Real PostgreSQL processing-path coverage for Phase 2B2H.

Set ``COMPAIR_TEST_POSTGRES_URL`` to a disposable PostgreSQL database. The test
uses and drops an isolated schema, and is skipped rather than emulated when the
database is unavailable.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest
from conftest import REAL_SQLALCHEMY_TEXT as text
from test_baseline_evidence_persistence import (
    make_persistence_environment,
    persistence_counts,
)
from test_phase2_baseline_processing import (
    _baseline_reference_rows,
    _install_actual_task_path,
    _run_task,
    run_in_isolated_pytest_if_needed,
)

from compair_core import db as core_db
from compair_core.compair.retrieval import new_processing_run_key

POSTGRES_URL = os.getenv("COMPAIR_TEST_POSTGRES_URL")


@pytest.mark.skipif(
    not POSTGRES_URL,
    reason="set COMPAIR_TEST_POSTGRES_URL to run real PostgreSQL baseline processing",
)
def test_postgres_actual_baseline_processing_and_retry(
    monkeypatch,
    request,
    tmp_path,
) -> None:
    if run_in_isolated_pytest_if_needed(request, tmp_path):
        return
    assert POSTGRES_URL is not None
    schema_name = f"baseline_processing_{uuid4().hex}"
    admin_engine = core_db.create_engine(POSTGRES_URL, pool_pre_ping=True)
    scoped_engine = None
    try:
        if admin_engine.dialect.name != "postgresql":
            pytest.fail("COMPAIR_TEST_POSTGRES_URL must select PostgreSQL")
        with admin_engine.begin() as connection:
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema_name}"')
        scoped_engine = core_db.create_engine(
            POSTGRES_URL,
            pool_pre_ping=True,
            connect_args={"options": f"-csearch_path={schema_name}"},
        )
        environment = make_persistence_environment(scoped_engine)
        generation, embedding, _events = _install_actual_task_path(
            monkeypatch, environment
        )
        parent_key = new_processing_run_key()

        first = _run_task(environment, parent_key)
        replay = _run_task(environment, parent_key)

        first_outcome = first["baseline_processing"]["outcomes"][0]
        replay_outcome = replay["baseline_processing"]["outcomes"][0]
        assert first_outcome["status"] == "references_persisted"
        assert first_outcome["group_id"] == environment.group_id
        assert first_outcome["idempotent_replay"] is False
        assert replay_outcome["group_id"] == environment.group_id
        assert replay_outcome["idempotent_replay"] is True
        assert persistence_counts(scoped_engine) == (1, 4, 4, 4, 0)
        assert [row.ordinal for row in _baseline_reference_rows(environment)] == [
            1,
            2,
            3,
            4,
        ]
        with scoped_engine.connect() as connection:
            assert (
                connection.execute(
                    text("SELECT count(*) FROM notification_event")
                ).scalar_one()
                == 0
            )
        generation.assert_not_called()
        embedding.assert_not_called()
    finally:
        if scoped_engine is not None:
            scoped_engine.dispose()
        with admin_engine.begin() as connection:
            connection.exec_driver_sql(
                f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'
            )
        admin_engine.dispose()
