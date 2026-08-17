"""Real PostgreSQL coverage for Phase 2B2G persistence.

Set ``COMPAIR_TEST_POSTGRES_URL`` to a disposable PostgreSQL database.  The
test creates and drops an isolated schema; it is skipped rather than emulated
when PostgreSQL is unavailable.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from uuid import uuid4

import pytest
from sqlalchemy import text
from test_baseline_evidence_persistence import (
    control_command,
    make_persistence_environment,
    persistence_counts,
    seed_running_control_job,
)
from test_baseline_generation import CapturingProvider

from compair_core import db as core_db
from compair_core.compair.retrieval.evidence_persistence import (
    BaselineEvidencePersistenceError,
    BaselineEvidencePersistenceService,
    PersistenceWriteStage,
)
from compair_core.compair.retrieval.generation import (
    BaselineGenerationCommand,
    BaselineGenerationService,
    BaselineGenerationState,
)

POSTGRES_URL = os.getenv("COMPAIR_TEST_POSTGRES_URL")


@pytest.mark.skipif(
    not POSTGRES_URL,
    reason="set COMPAIR_TEST_POSTGRES_URL to run real PostgreSQL persistence",
)
def test_postgres_transaction_idempotency_concurrency_and_retention() -> None:
    assert POSTGRES_URL is not None
    schema_name = f"baseline_persistence_{uuid4().hex}"
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
        service = BaselineEvidencePersistenceService(environment.sessions)

        first = service.persist(environment.command())
        replay = service.persist(environment.command())
        assert replay.replayed is True
        assert replay.run_id == first.run_id
        assert persistence_counts(scoped_engine) == (1, 4, 4, 4, 0)

        conflict = replace(
            environment.command(),
            retrieval_result=replace(
                environment.result, request_id="postgres-conflicting-intent"
            ),
        )
        with pytest.raises(BaselineEvidencePersistenceError) as caught:
            service.persist(conflict)
        assert caught.value.code == "idempotency_conflict"

        concurrent_command = environment.command("postgres-concurrent-retry")

        def persist_once(_ordinal: int):
            return BaselineEvidencePersistenceService(environment.sessions).persist(
                concurrent_command
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            receipts = tuple(executor.map(persist_once, range(4)))
        assert len({receipt.run_id for receipt in receipts}) == 1
        assert sum(not receipt.replayed for receipt in receipts) == 1
        assert persistence_counts(scoped_engine) == (2, 4, 8, 8, 0)

        legacy_stages = tuple(
            stage
            for stage in PersistenceWriteStage
            if stage
            not in {
                PersistenceWriteStage.CONTROL_RELATIONSHIP,
                PersistenceWriteStage.PROTECTED_PAYLOAD,
            }
        )
        for ordinal, stage in enumerate(legacy_stages, start=1):
            before = persistence_counts(scoped_engine)

            def fail_at(actual: PersistenceWriteStage, target=stage) -> None:
                if actual is target:
                    raise RuntimeError(f"postgres-injected-{target.value}")

            with pytest.raises(RuntimeError, match=f"postgres-injected-{stage.value}"):
                BaselineEvidencePersistenceService(
                    environment.sessions, stage_hook=fail_at
                ).persist(environment.command(f"postgres-rollback-{ordinal}"))
            assert persistence_counts(scoped_engine) == before

        invalid = replace(
            environment.command("postgres-invalid"),
            retrieval_result=replace(
                environment.result, status=environment.result.status.INSUFFICIENT
            ),
        )
        before = persistence_counts(scoped_engine)
        with pytest.raises(BaselineEvidencePersistenceError):
            service.persist(invalid)
        assert persistence_counts(scoped_engine) == before

        job_id, lease_token, caller = seed_running_control_job(environment)
        document_command = control_command(
            environment,
            job_id=job_id,
            lease_token=lease_token,
            caller_user_id=caller,
            key="postgres-document-scope-intent",
        )
        before_document = persistence_counts(scoped_engine)

        def fail_control_link(stage: PersistenceWriteStage) -> None:
            if stage is PersistenceWriteStage.CONTROL_RELATIONSHIP:
                raise RuntimeError("postgres-injected-control-relationship")

        with pytest.raises(RuntimeError, match="postgres-injected-control"):
            BaselineEvidencePersistenceService(
                environment.sessions, stage_hook=fail_control_link
            ).persist(document_command)
        assert persistence_counts(scoped_engine) == before_document
        control_receipt = service.persist(document_command)
        control_replay = service.persist(document_command)
        assert control_replay.replayed is True
        assert control_replay.run_id == control_receipt.run_id
        generated = BaselineGenerationService(
            environment.sessions, notifications_enabled=False
        ).generate(
            BaselineGenerationCommand(
                run_id=control_receipt.run_id,
                group_id=environment.group_id,
                caller_user_id=caller,
            ),
            CapturingProvider("postgres document finding"),
        )
        assert generated.state is BaselineGenerationState.SUCCEEDED
        with scoped_engine.connect() as connection:
            assert (
                connection.execute(
                    text(
                        "SELECT source_chunk_id FROM baseline_retrieval_run "
                        "WHERE run_id = :run_id"
                    ),
                    {"run_id": control_receipt.run_id},
                ).scalar_one()
                is None
            )
            assert (
                connection.execute(
                    text(
                        "SELECT source_chunk_id FROM feedback WHERE "
                        "baseline_retrieval_run_id = :run_id"
                    ),
                    {"run_id": control_receipt.run_id},
                ).scalar_one()
                is None
            )
            assert (
                connection.execute(
                    text(
                        "SELECT count(*) FROM baseline_notification_outbox "
                        "WHERE run_id = :run_id"
                    ),
                    {"run_id": control_receipt.run_id},
                ).scalar_one()
                == 1
            )

        with scoped_engine.begin() as connection:
            connection.execute(
                text("DELETE FROM document WHERE document_id = :document_id"),
                {"document_id": environment.source_document_id},
            )
        assert persistence_counts(scoped_engine) == (3, 4, 12, 12, 1)
        with scoped_engine.connect() as connection:
            assert (
                connection.execute(
                    text(
                        "SELECT count(*) FROM baseline_retrieval_run "
                        "WHERE source_chunk_id IS NULL AND source_document_id IS NULL"
                    )
                ).scalar_one()
                == 3
            )
            assert (
                connection.execute(
                    text(
                        "SELECT count(*) FROM reference "
                        "WHERE baseline_selected_evidence_id IS NOT NULL "
                        "AND source_chunk_id IS NULL"
                    )
                ).scalar_one()
                == 12
            )
            legacy_count = connection.execute(
                text(
                    "SELECT count(*) FROM reference "
                    "WHERE baseline_selected_evidence_id IS NULL"
                )
            ).scalar_one()
        # Legacy References keep legacy deletion behavior; baseline rows retain
        # immutable evidence after the source disappears.
        assert legacy_count == 0

        restarted_sessions = core_db.sessionmaker(scoped_engine, expire_on_commit=False)
        with restarted_sessions() as session:
            assert (
                session.execute(
                    text("SELECT count(*) FROM baseline_selected_evidence")
                ).scalar_one()
                == 12
            )
    finally:
        if scoped_engine is not None:
            scoped_engine.dispose()
        with admin_engine.begin() as connection:
            connection.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
        admin_engine.dispose()
