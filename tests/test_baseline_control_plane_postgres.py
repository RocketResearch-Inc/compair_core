"""Real PostgreSQL staging/publication-boundary coverage.

Run with::

    COMPAIR_TEST_POSTGRES_URL=postgresql+psycopg2://... \
      pytest -q tests/test_baseline_control_plane_postgres.py
"""

from __future__ import annotations

import copy
import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import func, select, text, update
from sqlalchemy.exc import DatabaseError
from test_baseline_control_plane import (
    _add_group_member,
    _begin_payload,
    _commit_payload,
    _continuation_status_payload,
    _continuation_worker,
    _part_payload,
    _registration_create_payload,
    _registration_state_payload,
    _row_counts,
    _single_continuation_id,
    _stage_success,
    _stage_worker_snapshot,
    make_control_environment,
)

from compair_core import db as core_db
from compair_core.baseline_control_plane_schema import (
    control_job,
    repository_approval,
    snapshot_content_part,
    snapshot_continuation_job,
    snapshot_staging,
)
from compair_core.compair.retrieval.continuation_worker import (
    ContinuationWorkerError,
    ContinuationWorkerStage,
    InternalContinuationWorkerIdentity,
)
from compair_core.compair.retrieval.control_plane import (
    BaselineControlPlaneService,
    ControlPlaneError,
    ControlWriteStage,
    canonicalize,
)
from compair_core.compair.retrieval.corpus import (
    RetrievalCorpus,
    RetrievalCorpusGeneration,
)
from compair_core.schema_migrations import read_schema_migration_state

POSTGRES_URL = os.getenv("COMPAIR_TEST_POSTGRES_URL")


@pytest.fixture
def postgres_control_environment():
    if not POSTGRES_URL:
        pytest.skip("set COMPAIR_TEST_POSTGRES_URL for real PostgreSQL staging tests")
    schema_name = f"baseline_control_{uuid4().hex}"
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
        yield make_control_environment(scoped_engine)
    finally:
        scoped_engine.dispose()
        with admin_engine.begin() as connection:
            connection.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
        admin_engine.dispose()


def test_postgres_migration_concurrent_staging_rollback_and_restart(
    postgres_control_environment,
) -> None:
    environment = postgres_control_environment
    state = read_schema_migration_state(environment.engine)
    assert state[-1].migration_id == "0013_baseline_database_worker_v1"
    assert state[-1].state == "applied"

    begin, content = _begin_payload(environment)
    barrier = threading.Barrier(2)

    def submit_begin():
        barrier.wait()
        return environment.service.begin_snapshot(
            begin, caller_user_id=environment.user_id
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        accepted = list(pool.map(lambda _ordinal: submit_begin(), range(2)))
    assert len({str(item["job_id"]) for item in accepted}) == 1
    assert _row_counts(environment.engine) == (1, 1, 0)

    job_id = str(accepted[0]["job_id"])
    part = _part_payload(begin, job_id, content)
    body_hash = hashlib.sha256(canonicalize(part)).hexdigest()
    barrier = threading.Barrier(2)

    def submit_part():
        barrier.wait()
        return environment.service.stage_content_part(
            part,
            caller_user_id=environment.user_id,
            request_body_sha256=body_hash,
            path_job_id=job_id,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        staged = list(pool.map(lambda _ordinal: submit_part(), range(2)))
    assert _row_counts(environment.engine) == (1, 1, 1)
    assert sorted(bool(item["replayed"]) for item in staged) == [False, True]

    commit = _commit_payload(begin, job_id, part)
    failing = BaselineControlPlaneService(
        environment.engine,
        stage_hook=lambda stage: (
            (_ for _ in ()).throw(RuntimeError("postgres_commit_failure"))
            if stage is ControlWriteStage.COMMIT
            else None
        ),
    )
    with pytest.raises(RuntimeError, match="postgres_commit_failure"):
        failing.commit_snapshot(
            commit,
            caller_user_id=environment.user_id,
            path_job_id=job_id,
        )
    with environment.engine.connect() as connection:
        assert (
            connection.execute(select(snapshot_staging.c.status)).scalar_one() == "open"
        )
        assert connection.execute(select(control_job.c.state)).scalar_one() == "queued"
        assert (
            connection.execute(
                select(func.count()).select_from(snapshot_continuation_job)
            ).scalar_one()
            == 0
        )

    sealed = environment.service.commit_snapshot(
        commit,
        caller_user_id=environment.user_id,
        path_job_id=job_id,
    )
    assert sealed["state"] == "succeeded"
    assert sealed["staging"]["state"] == "sealed"
    continuation = environment.service.continuation_status(
        _continuation_status_payload(
            environment,
            staging_job_id=str(sealed["job_id"]),
        ),
        caller_user_id=environment.user_id,
    )
    assert continuation["state"] == "queued"

    with pytest.raises(DatabaseError), environment.engine.begin() as connection:
        connection.execute(
            update(snapshot_content_part)
            .where(snapshot_content_part.c.part_ordinal == 1)
            .values(part_sha256="0" * 64)
        )
    with pytest.raises(DatabaseError), environment.engine.begin() as connection:
        connection.execute(
            update(snapshot_staging)
            .where(snapshot_staging.c.job_id == job_id)
            .values(canonical_manifest_hash="0" * 64)
        )

    environment.engine.dispose()
    restarted = BaselineControlPlaneService(environment.engine)
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(snapshot_content_part)
            ).scalar_one()
            == 1
        )
        assert (
            connection.execute(select(control_job.c.state)).scalar_one() == "succeeded"
        )
    assert restarted.engine.dialect.name == "postgresql"


def test_postgres_admin_registration_continuation_claim_and_revocation(
    postgres_control_environment,
) -> None:
    environment = postgres_control_environment
    member_id = _add_group_member(
        environment.engine,
        group_id=environment.group_id,
    )
    registration = _registration_create_payload(
        environment,
        repository_uid="postgres-upstream-repository-uid",
    )
    with pytest.raises(ControlPlaneError, match="not_found_or_forbidden"):
        environment.service.register_repository(
            registration,
            caller_user_id=member_id,
        )
    created = environment.service.register_repository(
        registration,
        caller_user_id=environment.user_id,
    )
    assert created["state"] == "active"
    conflicting_registration = copy.deepcopy(registration)
    conflicting_registration["request_id"] = str(uuid4())
    conflicting_registration["source_document_id"] = environment.source_document_id
    with pytest.raises(
        ControlPlaneError,
        match="repository_registration_conflict",
    ):
        environment.service.register_repository(
            conflicting_registration,
            caller_user_id=environment.user_id,
        )
    with pytest.raises(DatabaseError), environment.engine.begin() as connection:
        connection.execute(
            update(repository_approval)
            .where(repository_approval.c.registration_id == created["registration_id"])
            .values(repository_uid="mutated-postgres-uid")
        )

    _begin, _part, _commit, _staged, _sealed = _stage_success(environment)
    continuation_id = _single_continuation_id(environment.engine)
    barrier = threading.Barrier(2)

    def claim():
        barrier.wait()
        try:
            return environment.service.claim_continuation_job(
                caller_user_id=environment.user_id,
                group_id=environment.group_id,
                job_id=continuation_id,
            )
        except ControlPlaneError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda _ordinal: claim(), range(2)))
    assert sum(not isinstance(item, str) for item in claims) == 1
    assert sum(item == "job_lease_unavailable" for item in claims) == 1

    with environment.engine.begin() as connection:
        connection.execute(
            update(snapshot_continuation_job)
            .where(snapshot_continuation_job.c.continuation_job_id == continuation_id)
            .values(
                state="retryable_failed",
                lease_token=None,
                lease_expires_at=None,
            )
        )
    environment.service.set_repository_registration_state(
        _registration_state_payload(
            environment,
            environment.sibling_repository_id,
            active=False,
        ),
        caller_user_id=environment.user_id,
    )
    with pytest.raises(ControlPlaneError, match="repository_not_authorized"):
        environment.service.claim_continuation_job(
            caller_user_id=environment.user_id,
            group_id=environment.group_id,
            job_id=continuation_id,
        )
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(repository_approval)
            ).scalar_one()
            == 3
        )
        assert (
            connection.execute(
                select(func.count()).select_from(snapshot_continuation_job)
            ).scalar_one()
            == 1
        )

    environment.engine.dispose()
    restarted = BaselineControlPlaneService(environment.engine)
    status = restarted.continuation_status(
        _continuation_status_payload(
            environment,
            continuation_job_id=continuation_id,
        ),
        caller_user_id=environment.user_id,
    )
    assert status["state"] == "retryable_failed"
    assert status["result"]["corpus_eligible"] is False
    environment.service.set_repository_registration_state(
        _registration_state_payload(
            environment,
            environment.sibling_repository_id,
            active=True,
        ),
        caller_user_id=environment.user_id,
    )
    with environment.engine.begin() as connection:
        connection.execute(
            text(
                "DELETE FROM user_to_group "
                "WHERE user_id = :user_id AND group_id = :group_id"
            ),
            {
                "user_id": environment.user_id,
                "group_id": environment.group_id,
            },
        )
    with pytest.raises(ControlPlaneError, match="not_found_or_forbidden"):
        restarted.claim_continuation_job(
            caller_user_id=environment.user_id,
            group_id=environment.group_id,
            job_id=continuation_id,
        )
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(snapshot_continuation_job)
            ).scalar_one()
            == 1
        )


def test_postgres_expired_staging_cleanup_respects_an_active_lease(
    postgres_control_environment,
) -> None:
    environment = postgres_control_environment
    now = datetime(2026, 8, 16, tzinfo=timezone.utc)
    clock_value = [now]
    service = BaselineControlPlaneService(
        environment.engine,
        clock=lambda: clock_value[0],
    )
    begin, _content = _begin_payload(environment)
    accepted = service.begin_snapshot(begin, caller_user_id=environment.user_id)
    staging_job_id = str(accepted["job_id"])
    service.acquire_job_lease(
        caller_user_id=environment.user_id,
        group_id=environment.group_id,
        job_id=staging_job_id,
        lifetime=timedelta(hours=30),
    )
    clock_value[0] += timedelta(hours=25)
    assert service.expire_staging_sessions() == 0
    clock_value[0] += timedelta(hours=6)
    assert service.expire_staging_sessions() == 1
    with environment.engine.connect() as connection:
        assert connection.execute(select(snapshot_staging.c.status)).scalar_one() == (
            "expired"
        )
        assert connection.execute(select(control_job.c.state)).scalar_one() == (
            "terminal_failed"
        )
        assert (
            connection.execute(
                select(func.count()).select_from(snapshot_continuation_job)
            ).scalar_one()
            == 0
        )


def test_postgres_continuation_worker_publication_retry_and_concurrency(
    postgres_control_environment,
) -> None:
    environment = postgres_control_environment
    first_job = _stage_worker_snapshot(
        environment,
        content="postgres first snapshot\n",
        idempotency_key="opaque-postgres-worker-snapshot-0001",
    )
    first = _continuation_worker(environment).execute(
        identity=InternalContinuationWorkerIdentity.create("postgres-worker-1"),
        group_id=environment.group_id,
        continuation_job_id=first_job,
    )
    second_job = _stage_worker_snapshot(
        environment,
        content="postgres rollback snapshot\n",
        idempotency_key="opaque-postgres-worker-snapshot-0002",
    )

    def fail_publication(stage: ContinuationWorkerStage) -> None:
        if stage is ContinuationWorkerStage.BEFORE_SUCCESS:
            raise RuntimeError("postgres_injected_failure")

    with pytest.raises(ContinuationWorkerError) as failure:
        _continuation_worker(environment, stage_hook=fail_publication).execute(
            identity=InternalContinuationWorkerIdentity.create("postgres-worker-2"),
            group_id=environment.group_id,
            continuation_job_id=second_job,
        )
    assert failure.value.code == "corpus_ingestion_failed"
    sessions = core_db.sessionmaker(environment.engine, expire_on_commit=False)
    with sessions() as session:
        corpus = session.get(RetrievalCorpus, first.corpus_id)
        staged_count = (
            session.query(RetrievalCorpusGeneration)
            .filter_by(corpus_id=first.corpus_id)
            .count()
        )
    assert corpus is not None and corpus.active_generation_id == first.generation_id
    assert staged_count == 2

    resumed = _continuation_worker(environment).execute(
        identity=InternalContinuationWorkerIdentity.create("postgres-worker-3"),
        group_id=environment.group_id,
        continuation_job_id=second_job,
    )
    assert resumed.state == "succeeded"
    assert resumed.generation_id != first.generation_id

    third_job = _stage_worker_snapshot(
        environment,
        content="postgres concurrent snapshot\n",
        idempotency_key="opaque-postgres-worker-snapshot-0003",
    )
    barrier = threading.Barrier(2)

    def execute(ordinal: int):
        barrier.wait()
        try:
            return _continuation_worker(environment).execute(
                identity=InternalContinuationWorkerIdentity.create(
                    f"postgres-concurrent-{ordinal}"
                ),
                group_id=environment.group_id,
                continuation_job_id=third_job,
            )
        except ContinuationWorkerError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(execute, range(2)))
    assert any(not isinstance(item, str) for item in outcomes)
    with environment.engine.connect() as connection:
        continuation = (
            connection.execute(
                select(snapshot_continuation_job).where(
                    snapshot_continuation_job.c.continuation_job_id == third_job
                )
            )
            .mappings()
            .one()
        )
        parts_before = connection.execute(
            select(func.count()).select_from(snapshot_content_part)
        ).scalar_one()
    assert continuation["state"] == "succeeded"
    assert continuation["attempt_count"] == 1
    assert continuation["result_generation_id"] is not None
    status = environment.service.continuation_status(
        _continuation_status_payload(
            environment,
            continuation_job_id=third_job,
        ),
        caller_user_id=environment.user_id,
    )
    assert status["result"]["index_state"] == "incomplete"
    assert "postgres concurrent snapshot" not in str(status)

    with pytest.raises(DatabaseError), environment.engine.begin() as connection:
        connection.execute(
            snapshot_content_part.delete().where(
                snapshot_content_part.c.staging_id == continuation["staging_id"]
            )
        )
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(snapshot_content_part)
            ).scalar_one()
            == parts_before
        )
