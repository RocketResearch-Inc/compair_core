"""Real PostgreSQL compatible-index continuation checks.

Run with::

    COMPAIR_TEST_POSTGRES_URL=postgresql+psycopg2://... \
      pytest -q tests/test_baseline_index_continuation_postgres.py
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from sqlalchemy import select, update
from test_baseline_control_plane_postgres import (
    postgres_control_environment as _postgres_control_environment_fixture,  # noqa: F401
)
from test_baseline_index_continuation import (
    FixtureAdapter,
    _payload,
    _publish_corpus,
    _service,
)


@pytest.fixture
def postgres_control_environment(request: pytest.FixtureRequest):
    """Reuse the real PostgreSQL control-plane fixture."""

    return request.getfixturevalue("_postgres_control_environment_fixture")


from compair_core.baseline_control_plane_schema import (
    compatible_index_job,
    control_job,
    repository_approval,
)
from compair_core.compair.retrieval.control_plane import (
    PROTOCOL_SHA256,
    PROTOCOL_VERSION,
)
from compair_core.compair.retrieval.corpus import (
    RetrievalBaselineIndexPublication,
    RetrievalCorpus,
)
from compair_core.compair.retrieval.index_continuation import (
    IndexJobError,
    IndexJobStage,
    InternalIndexWorkerIdentity,
)
from compair_core.schema_migrations import read_schema_migration_state


def test_postgres_index_job_submission_publication_concurrency_and_restart(
    postgres_control_environment,
) -> None:
    environment = postgres_control_environment
    assert read_schema_migration_state(environment.engine)[-1].migration_id == (
        "0014_baseline_worker_runtime_attestation_v1"
    )
    _publish_corpus(environment)
    service = _service(environment)
    payload = _payload(environment)
    barrier = threading.Barrier(2)

    def submit():
        barrier.wait()
        return service.submit(payload, caller_user_id=environment.user_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        accepted = list(pool.map(lambda _ordinal: submit(), range(2)))
    assert len({str(item["job_id"]) for item in accepted}) == 1
    assert sorted(bool(item["replayed"]) for item in accepted) == [False, True]
    job_id = str(accepted[0]["job_id"])

    barrier = threading.Barrier(2)

    def execute(ordinal: int):
        barrier.wait()
        try:
            return service.execute(
                identity=InternalIndexWorkerIdentity.create(
                    f"postgres-index-worker-{ordinal}"
                ),
                group_id=environment.group_id,
                job_id=job_id,
            )
        except IndexJobError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(execute, range(2)))
    assert sum(not isinstance(item, str) for item in outcomes) == 1
    completed = next(item for item in outcomes if not isinstance(item, str))
    restarted = _service(environment).execute(
        identity=InternalIndexWorkerIdentity.create("postgres-restart-worker"),
        group_id=environment.group_id,
        job_id=job_id,
    )
    assert restarted == completed


def test_postgres_publication_rollback_vector_failure_and_revocation(
    postgres_control_environment,
) -> None:
    environment = postgres_control_environment
    _publish_corpus(environment, ordinal=1)
    first_service = _service(environment)
    first_job = first_service.submit(
        _payload(
            environment, idempotency_key="opaque-postgres-index-first-intent-0001"
        ),
        caller_user_id=environment.user_id,
    )
    first = first_service.execute(
        identity=InternalIndexWorkerIdentity.create("postgres-first-index"),
        group_id=environment.group_id,
        job_id=str(first_job["job_id"]),
    )

    _publish_corpus(environment, ordinal=2)

    def crash(stage: IndexJobStage) -> None:
        if stage is IndexJobStage.AFTER_PUBLICATION:
            raise RuntimeError("postgres injected publication rollback")

    crashing = _service(environment, stage_hook=crash)
    crash_job = crashing.submit(
        _payload(
            environment, idempotency_key="opaque-postgres-index-crash-intent-0002"
        ),
        caller_user_id=environment.user_id,
    )
    with pytest.raises(IndexJobError, match="index_publication_failed"):
        crashing.execute(
            identity=InternalIndexWorkerIdentity.create("postgres-crash-index"),
            group_id=environment.group_id,
            job_id=str(crash_job["job_id"]),
        )
    with crashing.sessions() as session:
        corpus = session.get(RetrievalCorpus, first.corpus_id)
        publication = session.get(
            RetrievalBaselineIndexPublication,
            first.corpus_id,
        )
    assert corpus is not None
    assert publication is not None and publication.index_id == first.index_id

    recovered = _service(environment).execute(
        identity=InternalIndexWorkerIdentity.create("postgres-recovered-index"),
        group_id=environment.group_id,
        job_id=str(crash_job["job_id"]),
    )
    assert recovered.attempt_count == 2

    _publish_corpus(environment, ordinal=3)
    nonfinite = _service(environment, FixtureAdapter(mode="nan"))
    vector_job = nonfinite.submit(
        _payload(
            environment, idempotency_key="opaque-postgres-index-vector-intent-0003"
        ),
        caller_user_id=environment.user_id,
    )
    with pytest.raises(IndexJobError, match="embedding_vector_nonfinite"):
        nonfinite.execute(
            identity=InternalIndexWorkerIdentity.create("postgres-vector-index"),
            group_id=environment.group_id,
            job_id=str(vector_job["job_id"]),
        )
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(control_job.c.state).where(
                    control_job.c.job_id == vector_job["job_id"]
                )
            ).scalar_one()
            == "terminal_failed"
        )
    terminal_status = environment.service.job_status(
        {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "message_type": "job_status_request",
            "request_id": "91000000-0000-4000-8000-000000000002",
            "group_id": environment.group_id,
            "job_id": vector_job["job_id"],
        },
        caller_user_id=environment.user_id,
    )
    assert terminal_status["result"] is None

    _publish_corpus(environment, ordinal=4)
    revoked = _service(environment)
    revoked_job = revoked.submit(
        _payload(
            environment, idempotency_key="opaque-postgres-index-revoked-intent-0004"
        ),
        caller_user_id=environment.user_id,
    )
    with environment.engine.begin() as connection:
        connection.execute(
            update(repository_approval)
            .where(
                repository_approval.c.registration_id
                == environment.sibling_repository_id
            )
            .values(state="disabled", disabled_at=environment.service.clock())
        )
    with pytest.raises(IndexJobError, match="repository_not_authorized"):
        revoked.execute(
            identity=InternalIndexWorkerIdentity.create("postgres-revoked-index"),
            group_id=environment.group_id,
            job_id=str(revoked_job["job_id"]),
        )


def test_postgres_precommit_rollback_and_postcommit_crash_are_atomic(
    postgres_control_environment,
) -> None:
    environment = postgres_control_environment
    _publish_corpus(environment, ordinal=1)
    first_service = _service(environment)
    first_job = first_service.submit(
        _payload(
            environment,
            idempotency_key="opaque-postgres-atomic-first-publication-0001",
        ),
        caller_user_id=environment.user_id,
    )
    first = first_service.execute(
        identity=InternalIndexWorkerIdentity.create("postgres-atomic-first-worker"),
        group_id=environment.group_id,
        job_id=str(first_job["job_id"]),
    )

    _publish_corpus(environment, ordinal=2)

    def fail_before_commit(stage: IndexJobStage) -> None:
        # The pointer and safe result have both been written by this point, but
        # the outer builder transaction has not committed.
        if stage is IndexJobStage.BEFORE_SUCCESS:
            raise RuntimeError("postgres precommit fault")

    precommit = _service(environment, stage_hook=fail_before_commit)
    precommit_job = precommit.submit(
        _payload(
            environment,
            idempotency_key="opaque-postgres-atomic-precommit-intent-0002",
        ),
        caller_user_id=environment.user_id,
    )
    with pytest.raises(IndexJobError, match="index_publication_failed"):
        precommit.execute(
            identity=InternalIndexWorkerIdentity.create("postgres-precommit-worker"),
            group_id=environment.group_id,
            job_id=str(precommit_job["job_id"]),
        )
    with precommit.sessions() as session:
        failed_control = (
            session.execute(
                select(control_job).where(
                    control_job.c.job_id == precommit_job["job_id"]
                )
            )
            .mappings()
            .one()
        )
        failed_extension = (
            session.execute(
                select(compatible_index_job).where(
                    compatible_index_job.c.job_id == precommit_job["job_id"]
                )
            )
            .mappings()
            .one()
        )
        publication = session.get(RetrievalBaselineIndexPublication, first.corpus_id)
    assert failed_control["state"] == "retryable_failed"
    assert failed_extension["result_index_id"] is None
    assert publication is not None and publication.index_id == first.index_id
    failed_status = environment.service.job_status(
        {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "message_type": "job_status_request",
            "request_id": "91000000-0000-4000-8000-000000000001",
            "group_id": environment.group_id,
            "job_id": precommit_job["job_id"],
        },
        caller_user_id=environment.user_id,
    )
    assert failed_status["result"] is None

    recovered = _service(environment).execute(
        identity=InternalIndexWorkerIdentity.create("postgres-precommit-recovery"),
        group_id=environment.group_id,
        job_id=str(precommit_job["job_id"]),
    )
    assert recovered.state == "succeeded"

    _publish_corpus(environment, ordinal=3)

    def fail_after_commit(stage: IndexJobStage) -> None:
        if stage is IndexJobStage.AFTER_COMMIT:
            raise RuntimeError("postgres postcommit response fault")

    postcommit = _service(environment, stage_hook=fail_after_commit)
    postcommit_job = postcommit.submit(
        _payload(
            environment,
            idempotency_key="opaque-postgres-atomic-postcommit-intent-0003",
        ),
        caller_user_id=environment.user_id,
    )
    with pytest.raises(IndexJobError, match="index_build_failed"):
        postcommit.execute(
            identity=InternalIndexWorkerIdentity.create("postgres-postcommit-worker"),
            group_id=environment.group_id,
            job_id=str(postcommit_job["job_id"]),
        )
    with postcommit.sessions() as session:
        durable_control = (
            session.execute(
                select(control_job).where(
                    control_job.c.job_id == postcommit_job["job_id"]
                )
            )
            .mappings()
            .one()
        )
        durable_extension = (
            session.execute(
                select(compatible_index_job).where(
                    compatible_index_job.c.job_id == postcommit_job["job_id"]
                )
            )
            .mappings()
            .one()
        )
        durable_publication = session.get(
            RetrievalBaselineIndexPublication,
            durable_extension["corpus_id"],
        )
    assert durable_control["state"] == "succeeded"
    assert durable_control["error_code"] is None
    assert durable_extension["result_index_id"] is not None
    assert durable_publication is not None
    assert durable_publication.index_id == durable_extension["result_index_id"]
