"""Real PostgreSQL protected baseline-run job checks.

Run with::

    COMPAIR_TEST_POSTGRES_URL=postgresql+psycopg2://... \
      pytest -q tests/test_baseline_run_jobs_postgres.py
"""

from __future__ import annotations

import copy
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import func, select, text, update
from test_baseline_control_plane import _add_group_member
from test_baseline_control_plane_postgres import (
    postgres_control_environment as _postgres_control_environment_fixture,  # noqa: F401
)
from test_baseline_run_jobs import (
    RAW_QUERY,
    _keyring,
    _rows,
    _run_payload,
)

from compair_core.baseline_control_plane_schema import (
    baseline_run_job,
    baseline_run_payload,
    repository_approval,
)
from compair_core.compair.retrieval.control_plane_v2 import parse_run_submission
from compair_core.compair.retrieval.run_jobs import (
    BaselineRunJobError,
    BaselineRunJobService,
    RunSubmissionStage,
)
from compair_core.schema_migrations import read_schema_migration_state


@pytest.fixture
def postgres_control_environment(request: pytest.FixtureRequest):
    return request.getfixturevalue("_postgres_control_environment_fixture")


def test_postgres_run_job_migration_concurrency_restart_rollback_and_erasure(
    postgres_control_environment,
) -> None:
    environment = postgres_control_environment
    assert read_schema_migration_state(environment.engine)[-1].migration_id == (
        "0014_baseline_worker_runtime_attestation_v1"
    )
    payload = _run_payload(environment)
    submission = parse_run_submission(payload)
    service = BaselineRunJobService(environment.engine, _keyring())
    barrier = threading.Barrier(2)

    def submit():
        barrier.wait()
        return service.submit(submission, caller_user_id=environment.user_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _ordinal: submit(), range(2)))
    assert len({str(result["job_id"]) for result in results}) == 1
    assert sorted(bool(result["replayed"]) for result in results) == [False, True]
    job_id = str(results[0]["job_id"])
    job, protected = _rows(environment, job_id)
    restarted = BaselineRunJobService(environment.engine, _keyring())
    assert (
        restarted.cipher.decrypt(job=job, payload=protected).retrieval_query
        == RAW_QUERY
    )
    status = restarted.read_status(
        request_id=str(uuid4()),
        group_id=environment.group_id,
        job_id=job_id,
        caller_user_id=environment.user_id,
    )
    assert status["state"] == "queued"
    assert RAW_QUERY not in str(status)

    rollback_payload = copy.deepcopy(payload)
    rollback_payload["request_id"] = str(uuid4())
    rollback_payload["idempotency_key"] = "opaque-postgres-run-rollback-intent-000001"

    def fail(stage: RunSubmissionStage) -> None:
        if stage is RunSubmissionStage.AFTER_PAYLOAD_INSERT:
            raise RuntimeError("injected postgres rollback")

    with pytest.raises(RuntimeError, match="injected postgres rollback"):
        BaselineRunJobService(environment.engine, _keyring(), stage_hook=fail).submit(
            parse_run_submission(rollback_payload),
            caller_user_id=environment.user_id,
        )
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(baseline_run_job)
            ).scalar_one()
            == 1
        )
        assert (
            connection.execute(
                select(func.count()).select_from(baseline_run_payload)
            ).scalar_one()
            == 1
        )

    tampered = bytearray(protected["ciphertext"])
    tampered[0] ^= 1
    with environment.engine.begin() as connection:
        connection.execute(
            update(baseline_run_payload)
            .where(baseline_run_payload.c.job_id == job_id)
            .values(ciphertext=bytes(tampered))
        )
    with pytest.raises(BaselineRunJobError, match="run_payload_authentication_failed"):
        restarted.verify_protected_payload_integrity(
            group_id=environment.group_id,
            job_id=job_id,
            caller_user_id=environment.user_id,
        )

    now = [datetime.now(timezone.utc)]
    expiring_payload = copy.deepcopy(payload)
    expiring_payload["request_id"] = str(uuid4())
    expiring_payload["idempotency_key"] = "opaque-postgres-run-expiring-intent-00001"
    expiring = BaselineRunJobService(
        environment.engine,
        _keyring(),
        clock=lambda: now[0],
        payload_lifetime=timedelta(minutes=1),
    )
    expiring_result = expiring.submit(
        parse_run_submission(expiring_payload), caller_user_id=environment.user_id
    )
    now[0] += timedelta(minutes=2)
    assert expiring.cleanup_protected_payloads() == 1
    with environment.engine.connect() as connection:
        expired = (
            connection.execute(
                select(baseline_run_job).where(
                    baseline_run_job.c.job_id == expiring_result["job_id"]
                )
            )
            .mappings()
            .one()
        )
        assert expired["state"] == "blocked"
        assert expired["reason_code"] == "payload_expired"
        assert (
            connection.execute(
                select(func.count()).select_from(baseline_run_payload)
            ).scalar_one()
            == 0
        )


def test_postgres_run_job_authorization_revocation_source_deletion_and_8mb_limit(
    postgres_control_environment,
) -> None:
    environment = postgres_control_environment
    payload = _run_payload(environment)
    boundary = "p" * 8_000_000
    encoded = boundary.encode()
    payload["retrieval_query"].update(
        text=boundary,
        byte_size=len(encoded),
        sha256=hashlib.sha256(encoded).hexdigest(),
    )
    member = _add_group_member(environment.engine, group_id=environment.group_id)
    service = BaselineRunJobService(environment.engine, _keyring())
    accepted = service.submit(parse_run_submission(payload), caller_user_id=member)
    assert accepted["state"] == "queued"

    outsider_payload = copy.deepcopy(payload)
    outsider_payload["request_id"] = str(uuid4())
    outsider_payload["idempotency_key"] = "opaque-postgres-run-outsider-intent-000001"
    with pytest.raises(BaselineRunJobError, match="not_found_or_forbidden"):
        service.submit(
            parse_run_submission(outsider_payload), caller_user_id=str(uuid4())
        )

    with environment.engine.begin() as connection:
        connection.execute(
            update(repository_approval)
            .where(
                repository_approval.c.registration_id
                == environment.changed_repository_id
            )
            .values(state="disabled", disabled_at=datetime.now(timezone.utc))
        )
    with pytest.raises(BaselineRunJobError, match="repository_not_authorized"):
        service.submit(parse_run_submission(payload), caller_user_id=member)

    with environment.engine.begin() as connection:
        connection.execute(
            text("DELETE FROM document WHERE document_id = :document_id"),
            {"document_id": environment.source_document_id},
        )
    with pytest.raises(BaselineRunJobError, match="job_not_found_or_forbidden"):
        service.read_status(
            request_id=str(uuid4()),
            group_id=environment.group_id,
            job_id=str(accepted["job_id"]),
            caller_user_id=member,
        )
    with environment.engine.connect() as connection:
        audit = (
            connection.execute(
                select(baseline_run_job).where(
                    baseline_run_job.c.job_id == accepted["job_id"]
                )
            )
            .mappings()
            .one()
        )
        assert audit["source_document_id"] is None
        assert boundary not in repr(dict(audit))


def test_postgres_stale_generation_rejected_without_run_write(
    postgres_control_environment,
) -> None:
    from test_baseline_index_continuation import _publish_corpus

    environment = postgres_control_environment
    payload = _run_payload(environment)
    _publish_corpus(environment, ordinal=2)
    with pytest.raises(BaselineRunJobError, match="index_publication_stale"):
        BaselineRunJobService(environment.engine, _keyring()).submit(
            parse_run_submission(payload), caller_user_id=environment.user_id
        )
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(baseline_run_job)
            ).scalar_one()
            == 0
        )
