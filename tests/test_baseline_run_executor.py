from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from sqlalchemy import func, select, text, update
from sqlalchemy.orm import sessionmaker
from test_baseline_control_plane import (
    ControlEnvironment,
)
from test_baseline_control_plane import (
    environment as _environment_fixture,  # noqa: F401
)
from test_baseline_index_continuation import FixtureAdapter, _identity
from test_baseline_run_jobs import RAW_QUERY, _keyring, _run_payload, _service

from compair_core.baseline_control_plane_schema import (
    BASELINE_RUN_WORKER_CONTRACT_VERSION,
    BASELINE_RUN_WORKER_SERVICE_ID,
    baseline_run_job,
    baseline_run_payload,
    repository_approval,
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
    BaselineRunExecutorStage,
    InternalBaselineRunWorkerIdentity,
)
from compair_core.compair.retrieval.types import RetrievalRequest, RetrievalResult


@pytest.fixture
def environment(request: pytest.FixtureRequest):
    return request.getfixturevalue("_environment_fixture")


class RecordingRetriever:
    def __init__(self, delegate, *, after=None) -> None:
        self.delegate = delegate
        self.after = after
        self.requests: list[RetrievalRequest] = []
        self.results: list[RetrievalResult] = []

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        self.requests.append(request)
        result = self.delegate.retrieve(request)
        self.results.append(result)
        if self.after is not None:
            self.after()
        return result


class RaisingRetriever:
    def __init__(self) -> None:
        self.requests: list[RetrievalRequest] = []

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        self.requests.append(request)
        raise RuntimeError("provider detail that must not escape")


class MalformedRetriever(RecordingRetriever):
    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        result = super().retrieve(request)
        return replace(result, engine_version="unsupported-engine-draft")


def _submit(environment: ControlEnvironment, *, suffix: str = "main") -> str:
    payload = _run_payload(
        environment,
        idempotency_key=f"opaque-executor-{suffix}-intent-000000000001",
    )
    accepted = _service(environment).submit(
        parse_run_submission(payload), caller_user_id=environment.user_id
    )
    return str(accepted["job_id"])


def _persistent(environment: ControlEnvironment, *, filtered: bool = False):
    return PersistentBaselineV1Retriever(
        sessionmaker(environment.engine, expire_on_commit=False),
        FixtureAdapter(),
        evidence_filter=(lambda _candidate: False) if filtered else None,
    )


def _executor(environment: ControlEnvironment, retriever, **kwargs):
    return BaselineDocumentRunExecutor(
        environment.engine,
        identity=InternalBaselineRunWorkerIdentity.create("sqlite-runner-1"),
        keyring=_keyring(),
        retriever_factory=lambda: retriever,
        **kwargs,
    )


def _job(environment: ControlEnvironment, job_id: str) -> dict[str, object]:
    with environment.engine.connect() as connection:
        return dict(
            connection.execute(
                select(baseline_run_job).where(baseline_run_job.c.job_id == job_id)
            )
            .mappings()
            .one()
        )


def _effect_counts(environment: ControlEnvironment) -> dict[str, int]:
    with environment.engine.connect() as connection:
        return {
            table: int(
                connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()
            )
            for table in (
                "baseline_retrieval_run",
                "baseline_evidence_artifact",
                "baseline_selected_evidence",
                "feedback",
                "baseline_notification_outbox",
                "notification_event",
            )
        }


def _payload_count(environment: ControlEnvironment, job_id: str) -> int:
    with environment.engine.connect() as connection:
        return int(
            connection.execute(
                select(func.count())
                .select_from(baseline_run_payload)
                .where(baseline_run_payload.c.job_id == job_id)
            ).scalar_one()
        )


def test_document_executor_invokes_one_complete_query_and_persists_once(
    environment: ControlEnvironment,
) -> None:
    job_id = _submit(environment)
    retriever = RecordingRetriever(_persistent(environment))
    outcome = _executor(environment, retriever).execute(job_id)

    assert len(retriever.requests) == 1
    request = retriever.requests[0]
    assert request.retrieval_query == RAW_QUERY
    assert request.group_id == environment.group_id
    assert request.source_document_id == environment.source_document_id
    assert request.changed_repository_id == environment.changed_repository_id
    assert request.repository_roots == ()
    assert request.changed_repository is None
    assert retriever.results[0].engine == "baseline_v1"
    assert retriever.results[0].fallback_engine is None
    assert outcome.state == "references_persisted"
    assert outcome.evidence_count == outcome.reference_count
    assert 1 <= outcome.evidence_count <= 4

    with environment.engine.connect() as connection:
        job = (
            connection.execute(
                select(baseline_run_job).where(baseline_run_job.c.job_id == job_id)
            )
            .mappings()
            .one()
        )
        selected = (
            connection.execute(
                text(
                    "SELECT s.selected_evidence_id, s.ordinal, s.selected_character_count, "
                    "r.reference_id, r.source_chunk_id, r.reference_type "
                    "FROM baseline_selected_evidence s JOIN reference r ON "
                    "r.baseline_selected_evidence_id = s.selected_evidence_id "
                    "WHERE s.run_id = :run_id ORDER BY s.ordinal"
                ),
                {"run_id": outcome.persisted_run_id},
            )
            .mappings()
            .all()
        )
        chunk_count = int(
            connection.execute(text("SELECT count(*) FROM chunk")).scalar_one()
        )
    assert job["worker_service_id"] == BASELINE_RUN_WORKER_SERVICE_ID
    assert job["worker_contract_version"] == BASELINE_RUN_WORKER_CONTRACT_VERSION
    assert job["retrieval_result_fingerprint"] == outcome.retrieval_result_fingerprint
    assert job["lease_token"] is None
    assert [row["ordinal"] for row in selected] == list(range(1, len(selected) + 1))
    assert [row["selected_evidence_id"] for row in selected] == list(
        outcome.selected_evidence_ids
    )
    assert [row["reference_id"] for row in selected] == list(outcome.reference_ids)
    assert all(row["source_chunk_id"] is None for row in selected)
    assert all(row["reference_type"] == "baseline_file" for row in selected)
    assert sum(int(row["selected_character_count"]) for row in selected) <= 16_000
    assert chunk_count == 0
    assert _payload_count(environment, job_id) == 0
    counts = _effect_counts(environment)
    assert counts["baseline_retrieval_run"] == 1
    assert counts["feedback"] == 0
    assert counts["baseline_notification_outbox"] == 0
    assert counts["notification_event"] == 0


def test_insufficient_is_terminal_zero_effect_and_erases_payload(
    environment: ControlEnvironment,
) -> None:
    job_id = _submit(environment, suffix="insufficient")
    retriever = RecordingRetriever(_persistent(environment, filtered=True))
    outcome = _executor(environment, retriever).execute(job_id)
    assert outcome.state == "insufficient"
    assert outcome.evidence_count == outcome.reference_count == 0
    assert len(retriever.requests) == 1
    assert _payload_count(environment, job_id) == 0
    assert _effect_counts(environment) == {
        "baseline_retrieval_run": 0,
        "baseline_evidence_artifact": 0,
        "baseline_selected_evidence": 0,
        "feedback": 0,
        "baseline_notification_outbox": 0,
        "notification_event": 0,
    }


def test_retryable_retrieval_failure_retains_payload_and_retries(
    environment: ControlEnvironment,
) -> None:
    job_id = _submit(environment, suffix="retry")
    failing = RaisingRetriever()
    with pytest.raises(BaselineRunExecutorError) as caught:
        _executor(environment, failing).execute(job_id)
    assert (caught.value.code, caught.value.retryable, caught.value.state) == (
        "retrieval_error",
        True,
        "retryable_failed",
    )
    assert _job(environment, job_id)["state"] == "retryable_failed"
    assert _payload_count(environment, job_id) == 1
    assert _effect_counts(environment)["baseline_retrieval_run"] == 0

    recovered = _executor(environment, _persistent(environment)).execute(job_id)
    assert recovered.state == "references_persisted"
    assert recovered.attempt_count == 2


def test_real_embedding_timeout_result_is_retryable_and_retains_payload(
    environment: ControlEnvironment,
) -> None:
    job_id = _submit(environment, suffix="embedding-timeout")
    timed_out = PersistentBaselineV1Retriever(
        sessionmaker(environment.engine, expire_on_commit=False),
        FixtureAdapter(mode="failure"),
    )
    with pytest.raises(BaselineRunExecutorError) as caught:
        _executor(environment, timed_out).execute(job_id)
    assert (caught.value.code, caught.value.retryable, caught.value.state) == (
        "query_embedding_failed",
        True,
        "retryable_failed",
    )
    assert _job(environment, job_id)["state"] == "retryable_failed"
    assert _payload_count(environment, job_id) == 1
    assert _effect_counts(environment)["baseline_retrieval_run"] == 0


def test_lease_renewal_expiry_reclaim_and_stale_token_rejection(
    environment: ControlEnvironment,
) -> None:
    job_id = _submit(environment, suffix="lease")
    now = [datetime.now(timezone.utc)]
    executor = _executor(
        environment,
        _persistent(environment),
        clock=lambda: now[0],
        token_factory=lambda _size: "first-lease-token-0000000000000001",
    )
    first = executor.claim(job_id, lifetime=timedelta(minutes=2))
    initial_expiry = first.lease_expires_at
    now[0] += timedelta(seconds=30)
    assert (
        executor.renew(job_id, first.lease_token, lifetime=timedelta(minutes=3))
        > initial_expiry
    )
    now[0] += timedelta(minutes=4)
    replacement = BaselineDocumentRunExecutor(
        environment.engine,
        identity=InternalBaselineRunWorkerIdentity.create("sqlite-runner-2"),
        keyring=_keyring(),
        retriever_factory=lambda: _persistent(environment),
        clock=lambda: now[0],
        token_factory=lambda _size: "second-lease-token-000000000000001",
    ).claim(job_id)
    assert replacement.attempt_count == 2
    assert replacement.lease_token != first.lease_token
    with pytest.raises(BaselineRunExecutorError, match="job_lease_unavailable"):
        executor.renew(job_id, first.lease_token)


def test_concurrent_claim_has_one_current_lease(
    environment: ControlEnvironment,
) -> None:
    job_id = _submit(environment, suffix="concurrent-claim")
    barrier = threading.Barrier(2)

    def claim(ordinal: int):
        worker = BaselineDocumentRunExecutor(
            environment.engine,
            identity=InternalBaselineRunWorkerIdentity.create(
                f"claim-worker-{ordinal}"
            ),
            keyring=_keyring(),
            retriever_factory=lambda: _persistent(environment),
        )
        barrier.wait()
        try:
            return worker.claim(job_id)
        except BaselineRunExecutorError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        attempts = tuple(pool.map(claim, range(2)))
    leases = tuple(value for value in attempts if not isinstance(value, Exception))
    errors = tuple(value for value in attempts if isinstance(value, Exception))
    assert len(leases) == 1
    assert len(errors) == 1
    assert errors[0].code == "job_lease_unavailable"
    assert _job(environment, job_id)["lease_token"] == leases[0].lease_token


def test_lease_loss_before_evidence_commit_blocks_stale_worker_and_reclaims(
    environment: ControlEnvironment,
) -> None:
    job_id = _submit(environment, suffix="lease-loss")

    def expire_after_retrieval(stage: BaselineRunExecutorStage) -> None:
        if stage is BaselineRunExecutorStage.AFTER_RETRIEVAL:
            with environment.engine.begin() as connection:
                connection.execute(
                    update(baseline_run_job)
                    .where(baseline_run_job.c.job_id == job_id)
                    .values(
                        lease_expires_at=datetime.now(timezone.utc)
                        - timedelta(seconds=1)
                    )
                )

    with pytest.raises(BaselineRunExecutorError) as caught:
        _executor(
            environment,
            _persistent(environment),
            stage_hook=expire_after_retrieval,
        ).execute(job_id)
    assert caught.value.code == "control_job_lease_invalid"
    assert _effect_counts(environment)["baseline_retrieval_run"] == 0
    assert _payload_count(environment, job_id) == 1
    recovered = _executor(environment, _persistent(environment)).execute(job_id)
    assert recovered.state == "references_persisted"
    assert recovered.attempt_count == 2


def test_claim_and_pre_effect_revocation_block_and_erase_payload(
    environment: ControlEnvironment,
) -> None:
    first_job = _submit(environment, suffix="claim-revoke")
    with environment.engine.begin() as connection:
        connection.execute(
            update(repository_approval)
            .where(
                repository_approval.c.registration_id
                == environment.changed_repository_id
            )
            .values(state="disabled", disabled_at=datetime.now(timezone.utc))
        )
    with pytest.raises(BaselineRunExecutorError) as caught:
        _executor(environment, _persistent(environment)).execute(first_job)
    assert caught.value.state == "blocked"
    assert _job(environment, first_job)["state"] == "blocked"
    assert _payload_count(environment, first_job) == 0


def test_pre_effect_revocation_rolls_back_all_evidence(
    environment: ControlEnvironment,
) -> None:
    job_id = _submit(environment, suffix="effect-revoke")

    def revoke() -> None:
        with environment.engine.begin() as connection:
            connection.execute(
                update(repository_approval)
                .where(
                    repository_approval.c.registration_id
                    == environment.changed_repository_id
                )
                .values(state="disabled", disabled_at=datetime.now(timezone.utc))
            )

    retriever = RecordingRetriever(_persistent(environment), after=revoke)
    with pytest.raises(BaselineRunExecutorError) as caught:
        _executor(environment, retriever).execute(job_id)
    assert caught.value.state == "blocked"
    assert _job(environment, job_id)["state"] == "blocked"
    assert _payload_count(environment, job_id) == 0
    assert _effect_counts(environment)["baseline_retrieval_run"] == 0


@pytest.mark.parametrize("mutation", ["ciphertext", "key", "query_hash"])
def test_payload_integrity_failures_are_sanitized_and_erased(
    environment: ControlEnvironment,
    mutation: str,
) -> None:
    job_id = _submit(environment, suffix=f"integrity-{mutation}")
    with environment.engine.begin() as connection:
        if mutation == "ciphertext":
            row = connection.execute(
                select(baseline_run_payload.c.ciphertext).where(
                    baseline_run_payload.c.job_id == job_id
                )
            ).scalar_one()
            value = bytearray(row)
            value[0] ^= 1
            connection.execute(
                update(baseline_run_payload)
                .where(baseline_run_payload.c.job_id == job_id)
                .values(ciphertext=bytes(value))
            )
        elif mutation == "key":
            connection.execute(
                update(baseline_run_payload)
                .where(baseline_run_payload.c.job_id == job_id)
                .values(key_id="unavailable-key")
            )
        else:
            connection.execute(
                update(baseline_run_job)
                .where(baseline_run_job.c.job_id == job_id)
                .values(query_sha256="0" * 64)
            )
    with pytest.raises(BaselineRunExecutorError) as caught:
        _executor(environment, _persistent(environment)).execute(job_id)
    assert caught.value.code == "payload_authentication_failed"
    assert str(caught.value) == "payload_authentication_failed"
    assert _job(environment, job_id)["state"] == "blocked"
    assert _payload_count(environment, job_id) == 0


def test_cancel_is_lease_guarded_and_erases_payload(
    environment: ControlEnvironment,
) -> None:
    job_id = _submit(environment, suffix="cancel")
    executor = _executor(environment, _persistent(environment))
    lease = executor.claim(job_id)
    with pytest.raises(BaselineRunExecutorError, match="job_lease_unavailable"):
        executor.cancel(job_id, "stale-token-value-000000000000000")
    executor.cancel(job_id, lease.lease_token)
    assert _job(environment, job_id)["state"] == "cancelled"
    assert _payload_count(environment, job_id) == 0


def test_queued_cancel_acquires_an_internal_lease_and_erases_payload(
    environment: ControlEnvironment,
) -> None:
    job_id = _submit(environment, suffix="queued-cancel")
    executor = _executor(environment, _persistent(environment))
    executor.cancel(job_id)
    job = _job(environment, job_id)
    assert job["state"] == "cancelled"
    assert job["attempt_count"] == 1
    assert job["worker_service_id"] == BASELINE_RUN_WORKER_SERVICE_ID
    assert job["lease_token"] is None
    assert _payload_count(environment, job_id) == 0


def test_malformed_retrieval_result_is_terminal_and_erases_payload(
    environment: ControlEnvironment,
) -> None:
    job_id = _submit(environment, suffix="malformed-result")
    with pytest.raises(BaselineRunExecutorError) as caught:
        _executor(environment, MalformedRetriever(_persistent(environment))).execute(
            job_id
        )
    assert (caught.value.code, caught.value.retryable) == (
        "retrieval_result_incompatible",
        False,
    )
    assert _job(environment, job_id)["state"] == "terminal_failed"
    assert _payload_count(environment, job_id) == 0
    assert _effect_counts(environment)["baseline_retrieval_run"] == 0


def test_incompatible_embedding_result_blocks_and_erases_payload(
    environment: ControlEnvironment,
) -> None:
    job_id = _submit(environment, suffix="embedding-mismatch")
    mismatched = PersistentBaselineV1Retriever(
        sessionmaker(environment.engine, expire_on_commit=False),
        FixtureAdapter(identity=_identity("mismatched-fixture-revision")),
    )
    with pytest.raises(BaselineRunExecutorError) as caught:
        _executor(environment, mismatched).execute(job_id)
    assert caught.value.state == "blocked"
    assert caught.value.retryable is False
    assert _job(environment, job_id)["state"] == "blocked"
    assert _payload_count(environment, job_id) == 0
    assert _effect_counts(environment)["baseline_retrieval_run"] == 0


def test_evidence_transaction_rollback_is_retryable(
    environment: ControlEnvironment,
) -> None:
    rollback_job = _submit(environment, suffix="rollback")

    def persistence_factory():
        def fail(stage: PersistenceWriteStage) -> None:
            if stage is PersistenceWriteStage.PROTECTED_PAYLOAD:
                raise RuntimeError("injected evidence rollback")

        return BaselineEvidencePersistenceService(
            sessionmaker(environment.engine, expire_on_commit=False), stage_hook=fail
        )

    with pytest.raises(BaselineRunExecutorError) as caught:
        _executor(
            environment,
            _persistent(environment),
            persistence_factory=persistence_factory,
        ).execute(rollback_job)
    assert caught.value.state == "retryable_failed"
    assert _effect_counts(environment)["baseline_retrieval_run"] == 0
    assert _payload_count(environment, rollback_job) == 1
    recovered = _executor(environment, _persistent(environment)).execute(rollback_job)
    assert recovered.state == "references_persisted"


def test_post_commit_response_loss_recovers_exact_ids_without_retrieval(
    environment: ControlEnvironment,
) -> None:
    lost_job = _submit(environment, suffix="lost-response")
    with environment.engine.connect() as connection:
        protected = dict(
            connection.execute(
                select(baseline_run_payload).where(
                    baseline_run_payload.c.job_id == lost_job
                )
            )
            .mappings()
            .one()
        )

    def crash(stage: BaselineRunExecutorStage) -> None:
        if stage is BaselineRunExecutorStage.AFTER_EVIDENCE_COMMIT:
            raise RuntimeError("simulated process response loss")

    retriever = RecordingRetriever(_persistent(environment))
    with pytest.raises(RuntimeError, match="simulated process response loss"):
        _executor(environment, retriever, stage_hook=crash).execute(lost_job)
    durable = _job(environment, lost_job)
    assert durable["state"] == "references_persisted"
    assert _payload_count(environment, lost_job) == 0
    # Simulate an old caller that lost its response after the evidence commit
    # but before its independent payload cleanup completed.
    with environment.engine.begin() as connection:
        connection.execute(baseline_run_payload.insert().values(**protected))
    replay_retriever = RecordingRetriever(_persistent(environment))
    replay = _executor(environment, replay_retriever).execute(lost_job)
    assert replay.replayed is True
    assert replay.persisted_run_id == durable["persisted_run_id"]
    assert replay_retriever.requests == []
    assert len(set(replay.reference_ids)) == replay.reference_count
    assert _payload_count(environment, lost_job) == 0


def test_worker_dispatch_and_observable_values_are_redacted(
    environment: ControlEnvironment,
    caplog: pytest.LogCaptureFixture,
) -> None:
    job_id = _submit(environment, suffix="redaction")
    executor = _executor(environment, RaisingRetriever())
    with pytest.raises(BaselineRunExecutorError):
        executor.execute(job_id)
    status = _service(environment).read_status(
        request_id=str(uuid4()),
        group_id=environment.group_id,
        job_id=job_id,
        caller_user_id=environment.user_id,
    )
    rendered = " ".join(
        (json.dumps(status, sort_keys=True), repr(executor.identity), caplog.text)
    )
    assert RAW_QUERY not in rendered
    assert "parent_processing_secret" not in rendered
    assert "ciphertext" not in rendered
    assert "lease_token" not in rendered
