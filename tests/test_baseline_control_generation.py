from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import text
from test_baseline_evidence_persistence import (
    control_command,
    seed_running_control_job,
)
from test_baseline_generation import (
    CapturingProvider,
    FailingProvider,
    RawOutputProvider,
    _environment,
)

from compair_core.baseline_generation.profile import (
    CPU_GENERATION_TIMEOUT_SECONDS,
    required_generation_lease_seconds,
)
from compair_core.compair.retrieval import generation as generation_module
from compair_core.compair.retrieval.evidence_persistence import (
    BaselineEvidencePersistenceService,
)
from compair_core.compair.retrieval.generation import (
    GENERATION_OUTPUT_SCHEMA_VERSION,
    BaselineGenerationBusyError,
    BaselineGenerationError,
    BaselineGenerationService,
    GenerationWriteStage,
)


def _persist_control(environment):
    job_id, lease_token, caller = seed_running_control_job(environment)
    persisted = BaselineEvidencePersistenceService(environment.sessions).persist(
        control_command(
            environment,
            job_id=job_id,
            lease_token=lease_token,
            caller_user_id=caller,
        )
    )
    return job_id, caller, persisted


def _structured(outcome: str, findings: list[str]) -> str:
    return json.dumps(
        {
            "schema_version": GENERATION_OUTPUT_SCHEMA_VERSION,
            "outcome": outcome,
            "findings": [{"feedback": finding} for finding in findings],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _state(environment, job_id: str, run_id: str):
    with environment.engine.connect() as connection:
        job = (
            connection.execute(
                text("SELECT * FROM baseline_control_run_job WHERE job_id = :job_id"),
                {"job_id": job_id},
            )
            .mappings()
            .one()
        )
        run = (
            connection.execute(
                text("SELECT * FROM baseline_retrieval_run WHERE run_id = :run_id"),
                {"run_id": run_id},
            )
            .mappings()
            .one()
        )
        feedback = (
            connection.execute(
                text(
                    "SELECT feedback_id, feedback, baseline_finding_ordinal FROM feedback "
                    "WHERE baseline_retrieval_run_id = :run_id "
                    "ORDER BY baseline_finding_ordinal"
                ),
                {"run_id": run_id},
            )
            .mappings()
            .all()
        )
        outbox = (
            connection.execute(
                text(
                    "SELECT outbox_id, state, finding_count FROM "
                    "baseline_notification_outbox WHERE run_id = :run_id"
                ),
                {"run_id": run_id},
            )
            .mappings()
            .all()
        )
        notifications = connection.execute(
            text("SELECT count(*) FROM notification_event")
        ).scalar_one()
    return job, run, feedback, outbox, notifications


class CoordinatedLeaseInspectingProvider(CapturingProvider):
    def __init__(self, environment, job_id: str, run_id: str) -> None:
        super().__init__("coordinated lease-safe finding")
        self.environment = environment
        self.job_id = job_id
        self.run_id = run_id
        self.control_remaining: timedelta | None = None
        self.generation_remaining: timedelta | None = None

    @staticmethod
    def _aware(value: datetime | str) -> datetime:
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)

    def generate(self, generation_input, *, idempotency_key: str) -> str:
        with self.environment.engine.connect() as connection:
            control_expiry = connection.execute(
                text(
                    "SELECT lease_expires_at FROM baseline_control_run_job "
                    "WHERE job_id = :job_id"
                ),
                {"job_id": self.job_id},
            ).scalar_one()
            generation_expiry = connection.execute(
                text(
                    "SELECT generation_lease_expires_at "
                    "FROM baseline_retrieval_run WHERE run_id = :run_id"
                ),
                {"run_id": self.run_id},
            ).scalar_one()
        now = datetime.now(timezone.utc)
        self.control_remaining = self._aware(control_expiry) - now
        self.generation_remaining = self._aware(generation_expiry) - now
        return super().generate(generation_input, idempotency_key=idempotency_key)


def test_control_generation_positive_is_ordered_atomic_and_replayed(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path, "control-generation-positive.db")
    try:
        job_id, _caller, persisted = _persist_control(environment)
        provider = CapturingProvider("first finding", "second finding")
        service = BaselineGenerationService(
            environment.sessions, notifications_enabled=False
        )

        receipt = service.generate_control(job_id, provider)
        replay = service.generate_control(job_id, provider)

        assert receipt.state == "feedback_persisted"
        assert receipt.feedback_ids
        assert replay.replayed is True
        assert replay.feedback_ids == receipt.feedback_ids
        assert len(provider.inputs) == 1
        with environment.engine.connect() as connection:
            renderer_outputs = (
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
            payload_count = connection.execute(
                text(
                    "SELECT count(*) FROM baseline_control_run_payload "
                    "WHERE job_id = :job_id"
                ),
                {"job_id": job_id},
            ).scalar_one()
        assert [item.renderer_output for item in provider.inputs[0].evidence] == list(
            renderer_outputs
        )
        assert payload_count == 0
        job, run, feedback, outbox, notifications = _state(
            environment, job_id, persisted.run_id
        )
        assert job["state"] == "feedback_persisted"
        assert bool(job["generation_invoked"]) is True
        assert job["generation_contract_version"] == "baseline-control-generation.v1"
        assert job["feedback_count"] == 2
        assert job["notification_outbox_count"] == 1
        assert job["lease_token"] is None
        assert run["generation_state"] == "succeeded"
        assert [row["feedback"] for row in feedback] == [
            "first finding",
            "second finding",
        ]
        assert [row["baseline_finding_ordinal"] for row in feedback] == [1, 2]
        assert len(outbox) == 1
        assert outbox[0]["state"] == "suppressed"
        assert outbox[0]["finding_count"] == 2
        assert notifications == 0
    finally:
        environment.engine.dispose()


def test_control_and_generation_leases_cover_cpu_provider_timeout(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path, "control-cpu-timeout-lease.db")
    try:
        job_id, _caller, persisted = _persist_control(environment)
        provider = CoordinatedLeaseInspectingProvider(
            environment,
            job_id,
            persisted.run_id,
        )
        service = BaselineGenerationService(
            environment.sessions,
            lease_seconds=required_generation_lease_seconds(
                CPU_GENERATION_TIMEOUT_SECONDS
            ),
            provider_timeout_seconds=CPU_GENERATION_TIMEOUT_SECONDS,
        )
        receipt = service.generate_control(job_id, provider)
        assert receipt.state == "feedback_persisted"
        assert provider.control_remaining is not None
        assert provider.generation_remaining is not None
        assert provider.control_remaining > timedelta(seconds=350)
        assert provider.generation_remaining > timedelta(seconds=350)
    finally:
        environment.engine.dispose()


def test_control_generation_zero_findings_is_success_without_rows(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path, "control-generation-zero.db")
    try:
        job_id, _caller, persisted = _persist_control(environment)
        provider = RawOutputProvider(_structured("no_findings", []))
        receipt = BaselineGenerationService(
            environment.sessions, notifications_enabled=False
        ).generate_control(job_id, provider)
        replay = BaselineGenerationService(
            environment.sessions, notifications_enabled=False
        ).generate_control(job_id, provider)

        assert receipt.state == "feedback_persisted"
        assert receipt.feedback_ids == ()
        assert receipt.notification_outbox_count == 0
        assert replay.replayed is True
        assert len(provider.inputs) == 1
        job, run, feedback, outbox, notifications = _state(
            environment, job_id, persisted.run_id
        )
        assert bool(job["generation_invoked"]) is True
        assert job["feedback_count"] == 0
        assert job["notification_outbox_count"] == 0
        assert run["generation_state"] == "succeeded"
        assert feedback == []
        assert outbox == []
        assert notifications == 0
    finally:
        environment.engine.dispose()


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "NONE",
        '"NONE"',
        "plain text",
        "{",
        (
            '{"schema_version":"baseline-generation-output.v2",'
            '"outcome":"no_findings","outcome":"findings","findings":[]}'
        ),
        (
            '{"schema_version":"baseline-generation-output.v2",'
            '"outcome":"no_findings","findings":[],"extra":true}'
        ),
        _structured("findings", []),
        _structured("no_findings", ["contradiction"]),
        _structured("findings", ["   "]),
        _structured("findings", ["NONE"]),
        (
            '{"schema_version":"baseline-generation-output.v2",'
            '"outcome":"findings","findings":[],"number":NaN}'
        ),
    ],
)
def test_strict_structured_output_rejects_invalid_values(
    tmp_path: Path, raw: str
) -> None:
    environment = _environment(tmp_path, f"invalid-{abs(hash(raw))}.db")
    try:
        job_id, _caller, persisted = _persist_control(environment)
        receipt = BaselineGenerationService(environment.sessions).generate_control(
            job_id, RawOutputProvider(raw)
        )
        assert receipt.state == "terminal_failed"
        assert receipt.error_code == "provider_malformed_output"
        job, run, feedback, outbox, _notifications = _state(
            environment, job_id, persisted.run_id
        )
        assert job["reason_code"] == "provider_malformed_output"
        assert run["generation_state"] == "terminal_failed"
        assert feedback == []
        assert outbox == []
    finally:
        environment.engine.dispose()


def test_findings_cannot_exceed_reference_count(tmp_path: Path) -> None:
    environment = _environment(tmp_path, "too-many-findings.db")
    try:
        job_id, _caller, _persisted = _persist_control(environment)
        receipt = BaselineGenerationService(environment.sessions).generate_control(
            job_id, CapturingProvider("one", "two", "three", "four", "five")
        )
        assert receipt.state == "terminal_failed"
        assert receipt.error_code == "provider_malformed_output"
    finally:
        environment.engine.dispose()


def test_transient_failure_retries_without_payload_or_retrieval(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path, "control-generation-retry.db")
    try:
        job_id, _caller, persisted = _persist_control(environment)
        failed = BaselineGenerationService(environment.sessions).generate_control(
            job_id, FailingProvider(retryable=True)
        )
        assert failed.state == "retryable_failed"
        recovered_provider = CapturingProvider("recovered")
        recovered = BaselineGenerationService(environment.sessions).generate_control(
            job_id, recovered_provider
        )
        assert recovered.state == "feedback_persisted"
        assert recovered.generation_attempt_count == 2
        assert len(recovered_provider.inputs) == 1
        job, run, feedback, _outbox, _notifications = _state(
            environment, job_id, persisted.run_id
        )
        assert job["persisted_run_id"] == persisted.run_id
        assert run["generation_attempt_count"] == 2
        assert len(feedback) == 1
        with environment.engine.connect() as connection:
            assert (
                connection.execute(
                    text(
                        "SELECT count(*) FROM baseline_control_run_payload "
                        "WHERE job_id = :job_id"
                    ),
                    {"job_id": job_id},
                ).scalar_one()
                == 0
            )
    finally:
        environment.engine.dispose()


def test_authorization_revocation_before_commit_blocks_atomically(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path, "control-generation-revoked.db")
    try:
        job_id, caller, persisted = _persist_control(environment)

        class RevokingProvider(CapturingProvider):
            def generate(self, generation_input, *, idempotency_key: str) -> str:
                output = super().generate(
                    generation_input, idempotency_key=idempotency_key
                )
                with environment.engine.begin() as connection:
                    connection.execute(
                        text(
                            "DELETE FROM user_to_group WHERE user_id = :user_id "
                            "AND group_id = :group_id"
                        ),
                        {"user_id": caller, "group_id": environment.group_id},
                    )
                return output

        receipt = BaselineGenerationService(environment.sessions).generate_control(
            job_id, RevokingProvider("must not persist")
        )
        assert receipt.state == "blocked"
        job, run, feedback, outbox, _notifications = _state(
            environment, job_id, persisted.run_id
        )
        assert job["reason_code"] == "generation_authorization_revoked"
        assert run["generation_state"] == "blocked"
        assert feedback == []
        assert outbox == []
    finally:
        environment.engine.dispose()


@pytest.mark.parametrize("stage", list(GenerationWriteStage))
def test_atomic_precommit_crash_rolls_back_and_retry_is_safe(
    tmp_path: Path, stage: GenerationWriteStage
) -> None:
    environment = _environment(tmp_path, f"control-generation-{stage.value}.db")
    try:
        job_id, _caller, persisted = _persist_control(environment)

        def fail(stage: GenerationWriteStage) -> None:
            if stage is selected_stage:
                raise RuntimeError("private provider response must not escape")

        selected_stage = stage

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
            job_id, CapturingProvider("committed")
        )
        assert recovered.state == "feedback_persisted"
        assert recovered.generation_attempt_count == 2
    finally:
        environment.engine.dispose()


def test_provider_idempotency_attestation_and_key_are_durable(tmp_path: Path) -> None:
    environment = _environment(tmp_path, "control-generation-idempotency.db")
    try:
        job_id, _caller, persisted = _persist_control(environment)

        class SupportedProvider(FailingProvider):
            supports_idempotency = True

        first = SupportedProvider(retryable=True)
        failed = BaselineGenerationService(environment.sessions).generate_control(
            job_id, first
        )
        assert failed.state == "retryable_failed"

        class RecoveredProvider(CapturingProvider):
            supports_idempotency = True

        second = RecoveredProvider("durably idempotent")
        succeeded = BaselineGenerationService(environment.sessions).generate_control(
            job_id, second
        )
        assert succeeded.state == "feedback_persisted"
        assert first.idempotency_keys == second.idempotency_keys
        job, _run, _feedback, _outbox, _notifications = _state(
            environment, job_id, persisted.run_id
        )
        assert bool(job["generation_provider_idempotency_supported"]) is True
        assert len(job["generation_provider_fingerprint"]) == 64
    finally:
        environment.engine.dispose()


def test_commit_success_response_loss_recovers_without_provider_recall(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path, "control-generation-response-loss.db")
    try:
        job_id, _caller, persisted = _persist_control(environment)
        provider = CapturingProvider("committed before response loss")

        class LostResponseService(BaselineGenerationService):
            def _commit_control_feedback(self, *args, **kwargs):
                super()._commit_control_feedback(*args, **kwargs)
                raise RuntimeError("simulated process loss after durable commit")

        with pytest.raises(BaselineGenerationError) as error:
            LostResponseService(environment.sessions).generate_control(job_id, provider)
        assert error.value.code == "database_commit_failed"
        replay = BaselineGenerationService(environment.sessions).generate_control(
            job_id, provider
        )
        assert replay.state == "feedback_persisted"
        assert replay.replayed is True
        assert len(provider.inputs) == 1
        _job, run, feedback, outbox, _notifications = _state(
            environment, job_id, persisted.run_id
        )
        assert run["generation_state"] == "succeeded"
        assert len(feedback) == 1
        assert len(outbox) == 1
    finally:
        environment.engine.dispose()


def test_revoked_approval_blocks_before_provider_and_status_is_content_free(
    tmp_path: Path, caplog
) -> None:
    environment = _environment(tmp_path, "control-generation-approval.db")
    try:
        job_id, _caller, persisted = _persist_control(environment)
        provider = CapturingProvider("private finding must not appear")
        with environment.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE baseline_control_repository_approval SET state = 'disabled', "
                    "disabled_at = CURRENT_TIMESTAMP "
                    "WHERE registration_id = (SELECT "
                    "changed_repository_registration_id FROM baseline_control_run_job "
                    "WHERE job_id = :job_id)"
                ),
                {"job_id": job_id},
            )
        receipt = BaselineGenerationService(environment.sessions).generate_control(
            job_id, provider
        )
        assert receipt.state == "blocked"
        assert provider.inputs == []
        job, run, feedback, outbox, _notifications = _state(
            environment, job_id, persisted.run_id
        )
        safe_status = json.dumps(dict(job), default=str)
        for forbidden in (
            "private finding must not appear",
            "authoritative source document",
            "alpha persistence query",
            "alpha evidence file",
        ):
            assert forbidden not in safe_status
            assert forbidden not in caplog.text
        assert run["generation_state"] == "blocked"
        assert feedback == []
        assert outbox == []
    finally:
        environment.engine.dispose()


def test_expired_lease_reclaim_and_stale_holder_cannot_commit(tmp_path: Path) -> None:
    environment = _environment(tmp_path, "control-generation-expiry.db")
    try:
        job_id, _caller, persisted = _persist_control(environment)
        service = BaselineGenerationService(environment.sessions, lease_seconds=60)
        provider = CapturingProvider("first")
        claimed = service._claim_control_generation(
            job_id,
            provider_name=provider.provider,
            model=provider.model,
            version=provider.version,
            supports_idempotency=False,
            provider_fingerprint=generation_module._provider_fingerprint(
                provider.provider, provider.model, provider.version, False
            ),
        )
        assert isinstance(claimed, tuple)
        old_token, command, old_input, _attempt = claimed
        expired = datetime.now(timezone.utc) - timedelta(seconds=1)
        with environment.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE baseline_control_run_job SET lease_expires_at = :expired "
                    "WHERE job_id = :job_id"
                ),
                {"expired": expired, "job_id": job_id},
            )
            connection.execute(
                text(
                    "UPDATE baseline_retrieval_run SET generation_lease_expires_at = :expired "
                    "WHERE run_id = :run_id"
                ),
                {"expired": expired, "run_id": persisted.run_id},
            )
        recovered = BaselineGenerationService(environment.sessions).generate_control(
            job_id, CapturingProvider("reclaimed")
        )
        assert recovered.state == "feedback_persisted"
        with pytest.raises(BaselineGenerationBusyError):
            service._commit_control_feedback(
                job_id,
                command=command,
                lease_token=old_token,
                expected_input=old_input,
                findings=("stale",),
                output_fingerprint="0" * 64,
                provider_name=provider.provider,
                model=provider.model,
                version=provider.version,
                provider_fingerprint="0" * 64,
                supports_idempotency=False,
            )
    finally:
        environment.engine.dispose()


def test_concurrent_control_generation_has_one_provider_call(tmp_path: Path) -> None:
    environment = _environment(tmp_path, "control-generation-concurrent.db")
    try:
        job_id, _caller, persisted = _persist_control(environment)
        started = threading.Event()
        release = threading.Event()

        class BlockingProvider(CapturingProvider):
            def generate(self, generation_input, *, idempotency_key: str) -> str:
                self.inputs.append(generation_input)
                self.idempotency_keys.append(idempotency_key)
                started.set()
                assert release.wait(timeout=10)
                return self.output

        provider = BlockingProvider("only finding")
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(
                BaselineGenerationService(environment.sessions).generate_control,
                job_id,
                provider,
            )
            assert started.wait(timeout=10)
            with pytest.raises(BaselineGenerationBusyError):
                BaselineGenerationService(environment.sessions).generate_control(
                    job_id, CapturingProvider("duplicate")
                )
            release.set()
            assert future.result(timeout=10).state == "feedback_persisted"
        assert len(provider.inputs) == 1
        _job, _run, feedback, _outbox, _notifications = _state(
            environment, job_id, persisted.run_id
        )
        assert len(feedback) == 1
    finally:
        environment.engine.dispose()
