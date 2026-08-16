from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from compair_core import db as core_db
from compair_core.compair.retrieval.evidence_persistence import (
    BaselineEvidencePersistenceCommand,
    BaselineEvidencePersistenceService,
)
from compair_core.compair.retrieval.generation import (
    GENERATION_FINDING_SEPARATOR,
    BaselineGenerationBusyError,
    BaselineGenerationCommand,
    BaselineGenerationError,
    BaselineGenerationProviderError,
    BaselineGenerationService,
    BaselineGenerationState,
    GenerationWriteStage,
    ReviewerBaselineGenerationProvider,
)
from compair_core.compair.retrieval import generation as generation_module
from test_baseline_evidence_persistence import make_persistence_environment


class CapturingProvider:
    provider = "fixture-generation"
    model = "fixture-reviewer"
    version = "fixture-reviewer-r1"
    supports_idempotency = False

    def __init__(self, output: str = "first finding") -> None:
        self.output = output
        self.inputs = []
        self.idempotency_keys = []

    def generate(self, generation_input, *, idempotency_key: str) -> str:
        self.inputs.append(generation_input)
        self.idempotency_keys.append(idempotency_key)
        return self.output


class FailingProvider(CapturingProvider):
    def __init__(self, *, retryable: bool) -> None:
        super().__init__()
        self.retryable = retryable

    def generate(self, generation_input, *, idempotency_key: str) -> str:
        self.inputs.append(generation_input)
        self.idempotency_keys.append(idempotency_key)
        raise BaselineGenerationProviderError(
            "fixture_provider_failed",
            "safe fixture failure",
            retryable=self.retryable,
        )


def _environment(tmp_path: Path, name: str = "generation.db"):
    engine = core_db.create_engine(
        f"sqlite:///{tmp_path / name}",
        connect_args={"check_same_thread": False, "timeout": 15},
    )
    return make_persistence_environment(engine)


def _caller_user_id(environment) -> str:
    with environment.engine.connect() as connection:
        return str(
            connection.execute(
                text(
                    "SELECT user_id FROM user_to_group "
                    "WHERE group_id = :group_id ORDER BY user_id"
                ),
                {"group_id": environment.group_id},
            ).scalar_one()
        )


def _persist(environment, key: str = "generation-intent"):
    caller = _caller_user_id(environment)
    receipt = BaselineEvidencePersistenceService(environment.sessions).persist(
        BaselineEvidencePersistenceCommand(
            group_id=environment.group_id,
            source_chunk_id=environment.source_chunk_id,
            source_document_id=environment.source_document_id,
            idempotency_key=key,
            retrieval_result=environment.result,
            caller_user_id=caller,
        )
    )
    return receipt, BaselineGenerationCommand(
        run_id=receipt.run_id,
        group_id=environment.group_id,
        caller_user_id=caller,
    )


def _feedback_rows(environment, run_id: str):
    with environment.engine.connect() as connection:
        return connection.execute(
            text(
                "SELECT feedback_id, feedback, baseline_finding_ordinal, "
                "generation_provider, generation_model, generation_model_version, "
                "generation_input_fingerprint, generation_output_fingerprint "
                "FROM feedback WHERE baseline_retrieval_run_id = :run_id "
                "ORDER BY baseline_finding_ordinal"
            ),
            {"run_id": run_id},
        ).mappings().all()


def test_exact_order_renderer_bytes_success_replay_and_restart(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    try:
        persisted, command = _persist(environment)
        provider = CapturingProvider(
            f"first finding{GENERATION_FINDING_SEPARATOR}second finding"
        )
        service = BaselineGenerationService(environment.sessions)

        receipt = service.generate(command, provider)
        replay = service.generate(command, provider)

        assert receipt.state is BaselineGenerationState.SUCCEEDED
        assert receipt.attempt_count == 1
        assert len(receipt.feedback_ids) == 2
        assert replay.state is BaselineGenerationState.SUCCEEDED
        assert replay.replayed is True
        assert replay.feedback_ids == receipt.feedback_ids
        assert len(provider.inputs) == 1
        generation_input = provider.inputs[0]
        assert [item.ordinal for item in generation_input.evidence] == [1, 2, 3, 4]
        with environment.engine.connect() as connection:
            stored = connection.execute(
                text(
                    "SELECT ordinal, renderer_output, a.repository_name, a.relative_path "
                    "FROM baseline_selected_evidence s "
                    "JOIN baseline_evidence_artifact a ON a.artifact_id = s.artifact_id "
                    "WHERE s.run_id = :run_id ORDER BY s.ordinal"
                ),
                {"run_id": persisted.run_id},
            ).all()
            assert connection.execute(
                text("SELECT count(*) FROM notification_event")
            ).scalar_one() == 0
        assert [item.renderer_output for item in generation_input.evidence] == [
            row.renderer_output for row in stored
        ]
        assert [
            (item.repository_name, item.relative_path)
            for item in generation_input.evidence
        ] == [(row.repository_name, row.relative_path) for row in stored]
        assert sum(len(item.renderer_output) for item in generation_input.evidence) == sum(
            len(row.renderer_output) for row in stored
        )
        rows = _feedback_rows(environment, persisted.run_id)
        assert [row["feedback"] for row in rows] == ["first finding", "second finding"]
        assert [row["baseline_finding_ordinal"] for row in rows] == [1, 2]
        assert all(row["generation_provider"] == provider.provider for row in rows)
        assert all(row["generation_model"] == provider.model for row in rows)
        assert all(row["generation_model_version"] == provider.version for row in rows)
        assert all(
            row["generation_input_fingerprint"] == receipt.input_fingerprint
            for row in rows
        )
        assert all(
            row["generation_output_fingerprint"] == receipt.output_fingerprint
            for row in rows
        )

        environment.engine.dispose()
        restarted_engine = core_db.create_engine(
            f"sqlite:///{tmp_path / 'generation.db'}",
            connect_args={"check_same_thread": False, "timeout": 15},
        )
        restarted_sessions = core_db.sessionmaker(
            restarted_engine, expire_on_commit=False
        )
        restarted = BaselineGenerationService(restarted_sessions).generate(
            command, provider
        )
        assert restarted.state is BaselineGenerationState.SUCCEEDED
        assert restarted.feedback_ids == receipt.feedback_ids
        assert len(provider.inputs) == 1
        restarted_engine.dispose()
    finally:
        environment.engine.dispose()


def test_configured_http_adapter_sends_ordered_renderer_values_verbatim(
    tmp_path: Path,
    monkeypatch,
) -> None:
    environment = _environment(tmp_path, "http-adapter.db")
    try:
        persisted, command = _persist(environment, "http-adapter")
        captured = {}

        class Response:
            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def json():
                return {"feedback": "HTTP baseline finding"}

        def post(endpoint, *, json, timeout):
            captured.update(endpoint=endpoint, payload=json, timeout=timeout)
            return Response()

        monkeypatch.setattr(generation_module.requests, "post", post)
        reviewer = SimpleNamespace(
            provider="local",
            model="local-fixture",
            endpoint="http://127.0.0.1:9000/generate",
        )
        provider = ReviewerBaselineGenerationProvider(reviewer)
        receipt = BaselineGenerationService(environment.sessions).generate(
            command, provider
        )
        assert receipt.state is BaselineGenerationState.SUCCEEDED
        with environment.engine.connect() as connection:
            stored = connection.execute(
                text(
                    "SELECT renderer_output FROM baseline_selected_evidence "
                    "WHERE run_id = :run_id ORDER BY ordinal"
                ),
                {"run_id": persisted.run_id},
            ).scalars().all()
        assert captured["endpoint"] == reviewer.endpoint
        assert captured["payload"]["references"] == stored
        assert captured["payload"]["contract_version"] == (
            "baseline-generation-input.v1"
        )
        assert captured["payload"]["document"] == "authoritative source chunk"
        assert len(captured["payload"]["idempotency_key"]) == 64
    finally:
        environment.engine.dispose()


@pytest.mark.parametrize(
    ("retryable", "expected_state"),
    [
        (True, BaselineGenerationState.RETRYABLE_FAILED),
        (False, BaselineGenerationState.TERMINAL_FAILED),
    ],
)
def test_provider_failure_is_sanitized_and_retry_safe(
    tmp_path: Path,
    retryable: bool,
    expected_state: BaselineGenerationState,
) -> None:
    environment = _environment(tmp_path, f"provider-{retryable}.db")
    try:
        persisted, command = _persist(environment, f"provider-{retryable}")
        failed = BaselineGenerationService(environment.sessions).generate(
            command, FailingProvider(retryable=retryable)
        )
        assert failed.state is expected_state
        assert failed.error_code == "fixture_provider_failed"
        assert _feedback_rows(environment, persisted.run_id) == []
        with environment.engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT generation_error_code, generation_error_fingerprint, "
                    "generation_lease_token FROM baseline_retrieval_run "
                    "WHERE run_id = :run_id"
                ),
                {"run_id": persisted.run_id},
            ).one()
        assert row.generation_error_code == "fixture_provider_failed"
        assert len(row.generation_error_fingerprint) == 64
        assert row.generation_lease_token is None

        if retryable:
            recovered = BaselineGenerationService(environment.sessions).generate(
                command, CapturingProvider("recovered finding")
            )
            assert recovered.state is BaselineGenerationState.SUCCEEDED
            assert recovered.attempt_count == 2
            assert len(_feedback_rows(environment, persisted.run_id)) == 1
    finally:
        environment.engine.dispose()


def test_expired_lease_is_recovered_transactionally(tmp_path: Path) -> None:
    environment = _environment(tmp_path, "expired.db")
    try:
        persisted, command = _persist(environment, "expired-lease")
        with environment.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE baseline_retrieval_run SET generation_state = 'running', "
                    "generation_lease_token = 'abandoned', generation_attempt_count = 1, "
                    "generation_lease_expires_at = :expired WHERE run_id = :run_id"
                ),
                {
                    "expired": datetime.now(timezone.utc) - timedelta(minutes=1),
                    "run_id": persisted.run_id,
                },
            )
        receipt = BaselineGenerationService(environment.sessions).generate(
            command, CapturingProvider()
        )
        assert receipt.state is BaselineGenerationState.SUCCEEDED
        assert receipt.attempt_count == 2
        assert len(_feedback_rows(environment, persisted.run_id)) == 1
    finally:
        environment.engine.dispose()


def test_authorization_is_rechecked_after_provider_before_feedback(tmp_path: Path) -> None:
    environment = _environment(tmp_path, "authorization.db")
    try:
        persisted, command = _persist(environment, "authorization")

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
    finally:
        environment.engine.dispose()


def test_deleted_source_blocks_before_provider_and_creates_no_feedback(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path, "deleted-source.db")
    try:
        persisted, command = _persist(environment, "deleted-source")
        with environment.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM document WHERE document_id = :document_id"),
                {"document_id": environment.source_document_id},
            )
        provider = CapturingProvider()
        receipt = BaselineGenerationService(environment.sessions).generate(
            command, provider
        )
        assert receipt.state is BaselineGenerationState.BLOCKED
        assert receipt.error_code == "generation_source_deleted"
        assert provider.inputs == []
        assert _feedback_rows(environment, persisted.run_id) == []
    finally:
        environment.engine.dispose()


def test_malformed_output_is_terminal_and_creates_no_feedback(tmp_path: Path) -> None:
    environment = _environment(tmp_path, "malformed.db")
    try:
        persisted, command = _persist(environment, "malformed")
        receipt = BaselineGenerationService(environment.sessions).generate(
            command, CapturingProvider("NONE")
        )
        assert receipt.state is BaselineGenerationState.TERMINAL_FAILED
        assert receipt.error_code == "provider_empty_output"
        assert _feedback_rows(environment, persisted.run_id) == []
    finally:
        environment.engine.dispose()


def test_stale_publication_blocks_before_provider(tmp_path: Path) -> None:
    environment = _environment(tmp_path, "stale-publication.db")
    try:
        persisted, command = _persist(environment, "stale-publication")
        with environment.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE retrieval_baseline_index_publication SET index_id = NULL "
                    "WHERE corpus_id = (SELECT corpus_id FROM baseline_retrieval_run "
                    "WHERE run_id = :run_id)"
                ),
                {"run_id": persisted.run_id},
            )
        provider = CapturingProvider()
        receipt = BaselineGenerationService(environment.sessions).generate(
            command, provider
        )
        assert receipt.state is BaselineGenerationState.BLOCKED
        assert receipt.error_code == "generation_publication_stale"
        assert provider.inputs == []
        assert _feedback_rows(environment, persisted.run_id) == []
    finally:
        environment.engine.dispose()


@pytest.mark.parametrize("stage", list(GenerationWriteStage))
def test_injected_feedback_transaction_failure_rolls_back(
    tmp_path: Path,
    stage: GenerationWriteStage,
) -> None:
    environment = _environment(tmp_path, f"rollback-{stage.value}.db")
    try:
        persisted, command = _persist(environment, f"rollback-{stage.value}")

        def fail(selected: GenerationWriteStage) -> None:
            if selected is stage:
                raise RuntimeError("injected database failure with sensitive details")

        with pytest.raises(BaselineGenerationError) as error:
            BaselineGenerationService(
                environment.sessions, stage_hook=fail
            ).generate(command, CapturingProvider())
        assert error.value.code == "database_commit_failed"
        assert _feedback_rows(environment, persisted.run_id) == []
        with environment.engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT generation_state, generation_error_code "
                    "FROM baseline_retrieval_run WHERE run_id = :run_id"
                ),
                {"run_id": persisted.run_id},
            ).one()
        assert row == ("retryable_failed", "database_commit_failed")

        recovered = BaselineGenerationService(environment.sessions).generate(
            command, CapturingProvider("retry succeeded")
        )
        assert recovered.state is BaselineGenerationState.SUCCEEDED
        assert recovered.attempt_count == 2
        assert len(_feedback_rows(environment, persisted.run_id)) == 1
    finally:
        environment.engine.dispose()


def test_concurrent_attempt_observes_active_lease_and_does_not_duplicate(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path, "concurrent.db")
    try:
        persisted, command = _persist(environment, "concurrent")
        started = threading.Event()
        release = threading.Event()

        class BlockingProvider(CapturingProvider):
            def generate(self, generation_input, *, idempotency_key: str) -> str:
                self.inputs.append(generation_input)
                started.set()
                assert release.wait(timeout=10)
                return self.output

        provider = BlockingProvider()
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(
                BaselineGenerationService(environment.sessions).generate,
                command,
                provider,
            )
            assert started.wait(timeout=10)
            with pytest.raises(BaselineGenerationBusyError) as error:
                BaselineGenerationService(environment.sessions).generate(
                    command, CapturingProvider()
                )
            assert error.value.code == "generation_lease_active"
            release.set()
            first = future.result(timeout=15)
        assert first.state is BaselineGenerationState.SUCCEEDED
        assert len(_feedback_rows(environment, persisted.run_id)) == 1
        assert len(provider.inputs) == 1
    finally:
        environment.engine.dispose()
