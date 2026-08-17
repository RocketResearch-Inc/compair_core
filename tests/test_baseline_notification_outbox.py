from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from test_baseline_generation import (
    CapturingProvider,
    FailingProvider,
    _environment,
    _persist,
)

from compair_core.compair.retrieval.evidence_persistence import (
    BaselineEvidencePersistenceCommand,
    BaselineEvidencePersistenceError,
    BaselineEvidencePersistenceService,
    LegacyChunkSource,
)
from compair_core.compair.retrieval.generation import (
    BaselineGenerationService,
    BaselineGenerationState,
)
from compair_core.compair.retrieval.notification_outbox import (
    BASELINE_NOTIFICATION_CHANNEL,
    BASELINE_NOTIFICATION_OUTBOX_TABLE,
    BASELINE_NOTIFICATION_PAYLOAD_SCHEMA_VERSION,
    BaselineNotificationOutboxDispatcher,
    BaselineNotificationOutboxError,
    BaselineNotificationSinkError,
    BaselineNotificationState,
    baseline_notifications_enabled,
    load_authorized_baseline_notification_digest,
)
from compair_core.compair.retrieval.types import RetrievalStatus


class CapturingSink:
    channel = BASELINE_NOTIFICATION_CHANNEL
    supports_idempotency = False

    def __init__(self) -> None:
        self.digests = []
        self.idempotency_keys = []

    def deliver(self, digest, *, idempotency_key: str) -> None:
        self.digests.append(digest)
        self.idempotency_keys.append(idempotency_key)


class FailingSink(CapturingSink):
    def __init__(self, *, retryable: bool) -> None:
        super().__init__()
        self.retryable = retryable

    def deliver(self, digest, *, idempotency_key: str) -> None:
        super().deliver(digest, idempotency_key=idempotency_key)
        raise BaselineNotificationSinkError(
            "fixture_sink_failed",
            "fixture detail that must not be persisted",
            retryable=self.retryable,
        )


def _outbox_rows(environment):
    with environment.engine.connect() as connection:
        return (
            connection.execute(
                text(
                    "SELECT * FROM baseline_notification_outbox "
                    "ORDER BY created_at, outbox_id"
                )
            )
            .mappings()
            .all()
        )


def _successful_generation(
    environment, *, enabled: bool, output: str | tuple[str, ...]
):
    findings = (output,) if isinstance(output, str) else output
    persisted, command = _persist(environment, f"notify-{findings[0][:12]}")
    receipt = BaselineGenerationService(
        environment.sessions,
        notifications_enabled=enabled,
    ).generate(command, CapturingProvider(*findings))
    assert receipt.state is BaselineGenerationState.SUCCEEDED
    return persisted, command, receipt


def test_default_off_creates_one_privacy_safe_suppressed_digest_and_replay(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.delenv("COMPAIR_BASELINE_NOTIFICATIONS_ENABLED", raising=False)
    assert baseline_notifications_enabled() is False
    environment = _environment(tmp_path, "default-off.db")
    try:
        persisted, command = _persist(environment, "default-off")
        provider = CapturingProvider("first private finding", "second private finding")
        service = BaselineGenerationService(environment.sessions)
        first = service.generate(command, provider)
        replay = service.generate(command, provider)

        assert replay.replayed is True
        assert replay.feedback_ids == first.feedback_ids
        rows = _outbox_rows(environment)
        assert len(rows) == 1
        row = rows[0]
        assert row["run_id"] == persisted.run_id
        assert row["recipient_user_id"] == command.caller_user_id
        assert row["state"] == BaselineNotificationState.SUPPRESSED.value
        assert row["error_code"] == "baseline_notifications_disabled"
        manifest = json.loads(row["finding_manifest"])
        assert manifest == {
            "findings": [
                {"feedback_id": first.feedback_ids[0], "ordinal": 1},
                {"feedback_id": first.feedback_ids[1], "ordinal": 2},
            ],
            "schema_version": BASELINE_NOTIFICATION_PAYLOAD_SCHEMA_VERSION,
        }
        serialized = json.dumps(dict(row), default=str)
        for forbidden in (
            "alpha persistence query",
            "authoritative source chunk",
            "alpha evidence file",
            "first private finding",
            "second private finding",
        ):
            assert forbidden not in serialized
            assert forbidden not in caplog.text
        assert inspect(environment.engine).has_table(BASELINE_NOTIFICATION_OUTBOX_TABLE)
        with environment.engine.connect() as connection:
            assert (
                connection.execute(
                    text("SELECT count(*) FROM notification_event")
                ).scalar_one()
                == 0
            )
    finally:
        environment.engine.dispose()


def test_enabled_dispatch_and_authenticated_read_preserve_finding_order(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path, "deliver.db")
    try:
        persisted, command, generated = _successful_generation(
            environment,
            enabled=True,
            output=("finding one", "finding two"),
        )
        sink = CapturingSink()
        receipt = BaselineNotificationOutboxDispatcher(
            environment.sessions, enabled=True
        ).dispatch_one(sink)

        assert receipt is not None
        assert receipt.state is BaselineNotificationState.DELIVERED
        assert len(sink.digests) == 1
        digest = sink.digests[0]
        assert digest.run_id == persisted.run_id
        assert [(item.ordinal, item.feedback_id) for item in digest.findings] == [
            (1, generated.feedback_ids[0]),
            (2, generated.feedback_ids[1]),
        ]
        assert sink.idempotency_keys == [digest.digest_key]
        with environment.sessions() as session:
            authorized = load_authorized_baseline_notification_digest(
                session,
                outbox_id=digest.outbox_id,
                recipient_user_id=command.caller_user_id,
                group_id=command.group_id,
            )
        assert authorized.findings == digest.findings
        assert (
            BaselineNotificationOutboxDispatcher(
                environment.sessions, enabled=True
            ).dispatch_one(sink)
            is None
        )
        assert len(_outbox_rows(environment)) == 1
    finally:
        environment.engine.dispose()


@pytest.mark.parametrize(
    ("delete_sql", "expected_state"),
    [
        (
            "DELETE FROM user_to_group WHERE user_id = :user_id AND group_id = :group_id",
            BaselineNotificationState.SUPPRESSED,
        ),
        (
            "DELETE FROM chunk WHERE chunk_id = :chunk_id",
            BaselineNotificationState.CANCELLED,
        ),
        (
            'DELETE FROM "user" WHERE user_id = :user_id',
            BaselineNotificationState.CANCELLED,
        ),
    ],
)
def test_delivery_reauthorization_suppresses_or_cancels_deleted_scope(
    tmp_path: Path,
    delete_sql: str,
    expected_state: BaselineNotificationState,
) -> None:
    environment = _environment(tmp_path, f"reauth-{expected_state.value}.db")
    try:
        _persisted, command, _generated = _successful_generation(
            environment, enabled=True, output="reauthorization finding"
        )
        with environment.engine.begin() as connection:
            connection.execute(
                text(delete_sql),
                {
                    "user_id": command.caller_user_id,
                    "group_id": command.group_id,
                    "chunk_id": environment.source_chunk_id,
                },
            )
        sink = CapturingSink()
        receipt = BaselineNotificationOutboxDispatcher(
            environment.sessions, enabled=True
        ).dispatch_one(sink)
        assert receipt is not None
        assert receipt.state is expected_state
        assert sink.digests == []
    finally:
        environment.engine.dispose()


def test_group_privacy_delete_cascades_digest(tmp_path: Path) -> None:
    environment = _environment(tmp_path, "group-delete.db")
    try:
        _successful_generation(environment, enabled=True, output="group finding")
        assert len(_outbox_rows(environment)) == 1
        with environment.engine.begin() as connection:
            connection.execute(
                text('DELETE FROM "group" WHERE group_id = :group_id'),
                {"group_id": environment.group_id},
            )
        assert _outbox_rows(environment) == []
    finally:
        environment.engine.dispose()


def test_no_outbox_for_pending_failed_blocked_or_legacy_rows(tmp_path: Path) -> None:
    environment = _environment(tmp_path, "no-outbox.db")
    try:
        # The environment contains a real legacy Reference but no baseline digest.
        with environment.engine.connect() as connection:
            assert (
                connection.execute(
                    text(
                        "SELECT count(*) FROM reference "
                        "WHERE baseline_selected_evidence_id IS NULL"
                    )
                ).scalar_one()
                == 1
            )
        pending, pending_command = _persist(environment, "pending-no-outbox")
        assert _outbox_rows(environment) == []

        failed = BaselineGenerationService(
            environment.sessions, notifications_enabled=True
        ).generate(pending_command, FailingProvider(retryable=False))
        assert failed.state is BaselineGenerationState.TERMINAL_FAILED
        assert _outbox_rows(environment) == []

        blocked, blocked_command = _persist(environment, "blocked-no-outbox")
        with environment.engine.begin() as connection:
            connection.execute(
                text(
                    "DELETE FROM user_to_group WHERE user_id = :user_id "
                    "AND group_id = :group_id"
                ),
                {
                    "user_id": blocked_command.caller_user_id,
                    "group_id": blocked_command.group_id,
                },
            )
        blocked_receipt = BaselineGenerationService(
            environment.sessions, notifications_enabled=True
        ).generate(blocked_command, CapturingProvider())
        assert blocked_receipt.state is BaselineGenerationState.BLOCKED
        assert _outbox_rows(environment) == []
        assert pending.run_id != blocked.run_id
    finally:
        environment.engine.dispose()


@pytest.mark.parametrize(
    "status", [RetrievalStatus.INSUFFICIENT, RetrievalStatus.ERROR]
)
def test_non_ok_retrieval_results_cannot_schedule_outbox(
    tmp_path: Path,
    status: RetrievalStatus,
) -> None:
    environment = _environment(tmp_path, f"non-ok-{status.value}.db")
    try:
        with environment.engine.connect() as connection:
            caller_user_id = str(
                connection.execute(
                    text(
                        "SELECT user_id FROM user_to_group WHERE group_id = :group_id"
                    ),
                    {"group_id": environment.group_id},
                ).scalar_one()
            )
        command = BaselineEvidencePersistenceCommand(
            group_id=environment.group_id,
            source=LegacyChunkSource(
                document_id=environment.source_document_id,
                chunk_id=environment.source_chunk_id,
            ),
            idempotency_key=f"non-ok-{status.value}",
            retrieval_result=replace(environment.result, status=status),
            caller_user_id=caller_user_id,
        )
        with pytest.raises(BaselineEvidencePersistenceError):
            BaselineEvidencePersistenceService(environment.sessions).persist(command)
        assert _outbox_rows(environment) == []
        with environment.engine.connect() as connection:
            assert (
                connection.execute(
                    text("SELECT count(*) FROM baseline_retrieval_run")
                ).scalar_one()
                == 0
            )
    finally:
        environment.engine.dispose()


@pytest.mark.parametrize(
    ("retryable", "expected"),
    [
        (True, BaselineNotificationState.RETRYABLE_FAILED),
        (False, BaselineNotificationState.TERMINAL_FAILED),
    ],
)
def test_typed_sink_failure_state_and_safe_error_metadata(
    tmp_path: Path,
    retryable: bool,
    expected: BaselineNotificationState,
) -> None:
    environment = _environment(tmp_path, f"sink-{expected.value}.db")
    try:
        _successful_generation(environment, enabled=True, output="sink finding")
        receipt = BaselineNotificationOutboxDispatcher(
            environment.sessions, enabled=True
        ).dispatch_one(FailingSink(retryable=retryable))
        assert receipt is not None
        assert receipt.state is expected
        row = _outbox_rows(environment)[0]
        assert row["error_code"] == "fixture_sink_failed"
        assert "fixture detail" not in json.dumps(dict(row), default=str)
    finally:
        environment.engine.dispose()


def test_concurrent_dispatchers_lease_once(tmp_path: Path) -> None:
    environment = _environment(tmp_path, "concurrent-dispatch.db")
    try:
        _successful_generation(environment, enabled=True, output="concurrent finding")
        entered = threading.Event()
        release = threading.Event()

        class BlockingSink(CapturingSink):
            def deliver(self, digest, *, idempotency_key: str) -> None:
                super().deliver(digest, idempotency_key=idempotency_key)
                entered.set()
                assert release.wait(timeout=10)

        sink = BlockingSink()
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                BaselineNotificationOutboxDispatcher(
                    environment.sessions, enabled=True
                ).dispatch_one,
                sink,
            )
            assert entered.wait(timeout=10)
            second = BaselineNotificationOutboxDispatcher(
                environment.sessions, enabled=True
            ).dispatch_one(sink)
            release.set()
            first_receipt = first.result(timeout=10)
        assert second is None
        assert first_receipt is not None
        assert first_receipt.state is BaselineNotificationState.DELIVERED
        assert len(sink.digests) == 1
        assert len(_outbox_rows(environment)) == 1
    finally:
        environment.engine.dispose()


def test_expired_lease_is_recovered_and_restart_persists(tmp_path: Path) -> None:
    environment = _environment(tmp_path, "lease-restart.db")
    try:
        _successful_generation(environment, enabled=True, output="lease finding")
        with environment.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE baseline_notification_outbox SET state = 'running', "
                    "lease_token = 'abandoned', lease_expires_at = :expired, "
                    "attempt_count = 1"
                ),
                {"expired": datetime.now(timezone.utc) - timedelta(minutes=1)},
            )
        sink = CapturingSink()
        receipt = BaselineNotificationOutboxDispatcher(
            environment.sessions, enabled=True
        ).dispatch_one(sink)
        assert receipt is not None
        assert receipt.state is BaselineNotificationState.DELIVERED
        assert receipt.attempt_count == 2

        environment.engine.dispose()
        rows = _outbox_rows(environment)
        assert rows[0]["state"] == BaselineNotificationState.DELIVERED.value
    finally:
        environment.engine.dispose()


def test_migration_guards_succeeded_run_and_immutable_payload(tmp_path: Path) -> None:
    environment = _environment(tmp_path, "migration-guards.db")
    try:
        persisted, command = _persist(environment, "migration-guard-pending")
        now = datetime.now(timezone.utc)
        with (
            pytest.raises(IntegrityError),
            environment.engine.begin() as connection,
        ):
            connection.execute(
                text(
                    "INSERT INTO baseline_notification_outbox "
                    "(outbox_id, run_id, group_id, recipient_user_id, channel, "
                    "digest_key, payload_schema_version, finding_count, "
                    "finding_manifest, finding_manifest_hash, state, attempt_count, "
                    "created_at, updated_at) VALUES "
                    "('guard-outbox', :run_id, :group_id, :recipient, 'in_app', "
                    ":digest, 'baseline-notification-digest.v1', 1, '{}', :hash, "
                    "'pending', 0, :now, :now)"
                ),
                {
                    "run_id": persisted.run_id,
                    "group_id": environment.group_id,
                    "recipient": command.caller_user_id,
                    "digest": "a" * 64,
                    "hash": "b" * 64,
                    "now": now,
                },
            )

        _successful_generation(environment, enabled=True, output="immutable finding")
        with (
            pytest.raises(IntegrityError),
            environment.engine.begin() as connection,
        ):
            connection.execute(
                text(
                    "UPDATE baseline_notification_outbox "
                    "SET finding_manifest = '{}' WHERE state = 'pending'"
                )
            )
    finally:
        environment.engine.dispose()


def test_invalid_default_off_configuration_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("COMPAIR_BASELINE_NOTIFICATIONS_ENABLED", "sometimes")
    with pytest.raises(BaselineNotificationOutboxError) as error:
        baseline_notifications_enabled()
    assert error.value.code == "baseline_notifications_config_invalid"
