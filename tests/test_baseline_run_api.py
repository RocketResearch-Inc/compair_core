from __future__ import annotations

import base64
import copy
import inspect
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text, update
from sqlalchemy.orm import sessionmaker
from test_baseline_control_generation import _structured
from test_baseline_control_plane import (
    ControlEnvironment,
    _add_group_member,
)
from test_baseline_control_plane import (
    environment as _environment_fixture,  # noqa: F401
)
from test_baseline_control_plane_v2_protocol import _validate_contract
from test_baseline_generation import CapturingProvider, RawOutputProvider
from test_baseline_index_continuation import FixtureAdapter
from test_baseline_run_executor import RecordingRetriever
from test_baseline_run_jobs import RAW_QUERY, _keyring, _run_payload

from compair_core import api as api_module
from compair_core.baseline_control_plane_schema import (
    baseline_run_job,
    baseline_run_payload,
)
from compair_core.compair.retrieval.control_plane_v2 import (
    PROTOCOL_V2_SHA256,
    PROTOCOL_V2_VERSION,
)
from compair_core.compair.retrieval.database_worker import (
    DatabaseWorkerAttestation,
    DatabaseWorkerRegistry,
)
from compair_core.compair.retrieval.generation import BaselineGenerationService
from compair_core.compair.retrieval.persistent import PersistentBaselineV1Retriever
from compair_core.compair.retrieval.preview import (
    BaselinePreviewCommand,
    BaselinePreviewService,
)
from compair_core.compair.retrieval.run_executor import (
    BaselineDocumentRunExecutor,
    InternalBaselineRunWorkerIdentity,
)
from compair_core.compair.retrieval.run_jobs import BaselineRunJobError
from compair_core.compair.retrieval.run_operator import (
    BaselineManualRunOperator,
    BaselineRunRuntime,
    BaselineRunRuntimeError,
    process_baseline_run_job,
)
from compair_core.runtime_config import build_runtime_configuration
from compair_core.server.settings import Settings

SCHEMA = json.loads(
    (
        Path(__file__).parents[1] / "protocol/baseline-control-plane.v2.schema.json"
    ).read_text(encoding="utf-8")
)


@pytest.fixture
def environment(request: pytest.FixtureRequest):
    return request.getfixturevalue("_environment_fixture")


def _keyring_json() -> str:
    return json.dumps(
        {
            "version": "baseline-run-keyring.v1",
            "active_key_id": "key-2026-08",
            "keys": [
                {
                    "key_id": "key-2026-08",
                    "key_base64": base64.b64encode(b"n" * 32).decode("ascii"),
                }
            ],
        }
    )


def _settings(
    *,
    enabled: bool,
    keyring: str | None = None,
    worker_mode: str = "manual",
    maximum_pending: int = 8,
):
    return SimpleNamespace(
        baseline_runs_enabled=enabled,
        baseline_worker_mode=worker_mode,
        baseline_worker_poll_interval_seconds=1.0,
        baseline_worker_heartbeat_interval_seconds=5.0,
        baseline_worker_heartbeat_ttl_seconds=30,
        baseline_worker_cleanup_interval_seconds=30,
        baseline_worker_max_pending_per_slot=maximum_pending,
        baseline_worker_max_attempts=5,
        baseline_worker_max_backoff_seconds=30.0,
        baseline_run_encryption_keyring=keyring,
        baseline_run_payload_ttl_seconds=900,
        baseline_control_plane_allow_insecure_loopback=False,
        baseline_control_plane_trusted_proxy_allowlist="",
        baseline_embedding_provider="http",
        baseline_embedding_endpoint="http://127.0.0.1:19091",
        baseline_embedding_model="BAAI/bge-small-en-v1.5",
        baseline_embedding_revision="fixture-bge-revision-r1",
        baseline_embedding_dimension=384,
        baseline_embedding_timeout_seconds=1.0,
        baseline_embedding_batch_size=8,
        baseline_embedding_allow_insecure_loopback=True,
    )


def _status(group_id: str, job_id: str) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_V2_VERSION,
        "protocol_sha256": PROTOCOL_V2_SHA256,
        "message_type": "job_status_request",
        "request_id": str(uuid4()),
        "group_id": group_id,
        "job_id": job_id,
        "operation": "baseline_run",
    }


def _capabilities(group_id: str) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_V2_VERSION,
        "protocol_sha256": PROTOCOL_V2_SHA256,
        "message_type": "capabilities_request",
        "request_id": str(uuid4()),
        "group_id": group_id,
    }


def _client(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    settings,
    *,
    runtime=None,
    user_id: str | None = None,
) -> TestClient:
    monkeypatch.setattr(api_module, "core_database_engine", environment.engine)
    monkeypatch.setattr(
        api_module, "_control_plane_service", lambda: environment.service
    )
    monkeypatch.setattr(api_module, "get_settings_dependency", lambda: settings)
    if runtime is not None:
        monkeypatch.setattr(api_module, "_baseline_run_runtime", lambda: runtime)
    app = api_module.create_fastapi_app()
    app.dependency_overrides[api_module.get_current_user] = lambda: SimpleNamespace(
        user_id=user_id or environment.user_id,
        username="baseline-run@example.test",
        name="Baseline Run Caller",
    )
    return TestClient(app, base_url="https://core.example.test")


def _row_count(environment: ControlEnvironment, table) -> int:
    with environment.engine.connect() as connection:
        return int(
            connection.execute(select(func.count()).select_from(table)).scalar_one()
        )


def _ready_runtime(environment: ControlEnvironment, settings) -> BaselineRunRuntime:
    runtime = BaselineRunRuntime(
        environment.engine,
        settings,
        provider_factory=lambda: CapturingProvider("readiness fixture"),
    )
    runtime.embedding_adapter = FixtureAdapter()
    return runtime


def test_baseline_runs_feature_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("COMPAIR_BASELINE_RUNS_ENABLED", raising=False)
    assert Settings().baseline_runs_enabled is False


def test_operator_public_callable_accepts_only_opaque_job_id() -> None:
    assert tuple(inspect.signature(process_baseline_run_job).parameters) == ("job_id",)
    source = (
        Path(__file__).parents[1] / "compair_core/compair/retrieval/run_operator.py"
    ).read_text(encoding="utf-8")
    assert "Celery" not in source
    assert "Thread(" not in source
    assert ".delay(" not in source


def _manual_operator(environment: ControlEnvironment, provider):
    sessions = sessionmaker(environment.engine, expire_on_commit=False)
    recording = RecordingRetriever(
        PersistentBaselineV1Retriever(sessions, FixtureAdapter())
    )
    executor = BaselineDocumentRunExecutor(
        environment.engine,
        identity=InternalBaselineRunWorkerIdentity.create("manual-api-test"),
        keyring=_keyring(),
        retriever_factory=lambda: recording,
    )
    operator = BaselineManualRunOperator(
        environment.engine,
        executor=executor,
        generation=BaselineGenerationService(
            sessions,
            notifications_enabled=False,
        ),
        provider=provider,
    )
    return operator, recording


def test_default_off_capability_and_endpoints_reject_before_write(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _run_payload(environment)
    settings = _settings(enabled=False)
    client = _client(environment, monkeypatch, settings)
    with client:
        capability = client.post(
            "/baseline/control/v2/capabilities",
            json=_capabilities(environment.group_id),
        )
        rejected = client.post("/baseline/control/v2/runs", json=payload)
        status = client.post(
            "/baseline/control/v2/runs/status",
            json=_status(environment.group_id, str(uuid4())),
        )
    operation = capability.json()["operations"]["baseline_run"]
    assert operation == {
        "submission": "unavailable",
        "endpoint": "unavailable",
        "dispatch": "unavailable",
        "readiness": "unavailable",
        "reason_code": "capability_unavailable",
    }
    assert rejected.status_code == status.status_code == 503
    assert rejected.json()["code"] == status.json()["code"] == "capability_unavailable"
    assert _row_count(environment, baseline_run_job) == 0
    assert _row_count(environment, baseline_run_payload) == 0


def test_runtime_readiness_requires_keyring_embedding_and_current_publication(
    environment: ControlEnvironment,
) -> None:
    payload = _run_payload(environment)
    settings = _settings(enabled=True, keyring=_keyring_json())
    runtime = _ready_runtime(environment, settings)
    capability = runtime.capability(
        group_id=environment.group_id,
        caller_user_id=environment.user_id,
    )
    assert capability.as_dict() == {
        "submission": "safe",
        "endpoint": "authenticated_post",
        "dispatch": "manual",
        "readiness": "ready",
        "reason_code": None,
    }

    runtime.embedding_adapter = FixtureAdapter(mode="unavailable")
    unavailable = runtime.capability(
        group_id=environment.group_id,
        caller_user_id=environment.user_id,
    )
    assert unavailable.readiness == "not_ready"
    assert unavailable.reason_code == "embedding_unavailable"

    with pytest.raises(BaselineRunRuntimeError):
        BaselineRunRuntime(
            environment.engine,
            _settings(enabled=True, keyring="malformed-keyring"),
            provider_factory=lambda: CapturingProvider("unused"),
        )
    assert payload["index_publication"]["index_publication_id"]


def test_database_mode_requires_recent_worker_and_admits_with_automatic_dispatch(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _run_payload(environment)
    settings = _settings(
        enabled=True,
        keyring=_keyring_json(),
        worker_mode="database",
        maximum_pending=1,
    )
    runtime = _ready_runtime(environment, settings)
    client = _client(environment, monkeypatch, settings, runtime=runtime)
    with client:
        unavailable = client.post(
            "/baseline/control/v2/capabilities",
            json=_capabilities(environment.group_id),
        )
        rejected = client.post("/baseline/control/v2/runs", json=payload)
    assert unavailable.json()["operations"]["baseline_run"] == {
        "submission": "safe",
        "endpoint": "authenticated_post",
        "dispatch": "automatic",
        "readiness": "not_ready",
        "reason_code": "worker_unavailable",
    }
    assert rejected.status_code == 503
    assert _row_count(environment, baseline_run_job) == 0

    registry = DatabaseWorkerRegistry(
        environment.engine,
        heartbeat_ttl=timedelta(seconds=30),
        attestation=DatabaseWorkerAttestation.from_runtime(
            build_runtime_configuration(
                settings,
                database_url=environment.engine.url,
            )
        ),
    )
    registry.register(str(uuid4()))
    with client:
        ready = client.post(
            "/baseline/control/v2/capabilities",
            json=_capabilities(environment.group_id),
        )
        accepted = client.post("/baseline/control/v2/runs", json=payload)
        full = client.post(
            "/baseline/control/v2/capabilities",
            json=_capabilities(environment.group_id),
        )
        replayed = client.post("/baseline/control/v2/runs", json=payload)
    assert ready.json()["operations"]["baseline_run"] == {
        "submission": "safe",
        "endpoint": "authenticated_post",
        "dispatch": "automatic",
        "readiness": "ready",
        "reason_code": None,
    }
    assert accepted.status_code == 202
    assert accepted.json()["state"] == "queued"
    assert full.json()["operations"]["baseline_run"]["readiness"] == "not_ready"
    assert full.json()["operations"]["baseline_run"]["reason_code"] == (
        "worker_unavailable"
    )
    assert replayed.status_code == 202
    assert replayed.json()["replayed"] is True
    assert replayed.json()["job_id"] == accepted.json()["job_id"]
    assert _row_count(environment, baseline_run_job) == 1


@pytest.mark.parametrize(
    ("mode", "expected_reason"),
    [
        ("keyring", "capability_unavailable"),
        ("embedding", "embedding_unavailable"),
        ("provider", "worker_unavailable"),
    ],
)
def test_not_ready_prerequisites_are_truthful_and_reject_before_write(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_reason: str,
) -> None:
    payload = _run_payload(environment)
    settings = _settings(
        enabled=True,
        keyring=None if mode == "keyring" else _keyring_json(),
    )
    if mode == "embedding":
        settings.baseline_embedding_provider = "disabled"
    if mode == "provider":
        monkeypatch.setattr(
            api_module,
            "_baseline_run_runtime",
            lambda: (_ for _ in ()).throw(
                BaselineRunRuntimeError("worker_unavailable")
            ),
        )
    client = _client(environment, monkeypatch, settings)
    with client:
        capability = client.post(
            "/baseline/control/v2/capabilities",
            json=_capabilities(environment.group_id),
        )
        rejected = client.post("/baseline/control/v2/runs", json=payload)
    assert capability.json()["operations"]["baseline_run"] == {
        "submission": "safe",
        "endpoint": "authenticated_post",
        "dispatch": "manual",
        "readiness": "not_ready",
        "reason_code": expected_reason,
    }
    assert rejected.status_code == 503
    assert rejected.json()["code"] == expected_reason
    assert _row_count(environment, baseline_run_job) == 0
    assert _row_count(environment, baseline_run_payload) == 0


def test_enabled_submission_is_encrypted_queued_replay_safe_and_read_only_status(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = _run_payload(environment)
    settings = _settings(enabled=True, keyring=_keyring_json())
    runtime = _ready_runtime(environment, settings)
    client = _client(environment, monkeypatch, settings, runtime=runtime)
    with client:
        capability = client.post(
            "/baseline/control/v2/capabilities",
            json=_capabilities(environment.group_id),
        )
        accepted = client.post("/baseline/control/v2/runs", json=payload)
        with environment.engine.connect() as connection:
            protected_before = dict(
                connection.execute(select(baseline_run_payload)).mappings().one()
            )
            job_before = dict(
                connection.execute(select(baseline_run_job)).mappings().one()
            )
        replay = client.post("/baseline/control/v2/runs", json=payload)
        queued = client.post(
            "/baseline/control/v2/runs/status",
            json=_status(environment.group_id, accepted.json()["job_id"]),
        )
    assert capability.json()["operations"]["baseline_run"] == {
        "submission": "safe",
        "endpoint": "authenticated_post",
        "dispatch": "manual",
        "readiness": "ready",
        "reason_code": None,
    }
    assert accepted.status_code == replay.status_code == 202
    assert accepted.json()["state"] == "queued"
    assert replay.json()["replayed"] is True
    assert replay.json()["job_id"] == accepted.json()["job_id"]
    assert queued.status_code == 200
    _validate_contract(queued.json(), SCHEMA)
    assert queued.json()["state"] == "queued"
    assert queued.json()["effects"]["persisted_run_id"] is None
    with environment.engine.connect() as connection:
        protected_after = dict(
            connection.execute(select(baseline_run_payload)).mappings().one()
        )
        job_after = dict(connection.execute(select(baseline_run_job)).mappings().one())
    assert protected_after["nonce"] == protected_before["nonce"]
    assert protected_after["ciphertext"] == protected_before["ciphertext"]
    assert protected_after["expires_at"] == protected_before["expires_at"]
    assert job_after["state"] == job_before["state"] == "queued"
    assert job_after["attempt_count"] == 0
    rendered = json.dumps(
        [capability.json(), accepted.json(), replay.json(), queued.json()],
        sort_keys=True,
    )
    for forbidden in (
        RAW_QUERY,
        payload["idempotency_key"],
        protected_before["ciphertext"].hex(),
        protected_before["nonce"].hex(),
        protected_before["key_id"],
        "lease_token",
        "parent_processing_secret",
    ):
        assert forbidden not in rendered
        assert forbidden not in caplog.text


def test_run_strict_boundary_conflict_and_submitter_authorization(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _run_payload(environment)
    settings = _settings(enabled=True, keyring=_keyring_json())
    runtime = _ready_runtime(environment, settings)
    client = _client(environment, monkeypatch, settings, runtime=runtime)
    with client:
        accepted = client.post("/baseline/control/v2/runs", json=payload)
        conflict_payload = copy.deepcopy(payload)
        conflict_payload["request_id"] = str(uuid4())
        conflict_payload["retrieval_query"]["head_revision"] = "3" * 40
        conflict = client.post("/baseline/control/v2/runs", json=conflict_payload)
        query_string = client.post(
            "/baseline/control/v2/runs?job=forbidden",
            json=payload,
        )
        obsolete = copy.deepcopy(payload)
        obsolete["protocol_sha256"] = "8" * 64
        old_hash = client.post("/baseline/control/v2/runs", json=obsolete)
        duplicate = json.dumps(payload, separators=(",", ":")).replace(
            f'"group_id":"{environment.group_id}"',
            f'"group_id":"{environment.group_id}","group_id":"{environment.group_id}"',
            1,
        )
        duplicate_response = client.post(
            "/baseline/control/v2/runs",
            content=duplicate.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
    assert accepted.status_code == 202
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "idempotency_conflict"
    assert query_string.status_code == 400
    assert old_hash.status_code == 409
    assert duplicate_response.status_code == 400
    assert _row_count(environment, baseline_run_job) == 1

    member = _add_group_member(environment.engine, group_id=environment.group_id)
    with pytest.raises(BaselineRunJobError, match="job_not_found_or_forbidden"):
        runtime.jobs.read_status(
            request_id=str(uuid4()),
            group_id=environment.group_id,
            job_id=accepted.json()["job_id"],
            caller_user_id=member,
        )


@pytest.mark.parametrize(
    ("state", "reason", "failure_stage", "exit_classification"),
    [
        ("running", None, None, "pending"),
        ("retryable_failed", "retrieval_error", "retrieval", "pending"),
        ("terminal_failed", "retrieval_error", "retrieval", "failed"),
        ("blocked", "worker_unavailable", "dispatch", "blocked"),
        ("cancelled", "job_cancelled", "dispatch", "cancelled"),
    ],
)
def test_status_projects_safe_non_success_states(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    reason: str | None,
    failure_stage: str | None,
    exit_classification: str,
) -> None:
    payload = _run_payload(environment)
    settings = _settings(enabled=True, keyring=_keyring_json())
    runtime = _ready_runtime(environment, settings)
    client = _client(environment, monkeypatch, settings, runtime=runtime)
    with client:
        accepted = client.post("/baseline/control/v2/runs", json=payload).json()
        values: dict[str, object] = {
            "state": state,
            "reason_code": reason,
            "failure_stage": failure_stage,
        }
        if state == "running":
            values.update(
                lease_token="opaque-state-test-lease",
                lease_expires_at=environment.service.clock().replace(year=2030),
            )
        else:
            values.update(lease_token=None, lease_expires_at=None)
        with environment.engine.begin() as connection:
            connection.execute(
                update(baseline_run_job)
                .where(baseline_run_job.c.job_id == accepted["job_id"])
                .values(**values)
            )
        response = client.post(
            "/baseline/control/v2/runs/status",
            json=_status(environment.group_id, accepted["job_id"]),
        )
    assert response.status_code == 200
    assert response.json()["state"] == state
    assert response.json()["exit_classification"] == exit_classification
    assert response.json()["reason_code"] == reason
    assert "feedback" not in response.json()


@pytest.mark.parametrize("findings", [("ordered finding",), ()])
def test_manual_operator_end_to_end_positive_and_zero_finding_preview(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    findings: tuple[str, ...],
) -> None:
    payload = _run_payload(environment)
    payload["idempotency_key"] = (
        "opaque-manual-positive-intent-000001"
        if findings
        else "opaque-manual-zero-intent-0000000001"
    )
    settings = _settings(enabled=True, keyring=_keyring_json())
    provider = (
        CapturingProvider(*findings)
        if findings
        else RawOutputProvider(_structured("no_findings", []))
    )
    operator, recording = _manual_operator(environment, provider)
    runtime = _ready_runtime(environment, settings)
    runtime.operator = operator
    client = _client(environment, monkeypatch, settings, runtime=runtime)
    with client:
        accepted = client.post("/baseline/control/v2/runs", json=payload)
        queued = client.post(
            "/baseline/control/v2/runs/status",
            json=_status(environment.group_id, accepted.json()["job_id"]),
        )
        outcome = operator.process(accepted.json()["job_id"])
        completed = client.post(
            "/baseline/control/v2/runs/status",
            json=_status(environment.group_id, accepted.json()["job_id"]),
        )
        replay = client.post("/baseline/control/v2/runs", json=payload)
    assert queued.json()["state"] == "queued"
    assert outcome.state == "feedback_persisted"
    assert completed.json()["state"] == "feedback_persisted"
    assert completed.json()["effects"]["feedback_count"] == len(findings)
    assert completed.json()["effects"]["generation_invoked"] is True
    assert replay.json()["job_id"] == accepted.json()["job_id"]
    assert replay.json()["replayed"] is True
    assert len(recording.requests) == 1
    assert recording.requests[0].retrieval_query == RAW_QUERY
    assert recording.results[0].fallback_engine is None
    assert _row_count(environment, baseline_run_payload) == 0

    preview = BaselinePreviewService(
        sessionmaker(environment.engine, expire_on_commit=False)
    ).load(
        BaselinePreviewCommand(
            caller_user_id=environment.user_id,
            request_id=str(uuid4()),
            group_id=environment.group_id,
            job_id=accepted.json()["job_id"],
        )
    )
    assert [item.ordinal for item in preview.feedback] == list(
        range(1, len(findings) + 1)
    )
    assert [item.feedback for item in preview.feedback] == list(findings)
    assert preview.retrieval.evidence_count == preview.retrieval.reference_count
    assert 1 <= preview.retrieval.evidence_count <= 4
    with environment.engine.connect() as connection:
        selected = (
            connection.execute(
                text(
                    "SELECT ordinal FROM baseline_selected_evidence "
                    "WHERE run_id = :run_id ORDER BY ordinal"
                ),
                {"run_id": outcome.persisted_run_id},
            )
            .scalars()
            .all()
        )
        references = (
            connection.execute(
                text(
                    "SELECT s.ordinal FROM reference r JOIN baseline_selected_evidence s "
                    "ON s.selected_evidence_id = r.baseline_selected_evidence_id "
                    "WHERE s.run_id = :run_id ORDER BY s.ordinal"
                ),
                {"run_id": outcome.persisted_run_id},
            )
            .scalars()
            .all()
        )
        outbox = (
            connection.execute(
                text(
                    "SELECT state, finding_count FROM baseline_notification_outbox "
                    "WHERE run_id = :run_id"
                ),
                {"run_id": outcome.persisted_run_id},
            )
            .mappings()
            .all()
        )
        notification_count = connection.execute(
            text("SELECT count(*) FROM notification_event")
        ).scalar_one()
    assert selected == references == list(range(1, len(selected) + 1))
    if findings:
        assert [dict(row) for row in outbox] == [
            {"state": "suppressed", "finding_count": len(findings)}
        ]
        assert preview.digest is not None and preview.digest.state == "suppressed"
    else:
        assert outbox == []
        assert preview.digest is None
    assert notification_count == 0


def test_generation_lease_projects_last_durable_references_boundary(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingProvider(CapturingProvider):
        def generate(self, generation_input, *, idempotency_key: str) -> str:
            self.inputs.append(generation_input)
            self.idempotency_keys.append(idempotency_key)
            started.set()
            assert release.wait(timeout=10)
            return self.output

    payload = _run_payload(environment)
    payload["idempotency_key"] = "opaque-public-boundary-intent-00000001"
    provider = BlockingProvider("ordered finding")
    operator, _recording = _manual_operator(environment, provider)
    settings = _settings(enabled=True, keyring=_keyring_json())
    runtime = _ready_runtime(environment, settings)
    runtime.operator = operator
    client = _client(environment, monkeypatch, settings, runtime=runtime)

    with client, ThreadPoolExecutor(max_workers=1) as pool:
        accepted = client.post("/baseline/control/v2/runs", json=payload).json()
        future = pool.submit(operator.process, accepted["job_id"])
        assert started.wait(timeout=10)
        try:
            status = client.post(
                "/baseline/control/v2/runs/status",
                json=_status(environment.group_id, accepted["job_id"]),
            )
            with environment.engine.connect() as connection:
                internal_state = connection.execute(
                    select(baseline_run_job.c.state).where(
                        baseline_run_job.c.job_id == accepted["job_id"]
                    )
                ).scalar_one()
        finally:
            release.set()
        assert future.result(timeout=10).state == "feedback_persisted"

    assert internal_state == "running"
    assert status.status_code == 200
    body = status.json()
    _validate_contract(body, SCHEMA)
    assert body["state"] == "references_persisted"
    assert body["terminal"] is False
    assert body["retrieval_status"] == "ok"
    assert 1 <= body["effects"]["evidence_count"] <= 4
    assert body["effects"]["evidence_count"] == body["effects"]["reference_count"]
    assert body["effects"]["persisted_run_id"] is not None
    assert body["effects"]["feedback_count"] == 0
    assert body["effects"]["generation_invoked"] is False
