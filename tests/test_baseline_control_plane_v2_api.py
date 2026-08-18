from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text, update
from test_baseline_control_plane import (
    ControlEnvironment,
    _add_group_member,
)
from test_baseline_control_plane import (
    environment as _environment_fixture,  # noqa: F401
)
from test_baseline_control_plane_v2_protocol import _validate_contract
from test_baseline_index_continuation import (
    FixtureAdapter,
    _identity,
    _publish_corpus,
    _service,
)

from compair_core import api as api_module
from compair_core.baseline_control_plane_schema import (
    compatible_index_job,
    control_job,
    repository_approval,
    snapshot_continuation_job,
)
from compair_core.compair.retrieval.control_plane import (
    PROTOCOL_SHA256,
    PROTOCOL_VERSION,
    canonicalize,
)
from compair_core.compair.retrieval.control_plane_v2 import (
    PROTOCOL_V2_SHA256,
    PROTOCOL_V2_VERSION,
    assess_index_build_capability,
)
from compair_core.compair.retrieval.database_worker import (
    DatabaseWorkerAttestation,
    DatabaseWorkerRegistry,
)
from compair_core.compair.retrieval.index_continuation import (
    InternalIndexWorkerIdentity,
)
from compair_core.compair.retrieval.indexing import (
    baseline_engine_config_fingerprint,
)
from compair_core.runtime_config import build_runtime_configuration

SCHEMA = json.loads(
    (
        Path(__file__).parents[1] / "protocol/baseline-control-plane.v2.schema.json"
    ).read_text(encoding="utf-8")
)


@pytest.fixture
def environment(request: pytest.FixtureRequest):
    return request.getfixturevalue("_environment_fixture")


def _v2_capabilities(environment: ControlEnvironment) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_V2_VERSION,
        "protocol_sha256": PROTOCOL_V2_SHA256,
        "message_type": "capabilities_request",
        "request_id": str(uuid4()),
        "group_id": environment.group_id,
    }


def _v2_payload(environment: ControlEnvironment) -> dict[str, object]:
    identity = _identity()
    with environment.engine.connect() as connection:
        continuation = (
            connection.execute(
                select(snapshot_continuation_job)
                .where(snapshot_continuation_job.c.state == "succeeded")
                .order_by(snapshot_continuation_job.c.finished_at.desc())
            )
            .mappings()
            .first()
        )
    assert continuation is not None
    return {
        "protocol_version": PROTOCOL_V2_VERSION,
        "protocol_sha256": PROTOCOL_V2_SHA256,
        "message_type": "index_build_submit",
        "request_id": str(uuid4()),
        "group_id": environment.group_id,
        "idempotency_key": "opaque-v2-index-build-intent-00000001",
        "ingestion_continuation_id": continuation["continuation_job_id"],
        "corpus_generation_id": continuation["result_generation_id"],
        "corpus_manifest_hash": continuation["result_manifest_hash"],
        "ingestion_provenance_fingerprint": continuation[
            "result_provenance_fingerprint"
        ],
        "index_intent": {
            "index_format_version": "baseline-index.v1",
            "tokenizer_version": "baseline_v1_frozen_tokenizer.v1",
            "retrieval_config_fingerprint": baseline_engine_config_fingerprint(
                identity
            ),
            "embedding": {
                "contract_version": "baseline-embedding-http.v1",
                "provider": identity.provider,
                "model": identity.model,
                "revision": identity.revision,
                "dimension": identity.dimension,
                "dtype": "float32",
                "fingerprint": identity.fingerprint,
            },
        },
    }


def _v2_status(environment: ControlEnvironment, job_id: str) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_V2_VERSION,
        "protocol_sha256": PROTOCOL_V2_SHA256,
        "message_type": "job_status_request",
        "request_id": str(uuid4()),
        "group_id": environment.group_id,
        "job_id": job_id,
        "operation": "index_build",
    }


def _client(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    *,
    adapter: FixtureAdapter | None = None,
    user_id: str | None = None,
) -> tuple[TestClient, object]:
    service = _service(environment, adapter or FixtureAdapter())
    monkeypatch.setattr(
        api_module, "_control_plane_service", lambda: environment.service
    )
    monkeypatch.setattr(api_module, "core_database_engine", environment.engine)
    monkeypatch.setattr(api_module, "_compatible_index_job_service", lambda: service)
    app = api_module.create_fastapi_app()
    app.dependency_overrides[api_module.get_current_user] = lambda: SimpleNamespace(
        user_id=user_id or environment.user_id,
        username="v2-control@example.test",
        name="V2 Control User",
    )
    return TestClient(app, base_url="https://core.example.test"), service


def test_v2_database_dispatch_capability_and_exact_replay_under_backpressure(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish_corpus(environment)
    settings = SimpleNamespace(
        baseline_runs_enabled=False,
        baseline_worker_mode="database",
        baseline_worker_heartbeat_ttl_seconds=30,
        baseline_worker_poll_interval_seconds=1.0,
        baseline_worker_max_backoff_seconds=30.0,
        baseline_worker_max_pending_per_slot=1,
        baseline_control_plane_allow_insecure_loopback=False,
        baseline_control_plane_trusted_proxy_allowlist="",
    )
    monkeypatch.setattr(api_module, "get_settings_dependency", lambda: settings)
    DatabaseWorkerRegistry(
        environment.engine,
        heartbeat_ttl=timedelta(seconds=30),
        attestation=DatabaseWorkerAttestation.from_runtime(
            build_runtime_configuration(
                settings,
                database_url=environment.engine.url,
            )
        ),
    ).register(str(uuid4()))
    client, _service_instance = _client(environment, monkeypatch)
    payload = _v2_payload(environment)
    with client:
        ready = client.post(
            "/baseline/control/v2/capabilities",
            json=_v2_capabilities(environment),
        )
        accepted = client.post("/baseline/control/v2/index-builds", json=payload)
        full = client.post(
            "/baseline/control/v2/capabilities",
            json=_v2_capabilities(environment),
        )
        replayed = client.post("/baseline/control/v2/index-builds", json=payload)

    assert ready.json()["operations"]["index_build"] == {
        "submission": "safe",
        "endpoint": "authenticated_post",
        "dispatch": "automatic",
        "readiness": "ready",
        "reason_code": None,
    }
    assert accepted.status_code == replayed.status_code == 202
    assert replayed.json()["replayed"] is True
    assert replayed.json()["job_id"] == accepted.json()["job_id"]
    assert full.json()["operations"]["index_build"] == {
        "submission": "safe",
        "endpoint": "authenticated_post",
        "dispatch": "automatic",
        "readiness": "not_ready",
        "reason_code": "worker_unavailable",
    }


def test_v2_ready_capability_submit_replay_status_and_existing_service_mapping(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish_corpus(environment)
    client, service = _client(environment, monkeypatch)
    payload = _v2_payload(environment)
    with client:
        capability = client.post(
            "/baseline/control/v2/capabilities", json=_v2_capabilities(environment)
        )
        assert capability.status_code == 200
        body = capability.json()
        _validate_contract(body, SCHEMA)
        assert body["operations"] == {
            "index_build": {
                "submission": "safe",
                "endpoint": "authenticated_post",
                "dispatch": "manual",
                "readiness": "ready",
                "reason_code": None,
            },
            "baseline_run": {
                "submission": "unavailable",
                "endpoint": "unavailable",
                "dispatch": "unavailable",
                "readiness": "unavailable",
                "reason_code": "capability_unavailable",
            },
        }

        accepted = client.post("/baseline/control/v2/index-builds", json=payload)
        replay = client.post("/baseline/control/v2/index-builds", json=payload)
        assert accepted.status_code == replay.status_code == 202
        _validate_contract(accepted.json(), SCHEMA)
        assert accepted.json()["replayed"] is False
        assert replay.json()["replayed"] is True
        assert replay.json()["job_id"] == accepted.json()["job_id"]
        job_id = accepted.json()["job_id"]

        queued = client.post(
            "/baseline/control/v2/index-builds/status",
            json=_v2_status(environment, job_id),
        )
        assert queued.status_code == 200
        _validate_contract(queued.json(), SCHEMA)
        assert queued.json()["state"] == "queued"
        assert queued.json()["result"] is None
        assert queued.json()["progress"] == {
            "document_count": 0,
            "vector_count": 0,
        }
        queued_serialized = json.dumps(queued.json(), sort_keys=True)
        for protected in (
            "benign index corpus",
            payload["idempotency_key"],
            "lease_token",
            "repository_authority",
            "retrieval_query",
        ):
            assert str(protected) not in queued_serialized

        outcome = service.execute(
            identity=InternalIndexWorkerIdentity.create("v2-index-worker"),
            group_id=environment.group_id,
            job_id=job_id,
        )
        succeeded = client.post(
            "/baseline/control/v2/index-builds/status",
            json=_v2_status(environment, job_id),
        )
        assert succeeded.status_code == 200
        _validate_contract(succeeded.json(), SCHEMA)
        assert succeeded.json()["state"] == "succeeded"
        assert succeeded.json()["result"]["index_publication_id"] == outcome.index_id
        assert succeeded.json()["result"]["document_count"] == 1
        assert succeeded.json()["result"]["vector_count"] == 1

    with environment.engine.connect() as connection:
        job = (
            connection.execute(
                select(control_job).where(control_job.c.job_id == job_id)
            )
            .mappings()
            .one()
        )
        assert job["protocol_version"] == PROTOCOL_V2_VERSION
        assert job["protocol_sha256"] == PROTOCOL_V2_SHA256
        assert (
            connection.execute(
                select(func.count()).select_from(compatible_index_job)
            ).scalar_one()
            == 1
        )


def test_v2_not_ready_embedding_rejects_before_any_write(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish_corpus(environment)
    client, _service_instance = _client(
        environment, monkeypatch, adapter=FixtureAdapter(mode="unavailable")
    )
    with client:
        capability = client.post(
            "/baseline/control/v2/capabilities", json=_v2_capabilities(environment)
        )
        operation = capability.json()["operations"]["index_build"]
        assert operation == {
            "submission": "safe",
            "endpoint": "authenticated_post",
            "dispatch": "manual",
            "readiness": "not_ready",
            "reason_code": "embedding_unavailable",
        }
        rejected = client.post(
            "/baseline/control/v2/index-builds", json=_v2_payload(environment)
        )
    assert rejected.status_code == 503
    assert rejected.json()["code"] == "embedding_unavailable"
    assert rejected.json()["stage"] == "capability"
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count())
                .select_from(control_job)
                .where(control_job.c.operation == "index_build")
            ).scalar_one()
            == 0
        )


def test_v2_status_reauthorizes_source_scope_and_returns_generic_not_found(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from compair_core.compair.models import document_to_group_table

    _publish_corpus(environment)
    client, _service_instance = _client(environment, monkeypatch)
    with client:
        accepted = client.post(
            "/baseline/control/v2/index-builds", json=_v2_payload(environment)
        )
        assert accepted.status_code == 202
        with environment.engine.begin() as connection:
            connection.execute(
                document_to_group_table.delete().where(
                    document_to_group_table.c.document_id
                    == environment.source_document_id,
                    document_to_group_table.c.group_id == environment.group_id,
                )
            )
        hidden = client.post(
            "/baseline/control/v2/index-builds/status",
            json=_v2_status(environment, accepted.json()["job_id"]),
        )
    assert hidden.status_code == 404
    assert hidden.json()["code"] == "job_not_found_or_forbidden"
    assert hidden.json()["stage"] == "status"
    assert "source_document" not in hidden.text


def test_v2_missing_schema_and_service_report_not_ready_without_repair(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from compair_core.compair.retrieval.index_continuation import (
        BaselineCompatibleIndexJobService,
    )

    with environment.engine.begin() as connection:
        connection.execute(text("DROP TABLE baseline_compatible_index_job"))
    incomplete_service = BaselineCompatibleIndexJobService(
        environment.engine, lambda: FixtureAdapter()
    )
    capability = assess_index_build_capability(incomplete_service)
    assert capability.readiness == "not_ready"
    assert capability.reason_code == "capability_unavailable"
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                text(
                    "SELECT COUNT(*) FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'baseline_compatible_index_job'"
                )
            ).scalar_one()
            == 0
        )

    client, _service_instance = _client(environment, monkeypatch)
    monkeypatch.setattr(
        api_module,
        "_compatible_index_job_service",
        lambda: (_ for _ in ()).throw(RuntimeError("service unavailable")),
    )
    with client:
        response = client.post(
            "/baseline/control/v2/capabilities", json=_v2_capabilities(environment)
        )
    assert response.status_code == 200
    assert response.json()["operations"]["index_build"] == {
        "submission": "safe",
        "endpoint": "authenticated_post",
        "dispatch": "manual",
        "readiness": "not_ready",
        "reason_code": "capability_unavailable",
    }


def test_v2_exact_protocol_negotiation_and_v1_capability_remain_independent(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, _service_instance = _client(environment, monkeypatch)
    v1 = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "message_type": "capabilities_request",
        "request_id": str(uuid4()),
        "group_id": environment.group_id,
    }
    obsolete_v2 = _v2_capabilities(environment)
    obsolete_v2["protocol_sha256"] = (
        "c9486b3deb1a494781513109df17d8e8df1281fbc9687960ace711485b50d174"
    )
    with client:
        old = client.post("/baseline/control/v1/capabilities", json=v1)
        wrong = client.post("/baseline/control/v2/capabilities", json=v1)
        obsolete = client.post("/baseline/control/v2/capabilities", json=obsolete_v2)
        v2_on_v1 = client.post(
            "/baseline/control/v1/capabilities", json=_v2_capabilities(environment)
        )
    assert old.status_code == 200
    assert old.json()["protocol_version"] == PROTOCOL_VERSION
    assert old.json()["operations"]["index_build"] == "unavailable"
    assert wrong.status_code == 409
    assert wrong.json()["code"] == "protocol_mismatch"
    assert obsolete.status_code == 409
    assert obsolete.json()["code"] == "protocol_mismatch"
    assert v2_on_v1.status_code == 409
    assert v2_on_v1.json()["protocol_version"] == PROTOCOL_VERSION


def test_v2_submitted_embedding_identity_must_match_live_configured_identity(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish_corpus(environment)
    configured = _identity(revision="different-pinned-fixture-revision")
    client, _service_instance = _client(
        environment,
        monkeypatch,
        adapter=FixtureAdapter(identity=configured),
    )
    with client:
        capability = client.post(
            "/baseline/control/v2/capabilities", json=_v2_capabilities(environment)
        )
        assert (
            capability.json()["required_index_identity"]["embedding"]["revision"]
            == configured.revision
        )
        rejected = client.post(
            "/baseline/control/v2/index-builds", json=_v2_payload(environment)
        )
    assert rejected.status_code == 409
    assert rejected.json()["code"] == "embedding_identity_mismatch"
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count())
                .select_from(control_job)
                .where(control_job.c.operation == "index_build")
            ).scalar_one()
            == 0
        )


@pytest.mark.parametrize(
    ("state", "error_code", "terminal", "exit_classification", "reason"),
    [
        ("queued", None, False, "pending", None),
        ("running", None, False, "pending", None),
        (
            "retryable_failed",
            "embedding_service_timeout",
            False,
            "pending",
            "embedding_unavailable",
        ),
        (
            "terminal_failed",
            "embedding_vector_nonfinite",
            True,
            "failed",
            "index_vector_invalid",
        ),
        ("cancelled", "worker_cancelled", True, "cancelled", "job_cancelled"),
    ],
)
def test_v2_safe_status_projection_for_non_success_states(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    error_code: str | None,
    terminal: bool,
    exit_classification: str,
    reason: str | None,
) -> None:
    _publish_corpus(environment)
    client, _service_instance = _client(environment, monkeypatch)
    with client:
        accepted = client.post(
            "/baseline/control/v2/index-builds", json=_v2_payload(environment)
        ).json()
        values: dict[str, object] = {
            "state": state,
            "error_code": error_code,
            "error_fingerprint": "a" * 64 if error_code else None,
        }
        if state == "running":
            values.update(
                lease_token="opaque-test-lease",
                lease_expires_at=environment.service.clock() + timedelta(minutes=5),
                attempt_count=1,
            )
        with environment.engine.begin() as connection:
            connection.execute(
                update(control_job)
                .where(control_job.c.job_id == accepted["job_id"])
                .values(**values)
            )
        response = client.post(
            "/baseline/control/v2/index-builds/status",
            json=_v2_status(environment, str(accepted["job_id"])),
        )
    assert response.status_code == 200
    _validate_contract(response.json(), SCHEMA)
    assert response.json()["terminal"] is terminal
    assert response.json()["exit_classification"] == exit_classification
    assert response.json()["reason_code"] == reason
    assert response.json()["result"] is None


def test_v2_member_cannot_submit_another_callers_continuation_and_cross_scope_is_hidden(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish_corpus(environment)
    member_id = _add_group_member(environment.engine, group_id=environment.group_id)
    client, _service_instance = _client(environment, monkeypatch, user_id=member_id)
    with client:
        capability = client.post(
            "/baseline/control/v2/capabilities", json=_v2_capabilities(environment)
        )
        denied = client.post(
            "/baseline/control/v2/index-builds", json=_v2_payload(environment)
        )
    assert capability.status_code == 200
    assert denied.status_code == 404
    assert denied.json()["code"] == "job_not_found_or_forbidden"
    cross_group = _v2_capabilities(environment)
    cross_group["group_id"] = "inaccessible-group"
    with client:
        hidden = client.post("/baseline/control/v2/capabilities", json=cross_group)
    assert hidden.status_code == 404
    assert hidden.json()["code"] == "job_not_found_or_forbidden"
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count())
                .select_from(control_job)
                .where(control_job.c.operation == "index_build")
            ).scalar_one()
            == 0
        )


def test_v2_revocation_staleness_conflict_and_status_privacy(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish_corpus(environment, ordinal=1)
    client, _service_instance = _client(environment, monkeypatch)
    first_payload = _v2_payload(environment)
    with client:
        first = client.post("/baseline/control/v2/index-builds", json=first_payload)
        assert first.status_code == 202
        _publish_corpus(environment, ordinal=2)
        stale_payload = dict(first_payload)
        stale_payload["request_id"] = str(uuid4())
        stale_payload["idempotency_key"] = "opaque-v2-stale-index-intent-00000002"
        stale_submit = client.post(
            "/baseline/control/v2/index-builds", json=stale_payload
        )
        conflicting = _v2_payload(environment)
        conflicting["idempotency_key"] = first_payload["idempotency_key"]
        conflict = client.post("/baseline/control/v2/index-builds", json=conflicting)
        stale = client.post(
            "/baseline/control/v2/index-builds/status",
            json=_v2_status(environment, first.json()["job_id"]),
        )
        with environment.engine.begin() as connection:
            connection.execute(
                update(repository_approval)
                .where(
                    repository_approval.c.registration_id
                    == environment.sibling_repository_id
                )
                .values(
                    state="disabled",
                    disabled_at=environment.service.clock(),
                )
            )
        revoked = client.post(
            "/baseline/control/v2/index-builds", json=_v2_payload(environment)
        )
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "idempotency_conflict"
    assert stale_submit.status_code == 409
    assert stale_submit.json()["code"] == "index_publication_stale"
    assert stale.status_code == 409
    assert stale.json()["code"] == "index_publication_stale"
    assert revoked.status_code == 404
    assert revoked.json()["code"] == "repository_not_authorized"


@pytest.mark.parametrize(
    ("body", "headers", "expected"),
    [
        (
            (
                b'{"protocol_version":"baseline-control-plane.v2",'
                b'"protocol_sha256":"b278abe007779f05e92509db068f555701c03cba5cf236151e8df231a9b44091",'
                b'"message_type":"capabilities_request","request_id":"00000000-0000-4000-8000-000000000001",'
                b'"group_id":"a","group_id":"b"}'
            ),
            {"Content-Type": "application/json"},
            400,
        ),
        (b'{"value":NaN}', {"Content-Type": "application/json"}, 400),
        (b"\xff", {"Content-Type": "application/json"}, 400),
        (b"{}", {"Content-Type": "text/plain"}, 400),
        (b" " * 64_001, {"Content-Type": "application/json"}, 413),
    ],
)
def test_v2_strict_request_boundary_rejects_malformed_inputs(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
    headers: dict[str, str],
    expected: int,
) -> None:
    client, _service_instance = _client(environment, monkeypatch)
    with client:
        response = client.post(
            "/baseline/control/v2/capabilities", content=body, headers=headers
        )
    assert response.status_code == expected
    assert response.json()["message_type"] == "error"
    assert response.json()["code"] in {"protocol_mismatch", "limit_exceeded"}
    assert "PRIVATE-REQUEST-BODY" not in response.text


def test_v2_proxy_spoof_and_disabled_run_endpoints_create_no_jobs(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_module, "_control_plane_service", lambda: environment.service
    )
    monkeypatch.setattr(
        api_module, "_compatible_index_job_service", lambda: _service(environment)
    )
    monkeypatch.setattr(
        api_module,
        "get_settings_dependency",
        lambda: SimpleNamespace(
            baseline_control_plane_allow_insecure_loopback=True,
            baseline_control_plane_trusted_proxy_allowlist="",
        ),
    )
    app = api_module.create_fastapi_app()
    app.dependency_overrides[api_module.get_current_user] = lambda: SimpleNamespace(
        user_id=environment.user_id
    )
    with TestClient(app, base_url="http://core.example.test") as client:
        spoofed = client.post(
            "/baseline/control/v2/capabilities",
            json=_v2_capabilities(environment),
            headers={
                "Forwarded": "for=127.0.0.1;proto=https",
                "X-Forwarded-Proto": "https",
                "Host": "localhost",
            },
        )
        run = client.post("/baseline/control/v2/runs", json={})
        run_status = client.post("/baseline/control/v2/runs/status", json={})
    assert spoofed.status_code == 503
    assert spoofed.json()["code"] == "transport_unavailable"
    assert run.status_code == run_status.status_code == 503
    assert run.json()["code"] == run_status.json()["code"] == "transport_unavailable"
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count())
                .select_from(control_job)
                .where(control_job.c.operation == "baseline_run")
            ).scalar_one()
            == 0
        )


def test_v2_duplicate_nested_index_fields_are_rejected_before_authorization_or_write(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish_corpus(environment)
    client, _service_instance = _client(environment, monkeypatch)
    raw = canonicalize(_v2_payload(environment))
    assert raw.count(b'"dtype":"float32"') == 1
    duplicate = raw.replace(
        b'"dtype":"float32"',
        b'"dtype":"float32","dtype":"float32"',
    )
    with client:
        response = client.post(
            "/baseline/control/v2/index-builds",
            content=duplicate,
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 400
    assert response.json()["code"] == "protocol_mismatch"
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count())
                .select_from(control_job)
                .where(control_job.c.operation == "index_build")
            ).scalar_one()
            == 0
        )


def test_v2_frozen_protocol_files_remain_unmodified() -> None:
    import hashlib

    root = Path(__file__).parents[1] / "protocol"
    assert (
        hashlib.sha256((root / "baseline-control-plane.v2.md").read_bytes()).hexdigest()
        == PROTOCOL_V2_SHA256
    )
    assert (
        hashlib.sha256((root / "baseline-control-plane.v1.md").read_bytes()).hexdigest()
        == PROTOCOL_SHA256
    )
