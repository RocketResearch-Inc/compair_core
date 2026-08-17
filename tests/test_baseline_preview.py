from __future__ import annotations

import json
from dataclasses import dataclass, replace
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text
from test_baseline_control_generation import _persist_control, _structured
from test_baseline_generation import CapturingProvider, RawOutputProvider, _environment

from compair_core import api
from compair_core.compair.retrieval.generation import BaselineGenerationService
from compair_core.compair.retrieval.notification_outbox import (
    BaselineNotificationOutboxDispatcher,
)
from compair_core.compair.retrieval.persistent import PersistentBaselineV1Retriever
from compair_core.compair.retrieval.preview import (
    BASELINE_PREVIEW_SCHEMA_VERSION,
    BaselinePreviewCommand,
    BaselinePreviewError,
    BaselinePreviewService,
    BaselinePreviewSource,
    parse_baseline_preview_request,
)


@dataclass(frozen=True)
class PreviewEnvironment:
    environment: object
    job_id: str
    caller_user_id: str
    run_id: str
    digest_id: str | None
    feedback_ids: tuple[str, ...]


def _request_id() -> str:
    return str(uuid4())


def _successful_control_preview(
    tmp_path: Path,
    name: str,
    findings: tuple[str, ...],
) -> PreviewEnvironment:
    environment = _environment(tmp_path, name)
    job_id, caller, persisted = _persist_control(environment)
    provider = (
        RawOutputProvider(_structured("no_findings", []))
        if not findings
        else CapturingProvider(*findings)
    )
    receipt = BaselineGenerationService(
        environment.sessions,
        notifications_enabled=False,
    ).generate_control(job_id, provider)
    assert receipt.state == "feedback_persisted"
    with environment.engine.connect() as connection:
        digest_id = connection.execute(
            text(
                "SELECT outbox_id FROM baseline_notification_outbox "
                "WHERE run_id = :run_id"
            ),
            {"run_id": persisted.run_id},
        ).scalar_one_or_none()
    return PreviewEnvironment(
        environment=environment,
        job_id=job_id,
        caller_user_id=caller,
        run_id=persisted.run_id,
        digest_id=str(digest_id) if digest_id is not None else None,
        feedback_ids=receipt.feedback_ids,
    )


def _load_by_job(value: PreviewEnvironment):
    return BaselinePreviewService(value.environment.sessions).load(
        BaselinePreviewCommand(
            caller_user_id=value.caller_user_id,
            request_id=_request_id(),
            group_id=value.environment.group_id,
            job_id=value.job_id,
        )
    )


def _assert_unavailable(command: BaselinePreviewCommand, sessions) -> None:
    with pytest.raises(BaselinePreviewError) as error:
        BaselinePreviewService(sessions).load(command)
    assert error.value.code == "baseline_preview_unavailable"
    assert str(error.value) == "baseline preview is unavailable"


def test_zero_finding_control_job_is_success_with_no_digest(tmp_path: Path) -> None:
    value = _successful_control_preview(tmp_path, "preview-zero.db", ())
    try:
        preview = _load_by_job(value)
        payload = preview.to_dict()
        assert payload["schema_version"] == BASELINE_PREVIEW_SCHEMA_VERSION
        assert payload["control_job"] | {"completed_at": "ignored"} == {
            "job_id": value.job_id,
            "state": "feedback_persisted",
            "completed_at": "ignored",
            "generation_invoked": True,
            "feedback_count": 0,
            "notification_outbox_count": 0,
        }
        assert payload["retrieval"] == {
            "persisted_run_id": value.run_id,
            "status": "ok",
            "evidence_count": 4,
            "reference_count": 4,
        }
        assert payload["source"] == {
            "group_id": value.environment.group_id,
            "document_id": value.environment.source_document_id,
            "source_scope": "control_document",
            "chunk_id": None,
        }
        assert payload["feedback"] == []
        assert payload["digest"] is None
        assert payload["provenance"]["query"] == {
            "sha256": value.environment.result.query_provenance.sha256,
            "length": value.environment.result.query_provenance.length,
            "origin": "explicit",
        }
        _assert_unavailable(
            BaselinePreviewCommand(
                caller_user_id=value.caller_user_id,
                request_id=_request_id(),
                group_id=value.environment.group_id,
                digest_id=str(uuid4()),
            ),
            value.environment.sessions,
        )
    finally:
        value.environment.engine.dispose()


def test_positive_job_and_digest_preview_are_ordered_suppressed_and_read_only(
    tmp_path: Path,
) -> None:
    findings = ("second-looking finding", "first-looking finding")
    value = _successful_control_preview(tmp_path, "preview-positive.db", findings)
    assert value.digest_id is not None
    try:
        service = BaselinePreviewService(value.environment.sessions)
        by_job = service.load(
            BaselinePreviewCommand(
                caller_user_id=value.caller_user_id,
                request_id=_request_id(),
                group_id=value.environment.group_id,
                job_id=value.job_id,
            )
        )
        by_digest = service.load(
            BaselinePreviewCommand(
                caller_user_id=value.caller_user_id,
                request_id=_request_id(),
                group_id=value.environment.group_id,
                digest_id=value.digest_id,
            )
        )
        assert by_job.control_job == by_digest.control_job
        assert by_job.retrieval == by_digest.retrieval
        assert by_job.feedback == by_digest.feedback
        assert [item.ordinal for item in by_job.feedback] == [1, 2]
        assert [item.feedback_id for item in by_job.feedback] == list(
            value.feedback_ids
        )
        assert [item.feedback for item in by_job.feedback] == list(findings)
        assert by_job.digest is not None
        assert by_job.digest.state == "suppressed"
        assert by_job.digest.digest_id == value.digest_id

        serialized = json.dumps(by_job.to_dict(), sort_keys=True)
        for forbidden_value in (
            "alpha persistence query",
            "alpha evidence file 1",
            "authoritative source document",
            "fixture-protected-payload",
            "http://baseline-embedding.internal/v1/embeddings",
        ):
            assert forbidden_value not in serialized
        forbidden_keys = {
            "retrieval_query",
            "evidence",
            "renderer_output",
            "idempotency_key",
            "lease_token",
            "provider_request",
            "provider_response",
            "prompt",
            "endpoint_url",
            "credentials",
        }

        def keys(item):
            if isinstance(item, dict):
                for key, child in item.items():
                    yield key
                    yield from keys(child)
            elif isinstance(item, list):
                for child in item:
                    yield from keys(child)

        assert forbidden_keys.isdisjoint(set(keys(by_job.to_dict())))
        with value.environment.engine.connect() as connection:
            assert (
                connection.execute(
                    text(
                        "SELECT state FROM baseline_notification_outbox "
                        "WHERE outbox_id = :digest_id"
                    ),
                    {"digest_id": value.digest_id},
                ).scalar_one()
                == "suppressed"
            )
            assert (
                connection.execute(
                    text("SELECT count(*) FROM baseline_notification_outbox")
                ).scalar_one()
                == 1
            )
            assert (
                connection.execute(
                    text("SELECT count(*) FROM notification_event")
                ).scalar_one()
                == 0
            )
    finally:
        value.environment.engine.dispose()


def test_preview_invokes_no_retrieval_generation_or_delivery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _successful_control_preview(
        tmp_path,
        "preview-no-side-effects.db",
        ("already durable finding",),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("preview invoked a write or model path")

    try:
        monkeypatch.setattr(PersistentBaselineV1Retriever, "retrieve", forbidden)
        monkeypatch.setattr(BaselineGenerationService, "generate_control", forbidden)
        monkeypatch.setattr(
            BaselineNotificationOutboxDispatcher,
            "dispatch_one",
            forbidden,
        )
        preview = _load_by_job(value)
        assert preview.control_job.state == "feedback_persisted"
    finally:
        value.environment.engine.dispose()


def test_retained_legacy_chunk_response_shape_keeps_chunk_identity(
    tmp_path: Path,
) -> None:
    """The read contract retains historical legacy scope without coercion.

    Current control-job persistence creates only control_document runs; this
    serializer regression covers a retained pre-control legacy representation
    without weakening the migration's new-write guard.
    """

    value = _successful_control_preview(
        tmp_path, "preview-retained-legacy-shape.db", ("retained finding",)
    )
    try:
        preview = _load_by_job(value)
        retained = replace(
            preview,
            source=BaselinePreviewSource(
                group_id=preview.source.group_id,
                document_id=preview.source.document_id,
                source_scope="legacy_chunk",
                chunk_id=value.environment.source_chunk_id,
            ),
        )
        assert retained.to_dict()["source"] == {
            "group_id": value.environment.group_id,
            "document_id": value.environment.source_document_id,
            "source_scope": "legacy_chunk",
            "chunk_id": value.environment.source_chunk_id,
        }
    finally:
        value.environment.engine.dispose()


def test_request_contract_uses_job_not_retrieval_run_identity() -> None:
    caller = str(uuid4())
    group = str(uuid4())
    job = str(uuid4())
    command = parse_baseline_preview_request(
        {
            "schema_version": "baseline-preview.v1",
            "request_id": _request_id(),
            "group_id": group,
            "job_id": job,
        },
        caller_user_id=caller,
    )
    assert command.job_id == job
    assert command.digest_id is None
    for payload in (
        {
            "schema_version": "baseline-preview.v1",
            "request_id": _request_id(),
            "group_id": group,
            "run_id": str(uuid4()),
        },
        {
            "schema_version": "baseline-preview.v1",
            "request_id": _request_id(),
            "group_id": group,
            "job_id": job,
            "digest_id": str(uuid4()),
        },
    ):
        with pytest.raises(BaselinePreviewError) as error:
            parse_baseline_preview_request(payload, caller_user_id=caller)
        assert error.value.code == "baseline_preview_request_invalid"


@pytest.mark.parametrize(
    "mutation",
    [
        "membership",
        "source_document",
        "group",
        "approval",
        "feedback_fingerprint",
        "retrieval_fingerprint",
        "query_provenance",
    ],
)
def test_authorization_deletion_and_manifest_drift_are_generic_not_found(
    tmp_path: Path, mutation: str
) -> None:
    value = _successful_control_preview(
        tmp_path, f"preview-{mutation}.db", ("private finding",)
    )
    try:
        statements = {
            "membership": (
                "DELETE FROM user_to_group WHERE user_id = :user_id "
                "AND group_id = :group_id"
            ),
            "source_document": (
                "DELETE FROM document WHERE document_id = :document_id"
            ),
            "group": 'DELETE FROM "group" WHERE group_id = :group_id',
            "approval": (
                "UPDATE baseline_control_repository_approval SET state = 'disabled', "
                "disabled_at = CURRENT_TIMESTAMP WHERE registration_id = "
                "(SELECT changed_repository_registration_id FROM "
                "baseline_control_run_job WHERE job_id = :job_id)"
            ),
            "feedback_fingerprint": (
                "UPDATE feedback SET generation_output_fingerprint = :bad_hash "
                "WHERE baseline_retrieval_run_id = :run_id"
            ),
            "retrieval_fingerprint": (
                "UPDATE baseline_retrieval_run SET generation_output_fingerprint = "
                ":bad_hash WHERE run_id = :run_id"
            ),
            "query_provenance": (
                "UPDATE baseline_retrieval_run SET query_sha256 = :bad_hash "
                "WHERE run_id = :run_id"
            ),
        }
        with value.environment.engine.begin() as connection:
            connection.execute(
                text(statements[mutation]),
                {
                    "user_id": value.caller_user_id,
                    "group_id": value.environment.group_id,
                    "document_id": value.environment.source_document_id,
                    "job_id": value.job_id,
                    "run_id": value.run_id,
                    "bad_hash": "0" * 64,
                },
            )
        _assert_unavailable(
            BaselinePreviewCommand(
                caller_user_id=value.caller_user_id,
                request_id=_request_id(),
                group_id=value.environment.group_id,
                job_id=value.job_id,
            ),
            value.environment.sessions,
        )
    finally:
        value.environment.engine.dispose()


def test_cross_user_and_group_are_indistinguishable(tmp_path: Path) -> None:
    value = _successful_control_preview(
        tmp_path, "preview-cross-scope.db", ("private finding",)
    )
    try:
        for caller, group in (
            (str(uuid4()), value.environment.group_id),
            (value.caller_user_id, str(uuid4())),
        ):
            _assert_unavailable(
                BaselinePreviewCommand(
                    caller_user_id=caller,
                    request_id=_request_id(),
                    group_id=group,
                    job_id=value.job_id,
                ),
                value.environment.sessions,
            )
    finally:
        value.environment.engine.dispose()


def test_corrupted_digest_manifest_is_generic_not_found(tmp_path: Path) -> None:
    value = _successful_control_preview(
        tmp_path,
        "preview-corrupt-digest.db",
        ("private finding",),
    )
    try:
        with value.environment.engine.begin() as connection:
            # Simulate on-disk corruption beyond the immutable new-write guard.
            connection.exec_driver_sql("DROP TRIGGER trg_bl_notify_payload_immutable")
            connection.execute(
                text(
                    "UPDATE baseline_notification_outbox SET "
                    "finding_manifest_hash = :bad_hash WHERE run_id = :run_id"
                ),
                {"bad_hash": "0" * 64, "run_id": value.run_id},
            )
        _assert_unavailable(
            BaselinePreviewCommand(
                caller_user_id=value.caller_user_id,
                request_id=_request_id(),
                group_id=value.environment.group_id,
                job_id=value.job_id,
            ),
            value.environment.sessions,
        )
    finally:
        value.environment.engine.dispose()


def _api_client(
    value: PreviewEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    *,
    base_url: str = "https://core.example.test",
    caller_user_id: str | None = None,
):
    monkeypatch.setattr(api.compair, "Session", value.environment.sessions)
    monkeypatch.setattr(
        api,
        "get_settings_dependency",
        lambda: SimpleNamespace(
            require_authentication=True,
            baseline_control_plane_allow_insecure_loopback=False,
            baseline_control_plane_trusted_proxy_allowlist="",
        ),
    )
    app = FastAPI()
    app.include_router(api.core_router)
    dependency = (
        signature(api.post_baseline_preview_v1).parameters["current_user"].default
    )
    assert dependency.dependency is api.get_current_user
    app.dependency_overrides[api.get_current_user] = lambda: SimpleNamespace(
        user_id=caller_user_id or value.caller_user_id
    )
    return TestClient(app, base_url=base_url)


def _preview_request(value: PreviewEnvironment) -> dict[str, object]:
    return {
        "schema_version": "baseline-preview.v1",
        "request_id": _request_id(),
        "group_id": value.environment.group_id,
        "job_id": value.job_id,
    }


def test_api_is_authenticated_post_body_only_and_does_not_log_feedback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    private_feedback = "PRIVATE-FEEDBACK-MUST-NOT-ENTER-LOGS"
    value = _successful_control_preview(tmp_path, "preview-api.db", (private_feedback,))
    try:
        client = _api_client(value, monkeypatch)
        with client:
            get_response = client.get(
                "/baseline/preview/v1?group_id=advertised&run_id=obsolete"
            )
            response = client.post("/baseline/preview/v1", json=_preview_request(value))
        assert get_response.status_code == 405
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json()["control_job"]["job_id"] == value.job_id
        assert response.json()["feedback"][0]["feedback"] == private_feedback
        assert private_feedback not in caplog.text
    finally:
        value.environment.engine.dispose()


def test_api_cross_user_is_generic_not_found_without_feedback_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    private_feedback = "PRIVATE-CROSS-USER-FEEDBACK"
    value = _successful_control_preview(
        tmp_path,
        "preview-api-cross-user.db",
        (private_feedback,),
    )
    try:
        client = _api_client(
            value,
            monkeypatch,
            caller_user_id=str(uuid4()),
        )
        with client:
            response = client.post(
                "/baseline/preview/v1",
                json=_preview_request(value),
            )
        assert response.status_code == 404
        assert response.json()["code"] == "preview_not_found"
        assert private_feedback not in response.text
        assert "job" not in response.text.lower()
    finally:
        value.environment.engine.dispose()


@pytest.mark.parametrize(
    ("body", "content_type", "expected_status"),
    [
        (
            b"".join(
                (
                    b'{"schema_version":"baseline-preview.v1",',
                    b'"request_id":"00000000-0000-4000-8000-000000000001",',
                    b'"group_id":"00000000-0000-4000-8000-000000000002",',
                    b'"group_id":"00000000-0000-4000-8000-000000000003"}',
                )
            ),
            "application/json",
            400,
        ),
        (b'{"value":NaN}', "application/json", 400),
        (b"\xff", "application/json", 400),
        (b"{}", "text/plain", 400),
        (b" " * 4097, "application/json", 413),
    ],
)
def test_api_strict_request_boundary_rejects_malformed_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
    content_type: str,
    expected_status: int,
) -> None:
    value = _successful_control_preview(tmp_path, f"strict-{uuid4()}.db", ())
    try:
        client = _api_client(value, monkeypatch)
        with client:
            response = client.post(
                "/baseline/preview/v1",
                content=body,
                headers={"Content-Type": content_type},
            )
        assert response.status_code == expected_status
        assert response.json()["schema_version"] == "baseline-preview.v1"
        assert "private" not in response.text.lower()
    finally:
        value.environment.engine.dispose()


def test_api_untrusted_proxy_spoof_is_not_a_loopback_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _successful_control_preview(tmp_path, "proxy-preview.db", ())
    try:
        monkeypatch.setattr(api.compair, "Session", value.environment.sessions)
        monkeypatch.setattr(
            api,
            "get_settings_dependency",
            lambda: SimpleNamespace(
                require_authentication=True,
                baseline_control_plane_allow_insecure_loopback=True,
                baseline_control_plane_trusted_proxy_allowlist="",
            ),
        )
        app = FastAPI()
        app.include_router(api.core_router)
        app.dependency_overrides[api.get_current_user] = lambda: SimpleNamespace(
            user_id=value.caller_user_id
        )
        with TestClient(app, base_url="http://core.example.test") as client:
            response = client.post(
                "/baseline/preview/v1",
                json=_preview_request(value),
                headers={
                    "Forwarded": "for=127.0.0.1;proto=https",
                    "X-Forwarded-Proto": "https",
                    "Host": "localhost",
                },
            )
        assert response.status_code == 503
        assert response.json()["code"] == "preview_transport_unavailable"
    finally:
        value.environment.engine.dispose()
