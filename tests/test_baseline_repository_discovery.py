from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from test_baseline_control_plane import _add_group_member, make_control_environment
from test_baseline_control_plane_protocol import (
    ContractValidationError,
    _validate_schema,
)

from compair_core import db as core_db
from compair_core.baseline_control_plane_schema import repository_registration
from compair_core.compair.retrieval.control_plane import (
    REPOSITORY_ADMIN_SCHEMA_VERSION,
    REPOSITORY_DESCRIPTOR_VERSION,
    REPOSITORY_DISCOVERY_SCHEMA_VERSION,
    ControlPlaneError,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocol"
ARTIFACTS = (
    "baseline-repository-discovery.v1.md",
    "baseline-repository-discovery.v1.schema.json",
    "fixtures/baseline-repository-discovery.v1.valid.json",
    "fixtures/baseline-repository-discovery.v1.invalid.json",
)
PINNED_ARTIFACTS = {
    "baseline-repository-discovery.v1.md": (
        "2cca4e44b97a81a3ae25a84458c124776d9578fd079acd75b39086f0931eee26"
    ),
    "baseline-repository-discovery.v1.schema.json": (
        "09f76a3fac443dbcda85f47389508e8174a0383a1255bef0b4ac04c4f5d3424b"
    ),
    "fixtures/baseline-repository-discovery.v1.valid.json": (
        "6d6c46ca5789d53c4a12632b82529dc52cd3c63744135071e1346a64057a4086"
    ),
    "fixtures/baseline-repository-discovery.v1.invalid.json": (
        "e36c64563e0aa3e0bc1fc65821d3b28b53188269bd7399128a96f5fb219a3b25"
    ),
}
LOCAL_AUTHORITY = "compair-local-repository.v1"


@pytest.fixture
def environment(tmp_path: Path):
    engine = core_db.create_engine(
        f"sqlite:///{tmp_path / 'baseline-repository-discovery.db'}",
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    try:
        yield make_control_environment(engine)
    finally:
        engine.dispose()


def _register_payload(environment, repository_uid: str, source_document_id=None):
    return {
        "schema_version": REPOSITORY_ADMIN_SCHEMA_VERSION,
        "message_type": "repository_registration_create",
        "request_id": str(uuid4()),
        "group_id": environment.group_id,
        "identity_descriptor": {
            "version": REPOSITORY_DESCRIPTOR_VERSION,
            "authority": LOCAL_AUTHORITY,
            "repository_uid": repository_uid,
        },
        "source_document_id": source_document_id,
    }


def _list_payload(environment):
    return {
        "schema_version": REPOSITORY_DISCOVERY_SCHEMA_VERSION,
        "message_type": "repository_list_request",
        "request_id": str(uuid4()),
        "group_id": environment.group_id,
    }


def _inspect_payload(environment, registration_id: str):
    return {
        "schema_version": REPOSITORY_DISCOVERY_SCHEMA_VERSION,
        "message_type": "repository_inspect_request",
        "request_id": str(uuid4()),
        "group_id": environment.group_id,
        "registration_id": registration_id,
    }


def test_repository_discovery_protocol_artifacts_are_valid_and_safe() -> None:
    schema = json.loads((PROTOCOL / ARTIFACTS[1]).read_text(encoding="utf-8"))
    valid = json.loads((PROTOCOL / ARTIFACTS[2]).read_text(encoding="utf-8"))
    for message in valid["messages"]:
        _validate_schema(message, schema, schema)
    invalid = json.loads((PROTOCOL / ARTIFACTS[3]).read_text(encoding="utf-8"))
    for case in invalid["cases"]:
        with pytest.raises(ContractValidationError):
            _validate_schema(case["value"], schema, schema)
    serialized = b"".join((PROTOCOL / name).read_bytes() for name in ARTIFACTS)
    for forbidden in (
        b"authenticated remote",
        b"content_utf8",
        b"retrieval_query",
        b"lease_token",
        b"idempotency_key",
    ):
        assert forbidden not in serialized


def test_admin_registration_replay_discovery_state_and_source_lifecycle(
    environment,
) -> None:
    member_id = _add_group_member(environment.engine, group_id=environment.group_id)
    payload = _register_payload(
        environment,
        "local-cli-random-uid-000001",
        environment.source_document_id,
    )
    created = environment.service.register_repository(
        payload, caller_user_id=environment.user_id
    )
    replayed = environment.service.register_repository(
        payload, caller_user_id=environment.user_id
    )
    assert created["registration_id"] == replayed["registration_id"]
    assert replayed["replayed"] is True

    with pytest.raises(ControlPlaneError, match="not_found_or_forbidden"):
        environment.service.list_repository_registrations(
            _list_payload(environment), caller_user_id=member_id
        )

    listed = environment.service.list_repository_registrations(
        _list_payload(environment), caller_user_id=environment.user_id
    )
    registration_ids = [item["registration_id"] for item in listed["repositories"]]
    assert registration_ids == sorted(registration_ids)
    assert created["registration_id"] in registration_ids

    inspected = environment.service.inspect_repository_registration(
        _inspect_payload(environment, str(created["registration_id"])),
        caller_user_id=member_id,
    )
    assert inspected["repository"]["source_document_id"] == (
        environment.source_document_id
    )
    safe = json.dumps(inspected, sort_keys=True)
    assert (
        inspected["repository"]["identity_descriptor"] == payload["identity_descriptor"]
    )
    for forbidden in (
        "local_path",
        "remote_url",
        "created_by_user_id",
    ):
        assert forbidden not in safe

    disabled = environment.service.set_repository_registration_state(
        {
            "schema_version": REPOSITORY_ADMIN_SCHEMA_VERSION,
            "message_type": "repository_registration_state",
            "request_id": str(uuid4()),
            "group_id": environment.group_id,
            "registration_id": created["registration_id"],
            "active": False,
        },
        caller_user_id=environment.user_id,
    )
    assert disabled["state"] == "disabled"
    assert (
        environment.service.inspect_repository_registration(
            _inspect_payload(environment, str(created["registration_id"])),
            caller_user_id=member_id,
        )["repository"]["state"]
        == "disabled"
    )

    with environment.engine.begin() as connection:
        connection.execute(
            text("DELETE FROM document WHERE document_id = :document_id"),
            {"document_id": environment.source_document_id},
        )
    assert (
        environment.service.inspect_repository_registration(
            _inspect_payload(environment, str(created["registration_id"])),
            caller_user_id=member_id,
        )["repository"]["source_document_id"]
        is None
    )


def test_repository_registration_conflicting_source_and_group_delete(
    environment,
) -> None:
    payload = _register_payload(
        environment,
        "local-cli-random-uid-000002",
        environment.source_document_id,
    )
    created = environment.service.register_repository(
        payload, caller_user_id=environment.user_id
    )
    conflict = dict(payload)
    conflict["source_document_id"] = None
    with pytest.raises(ControlPlaneError, match="repository_registration_conflict"):
        environment.service.register_repository(
            conflict, caller_user_id=environment.user_id
        )

    with environment.engine.begin() as connection:
        connection.execute(
            text('DELETE FROM "group" WHERE group_id = :group_id'),
            {"group_id": environment.group_id},
        )
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count())
                .select_from(repository_registration)
                .where(
                    repository_registration.c.registration_id
                    == created["registration_id"]
                )
            ).scalar_one()
            == 0
        )
    with pytest.raises(ControlPlaneError, match="not_found_or_forbidden"):
        environment.service.inspect_repository_registration(
            _inspect_payload(environment, str(created["registration_id"])),
            caller_user_id=environment.user_id,
        )


def test_repository_discovery_http_authorization_and_strict_parsing(
    environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    from compair_core import api as api_module

    member_id = _add_group_member(environment.engine, group_id=environment.group_id)
    current_user = [environment.user_id]
    app = api_module.create_fastapi_app()
    app.dependency_overrides[api_module.get_current_user] = lambda: SimpleNamespace(
        user_id=current_user[0], username="repository@example.test", name="Repository"
    )
    monkeypatch.setattr(
        api_module, "_control_plane_service", lambda: environment.service
    )
    with TestClient(app, base_url="https://core.example.test") as client:
        listed = client.post(
            "/baseline/control/admin/v1/repositories/list",
            json=_list_payload(environment),
        )
        assert listed.status_code == 200
        registration_id = listed.json()["repositories"][0]["registration_id"]

        current_user[0] = member_id
        denied = client.post(
            "/baseline/control/admin/v1/repositories/list",
            json=_list_payload(environment),
        )
        assert denied.status_code == 404
        inspected = client.post(
            "/baseline/control/v1/repositories/inspect",
            json=_inspect_payload(environment, registration_id),
        )
        assert inspected.status_code == 200

        duplicate = (
            '{"schema_version":"baseline-repository-discovery.v1",'
            '"message_type":"repository_inspect_request",'
            f'"request_id":"{uuid4()}",'
            f'"group_id":"{environment.group_id}",'
            f'"group_id":"{environment.group_id}",'
            f'"registration_id":"{registration_id}"}}'
        )
        rejected = client.post(
            "/baseline/control/v1/repositories/inspect",
            content=duplicate.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        assert rejected.status_code == 400
        assert "group_id" not in rejected.text


def test_repository_discovery_artifact_hashes_are_reportable() -> None:
    # The test pins exact files through deterministic digest calculation; CLI
    # carries the same bytes and pins the reported values independently.
    digests = {
        name: hashlib.sha256((PROTOCOL / name).read_bytes()).hexdigest()
        for name in ARTIFACTS
    }
    assert digests == PINNED_ARTIFACTS
