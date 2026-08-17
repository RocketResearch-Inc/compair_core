from __future__ import annotations

import copy
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
import rfc8785
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import Engine, func, select, text, update
from sqlalchemy.exc import DatabaseError
from starlette.datastructures import Headers

from compair_core import db as core_db
from compair_core.baseline_control_plane_schema import (
    control_job,
    repository_approval,
    repository_registration,
    snapshot_content_part,
    snapshot_continuation_job,
    snapshot_staging,
)
from compair_core.compair import models
from compair_core.compair.retrieval.continuation_worker import (
    BaselineContinuationWorker,
    ContinuationWorkerError,
    ContinuationWorkerStage,
    InternalContinuationWorkerIdentity,
)
from compair_core.compair.retrieval.control_plane import (
    PROTOCOL_SHA256,
    PROTOCOL_VERSION,
    BaselineControlPlaneService,
    ControlPlaneError,
    ControlTransportStatus,
    ControlWriteStage,
    assess_control_transport,
    canonical_sha256,
    canonicalize,
    capabilities_response,
    decode_json_object,
)
from compair_core.compair.retrieval.corpus import (
    RetrievalCorpus,
    RetrievalCorpusFile,
    RetrievalCorpusGeneration,
    RetrievalIndexState,
    ensure_retrieval_corpus_schema,
)
from compair_core.compair.retrieval.ingestion import CorpusIngestionService
from compair_core.schema_migrations import run_schema_migrations


@dataclass(frozen=True)
class ControlEnvironment:
    engine: Engine
    user_id: str
    group_id: str
    source_document_id: str
    changed_repository_id: str
    sibling_repository_id: str
    service: BaselineControlPlaneService


def _seed_scope(engine: Engine) -> tuple[str, str, str, str, str]:
    user_id = str(uuid4())
    group_id = str(uuid4())
    document_id = str(uuid4())
    admin_id = str(uuid4())
    changed_repository_id = str(uuid4())
    sibling_repository_id = str(uuid4())
    now = datetime.now(timezone.utc)
    with engine.begin() as connection:
        connection.execute(
            text(
                'INSERT INTO "user" '
                "(user_id, username, name, datetime_registered, password_hash, "
                "password_salt, status, include_own_documents_in_feedback, "
                "default_publish, preferred_feedback_length, hide_affiliations) "
                "VALUES (:user_id, :username, 'Control Test User', :now, 'hash', "
                "'salt', 'active', false, true, 'Brief', false)"
            ),
            {"user_id": user_id, "username": f"user-{user_id}", "now": now},
        )
        connection.execute(
            text(
                'INSERT INTO "group" '
                "(group_id, name, datetime_created, category, description, visibility) "
                "VALUES (:group_id, 'Control test', :now, 'Other', '', 'private')"
            ),
            {"group_id": group_id, "now": now},
        )
        connection.execute(
            text(
                "INSERT INTO user_to_group (user_id, group_id) "
                "VALUES (:user_id, :group_id)"
            ),
            {"user_id": user_id, "group_id": group_id},
        )
        connection.execute(
            text(
                "INSERT INTO administrator (admin_id, user_id) "
                "VALUES (:admin_id, :user_id)"
            ),
            {"admin_id": admin_id, "user_id": user_id},
        )
        connection.execute(
            text(
                "INSERT INTO admin_to_group (admin_id, group_id) "
                "VALUES (:admin_id, :group_id)"
            ),
            {"admin_id": admin_id, "group_id": group_id},
        )
        connection.execute(
            text(
                "INSERT INTO document "
                "(document_id, user_id, author_id, title, content, doc_type, "
                "datetime_created, datetime_modified, is_published) "
                "VALUES (:document_id, :user_id, :user_id, 'Changed repository', "
                "'benign source', 'text', :now, :now, true)"
            ),
            {"document_id": document_id, "user_id": user_id, "now": now},
        )
        connection.execute(
            text(
                "INSERT INTO document_to_group (document_id, group_id) "
                "VALUES (:document_id, :group_id)"
            ),
            {"document_id": document_id, "group_id": group_id},
        )
        for repository_id, repository_uid, source_document_id in (
            (changed_repository_id, "upstream-changed-uid", document_id),
            (sibling_repository_id, "upstream-sibling-uid", None),
        ):
            connection.execute(
                repository_registration.insert().values(
                    registration_id=repository_id,
                    group_id=group_id,
                    repository_id=repository_id,
                    repository_name=repository_id,
                    source_document_id=source_document_id,
                    created_by_user_id=user_id,
                    enabled=True,
                    created_at=now,
                    updated_at=now,
                )
            )
            descriptor = {
                "version": "repository-identity.v1",
                "authority": "example.test",
                "repository_uid": repository_uid,
            }
            connection.execute(
                repository_approval.insert().values(
                    registration_id=repository_id,
                    group_id=group_id,
                    descriptor_version="repository-identity.v1",
                    repository_authority="example.test",
                    repository_uid=repository_uid,
                    descriptor_hash=canonical_sha256(descriptor),
                    state="active",
                    approved_by_user_id=user_id,
                    disabled_by_user_id=None,
                    created_at=now,
                    updated_at=now,
                    disabled_at=None,
                )
            )
    return (
        user_id,
        group_id,
        document_id,
        changed_repository_id,
        sibling_repository_id,
    )


def make_control_environment(engine: Engine) -> ControlEnvironment:
    models.Base.metadata.create_all(engine)
    ensure_retrieval_corpus_schema(engine)
    run_schema_migrations(engine)
    (
        user_id,
        group_id,
        source_document_id,
        changed_repository_id,
        sibling_repository_id,
    ) = _seed_scope(engine)
    return ControlEnvironment(
        engine=engine,
        user_id=user_id,
        group_id=group_id,
        source_document_id=source_document_id,
        changed_repository_id=changed_repository_id,
        sibling_repository_id=sibling_repository_id,
        service=BaselineControlPlaneService(engine),
    )


@pytest.fixture
def environment(tmp_path: Path):
    engine = core_db.create_engine(
        f"sqlite:///{tmp_path / 'baseline-control.db'}",
        connect_args={"check_same_thread": False, "timeout": 10},
    )
    try:
        yield make_control_environment(engine)
    finally:
        engine.dispose()


def _snapshot(
    group_id: str,
    source_document_id: str,
    changed_repository_id: str,
    sibling_repository_id: str,
    *,
    content: str = "héllo\n",
):
    encoded = content.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    changed = {
        "repository_id": changed_repository_id,
        "repository_name": "changed",
        "repository_revision": "a" * 40,
        "role": "changed",
        "base_revision": "b" * 40,
        "head_revision": "a" * 40,
        "source_document_id": source_document_id,
        "expected_file_count": 0,
    }
    sibling = {
        "repository_id": sibling_repository_id,
        "repository_name": "sibling",
        "repository_revision": "c" * 40,
        "role": "sibling",
        "expected_file_count": 2,
    }
    files = [
        {
            "ordinal": 1,
            "repository_id": sibling_repository_id,
            "repository_name": "sibling",
            "repository_revision": "c" * 40,
            "relative_path": "src/café.py",
            "git_mode": "100644",
            "git_object_id": "d" * 40,
            "file_state": "supported",
            "skip_reason": None,
            "byte_size": len(encoded),
            "content_sha256": digest,
            "content_required": True,
        },
        {
            "ordinal": 2,
            "repository_id": sibling_repository_id,
            "repository_name": "sibling",
            "repository_revision": "c" * 40,
            "relative_path": "src/link.py",
            "git_mode": "120000",
            "git_object_id": "e" * 40,
            "file_state": "symlink_rejected",
            "skip_reason": "symlink",
            "byte_size": 12,
            "content_sha256": "f" * 64,
            "content_required": False,
        },
    ]
    canonical_manifest = {
        "schema_version": "baseline-snapshot.v1",
        "changed_repository": changed,
        "sibling_repositories": [sibling],
        "files": files,
    }
    manifest_hash = canonical_sha256(canonical_manifest)
    return {
        **canonical_manifest,
        "group_id": group_id,
        "repository_count": 1,
        "total_file_count": 2,
        "supported_file_count": 1,
        "supported_content_bytes": len(encoded),
        "canonical_manifest_hash": manifest_hash,
        "snapshot_id": f"bsnap_{manifest_hash}",
    }, content


def _begin_payload(
    environment: ControlEnvironment,
    *,
    idempotency_key: str = "opaque-client-intent-token-000001",
):
    snapshot, content = _snapshot(
        environment.group_id,
        environment.source_document_id,
        environment.changed_repository_id,
        environment.sibling_repository_id,
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "message_type": "snapshot_begin",
        "request_id": str(uuid4()),
        "group_id": environment.group_id,
        "idempotency_key": idempotency_key,
        "snapshot": snapshot,
    }, content


def _part_payload(begin: dict[str, object], job_id: str, content: str):
    encoded = content.encode("utf-8")
    item = {
        "file_ordinal": 1,
        "byte_size": len(encoded),
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
        "content_utf8": content,
    }
    items = [item]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "message_type": "snapshot_content_part",
        "request_id": str(uuid4()),
        "group_id": begin["group_id"],
        "job_id": job_id,
        "snapshot_id": begin["snapshot"]["snapshot_id"],
        "part_ordinal": 1,
        "part_sha256": canonical_sha256(items),
        "content_items": items,
    }


def _commit_payload(begin: dict[str, object], job_id: str, part: dict[str, object]):
    descriptors = [
        {"part_ordinal": part["part_ordinal"], "part_sha256": part["part_sha256"]}
    ]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "message_type": "snapshot_commit",
        "request_id": str(uuid4()),
        "group_id": begin["group_id"],
        "job_id": job_id,
        "snapshot_id": begin["snapshot"]["snapshot_id"],
        "parts": descriptors,
        "content_manifest_hash": canonical_sha256(descriptors),
    }


def _status_payload(environment: ControlEnvironment, job_id: str):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "message_type": "job_status_request",
        "request_id": str(uuid4()),
        "group_id": environment.group_id,
        "job_id": job_id,
    }


def _stage_success(environment: ControlEnvironment):
    begin, content = _begin_payload(environment)
    accepted = environment.service.begin_snapshot(
        begin, caller_user_id=environment.user_id
    )
    job_id = str(accepted["job_id"])
    part = _part_payload(begin, job_id, content)
    raw_part = canonicalize(part)
    staged = environment.service.stage_content_part(
        part,
        caller_user_id=environment.user_id,
        request_body_sha256=hashlib.sha256(raw_part).hexdigest(),
        path_job_id=job_id,
    )
    commit = _commit_payload(begin, job_id, part)
    sealed = environment.service.commit_snapshot(
        commit,
        caller_user_id=environment.user_id,
        path_job_id=job_id,
    )
    return begin, part, commit, staged, sealed


def _row_counts(engine: Engine) -> tuple[int, int, int]:
    with engine.connect() as connection:
        return (
            connection.execute(
                select(func.count()).select_from(control_job)
            ).scalar_one(),
            connection.execute(
                select(func.count()).select_from(snapshot_staging)
            ).scalar_one(),
            connection.execute(
                select(func.count()).select_from(snapshot_content_part)
            ).scalar_one(),
        )


def _single_continuation_id(engine: Engine) -> str:
    with engine.connect() as connection:
        return str(
            connection.execute(
                select(snapshot_continuation_job.c.continuation_job_id)
            ).scalar_one()
        )


def _add_group_member(
    engine: Engine,
    *,
    group_id: str,
    make_admin: bool = False,
) -> str:
    user_id = str(uuid4())
    now = datetime.now(timezone.utc)
    with engine.begin() as connection:
        connection.execute(
            text(
                'INSERT INTO "user" '
                "(user_id, username, name, datetime_registered, password_hash, "
                "password_salt, status, include_own_documents_in_feedback, "
                "default_publish, preferred_feedback_length, hide_affiliations) "
                "VALUES (:user_id, :username, 'Control Member', :now, 'hash', "
                "'salt', 'active', false, true, 'Brief', false)"
            ),
            {"user_id": user_id, "username": f"member-{user_id}", "now": now},
        )
        connection.execute(
            text(
                "INSERT INTO user_to_group (user_id, group_id) "
                "VALUES (:user_id, :group_id)"
            ),
            {"user_id": user_id, "group_id": group_id},
        )
        if make_admin:
            admin_id = str(uuid4())
            connection.execute(
                text(
                    "INSERT INTO administrator (admin_id, user_id) "
                    "VALUES (:admin_id, :user_id)"
                ),
                {"admin_id": admin_id, "user_id": user_id},
            )
            connection.execute(
                text(
                    "INSERT INTO admin_to_group (admin_id, group_id) "
                    "VALUES (:admin_id, :group_id)"
                ),
                {"admin_id": admin_id, "group_id": group_id},
            )
    return user_id


def _registration_create_payload(
    environment: ControlEnvironment,
    *,
    repository_uid: str,
    source_document_id: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "baseline-repository-registration-admin.v1",
        "message_type": "repository_registration_create",
        "request_id": str(uuid4()),
        "group_id": environment.group_id,
        "identity_descriptor": {
            "version": "repository-identity.v1",
            "authority": "git.example.test",
            "repository_uid": repository_uid,
        },
        "source_document_id": source_document_id,
    }


def _registration_state_payload(
    environment: ControlEnvironment,
    registration_id: str,
    *,
    active: bool,
) -> dict[str, object]:
    return {
        "schema_version": "baseline-repository-registration-admin.v1",
        "message_type": "repository_registration_state",
        "request_id": str(uuid4()),
        "group_id": environment.group_id,
        "registration_id": registration_id,
        "active": active,
    }


def _continuation_status_payload(
    environment: ControlEnvironment,
    *,
    staging_job_id: str | None = None,
    continuation_job_id: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "baseline-snapshot-continuation.v1",
        "message_type": "continuation_job_status_request",
        "request_id": str(uuid4()),
        "group_id": environment.group_id,
        "staging_job_id": staging_job_id,
        "continuation_job_id": continuation_job_id,
    }


def test_pinned_rfc8785_canonicalization_and_strict_json() -> None:
    assert rfc8785.__version__ == "0.1.4"
    value = {"é": '\u000f\n"\\', "a": 1.0, "b": 1e-7}
    assert canonicalize(value) == (b'{"a":1,"b":1e-7,"\xc3\xa9":"\\u000f\\n\\"\\\\"}')
    assert (
        canonicalize({"numbers": [333333333.33333329, 1e30, 4.50, 2e-3, 1e-27]})
        == b'{"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27]}'
    )

    with pytest.raises(ControlPlaneError, match="invalid_contract"):
        decode_json_object(b'{"duplicate":1,"duplicate":2}')
    with pytest.raises(ControlPlaneError, match="invalid_contract"):
        decode_json_object(b'{"number":NaN}')


def test_transport_policy_and_capability_are_narrow_and_safe(
    environment: ControlEnvironment,
) -> None:
    secure = assess_control_transport(
        connection_scheme="https",
        peer_host="203.0.113.10",
        allow_insecure_loopback=False,
    )
    local = assess_control_transport(
        connection_scheme="http",
        peer_host="::1",
        allow_insecure_loopback=True,
    )
    remote_plaintext = assess_control_transport(
        connection_scheme="http",
        peer_host="203.0.113.10",
        allow_insecure_loopback=True,
    )
    mixed = assess_control_transport(
        connection_scheme="http",
        peer_host="127.0.0.1",
        allow_insecure_loopback=True,
        proxy_headers_present=True,
    )

    assert secure.status is ControlTransportStatus.SAFE
    assert local.status is ControlTransportStatus.LOCAL_OVERRIDE
    assert remote_plaintext.status is ControlTransportStatus.UNAVAILABLE
    assert mixed.status is ControlTransportStatus.UNAVAILABLE

    response = capabilities_response(
        request_id=str(uuid4()), group_id=environment.group_id, transport=local
    )
    serialized = json.dumps(response, sort_keys=True)
    assert response["protocol_version"] == PROTOCOL_VERSION
    assert response["protocol_sha256"] == PROTOCOL_SHA256
    assert response["operations"] == {
        "snapshot_staging": "safe",
        "corpus_ingestion": "unavailable",
        "index_build": "unavailable",
        "baseline_run": "unavailable",
    }
    assert response["staging_is_corpus_eligible"] is False
    assert response["staging_is_index_eligible"] is False
    assert "endpoint" not in serialized
    assert "credential" not in serialized


def test_proxy_trust_uses_only_the_immediate_peer_and_unambiguous_scheme() -> None:
    untrusted_spoof = assess_control_transport(
        connection_scheme="http",
        peer_host="203.0.113.10",
        allow_insecure_loopback=True,
        trusted_proxy_allowlist="10.20.0.0/16",
        forwarded_values=("for=127.0.0.1;proto=https;host=localhost",),
        x_forwarded_proto_values=("https",),
        proxy_headers_present=True,
    )
    trusted = assess_control_transport(
        connection_scheme="http",
        peer_host="10.20.1.9",
        allow_insecure_loopback=False,
        trusted_proxy_allowlist="10.20.0.0/16,2001:db8:1234::/48",
        forwarded_values=("for=203.0.113.10;proto=https",),
        x_forwarded_proto_values=("https",),
        proxy_headers_present=True,
    )
    ambiguous = assess_control_transport(
        connection_scheme="http",
        peer_host="10.20.1.9",
        allow_insecure_loopback=False,
        trusted_proxy_allowlist="10.20.0.0/16",
        forwarded_values=("for=203.0.113.10;proto=https",),
        x_forwarded_proto_values=("http",),
        proxy_headers_present=True,
    )
    loopback_with_proxy_headers = assess_control_transport(
        connection_scheme="http",
        peer_host="127.0.0.1",
        allow_insecure_loopback=True,
        forwarded_values=("for=127.0.0.1;proto=https",),
        proxy_headers_present=True,
    )
    invalid_configuration = assess_control_transport(
        connection_scheme="http",
        peer_host="10.20.1.9",
        allow_insecure_loopback=True,
        trusted_proxy_allowlist="not-a-network",
        x_forwarded_proto_values=("https",),
        proxy_headers_present=True,
    )

    assert untrusted_spoof.status is ControlTransportStatus.UNAVAILABLE
    assert trusted.status is ControlTransportStatus.SAFE
    assert trusted.encrypted is True
    assert ambiguous.status is ControlTransportStatus.UNAVAILABLE
    assert loopback_with_proxy_headers.status is ControlTransportStatus.UNAVAILABLE
    assert invalid_configuration.status is ControlTransportStatus.UNAVAILABLE


def test_api_transport_wiring_honors_configured_proxy_cidr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from compair_core import api as api_module

    monkeypatch.setattr(
        api_module,
        "get_settings_dependency",
        lambda: SimpleNamespace(
            baseline_control_plane_allow_insecure_loopback=False,
            baseline_control_plane_trusted_proxy_allowlist="10.30.0.0/16",
        ),
    )
    request = SimpleNamespace(
        scope={"scheme": "http"},
        client=SimpleNamespace(host="10.30.4.8"),
        headers=Headers(
            {
                "host": "attacker.invalid",
                "forwarded": "for=198.51.100.8;proto=https;host=attacker.invalid",
                "x-forwarded-for": "127.0.0.1",
                "x-forwarded-proto": "https",
            }
        ),
    )

    capability = api_module._control_transport_capability(request)

    assert capability.status is ControlTransportStatus.SAFE
    assert capability.encrypted is True


def test_protocol_mismatch_and_authorization_fail_before_writes(
    environment: ControlEnvironment,
) -> None:
    payload, _content = _begin_payload(environment)
    mismatched = copy.deepcopy(payload)
    mismatched["protocol_sha256"] = "0" * 64
    with pytest.raises(ControlPlaneError, match="protocol_mismatch"):
        environment.service.begin_snapshot(
            mismatched, caller_user_id=environment.user_id
        )
    assert _row_counts(environment.engine) == (0, 0, 0)

    with pytest.raises(ControlPlaneError, match="not_found_or_forbidden"):
        environment.service.begin_snapshot(payload, caller_user_id=str(uuid4()))
    assert _row_counts(environment.engine) == (0, 0, 0)

    with environment.engine.begin() as connection:
        connection.execute(
            update(repository_registration)
            .where(
                repository_registration.c.registration_id
                == environment.sibling_repository_id
            )
            .values(enabled=False)
        )
    with pytest.raises(ControlPlaneError, match="repository_not_authorized"):
        environment.service.begin_snapshot(payload, caller_user_id=environment.user_id)
    assert _row_counts(environment.engine) == (0, 0, 0)


def test_repository_registration_is_group_admin_only_immutable_and_revocable(
    environment: ControlEnvironment,
) -> None:
    member_id = _add_group_member(
        environment.engine,
        group_id=environment.group_id,
    )
    payload = _registration_create_payload(
        environment,
        repository_uid="new-upstream-repository-uid",
    )
    with environment.engine.connect() as connection:
        approval_count = connection.execute(
            select(func.count()).select_from(repository_approval)
        ).scalar_one()

    with pytest.raises(ControlPlaneError, match="not_found_or_forbidden"):
        environment.service.register_repository(payload, caller_user_id=member_id)
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(repository_approval)
            ).scalar_one()
            == approval_count
        )

    unsafe = copy.deepcopy(payload)
    unsafe["local_path"] = "/private/working/tree"
    with pytest.raises(ControlPlaneError, match="invalid_contract"):
        environment.service.register_repository(
            unsafe, caller_user_id=environment.user_id
        )

    failing_service = BaselineControlPlaneService(
        environment.engine,
        stage_hook=lambda stage: (
            (_ for _ in ()).throw(RuntimeError("registration_failure"))
            if stage is ControlWriteStage.REGISTRATION
            else None
        ),
    )
    with pytest.raises(RuntimeError, match="registration_failure"):
        failing_service.register_repository(
            payload,
            caller_user_id=environment.user_id,
        )
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(repository_approval)
            ).scalar_one()
            == approval_count
        )

    created = environment.service.register_repository(
        payload,
        caller_user_id=environment.user_id,
    )
    replay = copy.deepcopy(payload)
    replay["request_id"] = str(uuid4())
    replayed = environment.service.register_repository(
        replay,
        caller_user_id=environment.user_id,
    )
    registration_id = str(created["registration_id"])
    assert UUID(registration_id)
    assert replayed["registration_id"] == registration_id
    assert replayed["replayed"] is True

    conflicting = copy.deepcopy(replay)
    conflicting["request_id"] = str(uuid4())
    conflicting["source_document_id"] = environment.source_document_id
    with pytest.raises(ControlPlaneError, match="repository_registration_conflict"):
        environment.service.register_repository(
            conflicting,
            caller_user_id=environment.user_id,
        )

    disable = _registration_state_payload(
        environment,
        registration_id,
        active=False,
    )
    with pytest.raises(ControlPlaneError, match="not_found_or_forbidden"):
        environment.service.set_repository_registration_state(
            disable,
            caller_user_id=member_id,
        )
    disabled = environment.service.set_repository_registration_state(
        disable,
        caller_user_id=environment.user_id,
    )
    assert disabled["state"] == "disabled"
    reactivated = environment.service.set_repository_registration_state(
        _registration_state_payload(environment, registration_id, active=True),
        caller_user_id=environment.user_id,
    )
    assert reactivated["state"] == "active"

    with environment.engine.connect() as connection:
        registration = (
            connection.execute(
                select(repository_registration).where(
                    repository_registration.c.registration_id == registration_id
                )
            )
            .mappings()
            .one()
        )
        approval = (
            connection.execute(
                select(repository_approval).where(
                    repository_approval.c.registration_id == registration_id
                )
            )
            .mappings()
            .one()
        )
    persisted = json.dumps(
        {**dict(registration), **dict(approval)}, default=str, sort_keys=True
    )
    assert "/private/working/tree" not in persisted
    assert "repository_revision" not in persisted
    assert registration["repository_id"] == registration_id
    assert approval["descriptor_hash"] == created["identity_descriptor_hash"]

    with pytest.raises(DatabaseError), environment.engine.begin() as connection:
        connection.execute(
            update(repository_approval)
            .where(repository_approval.c.registration_id == registration_id)
            .values(repository_uid="mutated-upstream-id")
        )


def test_revoked_registration_blocks_commit_and_claim_without_erasing_audit(
    environment: ControlEnvironment,
) -> None:
    begin, content = _begin_payload(environment)
    accepted = environment.service.begin_snapshot(
        begin, caller_user_id=environment.user_id
    )
    job_id = str(accepted["job_id"])
    part = _part_payload(begin, job_id, content)
    environment.service.stage_content_part(
        part,
        caller_user_id=environment.user_id,
        request_body_sha256=hashlib.sha256(canonicalize(part)).hexdigest(),
        path_job_id=job_id,
    )
    environment.service.set_repository_registration_state(
        _registration_state_payload(
            environment,
            environment.sibling_repository_id,
            active=False,
        ),
        caller_user_id=environment.user_id,
    )
    commit = _commit_payload(begin, job_id, part)
    with pytest.raises(ControlPlaneError, match="repository_not_authorized"):
        environment.service.commit_snapshot(
            commit,
            caller_user_id=environment.user_id,
            path_job_id=job_id,
        )
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(snapshot_continuation_job)
            ).scalar_one()
            == 0
        )

    environment.service.set_repository_registration_state(
        _registration_state_payload(
            environment,
            environment.sibling_repository_id,
            active=True,
        ),
        caller_user_id=environment.user_id,
    )
    environment.service.commit_snapshot(
        commit,
        caller_user_id=environment.user_id,
        path_job_id=job_id,
    )
    continuation_id = _single_continuation_id(environment.engine)
    environment.service.set_repository_registration_state(
        _registration_state_payload(
            environment,
            environment.sibling_repository_id,
            active=False,
        ),
        caller_user_id=environment.user_id,
    )
    with pytest.raises(ControlPlaneError, match="repository_not_authorized"):
        environment.service.claim_continuation_job(
            caller_user_id=environment.user_id,
            group_id=environment.group_id,
            job_id=continuation_id,
        )
    with environment.engine.connect() as connection:
        continuation = (
            connection.execute(select(snapshot_continuation_job)).mappings().one()
        )
        assert continuation["state"] == "queued"
        assert (
            connection.execute(
                select(func.count()).select_from(repository_approval)
            ).scalar_one()
            == 2
        )


def test_source_deletion_preserves_sealed_audit_but_blocks_claim(
    environment: ControlEnvironment,
) -> None:
    _begin, _part, _commit, _staged, _sealed = _stage_success(environment)
    continuation_id = _single_continuation_id(environment.engine)
    with environment.engine.begin() as connection:
        connection.execute(
            text("DELETE FROM document WHERE document_id = :document_id"),
            {"document_id": environment.source_document_id},
        )
    with pytest.raises(ControlPlaneError, match="source_not_authorized"):
        environment.service.claim_continuation_job(
            caller_user_id=environment.user_id,
            group_id=environment.group_id,
            job_id=continuation_id,
        )
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(snapshot_staging)
            ).scalar_one()
            == 1
        )
        assert (
            connection.execute(
                select(func.count()).select_from(snapshot_continuation_job)
            ).scalar_one()
            == 1
        )
        changed_registration = (
            connection.execute(
                select(repository_registration).where(
                    repository_registration.c.registration_id
                    == environment.changed_repository_id
                )
            )
            .mappings()
            .one()
        )
        assert changed_registration["source_document_id"] is None


def test_originating_user_membership_removal_blocks_claim_without_erasing_audit(
    environment: ControlEnvironment,
) -> None:
    _begin, _part, _commit, _staged, _sealed = _stage_success(environment)
    continuation_id = _single_continuation_id(environment.engine)
    with environment.engine.begin() as connection:
        connection.execute(
            text(
                "DELETE FROM user_to_group "
                "WHERE user_id = :user_id AND group_id = :group_id"
            ),
            {
                "user_id": environment.user_id,
                "group_id": environment.group_id,
            },
        )
    with pytest.raises(ControlPlaneError, match="not_found_or_forbidden"):
        environment.service.claim_continuation_job(
            caller_user_id=environment.user_id,
            group_id=environment.group_id,
            job_id=continuation_id,
        )
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(snapshot_continuation_job)
            ).scalar_one()
            == 1
        )


def test_group_deletion_cascades_all_control_plane_state(
    environment: ControlEnvironment,
) -> None:
    _stage_success(environment)
    with environment.engine.begin() as connection:
        connection.execute(
            text('DELETE FROM "group" WHERE group_id = :group_id'),
            {"group_id": environment.group_id},
        )
    with environment.engine.connect() as connection:
        for table in (
            repository_registration,
            repository_approval,
            control_job,
            snapshot_staging,
            snapshot_content_part,
            snapshot_continuation_job,
        ):
            assert (
                connection.execute(select(func.count()).select_from(table)).scalar_one()
                == 0
            )


def test_complete_staging_is_immutable_durable_and_never_eligible(
    environment: ControlEnvironment,
) -> None:
    begin, part, _commit, staged, sealed = _stage_success(environment)
    job_id = str(sealed["job_id"])

    assert staged["staging"]["state"] == "open"
    assert sealed["state"] == "succeeded"
    assert sealed["staging"]["state"] == "sealed"
    assert sealed["result"] == {
        "snapshot_id": begin["snapshot"]["snapshot_id"],
        "staging_state": "sealed",
        "corpus_eligible": False,
        "index_eligible": False,
    }
    continuation_status = environment.service.continuation_status(
        _continuation_status_payload(environment, staging_job_id=job_id),
        caller_user_id=environment.user_id,
    )
    assert continuation_status["operation"] == "sealed_snapshot_continue"
    assert continuation_status["state"] == "queued"
    assert continuation_status["result"]["corpus_eligible"] is False
    assert continuation_status["result"]["index_eligible"] is False
    assert _row_counts(environment.engine) == (1, 1, 1)

    status = environment.service.job_status(
        _status_payload(environment, job_id),
        caller_user_id=environment.user_id,
    )
    assert status["result"]["corpus_eligible"] is False
    assert status["result"]["index_eligible"] is False
    assert "content_utf8" not in json.dumps(status)

    with pytest.raises(DatabaseError), environment.engine.begin() as connection:
        connection.execute(
            update(snapshot_content_part)
            .where(snapshot_content_part.c.part_ordinal == 1)
            .values(part_sha256="0" * 64)
        )
    with pytest.raises(DatabaseError), environment.engine.begin() as connection:
        connection.execute(
            update(snapshot_staging)
            .where(snapshot_staging.c.job_id == job_id)
            .values(canonical_manifest_hash="0" * 64)
        )

    environment.engine.dispose()
    restarted = BaselineControlPlaneService(environment.engine)
    persisted = restarted.job_status(
        _status_payload(environment, job_id),
        caller_user_id=environment.user_id,
    )
    assert persisted["state"] == "succeeded"
    assert persisted["staging"]["state"] == "sealed"
    assert part["content_items"][0]["content_utf8"] not in json.dumps(persisted)
    persisted_continuation = restarted.continuation_status(
        _continuation_status_payload(environment, staging_job_id=job_id),
        caller_user_id=environment.user_id,
    )
    assert persisted_continuation["job_id"] == continuation_status["job_id"]


def test_continuation_replay_claim_failure_expiry_restart_and_no_content_mutation(
    environment: ControlEnvironment,
) -> None:
    now = datetime(2026, 8, 16, tzinfo=timezone.utc)
    clock_value = [now]
    service = BaselineControlPlaneService(
        environment.engine,
        clock=lambda: clock_value[0],
    )
    begin, content = _begin_payload(environment)
    accepted = service.begin_snapshot(begin, caller_user_id=environment.user_id)
    staging_job_id = str(accepted["job_id"])
    part = _part_payload(begin, staging_job_id, content)
    service.stage_content_part(
        part,
        caller_user_id=environment.user_id,
        request_body_sha256=hashlib.sha256(canonicalize(part)).hexdigest(),
        path_job_id=staging_job_id,
    )
    commit = _commit_payload(begin, staging_job_id, part)
    service.commit_snapshot(
        commit,
        caller_user_id=environment.user_id,
        path_job_id=staging_job_id,
    )
    continuation_id = _single_continuation_id(environment.engine)
    assert continuation_id != staging_job_id

    replay = copy.deepcopy(commit)
    replay["request_id"] = str(uuid4())
    replayed = service.commit_snapshot(
        replay,
        caller_user_id=environment.user_id,
        path_job_id=staging_job_id,
    )
    assert replayed["replayed"] is True
    assert _single_continuation_id(environment.engine) == continuation_id

    with environment.engine.connect() as connection:
        stored_content = connection.execute(
            select(snapshot_content_part.c.canonical_content_items_json)
        ).scalar_one()
        corpus_generation_count = connection.exec_driver_sql(
            "SELECT COUNT(*) FROM retrieval_corpus_generation"
        ).scalar_one()
        continuation_count = connection.execute(
            select(func.count()).select_from(snapshot_continuation_job)
        ).scalar_one()
    assert continuation_count == 1

    first = service.claim_continuation_job(
        caller_user_id=environment.user_id,
        group_id=environment.group_id,
        job_id=continuation_id,
    )
    with pytest.raises(ControlPlaneError, match="job_lease_unavailable"):
        service.claim_continuation_job(
            caller_user_id=environment.user_id,
            group_id=environment.group_id,
            job_id=continuation_id,
        )
    with pytest.raises(ControlPlaneError, match="invalid_contract"):
        service.record_continuation_failure(
            caller_user_id=environment.user_id,
            group_id=environment.group_id,
            job_id=continuation_id,
            lease_token=first.lease_token,
            error_code="PRIVATE /repository/path",
            retryable=True,
        )
    service.record_continuation_failure(
        caller_user_id=environment.user_id,
        group_id=environment.group_id,
        job_id=continuation_id,
        lease_token=first.lease_token,
        error_code="worker_unavailable",
        retryable=True,
    )
    second = service.claim_continuation_job(
        caller_user_id=environment.user_id,
        group_id=environment.group_id,
        job_id=continuation_id,
    )
    assert second.attempt_count == 2
    clock_value[0] += timedelta(minutes=6)
    third = service.claim_continuation_job(
        caller_user_id=environment.user_id,
        group_id=environment.group_id,
        job_id=continuation_id,
    )
    assert third.attempt_count == 3
    assert len({first.lease_token, second.lease_token, third.lease_token}) == 3

    environment.engine.dispose()
    restarted = BaselineControlPlaneService(environment.engine)
    status = restarted.continuation_status(
        _continuation_status_payload(
            environment,
            continuation_job_id=continuation_id,
        ),
        caller_user_id=environment.user_id,
    )
    assert status["operation"] == "sealed_snapshot_continue"
    assert status["state"] == "running"
    assert status["attempt"] == 3
    assert status["result"]["corpus_eligible"] is False
    assert status["result"]["index_eligible"] is False
    serialized = json.dumps(status, sort_keys=True)
    assert content not in serialized

    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(snapshot_content_part.c.canonical_content_items_json)
            ).scalar_one()
            == stored_content
        )
        assert (
            connection.exec_driver_sql(
                "SELECT COUNT(*) FROM retrieval_corpus_generation"
            ).scalar_one()
            == corpus_generation_count
        )


def test_concurrent_continuation_claim_has_one_winner(
    environment: ControlEnvironment,
) -> None:
    _begin, _part, _commit, _staged, _sealed = _stage_success(environment)
    continuation_id = _single_continuation_id(environment.engine)
    barrier = threading.Barrier(2)

    def claim():
        barrier.wait()
        try:
            return environment.service.claim_continuation_job(
                caller_user_id=environment.user_id,
                group_id=environment.group_id,
                job_id=continuation_id,
            )
        except ControlPlaneError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: claim(), range(2)))
    assert sum(not isinstance(result, str) for result in results) == 1
    assert sum(result == "job_lease_unavailable" for result in results) == 1
    with environment.engine.connect() as connection:
        continuation = (
            connection.execute(select(snapshot_continuation_job)).mappings().one()
        )
    assert continuation["state"] == "running"
    assert continuation["attempt_count"] == 1


def test_active_staging_lease_defers_expiry_cleanup(
    environment: ControlEnvironment,
) -> None:
    now = datetime(2026, 8, 16, tzinfo=timezone.utc)
    clock_value = [now]
    service = BaselineControlPlaneService(
        environment.engine,
        clock=lambda: clock_value[0],
    )
    begin, _content = _begin_payload(environment)
    accepted = service.begin_snapshot(begin, caller_user_id=environment.user_id)
    job_id = str(accepted["job_id"])
    service.acquire_job_lease(
        caller_user_id=environment.user_id,
        group_id=environment.group_id,
        job_id=job_id,
        lifetime=timedelta(hours=30),
    )
    clock_value[0] += timedelta(hours=25)
    assert service.expire_staging_sessions() == 0
    clock_value[0] += timedelta(hours=6)
    assert service.expire_staging_sessions() == 1


def test_replay_conflict_malformed_part_and_commit_are_fail_closed(
    environment: ControlEnvironment,
) -> None:
    begin, content = _begin_payload(environment)
    first = environment.service.begin_snapshot(
        begin, caller_user_id=environment.user_id
    )
    replay = copy.deepcopy(begin)
    replay["request_id"] = str(uuid4())
    replayed = environment.service.begin_snapshot(
        replay, caller_user_id=environment.user_id
    )
    assert replayed["job_id"] == first["job_id"]
    assert replayed["replayed"] is True

    conflict = copy.deepcopy(begin)
    conflict["request_id"] = str(uuid4())
    conflict["snapshot"]["changed_repository"]["base_revision"] = "9" * 40
    canonical_manifest = {
        key: conflict["snapshot"][key]
        for key in (
            "schema_version",
            "changed_repository",
            "sibling_repositories",
            "files",
        )
    }
    conflict_hash = canonical_sha256(canonical_manifest)
    conflict["snapshot"]["canonical_manifest_hash"] = conflict_hash
    conflict["snapshot"]["snapshot_id"] = f"bsnap_{conflict_hash}"
    with pytest.raises(ControlPlaneError, match="idempotency_conflict"):
        environment.service.begin_snapshot(conflict, caller_user_id=environment.user_id)

    job_id = str(first["job_id"])
    part = _part_payload(begin, job_id, content)
    malformed = copy.deepcopy(part)
    malformed["content_items"] = malformed["content_items"] * 1001
    with pytest.raises(ControlPlaneError, match="limit_exceeded"):
        environment.service.stage_content_part(
            malformed,
            caller_user_id=environment.user_id,
            request_body_sha256="1" * 64,
            path_job_id=job_id,
        )
    assert _row_counts(environment.engine) == (1, 1, 0)

    raw = canonicalize(part)
    environment.service.stage_content_part(
        part,
        caller_user_id=environment.user_id,
        request_body_sha256=hashlib.sha256(raw).hexdigest(),
        path_job_id=job_id,
    )
    conflicting_part = copy.deepcopy(part)
    conflicting_part["request_id"] = str(uuid4())
    with pytest.raises(ControlPlaneError, match="part_conflict"):
        environment.service.stage_content_part(
            conflicting_part,
            caller_user_id=environment.user_id,
            request_body_sha256=hashlib.sha256(
                canonicalize(conflicting_part)
            ).hexdigest(),
            path_job_id=job_id,
        )

    incomplete_commit = _commit_payload(begin, job_id, part)
    incomplete_commit["parts"] = []
    incomplete_commit["content_manifest_hash"] = canonical_sha256([])
    with pytest.raises(ControlPlaneError, match="incomplete_staging"):
        environment.service.commit_snapshot(
            incomplete_commit,
            caller_user_id=environment.user_id,
            path_job_id=job_id,
        )


@pytest.mark.parametrize(
    "failing_stage, expected_counts",
    [
        (ControlWriteStage.JOB, (0, 0, 0)),
        (ControlWriteStage.STAGING, (0, 0, 0)),
    ],
)
def test_begin_rolls_back_at_every_write_stage(
    environment: ControlEnvironment,
    failing_stage: ControlWriteStage,
    expected_counts: tuple[int, int, int],
) -> None:
    payload, _content = _begin_payload(environment)

    def fail(stage: ControlWriteStage) -> None:
        if stage is failing_stage:
            raise RuntimeError("injected_failure")

    service = BaselineControlPlaneService(environment.engine, stage_hook=fail)
    with pytest.raises(RuntimeError, match="injected_failure"):
        service.begin_snapshot(payload, caller_user_id=environment.user_id)
    assert _row_counts(environment.engine) == expected_counts


def test_part_and_commit_failures_roll_back_completely(
    environment: ControlEnvironment,
) -> None:
    begin, content = _begin_payload(environment)
    accepted = environment.service.begin_snapshot(
        begin, caller_user_id=environment.user_id
    )
    job_id = str(accepted["job_id"])
    part = _part_payload(begin, job_id, content)
    raw_hash = hashlib.sha256(canonicalize(part)).hexdigest()

    part_service = BaselineControlPlaneService(
        environment.engine,
        stage_hook=lambda stage: (
            (_ for _ in ()).throw(RuntimeError("part_failure"))
            if stage is ControlWriteStage.PART
            else None
        ),
    )
    with pytest.raises(RuntimeError, match="part_failure"):
        part_service.stage_content_part(
            part,
            caller_user_id=environment.user_id,
            request_body_sha256=raw_hash,
            path_job_id=job_id,
        )
    assert _row_counts(environment.engine) == (1, 1, 0)

    environment.service.stage_content_part(
        part,
        caller_user_id=environment.user_id,
        request_body_sha256=raw_hash,
        path_job_id=job_id,
    )
    commit = _commit_payload(begin, job_id, part)
    commit_service = BaselineControlPlaneService(
        environment.engine,
        stage_hook=lambda stage: (
            (_ for _ in ()).throw(RuntimeError("commit_failure"))
            if stage is ControlWriteStage.COMMIT
            else None
        ),
    )
    with pytest.raises(RuntimeError, match="commit_failure"):
        commit_service.commit_snapshot(
            commit,
            caller_user_id=environment.user_id,
            path_job_id=job_id,
        )
    with environment.engine.connect() as connection:
        assert (
            connection.execute(select(snapshot_staging.c.status)).scalar_one() == "open"
        )
        assert connection.execute(select(control_job.c.state)).scalar_one() == "queued"
        assert (
            connection.execute(
                select(func.count()).select_from(snapshot_continuation_job)
            ).scalar_one()
            == 0
        )


def test_continuation_creation_failure_rolls_back_seal_and_job(
    environment: ControlEnvironment,
) -> None:
    begin, content = _begin_payload(environment)
    accepted = environment.service.begin_snapshot(
        begin, caller_user_id=environment.user_id
    )
    job_id = str(accepted["job_id"])
    part = _part_payload(begin, job_id, content)
    environment.service.stage_content_part(
        part,
        caller_user_id=environment.user_id,
        request_body_sha256=hashlib.sha256(canonicalize(part)).hexdigest(),
        path_job_id=job_id,
    )
    service = BaselineControlPlaneService(
        environment.engine,
        stage_hook=lambda stage: (
            (_ for _ in ()).throw(RuntimeError("continuation_failure"))
            if stage is ControlWriteStage.CONTINUATION
            else None
        ),
    )
    with pytest.raises(RuntimeError, match="continuation_failure"):
        service.commit_snapshot(
            _commit_payload(begin, job_id, part),
            caller_user_id=environment.user_id,
            path_job_id=job_id,
        )
    with environment.engine.connect() as connection:
        assert (
            connection.execute(select(snapshot_staging.c.status)).scalar_one() == "open"
        )
        assert connection.execute(select(control_job.c.state)).scalar_one() == "queued"
        assert (
            connection.execute(
                select(func.count()).select_from(snapshot_continuation_job)
            ).scalar_one()
            == 0
        )


def test_conflicting_sealed_continuation_intent_fails_without_reinterpreting_staging(
    environment: ControlEnvironment,
) -> None:
    begin, content = _begin_payload(environment)
    accepted = environment.service.begin_snapshot(
        begin, caller_user_id=environment.user_id
    )
    job_id = str(accepted["job_id"])
    part = _part_payload(begin, job_id, content)
    environment.service.stage_content_part(
        part,
        caller_user_id=environment.user_id,
        request_body_sha256=hashlib.sha256(canonicalize(part)).hexdigest(),
        path_job_id=job_id,
    )
    commit = _commit_payload(begin, job_id, part)
    now = datetime.now(timezone.utc)
    with environment.engine.begin() as connection:
        staging = connection.execute(select(snapshot_staging)).mappings().one()
        staging_job = connection.execute(select(control_job)).mappings().one()
        connection.execute(
            snapshot_continuation_job.insert().values(
                continuation_job_id=str(uuid4()),
                group_id=environment.group_id,
                staging_id=staging["staging_id"],
                request_id=str(uuid4()),
                created_by_user_id=environment.user_id,
                contract_version="baseline-snapshot-continuation.v1",
                idempotency_key=staging_job["idempotency_key"],
                sealed_intent_hash="0" * 64,
                snapshot_id=begin["snapshot"]["snapshot_id"],
                canonical_manifest_hash=begin["snapshot"]["canonical_manifest_hash"],
                content_manifest_hash=commit["content_manifest_hash"],
                repository_set_hash="1" * 64,
                expected_repository_count=1,
                expected_file_count=2,
                expected_supported_file_count=1,
                expected_supported_content_bytes=len(content.encode("utf-8")),
                expected_part_count=1,
                state="queued",
                attempt_count=0,
                lease_token=None,
                lease_expires_at=None,
                error_code=None,
                error_fingerprint=None,
                created_at=now,
                updated_at=now,
                finished_at=None,
            )
        )
    with pytest.raises(ControlPlaneError, match="continuation_conflict"):
        environment.service.commit_snapshot(
            commit,
            caller_user_id=environment.user_id,
            path_job_id=job_id,
        )
    with environment.engine.connect() as connection:
        assert (
            connection.execute(select(snapshot_staging.c.status)).scalar_one() == "open"
        )
        assert connection.execute(select(control_job.c.state)).scalar_one() == "queued"


def test_expiry_and_lease_recovery_are_durable(environment: ControlEnvironment) -> None:
    now = datetime(2026, 8, 16, tzinfo=timezone.utc)
    clock_value = [now]
    service = BaselineControlPlaneService(
        environment.engine,
        clock=lambda: clock_value[0],
    )
    begin, _content = _begin_payload(environment)
    accepted = service.begin_snapshot(begin, caller_user_id=environment.user_id)
    job_id = str(accepted["job_id"])

    first_lease = service.acquire_job_lease(
        caller_user_id=environment.user_id,
        group_id=environment.group_id,
        job_id=job_id,
    )
    with pytest.raises(ControlPlaneError, match="job_lease_unavailable"):
        service.acquire_job_lease(
            caller_user_id=environment.user_id,
            group_id=environment.group_id,
            job_id=job_id,
        )
    clock_value[0] += timedelta(minutes=6)
    second_lease = service.acquire_job_lease(
        caller_user_id=environment.user_id,
        group_id=environment.group_id,
        job_id=job_id,
    )
    assert first_lease.lease_token != second_lease.lease_token
    assert second_lease.attempt_count == 2

    clock_value[0] = now + timedelta(hours=25)
    assert service.expire_staging_sessions() == 1
    status = service.job_status(
        _status_payload(environment, job_id),
        caller_user_id=environment.user_id,
    )
    assert status["state"] == "terminal_failed"
    assert status["error_code"] == "staging_expired"
    assert status["staging"]["state"] == "expired"


def test_concurrent_identical_begin_and_part_replays_do_not_duplicate(
    environment: ControlEnvironment,
) -> None:
    begin, content = _begin_payload(environment)
    barrier = threading.Barrier(2)

    def submit_begin():
        barrier.wait()
        return environment.service.begin_snapshot(
            begin, caller_user_id=environment.user_id
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: submit_begin(), range(2)))
    assert len({str(item["job_id"]) for item in results}) == 1
    assert _row_counts(environment.engine) == (1, 1, 0)

    job_id = str(results[0]["job_id"])
    part = _part_payload(begin, job_id, content)
    body_hash = hashlib.sha256(canonicalize(part)).hexdigest()
    barrier = threading.Barrier(2)

    def submit_part():
        barrier.wait()
        return environment.service.stage_content_part(
            part,
            caller_user_id=environment.user_id,
            request_body_sha256=body_hash,
            path_job_id=job_id,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: submit_part(), range(2)))
    assert _row_counts(environment.engine) == (1, 1, 1)
    assert sorted(bool(item["replayed"]) for item in results) == [False, True]


def test_status_errors_and_logs_never_expose_content_or_query(
    environment: ControlEnvironment,
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw_source = "PRIVATE-SOURCE-CONTENT-ζ"
    raw_query = "PRIVATE-RAW-DIFF-SHOULD-NEVER-APPEAR"
    begin, _content = _begin_payload(environment)
    snapshot, _ = _snapshot(
        environment.group_id,
        environment.source_document_id,
        environment.changed_repository_id,
        environment.sibling_repository_id,
        content=raw_source,
    )
    begin["snapshot"] = snapshot
    accepted = environment.service.begin_snapshot(
        begin, caller_user_id=environment.user_id
    )
    job_id = str(accepted["job_id"])
    part = _part_payload(begin, job_id, raw_source)
    environment.service.stage_content_part(
        part,
        caller_user_id=environment.user_id,
        request_body_sha256=hashlib.sha256(canonicalize(part)).hexdigest(),
        path_job_id=job_id,
    )

    error = ControlPlaneError("content_hash_mismatch", status_code=409).to_dict(
        str(uuid4())
    )
    status = environment.service.job_status(
        _status_payload(environment, job_id),
        caller_user_id=environment.user_id,
    )
    exposed = json.dumps({"error": error, "status": status}) + caplog.text
    assert raw_source not in exposed
    assert raw_query not in exposed

    with environment.engine.connect() as connection:
        job_row = connection.execute(select(control_job)).mappings().one()
        staging_row = connection.execute(select(snapshot_staging)).mappings().one()
    persisted_status = json.dumps(
        {**dict(job_row), **dict(staging_row)}, default=str, sort_keys=True
    )
    assert raw_source not in persisted_status
    assert raw_query not in persisted_status


def test_authenticated_post_api_contract_transport_limits_and_redaction(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from compair_core import api as api_module

    app = api_module.create_fastapi_app()
    app.dependency_overrides[api_module.get_current_user] = lambda: SimpleNamespace(
        user_id=environment.user_id,
        username="control@example.test",
        name="Control User",
    )
    monkeypatch.setattr(
        api_module,
        "_control_plane_service",
        lambda: environment.service,
    )
    secure_client = TestClient(app, base_url="https://core.example.test")
    begin, content = _begin_payload(environment)

    response = secure_client.post(
        "/baseline/control/v1/snapshots",
        content=canonicalize(begin),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 202
    accepted = response.json()
    assert accepted["message_type"] == "job_accepted"
    assert response.headers["cache-control"] == "no-store"

    job_id = accepted["job_id"]
    part = _part_payload(begin, job_id, content)
    response = secure_client.post(
        f"/baseline/control/v1/snapshots/{job_id}/parts",
        content=canonicalize(part),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200
    assert response.json()["staging"]["received_parts"] == 1

    commit = _commit_payload(begin, job_id, part)
    response = secure_client.post(
        f"/baseline/control/v1/snapshots/{job_id}/commit",
        json=commit,
    )
    assert response.status_code == 200
    assert response.json()["state"] == "succeeded"
    assert response.json()["result"]["corpus_eligible"] is False

    response = secure_client.post(
        "/baseline/control/v1/jobs/status",
        json=_status_payload(environment, job_id),
    )
    assert response.status_code == 200
    assert response.json()["state"] == "succeeded"
    assert content not in response.text

    response = secure_client.post(
        "/baseline/control/v1/continuations/status",
        json=_continuation_status_payload(
            environment,
            staging_job_id=job_id,
        ),
    )
    assert response.status_code == 200
    continuation_status = response.json()
    assert continuation_status["schema_version"] == (
        "baseline-snapshot-continuation.v1"
    )
    assert continuation_status["state"] == "queued"
    assert continuation_status["staging_job_id"] == job_id
    assert continuation_status["job_id"] != job_id
    assert continuation_status["result"]["corpus_eligible"] is False
    assert content not in response.text

    capabilities_request = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "message_type": "capabilities_request",
        "request_id": str(uuid4()),
        "group_id": environment.group_id,
    }
    response = secure_client.post(
        "/baseline/control/v1/capabilities",
        json=capabilities_request,
    )
    assert response.status_code == 200
    assert response.json()["protocol_sha256"] == PROTOCOL_SHA256
    assert response.json()["operations"]["corpus_ingestion"] == "unavailable"

    assert secure_client.get("/baseline/control/v1/snapshots").status_code == 405
    assert secure_client.get("/baseline/control/v1/jobs/status").status_code == 405
    assert (
        secure_client.get("/baseline/control/v1/continuations/status").status_code
        == 405
    )

    insecure_client = TestClient(app, base_url="http://core.example.test")
    insecure_begin, _content = _begin_payload(
        environment, idempotency_key="opaque-client-intent-token-000002"
    )
    response = insecure_client.post(
        "/baseline/control/v1/snapshots",
        json=insecure_begin,
    )
    assert response.status_code == 503
    assert response.json()["code"] == "transport_unavailable"
    response = insecure_client.post(
        "/baseline/control/v1/capabilities",
        json=capabilities_request,
    )
    assert response.status_code == 503
    assert response.json()["code"] == "transport_unavailable"

    oversized_begin, _content = _begin_payload(
        environment, idempotency_key="opaque-client-intent-token-000003"
    )
    response = secure_client.post(
        "/baseline/control/v1/snapshots",
        content=canonicalize(oversized_begin),
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(32_000_001),
        },
    )
    assert response.status_code == 413
    assert response.json()["code"] == "limit_exceeded"

    private_source = "DO-NOT-LOG-THIS-SOURCE"
    raw_duplicate = (
        b'{"protocol_version":"baseline-control-plane.v1",'
        b'"protocol_version":"baseline-control-plane.v1",'
        b'"message_type":"snapshot_begin","private":"' + private_source.encode() + b'"}'
    )
    response = secure_client.post(
        "/baseline/control/v1/snapshots",
        content=raw_duplicate,
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_contract"
    assert private_source not in response.text
    assert private_source not in caplog.text

    def reject_unauthenticated():
        raise HTTPException(status_code=401, detail="authentication required")

    app.dependency_overrides[api_module.get_current_user] = reject_unauthenticated
    response = secure_client.post(
        "/baseline/control/v1/capabilities",
        json=capabilities_request,
    )
    assert response.status_code == 401


def test_authenticated_repository_admin_endpoint_contract(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from compair_core import api as api_module

    member_id = _add_group_member(
        environment.engine,
        group_id=environment.group_id,
    )
    app = api_module.create_fastapi_app()
    current_user = [member_id]
    app.dependency_overrides[api_module.get_current_user] = lambda: SimpleNamespace(
        user_id=current_user[0],
        username="control@example.test",
        name="Control User",
    )
    monkeypatch.setattr(
        api_module,
        "_control_plane_service",
        lambda: environment.service,
    )
    payload = _registration_create_payload(
        environment,
        repository_uid="endpoint-upstream-repository-uid",
    )
    with TestClient(app, base_url="https://core.example.test") as client:
        denied = client.post(
            "/baseline/control/admin/v1/repositories/register",
            content=canonicalize(payload),
            headers={"Content-Type": "application/json"},
        )
        assert denied.status_code == 404

        current_user[0] = environment.user_id
        created = client.post(
            "/baseline/control/admin/v1/repositories/register",
            content=canonicalize(payload),
            headers={"Content-Type": "application/json"},
        )
        assert created.status_code == 201
        response = created.json()
        assert response["schema_version"] == (
            "baseline-repository-registration-admin.v1"
        )
        assert response["state"] == "active"
        serialized = json.dumps(response, sort_keys=True)
        assert "endpoint-upstream-repository-uid" not in serialized
        assert "git.example.test" not in serialized

        disabled = client.post(
            "/baseline/control/admin/v1/repositories/state",
            json=_registration_state_payload(
                environment,
                str(response["registration_id"]),
                active=False,
            ),
        )
        assert disabled.status_code == 200
        assert disabled.json()["state"] == "disabled"


def test_all_control_endpoints_reject_duplicate_keys_before_service_authorization(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from compair_core import api as api_module

    app = api_module.create_fastapi_app()
    app.dependency_overrides[api_module.get_current_user] = lambda: SimpleNamespace(
        user_id=environment.user_id,
        username="control@example.test",
        name="Control User",
    )

    def service_must_not_be_reached():
        pytest.fail("duplicate-key request reached domain authorization/service")

    monkeypatch.setattr(
        api_module,
        "_control_plane_service",
        service_must_not_be_reached,
    )
    secret = b"DUPLICATE-BODY-MUST-STAY-PRIVATE"
    duplicate_requests = (
        (
            "/baseline/control/admin/v1/repositories/register",
            b'{"group_id":"group-a","group_id":"group-b","private":"' + secret + b'"}',
        ),
        (
            "/baseline/control/admin/v1/repositories/state",
            b'{"registration_id":"one","registration_id":"two","private":"'
            + secret
            + b'"}',
        ),
        (
            "/baseline/control/v1/snapshots",
            b'{"group_id":"group-a","group_id":"group-b","private":"' + secret + b'"}',
        ),
        (
            "/baseline/control/v1/snapshots",
            b'{"snapshot":{"changed_repository":{"repository_id":"one",'
            b'"repository_id":"two"}},"private":"' + secret + b'"}',
        ),
        (
            "/baseline/control/v1/snapshots/job_duplicate/parts",
            b'{"content_items":[{"content_utf8":"'
            + secret
            + b'","content_utf8":"replacement"}]}',
        ),
        (
            "/baseline/control/v1/snapshots/job_duplicate/commit",
            b'{"protocol_sha256":"'
            + PROTOCOL_SHA256.encode()
            + b'","protocol_sha256":"'
            + (b"0" * 64)
            + b'","private":"'
            + secret
            + b'"}',
        ),
        (
            "/baseline/control/v1/jobs/status",
            b'{"group_id":"group-a","group_id":"group-b","private":"' + secret + b'"}',
        ),
        (
            "/baseline/control/v1/index-builds",
            b'{"group_id":"group-a","group_id":"group-b","private":"' + secret + b'"}',
        ),
        (
            "/baseline/control/v1/continuations/status",
            b'{"continuation_job_id":"one","continuation_job_id":"two",'
            b'"private":"' + secret + b'"}',
        ),
        (
            "/baseline/control/v1/capabilities",
            b'{"protocol_version":"baseline-control-plane.v1",'
            b'"protocol_version":"baseline-control-plane.v1","private":"'
            + secret
            + b'"}',
        ),
    )
    before = _row_counts(environment.engine)

    with TestClient(app, base_url="https://core.example.test") as client:
        for path, body in duplicate_requests:
            response = client.post(
                path,
                content=body,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            assert response.status_code == 400
            assert response.json()["code"] == "invalid_contract"
            assert secret.decode() not in response.text

    assert _row_counts(environment.engine) == before
    assert secret.decode() not in caplog.text


@pytest.mark.parametrize(
    ("body", "content_type"),
    (
        (b'{"private":"MALFORMED-UTF8-PRIVATE-\xff"}', "application/json"),
        (b'{"number":NaN,"private":"NONFINITE-PRIVATE"}', "application/json"),
        (
            b'{"number":Infinity,"private":"NONFINITE-PRIVATE"}',
            "application/json",
        ),
        (
            b'{"number":-Infinity,"private":"NONFINITE-PRIVATE"}',
            "application/json",
        ),
        (b'{"private":"WRONG-TYPE-PRIVATE"}', "text/plain"),
        (
            b'{"private":"WRONG-CHARSET-PRIVATE"}',
            "application/json; charset=iso-8859-1",
        ),
    ),
)
def test_control_api_strict_utf8_content_type_and_finite_numbers(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    body: bytes,
    content_type: str,
) -> None:
    from compair_core import api as api_module

    app = api_module.create_fastapi_app()
    app.dependency_overrides[api_module.get_current_user] = lambda: SimpleNamespace(
        user_id=environment.user_id,
        username="control@example.test",
        name="Control User",
    )

    def service_must_not_be_reached():
        pytest.fail("malformed request reached control-plane service")

    monkeypatch.setattr(
        api_module,
        "_control_plane_service",
        service_must_not_be_reached,
    )
    before = _row_counts(environment.engine)

    with TestClient(app, base_url="https://core.example.test") as client:
        response = client.post(
            "/baseline/control/v1/snapshots",
            content=body,
            headers={"Content-Type": content_type},
        )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_contract"
    assert _row_counts(environment.engine) == before
    for private_marker in (
        "MALFORMED-UTF8-PRIVATE",
        "NONFINITE-PRIVATE",
        "WRONG-TYPE-PRIVATE",
        "WRONG-CHARSET-PRIVATE",
    ):
        assert private_marker not in response.text
        assert private_marker not in caplog.text


def test_untrusted_peer_cannot_spoof_loopback_or_https_with_headers(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from compair_core import api as api_module

    app = api_module.create_fastapi_app()
    app.dependency_overrides[api_module.get_current_user] = lambda: SimpleNamespace(
        user_id=environment.user_id,
        username="control@example.test",
        name="Control User",
    )
    monkeypatch.setattr(
        api_module,
        "get_settings_dependency",
        lambda: SimpleNamespace(
            baseline_control_plane_allow_insecure_loopback=True,
            baseline_control_plane_trusted_proxy_allowlist="10.40.0.0/16",
        ),
    )

    def service_must_not_be_reached():
        pytest.fail("spoofed transport reached control-plane service")

    monkeypatch.setattr(
        api_module,
        "_control_plane_service",
        service_must_not_be_reached,
    )
    begin, _content = _begin_payload(environment)

    with TestClient(app, base_url="http://core.example.test") as client:
        response = client.post(
            "/baseline/control/v1/snapshots",
            content=canonicalize(begin),
            headers={
                "Content-Type": "application/json",
                "Host": "localhost",
                "Forwarded": "for=127.0.0.1;proto=https;host=localhost",
                "X-Forwarded-For": "127.0.0.1",
                "X-Forwarded-Host": "localhost",
                "X-Forwarded-Proto": "https",
            },
        )

    assert response.status_code == 503
    assert response.json()["code"] == "transport_unavailable"
    assert _row_counts(environment.engine) == (0, 0, 0)


def _continuation_worker(
    environment: ControlEnvironment,
    *,
    clock=None,
    stage_hook=None,
) -> BaselineContinuationWorker:
    sessions = core_db.sessionmaker(environment.engine, expire_on_commit=False)
    options = {"stage_hook": stage_hook}
    if clock is not None:
        options["clock"] = clock
    return BaselineContinuationWorker(
        environment.engine,
        CorpusIngestionService(sessions),
        **options,
    )


def _stage_worker_snapshot(
    environment: ControlEnvironment,
    *,
    content: str,
    idempotency_key: str,
) -> str:
    snapshot, supplied_content = _snapshot(
        environment.group_id,
        environment.source_document_id,
        environment.changed_repository_id,
        environment.sibling_repository_id,
        content=content,
    )
    begin = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "message_type": "snapshot_begin",
        "request_id": str(uuid4()),
        "group_id": environment.group_id,
        "idempotency_key": idempotency_key,
        "snapshot": snapshot,
    }
    accepted = environment.service.begin_snapshot(
        begin, caller_user_id=environment.user_id
    )
    staging_job_id = str(accepted["job_id"])
    part = _part_payload(begin, staging_job_id, supplied_content)
    environment.service.stage_content_part(
        part,
        caller_user_id=environment.user_id,
        request_body_sha256=hashlib.sha256(canonicalize(part)).hexdigest(),
        path_job_id=staging_job_id,
    )
    environment.service.commit_snapshot(
        _commit_payload(begin, staging_job_id, part),
        caller_user_id=environment.user_id,
        path_job_id=staging_job_id,
    )
    with environment.engine.connect() as connection:
        staging_id = connection.execute(
            select(snapshot_staging.c.staging_id).where(
                snapshot_staging.c.job_id == staging_job_id
            )
        ).scalar_one()
        return str(
            connection.execute(
                select(snapshot_continuation_job.c.continuation_job_id).where(
                    snapshot_continuation_job.c.staging_id == staging_id
                )
            ).scalar_one()
        )


def test_continuation_worker_reconstructs_publishes_and_replays_safely(
    environment: ControlEnvironment,
) -> None:
    _stage_success(environment)
    continuation_id = _single_continuation_id(environment.engine)
    first = _continuation_worker(environment).execute(
        identity=InternalContinuationWorkerIdentity.create("sqlite-worker-1"),
        group_id=environment.group_id,
        continuation_job_id=continuation_id,
    )
    environment.engine.dispose()
    replay = _continuation_worker(environment).execute(
        identity=InternalContinuationWorkerIdentity.create("sqlite-worker-2"),
        group_id=environment.group_id,
        continuation_job_id=continuation_id,
    )

    assert replay == first
    assert first.state == "succeeded"
    assert first.index_state == "incomplete"
    assert first.baseline_eligible is False
    sessions = core_db.sessionmaker(environment.engine, expire_on_commit=False)
    with sessions() as session:
        corpus = session.get(RetrievalCorpus, first.corpus_id)
        generation = session.get(RetrievalCorpusGeneration, first.generation_id)
        index_state = session.get(RetrievalIndexState, first.generation_id)
        rows = tuple(
            session.scalars(
                select(RetrievalCorpusFile)
                .where(RetrievalCorpusFile.generation_id == first.generation_id)
                .order_by(RetrievalCorpusFile.relative_path)
            )
        )
    assert corpus is not None and corpus.active_generation_id == first.generation_id
    assert generation is not None and generation.status == "active"
    assert index_state is not None and index_state.status == "incomplete"
    assert [(row.relative_path, row.file_state) for row in rows] == [
        ("src/café.py", "supported"),
        ("src/link.py", "symlink_rejected"),
    ]
    assert rows[0].content == "héllo\n"
    assert rows[1].content is None

    status = environment.service.continuation_status(
        _continuation_status_payload(
            environment,
            continuation_job_id=continuation_id,
        ),
        caller_user_id=environment.user_id,
    )
    assert status["state"] == "succeeded"
    assert status["progress"] == {"completed": 1, "total": 1}
    assert status["result"]["corpus_generation_id"] == first.generation_id
    assert status["result"]["index_state"] == "incomplete"
    assert status["result"]["baseline_eligible"] is False
    serialized = json.dumps(status, sort_keys=True)
    for forbidden in ("héllo", "content_utf8", "lease_token", "idempotency"):
        assert forbidden not in serialized
    with pytest.raises(DatabaseError), environment.engine.begin() as connection:
        connection.execute(
            update(snapshot_continuation_job)
            .where(
                snapshot_continuation_job.c.continuation_job_id == continuation_id
            )
            .values(result_manifest_hash="0" * 64)
        )


def test_continuation_worker_failure_rolls_back_publication_and_retry_resumes(
    environment: ControlEnvironment,
) -> None:
    first_job = _stage_worker_snapshot(
        environment,
        content="first benign snapshot\n",
        idempotency_key="opaque-worker-first-snapshot-0001",
    )
    first = _continuation_worker(environment).execute(
        identity=InternalContinuationWorkerIdentity.create("rollback-worker"),
        group_id=environment.group_id,
        continuation_job_id=first_job,
    )
    second_job = _stage_worker_snapshot(
        environment,
        content="second benign snapshot\n",
        idempotency_key="opaque-worker-second-snapshot-0002",
    )

    def fail_before_success(stage: ContinuationWorkerStage) -> None:
        if stage is ContinuationWorkerStage.BEFORE_SUCCESS:
            raise RuntimeError("injected_without_source_detail")

    with pytest.raises(ContinuationWorkerError) as failure:
        _continuation_worker(environment, stage_hook=fail_before_success).execute(
            identity=InternalContinuationWorkerIdentity.create("failure-worker"),
            group_id=environment.group_id,
            continuation_job_id=second_job,
        )
    assert (failure.value.code, failure.value.retryable) == (
        "corpus_ingestion_failed",
        True,
    )
    sessions = core_db.sessionmaker(environment.engine, expire_on_commit=False)
    with sessions() as session:
        corpus = (
            session.query(RetrievalCorpus)
            .filter_by(scope_key=f"group:{environment.group_id}")
            .one()
        )
    assert corpus.active_generation_id == first.generation_id
    with environment.engine.connect() as connection:
        failed = (
            connection.execute(
                select(snapshot_continuation_job).where(
                    snapshot_continuation_job.c.continuation_job_id == second_job
                )
            )
            .mappings()
            .one()
        )
    assert failed["state"] == "retryable_failed"
    assert failed["error_code"] == "corpus_ingestion_failed"
    assert failed["result_generation_id"] is None

    resumed = _continuation_worker(environment).execute(
        identity=InternalContinuationWorkerIdentity.create("resume-worker"),
        group_id=environment.group_id,
        continuation_job_id=second_job,
    )
    assert resumed.generation_id != first.generation_id
    with sessions() as session:
        corpus = session.get(RetrievalCorpus, resumed.corpus_id)
        count = (
            session.query(RetrievalCorpusGeneration)
            .filter_by(corpus_id=resumed.corpus_id)
            .count()
        )
    assert corpus is not None and corpus.active_generation_id == resumed.generation_id
    assert count == 2


def test_continuation_worker_rechecks_revocation_and_source_scope(
    environment: ControlEnvironment,
) -> None:
    _stage_success(environment)
    continuation_id = _single_continuation_id(environment.engine)

    def revoke_after_claim(stage: ContinuationWorkerStage) -> None:
        if stage is ContinuationWorkerStage.CLAIMED:
            environment.service.set_repository_registration_state(
                _registration_state_payload(
                    environment,
                    environment.sibling_repository_id,
                    active=False,
                ),
                caller_user_id=environment.user_id,
            )

    with pytest.raises(ContinuationWorkerError) as revoked:
        _continuation_worker(environment, stage_hook=revoke_after_claim).execute(
            identity=InternalContinuationWorkerIdentity.create("revocation-worker"),
            group_id=environment.group_id,
            continuation_job_id=continuation_id,
        )
    assert revoked.value.code == "repository_not_authorized"
    with environment.engine.connect() as connection:
        row = connection.execute(select(snapshot_continuation_job)).mappings().one()
        corpus_count = connection.exec_driver_sql(
            "SELECT COUNT(*) FROM retrieval_corpus"
        ).scalar_one()
    assert row["state"] == "terminal_failed"
    assert row["result_generation_id"] is None
    assert corpus_count == 0

    environment.service.set_repository_registration_state(
        _registration_state_payload(
            environment,
            environment.sibling_repository_id,
            active=True,
        ),
        caller_user_id=environment.user_id,
    )
    second_job = _stage_worker_snapshot(
        environment,
        content="source deletion snapshot\n",
        idempotency_key="opaque-source-deletion-snapshot-0003",
    )

    def delete_source_scope(stage: ContinuationWorkerStage) -> None:
        if stage is ContinuationWorkerStage.CLAIMED:
            with environment.engine.begin() as connection:
                connection.execute(
                    text(
                        "DELETE FROM document_to_group "
                        "WHERE document_id = :document_id AND group_id = :group_id"
                    ),
                    {
                        "document_id": environment.source_document_id,
                        "group_id": environment.group_id,
                    },
                )

    with pytest.raises(ContinuationWorkerError) as deleted:
        _continuation_worker(environment, stage_hook=delete_source_scope).execute(
            identity=InternalContinuationWorkerIdentity.create("source-worker"),
            group_id=environment.group_id,
            continuation_job_id=second_job,
        )
    assert deleted.value.code == "source_not_authorized"
    with environment.engine.connect() as connection:
        row = (
            connection.execute(
                select(snapshot_continuation_job).where(
                    snapshot_continuation_job.c.continuation_job_id == second_job
                )
            )
            .mappings()
            .one()
        )
    assert row["state"] == "terminal_failed"
    assert row["result_generation_id"] is None


def test_continuation_worker_lease_reclaim_and_corruption_defense(
    environment: ControlEnvironment,
) -> None:
    now = datetime(2026, 8, 16, tzinfo=timezone.utc)
    current = [now]
    advanced = [False]
    _stage_success(environment)
    continuation_id = _single_continuation_id(environment.engine)

    def expire_once(stage: ContinuationWorkerStage) -> None:
        if stage is ContinuationWorkerStage.RECONSTRUCTED and not advanced[0]:
            current[0] += timedelta(minutes=6)
            advanced[0] = True

    with pytest.raises(ContinuationWorkerError) as expired:
        _continuation_worker(
            environment,
            clock=lambda: current[0],
            stage_hook=expire_once,
        ).execute(
            identity=InternalContinuationWorkerIdentity.create("expired-worker"),
            group_id=environment.group_id,
            continuation_job_id=continuation_id,
        )
    assert expired.value.code == "job_lease_unavailable"
    sessions = core_db.sessionmaker(environment.engine, expire_on_commit=False)
    with sessions() as session:
        corpus = session.query(RetrievalCorpus).one()
        assert corpus.active_generation_id is None
    recovered = _continuation_worker(
        environment,
        clock=lambda: current[0],
    ).execute(
        identity=InternalContinuationWorkerIdentity.create("reclaim-worker"),
        group_id=environment.group_id,
        continuation_job_id=continuation_id,
    )
    assert recovered.attempt_count == 2

    second_job = _stage_worker_snapshot(
        environment,
        content="corruption defense snapshot\n",
        idempotency_key="opaque-corruption-defense-snapshot-04",
    )
    with environment.engine.connect() as connection:
        staging_id = connection.execute(
            select(snapshot_continuation_job.c.staging_id).where(
                snapshot_continuation_job.c.continuation_job_id == second_job
            )
        ).scalar_one()
        stored = dict(
            connection.execute(
                select(snapshot_content_part).where(
                    snapshot_content_part.c.staging_id == staging_id
                )
            )
            .mappings()
            .one()
        )
    duplicate = dict(stored)
    duplicate["part_id"] = str(uuid4())
    with pytest.raises(DatabaseError), environment.engine.begin() as connection:
        connection.execute(snapshot_content_part.insert().values(**duplicate))
    with pytest.raises(DatabaseError), environment.engine.begin() as connection:
        connection.execute(
            snapshot_content_part.delete().where(
                snapshot_content_part.c.part_id == stored["part_id"]
            )
        )
    with environment.engine.begin() as connection:
        connection.exec_driver_sql("DROP TRIGGER trg_bl_ctl_part_immutable")
        connection.execute(
            update(snapshot_content_part)
            .where(snapshot_content_part.c.part_id == stored["part_id"])
            .values(canonical_content_items_json='[{"content_utf8":"drift"}]')
        )
    with pytest.raises(ContinuationWorkerError) as drift:
        _continuation_worker(environment).execute(
            identity=InternalContinuationWorkerIdentity.create("drift-worker"),
            group_id=environment.group_id,
            continuation_job_id=second_job,
        )
    assert drift.value.code == "sealed_snapshot_invalid"
    status = environment.service.continuation_status(
        _continuation_status_payload(
            environment,
            continuation_job_id=second_job,
        ),
        caller_user_id=environment.user_id,
    )
    assert status["state"] == "terminal_failed"
    assert status["error_code"] == "sealed_snapshot_invalid"
    assert "drift" not in json.dumps(status)


@pytest.mark.parametrize("drift", ("missing", "out_of_order", "count", "utf8"))
def test_continuation_worker_rejects_part_shape_and_encoding_drift(
    environment: ControlEnvironment,
    drift: str,
) -> None:
    _stage_success(environment)
    continuation_id = _single_continuation_id(environment.engine)
    with environment.engine.begin() as connection:
        row = dict(connection.execute(select(snapshot_content_part)).mappings().one())
        if drift == "missing":
            connection.exec_driver_sql("DROP TRIGGER trg_bl_ctl_sealed_part_delete")
            connection.execute(
                snapshot_content_part.delete().where(
                    snapshot_content_part.c.part_id == row["part_id"]
                )
            )
        else:
            connection.exec_driver_sql("DROP TRIGGER trg_bl_ctl_part_immutable")
            if drift == "out_of_order":
                connection.execute(
                    update(snapshot_content_part)
                    .where(snapshot_content_part.c.part_id == row["part_id"])
                    .values(part_ordinal=2)
                )
            elif drift == "count":
                connection.execute(
                    update(snapshot_content_part)
                    .where(snapshot_content_part.c.part_id == row["part_id"])
                    .values(item_count=2)
                )
            else:
                connection.execute(
                    update(snapshot_content_part)
                    .where(snapshot_content_part.c.part_id == row["part_id"])
                    .values(canonical_content_items_json=b"\xff")
                )
    with pytest.raises(ContinuationWorkerError) as failure:
        _continuation_worker(environment).execute(
            identity=InternalContinuationWorkerIdentity.create(f"drift-{drift}"),
            group_id=environment.group_id,
            continuation_job_id=continuation_id,
        )
    assert failure.value.code in {
        "sealed_snapshot_incompatible",
        "sealed_snapshot_invalid",
    }
    with environment.engine.connect() as connection:
        assert (
            connection.exec_driver_sql(
                "SELECT COUNT(*) FROM retrieval_corpus"
            ).scalar_one()
            == 0
        )
        continuation = (
            connection.execute(select(snapshot_continuation_job)).mappings().one()
        )
    assert continuation["result_generation_id"] is None


def test_continuation_worker_cancellation_is_lease_guarded_and_preserves_audit(
    environment: ControlEnvironment,
) -> None:
    _stage_success(environment)
    continuation_id = _single_continuation_id(environment.engine)
    worker = _continuation_worker(environment)
    identity = InternalContinuationWorkerIdentity.create("cancellation-worker")
    receipt = worker.claim(
        identity=identity,
        group_id=environment.group_id,
        continuation_job_id=continuation_id,
    )
    with pytest.raises(ContinuationWorkerError):
        worker.cancel(
            identity=identity,
            group_id=environment.group_id,
            continuation_job_id=continuation_id,
            lease_token="wrong-token",
        )
    worker.cancel(
        identity=identity,
        group_id=environment.group_id,
        continuation_job_id=continuation_id,
        lease_token=receipt.lease_token,
    )
    with environment.engine.connect() as connection:
        continuation = (
            connection.execute(select(snapshot_continuation_job)).mappings().one()
        )
        assert (
            connection.execute(
                select(func.count()).select_from(snapshot_staging)
            ).scalar_one()
            == 1
        )
        assert (
            connection.execute(
                select(func.count()).select_from(snapshot_content_part)
            ).scalar_one()
            == 1
        )
    assert continuation["state"] == "cancelled"
    assert continuation["lease_token"] is None
    assert continuation["error_code"] == "worker_cancelled"


def test_concurrent_continuation_workers_publish_one_generation(
    environment: ControlEnvironment,
) -> None:
    _stage_success(environment)
    continuation_id = _single_continuation_id(environment.engine)
    barrier = threading.Barrier(2)

    def execute(ordinal: int):
        barrier.wait()
        try:
            return _continuation_worker(environment).execute(
                identity=InternalContinuationWorkerIdentity.create(
                    f"concurrent-worker-{ordinal}"
                ),
                group_id=environment.group_id,
                continuation_job_id=continuation_id,
            )
        except ContinuationWorkerError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(execute, range(2)))
    assert any(not isinstance(item, str) for item in outcomes)
    with environment.engine.connect() as connection:
        continuation = (
            connection.execute(select(snapshot_continuation_job)).mappings().one()
        )
        generation_count = connection.exec_driver_sql(
            "SELECT COUNT(*) FROM retrieval_corpus_generation"
        ).scalar_one()
    assert continuation["state"] == "succeeded"
    assert continuation["attempt_count"] == 1
    assert generation_count == 1
