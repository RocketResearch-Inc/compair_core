from __future__ import annotations

import base64
import copy
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
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
    _payload as _index_payload,
)
from test_baseline_index_continuation import (
    _publish_corpus,
)
from test_baseline_index_continuation import (
    _service as _index_service,
)

from compair_core.baseline_control_plane_schema import (
    baseline_run_job,
    baseline_run_payload,
    compatible_index_job,
    control_job,
    repository_approval,
)
from compair_core.baseline_evidence_schema import (
    baseline_evidence_artifact,
    baseline_retrieval_run,
    baseline_selected_evidence,
)
from compair_core.compair.retrieval.control_plane_v2 import (
    PROTOCOL_V2_SHA256,
    PROTOCOL_V2_VERSION,
    V2ControlPlaneError,
    parse_run_submission,
)
from compair_core.compair.retrieval.corpus import (
    RetrievalBaselineIndexBuild,
)
from compair_core.compair.retrieval.index_continuation import (
    InternalIndexWorkerIdentity,
)
from compair_core.compair.retrieval.persistent import published_index_fingerprint
from compair_core.compair.retrieval.run_jobs import (
    BaselineRunJobError,
    BaselineRunJobService,
    BaselineRunKeyring,
    RunSubmissionStage,
    keyring_from_settings,
)

SCHEMA = json.loads(
    (
        Path(__file__).parents[1] / "protocol/baseline-control-plane.v2.schema.json"
    ).read_text(encoding="utf-8")
)
RAW_QUERY = (
    "diff --git a/src/widget.py b/src/widget.py\n"
    "--- a/src/widget.py\n"
    "+++ b/src/widget.py\n"
    "@@ -1 +1 @@\n"
    "-return 'old synthetic value'\n"
    "+return 'new synthetic value'\n"
)


@pytest.fixture
def environment(request: pytest.FixtureRequest):
    return request.getfixturevalue("_environment_fixture")


def _keyring(*, active: str = "key-2026-08", include_old: bool = False):
    entries = []
    if include_old:
        entries.append(
            {
                "key_id": "key-2026-07",
                "key_base64": base64.b64encode(b"o" * 32).decode(),
            }
        )
    entries.append(
        {
            "key_id": "key-2026-08",
            "key_base64": base64.b64encode(b"n" * 32).decode(),
        }
    )
    if active == "key-2026-09":
        entries.append(
            {
                "key_id": "key-2026-09",
                "key_base64": base64.b64encode(b"r" * 32).decode(),
            }
        )
    return BaselineRunKeyring.from_json(
        json.dumps(
            {
                "version": "baseline-run-keyring.v1",
                "active_key_id": active,
                "keys": entries,
            }
        )
    )


def _publish_index(environment: ControlEnvironment):
    _publish_corpus(environment)
    service = _index_service(environment)
    accepted = service.submit(
        _index_payload(environment), caller_user_id=environment.user_id
    )
    outcome = service.execute(
        identity=InternalIndexWorkerIdentity.create("run-job-index-worker"),
        group_id=environment.group_id,
        job_id=str(accepted["job_id"]),
    )
    with service.sessions() as session:
        build = session.get(RetrievalBaselineIndexBuild, outcome.index_id)
        extension = (
            session.execute(
                select(compatible_index_job).where(
                    compatible_index_job.c.job_id == outcome.job_id
                )
            )
            .mappings()
            .one()
        )
        assert build is not None
        return outcome, dict(extension), published_index_fingerprint(build)


def _run_payload(
    environment: ControlEnvironment,
    *,
    raw_query: str = RAW_QUERY,
    idempotency_key: str = "opaque-baseline-run-intent-000000000001",
):
    outcome, extension, index_fingerprint = _publish_index(environment)
    encoded = raw_query.encode("utf-8")
    return {
        "protocol_version": PROTOCOL_V2_VERSION,
        "protocol_sha256": PROTOCOL_V2_SHA256,
        "message_type": "run_submit",
        "request_id": str(uuid4()),
        "group_id": environment.group_id,
        "idempotency_key": idempotency_key,
        "source_document_id": environment.source_document_id,
        "changed_repository_registration_id": environment.changed_repository_id,
        "index_publication": {
            "index_publication_id": outcome.index_id,
            "corpus_generation_id": outcome.generation_id,
            "corpus_manifest_hash": extension["corpus_manifest_hash"],
            "index_format_version": extension["index_format_version"],
            "tokenizer_version": extension["tokenizer_version"],
            "retrieval_config_fingerprint": extension["retrieval_config_fingerprint"],
            "embedding_fingerprint": extension["embedding_fingerprint"],
            "index_fingerprint": index_fingerprint,
        },
        "retrieval_query": {
            "representation": "raw_git_diff_v1",
            "origin": "explicit",
            "encoding": "utf-8",
            "base_revision": "1" * 40,
            "head_revision": "2" * 40,
            "byte_size": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "text": raw_query,
        },
    }


def _service(environment: ControlEnvironment, **kwargs):
    return BaselineRunJobService(environment.engine, _keyring(), **kwargs)


def _rows(environment: ControlEnvironment, job_id: str):
    with environment.engine.connect() as connection:
        job = (
            connection.execute(
                select(baseline_run_job).where(baseline_run_job.c.job_id == job_id)
            )
            .mappings()
            .one()
        )
        payload = (
            connection.execute(
                select(baseline_run_payload).where(
                    baseline_run_payload.c.job_id == job_id
                )
            )
            .mappings()
            .one()
        )
    return dict(job), dict(payload)


def test_encrypted_round_trip_unique_nonces_restart_status_and_zero_effects(
    environment: ControlEnvironment,
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = _run_payload(environment)
    submission = parse_run_submission(payload)
    service = _service(environment)
    first = service.submit(submission, caller_user_id=environment.user_id)
    replay = service.submit(submission, caller_user_id=environment.user_id)
    assert first["job_id"] == replay["job_id"]
    assert replay["replayed"] is True
    job, protected = _rows(environment, str(first["job_id"]))
    nonce = protected["nonce"]
    ciphertext = protected["ciphertext"]
    opened = service.cipher.decrypt(job=job, payload=protected)
    assert opened.retrieval_query == RAW_QUERY
    assert len(opened.parent_processing_secret) == 32

    second_payload = copy.deepcopy(payload)
    second_payload["request_id"] = str(uuid4())
    second_payload["idempotency_key"] = "opaque-baseline-run-intent-000000000002"
    second = service.submit(
        parse_run_submission(second_payload), caller_user_id=environment.user_id
    )
    _second_job, second_protected = _rows(environment, str(second["job_id"]))
    assert second_protected["nonce"] != nonce
    assert second_protected["ciphertext"] != ciphertext

    restarted = BaselineRunJobService(environment.engine, _keyring())
    status = restarted.read_status(
        request_id=str(uuid4()),
        group_id=environment.group_id,
        job_id=str(first["job_id"]),
        caller_user_id=environment.user_id,
    )
    _validate_contract(status, SCHEMA)
    assert status["state"] == "queued"
    assert status["effects"] == {
        "evidence_count": 0,
        "reference_count": 0,
        "feedback_count": 0,
        "generation_invoked": False,
        "notification_outbox_count": 0,
        "persisted_run_id": None,
    }
    serialized = json.dumps(status, sort_keys=True)
    for secret in (
        RAW_QUERY,
        payload["idempotency_key"],
        protected["key_id"],
        nonce.hex(),
        ciphertext.hex(),
    ):
        assert str(secret) not in serialized
        assert str(secret) not in caplog.text
    assert RAW_QUERY not in repr(submission)
    assert RAW_QUERY not in repr(opened)
    assert protected["key_id"] not in repr(service.keyring)

    with environment.engine.connect() as connection:
        ordinary = (
            connection.execute(
                select(baseline_run_job).where(
                    baseline_run_job.c.job_id == first["job_id"]
                )
            )
            .mappings()
            .one()
        )
        assert RAW_QUERY not in repr(dict(ordinary))
        assert payload["idempotency_key"] not in repr(dict(ordinary))
        assert (
            connection.execute(text("SELECT COUNT(*) FROM reference")).scalar_one() == 0
        )
        assert (
            connection.execute(text("SELECT COUNT(*) FROM feedback")).scalar_one() == 0
        )
        assert (
            connection.execute(
                select(func.count())
                .select_from(control_job)
                .where(control_job.c.operation == "baseline_run")
            ).scalar_one()
            == 0
        )
        for table in (
            baseline_retrieval_run,
            baseline_evidence_artifact,
            baseline_selected_evidence,
        ):
            assert (
                connection.execute(select(func.count()).select_from(table)).scalar_one()
                == 0
            )
        assert (
            connection.execute(
                text("SELECT COUNT(*) FROM baseline_notification_outbox")
            ).scalar_one()
            == 0
        )


def test_aad_ciphertext_wrong_key_unknown_key_and_rotation_fail_closed(
    environment: ControlEnvironment,
) -> None:
    submission = parse_run_submission(_run_payload(environment))
    service = _service(environment)
    accepted = service.submit(submission, caller_user_id=environment.user_id)
    job, protected = _rows(environment, str(accepted["job_id"]))

    aad_tamper = dict(job)
    aad_tamper["query_sha256"] = "f" * 64
    with pytest.raises(BaselineRunJobError, match="run_payload_authentication_failed"):
        service.cipher.decrypt(job=aad_tamper, payload=protected)

    ciphertext_tamper = dict(protected)
    mutated = bytearray(ciphertext_tamper["ciphertext"])
    mutated[-1] ^= 1
    ciphertext_tamper["ciphertext"] = bytes(mutated)
    with pytest.raises(BaselineRunJobError, match="run_payload_authentication_failed"):
        service.cipher.decrypt(job=job, payload=ciphertext_tamper)

    wrong = BaselineRunKeyring.from_json(
        json.dumps(
            {
                "version": "baseline-run-keyring.v1",
                "active_key_id": "key-2026-08",
                "keys": [
                    {
                        "key_id": "key-2026-08",
                        "key_base64": base64.b64encode(b"w" * 32).decode(),
                    }
                ],
            }
        )
    )
    with pytest.raises(BaselineRunJobError, match="run_payload_authentication_failed"):
        BaselineRunJobService(environment.engine, wrong).cipher.decrypt(
            job=job, payload=protected
        )

    unknown = dict(protected)
    unknown["key_id"] = "retired-key"
    with pytest.raises(BaselineRunJobError, match="run_payload_key_unavailable"):
        service.cipher.decrypt(job=job, payload=unknown)

    rotated = BaselineRunKeyring.from_json(
        json.dumps(
            {
                "version": "baseline-run-keyring.v1",
                "active_key_id": "key-2026-09",
                "keys": [
                    {
                        "key_id": "key-2026-08",
                        "key_base64": base64.b64encode(b"n" * 32).decode(),
                    },
                    {
                        "key_id": "key-2026-09",
                        "key_base64": base64.b64encode(b"r" * 32).decode(),
                    },
                ],
            }
        )
    )
    assert (
        BaselineRunJobService(environment.engine, rotated)
        .cipher.decrypt(job=job, payload=protected)
        .retrieval_query
        == RAW_QUERY
    )

    with environment.engine.begin() as connection:
        connection.execute(
            update(baseline_run_payload)
            .where(baseline_run_payload.c.job_id == accepted["job_id"])
            .values(ciphertext=ciphertext_tamper["ciphertext"])
        )
    with pytest.raises(BaselineRunJobError, match="run_payload_authentication_failed"):
        service.verify_protected_payload_integrity(
            group_id=environment.group_id,
            job_id=str(accepted["job_id"]),
            caller_user_id=environment.user_id,
        )
    with environment.engine.connect() as connection:
        blocked = (
            connection.execute(
                select(baseline_run_job).where(
                    baseline_run_job.c.job_id == accepted["job_id"]
                )
            )
            .mappings()
            .one()
        )
        assert blocked["state"] == "blocked"
        assert blocked["reason_code"] == "payload_authentication_failed"
        assert (
            connection.execute(
                select(func.count())
                .select_from(baseline_run_payload)
                .where(baseline_run_payload.c.job_id == accepted["job_id"])
            ).scalar_one()
            == 0
        )
    blocked_status = service.read_status(
        request_id=str(uuid4()),
        group_id=environment.group_id,
        job_id=str(accepted["job_id"]),
        caller_user_id=environment.user_id,
    )
    _validate_contract(blocked_status, SCHEMA)
    assert blocked_status["state"] == "blocked"
    assert blocked_status["reason_code"] == "internal_failure"
    assert "payload_authentication_failed" not in str(blocked_status)


@pytest.mark.parametrize(
    "raw",
    (
        "",
        "{}",
        '{"version":"baseline-run-keyring.v1","active_key_id":"missing","keys":[]}',
        '{"version":"baseline-run-keyring.v1","active_key_id":"a","active_key_id":"b","keys":[]}',
        json.dumps(
            {
                "version": "baseline-run-keyring.v1",
                "active_key_id": "short",
                "keys": [{"key_id": "short", "key_base64": "YWJj"}],
            }
        ),
        json.dumps(
            {
                "version": "baseline-run-keyring.v1",
                "active_key_id": "duplicate",
                "keys": [
                    {
                        "key_id": "duplicate",
                        "key_base64": base64.b64encode(b"a" * 32).decode(),
                    },
                    {
                        "key_id": "duplicate",
                        "key_base64": base64.b64encode(b"b" * 32).decode(),
                    },
                ],
            }
        ),
    ),
)
def test_malformed_key_configuration_is_redacted(raw: str) -> None:
    with pytest.raises(BaselineRunJobError) as caught:
        BaselineRunKeyring.from_json(raw)
    assert caught.value.code in {"run_keyring_unavailable", "run_keyring_invalid"}
    if raw:
        assert raw not in str(caught.value)


def test_settings_keyring_is_secret_and_fail_closed() -> None:
    configured = json.dumps(
        {
            "version": "baseline-run-keyring.v1",
            "active_key_id": "settings-key",
            "keys": [
                {
                    "key_id": "settings-key",
                    "key_base64": base64.b64encode(b"s" * 32).decode(),
                }
            ],
        }
    )
    from pydantic import SecretStr

    settings = SimpleNamespace(baseline_run_encryption_keyring=SecretStr(configured))
    keyring = keyring_from_settings(settings)
    assert "settings-key" not in repr(keyring)
    with pytest.raises(BaselineRunJobError, match="run_keyring_unavailable"):
        keyring_from_settings(SimpleNamespace(baseline_run_encryption_keyring=None))


def test_idempotency_conflict_member_cross_group_and_authorization_rechecks(
    environment: ControlEnvironment,
) -> None:
    payload = _run_payload(environment)
    submission = parse_run_submission(payload)
    service = _service(environment)
    first = service.submit(submission, caller_user_id=environment.user_id)
    job, protected = _rows(environment, str(first["job_id"]))

    conflict = copy.deepcopy(payload)
    conflict["request_id"] = str(uuid4())
    conflict["retrieval_query"]["text"] += "\n# different intent\n"
    raw = conflict["retrieval_query"]["text"].encode()
    conflict["retrieval_query"]["byte_size"] = len(raw)
    conflict["retrieval_query"]["sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(BaselineRunJobError, match="idempotency_conflict"):
        service.submit(
            parse_run_submission(conflict), caller_user_id=environment.user_id
        )
    assert _rows(environment, str(first["job_id"])) == (job, protected)

    member = _add_group_member(environment.engine, group_id=environment.group_id)
    member_payload = copy.deepcopy(payload)
    member_payload["request_id"] = str(uuid4())
    member_payload["idempotency_key"] = "opaque-baseline-run-member-intent-00000001"
    member_result = service.submit(
        parse_run_submission(member_payload), caller_user_id=member
    )
    assert member_result["replayed"] is False

    outsider = str(uuid4())
    outsider_payload = copy.deepcopy(payload)
    outsider_payload["request_id"] = str(uuid4())
    outsider_payload["idempotency_key"] = "opaque-baseline-run-outsider-intent-000001"
    with pytest.raises(BaselineRunJobError, match="not_found_or_forbidden"):
        service.submit(parse_run_submission(outsider_payload), caller_user_id=outsider)

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
        service.submit(submission, caller_user_id=environment.user_id)
    with pytest.raises(BaselineRunJobError, match="job_not_found_or_forbidden"):
        service.read_status(
            request_id=str(uuid4()),
            group_id=environment.group_id,
            job_id=str(first["job_id"]),
            caller_user_id=environment.user_id,
        )


def test_source_deletion_stale_corpus_and_noncurrent_publication_write_nothing(
    environment: ControlEnvironment,
) -> None:
    payload = _run_payload(environment)
    service = _service(environment)
    with environment.engine.begin() as connection:
        connection.execute(
            text("DELETE FROM document WHERE document_id = :document_id"),
            {"document_id": environment.source_document_id},
        )
    with pytest.raises(BaselineRunJobError, match="source_not_authorized"):
        service.submit(
            parse_run_submission(payload), caller_user_id=environment.user_id
        )
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(baseline_run_job)
            ).scalar_one()
            == 0
        )


def test_source_deletion_preserves_safe_audit_but_group_deletion_cascades(
    environment: ControlEnvironment,
) -> None:
    submission = parse_run_submission(_run_payload(environment))
    service = _service(environment)
    accepted = service.submit(submission, caller_user_id=environment.user_id)
    with environment.engine.begin() as connection:
        connection.execute(
            text("DELETE FROM document WHERE document_id = :document_id"),
            {"document_id": environment.source_document_id},
        )
    with environment.engine.connect() as connection:
        row = (
            connection.execute(
                select(baseline_run_job).where(
                    baseline_run_job.c.job_id == accepted["job_id"]
                )
            )
            .mappings()
            .one()
        )
        assert row["source_document_id"] is None
        assert (
            connection.execute(
                select(func.count())
                .select_from(baseline_run_payload)
                .where(baseline_run_payload.c.job_id == accepted["job_id"])
            ).scalar_one()
            == 1
        )
    with pytest.raises(BaselineRunJobError, match="job_not_found_or_forbidden"):
        service.read_status(
            request_id=str(uuid4()),
            group_id=environment.group_id,
            job_id=str(accepted["job_id"]),
            caller_user_id=environment.user_id,
        )
    with environment.engine.begin() as connection:
        connection.execute(
            text('DELETE FROM "group" WHERE group_id = :group_id'),
            {"group_id": environment.group_id},
        )
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(baseline_run_job)
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(
                select(func.count()).select_from(baseline_run_payload)
            ).scalar_one()
            == 0
        )


def test_stale_active_generation_and_fingerprint_mismatch_write_nothing(
    environment: ControlEnvironment,
) -> None:
    payload = _run_payload(environment)
    service = _service(environment)
    mismatched = copy.deepcopy(payload)
    mismatched["request_id"] = str(uuid4())
    mismatched["idempotency_key"] = "opaque-baseline-run-mismatch-intent-0000001"
    mismatched["index_publication"]["index_fingerprint"] = "f" * 64
    with pytest.raises(BaselineRunJobError, match="index_publication_stale"):
        service.submit(
            parse_run_submission(mismatched), caller_user_id=environment.user_id
        )
    _publish_corpus(environment, ordinal=2)
    with pytest.raises(BaselineRunJobError, match="index_publication_stale"):
        service.submit(
            parse_run_submission(payload), caller_user_id=environment.user_id
        )
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(baseline_run_job)
            ).scalar_one()
            == 0
        )


def test_query_eight_megabyte_boundary_and_oversized_rejection(
    environment: ControlEnvironment,
) -> None:
    payload = _run_payload(environment)
    boundary = "x" * 8_000_000
    encoded = boundary.encode()
    payload["retrieval_query"].update(
        text=boundary,
        byte_size=len(encoded),
        sha256=hashlib.sha256(encoded).hexdigest(),
    )
    accepted = _service(environment).submit(
        parse_run_submission(payload), caller_user_id=environment.user_id
    )
    assert accepted["state"] == "queued"

    oversized = copy.deepcopy(payload)
    oversized_text = boundary + "x"
    oversized["request_id"] = str(uuid4())
    oversized["idempotency_key"] = "opaque-baseline-run-oversized-intent-000001"
    oversized["retrieval_query"].update(
        text=oversized_text,
        byte_size=len(oversized_text.encode()),
        sha256=hashlib.sha256(oversized_text.encode()).hexdigest(),
    )
    with pytest.raises(V2ControlPlaneError) as caught:
        parse_run_submission(oversized)
    assert caught.value.code == "limit_exceeded"


@pytest.mark.parametrize(
    "failure_stage",
    (
        RunSubmissionStage.AFTER_JOB_INSERT,
        RunSubmissionStage.AFTER_PAYLOAD_ENCRYPTION,
        RunSubmissionStage.AFTER_PAYLOAD_INSERT,
    ),
)
def test_transaction_rollback_at_every_write_stage(
    environment: ControlEnvironment,
    failure_stage: RunSubmissionStage,
) -> None:
    submission = parse_run_submission(_run_payload(environment))

    def fail(stage: RunSubmissionStage) -> None:
        if stage is failure_stage:
            raise RuntimeError("injected safe test failure")

    with pytest.raises(RuntimeError, match="injected safe test failure"):
        _service(environment, stage_hook=fail).submit(
            submission, caller_user_id=environment.user_id
        )
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(baseline_run_job)
            ).scalar_one()
            == 0
        )
        assert (
            connection.execute(
                select(func.count()).select_from(baseline_run_payload)
            ).scalar_one()
            == 0
        )


def test_expiry_erases_payload_retains_safe_audit_and_is_lease_aware(
    environment: ControlEnvironment,
) -> None:
    now = [datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)]
    service = _service(
        environment,
        clock=lambda: now[0],
        payload_lifetime=timedelta(minutes=1),
    )
    accepted = service.submit(
        parse_run_submission(_run_payload(environment)),
        caller_user_id=environment.user_id,
    )
    job_id = str(accepted["job_id"])
    with environment.engine.begin() as connection:
        connection.execute(
            update(baseline_run_job)
            .where(baseline_run_job.c.job_id == job_id)
            .values(
                state="running",
                lease_token="opaque-internal-lease-token",
                lease_expires_at=now[0] + timedelta(minutes=3),
            )
        )
    now[0] += timedelta(minutes=2)
    assert service.cleanup_protected_payloads() == 0
    now[0] += timedelta(minutes=2)
    assert service.cleanup_protected_payloads() == 1
    assert service.cleanup_protected_payloads() == 0
    with environment.engine.connect() as connection:
        row = (
            connection.execute(
                select(baseline_run_job).where(baseline_run_job.c.job_id == job_id)
            )
            .mappings()
            .one()
        )
        assert row["state"] == "blocked"
        assert row["reason_code"] == "payload_expired"
        assert (
            connection.execute(
                select(func.count())
                .select_from(baseline_run_payload)
                .where(baseline_run_payload.c.job_id == job_id)
            ).scalar_one()
            == 0
        )
    status = service.read_status(
        request_id=str(uuid4()),
        group_id=environment.group_id,
        job_id=job_id,
        caller_user_id=environment.user_id,
    )
    _validate_contract(status, SCHEMA)
    assert status["state"] == "blocked"
    assert status["reason_code"] == "worker_unavailable"


def test_concurrent_identical_submission_creates_one_job_and_payload(
    environment: ControlEnvironment,
) -> None:
    submission = parse_run_submission(_run_payload(environment))
    service = _service(environment)
    barrier = threading.Barrier(2)

    def submit():
        barrier.wait()
        return service.submit(submission, caller_user_id=environment.user_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _ordinal: submit(), range(2)))
    assert len({str(item["job_id"]) for item in results}) == 1
    assert sorted(bool(item["replayed"]) for item in results) == [False, True]
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


def test_injected_nonce_reuse_is_rejected_without_second_job(
    environment: ControlEnvironment,
) -> None:
    payload = _run_payload(environment)
    service = BaselineRunJobService(
        environment.engine,
        _keyring(),
        nonce_factory=lambda size: b"z" * size,
        secret_factory=lambda size: b"s" * size,
    )
    first = service.submit(
        parse_run_submission(payload), caller_user_id=environment.user_id
    )
    second_payload = copy.deepcopy(payload)
    second_payload["request_id"] = str(uuid4())
    second_payload["idempotency_key"] = "opaque-baseline-run-nonce-collision-000001"
    with pytest.raises(BaselineRunJobError) as caught:
        service.submit(
            parse_run_submission(second_payload),
            caller_user_id=environment.user_id,
        )
    assert caught.value.code == "run_payload_encryption_failed"
    assert caught.value.retryable is True
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
    assert _rows(environment, str(first["job_id"]))[1]["nonce"] == b"z" * 12
