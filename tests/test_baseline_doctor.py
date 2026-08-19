from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import create_engine, inspect, update

from compair_core import doctor as doctor_module
from compair_core.baseline_control_plane_schema import (
    baseline_run_payload,
    baseline_worker_attestation,
    baseline_worker_instance,
)
from compair_core.baseline_embedding.manifest import load_baseline_model_manifest
from compair_core.baseline_generation.budget import qualified_budget_profile
from compair_core.baseline_generation.profile import (
    GIB,
    RECOMMENDED_GENERATION_MODEL,
    RECOMMENDED_GENERATION_MODEL_DIGEST,
)
from compair_core.compair import models
from compair_core.compair.retrieval.corpus import ensure_retrieval_corpus_schema
from compair_core.compair.retrieval.database_worker import (
    DatabaseWorkerAttestation,
    DatabaseWorkerRegistry,
)
from compair_core.doctor import (
    DOCTOR_RESULT_SCHEMA_VERSION,
    BaselineDoctorResult,
    DoctorComponent,
    GenerationResourceSnapshot,
    run_doctor,
)
from compair_core.runtime_config import build_runtime_configuration
from compair_core.schema_migrations import run_schema_migrations
from compair_core.server.settings import Settings


def _keyring() -> str:
    return json.dumps(
        {
            "version": "baseline-run-keyring.v1",
            "active_key_id": "active",
            "keys": [
                {
                    "key_id": "active",
                    "key_base64": base64.b64encode(b"k" * 32).decode("ascii"),
                }
            ],
        },
        separators=(",", ":"),
    )


def _settings(tmp_path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "retrieval_engine": "baseline_v1",
        "baseline_runs_enabled": True,
        "baseline_worker_mode": "database",
        "baseline_embedding_provider": "http",
        "baseline_embedding_endpoint": "http://127.0.0.1:9010",
        "baseline_embedding_revision": ("52398278842ec682c6f32300af41344b1c0b0bb2"),
        "baseline_embedding_allow_insecure_loopback": True,
        "baseline_model_cache": str(tmp_path / "models"),
        "baseline_generation_provider": "ollama",
        "baseline_generation_endpoint": "http://127.0.0.1:11434",
        "baseline_generation_model": RECOMMENDED_GENERATION_MODEL,
        "baseline_generation_model_digest": RECOMMENDED_GENERATION_MODEL_DIGEST,
        "baseline_generation_allow_loopback_http": True,
        "baseline_run_encryption_keyring": _keyring(),
        "baseline_notifications_enabled": False,
    }
    values.update(overrides)
    return Settings(**values)


def _engine(tmp_path, *, migrate: bool = True):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'doctor.db'}",
        connect_args={"check_same_thread": False},
    )
    models.Base.metadata.create_all(engine)
    ensure_retrieval_corpus_schema(engine)
    if migrate:
        run_schema_migrations(engine)
    return engine


def _embedding_client():
    manifest = load_baseline_model_manifest()

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/health"
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "contract_version": manifest.contract_version,
                "provider": manifest.provider,
                "model": manifest.logical_model,
                "revision": manifest.revision,
                "dimension": manifest.dimension,
            },
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


def _generation_factory(calls: list[str]):
    def factory():
        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            if request.url.path == "/api/version":
                return httpx.Response(200, json={"version": "0.32.14"})
            if request.url.path == "/api/tags":
                return httpx.Response(
                    200,
                    json={
                        "models": [
                            {
                                "name": RECOMMENDED_GENERATION_MODEL,
                                "digest": RECOMMENDED_GENERATION_MODEL_DIGEST,
                            }
                        ]
                    },
                )
            if request.url.path == "/api/chat":
                payload = json.loads(request.content)
                if payload.get("_debug_render_only") is True:
                    return httpx.Response(
                        200,
                        json={
                            "model": RECOMMENDED_GENERATION_MODEL,
                            "done": True,
                            "_debug_info": {
                                "rendered_template": qualified_budget_profile().attestation_render
                            },
                        },
                    )
                return httpx.Response(
                    200,
                    json={
                        "model": RECOMMENDED_GENERATION_MODEL,
                        "done": True,
                        "done_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "schema_version": ("baseline-generation-output.v2"),
                                    "outcome": "no_findings",
                                    "findings": [],
                                },
                                separators=(",", ":"),
                            )
                        },
                    },
                )
            raise AssertionError(request.url.path)

        return httpx.Client(transport=httpx.MockTransport(handler))

    return factory


def _verified_model():
    return SimpleNamespace(manifest=load_baseline_model_manifest())


@pytest.fixture(autouse=True)
def _stable_generation_resources(monkeypatch) -> None:
    monkeypatch.setattr(
        doctor_module,
        "_sample_generation_resources",
        lambda: GenerationResourceSnapshot(
            total_memory_bytes=32 * GIB,
            available_memory_bytes=24 * GIB,
            free_storage_bytes=40 * GIB,
        ),
    )


def _matching_worker(engine, settings, *, clock=None):
    runtime = build_runtime_configuration(settings, database_url=engine.url)
    registry = DatabaseWorkerRegistry(
        engine,
        heartbeat_ttl=timedelta(seconds=30),
        attestation=DatabaseWorkerAttestation.from_runtime(runtime),
        **({"clock": clock} if clock is not None else {}),
    )
    worker_id = str(uuid4())
    registry.register(worker_id)
    return registry, worker_id


def test_doctor_ready_json_contract_and_probe_is_opt_in(
    tmp_path,
    monkeypatch,
) -> None:
    engine = _engine(tmp_path)
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        doctor_module, "verify_baseline_model", lambda _root: _verified_model()
    )
    _matching_worker(engine, settings)
    generation_calls: list[str] = []

    without_probe = run_doctor(
        settings=settings,
        engine=engine,
        embedding_client_factory=_embedding_client,
        generation_client_factory=_generation_factory(generation_calls),
        clock=lambda: datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    assert without_probe.status == "ready"
    assert without_probe.generation_probed is False
    assert generation_calls.count("/api/chat") == 1
    payload = without_probe.as_dict()
    assert payload["schema_version"] == DOCTOR_RESULT_SCHEMA_VERSION
    assert payload["timestamp"] == "2026-01-02T00:00:00Z"
    assert without_probe.exit_code(require_baseline=True) == 0

    generation_calls.clear()
    probed = run_doctor(
        settings=settings,
        engine=engine,
        probe_generation=True,
        embedding_client_factory=_embedding_client,
        generation_client_factory=_generation_factory(generation_calls),
    )
    assert probed.generation_probed is True
    assert generation_calls.count("/api/chat") == 2
    assert without_probe.component("notifications").reason_code == (
        "notifications_default_off"
    )


def test_generation_resource_projection_warning_failure_and_attested_gpu() -> None:
    recommended = doctor_module._generation_resource_component(
        _settings(Path(".")),
        lambda: GenerationResourceSnapshot(
            total_memory_bytes=32 * GIB,
            available_memory_bytes=24 * GIB,
            free_storage_bytes=40 * GIB,
        ),
    )
    assert recommended.status == "ready"
    assert recommended.reason_code == "generation_resources_recommended"
    assert recommended.details["warning"] is False
    assert recommended.details["accelerator_memory_attested"] is False
    assert recommended.details["accelerator_memory_bytes"] is None
    assert recommended.details["measured_32k_inference_allocation_bytes"] == (15 * GIB)
    assert recommended.details["recommended_total_memory_bytes"] == 24 * GIB
    assert recommended.details["preferred_total_memory_bytes"] == 32 * GIB
    assert recommended.details["minimum_free_storage_bytes"] == 25 * GIB
    assert recommended.details["acquisition_free_storage_bytes"] == 40 * GIB

    warning = doctor_module._generation_resource_component(
        _settings(Path(".")),
        lambda: GenerationResourceSnapshot(
            total_memory_bytes=24 * GIB,
            available_memory_bytes=16 * GIB,
            free_storage_bytes=25 * GIB,
        ),
    )
    assert warning.status == "ready"
    assert warning.reason_code == "generation_resources_warning"
    assert warning.details["readiness_blocking"] is False
    assert warning.details["warning_codes"] == [
        "total_memory_below_preferred",
        "free_storage_below_acquisition_recommendation",
    ]

    insufficient = doctor_module._generation_resource_component(
        _settings(Path(".")),
        lambda: GenerationResourceSnapshot(
            total_memory_bytes=8 * GIB,
            available_memory_bytes=4 * GIB,
            free_storage_bytes=50 * GIB,
        ),
    )
    assert insufficient.status == "not_ready"
    assert insufficient.reason_code == "generation_resources_insufficient"
    assert insufficient.details["readiness_blocking"] is True

    advisory = doctor_module._generation_resource_component(
        Settings(),
        lambda: GenerationResourceSnapshot(
            total_memory_bytes=8 * GIB,
            available_memory_bytes=4 * GIB,
            free_storage_bytes=10 * GIB,
        ),
    )
    assert advisory.status == "ready"
    assert advisory.reason_code == "generation_resources_warning"
    assert advisory.details["recommended_profile_selected"] is False
    assert advisory.details["readiness_blocking"] is False

    dedicated = doctor_module._generation_resource_component(
        _settings(Path(".")),
        lambda: GenerationResourceSnapshot(
            total_memory_bytes=8 * GIB,
            available_memory_bytes=4 * GIB,
            free_storage_bytes=50 * GIB,
            accelerator_memory_attested=True,
            accelerator_memory_bytes=24 * GIB,
        ),
    )
    assert dedicated.status == "ready"
    assert dedicated.reason_code == "generation_resources_recommended"
    assert dedicated.details["assessment_mode"] == ("attested_dedicated_accelerator")
    rendered = json.dumps(dedicated.as_dict(), sort_keys=True)
    for prohibited in ("/Users/", "http://", "postgresql://", "credential"):
        assert prohibited not in rendered


def test_insufficient_generation_resources_fail_baseline_readiness(
    tmp_path,
    monkeypatch,
) -> None:
    engine = _engine(tmp_path)
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        doctor_module, "verify_baseline_model", lambda _root: _verified_model()
    )
    _matching_worker(engine, settings)
    result = run_doctor(
        settings=settings,
        engine=engine,
        embedding_client_factory=_embedding_client,
        generation_client_factory=_generation_factory([]),
        generation_resource_sampler=lambda: GenerationResourceSnapshot(
            total_memory_bytes=8 * GIB,
            available_memory_bytes=4 * GIB,
            free_storage_bytes=50 * GIB,
        ),
    )
    assert result.component("generation_resources").status == "not_ready"
    assert result.component("baseline_capability").reason_code == ("baseline_not_ready")
    assert result.exit_code(require_baseline=True) == 5


def test_doctor_does_not_create_missing_migration_registry(tmp_path) -> None:
    engine = _engine(tmp_path, migrate=False)
    assert "core_schema_migration" not in inspect(engine).get_table_names()
    result = run_doctor(settings=Settings(), engine=engine)
    assert result.component("migrations").reason_code == "migration_registry_missing"
    assert "core_schema_migration" not in inspect(engine).get_table_names()
    assert result.exit_code(require_baseline=False) == 3


def test_worker_mismatch_draining_capacity_and_stale_are_safe(
    tmp_path,
    monkeypatch,
) -> None:
    engine = _engine(tmp_path)
    settings = _settings(tmp_path)
    monkeypatch.setattr(
        doctor_module, "verify_baseline_model", lambda _root: _verified_model()
    )
    registry, worker_id = _matching_worker(engine, settings)
    with engine.begin() as connection:
        connection.execute(
            update(baseline_worker_attestation)
            .where(baseline_worker_attestation.c.worker_instance_id == worker_id)
            .values(runtime_config_fingerprint="0" * 64)
        )
    mismatch = run_doctor(
        settings=settings,
        engine=engine,
        embedding_client_factory=_embedding_client,
        generation_client_factory=_generation_factory([]),
    )
    component = mismatch.component("worker")
    assert component.reason_code == "worker_configuration_mismatch"
    assert component.details["mismatched_workers"] == 1
    assert mismatch.exit_code(require_baseline=True) == 6

    with engine.begin() as connection:
        runtime = build_runtime_configuration(settings, database_url=engine.url)
        connection.execute(
            update(baseline_worker_attestation)
            .where(baseline_worker_attestation.c.worker_instance_id == worker_id)
            .values(runtime_config_fingerprint=runtime.fingerprint)
        )
        connection.execute(
            update(baseline_worker_instance)
            .where(baseline_worker_instance.c.worker_instance_id == worker_id)
            .values(draining=True)
        )
    draining = run_doctor(
        settings=settings,
        engine=engine,
        embedding_client_factory=_embedding_client,
        generation_client_factory=_generation_factory([]),
    )
    assert draining.component("worker").details["draining_workers"] == 1

    old = datetime.now(timezone.utc) - timedelta(minutes=10)
    with engine.begin() as connection:
        connection.execute(
            update(baseline_worker_instance)
            .where(baseline_worker_instance.c.worker_instance_id == worker_id)
            .values(draining=False, last_heartbeat_at=old)
        )
    stale = run_doctor(
        settings=settings,
        engine=engine,
        embedding_client_factory=_embedding_client,
        generation_client_factory=_generation_factory([]),
    )
    assert stale.component("worker").details["stale_workers"] == 1
    assert registry is not None


def test_json_cli_emits_one_value_and_no_sensitive_strings(
    monkeypatch,
    capsys,
) -> None:
    result = BaselineDoctorResult(
        status="degraded",
        runtime_configuration_fingerprint="f" * 64,
        components=(
            DoctorComponent("configuration", "ready", "configuration_valid", {}),
            DoctorComponent("database", "ready", "database_reachable", {}),
            DoctorComponent("migrations", "ready", "migrations_current", {}),
            DoctorComponent("embedding", "degraded", "embedding_unavailable", {}),
            DoctorComponent("generation", "ready", "generation_identity_attested", {}),
            DoctorComponent("worker", "ready", "worker_ready", {}),
        ),
        generated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        generation_probed=False,
        recommended_actions=("start_or_correct_embedding_service",),
    )
    monkeypatch.setattr(doctor_module, "run_doctor", lambda **_kwargs: result)
    assert doctor_module.main(["doctor", "--json"]) == 1
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["schema_version"] == DOCTOR_RESULT_SCHEMA_VERSION
    assert captured.out.count("\n") == 1
    assert captured.err == ""
    for secret in (
        "postgresql://",
        "http://",
        "/Users/",
        "raw query",
        "private evidence",
        "job-id",
    ):
        assert secret not in captured.out


def test_doctor_reports_invalid_and_removed_referenced_keys(tmp_path) -> None:
    engine = _engine(tmp_path)
    invalid = run_doctor(
        settings=_settings(tmp_path, baseline_run_encryption_keyring="invalid"),
        engine=engine,
    )
    assert invalid.component("keyring").reason_code == "run_keyring_invalid"
    assert invalid.exit_code(require_baseline=False) == 2

    now = datetime.now(timezone.utc)
    connection = engine.connect()
    try:
        connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
        connection.execute(
            baseline_run_payload.insert().values(
                job_id=str(uuid4()),
                group_id=str(uuid4()),
                payload_schema_version="baseline-run-protected-payload.v1",
                algorithm="AES-256-GCM",
                key_id="removed-key",
                nonce=b"n" * 12,
                ciphertext=b"c" * 16,
                aad_version="baseline-run-aad.v1",
                created_at=now,
                expires_at=now + timedelta(minutes=5),
            )
        )
        connection.commit()
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
    finally:
        connection.close()
    removed = run_doctor(settings=_settings(tmp_path), engine=engine)
    keyring = removed.component("keyring")
    assert keyring.reason_code == "run_payload_key_unavailable"
    assert keyring.details["removed_referenced_key_count"] == 1


def test_doctor_reports_provider_identity_failures_without_private_output(
    tmp_path,
    monkeypatch,
) -> None:
    engine = _engine(tmp_path)
    settings = _settings(
        tmp_path,
        baseline_generation_model_digest="sha256:" + "9" * 64,
    )
    monkeypatch.setattr(
        doctor_module,
        "verify_baseline_model",
        lambda _root: _verified_model(),
    )

    def wrong_embedding():
        manifest = load_baseline_model_manifest()
        return httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    json={
                        "status": "ok",
                        "contract_version": manifest.contract_version,
                        "provider": manifest.provider,
                        "model": manifest.logical_model,
                        "revision": manifest.revision,
                        "dimension": manifest.dimension + 1,
                    },
                )
            )
        )

    result = run_doctor(
        settings=settings,
        engine=engine,
        embedding_client_factory=wrong_embedding,
        generation_client_factory=_generation_factory([]),
    )
    assert result.component("embedding").reason_code == ("embedding_identity_mismatch")
    assert result.component("generation").reason_code == ("generation_digest_mismatch")
    rendered = json.dumps(result.as_dict(), sort_keys=True)
    assert "http://" not in rendered
    assert str(tmp_path) not in rendered
    assert result.exit_code(require_baseline=True) == 4

    generation_only = run_doctor(
        settings=settings,
        engine=engine,
        embedding_client_factory=_embedding_client,
        generation_client_factory=_generation_factory([]),
    )
    assert generation_only.component("embedding").status == "ready"
    assert generation_only.exit_code(require_baseline=True) == 5


def test_doctor_internal_failure_is_sanitized_exit_seven(
    monkeypatch,
    capsys,
) -> None:
    def fail(**_kwargs):
        raise RuntimeError("private /path and endpoint http://secret.invalid")

    monkeypatch.setattr(doctor_module, "run_doctor", fail)
    assert doctor_module.main(["doctor", "--json"]) == 7
    output = capsys.readouterr()
    assert output.err == ""
    payload = json.loads(output.out)
    assert payload["status"] == "not_ready"
    assert "private" not in output.out
    assert "secret.invalid" not in output.out
