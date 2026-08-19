"""Privacy-safe operational diagnostics for the self-hosted baseline workflow."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import logging
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import psutil
from sqlalchemy import Engine, func, inspect, select

from .baseline_control_plane_schema import (
    baseline_run_job,
    baseline_run_payload,
    baseline_worker_attestation,
    baseline_worker_instance,
    control_job,
    snapshot_continuation_job,
)
from .baseline_embedding.cache import (
    BaselineModelCacheError,
    default_cache_root,
    verify_baseline_model,
)
from .baseline_embedding.manifest import (
    BaselineModelManifestError,
    load_baseline_model_manifest,
)
from .baseline_generation.profile import (
    ACQUISITION_FREE_STORAGE_BYTES,
    MEASURED_32K_INFERENCE_ALLOCATION_BYTES,
    MINIMUM_FREE_STORAGE_BYTES,
    MINIMUM_GENERATION_CAPACITY_BYTES,
    PREFERRED_TOTAL_MEMORY_BYTES,
    RECOMMENDED_GENERATION_MODEL,
    RECOMMENDED_GENERATION_MODEL_DIGEST,
    RECOMMENDED_TOTAL_MEMORY_BYTES,
)
from .config_init import add_config_init_arguments, run_config_init_command
from .runtime_config import (
    BASELINE_GENERATION_OUTPUT_SCHEMA_SHA256,
    RUNTIME_CONFIG_CONTRACT_VERSION,
    WORKER_CONTRACT_VERSION,
    WORKER_SUPPORTED_JOB_TYPES,
    RuntimeConfigurationAttestation,
    RuntimeConfigurationError,
    attest_keyring,
    build_runtime_configuration,
    validate_runtime_configuration,
)
from .schema_migrations import (
    CORE_SCHEMA_MIGRATIONS,
    MIGRATION_TABLE_NAME,
    schema_migration_table,
)
from .server.settings import Settings

DOCTOR_RESULT_SCHEMA_VERSION = "baseline-doctor-result.v1"
_READY = "ready"
_DEGRADED = "degraded"
_NOT_READY = "not_ready"
_PENDING_STATES = frozenset({"queued", "running"})
_RETRYABLE_STATES = frozenset({"retryable_failed", "references_persisted"})
_BLOCKED_STATES = frozenset({"blocked", "terminal_failed"})
_MINIMUM_OLLAMA = (0, 32, 13)
_DISK_MINIMUM_DATABASE_BYTES = 512 * 1024 * 1024
_DISK_MINIMUM_TEMP_BYTES = 512 * 1024 * 1024


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class DoctorComponent:
    name: str
    status: str
    reason_code: str
    details: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "reason_code": self.reason_code,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class GenerationResourceSnapshot:
    """Privacy-safe host/accelerator capacity supplied to doctor."""

    total_memory_bytes: int
    available_memory_bytes: int
    free_storage_bytes: int
    accelerator_memory_attested: bool = False
    accelerator_memory_bytes: int | None = None

    def validate(self) -> None:
        values = (
            self.total_memory_bytes,
            self.available_memory_bytes,
            self.free_storage_bytes,
        )
        if any(value < 0 for value in values):
            raise ValueError("resource measurement is invalid")
        if self.accelerator_memory_attested:
            if (
                self.accelerator_memory_bytes is None
                or self.accelerator_memory_bytes < 0
            ):
                raise ValueError("accelerator measurement is invalid")
        elif self.accelerator_memory_bytes is not None:
            raise ValueError("unattested accelerator memory is prohibited")


@dataclass(frozen=True, slots=True)
class BaselineDoctorResult:
    status: str
    runtime_configuration_fingerprint: str | None
    components: tuple[DoctorComponent, ...]
    generated_at: datetime
    generation_probed: bool
    recommended_actions: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": DOCTOR_RESULT_SCHEMA_VERSION,
            "status": self.status,
            "runtime_configuration_fingerprint": (
                self.runtime_configuration_fingerprint
            ),
            "components": [component.as_dict() for component in self.components],
            "timestamp": self.generated_at.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "generation_probed": self.generation_probed,
            "recommended_actions": list(self.recommended_actions),
        }

    def component(self, name: str) -> DoctorComponent:
        return next(item for item in self.components if item.name == name)

    def exit_code(self, *, require_baseline: bool) -> int:
        if self.status == _READY:
            return 0
        configuration = self.component("configuration")
        database = self.component("database")
        migrations = self.component("migrations")
        embedding = self.component("embedding")
        generation = self.component("generation")
        generation_resources = next(
            (
                component
                for component in self.components
                if component.name == "generation_resources"
            ),
            None,
        )
        worker = self.component("worker")
        if configuration.status == _NOT_READY:
            return 2
        if database.status == _NOT_READY or migrations.status == _NOT_READY:
            return 3
        if require_baseline:
            if embedding.status != _READY:
                return 4
            if generation.status != _READY:
                return 5
            if (
                generation_resources is not None
                and generation_resources.status == _NOT_READY
            ):
                return 5
            if worker.status != _READY:
                return 6
            return 2
        return 1


def _component(
    name: str,
    status: str,
    reason: str,
    **details: object,
) -> DoctorComponent:
    return DoctorComponent(name, status, reason, details)


def _safe_package_version() -> str:
    try:
        return version("compair-core")
    except PackageNotFoundError:
        return "0.10.4"


def _migration_component(engine: Engine) -> DoctorComponent:
    expected = {item.migration_id: item for item in CORE_SCHEMA_MIGRATIONS}
    try:
        table_names = set(inspect(engine).get_table_names())
        if MIGRATION_TABLE_NAME not in table_names:
            return _component(
                "migrations",
                _NOT_READY,
                "migration_registry_missing",
                expected_latest=CORE_SCHEMA_MIGRATIONS[-1].migration_id,
                applied_latest=None,
                pending_count=len(expected),
            )
        with engine.connect() as connection:
            rows = connection.execute(select(schema_migration_table)).mappings().all()
        observed = {str(row["migration_id"]): row for row in rows}
        invalid = [
            migration_id
            for migration_id, migration in expected.items()
            if migration_id in observed
            and (
                observed[migration_id]["state"] != "applied"
                or observed[migration_id]["checksum"] != migration.checksum
            )
        ]
        pending = [item for item in expected if item not in observed]
        applied = [item for item in expected if item in observed]
        if invalid:
            return _component(
                "migrations",
                _NOT_READY,
                "migration_state_mismatch",
                expected_latest=CORE_SCHEMA_MIGRATIONS[-1].migration_id,
                applied_latest=max(applied) if applied else None,
                pending_count=len(pending),
                invalid_count=len(invalid),
            )
        if pending:
            return _component(
                "migrations",
                _NOT_READY,
                "migrations_pending",
                expected_latest=CORE_SCHEMA_MIGRATIONS[-1].migration_id,
                applied_latest=max(applied) if applied else None,
                pending_count=len(pending),
            )
        return _component(
            "migrations",
            _READY,
            "migrations_current",
            expected_latest=CORE_SCHEMA_MIGRATIONS[-1].migration_id,
            applied_latest=CORE_SCHEMA_MIGRATIONS[-1].migration_id,
            pending_count=0,
        )
    except Exception:  # noqa: BLE001 - database details remain private
        return _component(
            "migrations",
            _NOT_READY,
            "migration_inspection_failed",
            expected_latest=CORE_SCHEMA_MIGRATIONS[-1].migration_id,
            applied_latest=None,
            pending_count=len(expected),
        )


def _database_component(
    engine: Engine,
    runtime: RuntimeConfigurationAttestation | None,
) -> DoctorComponent:
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1").scalar_one()
        return _component(
            "database",
            _READY,
            "database_reachable",
            backend=engine.dialect.name,
            identity_fingerprint=(
                runtime.database_identity_fingerprint if runtime else None
            ),
        )
    except Exception:  # noqa: BLE001 - DSN/provider details are prohibited
        return _component(
            "database",
            _NOT_READY,
            "database_unavailable",
            backend=engine.dialect.name,
            identity_fingerprint=(
                runtime.database_identity_fingerprint if runtime else None
            ),
        )


def _endpoint(value: object, *, allow_loopback_http: bool) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("endpoint_absent")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("endpoint_invalid")
    try:
        loopback = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        loopback = parsed.hostname.lower() == "localhost"
    if parsed.scheme == "http" and not (allow_loopback_http and loopback):
        raise ValueError("insecure_transport")
    return value.rstrip("/")


def _embedding_component(
    settings: Settings,
    runtime: RuntimeConfigurationAttestation | None,
    *,
    client_factory: Callable[[], httpx.Client] | None,
) -> DoctorComponent:
    fingerprint = runtime.embedding_identity_fingerprint if runtime else None
    if settings.baseline_embedding_provider != "http":
        return _component(
            "embedding",
            _DEGRADED,
            "embedding_provider_disabled",
            identity_fingerprint=fingerprint,
            cache_status="not_checked",
        )
    try:
        verified = verify_baseline_model(settings.baseline_model_cache)
        endpoint = _endpoint(
            settings.baseline_embedding_endpoint,
            allow_loopback_http=settings.baseline_embedding_allow_insecure_loopback,
        )
        factory = client_factory or (
            lambda: httpx.Client(
                timeout=httpx.Timeout(settings.baseline_embedding_timeout_seconds),
                follow_redirects=False,
                trust_env=False,
            )
        )
        with factory() as client:
            response = client.get(f"{endpoint}/v1/health")
            if response.status_code != 200:
                raise ValueError("embedding_service_unavailable")
            payload = response.json()
        manifest = verified.manifest
        expected = {
            "contract_version": manifest.contract_version,
            "provider": manifest.provider,
            "model": settings.baseline_embedding_model,
            "revision": settings.baseline_embedding_revision,
            "dimension": settings.baseline_embedding_dimension,
        }
        if not isinstance(payload, dict) or any(
            payload.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("embedding_identity_mismatch")
        return _component(
            "embedding",
            _READY,
            "embedding_identity_attested",
            identity_fingerprint=fingerprint,
            manifest_fingerprint=manifest.manifest_fingerprint,
            model_artifact_fingerprint=manifest.model_artifact_fingerprint,
            cache_status="verified",
            dimension=manifest.dimension,
            dtype=manifest.dtype,
        )
    except BaselineModelCacheError as exc:
        return _component(
            "embedding",
            _DEGRADED,
            exc.code,
            identity_fingerprint=fingerprint,
            cache_status="unavailable",
        )
    except (
        BaselineModelManifestError,
        ValueError,
        httpx.HTTPError,
        TypeError,
    ) as exc:
        reason = "embedding_unavailable"
        if isinstance(exc, ValueError) and str(exc) == "embedding_identity_mismatch":
            reason = "embedding_identity_mismatch"
        return _component(
            "embedding",
            _DEGRADED,
            reason,
            identity_fingerprint=fingerprint,
            cache_status="verified",
        )


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, value in values:
            if key in output:
                raise ValueError("duplicate")
            output[key] = value
        return output

    return json.loads(
        raw.decode("utf-8", errors="strict"),
        object_pairs_hook=pairs,
        parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonfinite")),
    )


def _generation_probe_outcome(value: object) -> str:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "outcome",
        "findings",
    }:
        raise ValueError("structured_output_unavailable")
    if value["schema_version"] != "baseline-generation-output.v2":
        raise ValueError("structured_output_unavailable")
    findings = value["findings"]
    if not isinstance(findings, list):
        raise TypeError("structured_output_unavailable")
    if value["outcome"] == "no_findings":
        if findings:
            raise ValueError("structured_output_unavailable")
        return "no_findings"
    if value["outcome"] != "findings" or not 1 <= len(findings) <= 4:
        raise ValueError("structured_output_unavailable")
    for finding in findings:
        if not isinstance(finding, dict) or set(finding) != {"feedback"}:
            raise ValueError("structured_output_unavailable")
        feedback = finding["feedback"]
        if (
            not isinstance(feedback, str)
            or len(feedback) > 100_000
            or not any(not character.isspace() for character in feedback)
        ):
            raise ValueError("structured_output_unavailable")
    return "findings"


def _generation_component(
    settings: Settings,
    runtime: RuntimeConfigurationAttestation | None,
    *,
    probe: bool,
    client_factory: Callable[[], httpx.Client] | None,
) -> DoctorComponent:
    fingerprint = runtime.generation_identity_fingerprint if runtime else None
    if settings.baseline_generation_provider != "ollama":
        return _component(
            "generation",
            _DEGRADED,
            "generation_provider_disabled",
            identity_fingerprint=fingerprint,
            probe_performed=probe,
            probe_outcome=None,
        )
    try:
        endpoint = _endpoint(
            settings.baseline_generation_endpoint,
            allow_loopback_http=settings.baseline_generation_allow_loopback_http,
        )
        factory = client_factory or (
            lambda: httpx.Client(
                timeout=httpx.Timeout(settings.baseline_generation_timeout_seconds),
                follow_redirects=False,
                trust_env=False,
            )
        )
        # Resolve the package data beside this module without importing
        # baseline_generation.__init__, whose production provider imports the
        # legacy application package and therefore owns startup migrations.
        schema_raw = (
            Path(__file__).resolve().parent
            / "baseline_generation"
            / "baseline-generation-output.v2.schema.json"
        ).read_bytes()
        if hashlib.sha256(schema_raw).hexdigest() != (
            BASELINE_GENERATION_OUTPUT_SCHEMA_SHA256
        ):
            raise ValueError("structured_output_unavailable")
        schema = _strict_json(schema_raw)
        with factory() as client:
            version_response = client.get(f"{endpoint}/api/version")
            tags_response = client.get(f"{endpoint}/api/tags")
            if version_response.status_code != 200 or tags_response.status_code != 200:
                raise ValueError("generation_unavailable")
            version_payload = version_response.json()
            tags_payload = tags_response.json()
            runtime_version = str(version_payload.get("version", ""))
            numeric = runtime_version.split("-", 1)[0].split("+", 1)[0]
            parsed_version = tuple(int(item) for item in numeric.split("."))
            if len(parsed_version) != 3 or parsed_version < _MINIMUM_OLLAMA:
                raise ValueError("generation_runtime_mismatch")
            models = tags_payload.get("models")
            if not isinstance(models, list):
                raise TypeError("generation_unavailable")
            selected = next(
                (
                    item
                    for item in models
                    if isinstance(item, dict)
                    and settings.baseline_generation_model
                    in {item.get("name"), item.get("model")}
                ),
                None,
            )
            if selected is None:
                raise ValueError("generation_model_absent")
            digest = str(selected.get("digest", ""))
            if digest and not digest.startswith("sha256:"):
                digest = f"sha256:{digest}"
            if digest != settings.baseline_generation_model_digest:
                raise ValueError("generation_digest_mismatch")
            probe_outcome: str | None = None
            if probe:
                body = {
                    "model": settings.baseline_generation_model,
                    "stream": False,
                    "think": False,
                    "format": schema,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Return one JSON object matching the schema. "
                                "Use no_findings with an empty findings array."
                            ),
                        },
                        {
                            "role": "user",
                            "content": "Synthetic private-data-free compatibility probe.",
                        },
                    ],
                    "options": {
                        "temperature": 0,
                        "seed": settings.baseline_generation_seed,
                        "num_ctx": settings.baseline_generation_context_tokens,
                        "num_predict": settings.baseline_generation_output_tokens,
                    },
                }
                response = client.post(f"{endpoint}/api/chat", json=body)
                if response.status_code != 200:
                    raise ValueError("structured_output_unavailable")
                payload = response.json()
                if (
                    not isinstance(payload, dict)
                    or payload.get("model") != settings.baseline_generation_model
                    or payload.get("done") is not True
                    or payload.get("done_reason") == "length"
                ):
                    raise ValueError("structured_output_unavailable")
                message = payload.get("message") if isinstance(payload, dict) else None
                content = message.get("content") if isinstance(message, dict) else None
                parsed = _strict_json(str(content).encode("utf-8"))
                probe_outcome = _generation_probe_outcome(parsed)
        return _component(
            "generation",
            _READY,
            "generation_identity_attested",
            identity_fingerprint=fingerprint,
            runtime_version=runtime_version,
            output_schema_sha256=BASELINE_GENERATION_OUTPUT_SCHEMA_SHA256,
            probe_performed=probe,
            probe_outcome=probe_outcome,
            supports_idempotency=False,
            model=settings.baseline_generation_model,
            model_digest=settings.baseline_generation_model_digest,
        )
    except (
        ValueError,
        TypeError,
        httpx.HTTPError,
        json.JSONDecodeError,
    ) as exc:
        reason = "generation_unavailable"
        if isinstance(exc, ValueError) and str(exc) in {
            "generation_digest_mismatch",
            "generation_model_absent",
            "generation_runtime_mismatch",
            "structured_output_unavailable",
        }:
            reason = str(exc)
        return _component(
            "generation",
            _DEGRADED,
            reason,
            identity_fingerprint=fingerprint,
            probe_performed=probe,
            probe_outcome=None,
        )


def _sample_generation_resources() -> GenerationResourceSnapshot:
    memory = psutil.virtual_memory()
    return GenerationResourceSnapshot(
        total_memory_bytes=int(memory.total),
        available_memory_bytes=int(memory.available),
        free_storage_bytes=int(shutil.disk_usage(tempfile.gettempdir()).free),
    )


def _generation_resource_component(
    settings: Settings,
    sampler: Callable[[], GenerationResourceSnapshot] | None,
) -> DoctorComponent:
    recommended_profile_selected = (
        settings.baseline_generation_provider == "ollama"
        and settings.baseline_generation_model == RECOMMENDED_GENERATION_MODEL
        and settings.baseline_generation_model_digest
        == RECOMMENDED_GENERATION_MODEL_DIGEST
    )
    try:
        snapshot = (sampler or _sample_generation_resources)()
        snapshot.validate()
    except (OSError, ValueError, psutil.Error):
        return _component(
            "generation_resources",
            _READY,
            "generation_resources_unattested",
            readiness_blocking=False,
            recommended_profile_selected=recommended_profile_selected,
            warning=True,
            warning_codes=["resource_measurement_unavailable"],
            assessment_mode="host_memory_conservative",
            accelerator_memory_attested=False,
            accelerator_memory_bytes=None,
            total_memory_bytes=None,
            available_memory_bytes=None,
            free_storage_bytes=None,
            measured_32k_inference_allocation_bytes=(
                MEASURED_32K_INFERENCE_ALLOCATION_BYTES
            ),
            recommended_total_memory_bytes=RECOMMENDED_TOTAL_MEMORY_BYTES,
            preferred_total_memory_bytes=PREFERRED_TOTAL_MEMORY_BYTES,
            minimum_free_storage_bytes=MINIMUM_FREE_STORAGE_BYTES,
            acquisition_free_storage_bytes=ACQUISITION_FREE_STORAGE_BYTES,
            recommended_model=RECOMMENDED_GENERATION_MODEL,
            recommended_model_digest=RECOMMENDED_GENERATION_MODEL_DIGEST,
        )

    attested_accelerator = (
        snapshot.accelerator_memory_bytes
        if snapshot.accelerator_memory_attested
        else None
    )
    accelerator_satisfies_floor = (
        attested_accelerator is not None
        and attested_accelerator >= MINIMUM_GENERATION_CAPACITY_BYTES
    )
    capacity_satisfies_floor = (
        snapshot.total_memory_bytes >= MINIMUM_GENERATION_CAPACITY_BYTES
        or accelerator_satisfies_floor
    )
    warnings: list[str] = []
    if not accelerator_satisfies_floor:
        if snapshot.total_memory_bytes < RECOMMENDED_TOTAL_MEMORY_BYTES:
            warnings.append("total_memory_below_recommended")
        elif snapshot.total_memory_bytes < PREFERRED_TOTAL_MEMORY_BYTES:
            warnings.append("total_memory_below_preferred")
    if (
        snapshot.available_memory_bytes < MEASURED_32K_INFERENCE_ALLOCATION_BYTES
        and not accelerator_satisfies_floor
    ):
        warnings.append("available_memory_below_measured_allocation")
    if snapshot.free_storage_bytes < MINIMUM_FREE_STORAGE_BYTES:
        warnings.append("free_storage_below_recommended_minimum")
    elif snapshot.free_storage_bytes < ACQUISITION_FREE_STORAGE_BYTES:
        warnings.append("free_storage_below_acquisition_recommendation")

    if not capacity_satisfies_floor and recommended_profile_selected:
        status = _NOT_READY
        reason = "generation_resources_insufficient"
    elif warnings:
        status = _READY
        reason = "generation_resources_warning"
    else:
        status = _READY
        reason = "generation_resources_recommended"
    return _component(
        "generation_resources",
        status,
        reason,
        readiness_blocking=status == _NOT_READY,
        recommended_profile_selected=recommended_profile_selected,
        warning=bool(warnings),
        warning_codes=warnings,
        assessment_mode=(
            "attested_dedicated_accelerator"
            if snapshot.accelerator_memory_attested
            else "host_memory_conservative"
        ),
        accelerator_memory_attested=snapshot.accelerator_memory_attested,
        accelerator_memory_bytes=attested_accelerator,
        total_memory_bytes=snapshot.total_memory_bytes,
        available_memory_bytes=snapshot.available_memory_bytes,
        free_storage_bytes=snapshot.free_storage_bytes,
        measured_32k_inference_allocation_bytes=(
            MEASURED_32K_INFERENCE_ALLOCATION_BYTES
        ),
        recommended_total_memory_bytes=RECOMMENDED_TOTAL_MEMORY_BYTES,
        preferred_total_memory_bytes=PREFERRED_TOTAL_MEMORY_BYTES,
        minimum_free_storage_bytes=MINIMUM_FREE_STORAGE_BYTES,
        acquisition_free_storage_bytes=ACQUISITION_FREE_STORAGE_BYTES,
        recommended_model=RECOMMENDED_GENERATION_MODEL,
        recommended_model_digest=RECOMMENDED_GENERATION_MODEL_DIGEST,
    )


def _safe_counts(engine: Engine, table_names: set[str]) -> dict[str, int]:
    counts = {
        "pending": 0,
        "retryable": 0,
        "blocked": 0,
        "expired_query_payloads": 0,
    }
    now = datetime.now(timezone.utc)
    with engine.connect() as connection:
        sources = (
            (snapshot_continuation_job, "state"),
            (control_job, "state"),
            (baseline_run_job, "state"),
        )
        for table, state_name in sources:
            if table.name not in table_names:
                continue
            state = getattr(table.c, state_name)
            counts["pending"] += int(
                connection.scalar(
                    select(func.count())
                    .select_from(table)
                    .where(state.in_(_PENDING_STATES))
                )
                or 0
            )
            counts["retryable"] += int(
                connection.scalar(
                    select(func.count())
                    .select_from(table)
                    .where(state.in_(_RETRYABLE_STATES))
                )
                or 0
            )
            counts["blocked"] += int(
                connection.scalar(
                    select(func.count())
                    .select_from(table)
                    .where(state.in_(_BLOCKED_STATES))
                )
                or 0
            )
        if baseline_run_payload.name in table_names:
            counts["expired_query_payloads"] = int(
                connection.scalar(
                    select(func.count())
                    .select_from(baseline_run_payload)
                    .where(baseline_run_payload.c.expires_at <= now)
                )
                or 0
            )
    return counts


def _keyring_component(
    settings: Settings,
    engine: Engine,
    table_names: set[str],
) -> DoctorComponent:
    attestation = attest_keyring(settings.baseline_run_encryption_keyring)
    referenced: set[str] = set()
    expired = 0
    if baseline_run_payload.name in table_names:
        try:
            with engine.connect() as connection:
                referenced = {
                    str(value)
                    for value in connection.execute(
                        select(baseline_run_payload.c.key_id).distinct()
                    ).scalars()
                }
                expired = int(
                    connection.scalar(
                        select(func.count())
                        .select_from(baseline_run_payload)
                        .where(
                            baseline_run_payload.c.expires_at
                            <= datetime.now(timezone.utc)
                        )
                    )
                    or 0
                )
        except Exception:  # noqa: BLE001 - database details are redacted
            return _component(
                "keyring",
                _NOT_READY,
                "keyring_reference_check_failed",
                active_key_id=attestation.active_key_id,
                identity_fingerprint=attestation.identity_fingerprint,
                referenced_key_count=0,
                removed_referenced_key_count=0,
                expired_payload_count=0,
            )
    removed = referenced - set(attestation.key_ids)
    if not attestation.valid:
        status, reason = _DEGRADED, attestation.reason_code or "run_keyring_invalid"
    elif removed:
        status, reason = _NOT_READY, "run_payload_key_unavailable"
    elif expired:
        status, reason = _DEGRADED, "expired_query_payloads_present"
    else:
        status, reason = _READY, "keyring_valid"
    return _component(
        "keyring",
        status,
        reason,
        active_key_id=attestation.active_key_id,
        identity_fingerprint=attestation.identity_fingerprint,
        referenced_key_count=len(referenced),
        removed_referenced_key_count=len(removed),
        expired_payload_count=expired,
    )


def _worker_component(
    settings: Settings,
    runtime: RuntimeConfigurationAttestation | None,
    engine: Engine,
    table_names: set[str],
    pending_count: int,
) -> DoctorComponent:
    if settings.baseline_worker_mode != "database":
        return _component(
            "worker",
            _DEGRADED,
            "worker_mode_manual",
            dispatch="manual",
            healthy_workers=0,
            matching_workers=0,
            mismatched_workers=0,
            draining_workers=0,
            stale_workers=0,
            total_capacity=0,
            active_count=0,
            pending_count=pending_count,
        )
    required_tables = {
        baseline_worker_instance.name,
        baseline_worker_attestation.name,
    }
    if not required_tables <= table_names or runtime is None:
        return _component(
            "worker",
            _NOT_READY,
            "worker_schema_unavailable",
            dispatch="automatic",
            healthy_workers=0,
            matching_workers=0,
            mismatched_workers=0,
            draining_workers=0,
            stale_workers=0,
            total_capacity=0,
            active_count=0,
            pending_count=pending_count,
        )
    cutoff = datetime.now(timezone.utc) - timedelta(
        seconds=settings.baseline_worker_heartbeat_ttl_seconds
    )
    try:
        with engine.connect() as connection:
            rows = (
                connection.execute(
                    select(baseline_worker_instance, baseline_worker_attestation).join(
                        baseline_worker_attestation,
                        baseline_worker_attestation.c.worker_instance_id
                        == baseline_worker_instance.c.worker_instance_id,
                    )
                )
                .mappings()
                .all()
            )
    except Exception:  # noqa: BLE001
        rows = []
    recent = [row for row in rows if _aware(row["last_heartbeat_at"]) >= cutoff]
    stale = len(rows) - len(recent)
    mismatched = [
        row
        for row in recent
        if row["runtime_config_contract_version"] != RUNTIME_CONFIG_CONTRACT_VERSION
        or row["runtime_config_fingerprint"] != runtime.fingerprint
        or row["embedding_identity_fingerprint"]
        != runtime.embedding_identity_fingerprint
        or row["generation_identity_fingerprint"]
        != runtime.generation_identity_fingerprint
    ]
    draining = [row for row in recent if bool(row["draining"])]
    required_support = {
        "supports_corpus_ingestion",
        "supports_index_build",
        "supports_baseline_run",
        "supports_cleanup",
    }
    matching = [
        row
        for row in recent
        if row not in mismatched
        and row not in draining
        and row["worker_contract_version"] == WORKER_CONTRACT_VERSION
        and all(bool(row[name]) for name in required_support)
    ]
    capacity = sum(int(row["concurrency_limit"]) for row in matching)
    active = sum(int(row["active_count"]) for row in matching)
    maximum_pending = capacity * settings.baseline_worker_max_pending_per_slot
    if not matching:
        reason = "worker_configuration_mismatch" if mismatched else "worker_unavailable"
        status = _NOT_READY
    elif pending_count >= maximum_pending:
        reason, status = "worker_capacity_unavailable", _NOT_READY
    else:
        reason, status = "worker_ready", _READY
    return _component(
        "worker",
        status,
        reason,
        dispatch="automatic",
        worker_contract_version=WORKER_CONTRACT_VERSION,
        supported_job_types=list(WORKER_SUPPORTED_JOB_TYPES),
        healthy_workers=len(recent),
        matching_workers=len(matching),
        mismatched_workers=len(mismatched),
        draining_workers=len(draining),
        stale_workers=stale,
        total_capacity=capacity,
        active_count=active,
        pending_count=pending_count,
        maximum_pending=maximum_pending,
    )


def _disk_component(settings: Settings, engine: Engine) -> DoctorComponent:
    checks: list[tuple[str, Path, int]] = []
    if engine.dialect.name == "sqlite" and engine.url.database:
        checks.append(
            (
                "database",
                Path(str(engine.url.database)).expanduser().resolve().parent,
                _DISK_MINIMUM_DATABASE_BYTES,
            )
        )
    try:
        cache = (
            Path(settings.baseline_model_cache).expanduser()
            if settings.baseline_model_cache
            else default_cache_root()
        )
        manifest = load_baseline_model_manifest()
        checks.append(("model_cache", cache, manifest.total_bytes // 4))
    except (BaselineModelCacheError, BaselineModelManifestError, OSError):
        pass
    checks.append(("temporary", Path(tempfile.gettempdir()), _DISK_MINIMUM_TEMP_BYTES))
    statuses: dict[str, object] = {}
    insufficient = False
    for name, path, threshold in checks:
        target = (
            path
            if path.exists()
            else next(
                (parent for parent in path.parents if parent.exists()),
                Path(tempfile.gettempdir()),
            )
        )
        try:
            free = int(shutil.disk_usage(target).free)
            enough = free >= threshold
            statuses[name] = {
                "sufficient": enough,
                "free_bytes": free,
                "required_bytes": threshold,
            }
            insufficient = insufficient or not enough
        except OSError:
            statuses[name] = {
                "sufficient": False,
                "free_bytes": None,
                "required_bytes": threshold,
            }
            insufficient = True
    return _component(
        "disk",
        _DEGRADED if insufficient else _READY,
        "disk_space_insufficient" if insufficient else "disk_space_sufficient",
        roots=statuses,
    )


def _staging_component(settings: Settings) -> DoctorComponent:
    try:
        cache = (
            Path(settings.baseline_model_cache).expanduser()
            if settings.baseline_model_cache
            else default_cache_root()
        )
        staging = cache / ".staging"
        count = 0
        if staging.exists() and staging.is_dir() and not staging.is_symlink():
            count = sum(1 for item in staging.iterdir() if not item.is_symlink())
        return _component(
            "model_staging",
            _DEGRADED if count else _READY,
            "model_staging_retained" if count else "model_staging_clear",
            retained_count=count,
        )
    except (BaselineModelCacheError, OSError):
        return _component(
            "model_staging",
            _DEGRADED,
            "model_staging_unavailable",
            retained_count=0,
        )


def run_doctor(
    *,
    settings: Settings,
    engine: Engine,
    probe_generation: bool = False,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    embedding_client_factory: Callable[[], httpx.Client] | None = None,
    generation_client_factory: Callable[[], httpx.Client] | None = None,
    generation_resource_sampler: Callable[[], GenerationResourceSnapshot] | None = None,
) -> BaselineDoctorResult:
    """Inspect readiness without applying migrations or executing private inference."""

    runtime: RuntimeConfigurationAttestation | None = None
    try:
        runtime = build_runtime_configuration(settings, database_url=engine.url)
        validate_runtime_configuration(settings, database_url=engine.url)
        configuration = _component(
            "configuration",
            _READY,
            "configuration_valid",
            core_version=_safe_package_version(),
            runtime_contract_version=RUNTIME_CONFIG_CONTRACT_VERSION,
            embedding_identity_fingerprint=(runtime.embedding_identity_fingerprint),
            generation_identity_fingerprint=(runtime.generation_identity_fingerprint),
            baseline_runs_enabled=settings.baseline_runs_enabled,
            notifications_enabled=settings.baseline_notifications_enabled,
        )
    except RuntimeConfigurationError as exc:
        configuration = _component(
            "configuration",
            _NOT_READY,
            exc.code,
            core_version=_safe_package_version(),
            runtime_contract_version=RUNTIME_CONFIG_CONTRACT_VERSION,
            embedding_identity_fingerprint=(
                runtime.embedding_identity_fingerprint if runtime else None
            ),
            generation_identity_fingerprint=(
                runtime.generation_identity_fingerprint if runtime else None
            ),
            baseline_runs_enabled=settings.baseline_runs_enabled,
            notifications_enabled=settings.baseline_notifications_enabled,
        )

    database = _database_component(engine, runtime)
    migrations = (
        _migration_component(engine)
        if database.status == _READY
        else _component(
            "migrations",
            _NOT_READY,
            "migration_inspection_failed",
            expected_latest=CORE_SCHEMA_MIGRATIONS[-1].migration_id,
            applied_latest=None,
            pending_count=len(CORE_SCHEMA_MIGRATIONS),
        )
    )
    try:
        table_names = (
            set(inspect(engine).get_table_names())
            if database.status == _READY
            else set()
        )
        counts = (
            _safe_counts(engine, table_names)
            if table_names
            else {
                "pending": 0,
                "retryable": 0,
                "blocked": 0,
                "expired_query_payloads": 0,
            }
        )
    except Exception:  # noqa: BLE001
        table_names = set()
        counts = {
            "pending": 0,
            "retryable": 0,
            "blocked": 0,
            "expired_query_payloads": 0,
        }
    keyring = _keyring_component(settings, engine, table_names)
    embedding = _embedding_component(
        settings,
        runtime,
        client_factory=embedding_client_factory,
    )
    generation = _generation_component(
        settings,
        runtime,
        probe=probe_generation,
        client_factory=generation_client_factory,
    )
    generation_resources = _generation_resource_component(
        settings, generation_resource_sampler
    )
    worker = _worker_component(
        settings,
        runtime,
        engine,
        table_names,
        counts["pending"] + counts["retryable"],
    )
    workflow_ready = (
        all(
            component.status == _READY
            for component in (
                configuration,
                database,
                migrations,
                keyring,
                embedding,
                generation,
                worker,
            )
        )
        and generation_resources.status != _NOT_READY
        and settings.baseline_runs_enabled
    )
    capability = _component(
        "baseline_capability",
        _READY if workflow_ready else _DEGRADED,
        "baseline_ready" if workflow_ready else "baseline_not_ready",
        manual_readiness=(
            "ready"
            if all(
                item.status == _READY
                for item in (
                    configuration,
                    database,
                    migrations,
                    keyring,
                    embedding,
                    generation,
                )
            )
            and generation_resources.status != _NOT_READY
            and settings.baseline_runs_enabled
            else "not_ready"
        ),
        automatic_readiness="ready" if workflow_ready else "not_ready",
        baseline_runs_enabled=settings.baseline_runs_enabled,
    )
    notifications = _component(
        "notifications",
        _READY if not settings.baseline_notifications_enabled else _DEGRADED,
        (
            "notifications_default_off"
            if not settings.baseline_notifications_enabled
            else "notifications_enabled"
        ),
        enabled=settings.baseline_notifications_enabled,
        default_enabled=False,
    )
    jobs = _component(
        "jobs",
        _DEGRADED if any(counts.values()) else _READY,
        "jobs_require_attention" if any(counts.values()) else "jobs_clear",
        **counts,
    )
    model_staging = _staging_component(settings)
    disk = _disk_component(settings, engine)
    components = (
        configuration,
        database,
        migrations,
        keyring,
        embedding,
        generation,
        generation_resources,
        worker,
        capability,
        notifications,
        jobs,
        model_staging,
        disk,
    )
    if any(
        item.status == _NOT_READY
        for item in (
            configuration,
            database,
            migrations,
            keyring,
            generation_resources,
        )
    ):
        overall = _NOT_READY
    elif workflow_ready and disk.status == _READY and model_staging.status == _READY:
        overall = _READY
    else:
        overall = _DEGRADED
    actions: list[str] = []
    action_map = {
        "configuration": "correct_runtime_configuration",
        "database": "restore_database_connectivity",
        "migrations": "apply_pending_migrations_at_startup",
        "keyring": "restore_query_keyring",
        "embedding": "start_or_correct_embedding_service",
        "generation": "start_or_correct_generation_service",
        "generation_resources": "review_generation_resource_capacity",
        "worker": "start_matching_database_worker",
        "model_staging": "review_retained_model_staging",
        "disk": "increase_available_disk_space",
    }
    for component in components:
        if component.status != _READY and component.name in action_map:
            actions.append(action_map[component.name])
        if component.name == "generation_resources" and component.details.get(
            "warning"
        ):
            actions.append("review_generation_resource_capacity")
    if not settings.baseline_runs_enabled:
        actions.append("enable_baseline_runs_after_validation")
    return BaselineDoctorResult(
        status=overall,
        runtime_configuration_fingerprint=(runtime.fingerprint if runtime else None),
        components=components,
        generated_at=clock(),
        generation_probed=probe_generation,
        recommended_actions=tuple(dict.fromkeys(actions)),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compair-core",
        description="Compair Core operational commands.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    doctor = subcommands.add_parser(
        "doctor",
        help="inspect privacy-safe baseline readiness",
    )
    doctor.add_argument("--json", action="store_true", dest="json_output")
    doctor.add_argument("--require-baseline", action="store_true")
    doctor.add_argument("--probe-generation", action="store_true")
    config = subcommands.add_parser(
        "config",
        help="manage private local Core configuration",
    )
    config_commands = config.add_subparsers(dest="config_command", required=True)
    config_init = config_commands.add_parser(
        "init",
        help="create a private baseline-run keyring secrets fragment",
    )
    add_config_init_arguments(config_init)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "config" and args.config_command == "init":
        return run_config_init_command(args)
    for logger_name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    try:
        # Delay engine creation until an actual diagnostic run. Importing the
        # command (including for ``--help`` discovery) must not create a local
        # SQLite directory or initialize the legacy application.
        from .db import engine as default_engine

        result = run_doctor(
            settings=Settings(),
            engine=default_engine,
            probe_generation=bool(args.probe_generation),
        )
    except Exception:  # noqa: BLE001 - final command boundary is non-reflective
        if args.json_output:
            fallback = BaselineDoctorResult(
                status=_NOT_READY,
                runtime_configuration_fingerprint=None,
                components=(
                    _component(
                        "configuration",
                        _NOT_READY,
                        "internal_diagnostic_failure",
                    ),
                ),
                generated_at=datetime.now(timezone.utc),
                generation_probed=bool(args.probe_generation),
                recommended_actions=("review_sanitized_core_logs",),
            )
            print(json.dumps(fallback.as_dict(), sort_keys=True, separators=(",", ":")))
        else:
            print("not_ready: internal_diagnostic_failure", file=sys.stderr)
        return 7
    if args.json_output:
        print(
            json.dumps(
                result.as_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    else:
        print(
            f"{result.status}: baseline readiness diagnostics complete",
            file=sys.stderr,
        )
        for component in result.components:
            print(
                f"{component.name}: {component.status} ({component.reason_code})",
                file=sys.stderr,
            )
    return result.exit_code(require_baseline=bool(args.require_baseline))


if __name__ == "__main__":  # pragma: no cover - installed entry point
    raise SystemExit(main())


__all__ = [
    "DOCTOR_RESULT_SCHEMA_VERSION",
    "BaselineDoctorResult",
    "DoctorComponent",
    "main",
    "run_doctor",
]
