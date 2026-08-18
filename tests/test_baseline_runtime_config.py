from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

from sqlalchemy.engine import make_url

from compair_core.runtime_config import (
    BASELINE_GENERATION_OUTPUT_SCHEMA_SHA256,
    BASELINE_GENERATION_OUTPUT_SPEC_SHA256,
    CONTROL_PLANE_V1_SHA256,
    CONTROL_PLANE_V2_SHA256,
    RUNTIME_CONFIG_CONTRACT_VERSION,
    attest_keyring,
    build_runtime_configuration,
    database_identity,
)
from compair_core.server.settings import Settings


def _keyring(*, active: str = "primary", primary: bytes = b"p" * 32) -> str:
    return json.dumps(
        {
            "version": "baseline-run-keyring.v1",
            "active_key_id": active,
            "keys": [
                {
                    "key_id": "primary",
                    "key_base64": base64.b64encode(primary).decode("ascii"),
                },
                {
                    "key_id": "rotated",
                    "key_base64": base64.b64encode(b"r" * 32).decode("ascii"),
                },
            ],
        },
        separators=(",", ":"),
    )


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "retrieval_engine": "baseline_v1",
        "baseline_runs_enabled": True,
        "baseline_worker_mode": "database",
        "baseline_embedding_provider": "http",
        "baseline_embedding_endpoint": "http://127.0.0.1:9010",
        "baseline_embedding_revision": ("52398278842ec682c6f32300af41344b1c0b0bb2"),
        "baseline_embedding_allow_insecure_loopback": True,
        "baseline_generation_provider": "ollama",
        "baseline_generation_endpoint": "http://127.0.0.1:11434",
        "baseline_generation_model": "qwen3:1.7b",
        "baseline_generation_model_digest": "sha256:" + "a" * 64,
        "baseline_generation_allow_loopback_http": True,
        "baseline_run_encryption_keyring": _keyring(),
    }
    values.update(overrides)
    return Settings(**values)


def test_runtime_configuration_is_canonical_and_deterministic() -> None:
    settings = _settings()
    first = build_runtime_configuration(
        settings,
        database_url="sqlite:////private/example/core.db",
    )
    second = build_runtime_configuration(
        _settings(),
        database_url=make_url("sqlite:////private/example/core.db"),
    )
    assert first == second
    assert first.contract_version == RUNTIME_CONFIG_CONTRACT_VERSION
    assert len(first.fingerprint) == 64
    assert first.safe_summary() == second.safe_summary()


def test_runtime_protocol_pins_match_the_frozen_artifacts() -> None:
    root = Path(__file__).parents[1]
    artifacts = {
        root / "protocol/baseline-control-plane.v1.md": CONTROL_PLANE_V1_SHA256,
        root / "protocol/baseline-control-plane.v2.md": CONTROL_PLANE_V2_SHA256,
        root / "protocol/baseline-generation-output.v2.md": (
            BASELINE_GENERATION_OUTPUT_SPEC_SHA256
        ),
        root
        / "compair_core/baseline_generation/baseline-generation-output.v2.schema.json": (
            BASELINE_GENERATION_OUTPUT_SCHEMA_SHA256
        ),
    }
    for path, expected in artifacts.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_key_value_change_changes_private_identity_without_disclosure() -> None:
    original = attest_keyring(_keyring())
    changed = attest_keyring(_keyring(primary=b"x" * 32))
    rotated = attest_keyring(_keyring(active="rotated"))
    assert original.valid and changed.valid and rotated.valid
    assert original.identity_fingerprint != changed.identity_fingerprint
    assert original.identity_fingerprint == rotated.identity_fingerprint
    assert original.active_key_id == "primary"
    assert rotated.active_key_id == "rotated"
    serialized = json.dumps(
        {
            "original": original.identity_fingerprint,
            "changed": changed.identity_fingerprint,
        }
    )
    assert base64.b64encode(b"p" * 32).decode("ascii") not in serialized
    assert base64.b64encode(b"x" * 32).decode("ascii") not in serialized


def test_endpoint_and_database_normalization_hashes_without_disclosure() -> None:
    first = build_runtime_configuration(
        _settings(baseline_embedding_endpoint="https://EXAMPLE.invalid:443/v1/"),
        database_url=(
            "postgresql+psycopg2://alice:first@example.invalid:5432/private_db"
            "?sslmode=require"
        ),
    )
    second = build_runtime_configuration(
        _settings(baseline_embedding_endpoint="https://example.invalid/v1"),
        database_url=(
            "postgresql+psycopg2://bob:second@example.invalid/private_db"
            "?sslmode=require"
        ),
    )
    assert first.database_identity_fingerprint == second.database_identity_fingerprint
    assert first.fingerprint == second.fingerprint
    rendered = json.dumps(first.safe_summary(), sort_keys=True)
    for prohibited in (
        "alice",
        "first",
        "bob",
        "second",
        "example.invalid",
        "private_db",
        "https://",
    ):
        assert prohibited not in rendered


def test_database_identity_changes_for_distinct_database_without_credentials() -> None:
    first = database_identity(
        "postgresql+psycopg2://one:secret@127.0.0.1/first?sslmode=require"
    )
    second = database_identity(
        "postgresql+psycopg2://two:different@127.0.0.1/second?sslmode=require"
    )
    same = database_identity(
        "postgresql+psycopg2://other:changed@127.0.0.1/first?sslmode=require"
    )
    assert first["identity_sha256"] != second["identity_sha256"]
    assert first["identity_sha256"] == same["identity_sha256"]
    assert "secret" not in repr(first)


def test_api_and_worker_effective_settings_match_and_drift_is_visible() -> None:
    api = build_runtime_configuration(
        _settings(),
        database_url="sqlite:////private/example/core.db",
    )
    worker = build_runtime_configuration(
        _settings(),
        database_url="sqlite:////private/example/core.db",
    )
    drifted = build_runtime_configuration(
        _settings(baseline_generation_seed=42),
        database_url="sqlite:////private/example/core.db",
    )
    assert api.fingerprint == worker.fingerprint
    assert api.fingerprint != drifted.fingerprint
    assert (
        api.generation_identity_fingerprint == drifted.generation_identity_fingerprint
    )
