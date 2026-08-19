from __future__ import annotations

from pathlib import Path

import pytest

from compair_core.baseline_generation.profile import (
    ACCELERATED_GENERATION_TIMEOUT_SECONDS,
    CPU_GENERATION_TIMEOUT_SECONDS,
    GENERATION_LEASE_COMMIT_MARGIN_SECONDS,
    QUALIFIED_CONTEXT_TOKENS,
    QUALIFIED_OUTPUT_TOKENS,
    RECOMMENDED_GENERATION_MODEL,
    RECOMMENDED_GENERATION_MODEL_DIGEST,
    RECOMMENDED_GENERATION_QUANTIZATION,
    required_generation_lease_seconds,
)
from compair_core.compair.retrieval.generation import BaselineGenerationService
from compair_core.runtime_config import build_runtime_configuration
from compair_core.server.settings import Settings

ROOT = Path(__file__).resolve().parents[1]


def test_recommended_qwen_identity_and_fail_closed_defaults() -> None:
    settings = Settings()
    assert settings.baseline_generation_provider == "disabled"
    assert settings.retrieval_engine == "legacy"
    assert settings.baseline_generation_model == RECOMMENDED_GENERATION_MODEL
    assert (
        settings.baseline_generation_model_digest == RECOMMENDED_GENERATION_MODEL_DIGEST
    )
    assert RECOMMENDED_GENERATION_MODEL == "qwen3:14b"
    assert RECOMMENDED_GENERATION_QUANTIZATION == "Q4_K_M"
    assert RECOMMENDED_GENERATION_MODEL_DIGEST == (
        "sha256:bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8"
    )
    assert settings.baseline_generation_context_tokens == QUALIFIED_CONTEXT_TOKENS
    assert settings.baseline_generation_output_tokens == QUALIFIED_OUTPUT_TOKENS
    assert (
        settings.baseline_generation_timeout_seconds
        == ACCELERATED_GENERATION_TIMEOUT_SECONDS
    )


def test_cpu_timeout_derives_lease_with_commit_margin() -> None:
    assert required_generation_lease_seconds(60) == 300
    assert required_generation_lease_seconds(CPU_GENERATION_TIMEOUT_SECONDS) == 360
    assert GENERATION_LEASE_COMMIT_MARGIN_SECONDS == 60
    service = BaselineGenerationService(
        object(),
        lease_seconds=360,
        provider_timeout_seconds=CPU_GENERATION_TIMEOUT_SECONDS,
    )
    assert service.lease_seconds == 360
    with pytest.raises(ValueError, match="shorter than the provider safety bound"):
        BaselineGenerationService(
            object(),
            lease_seconds=359,
            provider_timeout_seconds=CPU_GENERATION_TIMEOUT_SECONDS,
        )
    for timeout in (0, 300.1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            required_generation_lease_seconds(timeout)


def test_runtime_attestation_records_cpu_timeout_and_internal_lease() -> None:
    settings = Settings(baseline_generation_timeout_seconds=300)
    runtime = build_runtime_configuration(settings, database_url="sqlite:///fixture.db")
    limits = runtime.canonical_configuration["limits"]
    assert limits["generation_timeout_seconds"] == 300
    assert limits["generation_lease_seconds"] == 360
    assert limits["generation_lease_commit_margin_seconds"] == 60


def test_operator_examples_pin_qualified_identity_without_enabling_provider() -> None:
    environment = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "COMPAIR_BASELINE_GENERATION_PROVIDER=disabled" in environment
    assert f"COMPAIR_BASELINE_GENERATION_MODEL={RECOMMENDED_GENERATION_MODEL}" in (
        environment
    )
    assert (
        "COMPAIR_BASELINE_GENERATION_MODEL_DIGEST="
        f"{RECOMMENDED_GENERATION_MODEL_DIGEST}"
    ) in environment
    assert "COMPAIR_BASELINE_GENERATION_TIMEOUT_SECONDS=300" in environment
    for path in (
        ROOT / "README.md",
        ROOT / "docs/baseline-ollama-generation.md",
        ROOT / "docs/baseline-local-self-host-runbook-draft.md",
        ROOT / "docs/baseline-local-self-host-readiness.md",
    ):
        content = path.read_text(encoding="utf-8")
        assert RECOMMENDED_GENERATION_MODEL in content
        assert RECOMMENDED_GENERATION_MODEL_DIGEST in content
