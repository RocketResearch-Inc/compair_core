"""Supported recommended local baseline-generation profile.

These values describe an operator-selected profile only.  Core keeps baseline
generation disabled by default, never downloads model weights, and still
attests the exact configured Ollama tag and immutable digest before use.
"""

from __future__ import annotations

import math

RECOMMENDED_GENERATION_MODEL = "qwen3:14b"
RECOMMENDED_GENERATION_MODEL_DIGEST = (
    "sha256:bdbd181c33f2ed1b31c972991882db3cf4d192569092138a7d29e973cd9debe8"
)
RECOMMENDED_GENERATION_QUANTIZATION = "Q4_K_M"
QUALIFIED_OLLAMA_RUNTIME_VERSION = "0.32.14"
QUALIFIED_CONTEXT_TOKENS = 32_768
QUALIFIED_OUTPUT_TOKENS = 1_024
QUALIFIED_BUDGET_PROFILE_ASSET_SHA256 = (
    "af00a090678da236d35203b01cdb929543e30bdcbc59749efe60e8ad20e1a284"
)
QUALIFIED_BUDGET_PROFILE_FINGERPRINT = (
    "69ccd81f6ba8e62a34961559390c170879315431ba58a96cf99ba90ac035bda9"
)

# Sixty seconds remains the compatibility/default profile for accelerated
# deployments.  CPU-only operators should explicitly select the qualified
# bounded profile documented in the runbook.
ACCELERATED_GENERATION_TIMEOUT_SECONDS = 60.0
CPU_GENERATION_TIMEOUT_SECONDS = 300.0
MAXIMUM_GENERATION_TIMEOUT_SECONDS = 300.0

# Existing generation jobs historically used a five-minute lease.  Preserve
# that floor and extend it when the configured provider timeout would otherwise
# leave no time for response validation and the atomic Feedback commit.
MINIMUM_GENERATION_LEASE_SECONDS = 300
GENERATION_LEASE_COMMIT_MARGIN_SECONDS = 60

GIB = 1024**3
MEASURED_32K_INFERENCE_ALLOCATION_BYTES = 15 * GIB
MINIMUM_GENERATION_CAPACITY_BYTES = 16 * GIB
RECOMMENDED_TOTAL_MEMORY_BYTES = 24 * GIB
PREFERRED_TOTAL_MEMORY_BYTES = 32 * GIB
MINIMUM_FREE_STORAGE_BYTES = 25 * GIB
ACQUISITION_FREE_STORAGE_BYTES = 40 * GIB


def required_generation_lease_seconds(provider_timeout_seconds: float) -> int:
    """Return a lease that covers one provider call plus a commit margin."""

    timeout = float(provider_timeout_seconds)
    if (
        not math.isfinite(timeout)
        or timeout < 0.1
        or timeout > MAXIMUM_GENERATION_TIMEOUT_SECONDS
    ):
        raise ValueError("provider_timeout_seconds is outside the supported range")
    return max(
        MINIMUM_GENERATION_LEASE_SECONDS,
        math.ceil(timeout) + GENERATION_LEASE_COMMIT_MARGIN_SECONDS,
    )


def validate_generation_timeout_lease(
    provider_timeout_seconds: float,
    lease_seconds: int,
) -> None:
    """Reject a lease that could expire during an authorized provider call."""

    required = required_generation_lease_seconds(provider_timeout_seconds)
    if lease_seconds < required:
        raise ValueError("generation lease is shorter than the provider safety bound")


__all__ = [
    "ACCELERATED_GENERATION_TIMEOUT_SECONDS",
    "ACQUISITION_FREE_STORAGE_BYTES",
    "CPU_GENERATION_TIMEOUT_SECONDS",
    "GENERATION_LEASE_COMMIT_MARGIN_SECONDS",
    "GIB",
    "MAXIMUM_GENERATION_TIMEOUT_SECONDS",
    "MEASURED_32K_INFERENCE_ALLOCATION_BYTES",
    "MINIMUM_FREE_STORAGE_BYTES",
    "MINIMUM_GENERATION_CAPACITY_BYTES",
    "MINIMUM_GENERATION_LEASE_SECONDS",
    "PREFERRED_TOTAL_MEMORY_BYTES",
    "QUALIFIED_BUDGET_PROFILE_ASSET_SHA256",
    "QUALIFIED_BUDGET_PROFILE_FINGERPRINT",
    "QUALIFIED_CONTEXT_TOKENS",
    "QUALIFIED_OLLAMA_RUNTIME_VERSION",
    "QUALIFIED_OUTPUT_TOKENS",
    "RECOMMENDED_GENERATION_MODEL",
    "RECOMMENDED_GENERATION_MODEL_DIGEST",
    "RECOMMENDED_GENERATION_QUANTIZATION",
    "RECOMMENDED_TOTAL_MEMORY_BYTES",
    "required_generation_lease_seconds",
    "validate_generation_timeout_lease",
]
