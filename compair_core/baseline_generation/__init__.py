"""Supported baseline-only generation providers and verification tools."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .ollama import (
        OLLAMA_GENERATION_ADAPTER_CONTRACT,
        OllamaBaselineGenerationProvider,
        OllamaGenerationConfig,
        OllamaGenerationIdentity,
        OllamaGenerationReadiness,
        validate_baseline_generation_endpoint,
        verify_ollama_generation,
    )

__all__ = [
    "OLLAMA_GENERATION_ADAPTER_CONTRACT",
    "OllamaBaselineGenerationProvider",
    "OllamaGenerationConfig",
    "OllamaGenerationIdentity",
    "OllamaGenerationReadiness",
    "validate_baseline_generation_endpoint",
    "verify_ollama_generation",
]


def __getattr__(name: str) -> Any:
    """Load provider exports only when callers request the provider itself."""

    if name not in __all__:
        raise AttributeError(name)
    value = getattr(import_module(".ollama", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
