"""Installed, fail-closed local embedding support for ``baseline_v1``."""

from .manifest import (
    BaselineModelArtifact,
    BaselineModelManifest,
    load_baseline_model_manifest,
)

__all__ = [
    "BaselineModelArtifact",
    "BaselineModelManifest",
    "load_baseline_model_manifest",
]
