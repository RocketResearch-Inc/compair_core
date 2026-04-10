from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


FEATURE_NAMES = [
    "same_document",
    "vector_rank_inv",
    "lexical_rank_inv",
    "anchor_rank_inv",
    "lexical_score",
    "path_theme_score",
    "path_score",
    "artifact_score",
    "anchor_overlap",
    "anchor_conflict",
    "combined_signal",
    "source_public_surface",
    "candidate_public_surface",
    "public_surface_pair_min",
    "path_token_overlap",
    "basename_token_overlap",
    "same_extension",
    "same_basename",
    "doc_code_pair",
    "doc_config_pair",
    "config_code_pair",
    "manifest_legal_pair",
    "workflow_doc_pair",
]

_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_LEGAL_BASENAMES = {"license", "copying", "notice"}
_MANIFEST_BASENAMES = {
    "pyproject.toml",
    "package.json",
    "package-lock.json",
    "cargo.toml",
    "go.mod",
    "go.sum",
    "requirements.txt",
    "pipfile",
    "pipfile.lock",
    "poetry.lock",
    "composer.json",
    "gemfile",
    "mix.exs",
}
_CONFIG_EXTENSIONS = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf", ".env"}
_DOC_EXTENSIONS = {".md", ".rst", ".adoc", ".txt"}
_LOW_SIGNAL_PATH_TOKENS = {
    "src",
    "lib",
    "dist",
    "build",
    "internal",
    "public",
    "private",
    "main",
    "index",
    "compair",
}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _reciprocal_rank(value: Any) -> float:
    rank = _as_float(value)
    if rank <= 0.0:
        return 0.0
    return 1.0 / rank


def _path_tokens(path: str | None) -> set[str]:
    raw = (path or "").replace("\\", "/").lower()
    return {
        token
        for token in _TOKEN_SPLIT_RE.split(raw)
        if len(token) >= 2 and token not in _LOW_SIGNAL_PATH_TOKENS
    }


def _basename_tokens(path: str | None) -> set[str]:
    name = Path((path or "").replace("\\", "/")).name.lower()
    return {
        token
        for token in _TOKEN_SPLIT_RE.split(name)
        if len(token) >= 2 and token not in _LOW_SIGNAL_PATH_TOKENS
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / float(len(union))


def _path_kinds(path: str | None) -> set[str]:
    raw = (path or "").replace("\\", "/")
    lower = raw.lower()
    name = Path(lower).name
    suffix = Path(lower).suffix
    kinds: set[str] = set()

    if name in _LEGAL_BASENAMES:
        kinds.add("legal")
    if name in _MANIFEST_BASENAMES or name.startswith("dockerfile"):
        kinds.add("manifest")
    if ".github/workflows/" in lower or name in {"render.yaml", "render.yml", "docker-compose.yml", "docker-compose.yaml"}:
        kinds.add("workflow")
    if "/docs/" in lower or name == "readme.md" or suffix in _DOC_EXTENSIONS:
        kinds.add("doc")
    if suffix in _CONFIG_EXTENSIONS or any(token in lower for token in ("/config", "/settings", "_config", ".env")):
        kinds.add("config")

    if not kinds:
        kinds.add("code")
    elif kinds.isdisjoint({"doc", "config", "workflow", "manifest", "legal"}):
        kinds.add("code")
    return kinds


def _public_surface_score(path: str | None) -> float:
    kinds = _path_kinds(path)
    score = 0.0
    if "doc" in kinds:
        score += 1.0
    if "legal" in kinds:
        score += 1.0
    if "manifest" in kinds:
        score += 0.9
    if "workflow" in kinds:
        score += 0.8
    if "config" in kinds:
        score += 0.7
    if "code" in kinds:
        score += 0.2
    return min(score, 2.0)


def feature_dict_from_trace_row(row: Mapping[str, Any]) -> dict[str, float]:
    source_path = str(row.get("source_path") or "")
    candidate_path = str(row.get("candidate_path") or row.get("path") or "")
    source_kinds = _path_kinds(source_path)
    candidate_kinds = _path_kinds(candidate_path)
    source_public = _public_surface_score(source_path)
    candidate_public = _public_surface_score(candidate_path)
    source_tokens = _path_tokens(source_path)
    candidate_tokens = _path_tokens(candidate_path)
    source_basename_tokens = _basename_tokens(source_path)
    candidate_basename_tokens = _basename_tokens(candidate_path)
    source_name = Path(source_path.replace("\\", "/")).name.lower()
    candidate_name = Path(candidate_path.replace("\\", "/")).name.lower()
    source_suffix = Path(source_path).suffix.lower()
    candidate_suffix = Path(candidate_path).suffix.lower()

    return {
        "same_document": 1.0 if row.get("same_document") else 0.0,
        "vector_rank_inv": _reciprocal_rank(row.get("vector_rank")),
        "lexical_rank_inv": _reciprocal_rank(row.get("lexical_rank")),
        "anchor_rank_inv": _reciprocal_rank(row.get("anchor_rank")),
        "lexical_score": _as_float(row.get("lexical_score")),
        "path_theme_score": _as_float(row.get("path_theme_score")),
        "path_score": _as_float(row.get("path_score")),
        "artifact_score": _as_float(row.get("artifact_score")),
        "anchor_overlap": _as_float(row.get("anchor_overlap")),
        "anchor_conflict": _as_float(row.get("anchor_conflict")),
        "combined_signal": _as_float(row.get("combined_signal")),
        "source_public_surface": source_public,
        "candidate_public_surface": candidate_public,
        "public_surface_pair_min": min(source_public, candidate_public),
        "path_token_overlap": _jaccard(source_tokens, candidate_tokens),
        "basename_token_overlap": _jaccard(source_basename_tokens, candidate_basename_tokens),
        "same_extension": 1.0 if source_suffix and source_suffix == candidate_suffix else 0.0,
        "same_basename": 1.0 if source_name and source_name == candidate_name else 0.0,
        "doc_code_pair": 1.0 if ("doc" in source_kinds and "code" in candidate_kinds) or ("code" in source_kinds and "doc" in candidate_kinds) else 0.0,
        "doc_config_pair": 1.0 if ("doc" in source_kinds and {"config", "manifest", "workflow"} & candidate_kinds) or ("doc" in candidate_kinds and {"config", "manifest", "workflow"} & source_kinds) else 0.0,
        "config_code_pair": 1.0 if ("config" in source_kinds and "code" in candidate_kinds) or ("code" in source_kinds and "config" in candidate_kinds) else 0.0,
        "manifest_legal_pair": 1.0 if ("manifest" in source_kinds and "legal" in candidate_kinds) or ("legal" in source_kinds and "manifest" in candidate_kinds) else 0.0,
        "workflow_doc_pair": 1.0 if ("workflow" in source_kinds and "doc" in candidate_kinds) or ("doc" in source_kinds and "workflow" in candidate_kinds) else 0.0,
    }


def feature_vector_from_trace_row(row: Mapping[str, Any], feature_names: list[str] | None = None) -> list[float]:
    features = feature_dict_from_trace_row(row)
    names = feature_names or FEATURE_NAMES
    return [features[name] for name in names]


def load_model(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def score_feature_vector(vector: list[float], model: Mapping[str, Any]) -> float:
    means = [float(x) for x in model.get("feature_means", [])]
    stds = [float(x) for x in model.get("feature_stds", [])]
    weights = [float(x) for x in model.get("weights", [])]
    bias = float(model.get("bias") or 0.0)
    total = bias
    for idx, value in enumerate(vector):
        mean = means[idx] if idx < len(means) else 0.0
        std = stds[idx] if idx < len(stds) else 1.0
        weight = weights[idx] if idx < len(weights) else 0.0
        normalized = (float(value) - mean) / std if std > 0 else float(value) - mean
        total += weight * normalized
    return total


def score_trace_row(row: Mapping[str, Any], model: Mapping[str, Any]) -> float:
    feature_names = model.get("feature_names")
    return score_feature_vector(feature_vector_from_trace_row(row, feature_names=feature_names), model)
