from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FEATURE_NAMES = [
    "same_document",
    "same_repo",
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
    "shared_env_var_count",
    "shared_env_var_ratio",
    "shared_identifier_count",
    "shared_identifier_ratio",
    "shared_literal_count",
    "shared_literal_ratio",
    "source_behavioral_doc",
    "candidate_implementation",
    "behavior_impl_bridge",
    "doc_code_env_bridge",
    "doc_code_identifier_bridge",
    "same_repo_doc_code_bridge",
]

EMBEDDING_SCALAR_FEATURE_NAMES = [
    "embedding_dot",
    "embedding_cosine",
    "embedding_l2",
    "embedding_mean_abs_diff",
    "embedding_max_abs_diff",
    "embedding_same_sign_fraction",
]

REFERENCE_RERANKER_MANIFEST_FORMAT = "compair_reference_reranker_manifest_v1"
REFERENCE_RERANKER_LATEST_MANIFEST_NAME = "reference_reranker_latest.json"
REFERENCE_RERANKER_LATEST_SUMMARY_NAME = "reference_reranker_latest.summary.json"
REFERENCE_RERANKER_EMBEDDING_SCHEMA_V2 = "embedding_plus_lightweight_v2"

_TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")
_SUBTOKEN_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+")
_ENV_VAR_RE = re.compile(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]{2,}\b")
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
_QUOTED_TERM_RE = re.compile(r"(?:`([^`]{2,64})`|\"([^\"]{2,64})\"|'([^']{2,64})')")
_ENDPOINT_RE = re.compile(r"\b(?:GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)\s+(/[A-Za-z0-9._~%:+/-]+)", re.IGNORECASE)
_BEHAVIOR_VERB_RE = re.compile(
    r"\b(?:use|uses|default|defaults|support|supports|require|requires|return|returns|"
    r"send|sends|deliver|delivers|emit|emits|serve|serves|expose|exposes|provide|provides|"
    r"advertise|advertises|enable|enables|disable|disables|configure|configures|"
    r"verify|verifies|allow|allows|deny|denies|include|includes|logs)\b",
    re.IGNORECASE,
)
_PUBLIC_SURFACE_RE = re.compile(
    r"\b(?:api|endpoint|route|router|capability|capabilities|config|configuration|"
    r"setting|settings|preference|preferences|backend|provider|service|worker|queue|"
    r"webhook|mailer|smtp|stdout|notification|notifications|delivery|token|oauth|auth|"
    r"license|policy|schema|field|fields|storage|billing|ocr)\b",
    re.IGNORECASE,
)
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
_LOW_SIGNAL_IDENTIFIER_TOKENS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "true",
    "false",
    "none",
    "null",
    "file",
    "path",
    "chunk",
    "document",
    "return",
    "returns",
    "class",
    "function",
    "value",
    "data",
    "item",
    "items",
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


def _subtokens(value: str) -> set[str]:
    out: set[str] = set()
    for token in _IDENTIFIER_RE.findall(value):
        lowered = token.lower()
        if len(lowered) >= 3 and lowered not in _LOW_SIGNAL_IDENTIFIER_TOKENS:
            out.add(lowered)
        parts = [part for part in token.split("_") if part] if "_" in token else _SUBTOKEN_RE.findall(token)
        for part in parts:
            lowered = part.lower()
            if len(lowered) >= 3 and lowered not in _LOW_SIGNAL_IDENTIFIER_TOKENS:
                out.add(lowered)
    return out


def _env_vars(text: str | None) -> set[str]:
    return {match.group(0) for match in _ENV_VAR_RE.finditer(text or "")}


def _quoted_terms(text: str | None) -> set[str]:
    values: set[str] = set()
    for match in _QUOTED_TERM_RE.finditer(text or ""):
        raw = next((group for group in match.groups() if group), "")
        lowered = raw.strip().lower()
        if 2 <= len(lowered) <= 64:
            values.add(lowered)
            values.update(_subtokens(lowered))
    return values


def _endpoints(text: str | None) -> set[str]:
    return {match.group(1).strip().lower() for match in _ENDPOINT_RE.finditer(text or "")}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / float(len(union))


def _bounded_count(value: int | float, *, scale: float = 1.0, cap: float = 4.0) -> float:
    return min(cap, float(value) * scale)


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


def _behavioral_doc_score(path: str | None, preview: str | None) -> float:
    if "doc" not in _path_kinds(path):
        return 0.0
    text = preview or ""
    if not text:
        return 0.0
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("### File:")]
    if not lines:
        return 0.0
    body = "\n".join(lines)
    score = 0.0
    if any(line.count("|") >= 2 for line in lines):
        score += 0.8
    if any("`" in line and ("/" in line or "=" in line or "." in line) for line in lines):
        score += 0.45
    if _BEHAVIOR_VERB_RE.search(body):
        score += 0.7
    if _PUBLIC_SURFACE_RE.search(body):
        score += 0.7
    envs = _env_vars(body)
    endpoints = _endpoints(body)
    terms = _quoted_terms(body)
    if envs:
        score += min(1.0, 0.35 * len(envs))
    if endpoints:
        score += min(1.0, 0.4 * len(endpoints))
    if terms:
        score += min(0.8, 0.08 * len(terms))
    return min(score, 4.0)


def _implementation_score(path: str | None, preview: str | None) -> float:
    kinds = _path_kinds(path)
    if "code" not in kinds:
        return 0.0
    text = preview or ""
    lower = text.lower()
    path_lower = (path or "").replace("\\", "/").lower()
    score = 0.0
    if any(
        token in path_lower
        for token in (
            "/server/",
            "/client/",
            "/provider",
            "/providers/",
            "/mailer/",
            "/api",
            "/tasks",
            "/notifications/",
            "/storage/",
            "/billing/",
        )
    ):
        score += 0.8
    if any(
        token in lower
        for token in (
            "def ",
            "class ",
            "import ",
            "from ",
            "return ",
            "@router",
            "mapped_column",
            "os.environ",
            "getenv(",
            "requests.",
            "emailer",
            "send(",
        )
    ):
        score += 1.0
    envs = _env_vars(text)
    identifiers = _subtokens(text)
    if envs:
        score += min(1.0, 0.3 * len(envs))
    if identifiers:
        score += min(0.8, 0.04 * len(identifiers))
    return min(score, 4.0)


def feature_dict_from_trace_row(row: Mapping[str, Any]) -> dict[str, float]:
    source_path = str(row.get("source_path") or "")
    candidate_path = str(row.get("candidate_path") or row.get("path") or "")
    source_repo = str(row.get("source_repo") or "")
    candidate_repo = str(row.get("candidate_repo") or "")
    source_preview = str(row.get("source_preview") or "")
    candidate_preview = str(row.get("candidate_preview") or "")
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
    source_envs = _env_vars(source_preview)
    candidate_envs = _env_vars(candidate_preview)
    source_identifiers = _subtokens(source_preview)
    candidate_identifiers = _subtokens(candidate_preview)
    source_literals = _quoted_terms(source_preview) | _endpoints(source_preview)
    candidate_literals = _quoted_terms(candidate_preview) | _endpoints(candidate_preview)
    shared_envs = source_envs & candidate_envs
    shared_identifiers = source_identifiers & candidate_identifiers
    shared_literals = source_literals & candidate_literals
    source_behavioral = _behavioral_doc_score(source_path, source_preview)
    candidate_impl = _implementation_score(candidate_path, candidate_preview)
    doc_code_pair = 1.0 if ("doc" in source_kinds and "code" in candidate_kinds) or ("code" in source_kinds and "doc" in candidate_kinds) else 0.0
    same_document = bool(row.get("same_document"))
    same_repo = bool(source_repo and source_repo == candidate_repo) or same_document
    shared_identifier_ratio = _jaccard(source_identifiers, candidate_identifiers)
    shared_literal_ratio = _jaccard(source_literals, candidate_literals)
    shared_env_ratio = _jaccard({env.lower() for env in source_envs}, {env.lower() for env in candidate_envs})
    behavior_impl_bridge = min(source_behavioral, candidate_impl)

    return {
        "same_document": 1.0 if same_document else 0.0,
        "same_repo": 1.0 if same_repo else 0.0,
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
        "doc_code_pair": doc_code_pair,
        "doc_config_pair": 1.0 if ("doc" in source_kinds and {"config", "manifest", "workflow"} & candidate_kinds) or ("doc" in candidate_kinds and {"config", "manifest", "workflow"} & source_kinds) else 0.0,
        "config_code_pair": 1.0 if ("config" in source_kinds and "code" in candidate_kinds) or ("code" in source_kinds and "config" in candidate_kinds) else 0.0,
        "manifest_legal_pair": 1.0 if ("manifest" in source_kinds and "legal" in candidate_kinds) or ("legal" in source_kinds and "manifest" in candidate_kinds) else 0.0,
        "workflow_doc_pair": 1.0 if ("workflow" in source_kinds and "doc" in candidate_kinds) or ("doc" in source_kinds and "workflow" in candidate_kinds) else 0.0,
        "shared_env_var_count": _bounded_count(len(shared_envs), scale=1.0, cap=4.0),
        "shared_env_var_ratio": shared_env_ratio,
        "shared_identifier_count": _bounded_count(len(shared_identifiers), scale=0.2, cap=4.0),
        "shared_identifier_ratio": shared_identifier_ratio,
        "shared_literal_count": _bounded_count(len(shared_literals), scale=0.35, cap=4.0),
        "shared_literal_ratio": shared_literal_ratio,
        "source_behavioral_doc": source_behavioral,
        "candidate_implementation": candidate_impl,
        "behavior_impl_bridge": behavior_impl_bridge,
        "doc_code_env_bridge": doc_code_pair * min(1.0, shared_env_ratio + (0.25 if shared_envs else 0.0)),
        "doc_code_identifier_bridge": doc_code_pair * min(1.5, shared_identifier_ratio + (0.15 * min(len(shared_identifiers), 4))),
        "same_repo_doc_code_bridge": (1.0 if same_repo else 0.0) * doc_code_pair * min(2.0, behavior_impl_bridge + shared_identifier_ratio + (0.5 * shared_literal_ratio)),
    }


def feature_vector_from_trace_row(row: Mapping[str, Any], feature_names: list[str] | None = None) -> list[float]:
    features = feature_dict_from_trace_row(row)
    names = feature_names or FEATURE_NAMES
    return [features.get(name, 0.0) for name in names]


def _coerce_embedding(values: Sequence[float] | Iterable[float] | None) -> list[float]:
    if not values:
        return []
    return [float(value) for value in values]


def embedding_scalar_features(source_embedding: Sequence[float], candidate_embedding: Sequence[float]) -> dict[str, float]:
    source = _coerce_embedding(source_embedding)
    candidate = _coerce_embedding(candidate_embedding)
    if len(source) != len(candidate):
        raise ValueError(f"Embedding length mismatch: {len(source)} vs {len(candidate)}")
    if not source:
        return {name: 0.0 for name in EMBEDDING_SCALAR_FEATURE_NAMES}

    dot = 0.0
    source_sq = 0.0
    candidate_sq = 0.0
    squared_diff = 0.0
    abs_diff_total = 0.0
    max_abs_diff = 0.0
    same_sign = 0

    for left, right in zip(source, candidate):
        dot += left * right
        source_sq += left * left
        candidate_sq += right * right
        diff = left - right
        squared_diff += diff * diff
        abs_diff = abs(diff)
        abs_diff_total += abs_diff
        if abs_diff > max_abs_diff:
            max_abs_diff = abs_diff
        if (left == 0.0 and right == 0.0) or (left > 0.0 and right > 0.0) or (left < 0.0 and right < 0.0):
            same_sign += 1

    source_norm = math.sqrt(source_sq)
    candidate_norm = math.sqrt(candidate_sq)
    cosine = dot / (source_norm * candidate_norm) if source_norm > 0.0 and candidate_norm > 0.0 else 0.0
    return {
        "embedding_dot": dot,
        "embedding_cosine": cosine,
        "embedding_l2": math.sqrt(squared_diff),
        "embedding_mean_abs_diff": abs_diff_total / float(len(source)),
        "embedding_max_abs_diff": max_abs_diff,
        "embedding_same_sign_fraction": same_sign / float(len(source)),
    }


def combined_feature_names(embedding_dim: int) -> list[str]:
    names = list(FEATURE_NAMES)
    names.extend(EMBEDDING_SCALAR_FEATURE_NAMES)
    names.extend(f"source_emb_{idx}" for idx in range(embedding_dim))
    names.extend(f"candidate_emb_{idx}" for idx in range(embedding_dim))
    names.extend(f"pair_abs_diff_{idx}" for idx in range(embedding_dim))
    return names


def combined_feature_vector_from_trace_row(row: Mapping[str, Any]) -> list[float] | None:
    source_embedding = _coerce_embedding(row.get("source_embedding"))
    candidate_embedding = _coerce_embedding(row.get("candidate_embedding"))
    if not source_embedding or not candidate_embedding:
        return None
    if len(source_embedding) != len(candidate_embedding):
        return None

    scalars = embedding_scalar_features(source_embedding, candidate_embedding)
    vector = feature_vector_from_trace_row(row, feature_names=list(FEATURE_NAMES))
    vector.extend(scalars[name] for name in EMBEDDING_SCALAR_FEATURE_NAMES)
    vector.extend(source_embedding)
    vector.extend(candidate_embedding)
    vector.extend(abs(left - right) for left, right in zip(source_embedding, candidate_embedding))
    return vector


def _import_xgboost() -> Any:
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "xgboost is required to use embedding-aware reference reranker models."
        ) from exc
    return xgb


def _load_manifest(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    artifact_name = str(payload.get("artifact_path") or "").strip()
    if not artifact_name:
        raise ValueError(f"Reranker manifest at {path} is missing artifact_path")
    artifact_path = (path.parent / artifact_name).resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"Reranker artifact referenced by {path} is missing: {artifact_path}")
    model = load_model(artifact_path)
    if isinstance(model, dict):
        resolved = dict(model)
        resolved["model_version"] = str(payload.get("model_version") or resolved.get("model_version") or "unknown")
        resolved["feature_schema"] = str(payload.get("feature_schema") or resolved.get("feature_schema") or REFERENCE_RERANKER_EMBEDDING_SCHEMA_V2)
        resolved["manifest_path"] = str(path)
        if payload.get("summary_path"):
            resolved["summary_path"] = str((path.parent / str(payload["summary_path"])).resolve())
        return resolved
    raise TypeError(f"Unsupported reranker model payload loaded from {artifact_path}")


def _load_xgboost_model(path: Path) -> dict[str, Any]:
    xgb = _import_xgboost()
    booster = xgb.Booster()
    booster.load_model(str(path))
    return {
        "model_kind": "xgboost",
        "feature_schema": REFERENCE_RERANKER_EMBEDDING_SCHEMA_V2,
        "model_version": path.stem,
        "model_path": str(path),
        "booster": booster,
    }


def load_model(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if resolved.is_dir():
        latest_manifest = resolved / REFERENCE_RERANKER_LATEST_MANIFEST_NAME
        if latest_manifest.exists():
            return load_model(latest_manifest)
        raise FileNotFoundError(f"No {REFERENCE_RERANKER_LATEST_MANIFEST_NAME} found in {resolved}")

    payload = json.loads(resolved.read_text())
    if isinstance(payload, dict) and payload.get("model_format") == REFERENCE_RERANKER_MANIFEST_FORMAT:
        return _load_manifest(resolved, payload)
    if isinstance(payload, dict) and ("weights" in payload or "bias" in payload):
        model = dict(payload)
        model.setdefault("model_kind", "linear")
        model.setdefault("model_version", resolved.stem)
        model.setdefault("model_path", str(resolved))
        return model
    if isinstance(payload, dict) and "learner" in payload and "version" in payload:
        return _load_xgboost_model(resolved)
    raise ValueError(f"Unrecognized reference reranker model format at {resolved}")


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


def _score_xgboost_trace_row(row: Mapping[str, Any], model: Mapping[str, Any]) -> float:
    vector = combined_feature_vector_from_trace_row(row)
    if vector is None:
        return 0.0
    booster = model.get("booster")
    if booster is None:
        raise ValueError("XGBoost reranker model is missing booster")
    xgb = _import_xgboost()
    prediction = booster.predict(xgb.DMatrix([vector]))
    return float(prediction[0]) if len(prediction) else 0.0


def score_trace_row(row: Mapping[str, Any], model: Mapping[str, Any]) -> float:
    model_kind = str(model.get("model_kind") or "linear").lower()
    if model_kind == "xgboost":
        return _score_xgboost_trace_row(row, model)
    feature_names = model.get("feature_names")
    return score_feature_vector(feature_vector_from_trace_row(row, feature_names=feature_names), model)
