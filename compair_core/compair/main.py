from __future__ import annotations

from dataclasses import dataclass
import difflib
from functools import lru_cache
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, Optional

import Levenshtein
from sqlalchemy import or_, select
from sqlalchemy.orm.attributes import get_history
from sqlalchemy.orm import Session as SASession

from .embeddings import create_embedding, create_embeddings, Embedder
from .feedback import get_feedback, Reviewer, split_feedback_items
from .logger import log_event
from .local_summary import extract_artifacts
from .models import (
    Chunk,
    Document,
    Feedback,
    Group,
    Note,
    Reference,
    User,
    VECTOR_BACKEND,
    cosine_similarity,
)
from .reference_reranker import load_model as load_reference_reranker_model, score_trace_row as score_reference_trace_row
from .topic_tags import extract_topic_tags
from .utils import (
    chunk_text_with_mode,
    count_tokens,
    log_activity,
    stable_chunk_hash,
)


_CODE_REPO_DOC_TYPE = "code-repo"
_HIGH_SIGNAL_PATH_HINTS = (
    "/api/",
    "/auth/",
    "/route",
    "/router",
    "/server/",
    "/settings",
    "/config",
    "/billing",
    "/payment",
    "/notification",
    "/sync",
    "/main.",
    "/app.",
)
_HIGH_SIGNAL_METADATA_BASENAMES = {
    "pyproject.toml",
    "package.json",
    "go.mod",
    "cargo.toml",
    "setup.py",
    "setup.cfg",
    "license",
    "license.txt",
    "copying",
    "copying.txt",
    "notice",
    "notice.txt",
}
_REFERENCE_RERANKER_DEFAULT_MODEL_PATH = "/opt/compair/reference_reranker.json"
_REFERENCE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_REFERENCE_SUBTOKEN_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+")
_HTTP_METHOD_PATH_RE = re.compile(
    r"\b(GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD)\s+((?:/[A-Za-z0-9._~%:+-]+)+(?:/[A-Za-z0-9._~%:+-]*)?)",
    re.IGNORECASE,
)
_ENV_VAR_RE = re.compile(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]{2,}\b")
_LICENSE_TERM_RE = re.compile(
    r"\b(?:"
    r"MIT|"
    r"Apache(?:[- ]?2(?:\.0)?)?|"
    r"BSD(?:[- ]?(?:2|3)(?:[- ]Clause)?)?|"
    r"GPL(?:[- ]?(?:v)?(?:2|3)(?:\.0)?)?|"
    r"AGPL(?:[- ]?(?:v)?(?:3)(?:\.0)?)?|"
    r"LGPL(?:[- ]?(?:v)?(?:2|3)(?:\.0)?)?|"
    r"MPL(?:[- ]?2(?:\.0)?)?|"
    r"ISC|"
    r"Proprietary|"
    r"GNU General Public License|"
    r"GNU Affero General Public License|"
    r"Mozilla Public License"
    r")\b",
    re.IGNORECASE,
)
_BEHAVIORAL_CLAIM_VERB_RE = re.compile(
    r"\b(?:"
    r"use|uses|used|"
    r"default|defaults|defaults? to|"
    r"support|supports|supported|"
    r"require|requires|required|"
    r"return|returns|returned|"
    r"write|writes|written|"
    r"send|sends|sent|"
    r"emit|emits|emitted|"
    r"serve|serves|served|"
    r"expose|exposes|exposed|"
    r"provide|provides|provided|"
    r"advertise|advertises|advertised|"
    r"enable|enables|enabled|"
    r"disable|disables|disabled|"
    r"configure|configures|configured|"
    r"persist|persists|persisted|"
    r"store|stores|stored|"
    r"route|routes|routed|"
    r"map|maps|mapped|"
    r"deliver|delivers|delivered|"
    r"verify|verifies|verified|"
    r"allow|allows|allowed|"
    r"deny|denies|denied|"
    r"include|includes|included|"
    r"exclude|excludes|excluded|"
    r"available|unavailable"
    r")\b",
    re.IGNORECASE,
)
_PUBLIC_SURFACE_NOUN_RE = re.compile(
    r"\b(?:"
    r"api|endpoint|route|router|capability|capabilities|"
    r"config|configuration|setting|settings|preference|preferences|"
    r"backend|provider|service|worker|queue|webhook|mailer|smtp|stdout|"
    r"notification|notifications|delivery|token|oauth|auth|"
    r"license|policy|schema|field|fields"
    r")\b",
    re.IGNORECASE,
)
_CODEISH_IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9]+(?:[A-Z][A-Za-z0-9]+)+|[a-z]+_[a-z0-9_]{2,})\b"
)
_REFERENCE_TOKEN_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "return",
    "returns",
    "class",
    "function",
    "const",
    "true",
    "false",
    "none",
    "null",
    "json",
    "text",
    "type",
    "value",
    "data",
    "item",
    "items",
    "file",
    "path",
    "chunk",
    "document",
    "note",
}

logger = logging.getLogger(__name__)
_ENV_VAR_EXCLUDE = {
    "HTTP",
    "HTTPS",
    "JSON",
    "HTML",
    "NULL",
    "NONE",
    "TRUE",
    "FALSE",
}


@dataclass(frozen=True)
class ReferenceAnchorProfile:
    endpoint_pairs: frozenset[str]
    endpoint_paths: frozenset[str]
    methods: frozenset[str]
    env_vars: frozenset[str]
    license_terms: frozenset[str]
    key_names: frozenset[str]
    quoted_norm: frozenset[str]
    path_tokens: frozenset[str]
    basename: str


def is_code_review_document(doc: Document, chunk_mode: Optional[str]) -> bool:
    doc_type = (getattr(doc, "doc_type", "") or "").strip().lower()
    mode = (chunk_mode or "").strip().lower()
    return doc_type == _CODE_REPO_DOC_TYPE or mode in {"client", "preserve", "prechunked"}


def _extract_snapshot_file_path(chunk: str) -> str:
    match = re.search(r"^### File:\s+([^\n(]+)", chunk, re.MULTILINE)
    if not match:
        return ""
    return match.group(1).strip()


def _extract_snapshot_part(chunk: str) -> tuple[int, int] | None:
    match = re.search(r"^### File:\s+[^\n]+\(part\s+(\d+)/(\d+)\)", chunk, re.MULTILINE)
    if not match:
        return None
    try:
        return (int(match.group(1)), int(match.group(2)))
    except ValueError:
        return None


def _is_header_only_snapshot_chunk(chunk: str) -> bool:
    stripped = (chunk or "").strip()
    if not stripped.startswith("### File:"):
        return False
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) <= 1:
        return True
    body_lines = [line for line in lines[1:] if not line.startswith("```")]
    if not body_lines:
        return True
    return len(body_lines) == 1 and len(body_lines[0]) <= 24


def _chunk_priority_key(chunk: str, idx: int, code_focus: bool) -> tuple[int, int, int, int, int]:
    if not code_focus:
        return (0, 0, 0, -len(chunk), idx)

    stripped = chunk.lstrip()
    path = _extract_snapshot_file_path(chunk).lower()
    has_file_header = bool(path)
    has_code_fence = "```" in chunk

    if stripped.startswith("# Compair baseline snapshot") and not has_file_header:
        category = 3
    elif stripped.startswith("## Snapshot limits") and not has_file_header:
        category = 2
    elif has_file_header:
        category = 0
    else:
        category = 1

    path_rank = 2
    if path:
        if any(hint in path for hint in _HIGH_SIGNAL_PATH_HINTS) or _is_high_signal_metadata_path(path):
            path_rank = 0
        else:
            path_rank = 1

    fence_rank = 0 if has_code_fence else 1
    return (category, path_rank, fence_rank, -count_tokens(chunk), idx)


def _is_snapshot_metadata_chunk(chunk: str) -> bool:
    stripped = (chunk or "").lstrip()
    return stripped.startswith("# Compair baseline snapshot") or stripped.startswith("## Snapshot limits")


def _should_reanalyze_existing_chunks(*, reanalyze_existing: bool, meaningful_new_chunk_count: int) -> bool:
    return bool(reanalyze_existing and meaningful_new_chunk_count == 0)


def _is_code_review_chunk(doc: Document, text: str) -> bool:
    doc_type = (getattr(doc, "doc_type", "") or "").strip().lower()
    if doc_type == _CODE_REPO_DOC_TYPE:
        return True
    return "### File:" in (text or "") and "```" in (text or "")


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _reference_selection_config(code_focus: bool) -> tuple[int, int, int]:
    if code_focus:
        return (
            _int_env("COMPAIR_CODE_REPO_REFERENCE_CANDIDATES", 10),
            _int_env("COMPAIR_CODE_REPO_REFERENCE_LIMIT", 4),
            _int_env("COMPAIR_REFERENCE_MAX_PER_SOURCE", 2),
        )
    return (
        _int_env("COMPAIR_REFERENCE_CANDIDATES", 6),
        _int_env("COMPAIR_REFERENCE_LIMIT", 3),
        _int_env("COMPAIR_REFERENCE_MAX_PER_SOURCE", 2),
    )


def _reference_trace_enabled() -> bool:
    return _bool_env("COMPAIR_REFERENCE_TRACE", False)


def _reference_trace_max_candidates() -> int:
    raw = os.getenv("COMPAIR_REFERENCE_TRACE_MAX_CANDIDATES", "0").strip()
    try:
        value = int(raw)
    except ValueError:
        return 0
    return max(0, value)


def _reference_reranker_enabled() -> bool:
    return _bool_env("COMPAIR_REFERENCE_RERANKER_ENABLED", False)


@lru_cache(maxsize=1)
def _reference_reranker_state() -> tuple[dict[str, Any] | None, str | None]:
    if not _reference_reranker_enabled():
        return None, None
    raw_path = (os.getenv("COMPAIR_REFERENCE_RERANKER_MODEL_PATH") or "").strip()
    model_path = raw_path or _REFERENCE_RERANKER_DEFAULT_MODEL_PATH
    if not os.path.exists(model_path):
        logger.warning(
            "Reference reranker enabled but model artifact is missing at %s; falling back to heuristic ranking.",
            model_path,
        )
        return None, model_path
    try:
        model = load_reference_reranker_model(model_path)
    except Exception as exc:
        logger.warning(
            "Failed to load reference reranker model from %s; falling back to heuristic ranking: %s",
            model_path,
            exc,
        )
        return None, model_path
    logger.info(
        "Loaded reference reranker model %s from %s",
        str(model.get("version") or "unknown"),
        model_path,
    )
    return model, model_path


def _reference_reranker_score(row: Mapping[str, Any]) -> float | None:
    model, _ = _reference_reranker_state()
    if not model:
        return None
    return float(score_reference_trace_row(row, model))


def _identifier_tokens(text: str, limit: int = 64) -> set[str]:
    tokens: set[str] = set()
    for raw in _REFERENCE_TOKEN_RE.findall(text or ""):
        parts = [raw]
        parts.extend(
            subtoken
            for subtoken in _REFERENCE_SUBTOKEN_RE.findall(raw.replace("_", " "))
            if subtoken
        )
        for part in parts:
            token = part.lower()
            if len(token) < 3 or token in _REFERENCE_TOKEN_STOPWORDS:
                continue
            tokens.add(token)
            if len(tokens) >= limit:
                return tokens
    return tokens


def _path_token_set(path: str) -> set[str]:
    return {
        token
        for token in re.split(r"[/._:-]+", (path or "").lower())
        if len(token) >= 2 and token not in _REFERENCE_TOKEN_STOPWORDS
    }


def _path_overlap_score(target_path: str, candidate_path: str) -> float:
    if not target_path or not candidate_path:
        return 0.0
    target_norm = target_path.strip().lower()
    candidate_norm = candidate_path.strip().lower()
    if not target_norm or not candidate_norm:
        return 0.0
    score = 0.0
    if target_norm == candidate_norm:
        score += 3.0
    elif os.path.basename(target_norm) == os.path.basename(candidate_norm):
        score += 1.5
    shared = _path_token_set(target_norm) & _path_token_set(candidate_norm)
    if shared:
        score += min(2.0, 0.5 * len(shared))
    return score


def _token_overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / float(max(1, min(len(left), len(right))))


def _doc_body_lines(text: str) -> list[str]:
    if not text:
        return []
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("### File:"):
            continue
        lines.append(line)
    return lines


@lru_cache(maxsize=4096)
def _behavioral_doc_signal_score(text: str) -> float:
    chunk = text or ""
    path = _extract_snapshot_file_path(chunk)
    if not chunk or not _is_doc_like_path(path) or _is_snapshot_metadata_chunk(chunk):
        return 0.0

    profile = _reference_anchor_profile(chunk)
    artifacts = extract_artifacts(chunk)
    body_lines = _doc_body_lines(chunk)
    if not body_lines:
        return 0.0

    score = 0.0
    body = "\n".join(body_lines)
    if any(line.count("|") >= 2 for line in body_lines):
        score += 0.9
    if any(
        (line.startswith(("-", "*", "+")) or re.match(r"^\d+[.)]\s+", line))
        and ("`" in line or ":" in line or "=" in line)
        for line in body_lines
    ):
        score += 0.45
    if any("`" in line and ("/" in line or "=" in line or "." in line) for line in body_lines):
        score += 0.4
    if _BEHAVIORAL_CLAIM_VERB_RE.search(body):
        score += 0.7
    if _PUBLIC_SURFACE_NOUN_RE.search(body):
        score += 0.65
    if _CODEISH_IDENTIFIER_RE.search(body):
        score += 0.35

    if profile.endpoint_pairs:
        score += min(1.4, 0.8 * float(len(profile.endpoint_pairs)))
    elif profile.endpoint_paths:
        score += min(1.0, 0.45 * float(len(profile.endpoint_paths)))

    if profile.env_vars:
        score += min(1.2, 0.35 * float(len(profile.env_vars)))
    if profile.license_terms:
        score += min(1.6, 0.8 * float(len(profile.license_terms)))
    if artifacts.assignments:
        score += min(1.0, 0.22 * float(len(artifacts.assignments)))
    if artifacts.key_names:
        score += min(0.8, 0.12 * float(len(artifacts.key_names)))
    if profile.quoted_norm:
        score += min(0.8, 0.1 * float(len(profile.quoted_norm)))

    return min(score, 5.5)


@lru_cache(maxsize=4096)
def _structured_source_signal_score(text: str, *, code_focus: bool) -> float:
    if not code_focus:
        return 0.0

    chunk = text or ""
    if not chunk or _is_snapshot_metadata_chunk(chunk):
        return 0.0

    path = _extract_snapshot_file_path(chunk)
    profile = _reference_anchor_profile(chunk)
    artifacts = extract_artifacts(chunk)
    behavioral_doc_signal = _behavioral_doc_signal_score(chunk)

    score = 0.0
    if profile.endpoint_pairs:
        score += min(3.0, 1.35 * float(len(profile.endpoint_pairs)))
    elif profile.endpoint_paths:
        score += min(2.0, 0.85 * float(len(profile.endpoint_paths)))

    if profile.env_vars:
        score += min(1.8, 0.45 * float(len(profile.env_vars)))

    if profile.license_terms:
        score += min(1.8, 0.9 * float(len(profile.license_terms)))

    if artifacts.assignments:
        score += min(2.0, 0.3 * float(len(artifacts.assignments)))

    if artifacts.key_names:
        score += min(1.2, 0.15 * float(len(artifacts.key_names)))

    if profile.quoted_norm:
        score += min(1.0, 0.1 * float(len(profile.quoted_norm)))

    if _is_high_signal_metadata_path(path):
        score += 1.0
        if profile.license_terms or "license" in artifacts.key_names or os.path.basename(path).lower() in {
            "license",
            "license.txt",
            "copying",
            "copying.txt",
            "notice",
            "notice.txt",
        }:
            score += 1.25

    # Structured docs such as API guides and config references are often the
    # best source chunk for cross-repo contradictions even though they are docs.
    if _is_doc_like_path(path) and (
        profile.endpoint_pairs
        or profile.endpoint_paths
        or profile.env_vars
        or profile.license_terms
        or artifacts.assignments
    ):
        score += 1.2
    if behavioral_doc_signal > 0.0:
        score += behavioral_doc_signal

    if "```" in chunk:
        score += 0.25
    if "diff --git" in chunk or "@@" in chunk or "+++ b/" in chunk:
        score += 0.35

    return min(score, 7.5)


@lru_cache(maxsize=2048)
def _artifact_overlap_score(left_text: str, right_text: str) -> float:
    left = extract_artifacts(left_text or "")
    right = extract_artifacts(right_text or "")
    if not left.text or not right.text:
        return 0.0

    score = 0.0
    left_paths = {path.lower() for path in left.paths}
    right_paths = {path.lower() for path in right.paths}
    exact_path_overlap = left_paths & right_paths
    if exact_path_overlap:
        score += 3.0 + min(1.5, 0.5 * float(max(len(exact_path_overlap) - 1, 0)))

    path_token_overlap = left.path_tokens & right.path_tokens
    if path_token_overlap:
        score += min(2.5, 0.5 * float(len(path_token_overlap)))

    quoted_overlap = left.quoted_norm & right.quoted_norm
    if quoted_overlap:
        score += min(2.0, 0.75 * float(len(quoted_overlap)))

    key_overlap = left.key_names & right.key_names
    if key_overlap:
        score += min(1.5, 0.5 * float(len(key_overlap)))

    compound_overlap = left.compound_prefixes & right.compound_prefixes
    if compound_overlap:
        score += min(1.0, 0.25 * float(len(compound_overlap)))

    return score


def _normalize_license_term(value: str) -> str:
    lowered = (value or "").strip().lower()
    compact = lowered.replace(" ", "").replace("-", "")
    if "gnuafferogeneralpubliclicense" in compact or compact.startswith("agpl"):
        return "agpl"
    if "gnugeneralpubliclicense" in compact or compact.startswith("gpl"):
        return "gpl"
    if compact.startswith("lgpl"):
        return "lgpl"
    if lowered == "mit":
        return "mit"
    if lowered.startswith("apache"):
        return "apache"
    if lowered.startswith("bsd"):
        return "bsd"
    if lowered.startswith("mozilla public license") or lowered.startswith("mpl"):
        return "mpl"
    if lowered == "isc":
        return "isc"
    if lowered == "proprietary":
        return "proprietary"
    return lowered


def _extract_license_terms(text: str) -> frozenset[str]:
    terms = {
        _normalize_license_term(match.group(0))
        for match in _LICENSE_TERM_RE.finditer(text or "")
    }
    return frozenset(term for term in terms if term)


def _extract_env_vars(text: str) -> frozenset[str]:
    out: set[str] = set()
    for raw in _ENV_VAR_RE.findall(text or ""):
        candidate = raw.strip().upper()
        if candidate in _ENV_VAR_EXCLUDE or candidate in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            continue
        out.add(candidate)
    return frozenset(out)


@lru_cache(maxsize=2048)
def _reference_anchor_profile(text: str) -> ReferenceAnchorProfile:
    artifact_profile = extract_artifacts(text or "")
    endpoint_pairs: set[str] = set()
    endpoint_paths: set[str] = set()
    methods: set[str] = set()
    for match in _HTTP_METHOD_PATH_RE.finditer(text or ""):
        method = match.group(1).strip().lower()
        path = match.group(2).strip().lower()
        if not path:
            continue
        endpoint_pairs.add(f"{method} {path}")
        endpoint_paths.add(path)
        methods.add(method)
    for path in artifact_profile.paths:
        normalized = path.strip().lower()
        if normalized.startswith("/"):
            endpoint_paths.add(normalized)

    snapshot_path = _extract_snapshot_file_path(text).lower().replace("\\", "/")
    basename = os.path.basename(snapshot_path) if snapshot_path else ""
    return ReferenceAnchorProfile(
        endpoint_pairs=frozenset(endpoint_pairs),
        endpoint_paths=frozenset(endpoint_paths),
        methods=frozenset(methods),
        env_vars=_extract_env_vars(text or ""),
        license_terms=_extract_license_terms(text or ""),
        key_names=artifact_profile.key_names,
        quoted_norm=artifact_profile.quoted_norm,
        path_tokens=artifact_profile.path_tokens,
        basename=basename,
    )


@lru_cache(maxsize=2048)
def _assignment_contrast_score(left_text: str, right_text: str) -> float:
    left = extract_artifacts(left_text or "")
    right = extract_artifacts(right_text or "")
    if not left.assignments or not right.assignments:
        return 0.0

    right_by_key: dict[str, list] = {}
    for assignment in right.assignments:
        right_by_key.setdefault(assignment.key, []).append(assignment)

    score = 0.0
    for assignment in left.assignments:
        for peer in right_by_key.get(assignment.key, []):
            if assignment.normalized_value == peer.normalized_value:
                continue
            score += 1.6
            if assignment.paths or peer.paths:
                score += 0.7
            if assignment.value_tokens and peer.value_tokens and not (assignment.value_tokens & peer.value_tokens):
                score += 0.9
            if assignment.normalized_value in {"true", "false"} or peer.normalized_value in {"true", "false"}:
                score += 0.4
    return min(score, 4.5)


@lru_cache(maxsize=2048)
def _reference_anchor_overlap_score(left_text: str, right_text: str) -> float:
    left = _reference_anchor_profile(left_text or "")
    right = _reference_anchor_profile(right_text or "")
    score = 0.0
    shared_endpoint_pairs = left.endpoint_pairs & right.endpoint_pairs
    if shared_endpoint_pairs:
        score += 4.0 + min(1.0, 0.25 * float(max(len(shared_endpoint_pairs) - 1, 0)))
    shared_endpoint_paths = left.endpoint_paths & right.endpoint_paths
    if shared_endpoint_paths:
        score += 2.5 + min(1.0, 0.25 * float(max(len(shared_endpoint_paths) - 1, 0)))
    shared_env_vars = left.env_vars & right.env_vars
    if shared_env_vars:
        score += min(2.5, 0.9 * float(len(shared_env_vars)))
    shared_keys = left.key_names & right.key_names
    if shared_keys:
        score += min(2.0, 0.45 * float(len(shared_keys)))
    shared_quotes = left.quoted_norm & right.quoted_norm
    if shared_quotes:
        score += min(1.8, 0.6 * float(len(shared_quotes)))
    shared_licenses = left.license_terms & right.license_terms
    if shared_licenses:
        score += min(1.8, 0.9 * float(len(shared_licenses)))
    shared_path_tokens = left.path_tokens & right.path_tokens
    if shared_path_tokens:
        score += min(1.5, 0.25 * float(len(shared_path_tokens)))
    if left.basename and left.basename == right.basename:
        score += 1.0
    return score


@lru_cache(maxsize=2048)
def _reference_anchor_conflict_score(left_text: str, right_text: str) -> float:
    left = _reference_anchor_profile(left_text or "")
    right = _reference_anchor_profile(right_text or "")
    score = _assignment_contrast_score(left_text, right_text)

    shared_endpoint_paths = left.endpoint_paths & right.endpoint_paths
    if shared_endpoint_paths:
        same_method_pairs = left.endpoint_pairs & right.endpoint_pairs
        if not same_method_pairs and left.methods and right.methods and left.methods != right.methods:
            score += 3.2 + min(0.8, 0.2 * float(len(shared_endpoint_paths)))

    if left.license_terms and right.license_terms and left.license_terms != right.license_terms:
        score += 2.8
    elif (
        ("license" in left.key_names and right.license_terms)
        or ("license" in right.key_names and left.license_terms)
    ):
        score += 1.8

    if left.env_vars and right.env_vars and left.env_vars != right.env_vars:
        shared_prefixes = {
            env.split("_", 1)[0]
            for env in left.env_vars | right.env_vars
            if "_" in env
        }
        if shared_prefixes and not (left.env_vars & right.env_vars):
            score += 0.8
    return min(score, 6.0)


def _is_doc_like_path(path: str) -> bool:
    normalized = (path or "").strip().lower().replace("\\", "/")
    if not normalized:
        return False
    base = os.path.basename(normalized)
    doc_basenames = {
        "readme",
        "readme.md",
        "readme.rst",
        "readme.txt",
        "changelog",
        "changelog.md",
        "changelog.rst",
        "changelog.txt",
        "security",
        "security.md",
        "security.rst",
        "security.txt",
        "contributing",
        "contributing.md",
        "contributing.rst",
        "contributing.txt",
        "architecture",
        "architecture.md",
        "architecture.rst",
        "architecture.txt",
    }
    return base in doc_basenames or "/docs/" in normalized or "/doc/" in normalized


def _is_high_signal_metadata_path(path: str) -> bool:
    normalized = (path or "").strip().lower().replace("\\", "/")
    if not normalized:
        return False
    base = os.path.basename(normalized)
    return base in _HIGH_SIGNAL_METADATA_BASENAMES


def _chunk_relevance_score(
    chunk: str,
    idx: int,
    code_focus: bool,
    novelty_score: float,
) -> float:
    token_score = min(float(count_tokens(chunk)) / 240.0, 2.0)
    relevance = max(0.0, novelty_score) * 2.4
    if not code_focus:
        return relevance + token_score - (0.01 * float(min(idx, 50)))

    category, path_rank, _, _, _ = _chunk_priority_key(chunk, idx, code_focus)
    path = _extract_snapshot_file_path(chunk)
    structured_signal = _structured_source_signal_score(chunk, code_focus=code_focus)
    behavioral_doc_signal = _behavioral_doc_signal_score(chunk)
    relevance += token_score
    relevance += structured_signal
    relevance += max(0.0, 1.2 - (0.4 * float(category)))
    relevance += max(0.0, 0.7 - (0.35 * float(path_rank)))
    if _is_high_signal_metadata_path(path):
        relevance += 0.65
    if "```" in chunk:
        relevance += 0.4
    if "diff --git" in chunk or "@@" in chunk or "+++ b/" in chunk:
        relevance += 0.45
    if _is_doc_like_path(path):
        doc_penalty = 0.8
        if structured_signal > 0.0:
            doc_penalty = max(0.15, doc_penalty - min(0.65, structured_signal * 0.35))
        if behavioral_doc_signal > 0.0:
            doc_penalty = max(0.05, doc_penalty - min(0.45, behavioral_doc_signal * 0.18))
        relevance -= doc_penalty
    return relevance - (0.015 * float(min(idx, 50)))


def _chunk_redundancy_score(left: str, right: str, code_focus: bool) -> float:
    lexical = _token_overlap_ratio(_identifier_tokens(left), _identifier_tokens(right))
    if not code_focus:
        return lexical
    path_score = min(
        _path_overlap_score(_extract_snapshot_file_path(left), _extract_snapshot_file_path(right)) / 3.0,
        1.0,
    )
    return lexical + (0.35 * path_score)


def _reference_source_key(chunk: Chunk) -> str:
    if getattr(chunk, "document_id", None):
        return f"document:{chunk.document_id}"
    if getattr(chunk, "note_id", None):
        return f"note:{chunk.note_id}"
    return f"chunk:{getattr(chunk, 'chunk_id', '')}"


def _reference_chunk_key(chunk: Chunk) -> str:
    chunk_id = getattr(chunk, "chunk_id", None)
    if chunk_id:
        return str(chunk_id)
    source_key = _reference_source_key(chunk)
    content = getattr(chunk, "content", "") or ""
    return f"{source_key}:{stable_chunk_hash(content)}"


def _allow_same_document_feedback(user: User | None) -> bool:
    return bool(getattr(user, "include_own_documents_in_feedback", False))


def _same_file_self_reference_allowed(source_text: str, candidate_text: str, *, code_focus: bool) -> bool:
    source_path = _extract_snapshot_file_path(source_text).lower().replace("\\", "/")
    candidate_path = _extract_snapshot_file_path(candidate_text).lower().replace("\\", "/")
    if not source_path or not candidate_path or source_path != candidate_path:
        return True
    if _is_header_only_snapshot_chunk(candidate_text):
        return False

    source_part = _extract_snapshot_part(source_text)
    candidate_part = _extract_snapshot_part(candidate_text)
    if (
        source_part
        and candidate_part
        and source_part[1] == candidate_part[1]
        and abs(source_part[0] - candidate_part[0]) <= 1
    ):
        return False

    if not _bool_env("COMPAIR_ALLOW_SAME_FILE_SELF_FEEDBACK", False):
        return False

    if code_focus and _artifact_overlap_score(source_text, candidate_text) < 4.0:
        return False
    return True


def _reference_candidate_decision(
    candidate: Chunk,
    *,
    doc: Document,
    source_chunk: Chunk,
    allow_same_document: bool,
    code_focus: bool,
) -> tuple[bool, str | None]:
    if getattr(candidate, "chunk_type", "") != "document":
        return False, "non_document"
    if _reference_chunk_key(candidate) == _reference_chunk_key(source_chunk):
        return False, "same_chunk"

    source_content = getattr(source_chunk, "content", "") or ""
    candidate_content = getattr(candidate, "content", "") or ""
    if _is_header_only_snapshot_chunk(candidate_content):
        return False, "header_only"

    candidate_doc_id = getattr(candidate, "document_id", None)
    if candidate_doc_id != getattr(doc, "document_id", None):
        return True, None
    if not allow_same_document:
        return False, "same_document_disabled"
    if source_content and candidate_content and source_content == candidate_content:
        return False, "duplicate_content"
    if not _same_file_self_reference_allowed(source_content, candidate_content, code_focus=code_focus):
        return False, "same_file"
    return True, None


def _reference_candidate_allowed(
    candidate: Chunk,
    *,
    doc: Document,
    source_chunk: Chunk,
    allow_same_document: bool,
    code_focus: bool,
) -> bool:
    allowed, _ = _reference_candidate_decision(
        candidate,
        doc=doc,
        source_chunk=source_chunk,
        allow_same_document=allow_same_document,
        code_focus=code_focus,
    )
    return allowed


def _filter_reference_candidates(
    candidates: list[Chunk],
    *,
    doc: Document,
    source_chunk: Chunk,
    allow_same_document: bool,
    code_focus: bool,
) -> tuple[list[Chunk], dict[str, int]]:
    kept: list[Chunk] = []
    counts: dict[str, int] = {}
    for candidate in candidates:
        allowed, reason = _reference_candidate_decision(
            candidate,
            doc=doc,
            source_chunk=source_chunk,
            allow_same_document=allow_same_document,
            code_focus=code_focus,
        )
        if allowed:
            kept.append(candidate)
            continue
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
    return kept, counts


def _merge_reference_candidates(
    primary: list[Chunk],
    secondary: list[Chunk],
    limit: int,
) -> list[Chunk]:
    merged: list[Chunk] = []
    seen: set[str] = set()
    for candidate in [*primary, *secondary]:
        key = _reference_chunk_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        merged.append(candidate)
        if len(merged) >= limit:
            break
    return merged


def _lexical_reference_candidates(
    target_text: str,
    candidates: list[Chunk],
    *,
    limit: int,
    code_focus: bool,
) -> list[Chunk]:
    if not candidates or limit <= 0:
        return []

    target_tokens = _identifier_tokens(target_text, limit=96)
    target_path = _extract_snapshot_file_path(target_text)
    target_is_doc_like = code_focus and _is_doc_like_path(target_path)
    scored: list[tuple[float, int, Chunk]] = []
    for idx, candidate in enumerate(candidates):
        if getattr(candidate, "chunk_type", "") != "document":
            continue
        content = getattr(candidate, "content", "") or ""
        if not content or _is_snapshot_metadata_chunk(content):
            continue

        candidate_tokens = _identifier_tokens(content, limit=96)
        candidate_path = _extract_snapshot_file_path(content)
        lexical_score = _token_overlap_ratio(target_tokens, candidate_tokens)
        path_theme_score = _token_overlap_ratio(target_tokens, _path_token_set(candidate_path))
        path_score = _path_overlap_score(target_path, candidate_path)
        artifact_score = min(_artifact_overlap_score(target_text, content), 4.0)
        anchor_overlap = _reference_anchor_overlap_score(target_text, content)
        anchor_conflict = _reference_anchor_conflict_score(target_text, content)
        code_bonus = 0.35 if "```" in content else 0.0
        diff_bonus = 0.25 if "diff --git" in content or "@@" in content or "+++ b/" in content else 0.0
        doc_penalty = 0.0
        doc_bonus = 0.0
        metadata_bonus = 0.0
        if code_focus:
            candidate_is_doc_like = _is_doc_like_path(candidate_path)
            if target_is_doc_like and candidate_is_doc_like:
                doc_bonus = 0.75
            elif not target_is_doc_like and candidate_is_doc_like:
                doc_penalty = 0.75
            if _is_high_signal_metadata_path(target_path) and _is_high_signal_metadata_path(candidate_path):
                metadata_bonus = 0.85
            if anchor_overlap > 0.0 and target_is_doc_like != candidate_is_doc_like:
                doc_bonus += 0.45
        score = (
            (lexical_score * 5.0)
            + (path_theme_score * 2.5)
            + artifact_score
            + (anchor_overlap * 2.5)
            + (anchor_conflict * 3.0)
            + (path_score * 1.5)
            + code_bonus
            + diff_bonus
            + doc_bonus
            + metadata_bonus
            - doc_penalty
        )
        if score <= 0.0:
            continue
        scored.append((score, idx, candidate))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, _, candidate in scored[:limit]]


def _anchor_reference_candidates(
    target_text: str,
    candidates: list[Chunk],
    *,
    limit: int,
    code_focus: bool,
) -> list[Chunk]:
    if not candidates or limit <= 0:
        return []

    target_path = _extract_snapshot_file_path(target_text)
    target_is_doc_like = code_focus and _is_doc_like_path(target_path)
    scored: list[tuple[float, int, Chunk]] = []
    for idx, candidate in enumerate(candidates):
        if getattr(candidate, "chunk_type", "") != "document":
            continue
        content = getattr(candidate, "content", "") or ""
        if not content or _is_snapshot_metadata_chunk(content):
            continue

        candidate_path = _extract_snapshot_file_path(content)
        candidate_is_doc_like = _is_doc_like_path(candidate_path)
        anchor_overlap = _reference_anchor_overlap_score(target_text, content)
        anchor_conflict = _reference_anchor_conflict_score(target_text, content)
        if anchor_overlap <= 0.0 and anchor_conflict <= 0.0:
            continue
        artifact_score = min(_artifact_overlap_score(target_text, content), 4.0)
        path_score = _path_overlap_score(target_path, candidate_path)
        score = (anchor_overlap * 3.5) + (anchor_conflict * 4.0) + artifact_score + (path_score * 0.8)
        if code_focus:
            if _is_high_signal_metadata_path(target_path) and _is_high_signal_metadata_path(candidate_path):
                score += 1.0
            if target_is_doc_like != candidate_is_doc_like:
                score += 0.6
            if target_is_doc_like and candidate_is_doc_like:
                score += 0.25
        scored.append((score, idx, candidate))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, _, candidate in scored[:limit]]


def _rerank_reference_chunks(
    target_text: str,
    candidates: list[Chunk],
    code_focus: bool,
    *,
    doc: Document | None = None,
    raw_vector_candidates: list[Chunk] | None = None,
    lexical_candidates: list[Chunk] | None = None,
    anchor_candidates: list[Chunk] | None = None,
) -> list[Chunk]:
    candidate_limit, final_limit, max_per_source = _reference_selection_config(code_focus)
    if not candidates:
        return []

    trim_limit = max(candidate_limit, min(len(candidates), candidate_limit * 2))
    trimmed = candidates[:trim_limit]
    if not code_focus:
        return trimmed[:final_limit]

    target_path = _extract_snapshot_file_path(target_text)
    target_is_doc_like = _is_doc_like_path(target_path)
    target_tokens = _identifier_tokens(target_text)
    source_document_id = getattr(doc, "document_id", None)
    vector_rank = {
        _reference_chunk_key(chunk): idx + 1
        for idx, chunk in enumerate(raw_vector_candidates or [])
    }
    lexical_rank = {
        _reference_chunk_key(chunk): idx + 1
        for idx, chunk in enumerate(lexical_candidates or [])
    }
    anchor_rank = {
        _reference_chunk_key(chunk): idx + 1
        for idx, chunk in enumerate(anchor_candidates or [])
    }
    reranker_enabled = code_focus and _reference_reranker_state()[0] is not None
    used_indices: set[int] = set()
    source_counts: dict[str, int] = {}
    selected: list[Chunk] = []
    selected_tokens: list[set[str]] = []

    while len(selected) < final_limit:
        best_index = -1
        best_score = float("-inf")
        best_tokens: set[str] | None = None
        for idx, candidate in enumerate(trimmed):
            if idx in used_indices:
                continue
            if getattr(candidate, "chunk_type", "") != "document":
                continue
            content = getattr(candidate, "content", "") or ""
            if _is_snapshot_metadata_chunk(content):
                continue
            source_key = _reference_source_key(candidate)
            if source_counts.get(source_key, 0) >= max_per_source:
                continue

            candidate_tokens = _identifier_tokens(content)
            feature_row = _reference_candidate_feature_row(
                query_text=target_text,
                candidate=candidate,
                source_document_id=source_document_id,
                source_path=target_path,
                vector_rank=vector_rank,
                lexical_rank=lexical_rank,
                anchor_rank=anchor_rank,
            )
            diversity_penalty = 0.0
            if selected_tokens:
                diversity_penalty = max(_token_overlap_ratio(candidate_tokens, prev) for prev in selected_tokens)
            source_penalty = 0.75 * float(source_counts.get(source_key, 0))
            if reranker_enabled:
                reranker_score = float(feature_row.get("reranker_score") or 0.0)
                base_score = float(len(trimmed) - idx) / float(max(1, len(trimmed)))
                score = reranker_score + (base_score * 0.05) - (diversity_penalty * 1.35) - source_penalty
            else:
                candidate_path = str(feature_row.get("candidate_path") or "")
                base_score = float(len(trimmed) - idx) / float(max(1, len(trimmed)))
                lexical_score = float(feature_row.get("lexical_score") or 0.0)
                path_theme_score = float(feature_row.get("path_theme_score") or 0.0)
                path_score = float(feature_row.get("path_score") or 0.0)
                artifact_score = float(feature_row.get("artifact_score") or 0.0)
                anchor_overlap = float(feature_row.get("anchor_overlap") or 0.0)
                anchor_conflict = float(feature_row.get("anchor_conflict") or 0.0)
                code_bonus = 0.4 if "```" in content else 0.0
                doc_bonus = 0.0
                metadata_bonus = 0.0
                candidate_is_doc_like = _is_doc_like_path(candidate_path)
                if target_is_doc_like and candidate_is_doc_like:
                    doc_bonus = 0.75
                elif target_is_doc_like != candidate_is_doc_like and (anchor_overlap > 0.0 or anchor_conflict > 0.0):
                    doc_bonus = 0.55
                if _is_high_signal_metadata_path(target_path) and _is_high_signal_metadata_path(candidate_path):
                    metadata_bonus = 0.85
                score = (
                    (base_score * 3.0)
                    + (lexical_score * 4.0)
                    + (path_theme_score * 2.0)
                    + artifact_score
                    + (anchor_overlap * 3.5)
                    + (anchor_conflict * 4.0)
                    + path_score
                    + code_bonus
                    + doc_bonus
                    + metadata_bonus
                    - (diversity_penalty * 2.5)
                    - source_penalty
                )
            if score > best_score:
                best_score = score
                best_index = idx
                best_tokens = candidate_tokens

        if best_index < 0:
            break

        chosen = trimmed[best_index]
        used_indices.add(best_index)
        source_key = _reference_source_key(chosen)
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        selected.append(chosen)
        selected_tokens.append(best_tokens or set())

    if selected:
        return selected
    return trimmed[:final_limit]


def _reference_query_text(text: str, focus_text: str, change_context: str, *, code_focus: bool) -> str:
    if not code_focus or _is_snapshot_metadata_chunk(text):
        return text
    focus = (focus_text or "").strip()
    change = (change_context or "").strip()
    target = (text or "").strip()
    query = change or focus
    if not query or not target or query == target:
        return text

    focus_tokens = count_tokens(query)
    target_tokens = count_tokens(text)
    if focus_tokens <= 0 or target_tokens <= 0 or focus_tokens >= target_tokens:
        return text

    # Only switch to the compact changed-window query when it meaningfully narrows
    # the search surface; otherwise reuse the full chunk embedding.
    if (focus_tokens * 4) > (target_tokens * 3) and (len(query) * 4) > (len(target) * 3):
        return text
    return query


def _reference_candidate_feature_row(
    *,
    query_text: str,
    candidate: Chunk,
    source_document_id: str | None,
    source_path: str,
    vector_rank: Mapping[str, int],
    lexical_rank: Mapping[str, int],
    anchor_rank: Mapping[str, int],
) -> dict[str, object]:
    key = _reference_chunk_key(candidate)
    content = getattr(candidate, "content", "") or ""
    candidate_path = _extract_snapshot_file_path(content)
    query_tokens = _identifier_tokens(query_text)
    target_path = source_path or _extract_snapshot_file_path(query_text)
    lexical_score = _token_overlap_ratio(query_tokens, _identifier_tokens(content))
    path_theme_score = _token_overlap_ratio(query_tokens, _path_token_set(candidate_path))
    path_score = _path_overlap_score(target_path, candidate_path)
    artifact_score = min(_artifact_overlap_score(query_text, content), 4.0)
    anchor_overlap = _reference_anchor_overlap_score(query_text, content)
    anchor_conflict = _reference_anchor_conflict_score(query_text, content)
    combined_signal = (
        (lexical_score * 4.0)
        + (path_theme_score * 2.0)
        + artifact_score
        + (anchor_overlap * 3.0)
        + (anchor_conflict * 3.5)
        + path_score
    )
    row: dict[str, object] = {
        "source_path": source_path,
        "candidate_path": candidate_path,
        "same_document": getattr(candidate, "document_id", None) == source_document_id,
        "vector_rank": vector_rank.get(key),
        "lexical_rank": lexical_rank.get(key),
        "anchor_rank": anchor_rank.get(key),
        "lexical_score": round(lexical_score, 4),
        "path_theme_score": round(path_theme_score, 4),
        "path_score": round(path_score, 4),
        "artifact_score": round(artifact_score, 4),
        "anchor_overlap": round(anchor_overlap, 4),
        "anchor_conflict": round(anchor_conflict, 4),
        "combined_signal": round(combined_signal, 4),
    }
    reranker_score = _reference_reranker_score(row)
    if reranker_score is not None:
        row["reranker_score"] = round(reranker_score, 6)
    return row


def _reference_trace_entries(
    *,
    query_text: str,
    source_chunk: Chunk,
    doc: Document,
    raw_candidates: list[Chunk],
    raw_vector_candidates: list[Chunk],
    allow_same_document: bool,
    code_focus: bool,
    lexical_candidates: list[Chunk],
    anchor_candidates: list[Chunk],
    selected_references: list[Chunk],
) -> list[dict[str, object]]:
    if not raw_candidates:
        return []

    source_document_id = getattr(doc, "document_id", None)
    source_path = _extract_snapshot_file_path(query_text)
    vector_rank = {_reference_chunk_key(chunk): idx + 1 for idx, chunk in enumerate(raw_vector_candidates)}
    lexical_rank = {_reference_chunk_key(chunk): idx + 1 for idx, chunk in enumerate(lexical_candidates)}
    anchor_rank = {_reference_chunk_key(chunk): idx + 1 for idx, chunk in enumerate(anchor_candidates)}
    selected_rank = {_reference_chunk_key(chunk): idx + 1 for idx, chunk in enumerate(selected_references)}

    def _sort_key(entry: dict[str, object]) -> tuple[int, int, int, float]:
        status = str(entry.get("selection_status") or "")
        status_rank = {"selected": 0, "candidate": 1, "filtered": 2}.get(status, 3)
        best_rank = min(
            int(entry.get("selected_rank") or 9999),
            int(entry.get("anchor_rank") or 9999),
            int(entry.get("lexical_rank") or 9999),
            int(entry.get("vector_rank") or 9999),
        )
        same_doc_rank = 0 if entry.get("same_document") else 1
        combined = float(entry.get("combined_signal") or 0.0)
        return (status_rank, best_rank, same_doc_rank, -combined)

    entries: list[dict[str, object]] = []
    for candidate in raw_candidates:
        key = _reference_chunk_key(candidate)
        allowed, reason = _reference_candidate_decision(
            candidate,
            doc=doc,
            source_chunk=source_chunk,
            allow_same_document=allow_same_document,
            code_focus=code_focus,
        )
        feature_row = _reference_candidate_feature_row(
            query_text=query_text,
            candidate=candidate,
            source_document_id=source_document_id,
            source_path=source_path,
            vector_rank=vector_rank,
            lexical_rank=lexical_rank,
            anchor_rank=anchor_rank,
        )
        if key in selected_rank:
            selection_status = "selected"
        elif allowed:
            selection_status = "candidate"
        else:
            selection_status = "filtered"
        entries.append(
            {
                "chunk_id": getattr(candidate, "chunk_id", None),
                "document_id": getattr(candidate, "document_id", None),
                "path": feature_row.get("candidate_path"),
                "same_document": feature_row.get("same_document"),
                "selection_status": selection_status,
                "drop_reason": None if allowed else reason,
                "vector_rank": feature_row.get("vector_rank"),
                "lexical_rank": feature_row.get("lexical_rank"),
                "anchor_rank": feature_row.get("anchor_rank"),
                "selected_rank": selected_rank.get(key),
                "lexical_score": feature_row.get("lexical_score"),
                "path_theme_score": feature_row.get("path_theme_score"),
                "path_score": feature_row.get("path_score"),
                "artifact_score": feature_row.get("artifact_score"),
                "anchor_overlap": feature_row.get("anchor_overlap"),
                "anchor_conflict": feature_row.get("anchor_conflict"),
                "combined_signal": feature_row.get("combined_signal"),
                "reranker_score": feature_row.get("reranker_score"),
            }
        )

    entries.sort(key=_sort_key)
    trace_max = _reference_trace_max_candidates()
    if trace_max > 0:
        return entries[:trace_max]
    return entries


def process_document(
    user: User,
    session: SASession,
    embedder: Embedder,
    reviewer: Reviewer,
    doc: Document,
    generate_feedback: bool = True,
    chunk_mode: Optional[str] = None,
    reanalyze_existing: bool = False,
) -> Mapping[str, int]:
    new = False

    prev_content = get_history(doc, "content").deleted
    prev_chunks: list[str] = []
    if prev_content:
        prev_chunks = chunk_text_with_mode(prev_content[-1], chunk_mode=chunk_mode)

    feedback_limit_env = os.getenv("COMPAIR_CORE_FEEDBACK_LIMIT")
    try:
        feedback_limit = int(feedback_limit_env) if feedback_limit_env else None
    except ValueError:
        feedback_limit = None
    time_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

    recent_feedback_count = session.query(Feedback).filter(
        Feedback.source_chunk_id.in_(
            session.query(Chunk.chunk_id).filter(Chunk.document_id == doc.document_id)
        ),
        Feedback.timestamp >= time_cutoff,
    ).count()

    content = doc.content
    doc.topic_tags = extract_topic_tags(content)
    chunks = chunk_text_with_mode(content, chunk_mode=chunk_mode)
    prev_set = set(prev_chunks)
    new_chunks = [c for c in chunks if c not in prev_set]
    meaningful_new_chunk_count = 0

    prioritized_chunk_indices: list[int] = []
    code_focus = False
    if generate_feedback:
        code_focus = is_code_review_document(doc, chunk_mode)
        meaningful_new_chunk_count = len(
            [
                chunk
                for chunk in new_chunks
                if not (code_focus and _is_snapshot_metadata_chunk(chunk))
            ]
        )
        prioritized_chunk_indices = detect_significant_edits(
            prev_chunks=prev_chunks,
            new_chunks=new_chunks,
            code_focus=code_focus,
        )
        if code_focus:
            prioritized_chunk_indices = [
                i for i in prioritized_chunk_indices if not _is_snapshot_metadata_chunk(new_chunks[i])
            ]

    feedback_min_tokens = int(os.getenv("COMPAIR_FEEDBACK_MIN_TOKENS", "120"))
    feedback_fallback_min = int(os.getenv("COMPAIR_FEEDBACK_MIN_TOKENS_FALLBACK", "20"))
    token_lens = [count_tokens(c) for c in new_chunks]
    eligible_indices = [i for i, t in enumerate(token_lens) if t >= feedback_min_tokens]
    fallback_indices = list(range(len(token_lens)))
    if code_focus:
        eligible_indices = [i for i in eligible_indices if not _is_snapshot_metadata_chunk(new_chunks[i])]
        fallback_indices = [i for i in fallback_indices if not _is_snapshot_metadata_chunk(new_chunks[i])]
    if generate_feedback and feedback_min_tokens > 0:
        prioritized_chunk_indices = [i for i in prioritized_chunk_indices if i in eligible_indices]
        if not prioritized_chunk_indices and eligible_indices:
            fallback_idx = max(eligible_indices, key=lambda idx: token_lens[idx])
            prioritized_chunk_indices = [fallback_idx]
        elif not eligible_indices and fallback_indices:
            fallback_idx = max(fallback_indices, key=lambda idx: token_lens[idx])
            if token_lens[fallback_idx] >= feedback_fallback_min:
                prioritized_chunk_indices = [fallback_idx]

    if feedback_limit is None:
        indices_to_generate_feedback = prioritized_chunk_indices
    else:
        num_chunks_can_generate_feedback = max((feedback_limit - recent_feedback_count), 0)
        indices_to_generate_feedback = prioritized_chunk_indices[:num_chunks_can_generate_feedback]

    feedback_focus_by_new_index: dict[int, str] = {}
    feedback_change_context_by_new_index: dict[int, str] = {}
    for idx in indices_to_generate_feedback:
        focus_text = _focus_text_for_chunk(new_chunks[idx], prev_chunks, code_focus=code_focus)
        if focus_text:
            feedback_focus_by_new_index[idx] = focus_text
        change_context = _change_context_for_chunk(new_chunks[idx], prev_chunks, code_focus=code_focus)
        if change_context:
            feedback_change_context_by_new_index[idx] = change_context
    feedback_query_embedding_by_new_index: dict[int, list[float]] = {}
    query_embedding_requests: list[tuple[int, str]] = []
    for idx in indices_to_generate_feedback:
        query_text = _reference_query_text(
            new_chunks[idx],
            feedback_focus_by_new_index.get(idx, ""),
            feedback_change_context_by_new_index.get(idx, ""),
            code_focus=code_focus,
        )
        if query_text.strip() and query_text != new_chunks[idx]:
            query_embedding_requests.append((idx, query_text))
    if query_embedding_requests:
        query_embeddings = create_embeddings(
            embedder,
            [query_text for _, query_text in query_embedding_requests],
            user=user,
        )
        for (idx, _), query_embedding in zip(query_embedding_requests, query_embeddings):
            feedback_query_embedding_by_new_index[idx] = query_embedding

    existing_indices_to_generate_feedback: list[int] = []
    available_existing_candidate_count = 0
    should_reanalyze_existing = _should_reanalyze_existing_chunks(
        reanalyze_existing=reanalyze_existing,
        meaningful_new_chunk_count=meaningful_new_chunk_count,
    )

    if generate_feedback and should_reanalyze_existing:
        existing_indices = _existing_feedback_candidate_indices(
            session=session,
            doc=doc,
            chunks=chunks,
            code_focus=code_focus,
            feedback_min_tokens=feedback_min_tokens,
            feedback_fallback_min=feedback_fallback_min,
        )
        available_existing_candidate_count = len(existing_indices)
        if feedback_limit is None:
            existing_indices_to_generate_feedback = existing_indices
        else:
            remaining_slots = max((feedback_limit - recent_feedback_count - len(indices_to_generate_feedback)), 0)
            existing_indices_to_generate_feedback = existing_indices[:remaining_slots]
    elif generate_feedback and code_focus and not indices_to_generate_feedback:
        existing_indices = _existing_feedback_candidate_indices(
            session=session,
            doc=doc,
            chunks=chunks,
            code_focus=code_focus,
            feedback_min_tokens=feedback_min_tokens,
            feedback_fallback_min=feedback_fallback_min,
        )
        available_existing_candidate_count = len(existing_indices)

    if generate_feedback:
        log_event(
            "feedback_chunk_selection",
            document_id=doc.document_id,
            code_focus=code_focus,
            total_chunks=len(chunks),
            new_chunks=len(new_chunks),
            meaningful_new_chunks=meaningful_new_chunk_count,
            prioritized_new_chunks=len(prioritized_chunk_indices),
            selected_new_chunks=len(indices_to_generate_feedback),
            available_existing_candidates=available_existing_candidate_count,
            selected_existing_chunks=len(existing_indices_to_generate_feedback),
            reanalyze_existing=reanalyze_existing,
        )
        if (
            code_focus
            and available_existing_candidate_count > 0
            and not indices_to_generate_feedback
            and not existing_indices_to_generate_feedback
        ):
            skip_reason = "reanalyze_existing_disabled"
            if reanalyze_existing and meaningful_new_chunk_count > 0:
                skip_reason = "meaningful_new_chunks_present"
            log_event(
                "feedback_existing_candidates_skipped",
                document_id=doc.document_id,
                available_existing_candidates=available_existing_candidate_count,
                reason=skip_reason,
                new_chunks=len(new_chunks),
                meaningful_new_chunks=meaningful_new_chunk_count,
            )

    new_chunk_embeddings = create_embeddings(embedder, new_chunks, user=user) if new_chunks else []
    for i, chunk in enumerate(new_chunks):
        should_generate_feedback = i in indices_to_generate_feedback
        process_text(
            session=session,
            embedder=embedder,
            reviewer=reviewer,
            doc=doc,
            text=chunk,
            generate_feedback=should_generate_feedback,
            precomputed_embedding=new_chunk_embeddings[i],
            query_embedding=feedback_query_embedding_by_new_index.get(i),
            focus_text=feedback_focus_by_new_index.get(i, ""),
            change_context=feedback_change_context_by_new_index.get(i, ""),
        )

    for idx in existing_indices_to_generate_feedback:
        focus_text = _focus_text_for_chunk(chunks[idx], prev_chunks, code_focus=code_focus)
        change_context = _change_context_for_chunk(chunks[idx], prev_chunks, code_focus=code_focus)
        query_embedding = None
        query_text = _reference_query_text(chunks[idx], focus_text, change_context, code_focus=code_focus)
        if query_text.strip() and query_text != chunks[idx]:
            query_embedding = create_embedding(embedder, query_text, user=user)
        process_text(
            session=session,
            embedder=embedder,
            reviewer=reviewer,
            doc=doc,
            text=chunks[idx],
            generate_feedback=True,
            query_embedding=query_embedding,
            focus_text=focus_text,
            change_context=change_context,
        )

    removed = [c for c in prev_chunks if c not in set(chunks)]
    for chunk in removed:
        remove_text(session=session, text=chunk, document_id=doc.document_id)

    if doc.groups:
        log_activity(
            session=session,
            user_id=doc.author_id,
            group_id=doc.groups[0].group_id,
            action="update",
            object_id=doc.document_id,
            object_name=doc.title,
            object_type="document",
        )

    session.commit()
    return {"new": new}


def detect_significant_edits(
    prev_chunks: list[str],
    new_chunks: list[str],
    threshold: float = 0.5,
    code_focus: bool = False,
) -> list[int]:
    if not new_chunks:
        return []
    if not prev_chunks:
        novelty_scores = {i: 1.0 for i in range(len(new_chunks))}
        return prioritize_chunks(
            list(range(len(new_chunks))),
            new_chunks,
            code_focus=code_focus,
            novelty_scores=novelty_scores,
        )
    candidate_indices: list[int] = []
    novelty_scores: dict[int, float] = {}
    for idx, new_chunk in enumerate(new_chunks):
        if new_chunk in prev_chunks:
            continue
        best_match = max((Levenshtein.ratio(new_chunk, prev_chunk) for prev_chunk in prev_chunks), default=0.0)
        novelty_scores[idx] = max(0.0, 1.0 - best_match)
        if best_match < threshold:
            candidate_indices.append(idx)
    return prioritize_chunks(
        candidate_indices,
        new_chunks,
        code_focus=code_focus,
        novelty_scores=novelty_scores,
    )


def prioritize_chunks(
    indices: list[int],
    chunks: list[str],
    limit: int = 10,
    code_focus: bool = False,
    novelty_scores: Mapping[int, float] | None = None,
) -> list[int]:
    if not indices:
        return []
    if code_focus:
        indices = [i for i in indices if not _is_snapshot_metadata_chunk(chunks[i])]
        if not indices:
            return []
        try:
            limit = int(os.getenv("COMPAIR_CODE_REPO_FEEDBACK_CANDIDATES", str(limit)))
        except ValueError:
            pass
    else:
        indices.sort(key=lambda i: _chunk_priority_key(chunks[i], i, code_focus))
        return indices[:limit]

    if novelty_scores is None:
        novelty_scores = {}

    remaining = list(indices)
    selected: list[int] = []
    while remaining and len(selected) < limit:
        best_idx = -1
        best_score = float("-inf")
        for idx in remaining:
            relevance = _chunk_relevance_score(chunks[idx], idx, code_focus, novelty_scores.get(idx, 1.0))
            redundancy = 0.0
            if selected:
                redundancy = max(
                    _chunk_redundancy_score(chunks[idx], chunks[chosen_idx], code_focus)
                    for chosen_idx in selected
                )
            score = relevance - (2.6 * redundancy)
            if best_idx < 0 or score > best_score:
                best_idx = idx
                best_score = score
        if best_idx < 0:
            break
        if selected and best_score < 1.8:
            break
        selected.append(best_idx)
        remaining = [idx for idx in remaining if idx != best_idx]

    if selected:
        return selected
    indices.sort(key=lambda i: _chunk_priority_key(chunks[i], i, code_focus))
    return indices[:limit]


def _existing_feedback_candidate_indices(
    session: SASession,
    doc: Document,
    chunks: list[str],
    code_focus: bool,
    feedback_min_tokens: int,
    feedback_fallback_min: int,
) -> list[int]:
    if not chunks:
        return []

    chunk_set = set(chunks)
    content_to_chunk_id: dict[str, str] = {}
    existing_rows = session.query(Chunk.chunk_id, Chunk.content).filter(
        Chunk.document_id == doc.document_id,
        Chunk.chunk_type == "document",
    ).all()
    for chunk_id, content in existing_rows:
        if content in chunk_set and content not in content_to_chunk_id:
            content_to_chunk_id[content] = chunk_id
    if not content_to_chunk_id:
        return []

    feedback_chunk_ids = {
        source_chunk_id
        for (source_chunk_id,) in session.query(Feedback.source_chunk_id).filter(
            Feedback.source_chunk_id.in_(list(content_to_chunk_id.values()))
        ).distinct().all()
    }

    candidate_indices = [
        idx
        for idx, chunk in enumerate(chunks)
        if content_to_chunk_id.get(chunk) and content_to_chunk_id[chunk] not in feedback_chunk_ids
    ]
    if code_focus:
        candidate_indices = [idx for idx in candidate_indices if not _is_snapshot_metadata_chunk(chunks[idx])]
    if not candidate_indices:
        return []

    token_lens = {idx: count_tokens(chunks[idx]) for idx in candidate_indices}
    if feedback_min_tokens > 0:
        eligible = [idx for idx in candidate_indices if token_lens[idx] >= feedback_min_tokens]
        if eligible:
            candidate_indices = eligible
        else:
            fallback_idx = max(candidate_indices, key=lambda idx: token_lens[idx])
            if token_lens[fallback_idx] < feedback_fallback_min:
                return []
            candidate_indices = [fallback_idx]

    return prioritize_chunks(
        candidate_indices,
        chunks,
        code_focus=code_focus,
        novelty_scores={idx: 1.0 for idx in candidate_indices},
    )


def _best_previous_chunk_for_focus(
    chunk: str,
    prev_chunks: list[str],
    *,
    code_focus: bool,
) -> str:
    if not prev_chunks:
        return ""
    target_path = _extract_snapshot_file_path(chunk)
    candidates = prev_chunks
    if code_focus and target_path:
        same_path = [prev for prev in prev_chunks if _extract_snapshot_file_path(prev) == target_path]
        if same_path:
            candidates = same_path
    best_chunk = ""
    best_score = float("-inf")
    for prev_chunk in candidates:
        score = Levenshtein.ratio(chunk, prev_chunk)
        if score > best_score:
            best_score = score
            best_chunk = prev_chunk
    return best_chunk


def _merge_line_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ranges.sort()
    merged: list[tuple[int, int]] = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _initial_focus_window(chunk: str, *, max_lines: int = 12) -> str:
    lines = [line.rstrip() for line in (chunk or "").splitlines()]
    if not lines:
        return ""
    selected: list[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if idx == 0 and stripped.startswith("### File:"):
            selected.append(line)
            continue
        selected.append(line)
        if len(selected) >= max_lines:
            break
    return "\n".join(selected).strip()


def _compact_changed_focus_window(
    current_chunk: str,
    previous_chunk: str,
    *,
    context_lines: int = 1,
    max_lines: int = 18,
) -> str:
    if not current_chunk.strip():
        return ""
    if not previous_chunk.strip():
        return _initial_focus_window(current_chunk)

    current_lines = [line.rstrip() for line in current_chunk.splitlines()]
    previous_lines = [line.rstrip() for line in previous_chunk.splitlines()]
    matcher = difflib.SequenceMatcher(a=previous_lines, b=current_lines, autojunk=False)
    ranges: list[tuple[int, int]] = []
    for tag, _, _, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag in {"replace", "insert"} and j1 != j2:
            start = max(0, j1 - context_lines)
            end = min(len(current_lines), j2 + context_lines)
        else:
            anchor = min(max(j1, 0), max(len(current_lines) - 1, 0))
            start = max(0, anchor - context_lines)
            end = min(len(current_lines), anchor + context_lines + 1)
        if start < end:
            ranges.append((start, end))

    merged = _merge_line_ranges(ranges)
    if not merged:
        return _initial_focus_window(current_chunk)

    selected: list[str] = []
    if current_lines and current_lines[0].strip().startswith("### File:"):
        selected.append(current_lines[0])

    for idx, (start, end) in enumerate(merged):
        if idx > 0 and selected and selected[-1] != "...":
            selected.append("...")
        selected.extend(current_lines[start:end])

    compact = [line for line in selected if line.strip()]
    if len(compact) > max_lines:
        head = compact[:max_lines]
        if head[-1] != "...":
            head.append("...")
        compact = head
    return "\n".join(compact).strip()


def _compact_change_context(
    current_chunk: str,
    previous_chunk: str,
    *,
    max_lines: int = 12,
) -> str:
    if not current_chunk.strip() or not previous_chunk.strip():
        return ""

    current_lines = [line.rstrip() for line in current_chunk.splitlines()]
    previous_lines = [line.rstrip() for line in previous_chunk.splitlines()]
    matcher = difflib.SequenceMatcher(a=previous_lines, b=current_lines, autojunk=False)
    out: list[str] = []

    if current_lines and current_lines[0].strip().startswith("### File:"):
        out.append(current_lines[0])

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        before_lines = [f"- {line}" for line in previous_lines[i1:i2] if line.strip()]
        after_lines = [f"+ {line}" for line in current_lines[j1:j2] if line.strip()]
        for line in [*before_lines, *after_lines]:
            out.append(line)
            if len(out) >= max_lines:
                return "\n".join(out).strip()
    return "\n".join(out).strip()


def _focus_text_for_chunk(chunk: str, prev_chunks: list[str], *, code_focus: bool) -> str:
    previous_chunk = _best_previous_chunk_for_focus(chunk, prev_chunks, code_focus=code_focus)
    return _compact_changed_focus_window(chunk, previous_chunk)


def _change_context_for_chunk(chunk: str, prev_chunks: list[str], *, code_focus: bool) -> str:
    previous_chunk = _best_previous_chunk_for_focus(chunk, prev_chunks, code_focus=code_focus)
    return _compact_change_context(chunk, previous_chunk)


def process_text(
    session: SASession,
    embedder: Embedder,
    reviewer: Reviewer,
    doc: Document,
    text: str,
    generate_feedback: bool = True,
    note: Note | None = None,
    precomputed_embedding: list[float] | None = None,
    query_embedding: list[float] | None = None,
    focus_text: str = "",
    change_context: str = "",
) -> None:
    logger = logging.getLogger(__name__)
    chunk_hash = stable_chunk_hash(text)

    chunk_type = "note" if note else "document"
    note_id = note.note_id if note else None

    existing_chunks = session.query(Chunk).filter(
        Chunk.document_id == doc.document_id,
        Chunk.chunk_type == chunk_type,
        Chunk.note_id == note_id,
        Chunk.content == text,
    )

    user = session.query(User).filter(User.user_id == doc.author_id).first()
    existing_rows = existing_chunks.all()
    existing_chunk = existing_rows[0] if existing_rows else None

    embedding = precomputed_embedding
    if existing_rows:
        for chunk in existing_rows:
            if chunk.hash != chunk_hash:
                chunk.hash = chunk_hash
            if embedding is None and chunk.embedding is not None:
                embedding = chunk.embedding
        if embedding is None:
            embedding = create_embedding(embedder, text, user=user)
        for chunk in existing_rows:
            if chunk.embedding is None:
                chunk.embedding = embedding
        session.commit()
    else:
        chunk = Chunk(
            hash=chunk_hash,
            document_id=doc.document_id,
            note_id=note_id,
            chunk_type=chunk_type,
            content=text,
        )
        if embedding is None:
            embedding = create_embedding(embedder, text, user=user)
        chunk.embedding = embedding
        session.add(chunk)
        session.commit()
        existing_chunk = chunk
    if existing_chunk is None:
        existing_chunk = session.query(Chunk).filter(
            Chunk.document_id == doc.document_id,
            Chunk.chunk_type == chunk_type,
            Chunk.note_id == note_id,
            Chunk.content == text,
        ).first()

    references: list[Chunk] = []
    if generate_feedback and existing_chunk:
        doc_group_ids = [g.group_id for g in doc.groups]
        target_embedding = existing_chunk.embedding
        code_focus = _is_code_review_chunk(doc, text)
        allow_same_document = _allow_same_document_feedback(user)
        query_text = _reference_query_text(text, focus_text, change_context, code_focus=code_focus)
        query_vector = query_embedding or target_embedding
        if query_vector is None and query_text:
            query_vector = create_embedding(embedder, query_text, user=user)
        candidate_limit, _, _ = _reference_selection_config(code_focus)
        merge_limit = max(candidate_limit * 3, 24 if code_focus else candidate_limit * 2)
        vector_fetch_limit = candidate_limit
        if code_focus:
            vector_fetch_limit = max(candidate_limit * 6, merge_limit)

        if query_vector is not None:
            lexical_candidates: list[Chunk] = []
            anchor_candidates: list[Chunk] = []
            candidate_count = 0
            filtered_counts: dict[str, int] = {}
            raw_candidate_count = 0
            raw_all_candidates: list[Chunk] = []
            raw_vector_candidates: list[Chunk] = []
            base_query = None
            if doc_group_ids:
                base_query = (
                    session.query(Chunk)
                    .join(Chunk.document)
                    .join(Document.groups)
                    .filter(Group.group_id.in_(doc_group_ids))
                )
            elif allow_same_document:
                base_query = (
                    session.query(Chunk)
                    .join(Chunk.document)
                    .filter(Document.document_id == doc.document_id)
                )

            all_candidates: list[Chunk] | None = None
            if base_query is None:
                candidates = []
            else:
                published_filter = Document.is_published.is_(True)
                if allow_same_document:
                    published_filter = or_(Document.is_published.is_(True), Document.document_id == doc.document_id)
                else:
                    base_query = base_query.filter(Document.document_id != doc.document_id)
                base_query = base_query.filter(
                    published_filter,
                    Chunk.chunk_type == "document",
                )

            if base_query is None:
                candidates = []
            elif VECTOR_BACKEND == "pgvector":
                candidates = (
                    base_query.order_by(
                        Chunk.embedding.cosine_distance(query_vector)
                    )
                    .limit(vector_fetch_limit)
                    .all()
                )
                raw_vector_candidates = list(candidates)
                raw_candidate_count = len(candidates)
                if code_focus:
                    all_candidates = base_query.all()
                    raw_all_candidates = list(all_candidates)
                    raw_candidate_count = len(all_candidates)
            else:
                all_candidates = base_query.all()
                raw_all_candidates = list(all_candidates)
                raw_candidate_count = len(all_candidates)
                scored: list[tuple[float, Chunk]] = []
                for candidate in all_candidates:
                    score = cosine_similarity(candidate.embedding, query_vector)
                    if score is not None:
                        scored.append((score, candidate))
                scored.sort(key=lambda item: item[0], reverse=True)
                candidates = [chunk for _, chunk in scored[:vector_fetch_limit]]
                raw_vector_candidates = list(candidates)
            if not raw_all_candidates:
                raw_all_candidates = list(all_candidates or candidates)
            if all_candidates is not None:
                all_candidates, filtered_counts = _filter_reference_candidates(
                    all_candidates,
                    doc=doc,
                    source_chunk=existing_chunk,
                    allow_same_document=allow_same_document,
                    code_focus=code_focus,
                )
            candidates, candidate_filtered_counts = _filter_reference_candidates(
                candidates,
                doc=doc,
                source_chunk=existing_chunk,
                allow_same_document=allow_same_document,
                code_focus=code_focus,
            )
            for reason, count in candidate_filtered_counts.items():
                filtered_counts[reason] = max(filtered_counts.get(reason, 0), count)
            candidate_count = len(all_candidates) if all_candidates is not None else len(candidates)
            if code_focus:
                lexical_candidates = _lexical_reference_candidates(
                    query_text,
                    all_candidates or [],
                    limit=candidate_limit,
                    code_focus=True,
                )
                anchor_candidates = _anchor_reference_candidates(
                    query_text,
                    all_candidates or [],
                    limit=candidate_limit * 2,
                    code_focus=True,
                )
                candidates = _merge_reference_candidates(candidates, lexical_candidates, merge_limit)
                candidates = _merge_reference_candidates(candidates, anchor_candidates, merge_limit)
            references = _rerank_reference_chunks(
                query_text,
                candidates,
                code_focus=code_focus,
                doc=doc,
                raw_vector_candidates=raw_vector_candidates,
                lexical_candidates=lexical_candidates,
                anchor_candidates=anchor_candidates,
            )
            if _reference_trace_enabled():
                log_event(
                    "feedback_reference_trace",
                    document_id=doc.document_id,
                    source_chunk_id=existing_chunk.chunk_id,
                    source_path=_extract_snapshot_file_path(text),
                    query_path=_extract_snapshot_file_path(query_text),
                    code_focus=code_focus,
                    allow_same_document=allow_same_document,
                    raw_candidate_count=raw_candidate_count,
                    candidate_count=candidate_count,
                    lexical_candidate_count=len(lexical_candidates),
                    anchor_candidate_count=len(anchor_candidates),
                    selected_reference_count=len(references),
                    candidates=_reference_trace_entries(
                        query_text=query_text,
                        source_chunk=existing_chunk,
                        doc=doc,
                        raw_candidates=raw_all_candidates,
                        raw_vector_candidates=raw_vector_candidates,
                        allow_same_document=allow_same_document,
                        code_focus=code_focus,
                        lexical_candidates=lexical_candidates,
                        anchor_candidates=anchor_candidates,
                        selected_references=references,
                    ),
                )
            if not references:
                log_event(
                    "feedback_no_references",
                    document_id=doc.document_id,
                    source_chunk_id=existing_chunk.chunk_id,
                    group_ids=doc_group_ids,
                    code_focus=code_focus,
                    allow_same_document=allow_same_document,
                    raw_candidate_count=raw_candidate_count,
                    candidate_count=candidate_count,
                    lexical_candidate_count=len(lexical_candidates),
                    anchor_candidate_count=len(anchor_candidates),
                    filtered_header_only=filtered_counts.get("header_only", 0),
                    filtered_same_file=filtered_counts.get("same_file", 0),
                    filtered_same_document=filtered_counts.get("same_document_disabled", 0),
                    filtered_same_chunk=filtered_counts.get("same_chunk", 0),
                )
            else:
                log_event(
                    "feedback_references_selected",
                    document_id=doc.document_id,
                    source_chunk_id=existing_chunk.chunk_id,
                    group_ids=doc_group_ids,
                    code_focus=code_focus,
                    allow_same_document=allow_same_document,
                    raw_candidate_count=raw_candidate_count,
                    candidate_count=candidate_count,
                    lexical_candidate_count=len(lexical_candidates),
                    anchor_candidate_count=len(anchor_candidates),
                    selected_reference_count=len(references),
                    filtered_header_only=filtered_counts.get("header_only", 0),
                    filtered_same_file=filtered_counts.get("same_file", 0),
                    filtered_same_document=filtered_counts.get("same_document_disabled", 0),
                    filtered_same_chunk=filtered_counts.get("same_chunk", 0),
                )

        sql_references: list[Reference] = []
        for ref_chunk in references:
            sql_references.append(
                Reference(
                    source_chunk_id=existing_chunk.chunk_id,
                    reference_chunk_id=ref_chunk.chunk_id,
                    reference_type="document",
                    reference_document_id=ref_chunk.document_id,
                    reference_note_id=None,
                )
            )

        if sql_references:
            session.add_all(sql_references)
            session.commit()
        if not references:
            return

        feedback = get_feedback(
            reviewer,
            doc,
            text,
            references,
            user,
            focus_text=focus_text,
            change_context=change_context,
        )
        feedback_items = split_feedback_items(feedback)
        if not feedback_items:
            log_event(
                "feedback_generation_none",
                document_id=doc.document_id,
                source_chunk_id=existing_chunk.chunk_id,
                code_focus=code_focus,
                provider=getattr(reviewer, "provider", "unknown"),
                reference_count=len(references),
            )
            return
        if len(feedback_items) > 1:
            log_event(
                "feedback_generation_multiple",
                document_id=doc.document_id,
                source_chunk_id=existing_chunk.chunk_id,
                code_focus=code_focus,
                finding_count=len(feedback_items),
            )
        sql_feedback_entries: list[Feedback] = []
        for item_feedback in feedback_items:
            sql_feedback = Feedback(
                source_chunk_id=existing_chunk.chunk_id,
                feedback=item_feedback,
                model=reviewer.model,
            )
            session.add(sql_feedback)
            sql_feedback_entries.append(sql_feedback)
        session.commit()
        try:
            from .notifications.service import (
                NotificationCandidate,
                PeerCandidate,
                is_scoring_enabled,
                score_and_route_candidate,
            )

            if is_scoring_enabled() and doc.groups:
                peer_candidates: list[PeerCandidate] = []
                for ref_chunk in references:
                    ref_doc = ref_chunk.document
                    ref_user = ref_doc.user if ref_doc else None
                    peer_candidates.append(
                        PeerCandidate(
                            doc_id=ref_doc.document_id if ref_doc else "",
                            doc_title=ref_doc.title if ref_doc else "",
                            chunk_id=ref_chunk.chunk_id,
                            chunk_text=ref_chunk.content,
                            doc_type=ref_doc.doc_type if ref_doc else "",
                            author_role=ref_user.role if ref_user and ref_user.role else "",
                            author_team="",
                            last_modified_utc=(
                                ref_doc.datetime_modified.isoformat()
                                if ref_doc and ref_doc.datetime_modified
                                else None
                            ),
                            similarity=None,
                        )
                    )
                if peer_candidates:
                    for sql_feedback in sql_feedback_entries:
                        candidate = NotificationCandidate(
                            user_id=user.user_id if user else doc.user_id,
                            group_id=doc.groups[0].group_id,
                            target_doc_id=doc.document_id,
                            target_chunk_id=existing_chunk.chunk_id,
                            target_text=text,
                            target_doc_title=doc.title or "",
                            target_doc_type=doc.doc_type or "",
                            target_last_modified_utc=(
                                doc.datetime_modified.isoformat() if doc.datetime_modified else None
                            ),
                            user_role=user.role if user and user.role else "",
                            user_team="",
                            user_is_doc_author=True,
                            user_is_group_admin=False,
                            peer_candidates=tuple(peer_candidates),
                            generated_feedback={
                                "summary": sql_feedback.feedback,
                                "focus_text": focus_text,
                                "change_context": change_context,
                            },
                            run_id=f"feedback_{sql_feedback.feedback_id}",
                            now_utc=datetime.now(timezone.utc),
                        )
                        score_and_route_candidate(
                            session,
                            candidate,
                            commit=True,
                            delivery_channel="inbox_only",
                        )
        except Exception as exc:
            logger.warning("Notification scoring failed: %s", exc)


def remove_text(session: SASession, text: str, document_id: str) -> None:
    chunks = session.query(Chunk).filter(
        Chunk.document_id == document_id,
        Chunk.content == text,
    )
    chunks.delete(synchronize_session=False)
    session.commit()


def get_all_chunks_for_document(session: SASession, doc: Document) -> list[Chunk]:
    doc_chunks = session.query(Chunk).filter(Chunk.document_id == doc.document_id).all()
    note_chunks: list[Chunk] = []
    notes = session.query(Note).filter(Note.document_id == doc.document_id).all()
    for note in notes:
        note_text_chunks = chunk_text_with_mode(note.content)
        for text in note_text_chunks:
            chunk_hash = stable_chunk_hash(text)
            existing = session.query(Chunk).filter(
                Chunk.document_id == doc.document_id,
                Chunk.content == text,
            ).first()
            if not existing:
                embedding = create_embedding(Embedder(), text, user=doc.author_id)
                note_chunk = Chunk(
                    hash=str(chunk_hash),
                    document_id=doc.document_id,
                    content=text,
                    embedding=embedding,
                )
                session.add(note_chunk)
                session.commit()
                note_chunks.append(note_chunk)
            else:
                if existing.hash != chunk_hash:
                    existing.hash = chunk_hash
                    session.commit()
                note_chunks.append(existing)
    return doc_chunks + note_chunks
