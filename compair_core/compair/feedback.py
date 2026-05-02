from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
import re
import time
from typing import Any, Iterable, List

import requests

from .bundle_review import (
    FINDINGS_JSON_SCHEMA,
    SYSTEM_PROMPT_TEMPLATE,
    USER_PROMPT_TEMPLATE,
    build_document_bundle,
    estimate_tokens,
    estimate_usage_cost,
    extract_json_object,
    normalize_findings_payload,
    render_now_review_markdown,
)
from .logger import log_event
from .local_summary import (
    ReferenceText,
    best_reference_match,
    extract_artifacts,
    reference_payload_texts,
    summarize_reference_feedback,
)
from .models import Document, User

try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    openai = None  # type: ignore

try:
    from compair_cloud.feedback import Reviewer as CloudReviewer  # type: ignore
    from compair_cloud.feedback import get_feedback as cloud_get_feedback  # type: ignore
except (ImportError, ModuleNotFoundError):
    CloudReviewer = None  # type: ignore
    cloud_get_feedback = None  # type: ignore


_REASONING_PREFIXES = ("gpt-5", "o1", "o2", "o3", "o4")
_CODE_REVIEW_DOC_TYPE = "code-repo"
_SNAPSHOT_FILE_RE = re.compile(r"^### File:\s+(.+?)(?:\s+\(.*)?$", re.MULTILINE)
_FINDING_SEPARATOR = "<<<FINDING>>>"
logger = logging.getLogger(__name__)


def _openai_api_key() -> str | None:
    return os.getenv("COMPAIR_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")


def _openai_base_url() -> str | None:
    return os.getenv("COMPAIR_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")


def _openai_sdk_max_retries() -> int:
    raw = os.getenv("COMPAIR_OPENAI_SDK_MAX_RETRIES") or os.getenv("OPENAI_SDK_MAX_RETRIES")
    try:
        return max(0, int(raw)) if raw is not None else 0
    except ValueError:
        return 0


def _make_openai_client(api_key: str | None) -> Any:
    kwargs: dict[str, Any] = {"max_retries": _openai_sdk_max_retries()}
    base_url = _openai_base_url()
    effective_api_key = api_key
    if base_url and not effective_api_key:
        effective_api_key = "compair-local"
    if effective_api_key:
        kwargs["api_key"] = effective_api_key
    if base_url:
        kwargs["base_url"] = base_url
    try:  # pragma: no cover - optional dependency differences
        return openai.OpenAI(**kwargs)  # type: ignore[attr-defined]
    except TypeError:
        kwargs.pop("max_retries", None)
        return openai.OpenAI(**kwargs)  # type: ignore[attr-defined]


def _is_reasoning_model_name(model_name: str | None) -> bool:
    if not model_name:
        return False
    normalized = model_name.lower()
    for prefix in _REASONING_PREFIXES:
        if normalized == prefix or normalized.startswith(f"{prefix}-") or normalized.startswith(f"{prefix}."):
            return True
    return False


def _get_field(source: Any, key: str) -> Any:
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage_int(usage: Any, key: str) -> int | None:
    value = _get_field(usage, key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _is_code_review_document(doc: Document, text: str) -> bool:
    doc_type = (getattr(doc, "doc_type", "") or "").strip().lower()
    if doc_type == _CODE_REVIEW_DOC_TYPE:
        return True
    return "### File:" in (text or "") and "```" in (text or "")


def _is_snapshot_metadata_chunk(text: str) -> bool:
    stripped = (text or "").lstrip()
    return stripped.startswith("# Compair baseline snapshot") or stripped.startswith("## Snapshot limits")


def _extract_snapshot_file_path(text: str) -> str:
    match = _SNAPSHOT_FILE_RE.search(text or "")
    if not match:
        return ""
    return (match.group(1) or "").strip()


def _is_doc_like_path(path: str) -> bool:
    normalized = (path or "").replace("\\", "/").strip().lower()
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


def _is_doc_like_snapshot_chunk(text: str) -> bool:
    return _is_doc_like_path(_extract_snapshot_file_path(text))


def _format_changed_chunk_prompt(
    text: str,
    focus_text: str = "",
    *,
    primary_label: str = "Primary changed excerpt",
    secondary_label: str = "Surrounding chunk context (secondary)",
) -> str:
    full_text = (text or "").strip()
    focused = (focus_text or "").strip()
    if not full_text:
        return ""
    if not focused or focused == full_text:
        return f"{primary_label}:\n{full_text}"
    return (
        f"{primary_label}:\n{focused}\n\n"
        f"{secondary_label}:\n{full_text}"
    )


def _max_findings_per_chunk(*, code_review: bool) -> int:
    env_value = os.getenv("COMPAIR_MAX_FINDINGS_PER_CHUNK")
    default = 2 if code_review else 1
    try:
        value = int(env_value) if env_value else default
    except ValueError:
        value = default
    return max(1, min(value, 5))


def split_feedback_items(feedback: str, *, max_items: int | None = None) -> list[str]:
    text = (feedback or "").strip()
    if not text or text.upper() == "NONE":
        return []
    parts = [part.strip() for part in re.split(rf"\s*{re.escape(_FINDING_SEPARATOR)}\s*", text) if part.strip()]
    if not parts:
        return []
    limit = max_items if max_items is not None else len(parts)
    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = " ".join(part.split()).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(part)
        if len(deduped) >= limit:
            break
    return deduped


def _finding_prompt_instructions(max_findings: int) -> str:
    if max_findings <= 1:
        return "- Return a single compact paragraph."
    return (
        f"- If there are multiple distinct, well-supported observations, return up to {max_findings} findings.\n"
        f"- Separate distinct findings with a standalone line containing exactly: {_FINDING_SEPARATOR}\n"
        "- Only include multiple findings when they rely on meaningfully different evidence artifacts or product surfaces.\n"
        "- Each finding must be a compact paragraph with no bullets or headings."
    )


def _should_retry_without_reasoning(exc: Exception) -> bool:
    message = str(exc or "").lower()
    if "reasoning" not in message:
        return False
    unsupported_markers = (
        "unsupported",
        "not supported",
        "unknown parameter",
        "unexpected",
        "invalid parameter",
        "extra inputs are not permitted",
        "not permitted",
    )
    return any(marker in message for marker in unsupported_markers)


class Reviewer:
    """Edition-aware wrapper that selects a feedback provider based on configuration."""

    def __init__(self) -> None:
        self.edition = os.getenv("COMPAIR_EDITION", "core").lower()
        self.provider = os.getenv("COMPAIR_GENERATION_PROVIDER", "local").lower()
        self.length_map = {
            "Brief": "Respond in 1-2 short sentences focused on the single highest-signal observation.",
            "Detailed": "Respond in 3-5 concise sentences covering the strongest observation, why it matters, and what to verify or consider next.",
            "Verbose": "Respond in 6-8 concise sentences covering the strongest observations, likely impact, and concrete follow-up checks without repeating yourself.",
        }

        self._cloud_impl = None
        self._openai_client = None
        self.openai_model = os.getenv("COMPAIR_OPENAI_MODEL", "gpt-5.4-mini")
        self.code_openai_model = os.getenv("COMPAIR_OPENAI_CODE_MODEL", self.openai_model)
        self.openai_reasoning_effort = os.getenv("COMPAIR_OPENAI_REASONING_EFFORT", "minimal")
        self.uses_reasoning_model = _is_reasoning_model_name(self.openai_model)
        self.custom_endpoint = os.getenv("COMPAIR_GENERATION_ENDPOINT")

        if self.edition == "cloud" and CloudReviewer is not None:
            self._cloud_impl = CloudReviewer()
            self.provider = "cloud"
        else:
            if self.provider == "openai":
                api_key = _openai_api_key()
                if api_key and openai is not None:
                    # Support both legacy (ChatCompletion) and new SDKs
                    if hasattr(openai, "api_key"):
                        openai.api_key = api_key  # type: ignore[assignment]
                    if hasattr(openai, "OpenAI"):
                        try:  # pragma: no cover - optional runtime dependency
                            self._openai_client = _make_openai_client(api_key)
                        except Exception:  # pragma: no cover - if instantiation fails
                            self._openai_client = None
                if self._openai_client is None and not hasattr(openai, "ChatCompletion"):
                    log_event("openai_feedback_unavailable", reason="openai_library_missing")
                    self.provider = "fallback"
            if self.provider == "http" and not self.custom_endpoint:
                log_event("custom_feedback_unavailable", reason="missing_endpoint")
                self.provider = "fallback"
            if self.provider == "local":
                self.model = os.getenv("COMPAIR_LOCAL_GENERATION_MODEL", "local-feedback")
                base_url = os.getenv("COMPAIR_LOCAL_MODEL_URL", "http://127.0.0.1:9000")
                route = os.getenv("COMPAIR_LOCAL_GENERATION_ROUTE", "/generate")
                self.endpoint = f"{base_url.rstrip('/')}{route}"
            else:
                self.model = "external"
                self.endpoint = None
            if self.provider not in {"local", "openai", "http", "fallback"}:
                log_event("feedback_provider_unknown", provider=self.provider)
                self.provider = "fallback"

    @property
    def is_cloud(self) -> bool:
        return self._cloud_impl is not None


def _reference_snippets(references: Iterable[Any], limit: int = 4) -> List[str]:
    return reference_payload_texts(_local_references(list(references)[:limit]), limit=limit)


def _grounded_reference_context(
    text: str,
    references: list[Any],
    *,
    focus_text: str = "",
    change_context: str = "",
) -> str:
    match = best_reference_match(
        text,
        _local_references(references),
        focus_text=focus_text,
        change_context=change_context,
    )
    if match is None:
        return ""
    lines = [
        "Changed excerpt:",
        match.target_excerpt,
        "",
        f"Related excerpt from {match.reference_label}:",
        match.peer_excerpt,
    ]
    if match.relation.kind:
        lines.extend(["", f"Likely relation: {match.relation.kind}"])
    return "\n".join(lines).strip()


def _compact_local_excerpt(text: str, *, limit: int = 140) -> str:
    excerpt = " ".join((text or "").split())
    if len(excerpt) <= limit:
        return excerpt
    return excerpt[: max(0, limit - 1)].rstrip() + "…"


def _generic_local_quality_note(reference_label: str) -> str:
    return (
        f"Possible cross-repo drift detected between the changed excerpt and {reference_label}, "
        "but the bundled local review path could not ground a more specific summary."
    )


def _normalized_local_artifact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _looks_like_local_import_or_header(text: str) -> bool:
    candidate = (text or "").strip().lower()
    if not candidate:
        return True
    return candidate.startswith("#") or candidate.startswith("import ") or candidate.startswith("from ")


def _local_structured_overlap(target_excerpt: str, peer_excerpt: str) -> tuple[set[str], set[str], set[str], set[str]]:
    target_profile = extract_artifacts(target_excerpt)
    peer_profile = extract_artifacts(peer_excerpt)
    shared_keys = set(target_profile.key_names & peer_profile.key_names)
    shared_quotes = set(target_profile.quoted_norm & peer_profile.quoted_norm)
    shared_paths = set(target_profile.path_tokens & peer_profile.path_tokens)
    shared_tokens = set(target_profile.tokens & peer_profile.tokens)
    return shared_keys, shared_quotes, shared_paths, shared_tokens


def _local_docs_impl_focus(target_excerpt: str, peer_excerpt: str) -> str:
    shared_keys, shared_quotes, shared_paths, shared_tokens = _local_structured_overlap(target_excerpt, peer_excerpt)
    if shared_keys:
        return sorted(shared_keys)[0]
    if shared_quotes:
        return sorted(shared_quotes)[0]
    if shared_paths:
        return sorted(shared_paths)[0]
    if shared_tokens:
        return sorted(shared_tokens)[0]
    return ""


def _should_surface_local_reference_match(match: Any) -> bool:
    relation = getattr(match, "relation", None)
    if relation is None:
        return False

    kind = getattr(relation, "kind", None)
    confidence = int(getattr(relation, "confidence", 0) or 0)
    target_artifact = (getattr(relation, "target_artifact", "") or "").strip()
    peer_artifact = (getattr(relation, "peer_artifact", "") or "").strip()
    target_excerpt = getattr(match, "target_excerpt", "") or ""
    peer_excerpt = getattr(match, "peer_excerpt", "") or ""

    if kind is None or kind == "generic divergence" or confidence <= 1:
        return False
    if not target_excerpt or not peer_excerpt:
        return False

    if kind == "rename":
        if _normalized_local_artifact(target_artifact) == _normalized_local_artifact(peer_artifact):
            return False
        return False

    if kind == "docs-vs-impl mismatch":
        if _looks_like_local_import_or_header(target_excerpt) or _looks_like_local_import_or_header(peer_excerpt):
            return False
        shared_keys, shared_quotes, shared_paths, shared_tokens = _local_structured_overlap(target_excerpt, peer_excerpt)
        return bool(shared_keys or shared_quotes or shared_paths or len(shared_tokens) >= 5)

    if kind == "presence/absence":
        if not target_artifact:
            return False
        if _normalized_local_artifact(target_artifact) in {"settings", "config", "summary", "title", "readme"}:
            return False

    return True


def _render_local_reference_match(match: Any) -> str | None:
    relation = getattr(match, "relation", None)
    reference_label = getattr(match, "reference_label", "a related reference")
    target_excerpt = getattr(match, "target_excerpt", "") or ""
    peer_excerpt = getattr(match, "peer_excerpt", "") or ""
    if relation is None or not _should_surface_local_reference_match(match):
        return None

    kind = getattr(relation, "kind", None)
    confidence = int(getattr(relation, "confidence", 0) or 0)
    target_artifact = (getattr(relation, "target_artifact", "") or "").strip()
    peer_artifact = (getattr(relation, "peer_artifact", "") or "").strip()

    if kind is None or kind == "generic divergence" or confidence <= 1:
        return None

    if kind == "docs-vs-impl mismatch":
        focus = _local_docs_impl_focus(target_excerpt, peer_excerpt)
        focus_prefix = f' around "{focus}"' if focus else ""
        return (
            f'Possible docs-vs-implementation drift{focus_prefix}: the changed excerpt says '
            f'"{_compact_local_excerpt(target_excerpt)}", while related implementation or '
            f'config in {reference_label} indicates otherwise.'
        )

    if kind == "route/path mismatch" and target_artifact and peer_artifact:
        return (
            f'Possible route/path drift: the changed excerpt uses "{target_artifact}", '
            f'while {reference_label} uses "{peer_artifact}".'
        )

    if kind == "value mismatch" and target_artifact and peer_artifact:
        return (
            f'Possible value drift: the changed excerpt uses "{target_artifact}", '
            f'while {reference_label} uses "{peer_artifact}".'
        )

    if kind == "rename" and target_artifact and peer_artifact:
        return (
            f'Possible rename drift: the changed excerpt uses "{target_artifact}", '
            f'while {reference_label} still uses "{peer_artifact}".'
        )

    if kind == "presence/absence" and target_artifact:
        target_profile = extract_artifacts(target_excerpt)
        if target_artifact.startswith("/") and len(target_profile.paths) > 1:
            return None
        return (
            f'Possible cross-repo drift: the changed excerpt includes "{target_artifact}", '
            f'but {reference_label} does not show a corresponding item.'
        )

    rendered = summarize_reference_feedback(
        target_excerpt,
        [ReferenceText(label=reference_label, text=peer_excerpt)],
    )
    if not rendered:
        return None
    if len(rendered) > 360:
        return None
    return rendered


def _local_reference_feedback_text(
    text: str,
    references: list[Any],
    *,
    focus_text: str = "",
    change_context: str = "",
) -> str | None:
    match = best_reference_match(
        text,
        _local_references(references),
        focus_text=focus_text,
        change_context=change_context,
    )
    if match is None:
        return None
    return _render_local_reference_match(match)


def _fallback_feedback(
    text: str,
    references: list[Any],
    *,
    focus_text: str = "",
    change_context: str = "",
) -> str:
    summary = _local_reference_feedback_text(
        text,
        references,
        focus_text=focus_text,
        change_context=change_context,
    )
    return summary or "NONE"


def _bundle_review_model_name(reviewer: Reviewer, *, override: str | None = None) -> str:
    if override and override.strip():
        return override.strip()
    for candidate in (
        getattr(reviewer, "code_openai_model", None),
        getattr(reviewer, "openai_model", None),
        getattr(reviewer, "code_model", None),
        getattr(reviewer, "model", None),
        os.getenv("COMPAIR_OPENAI_CODE_MODEL"),
        os.getenv("COMPAIR_OPENAI_MODEL"),
    ):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return "gpt-5.4"


def _bundle_review_openai_client(reviewer: Reviewer) -> Any:
    client = getattr(reviewer, "_openai_client", None)
    if client is not None:
        return client
    if openai is None or not hasattr(openai, "OpenAI"):
        return None
    try:
        client = _make_openai_client(_openai_api_key())
    except Exception:  # pragma: no cover - runtime dependency variance
        return None
    reviewer._openai_client = client
    return client


def _completion_text(response: Any) -> str | None:
    if response is None:
        return None
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None
    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
    text = getattr(choice, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def review_documents_now(
    reviewer: Reviewer,
    documents: list[Document],
    user: User,
    *,
    group_name: str,
    max_findings: int = 12,
    model_override: str | None = None,
) -> dict[str, Any]:
    if not documents:
        raise ValueError("At least one document is required for now review")
    client = _bundle_review_openai_client(reviewer)
    if client is None:
        raise RuntimeError(
            "Now review requires an OpenAI-compatible generation client. "
            "Set COMPAIR_OPENAI_MODEL and optionally COMPAIR_OPENAI_BASE_URL."
        )

    doc_stats, bundle_text = build_document_bundle(documents)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(max_findings=max(1, min(max_findings, 12)))
    user_prompt = USER_PROMPT_TEMPLATE.format(bundle=bundle_text)
    prompt_text = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"
    model_name = _bundle_review_model_name(reviewer, override=model_override)
    uses_reasoning = _is_reasoning_model_name(model_name)

    started_at = time.time()
    response: Any = None
    raw_text: str | None = None
    response_mode = "responses"
    request_kwargs: dict[str, Any] = {"model": model_name}
    if uses_reasoning:
        request_kwargs["input"] = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if getattr(reviewer, "openai_reasoning_effort", None):
            request_kwargs["reasoning"] = {"effort": reviewer.openai_reasoning_effort}
    else:
        request_kwargs["instructions"] = system_prompt
        request_kwargs["input"] = user_prompt
    request_kwargs["text"] = {
        "format": {
            "type": "json_schema",
            "name": "compair_now_review_findings",
            "schema": FINDINGS_JSON_SCHEMA,
            "strict": True,
        }
    }

    last_exc: Exception | None = None
    try:
        if hasattr(client, "responses"):
            try:
                response = client.responses.create(**request_kwargs)
            except Exception as exc:
                if "reasoning" in request_kwargs and _should_retry_without_reasoning(exc):
                    reduced = dict(request_kwargs)
                    reduced.pop("reasoning", None)
                    response = client.responses.create(**reduced)
                else:
                    raise
            raw_text = _extract_response_text(response, reasoning_mode=uses_reasoning)
        else:
            response_mode = "chat"
    except Exception as exc:
        last_exc = exc
        response_mode = "chat"

    if not raw_text and hasattr(client, "chat") and hasattr(client.chat, "completions"):
        response_mode = "chat"
        chat_kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt + "\n\nReturn JSON only."},
            ],
            "temperature": 0.2,
            "max_tokens": 2200,
        }
        try:
            response = client.chat.completions.create(  # type: ignore[call-arg]
                **chat_kwargs,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = client.chat.completions.create(**chat_kwargs)  # type: ignore[call-arg]
        raw_text = _completion_text(response)

    if not raw_text:
        raise RuntimeError(f"Now review failed to produce content: {last_exc or 'empty response'}")

    parsed = normalize_findings_payload(extract_json_object(raw_text))
    usage = _get_field(response, "usage")
    duration_sec = round(time.time() - started_at, 3)
    meta = {
        "model": model_name,
        "response_mode": response_mode,
        "duration_sec": duration_sec,
        "response_id": _get_field(response, "id"),
        "bundle_estimated_tokens": estimate_tokens(bundle_text),
        "prompt_estimated_tokens": estimate_tokens(prompt_text),
        "usage": {
            "input_tokens": _usage_int(usage, "input_tokens"),
            "output_tokens": _usage_int(usage, "output_tokens"),
            "total_tokens": _usage_int(usage, "total_tokens"),
        },
        "cost_estimate_usd": estimate_usage_cost(
            model=model_name,
            input_tokens=_usage_int(usage, "input_tokens"),
            output_tokens=_usage_int(usage, "output_tokens"),
            prompt_estimated_tokens=estimate_tokens(prompt_text),
        ),
        "created_at": _utc_now(),
    }
    markdown = render_now_review_markdown(
        group_name=group_name,
        findings=parsed.get("findings", []),
        meta=meta,
        document_stats=doc_stats,
    )
    log_event(
        "openai_bundle_review_created",
        model=model_name,
        response_mode=response_mode,
        input_tokens=meta["usage"]["input_tokens"],
        output_tokens=meta["usage"]["output_tokens"],
        total_cost_usd=(meta["cost_estimate_usd"] or {}).get("total_cost_usd") if isinstance(meta.get("cost_estimate_usd"), dict) else None,
        duration_sec=duration_sec,
        document_count=len(documents),
        finding_count=len(parsed.get("findings", [])),
        created_at=_utc_now(),
        user_id=getattr(user, "user_id", None),
    )
    return {
        "group_name": group_name,
        "documents": doc_stats,
        "findings": parsed.get("findings", []),
        "meta": meta,
        "markdown": markdown,
        "bundle_text": bundle_text,
    }



def _local_reference_feedback(
    reviewer: Reviewer,
    text: str,
    references: list[Any],
    user: User,
    *,
    focus_text: str = "",
    change_context: str = "",
) -> str | None:
    return _local_reference_feedback_text(
        text,
        references,
        focus_text=focus_text,
        change_context=change_context,
    )


def _local_references(references: list[Any]) -> list[ReferenceText]:
    local_references: list[ReferenceText] = []
    for ref in references[:6]:
        snippet = getattr(ref, "content", "") or getattr(ref, "text", "")
        snippet = snippet.strip()
        if not snippet:
            continue
        doc = getattr(ref, "document", None)
        title = getattr(doc, "title", None) or "a related reference"
        path = _extract_snapshot_file_path(snippet)
        label = path or title
        if path and title and title != path:
            label = f"{title} [{path}]"
        local_references.append(ReferenceText(label=label, text=snippet))
    return local_references


def _openai_feedback(
    reviewer: Reviewer,
    doc: Document,
    text: str,
    references: list[Any],
    user: User,
    *,
    focus_text: str = "",
    change_context: str = "",
) -> str | None:
    if openai is None:
        return None
    started_at = time.time()
    instruction = reviewer.length_map.get(user.preferred_feedback_length, "1–2 short sentences")
    ref_text = "\n\n".join(_reference_snippets(references, limit=4))
    grounded_context = _grounded_reference_context(
        text,
        references,
        focus_text=focus_text,
        change_context=change_context,
    )
    is_code_review = _is_code_review_document(doc, text)
    is_doc_like = is_code_review and _is_doc_like_snapshot_chunk(text)
    max_findings = _max_findings_per_chunk(code_review=is_code_review)
    finding_prompt_instructions = _finding_prompt_instructions(max_findings if is_code_review else 1)
    if is_code_review and _is_snapshot_metadata_chunk(text):
        return "NONE"
    if is_doc_like:
        system_prompt = f"""# Identity
You are a cross-repo review assistant inside Compair. You compare repository documentation, API maps, config notes, and nearby implementation snippets to find concrete product-surface drift across repos.

# Purpose
Your goal is to identify a specific, evidence-backed mismatch between the changed document chunk and the related repo excerpts: feature-surface drift, docs-vs-implementation drift, auth or capability drift, route or endpoint drift, config/env drift, or rollout/status drift.

# Instructions

- Prioritize the single strongest cross-repo mismatch that is directly supported by the changed chunk and the references.
- If a primary changed excerpt is provided separately from surrounding chunk context, analyze the primary excerpt first and use the larger chunk only as supporting context.
- Treat documentation drift as valid when one repo claims behavior or availability that the related implementation/docs in other repos contradict.
- Mention concrete paths, endpoints, capability fields, auth modes, env vars, or product-surface claims when the evidence supports them.
- Do not fall back to vague architectural commentary or “verify everything” summaries.
- Stay grounded. If the references are too weak to support a concrete mismatch, respond with: **NONE**.
- Prefer one finding unless the changed chunk clearly contains multiple distinct, well-supported drifts.
{finding_prompt_instructions}
- Length: {instruction}
- Structure: Use one compact paragraph, not bullet lists or headings.

# Output Format
- If no concrete cross-repo mismatch is supported: **NONE**
        """
        changed_chunk_prompt = _format_changed_chunk_prompt(text, focus_text)
        change_block = f"\n\nRecent change summary (before/after):\n{change_context}" if change_context else ""
        evidence_block = f"\n\nStrongest grounded evidence:\n{grounded_context}" if grounded_context else ""
        user_prompt = (
            f"{changed_chunk_prompt}{change_block}{evidence_block}\n\nRelated repository chunks:\n{ref_text or 'None provided'}\n\n"
            f"{instruction} Focus on the strongest supported product-surface or docs-vs-implementation mismatch."
        )
    elif is_code_review:
        system_prompt = f"""# Identity
You are a code review assistant inside Compair. You compare chunks from repository snapshots and surface concrete implementation mismatches, integration risks, information gaps, or non-obvious overlaps between repos.

# Purpose
Your goal is to identify specific, evidence-backed code observations that matter across repositories: API drift, route/query mismatches, schema/config/env divergence, duplicated-but-divergent logic, missing downstream updates, meaningful hidden overlap, or a key information gap that only becomes visible across repos.

# Instructions

- Prioritize concrete implementation observations over broad architectural summaries.
- If a primary changed excerpt is provided separately from surrounding chunk context, analyze the primary excerpt first and use the larger chunk only as supporting context.
- Mention specific file paths, endpoints, env vars, settings, or interfaces when the evidence supports it.
- When the evidence points to hidden overlap, reinforcement, or an information gap rather than a conflict, say that plainly instead of forcing issue/fix framing.
- Stay grounded: only make claims that are directly supported by the provided chunk and references. If evidence is partial, say what to verify next instead of asserting it as fact.
- Do not suggest direct package dependencies across decoupled services unless the evidence clearly shows that is intended.
- Ignore weak signals like repo descriptions, version numbers, or stack recaps unless they imply a real compatibility issue.
- Prefer one finding unless the changed chunk clearly contains multiple distinct, well-supported issues.
{finding_prompt_instructions}
- Length: {instruction}
- Structure: Use one compact paragraph, not bullet lists or headings.
- If there is no concrete, code-focused observation, respond with: **NONE**.

# Output Format
- If no meaningful code-focused observation stands out: **NONE**
        """
        changed_chunk_prompt = _format_changed_chunk_prompt(text, focus_text)
        change_block = f"\n\nRecent change summary (before/after):\n{change_context}" if change_context else ""
        evidence_block = f"\n\nStrongest grounded evidence:\n{grounded_context}" if grounded_context else ""
        user_prompt = (
            f"{changed_chunk_prompt}{change_block}{evidence_block}\n\nRelated repository chunks:\n{ref_text or 'None provided'}\n\n"
            f"{instruction} Prefer one concrete observation."
        )
    else:
        system_prompt = f"""# Identity
You are a collaborative team member on Compair, a platform designed to help teammates uncover connections, share insights, and accelerate collective learning by comparing user documents with relevant references.

# Purpose
Your goal is to quickly surface **meaningful** connections or useful contrasts between a user’s main document and shared references—especially details that could help the document author or other team members work more effectively together.

# Instructions

- **Connect the Dots:** Highlight unique insights, similarities, differences, or answers between the main document and its references. Prioritize information that is truly meaningful or helpful to the author or team.
- **Qualified Sharing:** Only point out connections that matter—avoid commenting on trivial or already-obvious overlapping details. If nothing significant stands out, respond with: **NONE**.
- **Relay Messages:** If user documents or notes are being used to communicate with teammates, relay any important updates or questions to help foster further discussion or action.
- **Length:** {instruction}
- **Style:** Use one compact paragraph in a friendly, direct tone—never formal or repetitive.
- **Be Constructive:** Focus on actionable insights, especially those that could inform or inspire team decisions, workflow improvements, or new ideas.

# Output Format
- If no meaningful connections or insights are present: **NONE**

# Be sure NOT to:
- Repeat the user’s content back without adding value.
- Offer generic praise or vague observations.
- Use overly technical or robotic language.
    """
        user_prompt = (
            f"Document:\n{text}\n\nRelevant reference excerpts:\n{ref_text or 'None provided'}\n\n"
            f"Respond with {instruction}."
        )

    def _extract_response_text(response: Any, reasoning_mode: bool) -> str | None:
        if response is None:
            return None
        text_out = _get_field(response, "output_text")
        if isinstance(text_out, str) and text_out.strip():
            return text_out.strip()
        outputs = _get_field(response, "output") or _get_field(response, "outputs")
        pieces: list[str] = []
        if outputs:
            for item in outputs:
                item_type = _get_field(item, "type")
                if reasoning_mode and item_type and item_type not in {"message", "assistant"}:
                    continue
                content_field = _get_field(item, "content")
                if not content_field:
                    continue
                for part in content_field:
                    part_type = _get_field(part, "type")
                    if reasoning_mode and part_type and part_type not in {"output_text", "text"}:
                        continue
                    val = _get_field(part, "text") or _get_field(part, "output_text")
                    if val:
                        pieces.append(str(val))
                    elif part and not reasoning_mode:
                        pieces.append(str(part))
        if pieces:
            merged = "\n".join(piece.strip() for piece in pieces if piece and str(piece).strip())
            return merged or None
        message = _get_field(response, "message")
        if isinstance(message, dict):
            message_content = message.get("content") or message.get("text")
            if isinstance(message_content, str) and message_content.strip():
                return message_content.strip()
        return None

    try:
        client = reviewer._openai_client
        if client is None and hasattr(openai, "OpenAI"):
            api_key = _openai_api_key()
            client = _make_openai_client(api_key)
            reviewer._openai_client = client

        content: str | None = None
        uses_reasoning = reviewer.uses_reasoning_model
        model_name = reviewer.code_openai_model if is_code_review else reviewer.openai_model
        input_chars = len(text or "")
        reference_count = len(references or [])
        response: Any = None
        if client is not None and hasattr(client, "responses"):
            request_kwargs: dict[str, Any] = {
                "model": model_name,
            }
            if uses_reasoning:
                request_kwargs["input"] = [
                    {"role": "developer", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                if reviewer.openai_reasoning_effort:
                    request_kwargs["reasoning"] = {"effort": reviewer.openai_reasoning_effort}
            else:
                request_kwargs["instructions"] = system_prompt
                request_kwargs["input"] = user_prompt
            try:
                response = client.responses.create(**request_kwargs)
            except Exception as exc:
                if "reasoning" in request_kwargs and _should_retry_without_reasoning(exc):
                    log_event("openai_feedback_reasoning_retry", model=model_name)
                    reduced = dict(request_kwargs)
                    reduced.pop("reasoning", None)
                    response = client.responses.create(**reduced)
                else:
                    raise
            content = _extract_response_text(response, reasoning_mode=uses_reasoning)
        elif client is not None and hasattr(client, "chat") and hasattr(client.chat, "completions"):
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.3,
                max_tokens=256,
            )
            choices = getattr(response, "choices", None) or []
            if choices:
                message = getattr(choices[0], "message", None)
                if message is not None:
                    content = getattr(message, "content", None)
                if not content:
                    content = getattr(choices[0], "text", None)
                if isinstance(content, str):
                    content = content.strip()
        elif hasattr(openai, "ChatCompletion"):
            chat_response = openai.ChatCompletion.create(  # type: ignore[attr-defined]
                model=model_name,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.3,
                max_tokens=256,
            )
            response = chat_response
            content = chat_response["choices"][0]["message"]["content"].strip()  # type: ignore[index, assignment]
        if content:
            usage = _get_field(response, "usage")
            log_event(
                "openai_feedback_created",
                model=model_name,
                input_tokens=_usage_int(usage, "input_tokens"),
                output_tokens=_usage_int(usage, "output_tokens"),
                duration_sec=round(time.time() - started_at, 3),
                input_chars=input_chars,
                reference_count=reference_count,
                is_code_review=is_code_review,
                document_id=getattr(doc, "document_id", None),
                user_id=getattr(user, "user_id", None),
                created_at=_utc_now(),
            )
            return content.strip()
    except Exception as exc:  # pragma: no cover - network/API failure
        log_event("openai_feedback_failed", error=str(exc))
        logger.exception("OpenAI feedback generation failed")
    return None


def _local_feedback(
    reviewer: Reviewer,
    text: str,
    references: list[Any],
    user: User,
    *,
    focus_text: str = "",
    change_context: str = "",
) -> str | None:
    payload = {
        "document": text,
        "references": reference_payload_texts(_local_references(references)),
        "length_instruction": reviewer.length_map.get(
            user.preferred_feedback_length,
            "1–2 short sentences",
        ),
        "focus_text": focus_text or None,
        "change_context": change_context or None,
    }

    try:
        response = requests.post(reviewer.endpoint, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        feedback = data.get("feedback") or data.get("text")
        if feedback:
            return str(feedback).strip()
    except Exception as exc:  # pragma: no cover - network failures stay graceful
        log_event("local_feedback_failed", error=str(exc))
        logger.exception("Local feedback generation failed")

    return None


def _http_feedback(
    reviewer: Reviewer,
    text: str,
    references: list[Any],
    user: User,
    *,
    focus_text: str = "",
    change_context: str = "",
) -> str | None:
    if not reviewer.custom_endpoint:
        return None
    payload = {
        "document": text,
        "references": reference_payload_texts(_local_references(references)),
        "length_instruction": reviewer.length_map.get(
            user.preferred_feedback_length,
            "1–2 short sentences",
        ),
        "focus_text": focus_text or None,
        "change_context": change_context or None,
    }
    try:
        response = requests.post(reviewer.custom_endpoint, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        feedback = data.get("feedback") or data.get("text")
        if isinstance(feedback, str):
            feedback = feedback.strip()
        if feedback:
            return feedback
    except Exception as exc:  # pragma: no cover - network failures stay graceful
        log_event("custom_feedback_failed", error=str(exc))
        logger.exception("Custom feedback generation failed")
    return None


def get_feedback(
    reviewer: Reviewer,
    doc: Document,
    text: str,
    references: list[Any],
    user: User,
    *,
    focus_text: str = "",
    change_context: str = "",
) -> str:
    if _is_code_review_document(doc, text) and _is_snapshot_metadata_chunk(text):
        return "NONE"

    if reviewer.is_cloud and cloud_get_feedback is not None:
        return cloud_get_feedback(
            reviewer._cloud_impl,
            doc,
            text,
            references,
            user,
            focus_text=focus_text,
            change_context=change_context,
        )  # type: ignore[arg-type]

    if reviewer.provider == "openai":
        feedback = _openai_feedback(
            reviewer,
            doc,
            text,
            references,
            user,
            focus_text=focus_text,
            change_context=change_context,
        )
        if feedback:
            return feedback
        if _is_code_review_document(doc, text):
            if _is_doc_like_snapshot_chunk(text):
                fallback_feedback = _local_reference_feedback(
                    reviewer,
                    text,
                    references,
                    user,
                    focus_text=focus_text,
                    change_context=change_context,
                )
                if fallback_feedback:
                    log_event(
                        "openai_feedback_doc_fallback",
                        document_id=getattr(doc, "document_id", None),
                        provider="openai",
                    )
                    return fallback_feedback
            return "NONE"

    if reviewer.provider == "http":
        feedback = _http_feedback(
            reviewer,
            text,
            references,
            user,
            focus_text=focus_text,
            change_context=change_context,
        )
        if feedback:
            return feedback
        if _is_code_review_document(doc, text):
            return "NONE"

    if reviewer.provider == "local":
        if getattr(reviewer, "endpoint", None):
            feedback = _local_feedback(
                reviewer,
                text,
                references,
                user,
                focus_text=focus_text,
                change_context=change_context,
            )
            if feedback:
                return feedback
        feedback = _local_reference_feedback(
            reviewer,
            text,
            references,
            user,
            focus_text=focus_text,
            change_context=change_context,
        )
        if feedback:
            return feedback

    return _fallback_feedback(text, references, focus_text=focus_text, change_context=change_context)
