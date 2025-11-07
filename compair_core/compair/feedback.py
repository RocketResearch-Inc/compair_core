from __future__ import annotations

import os
from typing import Any, Iterable, List

import requests

from .logger import log_event
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


class Reviewer:
    """Edition-aware wrapper that selects a feedback provider based on configuration."""

    def __init__(self) -> None:
        self.edition = os.getenv("COMPAIR_EDITION", "core").lower()
        self.provider = os.getenv("COMPAIR_GENERATION_PROVIDER", "local").lower()
        self.length_map = {
            "Brief": "1–2 short sentences",
            "Detailed": "a couple short paragraphs",
            "Verbose": "as thorough as reasonably possible without repeating information",
        }

        self._cloud_impl = None
        self._openai_client = None
        self.openai_model = os.getenv("COMPAIR_OPENAI_MODEL", "gpt-5-nano")
        self.openai_reasoning_effort = os.getenv("COMPAIR_OPENAI_REASONING_EFFORT", "minimal")
        self.uses_reasoning_model = _is_reasoning_model_name(self.openai_model)
        self.custom_endpoint = os.getenv("COMPAIR_GENERATION_ENDPOINT")

        if self.edition == "cloud" and CloudReviewer is not None:
            self._cloud_impl = CloudReviewer()
            self.provider = "cloud"
        else:
            if self.provider == "openai":
                api_key = os.getenv("COMPAIR_OPENAI_API_KEY")
                if api_key and openai is not None:
                    # Support both legacy (ChatCompletion) and new SDKs
                    if hasattr(openai, "api_key"):
                        openai.api_key = api_key  # type: ignore[assignment]
                    if hasattr(openai, "OpenAI"):
                        try:  # pragma: no cover - optional runtime dependency
                            self._openai_client = openai.OpenAI(api_key=api_key)  # type: ignore[attr-defined]
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


def _reference_snippets(references: Iterable[Any], limit: int = 3) -> List[str]:
    snippets: List[str] = []
    for ref in references:
        snippet = getattr(ref, "content", "") or ""
        snippet = snippet.replace("\n", " ").strip()
        if snippet:
            snippets.append(snippet[:200])
        if len(snippets) == limit:
            break
    return snippets


def _fallback_feedback(text: str, references: list[Any]) -> str:
    snippets = _reference_snippets(references)
    if not snippets:
        return "NONE"
    joined = "; ".join(snippets)
    return f"Consider aligning with these reference passages: {joined}"



def _local_reference_feedback(
    reviewer: Reviewer,
    references: list[Any],
    user: User,
) -> str | None:
    if not references:
        return None
    summaries: list[str] = []
    for ref in references[:3]:
        doc = getattr(ref, "document", None)
        title = getattr(doc, "title", None) or "a related document"
        snippet = getattr(ref, "content", "") or getattr(ref, "text", "")
        snippet = snippet.replace("\n", " ").strip()
        if not snippet:
            continue
        summaries.append(f'"{title}" — {snippet[:200]}')
    if not summaries:
        return None
    instruction = reviewer.length_map.get(user.preferred_feedback_length, "1–2 short sentences")
    if len(summaries) == 1:
        body = summaries[0]
    else:
        body = "; ".join(summaries)
    return f"[local-feedback] {instruction}: Consider the guidance from {body}"


def _openai_feedback(
    reviewer: Reviewer,
    doc: Document,
    text: str,
    references: list[Any],
    user: User,
) -> str | None:
    if openai is None:
        return None
    instruction = reviewer.length_map.get(user.preferred_feedback_length, "1–2 short sentences")
    ref_text = "\n\n".join(_reference_snippets(references, limit=3))
    system_prompt = """# Identity
You are a collaborative team member on Compair, a platform designed to help teammates uncover connections, share insights, and accelerate collective learning by comparing user documents with relevant references.

# Purpose
Your goal is to quickly surface **meaningful** connections or useful contrasts between a user’s main document and shared references—especially details that could help the document author or other team members work more effectively together.

# Instructions

- **Connect the Dots:** Highlight unique insights, similarities, differences, or answers between the main document and its references. Prioritize information that is truly meaningful or helpful to the author or team.
- **Qualified Sharing:** Only point out connections that matter—avoid commenting on trivial or already-obvious overlapping details. If nothing significant stands out, respond with: **NONE**.
- **Relay Messages:** If user documents or notes are being used to communicate with teammates, relay any important updates or questions to help foster further discussion or action.
- **Be Conversational:** Respond in a friendly, direct tone—never formal or repetitive.
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
            api_key = os.getenv("COMPAIR_OPENAI_API_KEY") or None
            try:  # pragma: no cover - optional dependency differences
                client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
            except TypeError:
                client = openai.OpenAI()
            reviewer._openai_client = client

        content: str | None = None
        uses_reasoning = reviewer.uses_reasoning_model
        if client is not None and hasattr(client, "responses"):
            request_kwargs: dict[str, Any] = {
                "model": reviewer.openai_model,
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
            response = client.responses.create(
                **request_kwargs,
            )
            content = _extract_response_text(response, reasoning_mode=uses_reasoning)
        elif client is not None and hasattr(client, "chat") and hasattr(client.chat, "completions"):
            response = client.chat.completions.create(
                model=reviewer.openai_model,
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
                model=reviewer.openai_model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.3,
                max_tokens=256,
            )
            content = chat_response["choices"][0]["message"]["content"].strip()  # type: ignore[index, assignment]
        if content:
            return content.strip()
    except Exception as exc:  # pragma: no cover - network/API failure
        log_event("openai_feedback_failed", error=str(exc))
    return None


def _local_feedback(
    reviewer: Reviewer,
    text: str,
    references: list[Any],
    user: User,
) -> str | None:
    payload = {
        "document": text,
        "references": [getattr(ref, "content", "") for ref in references],
        "length_instruction": reviewer.length_map.get(
            user.preferred_feedback_length,
            "1–2 short sentences",
        ),
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

    return None


def _http_feedback(
    reviewer: Reviewer,
    text: str,
    references: list[Any],
    user: User,
) -> str | None:
    if not reviewer.custom_endpoint:
        return None
    payload = {
        "document": text,
        "references": [getattr(ref, "content", "") for ref in references],
        "length_instruction": reviewer.length_map.get(
            user.preferred_feedback_length,
            "1–2 short sentences",
        ),
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
    return None


def get_feedback(
    reviewer: Reviewer,
    doc: Document,
    text: str,
    references: list[Any],
    user: User,
) -> str:
    if reviewer.is_cloud and cloud_get_feedback is not None:
        return cloud_get_feedback(reviewer._cloud_impl, doc, text, references, user)  # type: ignore[arg-type]

    if reviewer.provider == "openai":
        feedback = _openai_feedback(reviewer, doc, text, references, user)
        if feedback:
            return feedback

    if reviewer.provider == "http":
        feedback = _http_feedback(reviewer, text, references, user)
        if feedback:
            return feedback

    if reviewer.provider == "local":
        feedback = _local_reference_feedback(reviewer, references, user)
        if feedback:
            return feedback
        if getattr(reviewer, "endpoint", None):
            feedback = _local_feedback(reviewer, text, references, user)
            if feedback:
                return feedback

    return _fallback_feedback(text, references)
