from __future__ import annotations

import os
import requests
from typing import Any

from .logger import log_event
from .models import Document, User

try:
    from compair_cloud.feedback import Reviewer as CloudReviewer  # type: ignore
    from compair_cloud.feedback import get_feedback as cloud_get_feedback  # type: ignore
except (ImportError, ModuleNotFoundError):
    CloudReviewer = None  # type: ignore
    cloud_get_feedback = None  # type: ignore


class Reviewer:
    """Edition-aware wrapper that falls back to the local feedback endpoint."""

    def __init__(self) -> None:
        self.edition = os.getenv("COMPAIR_EDITION", "core").lower()
        self._cloud_impl = None
        if self.edition == "cloud" and CloudReviewer is not None:
            self._cloud_impl = CloudReviewer()
        else:
            self.client = None
            self.model = os.getenv("COMPAIR_LOCAL_GENERATION_MODEL", "local-feedback")
            base_url = os.getenv("COMPAIR_LOCAL_MODEL_URL", "http://local-model:9000")
            route = os.getenv("COMPAIR_LOCAL_GENERATION_ROUTE", "/generate")
            self.endpoint = f"{base_url.rstrip('/')}{route}"

    @property
    def is_cloud(self) -> bool:
        return self._cloud_impl is not None


def _fallback_feedback(text: str, references: list[Any]) -> str:
    if not references:
        return "NONE"
    top_ref = references[0]
    snippet = getattr(top_ref, "content", "") or ""
    snippet = snippet.replace("\n", " ").strip()[:200]
    if not snippet:
        return "NONE"
    return f"Check alignment with this reference: {snippet}"


def get_feedback(
    reviewer: Reviewer,
    doc: Document,
    text: str,
    references: list[Any],
    user: User,
) -> str:
    if reviewer.is_cloud and cloud_get_feedback is not None:
        return cloud_get_feedback(reviewer._cloud_impl, doc, text, references, user)  # type: ignore[arg-type]

    payload = {
        "document": text,
        "references": [getattr(ref, "content", "") for ref in references],
        "length_instruction": {
            "Brief": "1–2 short sentences",
            "Detailed": "A couple short paragraphs",
            "Verbose": "As thorough as reasonably possible without repeating information",
        }.get(user.preferred_feedback_length, "1–2 short sentences"),
    }

    try:
        response = requests.post(reviewer.endpoint, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        feedback = data.get("feedback")
        if feedback:
            return feedback
    except Exception as exc:  # pragma: no cover - network failures stay graceful
        log_event("local_feedback_failed", error=str(exc))

    return _fallback_feedback(text, references)
