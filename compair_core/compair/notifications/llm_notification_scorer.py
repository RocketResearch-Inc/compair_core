"""
Notification scoring for Core.

Uses OpenAI when configured and falls back to a deterministic heuristic so
local Core parity does not depend on a hosted-only model path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import logging
import os
import re
import time

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency at runtime
    OpenAI = None  # type: ignore[assignment]

from .parse_llm_structured_output import (
    ParsedLLMNotificationAssessment,
    build_kv_fallback_prompt,
    build_repair_prompt,
    conservative_default,
    parse_llm_assessment,
)

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTIONS = """You are a notification scoring engine for a collaborative document platform.

You will be given:
- a target excerpt from the user's document,
- one or more peer excerpts that are semantically similar,
- metadata,
- and optional generated feedback text.

Return:
- intent: potential_conflict | relevant_update | hidden_overlap | quiet_validation
- relevance/novelty/severity/certainty: LOW, MEDIUM, or HIGH
- delivery: push or digest

Guidelines:
- Use only the provided input; do not speculate.
- Be conservative when unsure.
- evidence_target must be copied verbatim from the provided target text/excerpt.
- evidence_peer must be copied verbatim from one provided peer text/excerpt.
- hidden_overlap and quiet_validation should NEVER be push.
- potential_conflict should be push only if certainty is HIGH and severity is at least MEDIUM.
"""

_CONFLICT_TERMS = {
    "conflict", "drift", "mismatch", "regression", "contradict", "contradiction",
    "incompatible", "rename", "renamed", "break", "broken", "wrong", "stale",
    "mislead", "misleading", "silently", "fallback", "guard", "missing",
}
_OVERLAP_TERMS = {"overlap", "duplicate", "duplicated", "similar", "same logic", "parallel implementation"}
_VALIDATION_TERMS = {"aligns", "aligned", "reinforces", "confirms", "validated", "consistent"}
_HIGH_SEVERITY_TERMS = {"break", "broken", "silently", "mislead", "misleading", "error", "hang", "contract", "api"}
_HIGH_RELEVANCE_TERMS = {"api", "contract", "schema", "field", "endpoint", "env", "sync", "install", "publish"}
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]{3,}")


def _getenv(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        if not name:
            continue
        value = os.getenv(name)
        if value:
            return value
    return default


def build_user_prompt(payload: Dict[str, Any]) -> str:
    compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        "Score this notification candidate using the input JSON below.\n"
        "Return intent, LOW/MEDIUM/HIGH assessments, delivery, rationale bullets, and 1 evidence excerpt from target and peer.\n"
        "Use target.chunk_excerpt and candidates[*].peer_excerpt when they are present; otherwise quote directly from the supplied chunk text.\n"
        "OUTPUT: JSON only with keys: intent,relevance,novelty,severity,certainty,delivery,rationale,evidence_target,evidence_peer\n"
        "INPUT_JSON:\n"
        f"{compact}"
    )


def _tokenize(*values: Optional[str]) -> set[str]:
    out: set[str] = set()
    for value in values:
        if not value:
            continue
        for match in _TOKEN_RE.findall(value.lower()):
            out.add(match)
    return out


def _first_peer(payload: Dict[str, Any]) -> Dict[str, Any]:
    candidates = payload.get("candidates") or []
    if candidates and isinstance(candidates[0], dict):
        return candidates[0]
    return {}


def _feedback_summary(payload: Dict[str, Any]) -> str:
    feedback = payload.get("generated_feedback")
    if isinstance(feedback, dict):
        summary = feedback.get("summary")
        if isinstance(summary, str):
            return summary.strip()
    return ""


def _heuristic_assessment(payload: Dict[str, Any]) -> ParsedLLMNotificationAssessment:
    target = payload.get("target") or {}
    peer = _first_peer(payload)
    summary = _feedback_summary(payload)
    target_excerpt = (target.get("chunk_excerpt") or target.get("chunk_text") or "").strip()
    peer_excerpt = (peer.get("peer_excerpt") or peer.get("peer_chunk_text") or "").strip()

    text = " ".join([summary, target_excerpt, peer_excerpt]).lower()
    intent = "relevant_update"
    if any(term in text for term in _CONFLICT_TERMS):
        intent = "potential_conflict"
    elif any(term in text for term in _OVERLAP_TERMS):
        intent = "hidden_overlap"
    elif any(term in text for term in _VALIDATION_TERMS):
        intent = "quiet_validation"

    relevance = "MEDIUM"
    novelty = "MEDIUM"
    severity = "LOW"
    certainty = "LOW"
    delivery = "digest"

    signal_tokens = _tokenize(summary, target_excerpt, peer_excerpt)
    if target_excerpt and peer_excerpt:
        certainty = "HIGH"
    elif target_excerpt or peer_excerpt:
        certainty = "MEDIUM"

    if any(term in text for term in _HIGH_RELEVANCE_TERMS):
        relevance = "HIGH"
    elif intent in {"hidden_overlap", "quiet_validation"}:
        relevance = "LOW"

    if intent == "potential_conflict":
        severity = "MEDIUM"
        novelty = "HIGH" if any(term in text for term in {"regression", "rename", "drift", "mismatch"}) else "MEDIUM"
        if any(term in text for term in _HIGH_SEVERITY_TERMS):
            severity = "HIGH"
        if relevance == "HIGH" and severity == "HIGH" and certainty == "HIGH":
            delivery = "push"
    elif intent == "hidden_overlap":
        relevance = "MEDIUM"
        novelty = "LOW"
        severity = "LOW"
    elif intent == "quiet_validation":
        relevance = "LOW"
        novelty = "LOW"
        severity = "LOW"
    else:
        severity = "MEDIUM" if any(term in text for term in _HIGH_RELEVANCE_TERMS) else "LOW"

    rationale: List[str] = []
    if summary:
        rationale.append(summary[:220])
    if target_excerpt:
        rationale.append("Target excerpt contains the changed contract or claim under review.")
    if peer_excerpt:
        rationale.append("Peer excerpt provides directly comparable evidence from a related document.")
    if not rationale:
        rationale = ["Heuristic fallback used because structured notification scoring was unavailable."]

    preferred_target = target_excerpt[:280] if target_excerpt else ""
    preferred_peer = peer_excerpt[:280] if peer_excerpt else ""

    return ParsedLLMNotificationAssessment(
        intent=intent,
        relevance=relevance,
        novelty=novelty,
        severity=severity,
        certainty=certainty,
        delivery=delivery,
        rationale=rationale[:3],
        evidence_target=preferred_target,
        evidence_peer=preferred_peer,
        parse_mode="heuristic",
        raw_extracted=None,
        errors=[],
    )


@dataclass(frozen=True)
class NotificationScorerConfig:
    model: str = "gpt-5"
    provider: str = "auto"
    temperature: Optional[float] = None
    max_retries: int = 2
    timeout_s: float = 30.0


class NotificationScorer:
    def __init__(self, config: Optional[NotificationScorerConfig] = None, client: Optional[OpenAI] = None):
        cfg = config or NotificationScorerConfig()
        model_env = _getenv("COMPAIR_OPENAI_NOTIF_MODEL", "OPENAI_NOTIF_MODEL")
        provider_env = _getenv("COMPAIR_NOTIFICATION_SCORING_PROVIDER", "NOTIFICATION_SCORING_PROVIDER")
        temp_env = _getenv("COMPAIR_OPENAI_NOTIF_TEMPERATURE", "OPENAI_NOTIF_TEMPERATURE")
        if model_env:
            cfg = NotificationScorerConfig(
                model=model_env,
                provider=cfg.provider,
                temperature=cfg.temperature,
                max_retries=cfg.max_retries,
                timeout_s=cfg.timeout_s,
            )
        if provider_env:
            cfg = NotificationScorerConfig(
                model=cfg.model,
                provider=provider_env,
                temperature=cfg.temperature,
                max_retries=cfg.max_retries,
                timeout_s=cfg.timeout_s,
            )
        if temp_env:
            try:
                cfg = NotificationScorerConfig(
                    model=cfg.model,
                    provider=cfg.provider,
                    temperature=float(temp_env),
                    max_retries=cfg.max_retries,
                    timeout_s=cfg.timeout_s,
                )
            except ValueError:
                pass
        object.__setattr__(self, "config", cfg)

        api_key = _getenv("COMPAIR_OPENAI_API_KEY", "OPENAI_API_KEY")
        self.client = client
        if self.client is None and OpenAI is not None and api_key and cfg.provider.lower() != "heuristic":
            try:
                self.client = OpenAI(api_key=api_key)
            except Exception:
                self.client = None

    def score(self, payload: Dict[str, Any]) -> ParsedLLMNotificationAssessment:
        provider = (self.config.provider or "auto").lower()
        if provider == "heuristic":
            return _heuristic_assessment(payload)

        if self.client is None:
            return _heuristic_assessment(payload)

        system = SYSTEM_INSTRUCTIONS
        user = build_user_prompt(payload)

        first = self._score_once(system, user)
        if first.parse_mode != "failed_default":
            return first

        repaired_prompt = build_repair_prompt(first.raw_extracted or user)
        repaired = self._score_once(
            "You convert model outputs into strictly valid JSON. Output JSON only.",
            repaired_prompt,
        )
        if repaired.parse_mode != "failed_default":
            return repaired

        kv_prompt = build_kv_fallback_prompt(system + "\n\n" + user)
        kv_resp = self._score_once(
            "Return only key=value lines. No extra text.",
            kv_prompt,
        )
        if kv_resp.parse_mode != "failed_default":
            return kv_resp

        heuristic = _heuristic_assessment(payload)
        return ParsedLLMNotificationAssessment(
            **{**heuristic.__dict__, "errors": ["OpenAI scorer failed; heuristic fallback used."]}
        )

    def _score_once(self, system_prompt: str, user_prompt: str) -> ParsedLLMNotificationAssessment:
        raw = self._responses_create_text(
            model=self.config.model,
            input_items=[
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
        )
        if raw:
            return parse_llm_assessment(raw)
        return conservative_default("failed_default", errors=["Empty response"])

    def _responses_create_text(self, model: str, input_items: List[Dict[str, str]]) -> Optional[str]:
        if self.client is None:
            return None
        last_err: Optional[str] = None
        for attempt in range(self.config.max_retries):
            try:
                request: Dict[str, Any] = {
                    "model": model,
                    "input": input_items,
                    "timeout": self.config.timeout_s,
                }
                if self.config.temperature is not None:
                    request["temperature"] = self.config.temperature
                resp = self.client.responses.create(**request)
                text = getattr(resp, "output_text", None)
                if isinstance(text, str) and text.strip():
                    return text
                output = getattr(resp, "output", None)
                if output and isinstance(output, list):
                    for item in output:
                        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
                        if not content:
                            continue
                        for part in content:
                            maybe_text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                            if isinstance(maybe_text, str) and maybe_text.strip():
                                return maybe_text
                raw = str(resp)
                if raw.strip():
                    return raw
            except Exception as exc:
                last_err = repr(exc)
                logger.warning("Notification scoring request failed: %s", last_err)
                time.sleep(min(0.4 * (attempt + 1), 1.5))
        return None
