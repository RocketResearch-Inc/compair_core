"""
Notification scoring for Core.

Uses OpenAI when configured and falls back to a deterministic heuristic so
local Core parity does not depend on a hosted-only model path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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
    build_repair_prompt,
    conservative_default,
    parse_llm_assessment,
    validate_and_normalize,
)

try:
    from ..logger import log_event
except Exception:  # pragma: no cover - isolated test loaders may not build the full package tree
    def log_event(message: str, **fields: Any) -> None:
        return None

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage_int(usage: Any, key: str) -> Optional[int]:
    value = getattr(usage, key, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(key)
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

RUBRIC_SYSTEM_INSTRUCTIONS = """You are a notification scoring engine for a collaborative document platform.

Evaluate the candidate using the rubric fields below.

Return JSON only. Do not return LOW/MEDIUM/HIGH labels directly.

Rubric fields:
- same_surface_area: yes | no | unclear
  Does the target and peer discuss the same endpoint, field, capability, feature flag, workflow step, or product-surface claim?
- direct_contradiction: yes | no | unclear
  Do the target and peer explicitly disagree?
- docs_vs_impl_drift: yes | no | unclear
  Does one side document or claim behavior that the other side's implementation/config/workflow contradicts?
- user_or_runtime_impact: yes | no | unclear
  Could the mismatch mislead users, break integrations, or change observable behavior?
- policy_or_release_risk: yes | no | unclear
  Does the mismatch affect policy, compliance, security, packaging, or release gating?
- duplication_or_overlap: yes | no | unclear
  Do the target and peer substantially overlap without a clear contradiction?
- alignment_or_confirmation: yes | no | unclear
  Does the peer reinforce or confirm the target rather than conflict with it?
- novel_for_user: yes | no | unclear
  Is this likely new/non-redundant information for the user right now?

Requirements:
- Use only the provided input.
- evidence_target must be copied verbatim from the target excerpt when possible.
- evidence_peer must be copied verbatim from a peer excerpt when possible.
- If evidence is weak, use "unclear" rather than forcing "yes".
- rationale should be 1-3 concise bullets grounded in the text.
"""

NOTIFICATION_RUBRIC_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "same_surface_area",
        "direct_contradiction",
        "docs_vs_impl_drift",
        "user_or_runtime_impact",
        "policy_or_release_risk",
        "duplication_or_overlap",
        "alignment_or_confirmation",
        "novel_for_user",
        "rationale",
        "evidence_target",
        "evidence_peer",
    ],
    "properties": {
        "same_surface_area": {"type": "string", "enum": ["yes", "no", "unclear"]},
        "direct_contradiction": {"type": "string", "enum": ["yes", "no", "unclear"]},
        "docs_vs_impl_drift": {"type": "string", "enum": ["yes", "no", "unclear"]},
        "user_or_runtime_impact": {"type": "string", "enum": ["yes", "no", "unclear"]},
        "policy_or_release_risk": {"type": "string", "enum": ["yes", "no", "unclear"]},
        "duplication_or_overlap": {"type": "string", "enum": ["yes", "no", "unclear"]},
        "alignment_or_confirmation": {"type": "string", "enum": ["yes", "no", "unclear"]},
        "novel_for_user": {"type": "string", "enum": ["yes", "no", "unclear"]},
        "rationale": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 6,
        },
        "evidence_target": {"type": "string"},
        "evidence_peer": {"type": "string"},
    },
}

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
_WEAK_HEURISTIC_PREFIXES = (
    "possible cross-repo drift detected",
    "the changed content says ",
    "the changed text says ",
    "this may rename ",
)
_NOTIFICATION_EVIDENCE_CHARS = 600


def _getenv(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        if not name:
            continue
        value = os.getenv(name)
        if value:
            return value
    return default


def _openai_base_url() -> Optional[str]:
    return _getenv("COMPAIR_OPENAI_BASE_URL", "OPENAI_BASE_URL")


def _openai_sdk_max_retries() -> int:
    raw = _getenv("COMPAIR_OPENAI_SDK_MAX_RETRIES", "OPENAI_SDK_MAX_RETRIES")
    try:
        return max(0, int(raw)) if raw is not None else 0
    except ValueError:
        return 0


def _is_transport_error_text(value: str | None) -> bool:
    text = str(value or "").lower()
    if not text:
        return False
    markers = (
        "apitimeouterror",
        "timeout",
        "timed out",
        "apiconnectionerror",
        "connection error",
        "connectionerror",
        "rate limit",
        "ratelimit",
        "server error",
        "internalservererror",
        "bad gateway",
        "service unavailable",
    )
    return any(marker in text for marker in markers)


def _has_transport_error(errors: List[str] | None) -> bool:
    return any(_is_transport_error_text(error) for error in (errors or []))


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


def build_rubric_user_prompt(payload: Dict[str, Any]) -> str:
    compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        "Evaluate this notification candidate using the rubric schema.\n"
        "Return JSON only with the rubric fields requested by the schema.\n"
        "Use target.chunk_excerpt and candidates[*].peer_excerpt when they are present; otherwise quote directly from the supplied chunk text.\n"
        "INPUT_JSON:\n"
        f"{compact}"
    )


def _normalize_rubric_flag(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    mapping = {
        "yes": "yes",
        "true": "yes",
        "1": "yes",
        "y": "yes",
        "no": "no",
        "false": "no",
        "0": "no",
        "n": "no",
        "unclear": "unclear",
        "unknown": "unclear",
        "unsure": "unclear",
        "maybe": "unclear",
    }
    return mapping.get(normalized, "unclear")


def _env_int(*names: str) -> Optional[int]:
    for name in names:
        raw = os.getenv(name)
        if not raw:
            continue
        try:
            return int(raw.strip())
        except ValueError:
            continue
    return None


def _env_float(*names: str) -> Optional[float]:
    for name in names:
        raw = os.getenv(name)
        if not raw:
            continue
        try:
            return float(raw.strip())
        except ValueError:
            continue
    return None


def _rubric_assessment(obj: Dict[str, Any]) -> tuple[Optional[ParsedLLMNotificationAssessment], List[str]]:
    def flag(name: str) -> str:
        return _normalize_rubric_flag(obj.get(name))

    same_surface = flag("same_surface_area") == "yes"
    contradiction = flag("direct_contradiction") == "yes"
    docs_vs_impl = flag("docs_vs_impl_drift") == "yes"
    impact = flag("user_or_runtime_impact") == "yes"
    policy = flag("policy_or_release_risk") == "yes"
    overlap = flag("duplication_or_overlap") == "yes"
    alignment = flag("alignment_or_confirmation") == "yes"
    novel_flag = flag("novel_for_user")

    rationale = obj.get("rationale")
    rationale_items = [str(item).strip() for item in rationale or [] if str(item).strip()] if isinstance(rationale, list) else []
    if not rationale_items:
        if contradiction or docs_vs_impl:
            rationale_items = ["Target and peer disagree on the same product surface."]
        elif overlap:
            rationale_items = ["Target and peer overlap substantially without a direct contradiction."]
        elif alignment:
            rationale_items = ["Peer evidence reinforces the target rather than contradicting it."]
        else:
            rationale_items = ["Structured rubric response provided limited supporting detail."]

    evidence_target = str(obj.get("evidence_target") or "").strip()
    evidence_peer = str(obj.get("evidence_peer") or "").strip()
    grounded_target = bool(evidence_target)
    grounded_peer = bool(evidence_peer)
    grounded_both = grounded_target and grounded_peer

    if contradiction or docs_vs_impl:
        intent = "potential_conflict"
    elif overlap and not alignment:
        intent = "hidden_overlap"
    elif alignment and not contradiction and not docs_vs_impl:
        intent = "quiet_validation"
    else:
        intent = "relevant_update"

    relevance = "LOW"
    if policy or impact or contradiction:
        relevance = "HIGH"
    elif same_surface or docs_vs_impl or overlap or alignment:
        relevance = "MEDIUM"

    novelty = {"yes": "HIGH", "no": "LOW"}.get(novel_flag, "MEDIUM")
    weak_cross_repo_relation = not any([same_surface, contradiction, docs_vs_impl, impact, policy, overlap, alignment])
    if weak_cross_repo_relation:
        # If the rubric itself cannot establish a comparable cross-repo relationship,
        # keep the candidate conservative enough for routing to drop it.
        novelty = "LOW"

    severity = "LOW"
    if intent == "potential_conflict":
        if policy:
            severity = "HIGH"
        elif contradiction and (impact or docs_vs_impl):
            severity = "HIGH" if grounded_both or same_surface else "MEDIUM"
        elif impact or docs_vs_impl or (same_surface and contradiction):
            severity = "MEDIUM"
    elif intent == "relevant_update":
        if policy or impact:
            severity = "MEDIUM"

    certainty = "LOW"
    if grounded_both and (same_surface or contradiction or docs_vs_impl or overlap or alignment):
        certainty = "HIGH"
    elif grounded_both or (same_surface and (grounded_target or grounded_peer)):
        certainty = "MEDIUM"

    delivery = "digest"
    if intent == "potential_conflict" and severity == "HIGH" and certainty in {"HIGH", "MEDIUM"}:
        delivery = "push"
    elif intent == "relevant_update" and severity == "HIGH" and certainty == "HIGH":
        delivery = "push"

    parsed, errors = validate_and_normalize(
        {
            "intent": intent,
            "relevance": relevance,
            "novelty": novelty,
            "severity": severity,
            "certainty": certainty,
            "delivery": delivery,
            "rationale": rationale_items,
            "evidence_target": evidence_target,
            "evidence_peer": evidence_peer,
        }
    )
    return parsed, errors


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


def _is_weak_heuristic_summary(summary: str) -> bool:
    lowered = (summary or "").strip().lower()
    if not lowered:
        return False
    if lowered.startswith("possible route/path drift:"):
        return False
    if lowered.startswith("possible value drift:"):
        return False
    if lowered.startswith("possible docs-vs-implementation drift"):
        return False
    if lowered.startswith("possible cross-repo drift:"):
        return False
    return any(lowered.startswith(prefix) for prefix in _WEAK_HEURISTIC_PREFIXES)


def _heuristic_assessment(payload: Dict[str, Any]) -> ParsedLLMNotificationAssessment:
    target = payload.get("target") or {}
    peer = _first_peer(payload)
    summary = _feedback_summary(payload)
    target_excerpt = (target.get("chunk_excerpt") or target.get("chunk_text") or "").strip()
    peer_excerpt = (peer.get("peer_excerpt") or peer.get("peer_chunk_text") or "").strip()

    if _is_weak_heuristic_summary(summary):
        return ParsedLLMNotificationAssessment(
            intent="quiet_validation",
            relevance="LOW",
            novelty="LOW",
            severity="LOW",
            certainty="LOW",
            delivery="digest",
            rationale=["Heuristic local summary was too weak or generic to surface as a notification."],
            evidence_target=target_excerpt[:_NOTIFICATION_EVIDENCE_CHARS] if target_excerpt else "",
            evidence_peer=peer_excerpt[:_NOTIFICATION_EVIDENCE_CHARS] if peer_excerpt else "",
            parse_mode="heuristic",
            raw_extracted=None,
            errors=[],
        )

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

    preferred_target = target_excerpt[:_NOTIFICATION_EVIDENCE_CHARS] if target_excerpt else ""
    preferred_peer = peer_excerpt[:_NOTIFICATION_EVIDENCE_CHARS] if peer_excerpt else ""

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
        model_env = (
            _getenv("COMPAIR_OPENAI_NOTIF_MODEL", "OPENAI_NOTIF_MODEL")
            or _getenv("COMPAIR_OPENAI_MODEL", "OPENAI_MODEL")
        )
        provider_env = _getenv("COMPAIR_NOTIFICATION_SCORING_PROVIDER", "NOTIFICATION_SCORING_PROVIDER")
        temp_env = _getenv("COMPAIR_OPENAI_NOTIF_TEMPERATURE", "OPENAI_NOTIF_TEMPERATURE")
        max_retries_env = _env_int(
            "COMPAIR_NOTIFICATION_SCORING_MAX_RETRIES",
            "NOTIFICATION_SCORING_MAX_RETRIES",
        )
        timeout_env = _env_float(
            "COMPAIR_NOTIFICATION_SCORING_TIMEOUT_S",
            "NOTIFICATION_SCORING_TIMEOUT_S",
        )
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
        if max_retries_env is not None and max_retries_env >= 0:
            cfg = NotificationScorerConfig(
                model=cfg.model,
                provider=cfg.provider,
                temperature=cfg.temperature,
                max_retries=max_retries_env,
                timeout_s=cfg.timeout_s,
            )
        if timeout_env is not None and timeout_env > 0:
            cfg = NotificationScorerConfig(
                model=cfg.model,
                provider=cfg.provider,
                temperature=cfg.temperature,
                max_retries=cfg.max_retries,
                timeout_s=timeout_env,
            )
        object.__setattr__(self, "config", cfg)

        api_key = _getenv("COMPAIR_OPENAI_API_KEY", "OPENAI_API_KEY")
        self.client = client
        if self.client is None and OpenAI is not None and api_key and cfg.provider.lower() != "heuristic":
            base_url = _openai_base_url()
            try:
                kwargs: Dict[str, Any] = {"api_key": api_key, "max_retries": _openai_sdk_max_retries()}
                if base_url:
                    kwargs["base_url"] = base_url
                self.client = OpenAI(**kwargs)
            except TypeError:
                try:
                    kwargs = {"api_key": api_key}
                    if base_url:
                        kwargs["base_url"] = base_url
                    self.client = OpenAI(**kwargs)
                except Exception:
                    self.client = None
            except Exception:
                self.client = None

    def score(self, payload: Dict[str, Any]) -> ParsedLLMNotificationAssessment:
        return self._score_auto(payload)

    def score_rubric(self, payload: Dict[str, Any]) -> ParsedLLMNotificationAssessment:
        provider = (self.config.provider or "auto").lower()
        if provider == "heuristic" or self.client is None:
            return conservative_default(
                "failed_default",
                errors=["OpenAI scorer unavailable for rubric replay mode."],
            )
        return self._score_rubric(payload)

    def score_direct(self, payload: Dict[str, Any]) -> ParsedLLMNotificationAssessment:
        provider = (self.config.provider or "auto").lower()
        if provider == "heuristic" or self.client is None:
            return conservative_default(
                "failed_default",
                errors=["OpenAI scorer unavailable for direct replay mode."],
            )
        return self._score_direct(payload)

    def _score_auto(self, payload: Dict[str, Any]) -> ParsedLLMNotificationAssessment:
        provider = (self.config.provider or "auto").lower()
        if provider == "heuristic":
            return _heuristic_assessment(payload)

        if self.client is None:
            return _heuristic_assessment(payload)

        structured = self._score_rubric(payload)
        if structured.parse_mode != "failed_default":
            return structured
        if _has_transport_error(structured.errors):
            heuristic = _heuristic_assessment(payload)
            return ParsedLLMNotificationAssessment(
                **{
                    **heuristic.__dict__,
                    "errors": ["OpenAI scorer transport failure; heuristic fallback used.", *(structured.errors or [])],
                }
            )

        direct = self._score_direct(payload)
        if direct.parse_mode != "failed_default":
            return direct
        if _has_transport_error(direct.errors):
            heuristic = _heuristic_assessment(payload)
            return ParsedLLMNotificationAssessment(
                **{
                    **heuristic.__dict__,
                    "errors": ["OpenAI scorer transport failure; heuristic fallback used.", *(direct.errors or [])],
                }
            )

        heuristic = _heuristic_assessment(payload)
        return ParsedLLMNotificationAssessment(
            **{**heuristic.__dict__, "errors": ["OpenAI scorer failed; heuristic fallback used."]}
        )

    def _score_rubric(self, payload: Dict[str, Any]) -> ParsedLLMNotificationAssessment:
        rubric_system = RUBRIC_SYSTEM_INSTRUCTIONS
        rubric_user = build_rubric_user_prompt(payload)
        return self._score_once_structured(rubric_system, rubric_user)

    def _score_direct(self, payload: Dict[str, Any]) -> ParsedLLMNotificationAssessment:
        system = SYSTEM_INSTRUCTIONS
        user = build_user_prompt(payload)
        first = self._score_once(system, user)
        if first.parse_mode != "failed_default":
            return first
        if _has_transport_error(first.errors):
            return conservative_default("failed_default", errors=list(first.errors or []))

        repaired_prompt = build_repair_prompt(first.raw_extracted or user)
        repaired = self._score_once(
            "You convert model outputs into strictly valid JSON. Output JSON only.",
            repaired_prompt,
        )
        if repaired.parse_mode != "failed_default":
            return repaired
        return conservative_default("failed_default", errors=list(repaired.errors or ["All direct attempts failed"]))

    def _score_once_structured(self, system_prompt: str, user_prompt: str) -> ParsedLLMNotificationAssessment:
        raw, last_err = self._responses_create_text(
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
            json_schema=NOTIFICATION_RUBRIC_JSON_SCHEMA,
            schema_name="notification_score",
        )
        if not raw:
            errors = [f"structured_transport_error: {last_err}"] if _is_transport_error_text(last_err) else ["Empty structured response"]
            return conservative_default("failed_default", errors=errors)
        try:
            obj = json.loads(raw)
        except Exception as exc:
            return conservative_default("failed_default", errors=[f"json_schema_load_error: {exc}"])
        parsed, errors = _rubric_assessment(obj)
        if parsed:
            return ParsedLLMNotificationAssessment(
                **{**parsed.__dict__, "parse_mode": "json_schema_rubric", "raw_extracted": raw}
            )
        return conservative_default("failed_default", errors=errors or ["Invalid structured scorer output"])

    def _score_once(self, system_prompt: str, user_prompt: str) -> ParsedLLMNotificationAssessment:
        raw, last_err = self._responses_create_text(
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
        errors = [f"transport_error: {last_err}"] if _is_transport_error_text(last_err) else ["Empty response"]
        return conservative_default("failed_default", errors=errors)

    def _responses_create_text(
        self,
        model: str,
        input_items: List[Dict[str, Any]],
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        schema_name: str = "structured_output",
    ) -> tuple[Optional[str], Optional[str]]:
        if self.client is None:
            return None, None
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
                if json_schema is not None:
                    request["text"] = {
                        "format": {
                            "type": "json_schema",
                            "name": schema_name,
                            "schema": json_schema,
                            "strict": True,
                        }
                    }
                started_at = time.time()
                resp = self.client.responses.create(**request)
                usage = getattr(resp, "usage", None)
                log_event(
                    "openai_notification_scoring_created",
                    model=model,
                    input_tokens=_usage_int(usage, "input_tokens"),
                    output_tokens=_usage_int(usage, "output_tokens"),
                    duration_sec=round(time.time() - started_at, 3),
                    structured=bool(json_schema is not None),
                    schema_name=schema_name if json_schema is not None else None,
                    created_at=_utc_now(),
                )
                text = getattr(resp, "output_text", None)
                if isinstance(text, str) and text.strip():
                    return text, None
                output = getattr(resp, "output", None)
                if output and isinstance(output, list):
                    for item in output:
                        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
                        if not content:
                            continue
                        for part in content:
                            maybe_text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                            if isinstance(maybe_text, str) and maybe_text.strip():
                                return maybe_text, None
                raw = str(resp)
                if raw.strip():
                    return raw, None
            except Exception as exc:
                last_err = repr(exc)
                logger.warning("Notification scoring request failed: %s", last_err)
                time.sleep(min(0.4 * (attempt + 1), 1.5))
        return None, last_err
