"""
High-reliability parsing + validation for notification scoring outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import json
import re

ALLOWED_INTENTS = {
    "potential_conflict",
    "relevant_update",
    "hidden_overlap",
    "quiet_validation",
}

ALLOWED_BUCKETS = {"LOW", "MEDIUM", "HIGH"}
ALLOWED_DELIVERY = {"push", "digest"}

MAX_EVIDENCE_CHARS = 600
MAX_RATIONALE_ITEMS = 6
MAX_RATIONALE_ITEM_CHARS = 240


@dataclass(frozen=True)
class ParsedLLMNotificationAssessment:
    intent: str
    relevance: str
    novelty: str
    severity: str
    certainty: str
    delivery: str
    rationale: List[str]
    evidence_target: str
    evidence_peer: str
    parse_mode: str
    raw_extracted: Optional[str] = None
    errors: Optional[List[str]] = None


def conservative_default(parse_mode: str, errors: Optional[List[str]] = None) -> ParsedLLMNotificationAssessment:
    return ParsedLLMNotificationAssessment(
        intent="relevant_update",
        relevance="MEDIUM",
        novelty="LOW",
        severity="LOW",
        certainty="LOW",
        delivery="digest",
        rationale=["Fallback used due to parsing or scoring issues."],
        evidence_target="",
        evidence_peer="",
        parse_mode=parse_mode,
        raw_extracted=None,
        errors=errors or [],
    )


def validate_and_normalize(obj: Dict[str, Any]) -> Tuple[Optional[ParsedLLMNotificationAssessment], List[str]]:
    errors: List[str] = []

    def get_str(d: Dict[str, Any], key: str) -> Optional[str]:
        v = d.get(key)
        return v if isinstance(v, str) and v.strip() else None

    intent = get_str(obj, "intent")
    delivery = get_str(obj, "delivery")
    if not delivery and isinstance(obj.get("delivery"), dict):
        delivery = get_str(obj["delivery"], "recommended_channel")

    assessment = obj.get("assessment") if isinstance(obj.get("assessment"), dict) else obj

    relevance = get_str(assessment, "relevance")
    novelty = get_str(assessment, "novelty")
    severity = get_str(assessment, "severity")
    certainty = get_str(assessment, "certainty")

    def norm_bucket(x: Optional[str]) -> Optional[str]:
        if not x:
            return None
        x = x.strip().upper()
        mapping = {
            "LOW": "LOW",
            "L": "LOW",
            "MED": "MEDIUM",
            "MEDIUM": "MEDIUM",
            "M": "MEDIUM",
            "HIGH": "HIGH",
            "H": "HIGH",
        }
        return mapping.get(x)

    def norm_intent(x: Optional[str]) -> Optional[str]:
        if not x:
            return None
        x = x.strip().lower()
        mapping = {
            "potential_conflict": "potential_conflict",
            "conflict": "potential_conflict",
            "possible_conflict": "potential_conflict",
            "relevant_update": "relevant_update",
            "update": "relevant_update",
            "hidden_overlap": "hidden_overlap",
            "overlap": "hidden_overlap",
            "quiet_validation": "quiet_validation",
            "validation": "quiet_validation",
            "reinforcement": "quiet_validation",
        }
        return mapping.get(x)

    def norm_delivery(x: Optional[str]) -> Optional[str]:
        if not x:
            return None
        x = x.strip().lower()
        mapping = {
            "push": "push",
            "toast": "push",
            "notify": "push",
            "digest": "digest",
            "email_digest": "digest",
            "inbox": "digest",
        }
        return mapping.get(x)

    intent_n = norm_intent(intent)
    if intent_n is None:
        errors.append(f"Invalid or missing intent: {intent!r}")

    delivery_n = norm_delivery(delivery)
    if delivery_n is None:
        errors.append(f"Invalid or missing delivery: {delivery!r}")

    rel_n = norm_bucket(relevance)
    nov_n = norm_bucket(novelty)
    sev_n = norm_bucket(severity)
    cer_n = norm_bucket(certainty)
    for name, val in [("relevance", rel_n), ("novelty", nov_n), ("severity", sev_n), ("certainty", cer_n)]:
        if val is None:
            errors.append(f"Invalid or missing {name}: {assessment.get(name)!r}")

    rationale: List[str] = []
    if isinstance(obj.get("rationale"), list):
        rationale = [str(x).strip() for x in obj["rationale"] if str(x).strip()]
    elif isinstance(obj.get("rationales"), list):
        rationale = [str(x).strip() for x in obj["rationales"] if str(x).strip()]
    elif isinstance(obj.get("reason"), str):
        rationale = [obj["reason"].strip()]
    elif isinstance(obj.get("delivery"), dict) and isinstance(obj["delivery"].get("reason"), str):
        rationale = [obj["delivery"]["reason"].strip()]
    rationale = [item[:MAX_RATIONALE_ITEM_CHARS] for item in rationale[:MAX_RATIONALE_ITEMS]]

    evidence_target = ""
    evidence_peer = ""
    et = obj.get("evidence_target")
    ep = obj.get("evidence_peer")
    if isinstance(et, str):
        evidence_target = et.strip()
    if isinstance(ep, str):
        evidence_peer = ep.strip()

    if (not evidence_target and not evidence_peer) and isinstance(obj.get("evidence"), list):
        excerpts: List[str] = []
        for item in obj["evidence"]:
            if not isinstance(item, dict):
                continue
            excerpt = str(item.get("excerpt", "")).strip()
            if excerpt:
                excerpts.append(excerpt)
            source = str(item.get("source", "")).lower()
            if "target" in source and not evidence_target:
                evidence_target = excerpt
            if "peer" in source and not evidence_peer:
                evidence_peer = excerpt
        if not evidence_target and excerpts:
            evidence_target = excerpts[0]
        if not evidence_peer and len(excerpts) > 1:
            evidence_peer = excerpts[1]

    evidence_target = evidence_target[:MAX_EVIDENCE_CHARS]
    evidence_peer = evidence_peer[:MAX_EVIDENCE_CHARS]

    if intent_n and intent_n not in ALLOWED_INTENTS:
        errors.append(f"Intent not allowed after normalization: {intent_n}")
    for name, val in [("relevance", rel_n), ("novelty", nov_n), ("severity", sev_n), ("certainty", cer_n)]:
        if val and val not in ALLOWED_BUCKETS:
            errors.append(f"{name} not allowed after normalization: {val}")
    if delivery_n and delivery_n not in ALLOWED_DELIVERY:
        errors.append(f"Delivery not allowed after normalization: {delivery_n}")

    if errors:
        return None, errors

    return ParsedLLMNotificationAssessment(
        intent=intent_n,  # type: ignore[arg-type]
        relevance=rel_n,  # type: ignore[arg-type]
        novelty=nov_n,  # type: ignore[arg-type]
        severity=sev_n,  # type: ignore[arg-type]
        certainty=cer_n,  # type: ignore[arg-type]
        delivery=delivery_n,  # type: ignore[arg-type]
        rationale=rationale or ["No rationale provided."],
        evidence_target=evidence_target,
        evidence_peer=evidence_peer,
        parse_mode="json",
        raw_extracted=None,
        errors=[],
    ), []


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_probable_json(text: str) -> Optional[str]:
    if not text:
        return None
    fence = _CODE_FENCE_RE.search(text)
    if fence:
        return fence.group(1).strip()
    block = _JSON_BLOCK_RE.search(text)
    if block:
        return block.group(0).strip()
    return None


def repair_common_json_issues(s: str) -> str:
    repaired = s.strip()
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    repaired = re.sub(r"([\{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', repaired)
    return repaired


def parse_kv_fallback(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    rationale: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=" in line:
            key, value = line.split("=", 1)
        elif ":" in line:
            key, value = line.split(":", 1)
        else:
            rationale.append(line)
            continue
        key = key.strip().lower()
        value = value.strip()
        if key in {"rationale", "reason"}:
            rationale.append(value)
        else:
            out[key] = value
    if rationale:
        out["rationale"] = rationale
    return out


def parse_llm_assessment(raw_text: str) -> ParsedLLMNotificationAssessment:
    extracted = extract_probable_json(raw_text)
    errors: List[str] = []

    if extracted:
        try:
            obj = json.loads(extracted)
            parsed, parse_errors = validate_and_normalize(obj)
            if parsed:
                return ParsedLLMNotificationAssessment(
                    **{**parsed.__dict__, "parse_mode": "json", "raw_extracted": extracted}
                )
            errors.extend(parse_errors)
        except Exception as exc:
            errors.append(f"json_load_error: {exc}")
            try:
                repaired = repair_common_json_issues(extracted)
                obj = json.loads(repaired)
                parsed, parse_errors = validate_and_normalize(obj)
                if parsed:
                    return ParsedLLMNotificationAssessment(
                        **{**parsed.__dict__, "parse_mode": "json_repaired", "raw_extracted": repaired}
                    )
                errors.extend(parse_errors)
            except Exception as exc2:
                errors.append(f"json_repair_error: {exc2}")

    kv_obj = parse_kv_fallback(raw_text)
    if kv_obj:
        parsed, parse_errors = validate_and_normalize(kv_obj)
        if parsed:
            return ParsedLLMNotificationAssessment(
                **{**parsed.__dict__, "parse_mode": "kv_fallback", "raw_extracted": raw_text}
            )
        errors.extend(parse_errors)

    return conservative_default("failed_default", errors=errors or ["Unable to parse scorer output"])


def build_repair_prompt(raw_model_output: str) -> str:
    return (
        "Convert the following text into strict JSON with keys "
        "intent,relevance,novelty,severity,certainty,delivery,rationale,evidence_target,evidence_peer.\n"
        "Return JSON only.\n\n"
        f"{raw_model_output}"
    )


def build_kv_fallback_prompt(raw_model_output: str) -> str:
    return (
        "Convert the following text into plain key=value lines with keys "
        "intent,relevance,novelty,severity,certainty,delivery,rationale,evidence_target,evidence_peer.\n"
        "Return only key=value lines.\n\n"
        f"{raw_model_output}"
    )
