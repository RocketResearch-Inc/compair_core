from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence


INTENTS = ["potential_conflict", "relevant_update", "hidden_overlap", "quiet_validation"]
SEVERITIES = ["low", "medium", "high"]
CERTAINTIES = ["low", "medium", "high"]

FINDINGS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["findings"],
    "properties": {
        "findings": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "intent",
                    "severity",
                    "certainty",
                    "title",
                    "summary",
                    "why_it_matters",
                    "target_repos",
                    "target_files",
                    "peer_repos",
                    "peer_files",
                    "evidence_target",
                    "evidence_peer",
                    "follow_up",
                ],
                "properties": {
                    "intent": {"type": "string", "enum": INTENTS},
                    "severity": {"type": "string", "enum": SEVERITIES},
                    "certainty": {"type": "string", "enum": CERTAINTIES},
                    "title": {"type": "string"},
                    "summary": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "target_repos": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
                    "target_files": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
                    "peer_repos": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
                    "peer_files": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                    "evidence_target": {"type": "string"},
                    "evidence_peer": {"type": "string"},
                    "follow_up": {"type": "string"},
                },
            },
        }
    },
}

SYSTEM_PROMPT_TEMPLATE = """# Identity
You are a cross-repo review assistant inside Compair. You compare repository snapshots and surface concrete implementation mismatches, integration risks, information gaps, meaningful overlap, or strong confirmations across repos.

# Purpose
You are reviewing the current state of a multi-repo product surface without assuming anything is wrong. Your job is to surface only the strongest evidence-backed cross-repo findings.

# Allowed finding buckets
- potential_conflict: explicit contradiction, docs-vs-implementation drift, API/schema drift, route/path mismatch, rollout/status mismatch, release/policy mismatch
- relevant_update: a likely knowledge gap or important cross-repo update that one team should know about even if it is not a hard contradiction
- hidden_overlap: duplicated or parallel work across repos without a clear contradiction
- quiet_validation: strong reinforcement or confirmation across repos that reduces uncertainty

# Instructions
- Start from the repository snapshots below. There may be zero meaningful findings.
- Only report findings that require comparing multiple repos.
- Prioritize concrete observations over broad architecture commentary.
- Mention specific repo names, file paths, endpoints, capability fields, env vars, workflow steps, or product-surface claims when supported.
- Stay grounded in the provided text. If evidence is weak, omit the finding.
- Do not force an even distribution across the finding buckets.
- Prefer up to {max_findings} findings, ordered by likely user impact.
- Treat `relevant_update` as the bucket for knowledge-gap style findings.
- A larger context window does not guarantee equal attention; focus on the strongest supported contradictions, gaps, overlaps, or validations you can actually defend from the input.

# Output
Return JSON only, matching the provided schema exactly.
"""

USER_PROMPT_TEMPLATE = """Review the current multi-repo product surface below and surface the strongest cross-repo findings.

Do not assume that anything is wrong. Report only findings that are directly supported by the provided repo content.

Repository bundle:
{bundle}
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def estimate_tokens(text: str) -> int:
    return int(math.ceil(len(text or "") / 4.0))


def _float_env(*names: str) -> float | None:
    for name in names:
        raw = os.getenv(name)
        if raw is None or not raw.strip():
            continue
        try:
            return float(raw.strip())
        except ValueError:
            continue
    return None


def estimate_usage_cost(
    *,
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    prompt_estimated_tokens: int | None = None,
) -> dict[str, Any] | None:
    input_rate = _float_env(
        "COMPAIR_NOW_REVIEW_INPUT_COST_PER_1M_USD",
        "NOW_REVIEW_INPUT_COST_PER_1M_USD",
    )
    output_rate = _float_env(
        "COMPAIR_NOW_REVIEW_OUTPUT_COST_PER_1M_USD",
        "NOW_REVIEW_OUTPUT_COST_PER_1M_USD",
    )
    if input_rate is None and output_rate is None:
        return None

    effective_input = int(input_tokens if input_tokens is not None else (prompt_estimated_tokens or 0))
    effective_output = int(output_tokens or 0)
    input_cost = max(0.0, float(effective_input) / 1_000_000.0) * float(input_rate or 0.0)
    output_cost = max(0.0, float(effective_output) / 1_000_000.0) * float(output_rate or 0.0)
    return {
        "pricing_source": "env",
        "model": model,
        "input_tokens": effective_input,
        "output_tokens": effective_output,
        "input_rate_per_million_usd": float(input_rate or 0.0),
        "output_rate_per_million_usd": float(output_rate or 0.0),
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
        "estimated_input_tokens": input_tokens is None,
        "estimated_output_tokens": output_tokens is None,
    }


def _bundle_doc_sort_key(doc: Any) -> tuple[int, str, str]:
    doc_type = str(getattr(doc, "doc_type", "") or "").strip().lower()
    title = str(getattr(doc, "title", "") or "").strip().lower()
    document_id = str(getattr(doc, "document_id", "") or "").strip().lower()
    type_rank = 0 if doc_type == "code-repo" else 1
    return (type_rank, title, document_id)


def build_document_bundle(documents: Sequence[Any]) -> tuple[list[dict[str, Any]], str]:
    ordered_docs = sorted(documents, key=_bundle_doc_sort_key)
    stats: list[dict[str, Any]] = []
    parts: list[str] = []
    for doc in ordered_docs:
        title = str(getattr(doc, "title", "") or "").strip() or "Untitled document"
        doc_type = str(getattr(doc, "doc_type", "") or "").strip() or "document"
        document_id = str(getattr(doc, "document_id", "") or "").strip()
        content = str(getattr(doc, "content", "") or "")
        char_count = len(content)
        token_estimate = estimate_tokens(content)
        stats.append(
            {
                "document_id": document_id,
                "title": title,
                "doc_type": doc_type,
                "chars": char_count,
                "estimated_tokens": token_estimate,
            }
        )
        parts.append(f"===== DOCUMENT {title} [{document_id or 'unknown'}] =====")
        parts.append(f"<<<DOC TYPE {doc_type} TITLE {title} ID {document_id or 'unknown'}>>>")
        parts.append(content)
        parts.append("<<<END DOC>>>")
    return stats, "\n".join(parts)


def _coerce_str_list(value: Any, *, limit: int) -> list[str]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


def normalize_findings_payload(data: Mapping[str, Any]) -> dict[str, Any]:
    findings_in = data.get("findings")
    if not isinstance(findings_in, list):
        findings_in = []
    findings: list[dict[str, Any]] = []
    for item in findings_in[:12]:
        if not isinstance(item, Mapping):
            continue
        findings.append(
            {
                "intent": str(item.get("intent") or "relevant_update").strip().lower(),
                "severity": str(item.get("severity") or "low").strip().lower(),
                "certainty": str(item.get("certainty") or "low").strip().lower(),
                "title": str(item.get("title") or "Untitled finding").strip(),
                "summary": str(item.get("summary") or "").strip(),
                "why_it_matters": str(item.get("why_it_matters") or "").strip(),
                "target_repos": _coerce_str_list(item.get("target_repos"), limit=4),
                "target_files": _coerce_str_list(item.get("target_files"), limit=8),
                "peer_repos": _coerce_str_list(item.get("peer_repos"), limit=6),
                "peer_files": _coerce_str_list(item.get("peer_files"), limit=12),
                "evidence_target": str(item.get("evidence_target") or "").strip(),
                "evidence_peer": str(item.get("evidence_peer") or "").strip(),
                "follow_up": str(item.get("follow_up") or "").strip(),
            }
        )
    return {"findings": findings}


def extract_json_object(text: str) -> Mapping[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    if not cleaned:
        return {"findings": []}
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, Mapping):
        raise ValueError("Model response did not decode to a JSON object")
    return parsed


def render_now_review_markdown(
    *,
    group_name: str,
    findings: Sequence[Mapping[str, Any]],
    meta: Mapping[str, Any],
    document_stats: Sequence[Mapping[str, Any]],
) -> str:
    lines: list[str] = [
        f"# Compair Now Review: {group_name}",
        "",
        "## Summary",
        "",
        f"- Model: `{meta.get('model') or 'unknown'}`",
        f"- Documents reviewed: `{len(document_stats)}`",
        f"- Estimated prompt tokens: `{meta.get('prompt_estimated_tokens') or 0}`",
        f"- Findings returned: `{len(findings)}`",
    ]
    total_cost = meta.get("cost_estimate_usd", {}).get("total_cost_usd") if isinstance(meta.get("cost_estimate_usd"), Mapping) else None
    if total_cost is not None:
        lines.append(f"- Estimated cost: `${total_cost:.6f}`")
    duration = meta.get("duration_sec")
    if duration is not None:
        lines.append(f"- Duration: `{duration}`s")
    lines.extend(["", "## Findings", ""])
    if not findings:
        lines.append("No strong cross-repo findings were returned.")
    for idx, finding in enumerate(findings, start=1):
        title = str(finding.get("title") or f"Finding {idx}").strip()
        intent = str(finding.get("intent") or "relevant_update").strip()
        severity = str(finding.get("severity") or "low").strip()
        certainty = str(finding.get("certainty") or "low").strip()
        lines.append(f"### {idx}. {title}")
        lines.append("")
        lines.append(f"- Intent: `{intent}`")
        lines.append(f"- Severity: `{severity}`")
        lines.append(f"- Certainty: `{certainty}`")
        summary = str(finding.get("summary") or "").strip()
        if summary:
            lines.extend(["", summary])
        why = str(finding.get("why_it_matters") or "").strip()
        if why:
            lines.extend(["", f"Why it matters: {why}"])
        target = str(finding.get("evidence_target") or "").strip()
        if target:
            lines.extend(["", f"Target evidence: `{target}`"])
        peer = str(finding.get("evidence_peer") or "").strip()
        if peer:
            lines.extend(["", f"Peer evidence: `{peer}`"])
        follow_up = str(finding.get("follow_up") or "").strip()
        if follow_up:
            lines.extend(["", f"Follow-up: {follow_up}"])
        target_files = _coerce_str_list(finding.get("target_files"), limit=8)
        if target_files:
            lines.extend(["", "Target files:", *[f"- `{item}`" for item in target_files]])
        peer_files = _coerce_str_list(finding.get("peer_files"), limit=12)
        if peer_files:
            lines.extend(["", "Peer files:", *[f"- `{item}`" for item in peer_files]])
        lines.append("")
    if document_stats:
        lines.extend(["## Bundle", ""])
        for doc in document_stats:
            title = str(doc.get("title") or "Untitled document").strip()
            doc_type = str(doc.get("doc_type") or "document").strip()
            document_id = str(doc.get("document_id") or "").strip()
            estimated_tokens = int(doc.get("estimated_tokens") or 0)
            lines.append(f"- `{title}` ({doc_type}, {estimated_tokens} est. tokens, `{document_id or 'unknown'}`)")
    return "\n".join(lines).rstrip() + "\n"
