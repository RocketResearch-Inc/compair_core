"""
Deterministic routing and delivery logic for notification candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple
import hashlib
import re

from .parse_llm_structured_output import ParsedLLMNotificationAssessment

BUCKET_VALUE: Dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def b(x: str) -> int:
    return BUCKET_VALUE.get((x or "").upper().strip(), 0)


def normalize_excerpt(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s.strip()


@dataclass(frozen=True)
class CandidateContext:
    user_id: str
    group_id: str
    target_doc_id: str
    target_chunk_id: str
    peer_doc_ids: Tuple[str, ...]
    group_name: str = ""
    target_doc_title: str = ""
    peer_doc_titles: Tuple[str, ...] = tuple()
    now_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    run_id: str = ""


@dataclass(frozen=True)
class DeliveryDecision:
    action: str
    reason: str
    priority: int
    dedupe_key: str
    requires_human_review: bool = False
    digest_bucket: str = "general"


def compute_dedupe_key(ctx: CandidateContext, a: ParsedLLMNotificationAssessment) -> str:
    base = f"{ctx.user_id}|{ctx.group_id}|{a.intent}|{ctx.target_chunk_id}"
    evidence = normalize_excerpt(a.evidence_target) + "|" + normalize_excerpt(a.evidence_peer)
    digest = hashlib.sha256((base + "|" + evidence).encode("utf-8")).hexdigest()[:16]
    return f"{base}|{digest}"


def intent_digest_bucket(intent: str) -> str:
    return {
        "potential_conflict": "conflicts",
        "relevant_update": "updates",
        "hidden_overlap": "overlaps",
        "quiet_validation": "validation",
    }.get(intent, "general")


def default_priority(a: ParsedLLMNotificationAssessment) -> int:
    score = 35 * b(a.severity) + 30 * b(a.certainty) + 20 * b(a.relevance) + 15 * b(a.novelty)
    return max(0, min(100, score // 2))


def should_never_push(intent: str) -> bool:
    return intent in {"hidden_overlap", "quiet_validation"}


def routing_decision(
    ctx: CandidateContext,
    a: ParsedLLMNotificationAssessment,
    *,
    allow_push: bool = True,
    honor_model_delivery_hint: bool = False,
) -> DeliveryDecision:
    dedupe_key = compute_dedupe_key(ctx, a)
    priority = default_priority(a)

    if b(a.relevance) == 0 and b(a.novelty) == 0 and b(a.severity) == 0:
        return DeliveryDecision(
            action="drop",
            reason="All key dimensions LOW; not worth delivering.",
            priority=0,
            dedupe_key=dedupe_key,
            digest_bucket=intent_digest_bucket(a.intent),
            requires_human_review=False,
        )

    if should_never_push(a.intent):
        return DeliveryDecision(
            action="digest",
            reason=f"{a.intent} is digest-only by policy.",
            priority=priority,
            dedupe_key=dedupe_key,
            digest_bucket=intent_digest_bucket(a.intent),
            requires_human_review=(b(a.certainty) == 0),
        )

    if honor_model_delivery_hint and a.delivery == "push" and allow_push:
        if b(a.certainty) >= 2 and b(a.severity) >= 1 and b(a.relevance) >= 1:
            return DeliveryDecision(
                action="push",
                reason="Model recommended push and passed safety gates.",
                priority=max(priority, 80),
                dedupe_key=dedupe_key,
                digest_bucket=intent_digest_bucket(a.intent),
                requires_human_review=False,
            )

    if a.intent == "potential_conflict":
        if allow_push and b(a.certainty) >= 2 and b(a.severity) >= 2 and b(a.relevance) >= 1:
            return DeliveryDecision(
                action="push",
                reason="High-confidence, high-severity potential conflict.",
                priority=max(priority, 90),
                dedupe_key=dedupe_key,
                digest_bucket="conflicts",
                requires_human_review=False,
            )
        return DeliveryDecision(
            action="digest",
            reason="Conflict did not meet push gates (need HIGH certainty+severity).",
            priority=priority,
            dedupe_key=dedupe_key,
            digest_bucket="conflicts",
            requires_human_review=(b(a.certainty) <= 1),
        )

    if a.intent == "relevant_update":
        if allow_push and b(a.certainty) >= 2 and b(a.severity) >= 2 and b(a.relevance) >= 2 and b(a.novelty) >= 1:
            return DeliveryDecision(
                action="push",
                reason="Urgent, highly relevant update with high certainty.",
                priority=max(priority, 85),
                dedupe_key=dedupe_key,
                digest_bucket="updates",
                requires_human_review=False,
            )
        return DeliveryDecision(
            action="digest",
            reason="Update is better suited for digest (or not urgent enough).",
            priority=priority,
            dedupe_key=dedupe_key,
            digest_bucket="updates",
            requires_human_review=(b(a.certainty) == 0),
        )

    return DeliveryDecision(
        action="digest",
        reason="Defaulted to digest for safety.",
        priority=priority,
        dedupe_key=dedupe_key,
        digest_bucket=intent_digest_bucket(a.intent),
        requires_human_review=True,
    )


@dataclass(frozen=True)
class DeliveryPolicy:
    max_push_per_day: int = 1
    max_digest_items_per_day: int = 12
    dedupe_window_hours: int = 72
    push_cooldown_minutes: int = 180
    drop_low_certainty_push: bool = True


def apply_policy_overrides(
    ctx: CandidateContext,
    decision: DeliveryDecision,
    a: ParsedLLMNotificationAssessment,
    policy: DeliveryPolicy,
    *,
    pushes_sent_last_24h: int,
    last_push_sent_at: Optional[datetime],
    seen_dedupe_key_within_window: bool,
) -> DeliveryDecision:
    if seen_dedupe_key_within_window:
        return DeliveryDecision(
            action="drop",
            reason=f"Duplicate within {policy.dedupe_window_hours}h window.",
            priority=0,
            dedupe_key=decision.dedupe_key,
            digest_bucket=decision.digest_bucket,
            requires_human_review=False,
        )

    if decision.action == "push":
        if pushes_sent_last_24h >= policy.max_push_per_day:
            return DeliveryDecision(
                action="digest",
                reason="Push quota reached; downgraded to digest.",
                priority=min(decision.priority, 70),
                dedupe_key=decision.dedupe_key,
                digest_bucket=decision.digest_bucket,
                requires_human_review=False,
            )
        if last_push_sent_at:
            delta = ctx.now_utc - last_push_sent_at
            if delta < timedelta(minutes=policy.push_cooldown_minutes):
                return DeliveryDecision(
                    action="digest",
                    reason="In push cooldown window; downgraded to digest.",
                    priority=min(decision.priority, 70),
                    dedupe_key=decision.dedupe_key,
                    digest_bucket=decision.digest_bucket,
                    requires_human_review=False,
                )
        if policy.drop_low_certainty_push and b(a.certainty) < 2:
            return DeliveryDecision(
                action="digest",
                reason="Push requires HIGH certainty; downgraded to digest.",
                priority=min(decision.priority, 70),
                dedupe_key=decision.dedupe_key,
                digest_bucket=decision.digest_bucket,
                requires_human_review=True,
            )
    return decision


@dataclass(frozen=True)
class DigestItem:
    user_id: str
    group_id: str
    intent: str
    title: str
    one_liner: str
    priority: int
    dedupe_key: str
    evidence_target: str = ""
    evidence_peer: str = ""
    rationale: Tuple[str, ...] = tuple()
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def make_digest_item(ctx: CandidateContext, a: ParsedLLMNotificationAssessment, decision: DeliveryDecision) -> DigestItem:
    title = {
        "potential_conflict": "Potential conflict detected",
        "relevant_update": "Relevant update",
        "hidden_overlap": "Possible overlap",
        "quiet_validation": "Quiet validation",
    }.get(a.intent, "Compair notification")
    one_liner = (a.rationale[0] if a.rationale else "").strip() or "Compair surfaced a review-worthy update."
    return DigestItem(
        user_id=ctx.user_id,
        group_id=ctx.group_id,
        intent=a.intent,
        title=title,
        one_liner=one_liner[:240],
        priority=decision.priority,
        dedupe_key=decision.dedupe_key,
        evidence_target=(a.evidence_target or "")[:280],
        evidence_peer=(a.evidence_peer or "")[:280],
        rationale=tuple((a.rationale or [])[:3]),
    )


def decide_and_queue(
    ctx: CandidateContext,
    a: ParsedLLMNotificationAssessment,
    policy: DeliveryPolicy,
    *,
    pushes_sent_last_24h: int,
    last_push_sent_at: Optional[datetime],
    seen_dedupe_key_within_window: bool,
    allow_push: bool = True,
) -> Tuple[DeliveryDecision, Optional[DigestItem]]:
    decision = routing_decision(ctx, a, allow_push=allow_push)
    decision = apply_policy_overrides(
        ctx,
        decision,
        a,
        policy,
        pushes_sent_last_24h=pushes_sent_last_24h,
        last_push_sent_at=last_push_sent_at,
        seen_dedupe_key_within_window=seen_dedupe_key_within_window,
    )
    digest_item = None
    if decision.action == "digest":
        digest_item = make_digest_item(ctx, a, decision)
    return decision, digest_item


def maybe_escalate_conflict(
    decision: DeliveryDecision,
    assessment: ParsedLLMNotificationAssessment,
    *,
    times_seen_in_last_7d: int,
    times_acknowledged_in_last_7d: int,
) -> DeliveryDecision:
    if assessment.intent != "potential_conflict":
        return decision
    if decision.action != "digest":
        return decision
    if times_seen_in_last_7d >= 3 and times_acknowledged_in_last_7d == 0:
        return DeliveryDecision(
            action=decision.action,
            reason="Repeated unresolved conflict; prioritize in digest.",
            priority=max(decision.priority, 75),
            dedupe_key=decision.dedupe_key,
            requires_human_review=True,
            digest_bucket=decision.digest_bucket,
        )
    return decision
