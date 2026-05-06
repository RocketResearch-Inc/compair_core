"""
Notification scoring + routing glue for Core.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import re

from sqlalchemy.orm import Session

from ..logger import log_event
from ..local_summary import assess_relation, best_grounded_excerpt, excerpt_tokens
from ..models import Chunk, Document, NotificationEvent
from .delivery_logic import (
    CandidateContext,
    DeliveryDecision,
    DeliveryPolicy,
    DigestItem,
    compute_dedupe_key,
    decide_and_queue,
    maybe_escalate_conflict,
)
from .llm_notification_scorer import NotificationScorer, NotificationScorerConfig
from .parse_llm_structured_output import ParsedLLMNotificationAssessment

logger = logging.getLogger(__name__)
_NOTIFICATION_EVIDENCE_CHARS = 600
_CONFLICT_SUMMARY_TERMS = (
    "conflict",
    "drift",
    "mismatch",
    "regression",
    "contradict",
    "contradiction",
    "incompatible",
    "inconsistent",
    "rename",
    "renamed",
    "missing integration",
)
_HIGH_IMPACT_TERMS = (
    "api",
    "capabilities",
    "endpoint",
    "route",
    "field",
    "schema",
    "auth",
    "oauth",
    "env",
    "payload",
)
_RATIONALE_PROMOTION_BLOCKERS = (
    "no explicit contradiction",
    "no explicit disagreement",
    "no explicit conflicting",
    "no direct contradiction",
    "which is consistent",
    "aligning with",
    "aligns with",
    "different surfaces",
    "unrelated",
    "no evident schema drift or contradiction",
)
_RATIONALE_CONFLICT_DOWNGRADE_MARKERS = _RATIONALE_PROMOTION_BLOCKERS + (
    "not directly confirming or contradicting",
    "different files rather than conflicting behavior",
    "overlap without clear contradiction",
    "peer reinforces target",
    "peer reinforces the target",
    "aligning with target",
)
_RATIONALE_WEAK_EVIDENCE_MARKERS = (
    "incomplete",
    "truncated",
    "appears truncated",
)
_SNAPSHOT_FILE_RE = re.compile(r"^### File:\s+(.+?)(?:\s+\(.*)?$", re.MULTILINE)


def _as_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def is_scoring_enabled() -> bool:
    default = os.getenv("COMPAIR_EDITION", "core").lower() != "cloud"
    if os.getenv("COMPAIR_NOTIFICATION_SCORING_ENABLED") is not None:
        return _bool_env("COMPAIR_NOTIFICATION_SCORING_ENABLED", default)
    return _bool_env("NOTIFICATION_SCORING_ENABLED", default)


def _notification_score_trace_enabled() -> bool:
    return _bool_env("COMPAIR_NOTIFICATION_SCORING_TRACE", False) or _bool_env("NOTIFICATION_SCORING_TRACE", False)


@dataclass(frozen=True)
class PeerCandidate:
    doc_id: str
    doc_title: str
    chunk_id: str
    chunk_text: str
    doc_type: str = ""
    author_role: str = ""
    author_team: str = ""
    last_modified_utc: Optional[str] = None
    similarity: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class NotificationCandidate:
    user_id: str
    group_id: str
    target_doc_id: str
    target_chunk_id: str
    target_text: str
    peer_candidates: Tuple[PeerCandidate, ...]
    target_doc_title: str = ""
    target_doc_type: str = ""
    target_last_modified_utc: Optional[str] = None
    user_role: str = ""
    user_team: str = ""
    user_is_doc_author: bool = False
    user_is_group_admin: bool = False
    generated_feedback: Optional[Dict[str, Any]] = None
    run_id: str = ""
    now_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _grounded_excerpt_score(excerpt: str, signal_texts: List[str], preferred_text: str = "") -> int:
    excerpt_tokens_set = excerpt_tokens(excerpt)
    signal_tokens = excerpt_tokens(*signal_texts)
    preferred_tokens = excerpt_tokens(preferred_text)
    if not excerpt_tokens_set:
        return 0
    return len(excerpt_tokens_set & signal_tokens) + (2 * len(excerpt_tokens_set & preferred_tokens))


def _bucket_rank(value: str) -> int:
    return {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get((value or "").upper(), 0)


def _cap_bucket(value: str, max_value: str) -> str:
    return value if _bucket_rank(value) <= _bucket_rank(max_value) else max_value


def _raise_bucket(value: str, min_value: str) -> str:
    return value if _bucket_rank(value) >= _bucket_rank(min_value) else min_value


def _feedback_summary(candidate: NotificationCandidate) -> str:
    if not isinstance(candidate.generated_feedback, dict):
        return ""
    summary = candidate.generated_feedback.get("summary")
    return summary.strip() if isinstance(summary, str) else ""


def _feedback_focus_text(candidate: NotificationCandidate) -> str:
    if not isinstance(candidate.generated_feedback, dict):
        return ""
    focus_text = candidate.generated_feedback.get("focus_text")
    return focus_text.strip() if isinstance(focus_text, str) else ""


def _extract_snapshot_file_path(text: str) -> str:
    match = _SNAPSHOT_FILE_RE.search(text or "")
    if not match:
        return ""
    return (match.group(1) or "").strip()


def _ground_notification_assessment(
    candidate: NotificationCandidate,
    assessment: ParsedLLMNotificationAssessment,
) -> ParsedLLMNotificationAssessment:
    signal_texts = [
        text
        for text in [
            _feedback_focus_text(candidate),
            _feedback_summary(candidate),
            assessment.evidence_target,
            assessment.evidence_peer,
            *assessment.rationale,
        ]
        if text
    ]

    grounded_target = best_grounded_excerpt(
        candidate.target_text,
        signal_texts,
        assessment.evidence_target,
        limit=_NOTIFICATION_EVIDENCE_CHARS,
    )

    grounded_peer = ""
    best_peer_score = 0
    for peer in candidate.peer_candidates:
        excerpt = best_grounded_excerpt(
            peer.chunk_text,
            signal_texts,
            assessment.evidence_peer,
            limit=_NOTIFICATION_EVIDENCE_CHARS,
        )
        if not excerpt:
            continue
        score = _grounded_excerpt_score(excerpt, signal_texts, assessment.evidence_peer)
        if score > best_peer_score:
            best_peer_score = score
            grounded_peer = excerpt

    certainty = assessment.certainty
    delivery = assessment.delivery
    if not grounded_target and not grounded_peer:
        certainty = _cap_bucket(certainty, "LOW")
        delivery = "digest"
    elif not grounded_target or not grounded_peer:
        certainty = _cap_bucket(certainty, "MEDIUM")
        delivery = "digest"

    return replace(
        assessment,
        certainty=certainty,
        delivery=delivery,
        evidence_target=grounded_target,
        evidence_peer=grounded_peer,
    )


def _calibrate_assessment_from_feedback(
    candidate: NotificationCandidate,
    assessment: ParsedLLMNotificationAssessment,
) -> ParsedLLMNotificationAssessment:
    summary = _feedback_summary(candidate)
    if not summary:
        return assessment
    if assessment.intent not in {"hidden_overlap", "quiet_validation"}:
        return assessment

    lowered = summary.lower()
    if not any(term in lowered for term in _CONFLICT_SUMMARY_TERMS):
        return assessment
    rationale_text = " ".join(assessment.rationale).lower()
    if any(marker in rationale_text for marker in _RATIONALE_PROMOTION_BLOCKERS):
        return assessment

    target_excerpt = assessment.evidence_target or best_grounded_excerpt(
        candidate.target_text,
        [summary],
        summary,
        limit=_NOTIFICATION_EVIDENCE_CHARS,
    )
    peer_excerpt = assessment.evidence_peer
    relation = assess_relation(target_excerpt or "", peer_excerpt or "")

    promoted_intent = "relevant_update"
    if relation.kind in {"route/path mismatch", "value mismatch", "rename", "docs-vs-impl mismatch"}:
        promoted_intent = "potential_conflict"

    min_severity = "MEDIUM"
    min_relevance = "MEDIUM"
    min_novelty = "MEDIUM"
    min_certainty = "MEDIUM"
    if promoted_intent == "potential_conflict" and any(term in lowered for term in _HIGH_IMPACT_TERMS):
        min_severity = "HIGH"
        min_relevance = "HIGH"

    rationale = list(assessment.rationale[:2])
    rationale.append("Generated feedback describes a mismatch/drift, so this is not treated as benign overlap.")

    return replace(
        assessment,
        intent=promoted_intent,
        severity=_raise_bucket(assessment.severity, min_severity),
        relevance=_raise_bucket(assessment.relevance, min_relevance),
        novelty=_raise_bucket(assessment.novelty, min_novelty),
        certainty=_raise_bucket(assessment.certainty, min_certainty),
        delivery="digest",
        rationale=rationale[:3],
        evidence_target=target_excerpt or assessment.evidence_target,
        evidence_peer=peer_excerpt or assessment.evidence_peer,
    )


def _enforce_assessment_consistency(
    candidate: NotificationCandidate,
    assessment: ParsedLLMNotificationAssessment,
) -> ParsedLLMNotificationAssessment:
    rationale_text = " ".join(assessment.rationale).lower()
    summary = _feedback_summary(candidate)
    lowered_summary = summary.lower()
    denies_conflict = any(marker in rationale_text for marker in _RATIONALE_CONFLICT_DOWNGRADE_MARKERS)
    weak_evidence = any(marker in rationale_text for marker in _RATIONALE_WEAK_EVIDENCE_MARKERS)
    summary_signals_conflict = any(term in lowered_summary for term in _CONFLICT_SUMMARY_TERMS)

    if assessment.intent == "potential_conflict" and denies_conflict:
        downgraded_intent = "relevant_update" if summary_signals_conflict else "hidden_overlap"
        max_severity = "MEDIUM" if summary_signals_conflict and not weak_evidence else "LOW"
        max_relevance = "MEDIUM" if summary_signals_conflict else "LOW"
        max_novelty = "MEDIUM" if summary_signals_conflict else "LOW"
        return replace(
            assessment,
            intent=downgraded_intent,
            severity=_cap_bucket(assessment.severity, max_severity),
            relevance=_cap_bucket(assessment.relevance, max_relevance),
            novelty=_cap_bucket(assessment.novelty, max_novelty),
            certainty=_cap_bucket(assessment.certainty, "MEDIUM"),
            delivery="digest",
        )

    if assessment.intent == "potential_conflict" and weak_evidence and not summary_signals_conflict:
        return replace(
            assessment,
            intent="relevant_update",
            severity=_cap_bucket(assessment.severity, "LOW"),
            relevance=_cap_bucket(assessment.relevance, "LOW"),
            novelty=_cap_bucket(assessment.novelty, "LOW"),
            certainty=_cap_bucket(assessment.certainty, "MEDIUM"),
            delivery="digest",
        )

    return assessment


def _build_payload(candidate: NotificationCandidate) -> Dict[str, Any]:
    def _cap_text(value: Optional[str], limit: int) -> Optional[str]:
        if not value:
            return value
        value = value.strip()
        return value if len(value) <= limit else value[:limit]

    feedback_summary = _feedback_summary(candidate)
    feedback_focus = _feedback_focus_text(candidate)
    target_signals = [text for text in [feedback_focus, feedback_summary] if text]
    if not target_signals:
        target_signals = [peer.chunk_text for peer in candidate.peer_candidates[:2]]
    target_excerpt = best_grounded_excerpt(
        candidate.target_text,
        target_signals,
        feedback_focus or feedback_summary,
        anchor_texts=[feedback_focus] if feedback_focus else (),
        limit=360,
    )
    peers = []
    for p in candidate.peer_candidates:
        peer_signals = [text for text in [feedback_focus, feedback_summary] if text]
        if not peer_signals:
            peer_signals = [candidate.target_text]
        peer_excerpt = best_grounded_excerpt(
            p.chunk_text,
            peer_signals,
            feedback_focus or feedback_summary,
            anchor_texts=[feedback_focus] if feedback_focus else (),
            limit=360,
        )
        peers.append(
            {
                "peer_doc_id": p.doc_id,
                "peer_doc_title": p.doc_title,
                "peer_doc_type": p.doc_type,
                "peer_chunk_id": p.chunk_id,
                "peer_file_path": _extract_snapshot_file_path(p.chunk_text),
                "peer_chunk_text": _cap_text(p.chunk_text, 1600),
                "peer_excerpt": peer_excerpt,
                "peer_author_role": p.author_role,
                "peer_author_team": p.author_team,
                "peer_last_modified_utc": p.last_modified_utc,
                "similarity": p.similarity,
            }
        )

    return {
        "run_context": {
            "run_id": candidate.run_id,
            "now_utc": candidate.now_utc.isoformat(),
        },
        "user_context": {
            "user_id": candidate.user_id,
            "user_role": candidate.user_role,
            "user_team": candidate.user_team,
            "user_is_doc_author": candidate.user_is_doc_author,
            "user_is_group_admin": candidate.user_is_group_admin,
        },
        "target": {
            "doc_id": candidate.target_doc_id,
            "doc_title": candidate.target_doc_title,
            "doc_type": candidate.target_doc_type,
            "doc_last_modified_utc": candidate.target_last_modified_utc,
            "chunk_id": candidate.target_chunk_id,
            "chunk_file_path": _extract_snapshot_file_path(candidate.target_text),
            "chunk_text": _cap_text(candidate.target_text, 2000),
            "chunk_excerpt": target_excerpt,
        },
        "candidates": peers,
        "generated_feedback": candidate.generated_feedback,
    }


def _assessment_trace_dict(assessment: ParsedLLMNotificationAssessment) -> Dict[str, Any]:
    return {
        "intent": assessment.intent,
        "relevance": assessment.relevance,
        "novelty": assessment.novelty,
        "severity": assessment.severity,
        "certainty": assessment.certainty,
        "delivery": assessment.delivery,
        "parse_mode": assessment.parse_mode,
        "errors": list(assessment.errors or []),
        "rationale": list(assessment.rationale or []),
        "evidence_target": assessment.evidence_target,
        "evidence_peer": assessment.evidence_peer,
    }


def _count_pushes_last_24h(session: Session, user_id: str) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
    try:
        return (
            session.query(NotificationEvent)
            .filter(
                NotificationEvent.user_id == user_id,
                NotificationEvent.delivery_action == "push",
                NotificationEvent.created_at >= cutoff,
            )
            .count()
        )
    except Exception as exc:
        logger.warning("NotificationEvent push count failed: %s", exc)
        return 0


def _last_push_sent_at(session: Session, user_id: str) -> Optional[datetime]:
    try:
        record = (
            session.query(NotificationEvent)
            .filter(
                NotificationEvent.user_id == user_id,
                NotificationEvent.delivery_action == "push",
            )
            .order_by(NotificationEvent.created_at.desc())
            .first()
        )
        return _as_utc(record.created_at) if record else None
    except Exception as exc:
        logger.warning("NotificationEvent last push lookup failed: %s", exc)
        return None


def _seen_dedupe_key(session: Session, dedupe_key: str, hours: int) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    try:
        exists = (
            session.query(NotificationEvent.event_id)
            .filter(
                NotificationEvent.dedupe_key == dedupe_key,
                NotificationEvent.created_at >= cutoff,
            )
            .first()
        )
        return bool(exists)
    except Exception as exc:
        logger.warning("NotificationEvent dedupe lookup failed: %s", exc)
        return False


def _times_seen_in_last_7d(session: Session, dedupe_key: str) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    try:
        return (
            session.query(NotificationEvent.event_id)
            .filter(
                NotificationEvent.dedupe_key == dedupe_key,
                NotificationEvent.created_at >= cutoff,
            )
            .count()
        )
    except Exception as exc:
        logger.warning("NotificationEvent times-seen lookup failed: %s", exc)
        return 0


def _times_acknowledged_in_last_7d(session: Session, dedupe_key: str) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    try:
        return (
            session.query(NotificationEvent.event_id)
            .filter(
                NotificationEvent.dedupe_key == dedupe_key,
                NotificationEvent.acknowledged_at.isnot(None),
                NotificationEvent.acknowledged_at >= cutoff,
            )
            .count()
        )
    except Exception as exc:
        logger.warning("NotificationEvent acknowledgement lookup failed: %s", exc)
        return 0


def _persist_event(
    session: Session,
    candidate: NotificationCandidate,
    assessment: ParsedLLMNotificationAssessment,
    decision: DeliveryDecision,
    *,
    model: str,
    channel: Optional[str],
    run_id: str,
) -> Optional[NotificationEvent]:
    try:
        target_doc_id = candidate.target_doc_id or None
        if target_doc_id:
            exists = session.query(Document.document_id).filter(Document.document_id == target_doc_id).first()
            if not exists:
                logger.warning("NotificationEvent target_doc_id missing: %s", target_doc_id)
                target_doc_id = None

        target_chunk_id = candidate.target_chunk_id or None
        if target_chunk_id:
            exists = session.query(Chunk.chunk_id).filter(Chunk.chunk_id == target_chunk_id).first()
            if not exists:
                logger.warning("NotificationEvent target_chunk_id missing: %s", target_chunk_id)
                target_chunk_id = None

        event = NotificationEvent(
            user_id=candidate.user_id,
            group_id=candidate.group_id,
            intent=assessment.intent,
            dedupe_key=decision.dedupe_key,
            target_doc_id=target_doc_id,
            target_chunk_id=target_chunk_id,
            peer_doc_ids=[p.doc_id for p in candidate.peer_candidates],
            relevance=assessment.relevance,
            novelty=assessment.novelty,
            severity=assessment.severity,
            certainty=assessment.certainty,
            delivery_action=decision.action,
            channel=channel,
            parse_mode=assessment.parse_mode,
            model=model,
            run_id=run_id,
            digest_bucket=decision.digest_bucket,
            rationale=assessment.rationale[:3],
            evidence_target=assessment.evidence_target[:600],
            evidence_peer=assessment.evidence_peer[:600],
        )
        session.add(event)
        session.flush()
        return event
    except Exception as exc:
        logger.warning("NotificationEvent persist failed: %s", exc)
        return None


def score_and_route_candidate(
    session: Session,
    candidate: NotificationCandidate,
    *,
    scorer: Optional[NotificationScorer] = None,
    policy: Optional[DeliveryPolicy] = None,
    allow_push: bool = True,
    commit: bool = False,
    delivery_channel: str | None = "inbox_only",
) -> Optional[Tuple[DeliveryDecision, ParsedLLMNotificationAssessment, Optional[DigestItem], Optional[NotificationEvent]]]:
    if not is_scoring_enabled():
        logger.info("Notification scoring skipped (feature flag off).")
        return None

    policy = policy or DeliveryPolicy()
    scorer = scorer or NotificationScorer(NotificationScorerConfig())

    payload = _build_payload(candidate)
    if _notification_score_trace_enabled():
        log_event(
            "notification_score_trace",
            stage="input",
            run_id=candidate.run_id,
            target_doc_id=candidate.target_doc_id,
            target_chunk_id=candidate.target_chunk_id,
            payload=payload,
        )
    assessment = scorer.score(payload)
    assessment = _ground_notification_assessment(candidate, assessment)
    assessment = _calibrate_assessment_from_feedback(candidate, assessment)
    assessment = _enforce_assessment_consistency(candidate, assessment)
    if _notification_score_trace_enabled():
        log_event(
            "notification_score_trace",
            stage="result",
            run_id=candidate.run_id,
            target_doc_id=candidate.target_doc_id,
            target_chunk_id=candidate.target_chunk_id,
            assessment=_assessment_trace_dict(assessment),
        )

    ctx = CandidateContext(
        user_id=candidate.user_id,
        group_id=candidate.group_id,
        target_doc_id=candidate.target_doc_id,
        target_chunk_id=candidate.target_chunk_id,
        peer_doc_ids=tuple(p.doc_id for p in candidate.peer_candidates),
        target_doc_title=candidate.target_doc_title,
        peer_doc_titles=tuple(p.doc_title for p in candidate.peer_candidates),
        now_utc=candidate.now_utc,
        run_id=candidate.run_id,
    )

    dedupe_key = compute_dedupe_key(ctx, assessment)
    pushes_sent = _count_pushes_last_24h(session, candidate.user_id)
    last_push = _last_push_sent_at(session, candidate.user_id)
    seen_dedupe = _seen_dedupe_key(session, dedupe_key, policy.dedupe_window_hours)

    final_decision, digest_item = decide_and_queue(
        ctx,
        assessment,
        policy,
        pushes_sent_last_24h=pushes_sent,
        last_push_sent_at=last_push,
        seen_dedupe_key_within_window=seen_dedupe,
        allow_push=allow_push,
    )

    final_decision = maybe_escalate_conflict(
        final_decision,
        assessment,
        times_seen_in_last_7d=_times_seen_in_last_7d(session, final_decision.dedupe_key),
        times_acknowledged_in_last_7d=_times_acknowledged_in_last_7d(session, final_decision.dedupe_key),
    )

    channel = None if final_decision.action == "drop" else delivery_channel
    event = _persist_event(
        session,
        candidate,
        assessment,
        final_decision,
        model=scorer.config.model,
        channel=channel,
        run_id=candidate.run_id,
    )

    if commit:
        try:
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.warning("NotificationEvent commit failed: %s", exc)

    return final_decision, assessment, digest_item, event
