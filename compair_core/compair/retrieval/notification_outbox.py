"""Privacy-safe, injected in-app outbox for successful baseline generation.

The outbox stores only finding identifiers and ordinals.  It never imports or
invokes Core's legacy notification scorer/delivery path, and it has no default
external sink.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Protocol
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session

from ...baseline_evidence_schema import (
    SOURCE_SCOPE_CONTROL_DOCUMENT,
    SOURCE_SCOPE_LEGACY_CHUNK,
    SOURCE_SCOPE_VERSION,
)

BASELINE_NOTIFICATION_OUTBOX_TABLE = "baseline_notification_outbox"
BASELINE_NOTIFICATION_PAYLOAD_SCHEMA_VERSION = "baseline-notification-digest.v1"
BASELINE_NOTIFICATION_CHANNEL = "in_app"
BASELINE_NOTIFICATIONS_ENABLED_ENV = "COMPAIR_BASELINE_NOTIFICATIONS_ENABLED"
DEFAULT_OUTBOX_LEASE_SECONDS = 300


class BaselineNotificationState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DELIVERED = "delivered"
    RETRYABLE_FAILED = "retryable_failed"
    TERMINAL_FAILED = "terminal_failed"
    SUPPRESSED = "suppressed"
    CANCELLED = "cancelled"


class BaselineNotificationOutboxError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class BaselineNotificationSinkError(BaselineNotificationOutboxError):
    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        self.retryable = retryable
        super().__init__(code, message)


@dataclass(frozen=True, slots=True)
class BaselineNotificationFindingIdentifier:
    ordinal: int
    feedback_id: str


@dataclass(frozen=True, slots=True)
class BaselineNotificationDigest:
    outbox_id: str
    run_id: str
    group_id: str
    recipient_user_id: str
    channel: str
    digest_key: str
    finding_count: int
    findings: tuple[BaselineNotificationFindingIdentifier, ...]
    finding_manifest_hash: str
    payload_schema_version: str = BASELINE_NOTIFICATION_PAYLOAD_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class BaselineNotificationDispatchReceipt:
    outbox_id: str
    state: BaselineNotificationState
    attempt_count: int
    error_code: str | None = None


class BaselineNotificationSink(Protocol):
    channel: str
    supports_idempotency: bool

    def deliver(
        self,
        digest: BaselineNotificationDigest,
        *,
        idempotency_key: str,
    ) -> None: ...


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_error_code(value: object, fallback: str) -> str:
    candidate = re.sub(r"[^a-z0-9_.-]+", "_", str(value or "").strip().lower()).strip(
        "_.-"
    )
    return (candidate or fallback)[:128]


def baseline_notifications_enabled() -> bool:
    """Return the explicit baseline-only flag, defaulting securely to false."""

    raw = os.getenv(BASELINE_NOTIFICATIONS_ENABLED_ENV)
    if raw is None:
        return False
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise BaselineNotificationOutboxError(
        "baseline_notifications_config_invalid",
        f"{BASELINE_NOTIFICATIONS_ENABLED_ENV} must be an explicit boolean",
    )


def _begin_write(session: Session) -> None:
    if session.get_bind().dialect.name == "sqlite":
        session.connection().exec_driver_sql("BEGIN IMMEDIATE")
    else:
        session.begin()


def _canonical_manifest(
    feedback_ids: Sequence[str],
) -> tuple[str, tuple[BaselineNotificationFindingIdentifier, ...]]:
    if not 1 <= len(feedback_ids) <= 4:
        raise BaselineNotificationOutboxError(
            "baseline_notification_findings_invalid",
            "baseline notification finding count is invalid",
        )
    findings = tuple(
        BaselineNotificationFindingIdentifier(ordinal, str(feedback_id))
        for ordinal, feedback_id in enumerate(feedback_ids, start=1)
    )
    if any(
        not finding.feedback_id
        or len(finding.feedback_id) > 36
        or finding.feedback_id != finding.feedback_id.strip()
        for finding in findings
    ):
        raise BaselineNotificationOutboxError(
            "baseline_notification_findings_invalid",
            "baseline notification finding identity is invalid",
        )
    manifest = json.dumps(
        {
            "schema_version": BASELINE_NOTIFICATION_PAYLOAD_SCHEMA_VERSION,
            "findings": [
                {"ordinal": finding.ordinal, "feedback_id": finding.feedback_id}
                for finding in findings
            ],
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return manifest, findings


def _digest_key(run_id: str, recipient_user_id: str, channel: str) -> str:
    return _sha256(
        f"{BASELINE_NOTIFICATION_PAYLOAD_SCHEMA_VERSION}\x00{run_id}"
        f"\x00{recipient_user_id}\x00{channel}"
    )


def schedule_baseline_notification(
    session: Session,
    *,
    run_id: str,
    group_id: str,
    recipient_user_id: str,
    feedback_ids: Sequence[str],
    enabled: bool,
    now: datetime,
    control_generation_lease_token: str | None = None,
) -> str:
    """Insert one group-scoped digest inside the caller's Feedback transaction."""

    manifest, _findings = _canonical_manifest(feedback_ids)
    manifest_hash = _sha256(manifest)
    digest_key = _digest_key(run_id, recipient_user_id, BASELINE_NOTIFICATION_CHANNEL)
    authorized = session.execute(
        text(
            "SELECT 1 FROM baseline_retrieval_run r "
            "JOIN document_to_group dtg ON dtg.document_id = r.source_document_id "
            "AND dtg.group_id = r.group_id "
            "JOIN user_to_group utg ON utg.group_id = r.group_id "
            "AND utg.user_id = :recipient_user_id "
            'JOIN "user" u ON u.user_id = utg.user_id '
            "WHERE r.run_id = :run_id AND r.group_id = :group_id "
            "AND r.generation_state = 'succeeded' AND ("
            "(r.source_scope = 'legacy_chunk' AND EXISTS (SELECT 1 FROM chunk c "
            "WHERE c.chunk_id = r.source_chunk_id "
            "AND c.document_id = r.source_document_id)) OR "
            "(r.source_scope = 'control_document' AND r.source_chunk_id IS NULL "
            "AND EXISTS (SELECT 1 FROM baseline_control_run_job j WHERE "
            "j.persisted_run_id = r.run_id AND j.group_id = r.group_id "
            "AND j.source_document_id = r.source_document_id AND ("
            "j.state IN ('references_persisted', 'feedback_persisted') OR "
            "(j.state = 'running' AND j.failure_stage = 'generation' "
            "AND :control_generation_lease_token IS NOT NULL "
            "AND j.lease_token = :control_generation_lease_token)))))"
        ),
        {
            "run_id": run_id,
            "group_id": group_id,
            "recipient_user_id": recipient_user_id,
            "control_generation_lease_token": control_generation_lease_token,
        },
    ).scalar_one_or_none()
    if authorized is None:
        raise BaselineNotificationOutboxError(
            "baseline_notification_schedule_unauthorized",
            "recipient is not authorized for baseline notification scheduling",
        )
    durable_findings = session.execute(
        text(
            "SELECT feedback_id, baseline_finding_ordinal FROM feedback "
            "WHERE baseline_retrieval_run_id = :run_id "
            "ORDER BY baseline_finding_ordinal"
        ),
        {"run_id": run_id},
    ).all()
    if tuple(
        (int(row.baseline_finding_ordinal), str(row.feedback_id))
        for row in durable_findings
    ) != tuple(enumerate((str(value) for value in feedback_ids), start=1)):
        raise BaselineNotificationOutboxError(
            "baseline_notification_schedule_findings_mismatch",
            "durable baseline findings do not match notification scheduling intent",
        )

    outbox_id = str(uuid4())
    state = (
        BaselineNotificationState.PENDING
        if enabled
        else BaselineNotificationState.SUPPRESSED
    )
    error_code = None if enabled else "baseline_notifications_disabled"
    error_fingerprint = None if error_code is None else _sha256(f"policy:{error_code}")
    session.execute(
        text(
            "INSERT INTO baseline_notification_outbox "
            "(outbox_id, run_id, group_id, recipient_user_id, channel, digest_key, "
            "payload_schema_version, finding_count, finding_manifest, "
            "finding_manifest_hash, state, lease_token, lease_expires_at, "
            "attempt_count, error_code, error_fingerprint, created_at, updated_at, "
            "delivered_at, suppressed_at, cancelled_at) VALUES "
            "(:outbox_id, :run_id, :group_id, :recipient_user_id, :channel, "
            ":digest_key, :payload_schema_version, :finding_count, :finding_manifest, "
            ":finding_manifest_hash, :state, NULL, NULL, 0, :error_code, "
            ":error_fingerprint, :created_at, :updated_at, NULL, :suppressed_at, NULL) "
            "ON CONFLICT (run_id, recipient_user_id, channel) DO NOTHING"
        ),
        {
            "outbox_id": outbox_id,
            "run_id": run_id,
            "group_id": group_id,
            "recipient_user_id": recipient_user_id,
            "channel": BASELINE_NOTIFICATION_CHANNEL,
            "digest_key": digest_key,
            "payload_schema_version": BASELINE_NOTIFICATION_PAYLOAD_SCHEMA_VERSION,
            "finding_count": len(feedback_ids),
            "finding_manifest": manifest,
            "finding_manifest_hash": manifest_hash,
            "state": state.value,
            "error_code": error_code,
            "error_fingerprint": error_fingerprint,
            "created_at": now,
            "updated_at": now,
            "suppressed_at": now if not enabled else None,
        },
    )
    existing = (
        session.execute(
            text(
                "SELECT outbox_id, group_id, digest_key, payload_schema_version, "
                "finding_count, finding_manifest_hash FROM baseline_notification_outbox "
                "WHERE run_id = :run_id AND recipient_user_id = :recipient_user_id "
                "AND channel = :channel"
            ),
            {
                "run_id": run_id,
                "recipient_user_id": recipient_user_id,
                "channel": BASELINE_NOTIFICATION_CHANNEL,
            },
        )
        .mappings()
        .one()
    )
    expected = (
        group_id,
        digest_key,
        BASELINE_NOTIFICATION_PAYLOAD_SCHEMA_VERSION,
        len(feedback_ids),
        manifest_hash,
    )
    actual = (
        str(existing["group_id"]),
        str(existing["digest_key"]),
        str(existing["payload_schema_version"]),
        int(existing["finding_count"]),
        str(existing["finding_manifest_hash"]),
    )
    if actual != expected:
        raise BaselineNotificationOutboxError(
            "baseline_notification_idempotency_conflict",
            "existing baseline notification digest does not match this intent",
        )
    return str(existing["outbox_id"])


class BaselineNotificationOutboxDispatcher:
    """Lease and dispatch one internal in-app digest through an injected sink."""

    def __init__(
        self,
        session_factory,
        *,
        enabled: bool | None = None,
        lease_seconds: int = DEFAULT_OUTBOX_LEASE_SECONDS,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self._session_factory = session_factory
        self._enabled = baseline_notifications_enabled() if enabled is None else enabled
        self._lease_seconds = lease_seconds
        self._clock = clock

    def dispatch_one(
        self,
        sink: BaselineNotificationSink,
    ) -> BaselineNotificationDispatchReceipt | None:
        if sink.channel != BASELINE_NOTIFICATION_CHANNEL:
            raise BaselineNotificationOutboxError(
                "baseline_notification_channel_unsupported",
                "Phase 2B2J supports only the internal in-app channel",
            )
        claim = self._claim_one()
        if claim is None or isinstance(claim, BaselineNotificationDispatchReceipt):
            return claim
        lease_token, digest, attempt_count = claim
        validation = self._reauthorize_before_delivery(
            digest, lease_token=lease_token, attempt_count=attempt_count
        )
        if validation is not None:
            return validation
        try:
            sink.deliver(digest, idempotency_key=digest.digest_key)
        except BaselineNotificationSinkError as exc:
            return self._record_failure(
                digest,
                lease_token=lease_token,
                attempt_count=attempt_count,
                code=exc.code,
                retryable=exc.retryable,
                error_class="sink",
            )
        # An injected channel can raise arbitrary client exceptions. Persist only
        # the stable class-level failure code, never its potentially private text.
        except Exception:  # noqa: BLE001
            return self._record_failure(
                digest,
                lease_token=lease_token,
                attempt_count=attempt_count,
                code="baseline_notification_sink_unavailable",
                retryable=True,
                error_class="sink",
            )
        return self._mark_delivered(
            digest, lease_token=lease_token, attempt_count=attempt_count
        )

    def _claim_one(
        self,
    ) -> (
        tuple[str, BaselineNotificationDigest, int]
        | BaselineNotificationDispatchReceipt
        | None
    ):
        with self._session_factory() as session:
            try:
                _begin_write(session)
                now = self._clock()
                sql = (
                    "SELECT o.*, r.source_scope_version, r.source_scope, "
                    "r.source_chunk_id, r.source_document_id, "
                    "r.generation_state FROM baseline_notification_outbox o "
                    "JOIN baseline_retrieval_run r ON r.run_id = o.run_id "
                    "AND r.group_id = o.group_id "
                    "WHERE (o.state IN ('pending', 'retryable_failed') "
                    "OR (o.state = 'running' AND o.lease_expires_at <= :now)) "
                    "ORDER BY o.created_at, o.outbox_id LIMIT 1"
                )
                if session.get_bind().dialect.name == "postgresql":
                    sql += " FOR UPDATE OF o SKIP LOCKED"
                row = session.execute(text(sql), {"now": now}).mappings().one_or_none()
                if row is None:
                    session.commit()
                    return None
                row = dict(row)
                if not self._enabled:
                    receipt = self._finish_without_delivery(
                        session,
                        row,
                        state=BaselineNotificationState.SUPPRESSED,
                        code="baseline_notifications_disabled",
                    )
                    session.commit()
                    return receipt
                validation = self._validate_row(session, row)
                if validation is not None:
                    state, code = validation
                    receipt = self._finish_without_delivery(
                        session, row, state=state, code=code
                    )
                    session.commit()
                    return receipt
                digest = self._digest_from_row(row)
                lease_token = uuid4().hex
                attempt_count = int(row["attempt_count"]) + 1
                session.execute(
                    text(
                        "UPDATE baseline_notification_outbox SET state = 'running', "
                        "lease_token = :lease_token, lease_expires_at = :expires, "
                        "attempt_count = :attempt_count, error_code = NULL, "
                        "error_fingerprint = NULL, updated_at = :updated "
                        "WHERE outbox_id = :outbox_id"
                    ),
                    {
                        "lease_token": lease_token,
                        "expires": now + timedelta(seconds=self._lease_seconds),
                        "attempt_count": attempt_count,
                        "updated": now,
                        "outbox_id": digest.outbox_id,
                    },
                )
                session.commit()
                return lease_token, digest, attempt_count
            except Exception:
                session.rollback()
                raise

    def _reauthorize_before_delivery(
        self,
        digest: BaselineNotificationDigest,
        *,
        lease_token: str,
        attempt_count: int,
    ) -> BaselineNotificationDispatchReceipt | None:
        with self._session_factory() as session:
            try:
                _begin_write(session)
                row = self._load_leased_row(
                    session, digest.outbox_id, lease_token=lease_token, lock=True
                )
                validation = self._validate_row(session, row)
                if validation is not None:
                    state, code = validation
                    receipt = self._finish_without_delivery(
                        session, row, state=state, code=code
                    )
                    session.commit()
                    return receipt
                expiry = _as_utc(row["lease_expires_at"])
                if expiry is None or expiry <= self._clock():
                    receipt = self._finish_without_delivery(
                        session,
                        row,
                        state=BaselineNotificationState.RETRYABLE_FAILED,
                        code="baseline_notification_lease_expired",
                    )
                    session.commit()
                    return receipt
                session.commit()
                return None
            except Exception:
                session.rollback()
                raise

    def _mark_delivered(
        self,
        digest: BaselineNotificationDigest,
        *,
        lease_token: str,
        attempt_count: int,
    ) -> BaselineNotificationDispatchReceipt:
        with self._session_factory() as session:
            try:
                _begin_write(session)
                row = self._load_leased_row(
                    session, digest.outbox_id, lease_token=lease_token, lock=True
                )
                now = self._clock()
                session.execute(
                    text(
                        "UPDATE baseline_notification_outbox SET state = 'delivered', "
                        "lease_token = NULL, lease_expires_at = NULL, delivered_at = :now, "
                        "updated_at = :now, error_code = NULL, error_fingerprint = NULL "
                        "WHERE outbox_id = :outbox_id AND lease_token = :lease_token"
                    ),
                    {
                        "now": now,
                        "outbox_id": digest.outbox_id,
                        "lease_token": lease_token,
                    },
                )
                session.commit()
                return BaselineNotificationDispatchReceipt(
                    digest.outbox_id,
                    BaselineNotificationState.DELIVERED,
                    int(row["attempt_count"]),
                )
            except Exception:
                session.rollback()
                raise

    def _record_failure(
        self,
        digest: BaselineNotificationDigest,
        *,
        lease_token: str,
        attempt_count: int,
        code: str,
        retryable: bool,
        error_class: str,
    ) -> BaselineNotificationDispatchReceipt:
        safe_code = _safe_error_code(code, "baseline_notification_failed")
        state = (
            BaselineNotificationState.RETRYABLE_FAILED
            if retryable
            else BaselineNotificationState.TERMINAL_FAILED
        )
        with self._session_factory() as session:
            try:
                _begin_write(session)
                row = self._load_leased_row(
                    session, digest.outbox_id, lease_token=lease_token, lock=True
                )
                session.execute(
                    text(
                        "UPDATE baseline_notification_outbox SET state = :state, "
                        "lease_token = NULL, lease_expires_at = NULL, error_code = :code, "
                        "error_fingerprint = :fingerprint, updated_at = :updated "
                        "WHERE outbox_id = :outbox_id AND lease_token = :lease_token"
                    ),
                    {
                        "state": state.value,
                        "code": safe_code,
                        "fingerprint": _sha256(f"{error_class}:{safe_code}"),
                        "updated": self._clock(),
                        "outbox_id": digest.outbox_id,
                        "lease_token": lease_token,
                    },
                )
                session.commit()
                return BaselineNotificationDispatchReceipt(
                    digest.outbox_id, state, int(row["attempt_count"]), safe_code
                )
            except Exception:
                session.rollback()
                raise

    def _load_leased_row(
        self,
        session: Session,
        outbox_id: str,
        *,
        lease_token: str,
        lock: bool,
    ) -> dict[str, object]:
        sql = (
            "SELECT o.*, r.source_scope_version, r.source_scope, "
            "r.source_chunk_id, r.source_document_id, "
            "r.generation_state FROM baseline_notification_outbox o "
            "JOIN baseline_retrieval_run r ON r.run_id = o.run_id "
            "AND r.group_id = o.group_id "
            "WHERE o.outbox_id = :outbox_id AND o.state = 'running' "
            "AND o.lease_token = :lease_token"
        )
        if lock and session.get_bind().dialect.name == "postgresql":
            sql += " FOR UPDATE"
        row = (
            session.execute(
                text(sql), {"outbox_id": outbox_id, "lease_token": lease_token}
            )
            .mappings()
            .one_or_none()
        )
        if row is None:
            raise BaselineNotificationOutboxError(
                "baseline_notification_lease_lost",
                "baseline notification lease is no longer active",
            )
        return dict(row)

    def _validate_row(
        self, session: Session, row: dict[str, object]
    ) -> tuple[BaselineNotificationState, str] | None:
        if row.get("generation_state") != "succeeded":
            return (
                BaselineNotificationState.TERMINAL_FAILED,
                "baseline_notification_run_not_succeeded",
            )
        recipient_user_id = row.get("recipient_user_id")
        source_scope_version = row.get("source_scope_version")
        source_scope = row.get("source_scope")
        source_chunk_id = row.get("source_chunk_id")
        source_document_id = row.get("source_document_id")
        if not recipient_user_id:
            return (
                BaselineNotificationState.CANCELLED,
                "baseline_notification_recipient_deleted",
            )
        if not source_document_id:
            return (
                BaselineNotificationState.CANCELLED,
                "baseline_notification_source_deleted",
            )
        if source_scope_version != SOURCE_SCOPE_VERSION or source_scope not in {
            SOURCE_SCOPE_LEGACY_CHUNK,
            SOURCE_SCOPE_CONTROL_DOCUMENT,
        }:
            return (
                BaselineNotificationState.TERMINAL_FAILED,
                "baseline_notification_source_scope_invalid",
            )
        if source_scope == SOURCE_SCOPE_LEGACY_CHUNK and not source_chunk_id:
            return (
                BaselineNotificationState.CANCELLED,
                "baseline_notification_source_deleted",
            )
        if source_scope == SOURCE_SCOPE_CONTROL_DOCUMENT and source_chunk_id:
            return (
                BaselineNotificationState.TERMINAL_FAILED,
                "baseline_notification_source_scope_invalid",
            )
        entity_count = session.execute(
            text(
                "SELECT "
                '(SELECT count(*) FROM "group" WHERE group_id = :group_id), '
                '(SELECT count(*) FROM "user" WHERE user_id = :recipient_user_id), '
                "(SELECT count(*) FROM document WHERE document_id = :source_document_id)"
            ),
            {
                "group_id": row["group_id"],
                "recipient_user_id": recipient_user_id,
                "source_document_id": source_document_id,
            },
        ).one()
        if any(int(value) != 1 for value in entity_count):
            return (
                BaselineNotificationState.CANCELLED,
                "baseline_notification_scope_deleted",
            )
        if source_scope == SOURCE_SCOPE_LEGACY_CHUNK:
            chunk_exists = session.execute(
                text(
                    "SELECT 1 FROM chunk WHERE chunk_id = :source_chunk_id "
                    "AND document_id = :source_document_id"
                ),
                {
                    "source_chunk_id": source_chunk_id,
                    "source_document_id": source_document_id,
                },
            ).scalar_one_or_none()
            if chunk_exists is None:
                return (
                    BaselineNotificationState.CANCELLED,
                    "baseline_notification_scope_deleted",
                )
        else:
            control_link = session.execute(
                text(
                    "SELECT 1 FROM baseline_control_run_job WHERE "
                    "persisted_run_id = :run_id AND group_id = :group_id "
                    "AND source_document_id = :source_document_id "
                    "AND state IN ('references_persisted', 'feedback_persisted')"
                ),
                {
                    "run_id": row["run_id"],
                    "group_id": row["group_id"],
                    "source_document_id": source_document_id,
                },
            ).scalar_one_or_none()
            if control_link is None:
                return (
                    BaselineNotificationState.CANCELLED,
                    "baseline_notification_scope_deleted",
                )
        authorized = session.execute(
            text(
                "SELECT 1 FROM user_to_group utg "
                "JOIN document_to_group dtg ON dtg.group_id = utg.group_id "
                "WHERE utg.user_id = :recipient_user_id "
                "AND utg.group_id = :group_id "
                "AND dtg.document_id = :source_document_id"
            ),
            {
                "recipient_user_id": recipient_user_id,
                "group_id": row["group_id"],
                "source_document_id": source_document_id,
            },
        ).scalar_one_or_none()
        if authorized is None:
            return (
                BaselineNotificationState.SUPPRESSED,
                "baseline_notification_authorization_revoked",
            )
        try:
            digest = self._digest_from_row(row)
        except BaselineNotificationOutboxError:
            return (
                BaselineNotificationState.TERMINAL_FAILED,
                "baseline_notification_manifest_invalid",
            )
        feedback_rows = session.execute(
            text(
                "SELECT feedback_id, baseline_finding_ordinal FROM feedback "
                "WHERE baseline_retrieval_run_id = :run_id "
                "ORDER BY baseline_finding_ordinal"
            ),
            {"run_id": row["run_id"]},
        ).all()
        actual = tuple(
            (int(feedback.baseline_finding_ordinal), str(feedback.feedback_id))
            for feedback in feedback_rows
        )
        expected = tuple(
            (finding.ordinal, finding.feedback_id) for finding in digest.findings
        )
        if actual != expected:
            return (
                BaselineNotificationState.TERMINAL_FAILED,
                "baseline_notification_manifest_stale",
            )
        return None

    def _digest_from_row(self, row: dict[str, object]) -> BaselineNotificationDigest:
        manifest_text = str(row["finding_manifest"])
        if _sha256(manifest_text) != row["finding_manifest_hash"]:
            raise BaselineNotificationOutboxError(
                "baseline_notification_manifest_invalid",
                "baseline notification manifest hash is invalid",
            )
        try:
            manifest = json.loads(manifest_text)
        except (TypeError, ValueError) as exc:
            raise BaselineNotificationOutboxError(
                "baseline_notification_manifest_invalid",
                "baseline notification manifest is invalid",
            ) from exc
        if not isinstance(manifest, dict) or set(manifest) != {
            "schema_version",
            "findings",
        }:
            raise BaselineNotificationOutboxError(
                "baseline_notification_manifest_invalid",
                "baseline notification manifest shape is invalid",
            )
        if manifest["schema_version"] != BASELINE_NOTIFICATION_PAYLOAD_SCHEMA_VERSION:
            raise BaselineNotificationOutboxError(
                "baseline_notification_manifest_invalid",
                "baseline notification manifest version is invalid",
            )
        values = manifest["findings"]
        if not isinstance(values, list) or len(values) != int(row["finding_count"]):
            raise BaselineNotificationOutboxError(
                "baseline_notification_manifest_invalid",
                "baseline notification finding count is invalid",
            )
        findings: list[BaselineNotificationFindingIdentifier] = []
        for expected_ordinal, value in enumerate(values, start=1):
            if (
                not isinstance(value, dict)
                or set(value) != {"feedback_id", "ordinal"}
                or value.get("ordinal") != expected_ordinal
                or not isinstance(value.get("feedback_id"), str)
                or not value["feedback_id"]
            ):
                raise BaselineNotificationOutboxError(
                    "baseline_notification_manifest_invalid",
                    "baseline notification finding identity is invalid",
                )
            findings.append(
                BaselineNotificationFindingIdentifier(
                    expected_ordinal, value["feedback_id"]
                )
            )
        return BaselineNotificationDigest(
            outbox_id=str(row["outbox_id"]),
            run_id=str(row["run_id"]),
            group_id=str(row["group_id"]),
            recipient_user_id=str(row["recipient_user_id"]),
            channel=str(row["channel"]),
            digest_key=str(row["digest_key"]),
            finding_count=int(row["finding_count"]),
            findings=tuple(findings),
            finding_manifest_hash=str(row["finding_manifest_hash"]),
        )

    def _finish_without_delivery(
        self,
        session: Session,
        row: dict[str, object],
        *,
        state: BaselineNotificationState,
        code: str,
    ) -> BaselineNotificationDispatchReceipt:
        if state not in {
            BaselineNotificationState.RETRYABLE_FAILED,
            BaselineNotificationState.TERMINAL_FAILED,
            BaselineNotificationState.SUPPRESSED,
            BaselineNotificationState.CANCELLED,
        }:
            raise ValueError(state)
        safe_code = _safe_error_code(code, "baseline_notification_not_delivered")
        now = self._clock()
        session.execute(
            text(
                "UPDATE baseline_notification_outbox SET state = :state, "
                "lease_token = NULL, lease_expires_at = NULL, error_code = :code, "
                "error_fingerprint = :fingerprint, updated_at = :updated, "
                "suppressed_at = :suppressed_at, cancelled_at = :cancelled_at "
                "WHERE outbox_id = :outbox_id"
            ),
            {
                "state": state.value,
                "code": safe_code,
                "fingerprint": _sha256(f"policy:{safe_code}"),
                "updated": now,
                "suppressed_at": (
                    now if state is BaselineNotificationState.SUPPRESSED else None
                ),
                "cancelled_at": (
                    now if state is BaselineNotificationState.CANCELLED else None
                ),
                "outbox_id": row["outbox_id"],
            },
        )
        return BaselineNotificationDispatchReceipt(
            str(row["outbox_id"]),
            state,
            int(row["attempt_count"]),
            safe_code,
        )


def load_authorized_baseline_notification_digest(
    session: Session,
    *,
    outbox_id: str,
    recipient_user_id: str,
    group_id: str,
) -> BaselineNotificationDigest:
    """Load ordered identifiers for a future authenticated in-app API consumer."""

    row = (
        session.execute(
            text(
                "SELECT o.*, r.source_scope_version, r.source_scope, "
                "r.source_chunk_id, r.source_document_id, "
                "r.generation_state FROM baseline_notification_outbox o "
                "JOIN baseline_retrieval_run r ON r.run_id = o.run_id "
                "AND r.group_id = o.group_id "
                "JOIN user_to_group utg ON utg.user_id = :recipient_user_id "
                "AND utg.group_id = o.group_id "
                "JOIN document_to_group dtg ON dtg.document_id = r.source_document_id "
                "AND dtg.group_id = o.group_id "
                "WHERE o.outbox_id = :outbox_id AND o.group_id = :group_id "
                "AND o.recipient_user_id = :recipient_user_id"
            ),
            {
                "outbox_id": outbox_id,
                "recipient_user_id": recipient_user_id,
                "group_id": group_id,
            },
        )
        .mappings()
        .one_or_none()
    )
    if row is None:
        raise BaselineNotificationOutboxError(
            "baseline_notification_unauthorized",
            "baseline notification digest is unavailable",
        )
    dispatcher = BaselineNotificationOutboxDispatcher(lambda: session, enabled=True)
    validation = dispatcher._validate_row(session, dict(row))
    if validation is not None:
        raise BaselineNotificationOutboxError(
            validation[1], "baseline notification digest is unavailable"
        )
    return dispatcher._digest_from_row(dict(row))


__all__ = [
    "BASELINE_NOTIFICATIONS_ENABLED_ENV",
    "BASELINE_NOTIFICATION_CHANNEL",
    "BASELINE_NOTIFICATION_OUTBOX_TABLE",
    "BASELINE_NOTIFICATION_PAYLOAD_SCHEMA_VERSION",
    "BaselineNotificationDigest",
    "BaselineNotificationDispatchReceipt",
    "BaselineNotificationFindingIdentifier",
    "BaselineNotificationOutboxDispatcher",
    "BaselineNotificationOutboxError",
    "BaselineNotificationSink",
    "BaselineNotificationSinkError",
    "BaselineNotificationState",
    "baseline_notifications_enabled",
    "load_authorized_baseline_notification_digest",
    "schedule_baseline_notification",
]
