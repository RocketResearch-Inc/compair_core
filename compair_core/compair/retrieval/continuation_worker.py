"""Internal sealed-snapshot continuation worker.

The worker receives only opaque scope and job identifiers. It reloads all
source material from the durable sealed staging tables, reauthorizes the
stored submitting user, and delegates publication to ``CorpusIngestionService``.
There is intentionally no HTTP or end-user dispatch surface in this module.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlalchemy import Engine, select, text, update
from sqlalchemy.orm import Session

from compair_core.baseline_control_plane_schema import (
    snapshot_content_part,
    snapshot_continuation_job,
    snapshot_staging,
)

from .control_document_scope import (
    ControlDocumentCorpusScopeError,
    choose_control_document_corpus_scope_key,
    control_document_corpus_identity,
)
from .control_plane import (
    DEFAULT_LEASE_LIFETIME,
    MAX_CONTENT_PART_REQUEST_BYTES,
    BaselineControlPlaneService,
    ControlPlaneError,
    LeaseReceipt,
    canonical_sha256,
)
from .corpus import (
    CorpusFileInput,
    CorpusFileSkipReason,
    CorpusFileState,
    RetrievalCorpus,
    RetrievalIndexState,
)
from .ingestion import (
    CorpusIngestionResult,
    CorpusIngestionService,
    CorpusRepositoryInput,
    CorpusSnapshotInput,
)

WORKER_CONTRACT_VERSION = "baseline-continuation-worker.v1"
WORKER_SERVICE_ID = "compair-core-corpus-ingestion"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]*$")


class ContinuationWorkerStage(str, Enum):
    CLAIMED = "claimed"
    RECONSTRUCTED = "reconstructed"
    BEFORE_INGESTION = "before_ingestion"
    BEFORE_SUCCESS = "before_success"


@dataclass(frozen=True, slots=True)
class InternalContinuationWorkerIdentity:
    """Process-local worker convention, never accepted from an API request."""

    instance_id: str
    service_id: str = WORKER_SERVICE_ID
    contract_version: str = WORKER_CONTRACT_VERSION

    @classmethod
    def create(
        cls, instance_id: str | None = None
    ) -> InternalContinuationWorkerIdentity:
        return cls(instance_id=instance_id or f"worker-{uuid4()}")

    def validate(self) -> None:
        if (
            self.service_id != WORKER_SERVICE_ID
            or self.contract_version != WORKER_CONTRACT_VERSION
            or not 1 <= len(self.instance_id) <= 128
            or _SAFE_ID.fullmatch(self.instance_id) is None
        ):
            raise ContinuationWorkerError("worker_identity_invalid", retryable=False)


@dataclass(frozen=True, slots=True)
class ContinuationIngestionOutcome:
    continuation_job_id: str
    group_id: str
    state: str
    attempt_count: int
    corpus_id: str
    generation_id: str
    generation_version: str
    manifest_hash: str
    provenance_fingerprint: str
    index_state: str = "incomplete"
    baseline_eligible: bool = False


class ContinuationWorkerError(RuntimeError):
    """A bounded worker failure that contains no source or provider detail."""

    def __init__(self, code: str, *, retryable: bool) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(code)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate")
        result[key] = value
    return result


def _nonfinite(_value: str) -> None:
    raise ValueError("nonfinite")


def continuation_result_provenance_fingerprint(
    *,
    continuation: Mapping[str, Any],
    result: CorpusIngestionResult,
) -> str:
    """Recompute the immutable safe ingestion-continuation provenance."""

    return canonical_sha256(
        {
            "worker_contract_version": WORKER_CONTRACT_VERSION,
            "continuation_job_id": str(continuation["continuation_job_id"]),
            "group_id": str(continuation["group_id"]),
            "snapshot_id": str(continuation["snapshot_id"]),
            "sealed_intent_hash": str(continuation["sealed_intent_hash"]),
            "canonical_manifest_hash": str(continuation["canonical_manifest_hash"]),
            "content_manifest_hash": str(continuation["content_manifest_hash"]),
            "repository_set_hash": str(continuation["repository_set_hash"]),
            "corpus_id": result.corpus_id,
            "generation_id": result.generation_id,
            "generation_version": result.generation_version,
            "ingestion_manifest_hash": result.manifest_hash,
        }
    )


_FILE_STATES = {item.value: item for item in CorpusFileState}
_SKIP_REASONS = {item.value: item for item in CorpusFileSkipReason}


class BaselineContinuationWorker:
    """Claim, reconstruct, and publish one durable continuation."""

    def __init__(
        self,
        engine: Engine,
        ingestion_service: CorpusIngestionService,
        *,
        clock: Callable[[], datetime] = _utcnow,
        stage_hook: Callable[[ContinuationWorkerStage], None] | None = None,
    ) -> None:
        self.engine = engine
        self.ingestion_service = ingestion_service
        self.clock = clock
        self.stage_hook = stage_hook
        self.control = BaselineControlPlaneService(engine, clock=clock)

    def _stage(self, stage: ContinuationWorkerStage) -> None:
        if self.stage_hook is not None:
            self.stage_hook(stage)

    @staticmethod
    def _validate_selector(value: str, label: str) -> str:
        if not 1 <= len(value) <= 128 or _SAFE_ID.fullmatch(value) is None:
            raise ContinuationWorkerError(f"{label}_invalid", retryable=False)
        return value

    def claim(
        self,
        *,
        identity: InternalContinuationWorkerIdentity,
        group_id: str,
        continuation_job_id: str,
        lifetime: timedelta = DEFAULT_LEASE_LIFETIME,
    ) -> LeaseReceipt:
        identity.validate()
        group_id = self._validate_selector(group_id, "group")
        continuation_job_id = self._validate_selector(
            continuation_job_id, "continuation"
        )
        if lifetime <= timedelta(0) or lifetime > timedelta(hours=1):
            raise ContinuationWorkerError("lease_lifetime_invalid", retryable=False)
        now = self.clock()
        token = secrets.token_urlsafe(32)
        with self.engine.begin() as connection:
            statement = select(snapshot_continuation_job).where(
                snapshot_continuation_job.c.group_id == group_id,
                snapshot_continuation_job.c.continuation_job_id == continuation_job_id,
            )
            if connection.dialect.name == "postgresql":
                statement = statement.with_for_update()
            row = connection.execute(statement).mappings().first()
            if row is None:
                raise ContinuationWorkerError("not_found_or_forbidden", retryable=False)
            submitter = row["created_by_user_id"]
            if submitter is None:
                raise ContinuationWorkerError("source_not_authorized", retryable=False)
            self._authorize_submitter(
                connection,
                user_id=str(submitter),
                group_id=group_id,
            )
            self.control._validate_continuation_claim(
                connection,
                continuation=row,
                caller_user_id=str(submitter),
            )
            expired = (
                row["state"] == "running"
                and row["lease_expires_at"] is not None
                and _aware(row["lease_expires_at"]) <= now
            )
            if row["state"] not in {"queued", "retryable_failed"} and not expired:
                raise ContinuationWorkerError("job_lease_unavailable", retryable=True)
            attempt = int(row["attempt_count"]) + 1
            expires = now + lifetime
            claimed = connection.execute(
                update(snapshot_continuation_job)
                .where(
                    snapshot_continuation_job.c.group_id == group_id,
                    snapshot_continuation_job.c.continuation_job_id
                    == continuation_job_id,
                    (
                        snapshot_continuation_job.c.state.in_(
                            {"queued", "retryable_failed"}
                        )
                        | (
                            (snapshot_continuation_job.c.state == "running")
                            & (snapshot_continuation_job.c.lease_expires_at <= now)
                        )
                    ),
                )
                .values(
                    state="running",
                    attempt_count=attempt,
                    lease_token=token,
                    lease_expires_at=expires,
                    error_code=None,
                    error_fingerprint=None,
                    updated_at=now,
                    finished_at=None,
                )
            )
            if claimed.rowcount != 1:
                raise ContinuationWorkerError("job_lease_unavailable", retryable=True)
        receipt = LeaseReceipt(continuation_job_id, token, expires, attempt)
        self._stage(ContinuationWorkerStage.CLAIMED)
        return receipt

    @staticmethod
    def _authorize_submitter(
        connection: Any,
        *,
        user_id: str,
        group_id: str,
    ) -> None:
        active = connection.execute(
            text('SELECT status FROM "user" WHERE user_id = :user_id'),
            {"user_id": user_id},
        ).scalar_one_or_none()
        if active != "active":
            raise ControlPlaneError("not_found_or_forbidden", status_code=404)
        BaselineControlPlaneService._authorize_group(
            connection,
            user_id=user_id,
            group_id=group_id,
        )

    def execute(
        self,
        *,
        identity: InternalContinuationWorkerIdentity,
        group_id: str,
        continuation_job_id: str,
        lifetime: timedelta = DEFAULT_LEASE_LIFETIME,
    ) -> ContinuationIngestionOutcome:
        identity.validate()
        completed = self._completed_outcome(
            group_id=group_id,
            continuation_job_id=continuation_job_id,
        )
        if completed is not None:
            return completed
        try:
            receipt = self.claim(
                identity=identity,
                group_id=group_id,
                continuation_job_id=continuation_job_id,
                lifetime=lifetime,
            )
        except ControlPlaneError as exc:
            raise ContinuationWorkerError(
                exc.code,
                retryable=exc.retryable,
            ) from None
        try:
            snapshot = self._reconstruct(
                group_id=group_id,
                continuation_job_id=continuation_job_id,
                lease_token=receipt.lease_token,
            )
            self._stage(ContinuationWorkerStage.RECONSTRUCTED)
            self._stage(ContinuationWorkerStage.BEFORE_INGESTION)

            def publish(session: Session, result: CorpusIngestionResult) -> None:
                self._publish_success(
                    session,
                    identity=identity,
                    group_id=group_id,
                    continuation_job_id=continuation_job_id,
                    lease_token=receipt.lease_token,
                    result=result,
                )

            result = self.ingestion_service.ingest_resumable(
                snapshot,
                publication_callback=publish,
            )
            return self._outcome(
                group_id=group_id,
                continuation_job_id=continuation_job_id,
                result=result,
                attempt_count=receipt.attempt_count,
            )
        except ContinuationWorkerError:
            raise
        except ControlPlaneError as exc:
            retryable = exc.retryable or exc.code == "job_lease_unavailable"
            self._record_failure(
                group_id=group_id,
                continuation_job_id=continuation_job_id,
                lease_token=receipt.lease_token,
                code=exc.code,
                retryable=retryable,
            )
            raise ContinuationWorkerError(exc.code, retryable=retryable) from None
        except ControlDocumentCorpusScopeError as exc:
            self._record_failure(
                group_id=group_id,
                continuation_job_id=continuation_job_id,
                lease_token=receipt.lease_token,
                code=exc.code,
                retryable=False,
            )
            raise ContinuationWorkerError(exc.code, retryable=False) from None
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError):
            self._record_failure(
                group_id=group_id,
                continuation_job_id=continuation_job_id,
                lease_token=receipt.lease_token,
                code="sealed_snapshot_invalid",
                retryable=False,
            )
            raise ContinuationWorkerError(
                "sealed_snapshot_invalid", retryable=False
            ) from None
        except Exception:  # noqa: BLE001 - sanitize the worker execution boundary
            self._record_failure(
                group_id=group_id,
                continuation_job_id=continuation_job_id,
                lease_token=receipt.lease_token,
                code="corpus_ingestion_failed",
                retryable=True,
            )
            raise ContinuationWorkerError(
                "corpus_ingestion_failed", retryable=True
            ) from None

    def _load_leased(
        self,
        connection: Any,
        *,
        group_id: str,
        continuation_job_id: str,
        lease_token: str,
    ) -> Mapping[str, Any]:
        row = (
            connection.execute(
                select(snapshot_continuation_job).where(
                    snapshot_continuation_job.c.group_id == group_id,
                    snapshot_continuation_job.c.continuation_job_id
                    == continuation_job_id,
                    snapshot_continuation_job.c.state == "running",
                    snapshot_continuation_job.c.lease_token == lease_token,
                )
            )
            .mappings()
            .first()
        )
        if (
            row is None
            or row["lease_expires_at"] is None
            or _aware(row["lease_expires_at"]) <= self.clock()
        ):
            raise ContinuationWorkerError("job_lease_unavailable", retryable=True)
        submitter = row["created_by_user_id"]
        if submitter is None:
            raise ControlPlaneError("source_not_authorized", status_code=404)
        self._authorize_submitter(
            connection,
            user_id=str(submitter),
            group_id=group_id,
        )
        self.control._validate_continuation_claim(
            connection,
            continuation=row,
            caller_user_id=str(submitter),
        )
        return row

    def _reconstruct(
        self,
        *,
        group_id: str,
        continuation_job_id: str,
        lease_token: str,
    ) -> CorpusSnapshotInput:
        with self.engine.connect() as connection:
            continuation = self._load_leased(
                connection,
                group_id=group_id,
                continuation_job_id=continuation_job_id,
                lease_token=lease_token,
            )
            staging = (
                connection.execute(
                    select(snapshot_staging).where(
                        snapshot_staging.c.group_id == group_id,
                        snapshot_staging.c.staging_id == continuation["staging_id"],
                    )
                )
                .mappings()
                .one()
            )
            snapshot = self.control._snapshot_from_staging(staging)
            if (
                snapshot.canonical_manifest.decode("utf-8")
                != staging["canonical_manifest_json"]
            ):
                raise ValueError("manifest")
            part_rows = (
                connection.execute(
                    select(snapshot_content_part)
                    .where(
                        snapshot_content_part.c.staging_id == staging["staging_id"],
                        snapshot_content_part.c.group_id == group_id,
                    )
                    .order_by(snapshot_content_part.c.part_ordinal)
                )
                .mappings()
                .all()
            )
            changed = snapshot.manifest["changed_repository"]
            scope_identity = control_document_corpus_identity(
                group_id=group_id,
                changed_repository_registration_id=str(changed["repository_id"]),
                source_document_id=str(changed["source_document_id"]),
            )
            corpus_rows = connection.execute(
                select(
                    RetrievalCorpus.scope_key,
                    RetrievalCorpus.changed_repository_id,
                    RetrievalCorpus.source_document_id,
                ).where(
                    RetrievalCorpus.scope_key.in_(scope_identity.accepted_scope_keys)
                )
            ).all()
            scope_key = choose_control_document_corpus_scope_key(
                scope_identity,
                tuple(tuple(row) for row in corpus_rows),
            )

        if [int(row["part_ordinal"]) for row in part_rows] != list(
            range(1, len(snapshot.expected_parts) + 1)
        ):
            raise ValueError("part_order")
        content_by_file: dict[int, str] = {}
        part_descriptors: list[dict[str, object]] = []
        total_items = 0
        total_bytes = 0
        for row in part_rows:
            raw = str(row["canonical_content_items_json"]).encode("utf-8")
            if len(raw) > MAX_CONTENT_PART_REQUEST_BYTES:
                raise ValueError("part_limit")
            try:
                items = json.loads(
                    raw.decode("utf-8", errors="strict"),
                    object_pairs_hook=_strict_pairs,
                    parse_constant=_nonfinite,
                )
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise ValueError("part_json") from exc
            if not isinstance(items, list):
                raise TypeError("part_json")
            try:
                ordinal, part_hash, canonical_items, item_count, content_bytes = (
                    self.control._validated_part(
                        {
                            "snapshot_id": snapshot.snapshot_id,
                            "part_ordinal": int(row["part_ordinal"]),
                            "part_sha256": str(row["part_sha256"]),
                            "content_items": items,
                        },
                        snapshot=snapshot,
                    )
                )
            except ControlPlaneError as exc:
                raise ValueError("part_contract") from exc
            if (
                ordinal != int(row["part_ordinal"])
                or part_hash != row["part_sha256"]
                or canonical_items != raw
                or item_count != int(row["item_count"])
                or content_bytes != int(row["content_bytes"])
            ):
                raise ValueError("part_drift")
            for item in items:
                file_ordinal = int(item["file_ordinal"])
                if file_ordinal in content_by_file:
                    raise ValueError("duplicate_content")
                content_by_file[file_ordinal] = str(item["content_utf8"])
            part_descriptors.append({"part_ordinal": ordinal, "part_sha256": part_hash})
            total_items += item_count
            total_bytes += content_bytes
        if (
            canonical_sha256(part_descriptors) != continuation["content_manifest_hash"]
            or total_items != continuation["expected_supported_file_count"]
            or total_bytes != continuation["expected_supported_content_bytes"]
        ):
            raise ValueError("content_manifest")
        return self._to_corpus_snapshot(
            group_id=group_id,
            scope_key=scope_key,
            continuation_job_id=continuation_job_id,
            continuation=continuation,
            snapshot=snapshot,
            content_by_file=content_by_file,
        )

    @staticmethod
    def _to_corpus_snapshot(
        *,
        group_id: str,
        scope_key: str,
        continuation_job_id: str,
        continuation: Mapping[str, Any],
        snapshot: Any,
        content_by_file: Mapping[int, str],
    ) -> CorpusSnapshotInput:
        manifest = snapshot.manifest
        changed = manifest["changed_repository"]
        siblings = tuple(
            CorpusRepositoryInput(
                repository_id=str(item["repository_id"]),
                repository_name=str(item["repository_name"]),
                expected_file_count=int(item["expected_file_count"]),
                repository_revision=str(item["repository_revision"]),
            )
            for item in manifest["sibling_repositories"]
        )
        files: list[CorpusFileInput] = []
        expected_supported: set[int] = set()
        for item in manifest["files"]:
            ordinal = int(item["ordinal"])
            state = _FILE_STATES.get(str(item["file_state"]))
            if state is None:
                raise ValueError("file_state")
            reason_value = item["skip_reason"]
            reason = None if reason_value is None else _SKIP_REASONS.get(reason_value)
            content = content_by_file.get(ordinal)
            if bool(item["content_required"]):
                expected_supported.add(ordinal)
                if content is None:
                    raise ValueError("missing_content")
            elif content is not None:
                raise ValueError("unexpected_content")
            content_hash = item["content_sha256"]
            if content_hash is None:
                raise ValueError("content_hash")
            files.append(
                CorpusFileInput(
                    repository_id=str(item["repository_id"]),
                    repository_name=str(item["repository_name"]),
                    relative_path=str(item["relative_path"]),
                    file_state=state,
                    content_hash=str(content_hash),
                    byte_size=int(item["byte_size"]),
                    content=content,
                    source_snapshot_id=str(item["repository_revision"]),
                    repository_revision=str(item["repository_revision"]),
                    skip_reason=reason,
                    derived_from_symlink=False,
                )
            )
        if set(content_by_file) != expected_supported:
            raise ValueError("content_coverage")
        return CorpusSnapshotInput.create(
            scope_key=scope_key,
            generation_version=str(snapshot.snapshot_id),
            changed_repository=CorpusRepositoryInput(
                repository_id=str(changed["repository_id"]),
                repository_name=str(changed["repository_name"]),
                expected_file_count=0,
                repository_revision=str(changed["head_revision"]),
                document_id=str(changed["source_document_id"]),
            ),
            sibling_repositories=siblings,
            files=tuple(files),
            producer_id=WORKER_SERVICE_ID,
            producer_version=WORKER_CONTRACT_VERSION,
            snapshot_id=str(snapshot.snapshot_id),
            source_revision=str(changed["head_revision"]),
            source_manifest_hash=str(continuation["canonical_manifest_hash"]),
        )

    def _publish_success(
        self,
        session: Session,
        *,
        identity: InternalContinuationWorkerIdentity,
        group_id: str,
        continuation_job_id: str,
        lease_token: str,
        result: CorpusIngestionResult,
    ) -> None:
        identity.validate()
        statement = select(snapshot_continuation_job).where(
            snapshot_continuation_job.c.group_id == group_id,
            snapshot_continuation_job.c.continuation_job_id == continuation_job_id,
        )
        if session.get_bind().dialect.name == "postgresql":
            statement = statement.with_for_update()
        continuation = session.execute(statement).mappings().first()
        if (
            continuation is None
            or continuation["state"] != "running"
            or continuation["lease_token"] != lease_token
            or continuation["lease_expires_at"] is None
            or _aware(continuation["lease_expires_at"]) <= self.clock()
        ):
            raise ControlPlaneError(
                "job_lease_unavailable", status_code=409, retryable=True
            )
        submitter = continuation["created_by_user_id"]
        if submitter is None:
            raise ControlPlaneError("source_not_authorized", status_code=404)
        self._authorize_submitter(
            session,
            user_id=str(submitter),
            group_id=group_id,
        )
        self.control._validate_continuation_claim(
            session,
            continuation=continuation,
            caller_user_id=str(submitter),
        )
        index_state = session.get(RetrievalIndexState, result.generation_id)
        if index_state is None or index_state.status != "incomplete":
            raise ControlPlaneError("index_state_incompatible", status_code=409)
        provenance = self._provenance_fingerprint(
            continuation=continuation,
            result=result,
        )
        self._stage(ContinuationWorkerStage.BEFORE_SUCCESS)
        now = self.clock()
        updated = session.execute(
            update(snapshot_continuation_job)
            .where(
                snapshot_continuation_job.c.group_id == group_id,
                snapshot_continuation_job.c.continuation_job_id == continuation_job_id,
                snapshot_continuation_job.c.state == "running",
                snapshot_continuation_job.c.lease_token == lease_token,
                snapshot_continuation_job.c.lease_expires_at > now,
            )
            .values(
                state="succeeded",
                lease_token=None,
                lease_expires_at=None,
                error_code=None,
                error_fingerprint=None,
                result_corpus_id=result.corpus_id,
                result_generation_id=result.generation_id,
                result_generation_version=result.generation_version,
                result_manifest_hash=result.manifest_hash,
                result_provenance_fingerprint=provenance,
                result_worker_contract_version=WORKER_CONTRACT_VERSION,
                result_published_at=now,
                updated_at=now,
                finished_at=now,
            )
        )
        if updated.rowcount != 1:
            raise ControlPlaneError(
                "job_lease_unavailable", status_code=409, retryable=True
            )

    @staticmethod
    def _provenance_fingerprint(
        *,
        continuation: Mapping[str, Any],
        result: CorpusIngestionResult,
    ) -> str:
        return continuation_result_provenance_fingerprint(
            continuation=continuation,
            result=result,
        )

    def _record_failure(
        self,
        *,
        group_id: str,
        continuation_job_id: str,
        lease_token: str,
        code: str,
        retryable: bool,
    ) -> bool:
        safe_code = (
            code if _SAFE_ID.fullmatch(code) and len(code) <= 128 else "worker_failed"
        )
        now = self.clock()
        with self.engine.begin() as connection:
            updated = connection.execute(
                update(snapshot_continuation_job)
                .where(
                    snapshot_continuation_job.c.group_id == group_id,
                    snapshot_continuation_job.c.continuation_job_id
                    == continuation_job_id,
                    snapshot_continuation_job.c.state == "running",
                    snapshot_continuation_job.c.lease_token == lease_token,
                    snapshot_continuation_job.c.lease_expires_at > now,
                )
                .values(
                    state="retryable_failed" if retryable else "terminal_failed",
                    lease_token=None,
                    lease_expires_at=None,
                    error_code=safe_code,
                    error_fingerprint=hashlib.sha256(
                        safe_code.encode("utf-8")
                    ).hexdigest(),
                    updated_at=now,
                    finished_at=None if retryable else now,
                )
            )
            return updated.rowcount == 1

    def cancel(
        self,
        *,
        identity: InternalContinuationWorkerIdentity,
        group_id: str,
        continuation_job_id: str,
        lease_token: str,
    ) -> None:
        identity.validate()
        now = self.clock()
        with self.engine.begin() as connection:
            updated = connection.execute(
                update(snapshot_continuation_job)
                .where(
                    snapshot_continuation_job.c.group_id == group_id,
                    snapshot_continuation_job.c.continuation_job_id
                    == continuation_job_id,
                    snapshot_continuation_job.c.state == "running",
                    snapshot_continuation_job.c.lease_token == lease_token,
                    snapshot_continuation_job.c.lease_expires_at > now,
                )
                .values(
                    state="cancelled",
                    lease_token=None,
                    lease_expires_at=None,
                    error_code="worker_cancelled",
                    error_fingerprint=hashlib.sha256(b"worker_cancelled").hexdigest(),
                    updated_at=now,
                    finished_at=now,
                )
            )
            if updated.rowcount != 1:
                raise ContinuationWorkerError("job_lease_unavailable", retryable=True)

    def _completed_outcome(
        self,
        *,
        group_id: str,
        continuation_job_id: str,
    ) -> ContinuationIngestionOutcome | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(snapshot_continuation_job).where(
                        snapshot_continuation_job.c.group_id == group_id,
                        snapshot_continuation_job.c.continuation_job_id
                        == continuation_job_id,
                    )
                )
                .mappings()
                .first()
            )
            if row is None or row["state"] != "succeeded":
                return None
            submitter = row["created_by_user_id"]
            if submitter is None:
                raise ContinuationWorkerError("source_not_authorized", retryable=False)
            try:
                self._authorize_submitter(
                    connection,
                    user_id=str(submitter),
                    group_id=group_id,
                )
                self.control._validate_continuation_claim(
                    connection,
                    continuation=row,
                    caller_user_id=str(submitter),
                )
            except ControlPlaneError as exc:
                raise ContinuationWorkerError(exc.code, retryable=False) from None
            return ContinuationIngestionOutcome(
                continuation_job_id=continuation_job_id,
                group_id=group_id,
                state="succeeded",
                attempt_count=int(row["attempt_count"]),
                corpus_id=str(row["result_corpus_id"]),
                generation_id=str(row["result_generation_id"]),
                generation_version=str(row["result_generation_version"]),
                manifest_hash=str(row["result_manifest_hash"]),
                provenance_fingerprint=str(row["result_provenance_fingerprint"]),
            )

    def _outcome(
        self,
        *,
        group_id: str,
        continuation_job_id: str,
        result: CorpusIngestionResult,
        attempt_count: int,
    ) -> ContinuationIngestionOutcome:
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(snapshot_continuation_job).where(
                        snapshot_continuation_job.c.continuation_job_id
                        == continuation_job_id
                    )
                )
                .mappings()
                .one()
            )
        return ContinuationIngestionOutcome(
            continuation_job_id=continuation_job_id,
            group_id=group_id,
            state="succeeded",
            attempt_count=attempt_count,
            corpus_id=result.corpus_id,
            generation_id=result.generation_id,
            generation_version=result.generation_version,
            manifest_hash=result.manifest_hash,
            provenance_fingerprint=str(row["result_provenance_fingerprint"]),
        )


__all__ = [
    "WORKER_CONTRACT_VERSION",
    "WORKER_SERVICE_ID",
    "BaselineContinuationWorker",
    "ContinuationIngestionOutcome",
    "ContinuationWorkerError",
    "ContinuationWorkerStage",
    "InternalContinuationWorkerIdentity",
    "continuation_result_provenance_fingerprint",
]
