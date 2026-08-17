"""Internal protected baseline-run submission state.

No API route, task dispatch, retrieval, evidence persistence, generation, or
notification behavior is defined here.  A later lease-owning worker may use
the low-level cipher after it has independently claimed and reauthorized a job.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any
from uuid import UUID, uuid4

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sqlalchemy import delete, select, text, update
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from compair_core.baseline_control_plane_schema import (
    baseline_run_job,
    baseline_run_payload,
    compatible_index_job,
    control_job,
    repository_approval,
    repository_registration,
)

from .control_plane import BaselineControlPlaneService, ControlPlaneError, canonicalize
from .control_plane_v2 import (
    PROTOCOL_V2_SHA256,
    PROTOCOL_V2_VERSION,
    V2IndexPublication,
    V2RunSubmission,
)
from .corpus import (
    BaselineIndexBuildStatus,
    CorpusGenerationStatus,
    CorpusIngestionStatus,
    IndexStateStatus,
    RetrievalBaselineIndexBuild,
    RetrievalBaselineIndexPublication,
    RetrievalCorpus,
    RetrievalCorpusGeneration,
    RetrievalCorpusIngestion,
    RetrievalIndexState,
)
from .indexing import BaselineIndexLifecycle
from .persistent import published_index_fingerprint

RUN_JOB_CONTRACT_VERSION = "baseline-run-job.v1"
RUN_PAYLOAD_SCHEMA_VERSION = "baseline-run-protected-payload.v1"
RUN_PAYLOAD_AAD_VERSION = "baseline-run-aad.v1"
RUN_KEYRING_VERSION = "baseline-run-keyring.v1"
RUN_ENCRYPTION_ALGORITHM = "AES-256-GCM"
RUN_NONCE_BYTES = 12
RUN_KEY_BYTES = 32
RUN_PARENT_SECRET_BYTES = 32
DEFAULT_PAYLOAD_LIFETIME = timedelta(minutes=15)
MIN_PAYLOAD_LIFETIME = timedelta(minutes=1)
MAX_PAYLOAD_LIFETIME = timedelta(hours=1)

_SAFE_KEY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]{0,127}$")
_TERMINAL_STATES = frozenset(
    {
        "feedback_persisted",
        "insufficient",
        "terminal_failed",
        "blocked",
        "cancelled",
    }
)
_FROZEN_SAFE_REASONS = frozenset(
    {
        "authorization_revoked",
        "capability_unavailable",
        "corpus_incompatible",
        "embedding_identity_mismatch",
        "embedding_unavailable",
        "generation_blocked",
        "generation_malformed",
        "generation_terminal_failure",
        "idempotency_conflict",
        "index_build_failed",
        "index_publication_stale",
        "index_vector_invalid",
        "internal_failure",
        "job_cancelled",
        "job_not_found_or_forbidden",
        "limit_exceeded",
        "protocol_mismatch",
        "repository_not_authorized",
        "retrieval_error",
        "retrieval_insufficient",
        "source_not_authorized",
        "transport_unavailable",
        "worker_unavailable",
    }
)


def _safe_status_reason(code: str, *, stage: str, state: str) -> str:
    if code in _FROZEN_SAFE_REASONS:
        return code
    if code in {"payload_expired", "run_payload_expired"}:
        return "worker_unavailable"
    if "authorization" in code or code in {
        "generation_source_deleted",
        "generation_source_unavailable",
    }:
        return "authorization_revoked"
    if stage == "generation":
        if "malformed" in code or "output" in code:
            return "generation_malformed"
        if "unavailable" in code or "database" in code:
            return (
                "worker_unavailable"
                if state == "retryable_failed"
                else "generation_terminal_failure"
            )
        return "generation_blocked"
    if "embedding" in code:
        return (
            "embedding_identity_mismatch"
            if any(marker in code for marker in ("identity", "dimension", "mismatch"))
            else "embedding_unavailable"
        )
    if code.startswith(("corpus_", "index_state_")):
        return "corpus_incompatible"
    if "publication" in code or "stale" in code:
        return "index_publication_stale"
    return "internal_failure"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class BaselineRunJobError(RuntimeError):
    """A stable error that never includes protected values or crypto details."""

    def __init__(
        self,
        code: str,
        *,
        status_code: int = 409,
        retryable: bool = False,
    ) -> None:
        self.code = code
        self.status_code = status_code
        self.retryable = retryable
        super().__init__(code)


class RunSubmissionStage(str, Enum):
    AFTER_JOB_INSERT = "after_job_insert"
    AFTER_PAYLOAD_ENCRYPTION = "after_payload_encryption"
    AFTER_PAYLOAD_INSERT = "after_payload_insert"


@dataclass(frozen=True, slots=True, repr=False)
class ProtectedRunPayload:
    retrieval_query: str
    parent_processing_secret: bytes

    def __repr__(self) -> str:
        return "ProtectedRunPayload(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class BaselineRunKeyring:
    """Validated external AES keyring with intentionally redacted repr."""

    active_key_id: str
    _keys: Mapping[str, bytes]

    def __repr__(self) -> str:
        return "BaselineRunKeyring(<redacted>)"

    @classmethod
    def from_json(cls, raw: str) -> BaselineRunKeyring:
        if not isinstance(raw, str) or not raw:
            raise BaselineRunJobError(
                "run_keyring_unavailable", status_code=503, retryable=False
            )

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise BaselineRunJobError(
                        "run_keyring_invalid", status_code=503, retryable=False
                    )
                result[key] = value
            return result

        try:
            value = json.loads(raw, object_pairs_hook=reject_duplicates)
        except BaselineRunJobError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError):
            raise BaselineRunJobError(
                "run_keyring_invalid", status_code=503, retryable=False
            ) from None
        if not isinstance(value, Mapping) or set(value) != {
            "version",
            "active_key_id",
            "keys",
        }:
            raise BaselineRunJobError(
                "run_keyring_invalid", status_code=503, retryable=False
            )
        active = value["active_key_id"]
        entries = value["keys"]
        if (
            value["version"] != RUN_KEYRING_VERSION
            or not isinstance(active, str)
            or _SAFE_KEY_ID.fullmatch(active) is None
            or not isinstance(entries, list)
            or not entries
        ):
            raise BaselineRunJobError(
                "run_keyring_invalid", status_code=503, retryable=False
            )
        keys: dict[str, bytes] = {}
        for entry in entries:
            if not isinstance(entry, Mapping) or set(entry) != {"key_id", "key_base64"}:
                raise BaselineRunJobError(
                    "run_keyring_invalid", status_code=503, retryable=False
                )
            key_id = entry["key_id"]
            encoded = entry["key_base64"]
            if (
                not isinstance(key_id, str)
                or _SAFE_KEY_ID.fullmatch(key_id) is None
                or key_id in keys
                or not isinstance(encoded, str)
            ):
                raise BaselineRunJobError(
                    "run_keyring_invalid", status_code=503, retryable=False
                )
            try:
                key = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError):
                raise BaselineRunJobError(
                    "run_keyring_invalid", status_code=503, retryable=False
                ) from None
            if len(key) != RUN_KEY_BYTES:
                raise BaselineRunJobError(
                    "run_keyring_invalid", status_code=503, retryable=False
                )
            keys[key_id] = key
        if active not in keys:
            raise BaselineRunJobError(
                "run_keyring_invalid", status_code=503, retryable=False
            )
        return cls(active_key_id=active, _keys=MappingProxyType(keys))

    def active_key(self) -> bytes:
        return self._keys[self.active_key_id]

    def decryption_key(self, key_id: str) -> bytes:
        key = self._keys.get(key_id)
        if key is None:
            raise BaselineRunJobError(
                "run_payload_key_unavailable", status_code=503, retryable=False
            )
        return key


def keyring_from_settings(settings: Any) -> BaselineRunKeyring:
    configured = getattr(settings, "baseline_run_encryption_keyring", None)
    if configured is None:
        raise BaselineRunJobError(
            "run_keyring_unavailable", status_code=503, retryable=False
        )
    raw = (
        configured.get_secret_value()
        if hasattr(configured, "get_secret_value")
        else configured
    )
    return BaselineRunKeyring.from_json(raw)


def _aad(job: Mapping[str, Any]) -> bytes:
    return canonicalize(
        {
            "aad_version": RUN_PAYLOAD_AAD_VERSION,
            "payload_schema_version": RUN_PAYLOAD_SCHEMA_VERSION,
            "job_id": str(job["job_id"]),
            "group_id": str(job["group_id"]),
            "submitted_by_user_id": str(job["submitted_by_user_id"]),
            "source_document_id": str(job["source_document_id"]),
            "changed_repository_registration_id": str(
                job["changed_repository_registration_id"]
            ),
            "corpus_generation_id": str(job["corpus_generation_id"]),
            "index_publication_id": str(job["index_publication_id"]),
            "protocol_version": str(job["protocol_version"]),
            "protocol_sha256": str(job["protocol_sha256"]),
            "query_sha256": str(job["query_sha256"]),
            "query_byte_length": int(job["query_byte_length"]),
            "query_origin": str(job["query_origin"]),
        }
    )


class BaselineRunPayloadCipher:
    """AES-256-GCM envelope primitive; persistence/authorization live elsewhere."""

    def __init__(
        self,
        keyring: BaselineRunKeyring,
        *,
        nonce_factory: Callable[[int], bytes] = secrets.token_bytes,
    ) -> None:
        self.keyring = keyring
        self.nonce_factory = nonce_factory

    def encrypt(
        self,
        *,
        job: Mapping[str, Any],
        retrieval_query: str,
        parent_processing_secret: bytes,
        created_at: datetime,
        expires_at: datetime,
    ) -> dict[str, object]:
        nonce = self.nonce_factory(RUN_NONCE_BYTES)
        if not isinstance(nonce, bytes) or len(nonce) != RUN_NONCE_BYTES:
            raise BaselineRunJobError("run_crypto_randomness_failed", status_code=503)
        if len(parent_processing_secret) != RUN_PARENT_SECRET_BYTES:
            raise BaselineRunJobError("run_crypto_randomness_failed", status_code=503)
        plaintext = canonicalize(
            {
                "schema_version": RUN_PAYLOAD_SCHEMA_VERSION,
                "retrieval_query": retrieval_query,
                "parent_processing_secret": base64.b64encode(
                    parent_processing_secret
                ).decode("ascii"),
            }
        )
        try:
            ciphertext = AESGCM(self.keyring.active_key()).encrypt(
                nonce, plaintext, _aad(job)
            )
        except Exception:  # noqa: BLE001 - cryptographic boundary is sanitized
            raise BaselineRunJobError(
                "run_payload_encryption_failed", status_code=503, retryable=True
            ) from None
        return {
            "job_id": job["job_id"],
            "group_id": job["group_id"],
            "payload_schema_version": RUN_PAYLOAD_SCHEMA_VERSION,
            "algorithm": RUN_ENCRYPTION_ALGORITHM,
            "key_id": self.keyring.active_key_id,
            "nonce": nonce,
            "ciphertext": ciphertext,
            "aad_version": RUN_PAYLOAD_AAD_VERSION,
            "created_at": created_at,
            "expires_at": expires_at,
        }

    def decrypt(
        self,
        *,
        job: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> ProtectedRunPayload:
        if (
            payload.get("payload_schema_version") != RUN_PAYLOAD_SCHEMA_VERSION
            or payload.get("algorithm") != RUN_ENCRYPTION_ALGORITHM
            or payload.get("aad_version") != RUN_PAYLOAD_AAD_VERSION
            or not isinstance(payload.get("nonce"), bytes)
            or len(payload["nonce"]) != RUN_NONCE_BYTES
            or not isinstance(payload.get("ciphertext"), bytes)
        ):
            raise BaselineRunJobError("run_payload_authentication_failed")
        key = self.keyring.decryption_key(str(payload.get("key_id", "")))
        try:
            raw = AESGCM(key).decrypt(
                payload["nonce"], payload["ciphertext"], _aad(job)
            )
        except (InvalidTag, ValueError, TypeError):
            raise BaselineRunJobError("run_payload_authentication_failed") from None
        try:
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, Mapping) or set(value) != {
                "schema_version",
                "retrieval_query",
                "parent_processing_secret",
            }:
                raise ValueError
            query = value["retrieval_query"]
            encoded_secret = value["parent_processing_secret"]
            if (
                value["schema_version"] != RUN_PAYLOAD_SCHEMA_VERSION
                or not isinstance(query, str)
                or not isinstance(encoded_secret, str)
            ):
                raise ValueError
            query_bytes = query.encode("utf-8")
            parent_secret = base64.b64decode(encoded_secret, validate=True)
        except (UnicodeError, ValueError, TypeError, json.JSONDecodeError):
            raise BaselineRunJobError("run_payload_authentication_failed") from None
        if (
            len(query_bytes) != int(job["query_byte_length"])
            or len(query) != int(job["query_length"])
            or hashlib.sha256(query_bytes).hexdigest() != job["query_sha256"]
            or len(parent_secret) != RUN_PARENT_SECRET_BYTES
            or hashlib.sha256(parent_secret).hexdigest()
            != job["parent_processing_identity_fingerprint"]
        ):
            raise BaselineRunJobError("run_payload_authentication_failed")
        return ProtectedRunPayload(query, parent_secret)


@dataclass(frozen=True, slots=True)
class _AuthorizedPublication:
    index_job_id: str
    corpus_id: str
    document_count: int


class BaselineRunJobService:
    """Transactional internal run submission/status/expiry service."""

    def __init__(
        self,
        engine: Engine,
        keyring: BaselineRunKeyring,
        *,
        payload_lifetime: timedelta = DEFAULT_PAYLOAD_LIFETIME,
        clock: Callable[[], datetime] = _utcnow,
        nonce_factory: Callable[[int], bytes] = secrets.token_bytes,
        secret_factory: Callable[[int], bytes] = secrets.token_bytes,
        stage_hook: Callable[[RunSubmissionStage], None] | None = None,
    ) -> None:
        if not MIN_PAYLOAD_LIFETIME <= payload_lifetime <= MAX_PAYLOAD_LIFETIME:
            raise BaselineRunJobError("run_payload_lifetime_invalid", status_code=503)
        self.engine = engine
        self.sessions = sessionmaker(engine, expire_on_commit=False)
        self.keyring = keyring
        self.cipher = BaselineRunPayloadCipher(keyring, nonce_factory=nonce_factory)
        self.payload_lifetime = payload_lifetime
        self.clock = clock
        self.secret_factory = secret_factory
        self.stage_hook = stage_hook
        self.control = BaselineControlPlaneService(engine, clock=clock)

    @classmethod
    def from_settings(cls, engine: Engine, settings: Any) -> BaselineRunJobService:
        return cls(
            engine,
            keyring_from_settings(settings),
            payload_lifetime=timedelta(
                seconds=int(settings.baseline_run_payload_ttl_seconds)
            ),
        )

    def _stage(self, stage: RunSubmissionStage) -> None:
        if self.stage_hook is not None:
            self.stage_hook(stage)

    @staticmethod
    def _intent_hash(submission: V2RunSubmission) -> str:
        publication = submission.index_publication
        query = submission.retrieval_query
        return hashlib.sha256(
            canonicalize(
                {
                    "protocol_version": PROTOCOL_V2_VERSION,
                    "protocol_sha256": PROTOCOL_V2_SHA256,
                    "group_id": submission.group_id,
                    "source_document_id": submission.source_document_id,
                    "changed_repository_registration_id": (
                        submission.changed_repository_registration_id
                    ),
                    "index_publication": {
                        "index_publication_id": publication.index_publication_id,
                        "corpus_generation_id": publication.corpus_generation_id,
                        "corpus_manifest_hash": publication.corpus_manifest_hash,
                        "index_format_version": publication.index_format_version,
                        "tokenizer_version": publication.tokenizer_version,
                        "retrieval_config_fingerprint": (
                            publication.retrieval_config_fingerprint
                        ),
                        "embedding_fingerprint": publication.embedding_fingerprint,
                        "index_fingerprint": publication.index_fingerprint,
                    },
                    "retrieval_query": {
                        "representation": query.representation,
                        "origin": query.origin,
                        "encoding": query.encoding,
                        "base_revision": query.base_revision,
                        "head_revision": query.head_revision,
                        "byte_size": query.byte_size,
                        "sha256": query.sha256,
                    },
                }
            )
        ).hexdigest()

    @staticmethod
    def _idempotency_hash(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    @staticmethod
    def _lock(statement: Any, session: Session) -> Any:
        return (
            statement.with_for_update()
            if session.get_bind().dialect.name == "postgresql"
            else statement
        )

    def _authorize_publication(
        self,
        session: Session,
        *,
        caller_user_id: str,
        group_id: str,
        source_document_id: str,
        changed_registration_id: str,
        publication: V2IndexPublication,
        lock: bool,
    ) -> _AuthorizedPublication:
        try:
            self.control._authorize_group(
                session, user_id=caller_user_id, group_id=group_id
            )
        except ControlPlaneError:
            raise BaselineRunJobError(
                "not_found_or_forbidden", status_code=404
            ) from None
        source = session.execute(
            text(
                "SELECT d.document_id FROM document d "
                "JOIN document_to_group dtg ON dtg.document_id = d.document_id "
                "WHERE d.document_id = :document_id AND dtg.group_id = :group_id"
                + (
                    " FOR UPDATE OF d, dtg"
                    if lock and session.get_bind().dialect.name == "postgresql"
                    else ""
                )
            ),
            {"document_id": source_document_id, "group_id": group_id},
        ).first()
        if source is None:
            raise BaselineRunJobError("source_not_authorized", status_code=404)
        approval_statement = (
            select(repository_registration, repository_approval)
            .join(
                repository_approval,
                (
                    repository_approval.c.registration_id
                    == repository_registration.c.registration_id
                )
                & (
                    repository_approval.c.group_id == repository_registration.c.group_id
                ),
            )
            .where(
                repository_registration.c.group_id == group_id,
                repository_registration.c.registration_id == changed_registration_id,
                repository_registration.c.enabled.is_(True),
                repository_registration.c.source_document_id == source_document_id,
                repository_approval.c.state == "active",
            )
        )
        if lock:
            approval_statement = self._lock(approval_statement, session)
        if session.execute(approval_statement).mappings().first() is None:
            raise BaselineRunJobError("repository_not_authorized", status_code=404)

        extension_statement = select(compatible_index_job).where(
            compatible_index_job.c.group_id == group_id,
            compatible_index_job.c.result_index_id == publication.index_publication_id,
            compatible_index_job.c.generation_id == publication.corpus_generation_id,
        )
        if lock:
            extension_statement = self._lock(extension_statement, session)
        extension = session.execute(extension_statement).mappings().first()
        if extension is None:
            raise BaselineRunJobError("index_publication_stale")
        job_statement = select(control_job).where(
            control_job.c.job_id == extension["job_id"],
            control_job.c.group_id == group_id,
            control_job.c.operation == "index_build",
            control_job.c.state == "succeeded",
        )
        if lock:
            job_statement = self._lock(job_statement, session)
        if session.execute(job_statement).mappings().first() is None:
            raise BaselineRunJobError("index_publication_stale")

        corpus_statement = select(RetrievalCorpus).where(
            RetrievalCorpus.corpus_id == extension["corpus_id"]
        )
        generation_statement = select(RetrievalCorpusGeneration).where(
            RetrievalCorpusGeneration.generation_id == publication.corpus_generation_id
        )
        if lock:
            corpus_statement = self._lock(corpus_statement, session)
            generation_statement = self._lock(generation_statement, session)
        corpus = session.scalar(corpus_statement)
        generation = session.scalar(generation_statement)
        ingestion = session.get(
            RetrievalCorpusIngestion, publication.corpus_generation_id
        )
        current = session.get(
            RetrievalBaselineIndexPublication,
            str(extension["corpus_id"]),
        )
        build = session.get(
            RetrievalBaselineIndexBuild, publication.index_publication_id
        )
        index_state = session.get(RetrievalIndexState, publication.corpus_generation_id)
        if (
            corpus is None
            or generation is None
            or ingestion is None
            or current is None
            or build is None
            or index_state is None
            or corpus.scope_key != f"group:{group_id}"
            or corpus.source_document_id != source_document_id
            or corpus.changed_repository_id != changed_registration_id
            or corpus.active_generation_id != publication.corpus_generation_id
            or generation.corpus_id != corpus.corpus_id
            or generation.status != CorpusGenerationStatus.ACTIVE.value
            or generation.manifest_hash is None
            or ingestion.status != CorpusIngestionStatus.ACTIVE.value
            or current.index_id != publication.index_publication_id
            or build.generation_id != publication.corpus_generation_id
            or build.status != BaselineIndexBuildStatus.COMPATIBLE.value
            or index_state.status != IndexStateStatus.COMPATIBLE.value
            or extension["corpus_manifest_hash"] != publication.corpus_manifest_hash
            or build.index_schema_version != publication.index_format_version
            or build.tokenizer_version != publication.tokenizer_version
            or build.engine_config_fingerprint
            != publication.retrieval_config_fingerprint
            or build.embedding_fingerprint != publication.embedding_fingerprint
        ):
            raise BaselineRunJobError("index_publication_stale")
        try:
            fingerprint = published_index_fingerprint(build)
        except ValueError:
            raise BaselineRunJobError("index_publication_stale") from None
        if (
            fingerprint != publication.index_fingerprint
            or BaselineIndexLifecycle.validation_error(session, build.index_id)
            is not None
        ):
            raise BaselineRunJobError("index_publication_stale")
        return _AuthorizedPublication(
            index_job_id=str(extension["job_id"]),
            corpus_id=corpus.corpus_id,
            document_count=build.indexed_document_count,
        )

    @staticmethod
    def _accepted(
        submission: V2RunSubmission,
        row: Mapping[str, Any],
        *,
        replayed: bool,
    ) -> dict[str, object]:
        return {
            "protocol_version": PROTOCOL_V2_VERSION,
            "protocol_sha256": PROTOCOL_V2_SHA256,
            "message_type": "job_accepted",
            "request_id": submission.request_id,
            "group_id": submission.group_id,
            "job_id": str(row["job_id"]),
            "operation": "baseline_run",
            "state": "queued",
            "replayed": replayed,
            "processing_run_id": str(row["processing_run_id"]),
        }

    def find_replay(
        self,
        submission: V2RunSubmission,
        *,
        caller_user_id: str,
    ) -> dict[str, object] | None:
        """Return an authorized exact replay without mutating payload lifetime.

        This read-only check lets an already accepted intent remain replayable when
        automatic admission is temporarily backpressured.  It deliberately does
        not refresh timestamps, payload expiry, or job state.
        """

        intent_hash = self._intent_hash(submission)
        key_hash = self._idempotency_hash(submission.idempotency_key)
        with self.sessions() as session:
            existing = (
                session.execute(
                    select(baseline_run_job).where(
                        baseline_run_job.c.group_id == submission.group_id,
                        baseline_run_job.c.idempotency_key_hash == key_hash,
                    )
                )
                .mappings()
                .first()
            )
            if existing is None:
                return None
            self._authorize_publication(
                session,
                caller_user_id=caller_user_id,
                group_id=submission.group_id,
                source_document_id=submission.source_document_id,
                changed_registration_id=(
                    submission.changed_repository_registration_id
                ),
                publication=submission.index_publication,
                lock=False,
            )
            if (
                existing["intent_hash"] != intent_hash
                or existing["submitted_by_user_id"] != caller_user_id
            ):
                raise BaselineRunJobError("idempotency_conflict")
            return self._accepted(submission, existing, replayed=True)

    def submit(
        self,
        submission: V2RunSubmission,
        *,
        caller_user_id: str,
    ) -> dict[str, object]:
        intent_hash = self._intent_hash(submission)
        key_hash = self._idempotency_hash(submission.idempotency_key)
        try:
            with self.sessions.begin() as session:
                authorized = self._authorize_publication(
                    session,
                    caller_user_id=caller_user_id,
                    group_id=submission.group_id,
                    source_document_id=submission.source_document_id,
                    changed_registration_id=(
                        submission.changed_repository_registration_id
                    ),
                    publication=submission.index_publication,
                    lock=True,
                )
                existing_statement = select(baseline_run_job).where(
                    baseline_run_job.c.group_id == submission.group_id,
                    baseline_run_job.c.idempotency_key_hash == key_hash,
                )
                existing_statement = self._lock(existing_statement, session)
                existing = session.execute(existing_statement).mappings().first()
                if existing is not None:
                    if (
                        existing["intent_hash"] != intent_hash
                        or existing["submitted_by_user_id"] != caller_user_id
                    ):
                        raise BaselineRunJobError("idempotency_conflict")
                    return self._accepted(submission, existing, replayed=True)

                now = self.clock()
                expires_at = now + self.payload_lifetime
                job_id = str(uuid4())
                processing_run_id = str(uuid4())
                parent_secret = self.secret_factory(RUN_PARENT_SECRET_BYTES)
                if (
                    not isinstance(parent_secret, bytes)
                    or len(parent_secret) != RUN_PARENT_SECRET_BYTES
                ):
                    raise BaselineRunJobError(
                        "run_crypto_randomness_failed", status_code=503
                    )
                query = submission.retrieval_query
                publication = submission.index_publication
                values: dict[str, object] = {
                    "job_id": job_id,
                    "group_id": submission.group_id,
                    "submitted_by_user_id": caller_user_id,
                    "source_document_id": submission.source_document_id,
                    "changed_repository_registration_id": (
                        submission.changed_repository_registration_id
                    ),
                    "index_job_id": authorized.index_job_id,
                    "corpus_id": authorized.corpus_id,
                    "corpus_generation_id": publication.corpus_generation_id,
                    "index_publication_id": publication.index_publication_id,
                    "corpus_manifest_hash": publication.corpus_manifest_hash,
                    "index_format_version": publication.index_format_version,
                    "tokenizer_version": publication.tokenizer_version,
                    "retrieval_config_fingerprint": (
                        publication.retrieval_config_fingerprint
                    ),
                    "embedding_fingerprint": publication.embedding_fingerprint,
                    "index_fingerprint": publication.index_fingerprint,
                    "contract_version": RUN_JOB_CONTRACT_VERSION,
                    "protocol_version": PROTOCOL_V2_VERSION,
                    "protocol_sha256": PROTOCOL_V2_SHA256,
                    "request_id": submission.request_id,
                    "idempotency_key_hash": key_hash,
                    "intent_hash": intent_hash,
                    "processing_run_id": processing_run_id,
                    "parent_processing_identity_fingerprint": hashlib.sha256(
                        parent_secret
                    ).hexdigest(),
                    "query_representation": query.representation,
                    "query_encoding": query.encoding,
                    "query_base_revision": query.base_revision,
                    "query_head_revision": query.head_revision,
                    "query_sha256": query.sha256,
                    "query_length": len(query.text),
                    "query_byte_length": query.byte_size,
                    "query_origin": query.origin,
                    "state": "queued",
                    "attempt_count": 0,
                    "payload_expires_at": expires_at,
                    "evidence_count": 0,
                    "reference_count": 0,
                    "feedback_count": 0,
                    "generation_invoked": False,
                    "notification_outbox_count": 0,
                    "created_at": now,
                    "updated_at": now,
                }
                session.execute(baseline_run_job.insert().values(**values))
                self._stage(RunSubmissionStage.AFTER_JOB_INSERT)
                encrypted = self.cipher.encrypt(
                    job=values,
                    retrieval_query=query.text,
                    parent_processing_secret=parent_secret,
                    created_at=now,
                    expires_at=expires_at,
                )
                self._stage(RunSubmissionStage.AFTER_PAYLOAD_ENCRYPTION)
                session.execute(baseline_run_payload.insert().values(**encrypted))
                self._stage(RunSubmissionStage.AFTER_PAYLOAD_INSERT)
                return self._accepted(submission, values, replayed=False)
        except IntegrityError:
            with self.sessions.begin() as session:
                self._authorize_publication(
                    session,
                    caller_user_id=caller_user_id,
                    group_id=submission.group_id,
                    source_document_id=submission.source_document_id,
                    changed_registration_id=(
                        submission.changed_repository_registration_id
                    ),
                    publication=submission.index_publication,
                    lock=True,
                )
                existing = (
                    session.execute(
                        select(baseline_run_job).where(
                            baseline_run_job.c.group_id == submission.group_id,
                            baseline_run_job.c.idempotency_key_hash == key_hash,
                        )
                    )
                    .mappings()
                    .first()
                )
                if existing is None:
                    raise BaselineRunJobError(
                        "run_payload_encryption_failed",
                        status_code=503,
                        retryable=True,
                    ) from None
                if (
                    existing["intent_hash"] != intent_hash
                    or existing["submitted_by_user_id"] != caller_user_id
                ):
                    raise BaselineRunJobError("idempotency_conflict") from None
                return self._accepted(submission, existing, replayed=True)

    @staticmethod
    def _publication_from_row(row: Mapping[str, Any]) -> V2IndexPublication:
        return V2IndexPublication(
            index_publication_id=str(row["index_publication_id"]),
            corpus_generation_id=str(row["corpus_generation_id"]),
            corpus_manifest_hash=str(row["corpus_manifest_hash"]),
            index_format_version=str(row["index_format_version"]),
            tokenizer_version=str(row["tokenizer_version"]),
            retrieval_config_fingerprint=str(row["retrieval_config_fingerprint"]),
            embedding_fingerprint=str(row["embedding_fingerprint"]),
            index_fingerprint=str(row["index_fingerprint"]),
        )

    def read_status(
        self,
        *,
        request_id: str,
        group_id: str,
        job_id: str,
        caller_user_id: str,
    ) -> dict[str, object]:
        try:
            if str(UUID(request_id)) != request_id or str(UUID(job_id)) != job_id:
                raise ValueError
        except (ValueError, AttributeError):
            raise BaselineRunJobError("protocol_mismatch", status_code=422) from None
        with self.sessions() as session:
            row = (
                session.execute(
                    select(baseline_run_job).where(
                        baseline_run_job.c.group_id == group_id,
                        baseline_run_job.c.job_id == job_id,
                    )
                )
                .mappings()
                .first()
            )
            if (
                row is None
                or row["source_document_id"] is None
                or row["submitted_by_user_id"] is None
                or row["submitted_by_user_id"] != caller_user_id
            ):
                raise BaselineRunJobError("job_not_found_or_forbidden", status_code=404)
            try:
                self._authorize_publication(
                    session,
                    caller_user_id=caller_user_id,
                    group_id=group_id,
                    source_document_id=str(row["source_document_id"]),
                    changed_registration_id=str(
                        row["changed_repository_registration_id"]
                    ),
                    publication=self._publication_from_row(row),
                    lock=False,
                )
            except BaselineRunJobError:
                raise BaselineRunJobError(
                    "job_not_found_or_forbidden", status_code=404
                ) from None
            state = str(row["state"])
            # Generation uses the internal ``running`` state while holding its
            # provider lease.  Once retrieval effects are durable, the frozen
            # public v2 contract must continue to expose that last durable
            # boundary as ``references_persisted`` until Feedback commits.
            if (
                state == "running"
                and row["persisted_run_id"] is not None
                and int(row["evidence_count"]) > 0
                and int(row["evidence_count"]) == int(row["reference_count"])
                and row["failure_stage"] == "generation"
            ):
                state = "references_persisted"
            internal_reason = str(row["reason_code"] or "")
            safe_reason = _safe_status_reason(
                internal_reason,
                stage=str(row["failure_stage"] or "dispatch"),
                state=state,
            )
            state_contract = {
                "queued": (False, "pending", "pending", None, None),
                "running": (False, "pending", "pending", None, None),
                "references_persisted": (False, "pending", "ok", None, None),
                "feedback_persisted": (True, "success", "ok", None, None),
                "insufficient": (
                    True,
                    "insufficient",
                    "insufficient",
                    "retrieval_insufficient",
                    "retrieval",
                ),
                "retryable_failed": (
                    False,
                    "pending",
                    "error",
                    safe_reason,
                    str(row["failure_stage"] or "dispatch"),
                ),
                "terminal_failed": (
                    True,
                    "failed",
                    "error",
                    safe_reason,
                    str(row["failure_stage"] or "dispatch"),
                ),
                "blocked": (
                    True,
                    "blocked",
                    "error",
                    "worker_unavailable"
                    if row["reason_code"] == "payload_expired"
                    else safe_reason,
                    str(row["failure_stage"] or "dispatch"),
                ),
                "cancelled": (
                    True,
                    "cancelled",
                    "error",
                    "job_cancelled",
                    "dispatch",
                ),
            }.get(state)
            if state_contract is None:
                raise BaselineRunJobError("run_job_state_incompatible")
            terminal, exit_class, retrieval_status, reason, failure_stage = (
                state_contract
            )
            publication = self._publication_from_row(row)
            return {
                "protocol_version": PROTOCOL_V2_VERSION,
                "protocol_sha256": PROTOCOL_V2_SHA256,
                "message_type": "job_status",
                "request_id": request_id,
                "group_id": group_id,
                "job_id": job_id,
                "operation": "baseline_run",
                "processing_run_id": str(row["processing_run_id"]),
                "source_document_id": str(row["source_document_id"]),
                "changed_repository_registration_id": str(
                    row["changed_repository_registration_id"]
                ),
                "index_publication": {
                    "index_publication_id": publication.index_publication_id,
                    "corpus_generation_id": publication.corpus_generation_id,
                    "corpus_manifest_hash": publication.corpus_manifest_hash,
                    "index_format_version": publication.index_format_version,
                    "tokenizer_version": publication.tokenizer_version,
                    "retrieval_config_fingerprint": (
                        publication.retrieval_config_fingerprint
                    ),
                    "embedding_fingerprint": publication.embedding_fingerprint,
                    "index_fingerprint": publication.index_fingerprint,
                },
                "state": state,
                "terminal": terminal,
                "exit_classification": exit_class,
                "attempt": int(row["attempt_count"]),
                "created_at": _aware(row["created_at"])
                .isoformat()
                .replace("+00:00", "Z"),
                "updated_at": _aware(row["updated_at"])
                .isoformat()
                .replace("+00:00", "Z"),
                "retrieval_status": retrieval_status,
                "query_provenance": {
                    "sha256": str(row["query_sha256"]),
                    "length": int(row["query_length"]),
                    "byte_size": int(row["query_byte_length"]),
                    "origin": str(row["query_origin"]),
                },
                "effects": {
                    "evidence_count": int(row["evidence_count"]),
                    "reference_count": int(row["reference_count"]),
                    "feedback_count": int(row["feedback_count"]),
                    "generation_invoked": bool(row["generation_invoked"]),
                    "notification_outbox_count": int(row["notification_outbox_count"]),
                    "persisted_run_id": row["persisted_run_id"],
                },
                "reason_code": reason,
                "failure_stage": failure_stage,
                "replayed": False,
            }

    def verify_protected_payload_integrity(
        self,
        *,
        group_id: str,
        job_id: str,
        caller_user_id: str,
    ) -> bool:
        """Authenticate an envelope without releasing either protected value.

        This is not a worker claim or execution path.  It exists for startup,
        rotation, and corruption checks.  A failed authentication is erased
        and durably blocked with only a sanitized internal reason.
        """

        failure = False
        with self.sessions.begin() as session:
            job = (
                session.execute(
                    select(baseline_run_job).where(
                        baseline_run_job.c.group_id == group_id,
                        baseline_run_job.c.job_id == job_id,
                    )
                )
                .mappings()
                .first()
            )
            payload = (
                session.execute(
                    select(baseline_run_payload).where(
                        baseline_run_payload.c.group_id == group_id,
                        baseline_run_payload.c.job_id == job_id,
                    )
                )
                .mappings()
                .first()
            )
            if (
                job is None
                or payload is None
                or job["source_document_id"] is None
                or job["submitted_by_user_id"] is None
            ):
                raise BaselineRunJobError("job_not_found_or_forbidden", status_code=404)
            try:
                self._authorize_publication(
                    session,
                    caller_user_id=caller_user_id,
                    group_id=group_id,
                    source_document_id=str(job["source_document_id"]),
                    changed_registration_id=str(
                        job["changed_repository_registration_id"]
                    ),
                    publication=self._publication_from_row(job),
                    lock=True,
                )
                self.cipher.decrypt(job=job, payload=payload)
            except BaselineRunJobError as exc:
                if exc.code in {
                    "run_payload_authentication_failed",
                    "run_payload_key_unavailable",
                }:
                    now = self.clock()
                    session.execute(
                        delete(baseline_run_payload).where(
                            baseline_run_payload.c.job_id == job_id,
                            baseline_run_payload.c.group_id == group_id,
                        )
                    )
                    session.execute(
                        update(baseline_run_job)
                        .where(
                            baseline_run_job.c.job_id == job_id,
                            baseline_run_job.c.group_id == group_id,
                            baseline_run_job.c.state.notin_(_TERMINAL_STATES),
                        )
                        .values(
                            state="blocked",
                            lease_token=None,
                            lease_expires_at=None,
                            reason_code="payload_authentication_failed",
                            failure_stage="dispatch",
                            updated_at=now,
                            finished_at=now,
                        )
                    )
                    failure = True
                else:
                    raise
        if failure:
            raise BaselineRunJobError("run_payload_authentication_failed")
        return True

    def cleanup_protected_payloads(self, *, limit: int = 100) -> int:
        if not 1 <= limit <= 1000:
            raise BaselineRunJobError("limit_exceeded", status_code=422)
        now = self.clock()
        erased = 0
        with self.sessions.begin() as session:
            statement = (
                select(baseline_run_job, baseline_run_payload)
                .join(
                    baseline_run_payload,
                    (baseline_run_payload.c.job_id == baseline_run_job.c.job_id)
                    & (baseline_run_payload.c.group_id == baseline_run_job.c.group_id),
                )
                .where(
                    (baseline_run_payload.c.expires_at <= now)
                    | (baseline_run_job.c.state.in_(_TERMINAL_STATES))
                    | (baseline_run_job.c.state == "references_persisted")
                )
                .order_by(
                    baseline_run_payload.c.expires_at,
                    baseline_run_payload.c.job_id,
                )
                .limit(limit)
            )
            if session.get_bind().dialect.name == "postgresql":
                statement = statement.with_for_update(skip_locked=True)
            rows = session.execute(statement).mappings().all()
            for row in rows:
                if (
                    row["state"] == "running"
                    and row["lease_expires_at"] is not None
                    and _aware(row["lease_expires_at"]) > now
                ):
                    continue
                session.execute(
                    delete(baseline_run_payload).where(
                        baseline_run_payload.c.job_id == row["job_id"],
                        baseline_run_payload.c.group_id == row["group_id"],
                    )
                )
                erased += 1
                if row["state"] == "references_persisted":
                    session.execute(
                        update(baseline_run_job)
                        .where(
                            baseline_run_job.c.job_id == row["job_id"],
                            baseline_run_job.c.group_id == row["group_id"],
                            baseline_run_job.c.state == "references_persisted",
                        )
                        .values(
                            lease_token=None,
                            lease_expires_at=None,
                            updated_at=now,
                        )
                    )
                elif row["state"] not in _TERMINAL_STATES:
                    session.execute(
                        update(baseline_run_job)
                        .where(
                            baseline_run_job.c.job_id == row["job_id"],
                            baseline_run_job.c.group_id == row["group_id"],
                        )
                        .values(
                            state="blocked",
                            lease_token=None,
                            lease_expires_at=None,
                            reason_code="payload_expired",
                            failure_stage="dispatch",
                            updated_at=now,
                            finished_at=now,
                        )
                    )
        return erased


__all__ = [
    "RUN_ENCRYPTION_ALGORITHM",
    "RUN_JOB_CONTRACT_VERSION",
    "RUN_KEYRING_VERSION",
    "RUN_PAYLOAD_AAD_VERSION",
    "RUN_PAYLOAD_SCHEMA_VERSION",
    "BaselineRunJobError",
    "BaselineRunJobService",
    "BaselineRunKeyring",
    "BaselineRunPayloadCipher",
    "ProtectedRunPayload",
    "RunSubmissionStage",
    "keyring_from_settings",
]
