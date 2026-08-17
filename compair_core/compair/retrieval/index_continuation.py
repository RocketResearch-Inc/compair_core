"""Lease-safe compatible-index continuation over the existing index builder.

This module owns job orchestration only.  The frozen tokenizer, lexical/dense
artifacts, validation, and compatible-publication pointer remain owned by
``indexing.py``.  It has no retrieval-query, baseline-run, generation, or
notification entry point.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Protocol
from uuid import UUID, uuid4

from sqlalchemy import Engine, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from compair_core.baseline_control_plane_schema import (
    compatible_index_job,
    control_job,
    snapshot_continuation_job,
)

from .baseline import BASELINE_TOKENIZER_VERSION
from .continuation_worker import (
    BaselineContinuationWorker,
    continuation_result_provenance_fingerprint,
)
from .control_plane import (
    DEFAULT_LEASE_LIFETIME,
    PROTOCOL_SHA256,
    PROTOCOL_VERSION,
    BaselineControlPlaneService,
    ControlPlaneError,
    LeaseReceipt,
    canonical_sha256,
)
from .corpus import (
    CorpusGenerationStatus,
    CorpusIngestionStatus,
    CorpusLifecycle,
    IndexStateStatus,
    RetrievalBaselineIndexBuild,
    RetrievalBaselineIndexPublication,
    RetrievalBaselineIndexVector,
    RetrievalCorpus,
    RetrievalCorpusFile,
    RetrievalCorpusGeneration,
    RetrievalCorpusIngestion,
    RetrievalIndexState,
)
from .embedding import (
    BASELINE_EMBEDDING_HTTP_CONTRACT,
    BASELINE_EMBEDDING_HTTP_PROVIDER,
    BaselineEmbeddingAdapterError,
)
from .indexing import (
    BASELINE_INDEX_SCHEMA_VERSION,
    BaselineEmbeddingIdentity,
    BaselineIndexBuilder,
    BaselineIndexBuildError,
    BaselineIndexBuildResult,
    BaselineIndexLifecycle,
    baseline_engine_config_fingerprint,
)
from .ingestion import CorpusGenerationFreshness, CorpusIngestionResult
from .persistent import published_index_fingerprint

INDEX_JOB_CONTRACT_VERSION = "baseline-index-build-continuation.v1"
INDEX_JOB_WORKER_SERVICE_ID = "compair-core-compatible-index"
BASELINE_EMBEDDING_DTYPE = "float32"
PINNED_BASELINE_MODEL = "BAAI/bge-small-en-v1.5"
PINNED_BASELINE_DIMENSION = 384
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]*$")
_HEX = frozenset("0123456789abcdef")


class AttestedBaselineEmbeddingAdapter(Protocol):
    provider: str
    model: str
    revision: str
    dimension: int
    fingerprint: str

    @property
    def identity(self) -> BaselineEmbeddingIdentity: ...

    def attest(self) -> BaselineEmbeddingIdentity: ...

    def embed(self, texts: Any) -> Any: ...


class IndexJobStage(str, Enum):
    CLAIMED = "claimed"
    ATTESTED = "attested"
    BEFORE_BUILD = "before_build"
    BEFORE_PUBLICATION = "before_publication"
    AFTER_PUBLICATION = "after_publication"
    BEFORE_SUCCESS = "before_success"
    AFTER_COMMIT = "after_commit"


@dataclass(frozen=True, slots=True)
class InternalIndexWorkerIdentity:
    instance_id: str
    service_id: str = INDEX_JOB_WORKER_SERVICE_ID
    contract_version: str = INDEX_JOB_CONTRACT_VERSION

    @classmethod
    def create(cls, instance_id: str | None = None) -> InternalIndexWorkerIdentity:
        return cls(instance_id=instance_id or f"worker-{uuid4()}")

    def validate(self) -> None:
        if (
            self.service_id != INDEX_JOB_WORKER_SERVICE_ID
            or self.contract_version != INDEX_JOB_CONTRACT_VERSION
            or not 1 <= len(self.instance_id) <= 128
            or _SAFE_ID.fullmatch(self.instance_id) is None
        ):
            raise IndexJobError("worker_identity_invalid", retryable=False)


@dataclass(frozen=True, slots=True)
class IndexBuildIntent:
    request_id: str
    group_id: str
    idempotency_key: str
    snapshot_id: str
    generation_id: str
    control_manifest_hash: str
    index_format_version: str
    tokenizer_version: str
    retrieval_config_fingerprint: str
    embedding_contract_version: str
    embedding: BaselineEmbeddingIdentity

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(
            {
                "contract_version": INDEX_JOB_CONTRACT_VERSION,
                "group_id": self.group_id,
                "snapshot_id": self.snapshot_id,
                "generation_id": self.generation_id,
                "control_manifest_hash": self.control_manifest_hash,
                "index_format_version": self.index_format_version,
                "tokenizer_version": self.tokenizer_version,
                "retrieval_config_fingerprint": self.retrieval_config_fingerprint,
                "embedding": {
                    "contract_version": self.embedding_contract_version,
                    "provider": self.embedding.provider,
                    "model": self.embedding.model,
                    "revision": self.embedding.revision,
                    "dimension": self.embedding.dimension,
                    "dtype": BASELINE_EMBEDDING_DTYPE,
                    "fingerprint": self.embedding.fingerprint,
                },
            }
        )


@dataclass(frozen=True, slots=True)
class IndexJobOutcome:
    job_id: str
    group_id: str
    state: str
    attempt_count: int
    corpus_id: str
    generation_id: str
    index_id: str
    document_count: int
    document_manifest_hash: str
    lexical_manifest_hash: str
    dense_manifest_hash: str
    retrieval_config_fingerprint: str
    embedding_fingerprint: str


class IndexJobError(RuntimeError):
    """Bounded orchestration failure containing no source/provider payload."""

    def __init__(
        self,
        code: str,
        *,
        retryable: bool,
        status_code: int = 409,
    ) -> None:
        self.code = code
        self.retryable = retryable
        self.status_code = status_code
        super().__init__(code)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _safe_identifier(value: Any, label: str, maximum: int = 128) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= maximum
        or value != value.strip()
        or _SAFE_ID.fullmatch(value) is None
    ):
        raise IndexJobError(f"{label}_invalid", retryable=False, status_code=422)
    return value


def _canonical_text(value: Any, label: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= maximum
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise IndexJobError(f"{label}_invalid", retryable=False, status_code=422)
    return value


def _uuid(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise IndexJobError(f"{label}_invalid", retryable=False, status_code=422)
    try:
        parsed = UUID(value)
    except (ValueError, AttributeError):
        raise IndexJobError(
            f"{label}_invalid", retryable=False, status_code=422
        ) from None
    if str(parsed) != value.lower():
        raise IndexJobError(f"{label}_invalid", retryable=False, status_code=422)
    return str(parsed)


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise IndexJobError(f"{label}_invalid", retryable=False, status_code=422)
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in _HEX for character in normalized):
        raise IndexJobError(f"{label}_invalid", retryable=False, status_code=422)
    return normalized


def parse_index_build_intent(payload: Mapping[str, Any]) -> IndexBuildIntent:
    expected = {
        "protocol_version",
        "protocol_sha256",
        "message_type",
        "request_id",
        "group_id",
        "idempotency_key",
        "snapshot_id",
        "corpus_generation_id",
        "canonical_manifest_hash",
        "index_format_version",
        "tokenizer_version",
        "retrieval_config_fingerprint",
        "embedding",
    }
    if set(payload) != expected:
        raise IndexJobError("invalid_contract", retryable=False, status_code=422)
    if (
        payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("protocol_sha256") != PROTOCOL_SHA256
        or payload.get("message_type") != "index_build_submit"
    ):
        raise IndexJobError("protocol_mismatch", retryable=False, status_code=409)
    embedding = payload.get("embedding")
    if not isinstance(embedding, Mapping) or set(embedding) != {
        "contract_version",
        "provider",
        "model",
        "revision",
        "dimension",
        "fingerprint",
    }:
        raise IndexJobError(
            "embedding_identity_invalid", retryable=False, status_code=422
        )
    dimension = embedding.get("dimension")
    if not isinstance(dimension, int) or isinstance(dimension, bool):
        raise IndexJobError(
            "embedding_identity_invalid", retryable=False, status_code=422
        )
    identity = BaselineEmbeddingIdentity(
        provider=_safe_identifier(embedding.get("provider"), "embedding_provider"),
        model=_canonical_text(embedding.get("model"), "embedding_model", 256),
        revision=_canonical_text(embedding.get("revision"), "embedding_revision", 256),
        dimension=dimension,
        fingerprint=_sha256(embedding.get("fingerprint"), "embedding_fingerprint"),
    )
    snapshot_id = _safe_identifier(payload.get("snapshot_id"), "snapshot", 72)
    if not snapshot_id.startswith("bsnap_") or len(snapshot_id) != 70:
        raise IndexJobError("snapshot_invalid", retryable=False, status_code=422)
    idempotency_key = _safe_identifier(
        payload.get("idempotency_key"), "idempotency_key", 128
    )
    if len(idempotency_key) < 32:
        raise IndexJobError("idempotency_key_invalid", retryable=False, status_code=422)
    return IndexBuildIntent(
        request_id=_uuid(payload.get("request_id"), "request"),
        group_id=_safe_identifier(payload.get("group_id"), "group", 64),
        idempotency_key=idempotency_key,
        snapshot_id=snapshot_id,
        generation_id=_uuid(payload.get("corpus_generation_id"), "generation"),
        control_manifest_hash=_sha256(
            payload.get("canonical_manifest_hash"), "control_manifest_hash"
        ),
        index_format_version=_safe_identifier(
            payload.get("index_format_version"), "index_format", 64
        ),
        tokenizer_version=_safe_identifier(
            payload.get("tokenizer_version"), "tokenizer", 128
        ),
        retrieval_config_fingerprint=_sha256(
            payload.get("retrieval_config_fingerprint"), "retrieval_config"
        ),
        embedding_contract_version=_safe_identifier(
            embedding.get("contract_version"), "embedding_contract", 64
        ),
        embedding=identity,
    )


class BaselineCompatibleIndexJobService:
    """Submit, claim, execute, and inspect one compatible-index continuation."""

    def __init__(
        self,
        engine: Engine,
        adapter_factory: Callable[[], AttestedBaselineEmbeddingAdapter],
        *,
        clock: Callable[[], datetime] = _utcnow,
        stage_hook: Callable[[IndexJobStage], None] | None = None,
    ) -> None:
        self.engine = engine
        self.sessions = sessionmaker(engine, expire_on_commit=False)
        self.adapter_factory = adapter_factory
        self.clock = clock
        self.stage_hook = stage_hook
        self.control = BaselineControlPlaneService(engine, clock=clock)

    def _stage(self, stage: IndexJobStage) -> None:
        if self.stage_hook is not None:
            self.stage_hook(stage)

    @staticmethod
    def _required_identity(identity: BaselineEmbeddingIdentity) -> None:
        if (
            identity.provider != BASELINE_EMBEDDING_HTTP_PROVIDER
            or identity.model != PINNED_BASELINE_MODEL
            or identity.dimension != PINNED_BASELINE_DIMENSION
        ):
            raise IndexJobError("embedding_identity_mismatch", retryable=False)

    def _configured_adapter(self) -> AttestedBaselineEmbeddingAdapter:
        try:
            adapter = self.adapter_factory()
        except BaselineEmbeddingAdapterError as exc:
            raise IndexJobError(exc.code, retryable=True, status_code=503) from None
        except Exception:  # noqa: BLE001 - adapter construction boundary
            raise IndexJobError(
                "embedding_adapter_unavailable", retryable=True, status_code=503
            ) from None
        self._required_identity(adapter.identity)
        return adapter

    def attest_configured_identity(self) -> BaselineEmbeddingIdentity:
        """Attest the configured production identity without building an index."""

        adapter = self._configured_adapter()
        try:
            attested = adapter.attest()
        except BaselineEmbeddingAdapterError as exc:
            raise IndexJobError(exc.code, retryable=True, status_code=503) from None
        except Exception:  # noqa: BLE001 - provider capability boundary
            raise IndexJobError(
                "embedding_adapter_unavailable", retryable=True, status_code=503
            ) from None
        self._required_identity(attested)
        if attested != adapter.identity:
            raise IndexJobError("embedding_identity_mismatch", retryable=False)
        return attested

    def _validate_intent_against_adapter(
        self,
        intent: IndexBuildIntent,
        adapter: AttestedBaselineEmbeddingAdapter,
    ) -> None:
        self._required_identity(intent.embedding)
        if (
            intent.index_format_version != BASELINE_INDEX_SCHEMA_VERSION
            or intent.tokenizer_version != BASELINE_TOKENIZER_VERSION
            or intent.embedding_contract_version != BASELINE_EMBEDDING_HTTP_CONTRACT
            or intent.embedding != adapter.identity
            or intent.retrieval_config_fingerprint
            != baseline_engine_config_fingerprint(intent.embedding)
        ):
            raise IndexJobError("index_intent_incompatible", retryable=False)

    @staticmethod
    def _continuation_result(continuation: Mapping[str, Any]) -> CorpusIngestionResult:
        return CorpusIngestionResult(
            corpus_id=str(continuation["result_corpus_id"]),
            generation_id=str(continuation["result_generation_id"]),
            generation_version=str(continuation["result_generation_version"]),
            manifest_hash=str(continuation["result_manifest_hash"]),
            status=CorpusGenerationFreshness.ACTIVE,
        )

    def _validate_context(
        self,
        session: Session,
        *,
        job: Mapping[str, Any] | None,
        extension: Mapping[str, Any] | None,
        continuation_id: str,
        group_id: str,
        expected_generation_id: str,
        lock: bool,
    ) -> tuple[Mapping[str, Any], RetrievalCorpusGeneration, RetrievalCorpusIngestion]:
        statement = select(snapshot_continuation_job).where(
            snapshot_continuation_job.c.group_id == group_id,
            snapshot_continuation_job.c.continuation_job_id == continuation_id,
        )
        if lock and session.get_bind().dialect.name == "postgresql":
            statement = statement.with_for_update()
        continuation = session.execute(statement).mappings().first()
        if continuation is None or continuation["state"] != "succeeded":
            raise IndexJobError("ingestion_continuation_not_succeeded", retryable=False)
        submitter = continuation["created_by_user_id"]
        if submitter is None or (job is not None and job["group_id"] != group_id):
            raise IndexJobError(
                "source_not_authorized", retryable=False, status_code=404
            )
        if extension is not None and extension["submitted_by_user_id"] != submitter:
            raise IndexJobError(
                "source_not_authorized", retryable=False, status_code=404
            )
        try:
            BaselineContinuationWorker._authorize_submitter(
                session,
                user_id=str(submitter),
                group_id=group_id,
            )
            self.control._validate_continuation_claim(
                session,
                continuation=continuation,
                caller_user_id=str(submitter),
            )
        except ControlPlaneError as exc:
            raise IndexJobError(
                exc.code, retryable=False, status_code=exc.status_code
            ) from None

        generation_id = str(continuation["result_generation_id"])
        if generation_id != expected_generation_id:
            raise IndexJobError("corpus_generation_mismatch", retryable=False)
        generation_statement = select(RetrievalCorpusGeneration).where(
            RetrievalCorpusGeneration.generation_id == generation_id
        )
        if lock and session.get_bind().dialect.name == "postgresql":
            generation_statement = generation_statement.with_for_update()
        generation = session.scalar(generation_statement)
        if generation is None:
            raise IndexJobError("corpus_generation_absent", retryable=False)
        corpus = session.get(RetrievalCorpus, generation.corpus_id)
        ingestion = session.get(RetrievalCorpusIngestion, generation_id)
        if (
            corpus is None
            or ingestion is None
            or corpus.corpus_id != continuation["result_corpus_id"]
            or corpus.scope_key != f"group:{group_id}"
            or corpus.active_generation_id != generation_id
            or generation.status != CorpusGenerationStatus.ACTIVE.value
            or generation.generation_version
            != continuation["result_generation_version"]
            or ingestion.status != CorpusIngestionStatus.ACTIVE.value
            or ingestion.canonical_manifest_hash != continuation["result_manifest_hash"]
            or ingestion.source_manifest_hash != continuation["canonical_manifest_hash"]
            or ingestion.snapshot_id != continuation["snapshot_id"]
            or generation.manifest_hash is None
        ):
            raise IndexJobError("corpus_generation_stale", retryable=False)
        if (
            CorpusLifecycle.generation_integrity_error(session, generation_id)
            is not None
        ):
            raise IndexJobError("corpus_generation_incompatible", retryable=False)
        files = tuple(
            session.scalars(
                select(RetrievalCorpusFile).where(
                    RetrievalCorpusFile.generation_id == generation_id
                )
            )
        )
        if any(
            not isinstance(item.content_hash, str)
            or len(item.content_hash) != 64
            or any(character not in _HEX for character in item.content_hash.lower())
            for item in files
        ):
            raise IndexJobError("corpus_file_hash_absent", retryable=False)
        provenance = continuation_result_provenance_fingerprint(
            continuation=continuation,
            result=self._continuation_result(continuation),
        )
        if provenance != continuation["result_provenance_fingerprint"]:
            raise IndexJobError("ingestion_provenance_mismatch", retryable=False)
        if extension is not None and (
            extension["snapshot_id"] != continuation["snapshot_id"]
            or extension["corpus_id"] != corpus.corpus_id
            or extension["generation_id"] != generation_id
            or extension["generation_version"] != generation.generation_version
            or extension["control_manifest_hash"]
            != continuation["canonical_manifest_hash"]
            or extension["corpus_manifest_hash"] != continuation["result_manifest_hash"]
            or extension["corpus_file_manifest_hash"] != generation.manifest_hash
            or extension["ingestion_provenance_fingerprint"] != provenance
        ):
            raise IndexJobError("ingestion_provenance_mismatch", retryable=False)
        state = session.get(RetrievalIndexState, generation_id)
        if state is None or state.status not in {
            IndexStateStatus.INCOMPLETE.value,
            IndexStateStatus.COMPATIBLE.value,
        }:
            raise IndexJobError("index_state_incompatible", retryable=False)
        return continuation, generation, ingestion

    def submit(
        self,
        payload: Mapping[str, Any],
        *,
        caller_user_id: str,
        stored_protocol_version: str = PROTOCOL_VERSION,
        stored_protocol_sha256: str = PROTOCOL_SHA256,
        intent_fingerprint_override: str | None = None,
        allow_new: bool = True,
    ) -> dict[str, object]:
        intent = parse_index_build_intent(payload)
        intent_fingerprint = (
            _sha256(intent_fingerprint_override, "intent_fingerprint")
            if intent_fingerprint_override is not None
            else intent.fingerprint
        )
        adapter = self._configured_adapter()
        self._validate_intent_against_adapter(intent, adapter)
        try:
            with self.sessions.begin() as session:
                self.control._authorize_group(
                    session,
                    user_id=caller_user_id,
                    group_id=intent.group_id,
                )
                continuation = (
                    session.execute(
                        select(snapshot_continuation_job).where(
                            snapshot_continuation_job.c.group_id == intent.group_id,
                            snapshot_continuation_job.c.result_generation_id
                            == intent.generation_id,
                        )
                    )
                    .mappings()
                    .first()
                )
                if continuation is None:
                    raise IndexJobError(
                        "not_found_or_forbidden", retryable=False, status_code=404
                    )
                if continuation["created_by_user_id"] != caller_user_id:
                    raise IndexJobError(
                        "not_found_or_forbidden", retryable=False, status_code=404
                    )
                _continuation, generation, _ingestion = self._validate_context(
                    session,
                    job=None,
                    extension=None,
                    continuation_id=str(continuation["continuation_job_id"]),
                    group_id=intent.group_id,
                    expected_generation_id=intent.generation_id,
                    lock=True,
                )
                if (
                    intent.snapshot_id != continuation["snapshot_id"]
                    or intent.control_manifest_hash
                    != continuation["canonical_manifest_hash"]
                ):
                    raise IndexJobError("index_intent_incompatible", retryable=False)
                existing_key = (
                    session.execute(
                        select(control_job).where(
                            control_job.c.group_id == intent.group_id,
                            control_job.c.operation == "index_build",
                            control_job.c.idempotency_key == intent.idempotency_key,
                        )
                    )
                    .mappings()
                    .first()
                )
                exact = (
                    session.execute(
                        select(compatible_index_job).where(
                            compatible_index_job.c.group_id == intent.group_id,
                            compatible_index_job.c.generation_id
                            == intent.generation_id,
                            compatible_index_job.c.index_intent_hash
                            == intent_fingerprint,
                        )
                    )
                    .mappings()
                    .first()
                )
                existing = existing_key
                if (
                    existing_key is not None
                    and existing_key["intent_hash"] != intent_fingerprint
                ):
                    raise IndexJobError("index_build_conflict", retryable=False)
                if exact is not None:
                    existing = (
                        session.execute(
                            select(control_job).where(
                                control_job.c.job_id == exact["job_id"]
                            )
                        )
                        .mappings()
                        .one()
                    )
                if existing is not None:
                    extension = (
                        session.execute(
                            select(compatible_index_job).where(
                                compatible_index_job.c.job_id == existing["job_id"]
                            )
                        )
                        .mappings()
                        .one()
                    )
                    self._validate_context(
                        session,
                        job=existing,
                        extension=extension,
                        continuation_id=str(extension["continuation_job_id"]),
                        group_id=intent.group_id,
                        expected_generation_id=intent.generation_id,
                        lock=True,
                    )
                    return self._accepted(intent, existing, replayed=True)

                if not allow_new:
                    raise IndexJobError(
                        "worker_unavailable",
                        retryable=True,
                        status_code=503,
                    )

                job_id = str(uuid4())
                now = self.clock()
                session.execute(
                    control_job.insert().values(
                        job_id=job_id,
                        group_id=intent.group_id,
                        request_id=intent.request_id,
                        operation="index_build",
                        idempotency_key=intent.idempotency_key,
                        intent_hash=intent_fingerprint,
                        protocol_version=stored_protocol_version,
                        protocol_sha256=stored_protocol_sha256,
                        state="queued",
                        attempt_count=0,
                        progress_completed=0,
                        progress_total=1,
                        created_at=now,
                        updated_at=now,
                    )
                )
                session.execute(
                    compatible_index_job.insert().values(
                        job_id=job_id,
                        group_id=intent.group_id,
                        continuation_job_id=continuation["continuation_job_id"],
                        submitted_by_user_id=caller_user_id,
                        contract_version=INDEX_JOB_CONTRACT_VERSION,
                        index_intent_hash=intent_fingerprint,
                        snapshot_id=intent.snapshot_id,
                        corpus_id=continuation["result_corpus_id"],
                        generation_id=intent.generation_id,
                        generation_version=generation.generation_version,
                        control_manifest_hash=intent.control_manifest_hash,
                        corpus_manifest_hash=continuation["result_manifest_hash"],
                        corpus_file_manifest_hash=generation.manifest_hash,
                        ingestion_provenance_fingerprint=continuation[
                            "result_provenance_fingerprint"
                        ],
                        index_format_version=intent.index_format_version,
                        tokenizer_version=intent.tokenizer_version,
                        retrieval_config_fingerprint=intent.retrieval_config_fingerprint,
                        embedding_contract_version=intent.embedding_contract_version,
                        embedding_provider=intent.embedding.provider,
                        embedding_model=intent.embedding.model,
                        embedding_revision=intent.embedding.revision,
                        embedding_dimension=intent.embedding.dimension,
                        embedding_dtype=BASELINE_EMBEDDING_DTYPE,
                        embedding_fingerprint=intent.embedding.fingerprint,
                        created_at=now,
                        updated_at=now,
                    )
                )
                row = (
                    session.execute(
                        select(control_job).where(control_job.c.job_id == job_id)
                    )
                    .mappings()
                    .one()
                )
                return self._accepted(intent, row, replayed=False)
        except IntegrityError:
            # A concurrent exact submit won the unique generation-intent race.
            with self.sessions.begin() as session:
                self.control._authorize_group(
                    session,
                    user_id=caller_user_id,
                    group_id=intent.group_id,
                )
                exact = (
                    session.execute(
                        select(compatible_index_job).where(
                            compatible_index_job.c.group_id == intent.group_id,
                            compatible_index_job.c.generation_id
                            == intent.generation_id,
                            compatible_index_job.c.index_intent_hash
                            == intent_fingerprint,
                        )
                    )
                    .mappings()
                    .first()
                )
                if exact is None:
                    raise IndexJobError(
                        "index_build_conflict", retryable=False
                    ) from None
                row = (
                    session.execute(
                        select(control_job).where(
                            control_job.c.job_id == exact["job_id"]
                        )
                    )
                    .mappings()
                    .one()
                )
                self._validate_context(
                    session,
                    job=row,
                    extension=exact,
                    continuation_id=str(exact["continuation_job_id"]),
                    group_id=intent.group_id,
                    expected_generation_id=intent.generation_id,
                    lock=True,
                )
                return self._accepted(intent, row, replayed=True)
        except ControlPlaneError as exc:
            raise IndexJobError(
                exc.code, retryable=exc.retryable, status_code=exc.status_code
            ) from None

    def submit_bound_v2(
        self,
        *,
        request_id: str,
        group_id: str,
        idempotency_key: str,
        continuation_id: str,
        generation_id: str,
        corpus_manifest_hash: str,
        ingestion_provenance_fingerprint: str,
        index_format_version: str,
        tokenizer_version: str,
        retrieval_config_fingerprint: str,
        embedding_contract_version: str,
        embedding: BaselineEmbeddingIdentity,
        caller_user_id: str,
        protocol_version: str,
        protocol_sha256: str,
        allow_new: bool = True,
    ) -> dict[str, object]:
        """Bind frozen v2 provenance, then reuse the existing v1 job service.

        The first transaction performs no writes.  Succeeded continuation rows
        and their result fields are migration-guarded immutable; ``submit``
        nevertheless repeats authorization and corpus freshness validation in
        its write transaction.
        """

        try:
            with self.sessions() as session:
                self.control._authorize_group(
                    session, user_id=caller_user_id, group_id=group_id
                )
                continuation = (
                    session.execute(
                        select(snapshot_continuation_job).where(
                            snapshot_continuation_job.c.group_id == group_id,
                            snapshot_continuation_job.c.continuation_job_id
                            == continuation_id,
                        )
                    )
                    .mappings()
                    .first()
                )
                if (
                    continuation is None
                    or continuation["created_by_user_id"] != caller_user_id
                ):
                    raise IndexJobError(
                        "not_found_or_forbidden",
                        retryable=False,
                        status_code=404,
                    )
                self._validate_context(
                    session,
                    job=None,
                    extension=None,
                    continuation_id=continuation_id,
                    group_id=group_id,
                    expected_generation_id=generation_id,
                    lock=False,
                )
                if (
                    continuation["result_manifest_hash"] != corpus_manifest_hash
                    or continuation["result_provenance_fingerprint"]
                    != ingestion_provenance_fingerprint
                ):
                    raise IndexJobError(
                        "ingestion_provenance_mismatch", retryable=False
                    )
                legacy_payload: dict[str, object] = {
                    "protocol_version": PROTOCOL_VERSION,
                    "protocol_sha256": PROTOCOL_SHA256,
                    "message_type": "index_build_submit",
                    "request_id": request_id,
                    "group_id": group_id,
                    "idempotency_key": idempotency_key,
                    "snapshot_id": str(continuation["snapshot_id"]),
                    "corpus_generation_id": generation_id,
                    "canonical_manifest_hash": str(
                        continuation["canonical_manifest_hash"]
                    ),
                    "index_format_version": index_format_version,
                    "tokenizer_version": tokenizer_version,
                    "retrieval_config_fingerprint": retrieval_config_fingerprint,
                    "embedding": {
                        "contract_version": embedding_contract_version,
                        "provider": embedding.provider,
                        "model": embedding.model,
                        "revision": embedding.revision,
                        "dimension": embedding.dimension,
                        "fingerprint": embedding.fingerprint,
                    },
                }
                v2_intent_fingerprint = canonical_sha256(
                    {
                        "protocol_version": protocol_version,
                        "group_id": group_id,
                        "ingestion_continuation_id": continuation_id,
                        "corpus_generation_id": generation_id,
                        "corpus_manifest_hash": corpus_manifest_hash,
                        "ingestion_provenance_fingerprint": (
                            ingestion_provenance_fingerprint
                        ),
                        "index_intent": {
                            "index_format_version": index_format_version,
                            "tokenizer_version": tokenizer_version,
                            "retrieval_config_fingerprint": (
                                retrieval_config_fingerprint
                            ),
                            "embedding": {
                                "contract_version": embedding_contract_version,
                                "provider": embedding.provider,
                                "model": embedding.model,
                                "revision": embedding.revision,
                                "dimension": embedding.dimension,
                                "dtype": BASELINE_EMBEDDING_DTYPE,
                                "fingerprint": embedding.fingerprint,
                            },
                        },
                    }
                )
        except ControlPlaneError as exc:
            raise IndexJobError(
                exc.code,
                retryable=exc.retryable,
                status_code=exc.status_code,
            ) from None
        return self.submit(
            legacy_payload,
            caller_user_id=caller_user_id,
            stored_protocol_version=protocol_version,
            stored_protocol_sha256=protocol_sha256,
            intent_fingerprint_override=v2_intent_fingerprint,
            allow_new=allow_new,
        )

    @staticmethod
    def _accepted(
        intent: IndexBuildIntent,
        job: Mapping[str, Any],
        *,
        replayed: bool,
    ) -> dict[str, object]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "message_type": "job_accepted",
            "request_id": intent.request_id,
            "group_id": intent.group_id,
            "job_id": str(job["job_id"]),
            "operation": "index_build",
            "state": str(job["state"]),
            "replayed": replayed,
        }

    def claim(
        self,
        *,
        identity: InternalIndexWorkerIdentity,
        group_id: str,
        job_id: str,
        lifetime: timedelta = DEFAULT_LEASE_LIFETIME,
    ) -> LeaseReceipt:
        identity.validate()
        group_id = _safe_identifier(group_id, "group", 64)
        job_id = _uuid(job_id, "job")
        if lifetime <= timedelta(0) or lifetime > timedelta(hours=1):
            raise IndexJobError("lease_lifetime_invalid", retryable=False)
        now = self.clock()
        terminal: IndexJobError | None = None
        receipt: LeaseReceipt | None = None
        with self.sessions.begin() as session:
            statement = select(control_job).where(
                control_job.c.group_id == group_id,
                control_job.c.job_id == job_id,
                control_job.c.operation == "index_build",
            )
            if session.get_bind().dialect.name == "postgresql":
                statement = statement.with_for_update()
            job = session.execute(statement).mappings().first()
            extension = (
                session.execute(
                    select(compatible_index_job).where(
                        compatible_index_job.c.group_id == group_id,
                        compatible_index_job.c.job_id == job_id,
                    )
                )
                .mappings()
                .first()
            )
            if job is None or extension is None:
                raise IndexJobError(
                    "not_found_or_forbidden", retryable=False, status_code=404
                )
            try:
                self._validate_context(
                    session,
                    job=job,
                    extension=extension,
                    continuation_id=str(extension["continuation_job_id"]),
                    group_id=group_id,
                    expected_generation_id=str(extension["generation_id"]),
                    lock=True,
                )
            except IndexJobError as exc:
                safe = (
                    exc.code if _SAFE_ID.fullmatch(exc.code) else "index_job_ineligible"
                )
                session.execute(
                    update(control_job)
                    .where(control_job.c.job_id == job_id)
                    .values(
                        state="terminal_failed",
                        lease_token=None,
                        lease_expires_at=None,
                        error_code=safe,
                        error_fingerprint=hashlib.sha256(safe.encode()).hexdigest(),
                        updated_at=now,
                        finished_at=now,
                    )
                )
                terminal = exc
            if terminal is None:
                expired = (
                    job["state"] == "running"
                    and job["lease_expires_at"] is not None
                    and _aware(job["lease_expires_at"]) <= now
                )
                if job["state"] not in {"queued", "retryable_failed"} and not expired:
                    raise IndexJobError("job_lease_unavailable", retryable=True)
                attempt = int(job["attempt_count"]) + 1
                token = secrets.token_urlsafe(32)
                expires = now + lifetime
                changed = session.execute(
                    update(control_job)
                    .where(
                        control_job.c.job_id == job_id,
                        (
                            control_job.c.state.in_({"queued", "retryable_failed"})
                            | (
                                (control_job.c.state == "running")
                                & (control_job.c.lease_expires_at <= now)
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
                if changed.rowcount != 1:
                    raise IndexJobError("job_lease_unavailable", retryable=True)
                receipt = LeaseReceipt(job_id, token, expires, attempt)
        if terminal is not None:
            raise terminal
        assert receipt is not None
        self._stage(IndexJobStage.CLAIMED)
        return receipt

    def execute(
        self,
        *,
        identity: InternalIndexWorkerIdentity,
        group_id: str,
        job_id: str,
        lifetime: timedelta = DEFAULT_LEASE_LIFETIME,
    ) -> IndexJobOutcome:
        completed = self._completed_outcome(group_id=group_id, job_id=job_id)
        if completed is not None:
            return completed
        receipt = self.claim(
            identity=identity,
            group_id=group_id,
            job_id=job_id,
            lifetime=lifetime,
        )
        try:
            adapter = self._configured_adapter()
            try:
                attested = adapter.attest()
            except BaselineEmbeddingAdapterError as exc:
                raise IndexJobError(exc.code, retryable=True) from None
            with self.sessions() as session:
                extension = (
                    session.execute(
                        select(compatible_index_job).where(
                            compatible_index_job.c.job_id == job_id
                        )
                    )
                    .mappings()
                    .one()
                )
            expected = BaselineEmbeddingIdentity(
                provider=str(extension["embedding_provider"]),
                model=str(extension["embedding_model"]),
                revision=str(extension["embedding_revision"]),
                dimension=int(extension["embedding_dimension"]),
                fingerprint=str(extension["embedding_fingerprint"]),
            )
            if attested != expected or adapter.identity != expected:
                raise IndexJobError("embedding_identity_mismatch", retryable=False)
            self._stage(IndexJobStage.ATTESTED)
            self._stage(IndexJobStage.BEFORE_BUILD)

            def publish(session: Session, index_id: str) -> None:
                self._publish_success(
                    session,
                    group_id=group_id,
                    job_id=job_id,
                    lease_token=receipt.lease_token,
                    adapter=adapter,
                    index_id=index_id,
                )

            result = BaselineIndexBuilder(
                self.sessions,
                publish_index=publish,
            ).build(
                generation_id=str(extension["generation_id"]),
                index_version=f"{job_id}.attempt-{receipt.attempt_count}",
                embedding=expected,
                provider=adapter,
            )
            # ``build`` returns only after the outer publication transaction has
            # committed.  This hook models a response/process failure after that
            # durability boundary; the lease-guarded failure recorder must not
            # demote the already-succeeded job.
            self._stage(IndexJobStage.AFTER_COMMIT)
            return self._outcome(
                group_id=group_id,
                job_id=job_id,
                result=result,
                attempt_count=receipt.attempt_count,
            )
        except IndexJobError as exc:
            self._record_failure(
                group_id=group_id,
                job_id=job_id,
                lease_token=receipt.lease_token,
                code=exc.code,
                retryable=exc.retryable,
            )
            raise
        except BaselineIndexBuildError as exc:
            retryable = exc.code in {
                "embedding_adapter_failed",
                "embedding_adapter_unavailable",
                "index_publication_failed",
                "artifact_staging_failed",
            }
            self._record_failure(
                group_id=group_id,
                job_id=job_id,
                lease_token=receipt.lease_token,
                code=exc.code,
                retryable=retryable,
            )
            raise IndexJobError(exc.code, retryable=retryable) from None
        except Exception:  # noqa: BLE001 - durable worker failure boundary
            self._record_failure(
                group_id=group_id,
                job_id=job_id,
                lease_token=receipt.lease_token,
                code="index_build_failed",
                retryable=True,
            )
            raise IndexJobError("index_build_failed", retryable=True) from None

    def _publish_success(
        self,
        session: Session,
        *,
        group_id: str,
        job_id: str,
        lease_token: str,
        adapter: AttestedBaselineEmbeddingAdapter,
        index_id: str,
    ) -> None:
        if not session.in_transaction():  # pragma: no cover - builder contract guard
            raise BaselineIndexBuildError(
                "index_publication_transaction_absent",
                "compatible publication requires the builder transaction",
                index_id=index_id,
            )
        now = self.clock()
        statement = select(control_job).where(
            control_job.c.group_id == group_id,
            control_job.c.job_id == job_id,
            control_job.c.operation == "index_build",
        )
        if session.get_bind().dialect.name == "postgresql":
            statement = statement.with_for_update()
        job = session.execute(statement).mappings().first()
        extension = (
            session.execute(
                select(compatible_index_job).where(
                    compatible_index_job.c.job_id == job_id,
                    compatible_index_job.c.group_id == group_id,
                )
            )
            .mappings()
            .first()
        )
        if (
            job is None
            or extension is None
            or job["state"] != "running"
            or job["lease_token"] != lease_token
            or job["lease_expires_at"] is None
            or _aware(job["lease_expires_at"]) <= now
        ):
            raise BaselineIndexBuildError(
                "job_lease_unavailable",
                "index job lease is unavailable",
                index_id=index_id,
            )
        try:
            self._validate_context(
                session,
                job=job,
                extension=extension,
                continuation_id=str(extension["continuation_job_id"]),
                group_id=group_id,
                expected_generation_id=str(extension["generation_id"]),
                lock=True,
            )
        except IndexJobError as exc:
            raise BaselineIndexBuildError(
                exc.code, exc.code, index_id=index_id
            ) from None
        expected = BaselineEmbeddingIdentity(
            provider=str(extension["embedding_provider"]),
            model=str(extension["embedding_model"]),
            revision=str(extension["embedding_revision"]),
            dimension=int(extension["embedding_dimension"]),
            fingerprint=str(extension["embedding_fingerprint"]),
        )
        build = session.get(RetrievalBaselineIndexBuild, index_id)
        if (
            build is None
            or adapter.identity != expected
            or build.generation_id != extension["generation_id"]
            or build.index_schema_version != extension["index_format_version"]
            or build.tokenizer_version != extension["tokenizer_version"]
            or build.corpus_manifest_hash != extension["corpus_file_manifest_hash"]
            or build.engine_config_fingerprint
            != extension["retrieval_config_fingerprint"]
            or build.embedding_provider != expected.provider
            or build.embedding_model != expected.model
            or build.embedding_revision != expected.revision
            or build.embedding_dimension != expected.dimension
            or build.embedding_fingerprint != expected.fingerprint
        ):
            raise BaselineIndexBuildError(
                "index_fingerprint_mismatch",
                "staged index does not match the pinned job",
                index_id=index_id,
            )
        self._stage(IndexJobStage.BEFORE_PUBLICATION)
        BaselineIndexLifecycle.publish(session, index_id)
        publication = session.get(
            RetrievalBaselineIndexPublication, extension["corpus_id"]
        )
        if publication is None or publication.index_id != index_id:
            raise BaselineIndexBuildError(
                "index_publication_mismatch",
                "compatible publication mismatch",
                index_id=index_id,
            )
        self._stage(IndexJobStage.AFTER_PUBLICATION)
        updated_extension = session.execute(
            update(compatible_index_job)
            .where(
                compatible_index_job.c.job_id == job_id,
                compatible_index_job.c.result_index_id.is_(None),
            )
            .values(
                result_index_id=index_id,
                result_document_count=build.indexed_document_count,
                result_total_token_count=build.total_token_count,
                result_document_manifest_hash=build.document_manifest_hash,
                result_lexical_manifest_hash=build.lexical_manifest_hash,
                result_dense_manifest_hash=build.dense_manifest_hash,
                result_published_at=now,
                updated_at=now,
            )
        )
        if updated_extension.rowcount != 1:
            raise BaselineIndexBuildError(
                "index_job_result_conflict",
                "index job result conflict",
                index_id=index_id,
            )
        self._stage(IndexJobStage.BEFORE_SUCCESS)
        updated_job = session.execute(
            update(control_job)
            .where(
                control_job.c.job_id == job_id,
                control_job.c.state == "running",
                control_job.c.lease_token == lease_token,
                control_job.c.lease_expires_at > now,
            )
            .values(
                state="succeeded",
                lease_token=None,
                lease_expires_at=None,
                progress_completed=1,
                error_code=None,
                error_fingerprint=None,
                updated_at=now,
                finished_at=now,
            )
        )
        if updated_job.rowcount != 1:
            raise BaselineIndexBuildError(
                "job_lease_unavailable",
                "index job lease is unavailable",
                index_id=index_id,
            )

    def _record_failure(
        self,
        *,
        group_id: str,
        job_id: str,
        lease_token: str,
        code: str,
        retryable: bool,
    ) -> bool:
        safe = (
            code
            if len(code) <= 128 and _SAFE_ID.fullmatch(code)
            else "index_build_failed"
        )
        now = self.clock()
        with self.sessions.begin() as session:
            changed = session.execute(
                update(control_job)
                .where(
                    control_job.c.group_id == group_id,
                    control_job.c.job_id == job_id,
                    control_job.c.state == "running",
                    control_job.c.lease_token == lease_token,
                    control_job.c.lease_expires_at > now,
                )
                .values(
                    state="retryable_failed" if retryable else "terminal_failed",
                    lease_token=None,
                    lease_expires_at=None,
                    error_code=safe,
                    error_fingerprint=hashlib.sha256(safe.encode()).hexdigest(),
                    updated_at=now,
                    finished_at=None if retryable else now,
                )
            )
            return changed.rowcount == 1

    def cancel(
        self,
        *,
        identity: InternalIndexWorkerIdentity,
        group_id: str,
        job_id: str,
        lease_token: str,
    ) -> None:
        identity.validate()
        now = self.clock()
        with self.sessions.begin() as session:
            changed = session.execute(
                update(control_job)
                .where(
                    control_job.c.group_id == group_id,
                    control_job.c.job_id == job_id,
                    control_job.c.state == "running",
                    control_job.c.lease_token == lease_token,
                    control_job.c.lease_expires_at > now,
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
            if changed.rowcount != 1:
                raise IndexJobError("job_lease_unavailable", retryable=True)

    def _completed_outcome(
        self, *, group_id: str, job_id: str
    ) -> IndexJobOutcome | None:
        with self.sessions() as session:
            job = (
                session.execute(
                    select(control_job).where(
                        control_job.c.group_id == group_id,
                        control_job.c.job_id == job_id,
                        control_job.c.operation == "index_build",
                    )
                )
                .mappings()
                .first()
            )
            extension = (
                session.execute(
                    select(compatible_index_job).where(
                        compatible_index_job.c.group_id == group_id,
                        compatible_index_job.c.job_id == job_id,
                    )
                )
                .mappings()
                .first()
            )
            if job is None or extension is None or job["state"] != "succeeded":
                return None
            self._validate_context(
                session,
                job=job,
                extension=extension,
                continuation_id=str(extension["continuation_job_id"]),
                group_id=group_id,
                expected_generation_id=str(extension["generation_id"]),
                lock=False,
            )
            return self._outcome_from_rows(job, extension)

    @staticmethod
    def _outcome_from_rows(
        job: Mapping[str, Any], extension: Mapping[str, Any]
    ) -> IndexJobOutcome:
        return IndexJobOutcome(
            job_id=str(job["job_id"]),
            group_id=str(job["group_id"]),
            state=str(job["state"]),
            attempt_count=int(job["attempt_count"]),
            corpus_id=str(extension["corpus_id"]),
            generation_id=str(extension["generation_id"]),
            index_id=str(extension["result_index_id"]),
            document_count=int(extension["result_document_count"]),
            document_manifest_hash=str(extension["result_document_manifest_hash"]),
            lexical_manifest_hash=str(extension["result_lexical_manifest_hash"]),
            dense_manifest_hash=str(extension["result_dense_manifest_hash"]),
            retrieval_config_fingerprint=str(extension["retrieval_config_fingerprint"]),
            embedding_fingerprint=str(extension["embedding_fingerprint"]),
        )

    def _outcome(
        self,
        *,
        group_id: str,
        job_id: str,
        result: BaselineIndexBuildResult,
        attempt_count: int,
    ) -> IndexJobOutcome:
        with self.sessions() as session:
            job = (
                session.execute(
                    select(control_job).where(control_job.c.job_id == job_id)
                )
                .mappings()
                .one()
            )
            extension = (
                session.execute(
                    select(compatible_index_job).where(
                        compatible_index_job.c.job_id == job_id
                    )
                )
                .mappings()
                .one()
            )
        if (
            job["state"] != "succeeded"
            or extension["result_index_id"] != result.index_id
            or int(job["attempt_count"]) != attempt_count
        ):
            raise IndexJobError("index_job_result_incompatible", retryable=False)
        return self._outcome_from_rows(job, extension)

    def status_extension(
        self,
        *,
        caller_user_id: str,
        group_id: str,
        job_id: str,
    ) -> Mapping[str, Any] | None:
        with self.sessions() as session:
            try:
                self.control._authorize_group(
                    session, user_id=caller_user_id, group_id=group_id
                )
            except ControlPlaneError as exc:
                raise IndexJobError(
                    exc.code, retryable=False, status_code=exc.status_code
                ) from None
            extension = (
                session.execute(
                    select(compatible_index_job).where(
                        compatible_index_job.c.group_id == group_id,
                        compatible_index_job.c.job_id == job_id,
                    )
                )
                .mappings()
                .first()
            )
            if extension is None:
                return None
            return dict(extension)

    def read_status_snapshot(
        self,
        *,
        caller_user_id: str,
        group_id: str,
        job_id: str,
    ) -> dict[str, Any]:
        """Read and reauthorize one index job without changing durable state."""

        try:
            with self.sessions() as session:
                self.control._authorize_group(
                    session, user_id=caller_user_id, group_id=group_id
                )
                job = (
                    session.execute(
                        select(control_job).where(
                            control_job.c.group_id == group_id,
                            control_job.c.job_id == job_id,
                            control_job.c.operation == "index_build",
                        )
                    )
                    .mappings()
                    .first()
                )
                extension = (
                    session.execute(
                        select(compatible_index_job).where(
                            compatible_index_job.c.group_id == group_id,
                            compatible_index_job.c.job_id == job_id,
                        )
                    )
                    .mappings()
                    .first()
                )
                if job is None or extension is None:
                    raise IndexJobError(
                        "not_found_or_forbidden",
                        retryable=False,
                        status_code=404,
                    )
                self._validate_context(
                    session,
                    job=job,
                    extension=extension,
                    continuation_id=str(extension["continuation_job_id"]),
                    group_id=group_id,
                    expected_generation_id=str(extension["generation_id"]),
                    lock=False,
                )
                status_identity = BaselineEmbeddingIdentity(
                    provider=str(extension["embedding_provider"]),
                    model=str(extension["embedding_model"]),
                    revision=str(extension["embedding_revision"]),
                    dimension=int(extension["embedding_dimension"]),
                    fingerprint=str(extension["embedding_fingerprint"]),
                )
                if (
                    extension["index_format_version"] != BASELINE_INDEX_SCHEMA_VERSION
                    or extension["tokenizer_version"] != BASELINE_TOKENIZER_VERSION
                    or extension["embedding_contract_version"]
                    != BASELINE_EMBEDDING_HTTP_CONTRACT
                    or extension["embedding_dtype"] != BASELINE_EMBEDDING_DTYPE
                    or extension["retrieval_config_fingerprint"]
                    != baseline_engine_config_fingerprint(status_identity)
                ):
                    raise IndexJobError(
                        "index_job_result_incompatible", retryable=False
                    )
                state = str(job["state"])
                result_fields = (
                    extension["result_index_id"],
                    extension["result_document_count"],
                    extension["result_total_token_count"],
                    extension["result_document_manifest_hash"],
                    extension["result_lexical_manifest_hash"],
                    extension["result_dense_manifest_hash"],
                    extension["result_published_at"],
                )
                has_any_result = any(value is not None for value in result_fields)
                has_complete_result = all(value is not None for value in result_fields)
                if (state == "succeeded" and not has_complete_result) or (
                    state != "succeeded" and has_any_result
                ):
                    raise IndexJobError(
                        "index_job_result_incompatible", retryable=False
                    )
                if (
                    state in {"retryable_failed", "terminal_failed"}
                    and not job["error_code"]
                ):
                    raise IndexJobError(
                        "index_job_result_incompatible", retryable=False
                    )
                snapshot = {**dict(job), **dict(extension)}
                snapshot["document_count"] = 0
                snapshot["result"] = None
                if state == "succeeded":
                    index_id = str(extension["result_index_id"])
                    build = session.get(RetrievalBaselineIndexBuild, index_id)
                    vector_count = session.scalar(
                        select(func.count())
                        .select_from(RetrievalBaselineIndexVector)
                        .where(RetrievalBaselineIndexVector.index_id == index_id)
                    )
                    if (
                        build is None
                        or build.generation_id != extension["generation_id"]
                        or build.corpus_manifest_hash
                        != extension["corpus_file_manifest_hash"]
                        or build.document_manifest_hash
                        != extension["result_document_manifest_hash"]
                        or build.lexical_manifest_hash
                        != extension["result_lexical_manifest_hash"]
                        or build.dense_manifest_hash
                        != extension["result_dense_manifest_hash"]
                        or build.engine_config_fingerprint
                        != extension["retrieval_config_fingerprint"]
                        or build.embedding_fingerprint
                        != extension["embedding_fingerprint"]
                        or int(vector_count or 0)
                        != int(extension["result_document_count"])
                    ):
                        raise IndexJobError(
                            "index_job_result_incompatible", retryable=False
                        )
                    document_count = int(extension["result_document_count"])
                    snapshot["document_count"] = document_count
                    snapshot["result"] = {
                        "index_publication_id": index_id,
                        "corpus_generation_id": str(extension["generation_id"]),
                        "corpus_manifest_hash": str(extension["corpus_manifest_hash"]),
                        "index_fingerprint": published_index_fingerprint(build),
                        "retrieval_config_fingerprint": str(
                            extension["retrieval_config_fingerprint"]
                        ),
                        "embedding_fingerprint": str(
                            extension["embedding_fingerprint"]
                        ),
                        "document_count": document_count,
                        "vector_count": document_count,
                    }
                return snapshot
        except ControlPlaneError as exc:
            raise IndexJobError(
                exc.code,
                retryable=False,
                status_code=exc.status_code,
            ) from None


__all__ = [
    "BASELINE_EMBEDDING_DTYPE",
    "INDEX_JOB_CONTRACT_VERSION",
    "INDEX_JOB_WORKER_SERVICE_ID",
    "PINNED_BASELINE_DIMENSION",
    "PINNED_BASELINE_MODEL",
    "BaselineCompatibleIndexJobService",
    "IndexBuildIntent",
    "IndexJobError",
    "IndexJobOutcome",
    "IndexJobStage",
    "InternalIndexWorkerIdentity",
    "parse_index_build_intent",
]
