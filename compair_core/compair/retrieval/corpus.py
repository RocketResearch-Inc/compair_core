"""Durable corpus generations and fail-closed index compatibility metadata.

Phase 2B1 deliberately stores whole-file snapshots and index provenance only.
It does not scan filesystems, tokenize, embed, rank, or invoke retrieval.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import PurePosixPath
from uuid import uuid4

from sqlalchemy import (
    DateTime,
    Engine,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    select,
    update,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    MappedAsDataclass,
    Session,
    mapped_column,
)
from sqlalchemy.schema import CreateTable

CORPUS_SCHEMA_VERSION = "retrieval-corpus.v1"
CORPUS_SNAPSHOT_SCHEMA_VERSION = "corpus-snapshot-input.v1"
TOKENIZER_VERSION_PLACEHOLDER = "baseline_v1_tokenizer_pending"


class CorpusGenerationStatus(str, Enum):
    STAGING = "staging"
    VALIDATED = "validated"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    FAILED = "failed"


class CorpusFileState(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED_UTF8 = "unsupported_utf8"
    OVERSIZED = "oversized"
    SYMLINK_REJECTED = "symlink_rejected"
    UNREADABLE = "unreadable"
    EXCLUDED = "excluded"


class CorpusFileSkipReason(str, Enum):
    NON_UTF8 = "non_utf8"
    OVERSIZED = "oversized"
    SYMLINK = "symlink"
    UNREADABLE = "unreadable"
    EXCLUDED_DIRECTORY = "excluded_directory"
    UNSUPPORTED_FILE_TYPE = "unsupported_file_type"
    PRODUCER_SKIPPED = "producer_skipped"


class CorpusIngestionStatus(str, Enum):
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"
    ACTIVE = "active"
    STALE = "stale"
    FAILED = "failed"


class IndexStateStatus(str, Enum):
    INCOMPLETE = "incomplete"
    COMPATIBLE = "compatible"
    STALE = "stale"
    INCOMPATIBLE = "incompatible"


class BaselineIndexBuildStatus(str, Enum):
    STAGING = "staging"
    VALIDATED = "validated"
    COMPATIBLE = "compatible"
    STALE = "stale"
    INCOMPATIBLE = "incompatible"
    FAILED = "failed"


class RetrievalCorpusBase(DeclarativeBase, MappedAsDataclass):
    """Separate additive metadata namespace from Core's legacy ORM tables."""


class RetrievalCorpus(RetrievalCorpusBase):
    """Stable identity and atomic active-generation pointer for one corpus."""

    __tablename__ = "retrieval_corpus"

    corpus_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, init=False, default_factory=lambda: str(uuid4())
    )
    scope_key: Mapped[str] = mapped_column(String(256), unique=True, index=True)
    changed_repository_id: Mapped[str] = mapped_column(String(256), index=True)
    source_document_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, default=None, index=True
    )
    active_generation_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, default=None, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        init=False,
        default_factory=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        init=False,
        default_factory=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class RetrievalCorpusGeneration(RetrievalCorpusBase):
    """Immutable full-snapshot generation until its lifecycle state changes."""

    __tablename__ = "retrieval_corpus_generation"
    __table_args__ = (
        UniqueConstraint(
            "corpus_id",
            "generation_version",
            name="uq_retrieval_corpus_generation_version",
        ),
    )

    generation_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, init=False, default_factory=lambda: str(uuid4())
    )
    corpus_id: Mapped[str] = mapped_column(
        ForeignKey("retrieval_corpus.corpus_id", ondelete="CASCADE"), index=True
    )
    generation_version: Mapped[str] = mapped_column(String(128))
    expected_repository_count: Mapped[int] = mapped_column(Integer)
    expected_file_count: Mapped[int] = mapped_column(Integer)
    source_revision: Mapped[str | None] = mapped_column(
        String(256), nullable=True, default=None
    )
    status: Mapped[str] = mapped_column(String(24), default="staging", index=True)
    manifest_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, default=None
    )
    failure_code: Mapped[str | None] = mapped_column(
        String(64), nullable=True, default=None
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        init=False,
        default_factory=lambda: datetime.now(timezone.utc),
    )
    validated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, init=False, default=None
    )
    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, init=False, default=None
    )


class RetrievalCorpusIngestion(RetrievalCorpusBase):
    """Trusted snapshot provenance and canonical metadata-only manifest."""

    __tablename__ = "retrieval_corpus_ingestion"

    generation_id: Mapped[str] = mapped_column(
        ForeignKey("retrieval_corpus_generation.generation_id", ondelete="CASCADE"),
        primary_key=True,
    )
    snapshot_schema_version: Mapped[str] = mapped_column(String(64))
    ingestion_source: Mapped[str] = mapped_column(String(64))
    producer_id: Mapped[str] = mapped_column(String(128))
    canonical_manifest_hash: Mapped[str] = mapped_column(String(64))
    canonical_manifest_json: Mapped[str] = mapped_column(Text)
    repository_count: Mapped[int] = mapped_column(Integer)
    file_count: Mapped[int] = mapped_column(Integer)
    snapshot_id: Mapped[str | None] = mapped_column(
        String(256), nullable=True, default=None
    )
    producer_version: Mapped[str | None] = mapped_column(
        String(128), nullable=True, default=None
    )
    source_manifest_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, default=None
    )
    status: Mapped[str] = mapped_column(
        String(24), default=CorpusIngestionStatus.INCOMPLETE.value, index=True
    )
    failure_code: Mapped[str | None] = mapped_column(
        String(64), nullable=True, default=None
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        init=False,
        default_factory=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        init=False,
        default_factory=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class RetrievalCorpusFile(RetrievalCorpusBase):
    """One immutable file record in a corpus generation."""

    __tablename__ = "retrieval_corpus_file"
    __table_args__ = (
        UniqueConstraint(
            "generation_id",
            "repository_id",
            "relative_path",
            name="uq_retrieval_corpus_file_path",
        ),
        Index(
            "ix_retrieval_corpus_file_order",
            "generation_id",
            "repository_name",
            "relative_path",
            "repository_id",
        ),
    )

    file_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, init=False, default_factory=lambda: str(uuid4())
    )
    generation_id: Mapped[str] = mapped_column(
        ForeignKey("retrieval_corpus_generation.generation_id", ondelete="CASCADE"),
        index=True,
    )
    repository_id: Mapped[str] = mapped_column(String(256), index=True)
    repository_name: Mapped[str] = mapped_column(String(256))
    relative_path: Mapped[str] = mapped_column(String(1024))
    file_state: Mapped[str] = mapped_column(String(32))
    content_hash: Mapped[str] = mapped_column(String(64))
    byte_size: Mapped[int] = mapped_column(Integer)
    document_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, default=None, index=True
    )
    content: Mapped[str | None] = mapped_column(Text, nullable=True, default=None)
    source_snapshot_id: Mapped[str | None] = mapped_column(
        String(256), nullable=True, default=None
    )


class RetrievalIndexState(RetrievalCorpusBase):
    """Persistent compatibility metadata; Phase 2B1 builds no ranking index."""

    __tablename__ = "retrieval_index_state"

    generation_id: Mapped[str] = mapped_column(
        ForeignKey("retrieval_corpus_generation.generation_id", ondelete="CASCADE"),
        primary_key=True,
    )
    status: Mapped[str] = mapped_column(String(24), default="incomplete", index=True)
    corpus_manifest_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, default=None
    )
    tokenizer_version: Mapped[str | None] = mapped_column(
        String(128), nullable=True, default=None
    )
    embedding_provider: Mapped[str | None] = mapped_column(
        String(128), nullable=True, default=None
    )
    embedding_model: Mapped[str | None] = mapped_column(
        String(256), nullable=True, default=None
    )
    embedding_revision: Mapped[str | None] = mapped_column(
        String(256), nullable=True, default=None
    )
    embedding_dimension: Mapped[int | None] = mapped_column(
        Integer, nullable=True, default=None
    )
    embedding_fingerprint: Mapped[str | None] = mapped_column(
        String(64), nullable=True, default=None
    )
    engine_config_fingerprint: Mapped[str | None] = mapped_column(
        String(64), nullable=True, default=None
    )
    indexed_file_count: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        init=False,
        default_factory=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class RetrievalBaselineIndexBuild(RetrievalCorpusBase):
    """One immutable attempted baseline index build for a corpus generation."""

    __tablename__ = "retrieval_baseline_index_build"
    __table_args__ = (
        UniqueConstraint(
            "generation_id",
            "index_version",
            name="uq_retrieval_baseline_index_build_version",
        ),
    )

    index_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, init=False, default_factory=lambda: str(uuid4())
    )
    generation_id: Mapped[str] = mapped_column(
        ForeignKey("retrieval_corpus_generation.generation_id", ondelete="CASCADE"),
        index=True,
    )
    index_version: Mapped[str] = mapped_column(String(128))
    index_schema_version: Mapped[str] = mapped_column(String(64))
    document_format_version: Mapped[str] = mapped_column(String(64))
    corpus_manifest_hash: Mapped[str] = mapped_column(String(64))
    tokenizer_version: Mapped[str] = mapped_column(String(128))
    embedding_provider: Mapped[str] = mapped_column(String(128))
    embedding_model: Mapped[str] = mapped_column(String(256))
    embedding_revision: Mapped[str] = mapped_column(String(256))
    embedding_dimension: Mapped[int] = mapped_column(Integer)
    embedding_fingerprint: Mapped[str] = mapped_column(String(64))
    engine_config_fingerprint: Mapped[str] = mapped_column(String(64))
    expected_document_count: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(
        String(24), default=BaselineIndexBuildStatus.STAGING.value, index=True
    )
    indexed_document_count: Mapped[int] = mapped_column(Integer, default=0)
    total_token_count: Mapped[int] = mapped_column(Integer, default=0)
    document_manifest_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, default=None
    )
    lexical_manifest_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, default=None
    )
    dense_manifest_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, default=None
    )
    failure_code: Mapped[str | None] = mapped_column(
        String(64), nullable=True, default=None
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        init=False,
        default_factory=lambda: datetime.now(timezone.utc),
    )
    validated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, init=False, default=None
    )
    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, init=False, default=None
    )


class RetrievalBaselineIndexDocument(RetrievalCorpusBase):
    """One frozen whole-file ranking document in an index build."""

    __tablename__ = "retrieval_baseline_index_document"
    __table_args__ = (
        UniqueConstraint(
            "index_id",
            "ordinal",
            name="uq_retrieval_baseline_index_document_order",
        ),
        UniqueConstraint(
            "index_id",
            "corpus_file_id",
            name="uq_retrieval_baseline_index_document_file",
        ),
        Index(
            "ix_retrieval_baseline_index_document_path",
            "index_id",
            "repository_name",
            "relative_path",
            "repository_id",
        ),
    )

    index_document_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, init=False, default_factory=lambda: str(uuid4())
    )
    index_id: Mapped[str] = mapped_column(
        ForeignKey("retrieval_baseline_index_build.index_id", ondelete="CASCADE"),
        index=True,
    )
    corpus_file_id: Mapped[str] = mapped_column(
        ForeignKey("retrieval_corpus_file.file_id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    repository_id: Mapped[str] = mapped_column(String(256))
    repository_name: Mapped[str] = mapped_column(String(256))
    relative_path: Mapped[str] = mapped_column(String(1024))
    ranking_text: Mapped[str] = mapped_column(Text)
    source_content_hash: Mapped[str] = mapped_column(String(64))
    indexed_document_hash: Mapped[str] = mapped_column(String(64))
    token_count: Mapped[int] = mapped_column(Integer)


class RetrievalBaselineIndexTerm(RetrievalCorpusBase):
    """Portable exact per-document term frequency for baseline BM25."""

    __tablename__ = "retrieval_baseline_index_term"
    __table_args__ = (
        UniqueConstraint(
            "index_document_id",
            "term_hash",
            name="uq_retrieval_baseline_index_document_term_hash",
        ),
        Index(
            "ix_retrieval_baseline_index_term_lookup",
            "index_id",
            "term_hash",
        ),
    )

    term_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, init=False, default_factory=lambda: str(uuid4())
    )
    index_id: Mapped[str] = mapped_column(
        ForeignKey("retrieval_baseline_index_build.index_id", ondelete="CASCADE"),
        index=True,
    )
    index_document_id: Mapped[str] = mapped_column(
        ForeignKey(
            "retrieval_baseline_index_document.index_document_id",
            ondelete="CASCADE",
        ),
        index=True,
    )
    term_hash: Mapped[str] = mapped_column(String(64))
    term: Mapped[str] = mapped_column(Text)
    term_frequency: Mapped[int] = mapped_column(Integer)


class RetrievalBaselineIndexVector(RetrievalCorpusBase):
    """Portable little-endian float32 dense vector for one ranking document."""

    __tablename__ = "retrieval_baseline_index_vector"

    index_document_id: Mapped[str] = mapped_column(
        ForeignKey(
            "retrieval_baseline_index_document.index_document_id",
            ondelete="CASCADE",
        ),
        primary_key=True,
    )
    index_id: Mapped[str] = mapped_column(
        ForeignKey("retrieval_baseline_index_build.index_id", ondelete="CASCADE"),
        index=True,
    )
    dimension: Mapped[int] = mapped_column(Integer)
    vector_bytes: Mapped[bytes] = mapped_column(LargeBinary)
    vector_hash: Mapped[str] = mapped_column(String(64))


class RetrievalBaselineIndexPublication(RetrievalCorpusBase):
    """Atomic pointer to the last published compatible build for one corpus."""

    __tablename__ = "retrieval_baseline_index_publication"

    corpus_id: Mapped[str] = mapped_column(
        ForeignKey("retrieval_corpus.corpus_id", ondelete="CASCADE"), primary_key=True
    )
    index_id: Mapped[str | None] = mapped_column(
        ForeignKey("retrieval_baseline_index_build.index_id", ondelete="SET NULL"),
        nullable=True,
        unique=True,
    )
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        init=False,
        default_factory=lambda: datetime.now(timezone.utc),
    )


@dataclass(frozen=True, slots=True)
class CorpusFileInput:
    repository_id: str
    repository_name: str
    relative_path: str
    file_state: CorpusFileState
    content_hash: str
    byte_size: int
    document_id: str | None = None
    content: str | None = None
    source_snapshot_id: str | None = None
    repository_revision: str | None = None
    document_revision: str | None = None
    skip_reason: CorpusFileSkipReason | None = None
    derived_from_symlink: bool = False

    @classmethod
    def supported_text(
        cls,
        *,
        repository_id: str,
        repository_name: str,
        relative_path: str,
        content: str,
        document_id: str | None = None,
        source_snapshot_id: str | None = None,
    ) -> CorpusFileInput:
        raw = content.encode("utf-8")
        return cls(
            repository_id=repository_id,
            repository_name=repository_name,
            relative_path=relative_path,
            file_state=CorpusFileState.SUPPORTED,
            content_hash=hashlib.sha256(raw).hexdigest(),
            byte_size=len(raw),
            document_id=document_id,
            content=content,
            source_snapshot_id=source_snapshot_id,
        )


@dataclass(frozen=True, slots=True)
class GenerationValidation:
    complete: bool
    generation_id: str
    manifest_hash: str | None
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class IndexRequirements:
    tokenizer_version: str
    embedding_provider: str
    embedding_model: str
    embedding_revision: str
    embedding_dimension: int
    embedding_fingerprint: str
    engine_config_fingerprint: str


@dataclass(frozen=True, slots=True)
class CorpusReadiness:
    ready: bool
    code: str
    corpus_id: str | None = None
    generation_id: str | None = None
    generation_version: str | None = None
    changed_repository_id: str | None = None


RETRIEVAL_CORPUS_TABLES = (
    RetrievalCorpus.__table__,
    RetrievalCorpusGeneration.__table__,
    RetrievalCorpusIngestion.__table__,
    RetrievalCorpusFile.__table__,
    RetrievalIndexState.__table__,
    RetrievalBaselineIndexBuild.__table__,
    RetrievalBaselineIndexDocument.__table__,
    RetrievalBaselineIndexTerm.__table__,
    RetrievalBaselineIndexVector.__table__,
    RetrievalBaselineIndexPublication.__table__,
)


_RETRIEVAL_CORPUS_DDL = {
    "postgresql": tuple(
        str(CreateTable(table).compile(dialect=postgresql.dialect()))
        for table in RETRIEVAL_CORPUS_TABLES
    ),
    "sqlite": tuple(
        str(CreateTable(table).compile(dialect=sqlite.dialect()))
        for table in RETRIEVAL_CORPUS_TABLES
    ),
}


def ensure_retrieval_corpus_schema(engine: Engine) -> None:
    """Create missing Phase 2B1 tables without altering or rebuilding tables."""

    RetrievalCorpusBase.metadata.create_all(
        bind=engine,
        tables=list(RETRIEVAL_CORPUS_TABLES),
        checkfirst=True,
    )


def compile_retrieval_corpus_ddl(dialect_name: str) -> tuple[str, ...]:
    """Compile additive table DDL for migration review on supported backends."""

    try:
        return _RETRIEVAL_CORPUS_DDL[dialect_name]
    except KeyError as exc:
        raise ValueError(f"unsupported retrieval corpus dialect: {dialect_name}") from exc


def normalize_relative_path(value: str) -> str:
    """Return one safe normalized POSIX relative path."""

    raw = value or ""
    if not raw or "\x00" in raw or "\\" in raw or raw.startswith("/"):
        raise ValueError("relative path must be a non-empty repository-relative path")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("relative path contains an unsafe component")
    normalized = path.as_posix()
    if normalized != raw or normalized == "." or ":" in path.parts[0]:
        raise ValueError("relative path must not contain a drive or empty path")
    return normalized


_FILE_SKIP_REASONS = {
    CorpusFileState.UNSUPPORTED_UTF8: {CorpusFileSkipReason.NON_UTF8},
    CorpusFileState.OVERSIZED: {CorpusFileSkipReason.OVERSIZED},
    CorpusFileState.SYMLINK_REJECTED: {CorpusFileSkipReason.SYMLINK},
    CorpusFileState.UNREADABLE: {CorpusFileSkipReason.UNREADABLE},
    CorpusFileState.EXCLUDED: {
        CorpusFileSkipReason.EXCLUDED_DIRECTORY,
        CorpusFileSkipReason.UNSUPPORTED_FILE_TYPE,
        CorpusFileSkipReason.PRODUCER_SKIPPED,
    },
}


def validate_corpus_file_input(item: CorpusFileInput) -> CorpusFileInput:
    """Validate and normalize one trusted whole-file source record."""

    repository_id = item.repository_id.strip()
    repository_name = item.repository_name.strip()
    if not repository_id or not repository_name:
        raise ValueError("repository identity and name are required")
    if "/" in repository_name or "\\" in repository_name:
        raise ValueError("repository name must be a single path component")
    relative_path = normalize_relative_path(item.relative_path)
    if item.byte_size < 0:
        raise ValueError("file byte size must be non-negative")
    if item.derived_from_symlink:
        raise ValueError("content derived from a symlink is not trusted")
    content_hash = item.content_hash.lower()
    if len(content_hash) != 64 or any(c not in "0123456789abcdef" for c in content_hash):
        raise ValueError("file content hash must be a SHA-256 hex digest")
    if item.file_state is CorpusFileState.SUPPORTED:
        if item.skip_reason is not None:
            raise ValueError("supported files must not have a skip reason")
        if item.content is None:
            raise ValueError("supported files require UTF-8 text content")
        raw = item.content.encode("utf-8")
        if len(raw) != item.byte_size:
            raise ValueError("supported file byte size does not match content")
        if hashlib.sha256(raw).hexdigest() != content_hash:
            raise ValueError("supported file content hash does not match content")
    else:
        if item.content is not None:
            raise ValueError("unsupported file states must not persist text content")
        allowed_reasons = _FILE_SKIP_REASONS.get(item.file_state, set())
        if item.skip_reason not in allowed_reasons:
            raise ValueError("unsupported file state requires a compatible skip reason")
    return CorpusFileInput(
        repository_id=repository_id,
        repository_name=repository_name,
        relative_path=relative_path,
        file_state=item.file_state,
        content_hash=content_hash,
        byte_size=item.byte_size,
        document_id=item.document_id,
        content=item.content,
        source_snapshot_id=item.source_snapshot_id,
        repository_revision=item.repository_revision,
        document_revision=item.document_revision,
        skip_reason=item.skip_reason,
        derived_from_symlink=False,
    )


def _validate_file_input(item: CorpusFileInput) -> CorpusFileInput:
    """Compatibility wrapper for the Phase 2B1 internal helper."""

    return validate_corpus_file_input(item)


def _manifest_hash(files: Iterable[RetrievalCorpusFile]) -> str:
    rows = [
        {
            "byte_size": row.byte_size,
            "content_hash": row.content_hash,
            "document_id": row.document_id,
            "file_state": row.file_state,
            "relative_path": row.relative_path,
            "repository_id": row.repository_id,
            "repository_name": row.repository_name,
            "source_snapshot_id": row.source_snapshot_id,
        }
        for row in files
    ]
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _ingestion_repository_counts(
    ingestion: RetrievalCorpusIngestion | None,
) -> Mapping[str, int] | None:
    if ingestion is None:
        return None
    try:
        payload = json.loads(ingestion.canonical_manifest_json)
        repositories = payload["sibling_repositories"]
        return {
            str(repository["repository_id"]): int(
                repository["expected_file_count"]
            )
            for repository in repositories
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


class CorpusLifecycle:
    """Transactional lifecycle operations over immutable corpus snapshots."""

    @staticmethod
    def get_or_create_corpus(
        session: Session,
        *,
        scope_key: str,
        changed_repository_id: str,
        source_document_id: str | None = None,
    ) -> RetrievalCorpus:
        normalized_scope = scope_key.strip()
        normalized_repository = changed_repository_id.strip()
        if not normalized_scope or not normalized_repository:
            raise ValueError("scope key and changed repository identity are required")
        corpus = session.scalar(
            select(RetrievalCorpus).where(RetrievalCorpus.scope_key == normalized_scope)
        )
        if corpus is not None:
            if (
                corpus.changed_repository_id != normalized_repository
                or corpus.source_document_id != source_document_id
            ):
                raise ValueError("corpus scope identity is immutable")
            return corpus
        corpus = RetrievalCorpus(
            scope_key=normalized_scope,
            changed_repository_id=normalized_repository,
            source_document_id=source_document_id,
        )
        session.add(corpus)
        session.flush()
        return corpus

    @staticmethod
    def stage_generation(
        session: Session,
        *,
        corpus: RetrievalCorpus,
        generation_version: str,
        files: Iterable[CorpusFileInput],
        expected_repository_count: int,
        expected_file_count: int,
        source_revision: str | None = None,
    ) -> RetrievalCorpusGeneration:
        if expected_repository_count < 0 or expected_file_count < 0:
            raise ValueError("expected corpus counts must be non-negative")
        version = generation_version.strip()
        if not version:
            raise ValueError("generation version is required")
        normalized_files = [_validate_file_input(item) for item in files]
        keys = [
            (item.repository_id, item.relative_path) for item in normalized_files
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("generation contains duplicate repository paths")

        generation = RetrievalCorpusGeneration(
            corpus_id=corpus.corpus_id,
            generation_version=version,
            expected_repository_count=expected_repository_count,
            expected_file_count=expected_file_count,
            source_revision=source_revision,
        )
        session.add(generation)
        session.flush()
        for item in sorted(
            normalized_files,
            key=lambda row: (
                row.repository_name,
                row.relative_path,
                row.repository_id,
            ),
        ):
            session.add(
                RetrievalCorpusFile(
                    generation_id=generation.generation_id,
                    repository_id=item.repository_id,
                    repository_name=item.repository_name,
                    relative_path=item.relative_path,
                    file_state=item.file_state.value,
                    content_hash=item.content_hash,
                    byte_size=item.byte_size,
                    document_id=item.document_id,
                    content=item.content,
                    source_snapshot_id=(
                        item.source_snapshot_id or item.repository_revision
                    ),
                )
            )
        session.add(
            RetrievalIndexState(
                generation_id=generation.generation_id,
                tokenizer_version=TOKENIZER_VERSION_PLACEHOLDER,
            )
        )
        session.flush()
        return generation

    @staticmethod
    def ordered_files(
        session: Session,
        generation_id: str,
    ) -> tuple[RetrievalCorpusFile, ...]:
        return tuple(
            session.scalars(
                select(RetrievalCorpusFile)
                .where(RetrievalCorpusFile.generation_id == generation_id)
                .order_by(
                    RetrievalCorpusFile.repository_name,
                    RetrievalCorpusFile.relative_path,
                    RetrievalCorpusFile.repository_id,
                )
            )
        )

    @classmethod
    def validate_generation(
        cls,
        session: Session,
        generation_id: str,
    ) -> GenerationValidation:
        generation = session.get(RetrievalCorpusGeneration, generation_id)
        if generation is None:
            raise ValueError("unknown corpus generation")
        if generation.status != CorpusGenerationStatus.STAGING.value:
            raise ValueError("only a staging generation can be validated")
        files = cls.ordered_files(session, generation_id)
        ingestion = session.get(RetrievalCorpusIngestion, generation_id)
        error_code = cls._generation_count_error(
            generation,
            files,
            repository_counts=_ingestion_repository_counts(ingestion),
        )

        if error_code is not None:
            generation.status = CorpusGenerationStatus.FAILED.value
            generation.failure_code = error_code
            if ingestion is not None:
                ingestion.status = CorpusIngestionStatus.FAILED.value
                ingestion.failure_code = error_code
                ingestion.updated_at = datetime.now(timezone.utc)
            return GenerationValidation(False, generation_id, None, error_code)

        manifest_hash = _manifest_hash(files)
        generation.manifest_hash = manifest_hash
        generation.status = CorpusGenerationStatus.VALIDATED.value
        generation.validated_at = datetime.now(timezone.utc)
        generation.failure_code = None
        index_state = session.get(RetrievalIndexState, generation_id)
        if index_state is None:
            index_state = RetrievalIndexState(
                generation_id=generation_id,
                tokenizer_version=TOKENIZER_VERSION_PLACEHOLDER,
            )
            session.add(index_state)
        index_state.corpus_manifest_hash = manifest_hash
        if ingestion is not None:
            ingestion.status = CorpusIngestionStatus.COMPLETE.value
            ingestion.failure_code = None
            ingestion.updated_at = datetime.now(timezone.utc)
        return GenerationValidation(True, generation_id, manifest_hash)

    @staticmethod
    def _generation_count_error(
        generation: RetrievalCorpusGeneration,
        files: tuple[RetrievalCorpusFile, ...],
        *,
        repository_counts: Mapping[str, int] | None = None,
    ) -> str | None:
        if len(files) != generation.expected_file_count:
            return "file_count_mismatch"
        if repository_counts is not None:
            if len(repository_counts) != generation.expected_repository_count:
                return "repository_count_mismatch"
            actual_counts = Counter(row.repository_id for row in files)
            if actual_counts != Counter(repository_counts):
                return "repository_file_count_mismatch"
            return None
        if (
            len({row.repository_id for row in files})
            != generation.expected_repository_count
        ):
            return "repository_count_mismatch"
        return None

    @classmethod
    def generation_integrity_error(
        cls,
        session: Session,
        generation_id: str,
    ) -> str | None:
        """Return a fail-closed code when persisted corpus artifacts changed."""

        generation = session.get(RetrievalCorpusGeneration, generation_id)
        if generation is None:
            return "corpus_generation_absent"
        files = cls.ordered_files(session, generation_id)
        ingestion = session.get(RetrievalCorpusIngestion, generation_id)
        count_error = cls._generation_count_error(
            generation,
            files,
            repository_counts=_ingestion_repository_counts(ingestion),
        )
        if count_error is not None:
            return count_error
        if generation.manifest_hash is None or _manifest_hash(files) != generation.manifest_hash:
            return "corpus_manifest_mismatch"
        if ingestion is not None and (
            hashlib.sha256(
                ingestion.canonical_manifest_json.encode("utf-8")
            ).hexdigest()
            != ingestion.canonical_manifest_hash
        ):
            return "corpus_ingestion_manifest_mismatch"
        return None

    @staticmethod
    def activate_generation(session: Session, generation_id: str) -> None:
        generation = session.get(RetrievalCorpusGeneration, generation_id)
        if generation is None:
            raise ValueError("unknown corpus generation")
        if generation.status != CorpusGenerationStatus.VALIDATED.value:
            raise ValueError("only a complete validated generation can be activated")
        files = CorpusLifecycle.ordered_files(session, generation_id)
        ingestion = session.get(RetrievalCorpusIngestion, generation_id)
        if (
            CorpusLifecycle._generation_count_error(
                generation,
                files,
                repository_counts=_ingestion_repository_counts(ingestion),
            )
            is not None
            or generation.manifest_hash is None
            or _manifest_hash(files) != generation.manifest_hash
        ):
            raise ValueError("validated generation is no longer complete")

        statement = select(RetrievalCorpus).where(
            RetrievalCorpus.corpus_id == generation.corpus_id
        )
        if session.get_bind().dialect.name == "postgresql":
            statement = statement.with_for_update()
        corpus = session.scalar(statement)
        if corpus is None:
            raise ValueError("generation corpus does not exist")

        prior_active_ids = tuple(
            session.scalars(
                select(RetrievalCorpusGeneration.generation_id).where(
                    RetrievalCorpusGeneration.corpus_id == corpus.corpus_id,
                    RetrievalCorpusGeneration.generation_id != generation_id,
                    RetrievalCorpusGeneration.status
                    == CorpusGenerationStatus.ACTIVE.value,
                )
            )
        )
        session.execute(
            update(RetrievalCorpusGeneration)
            .where(
                RetrievalCorpusGeneration.corpus_id == corpus.corpus_id,
                RetrievalCorpusGeneration.generation_id != generation_id,
                RetrievalCorpusGeneration.status
                == CorpusGenerationStatus.ACTIVE.value,
            )
            .values(status=CorpusGenerationStatus.SUPERSEDED.value)
        )
        if prior_active_ids:
            session.execute(
                update(RetrievalCorpusIngestion)
                .where(
                    RetrievalCorpusIngestion.generation_id.in_(prior_active_ids)
                )
                .values(
                    status=CorpusIngestionStatus.STALE.value,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            session.execute(
                update(RetrievalIndexState)
                .where(
                    RetrievalIndexState.generation_id.in_(prior_active_ids),
                    RetrievalIndexState.status == IndexStateStatus.COMPATIBLE.value,
                )
                .values(
                    status=IndexStateStatus.STALE.value,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            session.execute(
                update(RetrievalBaselineIndexBuild)
                .where(
                    RetrievalBaselineIndexBuild.generation_id.in_(prior_active_ids),
                    RetrievalBaselineIndexBuild.status
                    == BaselineIndexBuildStatus.COMPATIBLE.value,
                )
                .values(status=BaselineIndexBuildStatus.STALE.value)
            )

        now = datetime.now(timezone.utc)
        generation.status = CorpusGenerationStatus.ACTIVE.value
        generation.activated_at = now
        corpus.active_generation_id = generation.generation_id
        corpus.updated_at = now
        if ingestion is not None:
            ingestion.status = CorpusIngestionStatus.ACTIVE.value
            ingestion.failure_code = None
            ingestion.updated_at = now

    @staticmethod
    def record_index_state(
        session: Session,
        generation_id: str,
        *,
        status: IndexStateStatus,
        tokenizer_version: str | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        embedding_revision: str | None = None,
        embedding_dimension: int | None = None,
        embedding_fingerprint: str | None = None,
        engine_config_fingerprint: str | None = None,
        indexed_file_count: int = 0,
    ) -> RetrievalIndexState:
        state = session.get(RetrievalIndexState, generation_id)
        if state is None:
            raise ValueError("index state requires an existing corpus generation")
        if embedding_dimension is not None and embedding_dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        if indexed_file_count < 0:
            raise ValueError("indexed file count must be non-negative")
        state.status = status.value
        state.tokenizer_version = tokenizer_version
        state.embedding_provider = embedding_provider
        state.embedding_model = embedding_model
        state.embedding_revision = embedding_revision
        state.embedding_dimension = embedding_dimension
        state.embedding_fingerprint = embedding_fingerprint
        state.engine_config_fingerprint = engine_config_fingerprint
        state.indexed_file_count = indexed_file_count
        state.updated_at = datetime.now(timezone.utc)
        return state

    @classmethod
    def assess_active_corpus(
        cls,
        session: Session,
        *,
        scope_key: str,
        requirements: IndexRequirements,
    ) -> CorpusReadiness:
        corpus = session.scalar(
            select(RetrievalCorpus).where(RetrievalCorpus.scope_key == scope_key)
        )
        if corpus is None or not corpus.active_generation_id:
            return CorpusReadiness(False, "active_corpus_absent")
        generation = session.get(
            RetrievalCorpusGeneration, corpus.active_generation_id
        )
        common = {
            "corpus_id": corpus.corpus_id,
            "generation_id": corpus.active_generation_id,
            "changed_repository_id": corpus.changed_repository_id,
        }
        if generation is None or generation.status != CorpusGenerationStatus.ACTIVE.value:
            return CorpusReadiness(False, "active_generation_inconsistent", **common)
        common["generation_version"] = generation.generation_version
        ingestion = session.get(RetrievalCorpusIngestion, generation.generation_id)
        if ingestion is None:
            return CorpusReadiness(
                False,
                "corpus_ingestion_provenance_absent",
                **common,
            )
        if ingestion.status != CorpusIngestionStatus.ACTIVE.value:
            return CorpusReadiness(False, "corpus_ingestion_not_active", **common)
        if (
            ingestion.snapshot_schema_version != CORPUS_SNAPSHOT_SCHEMA_VERSION
            or ingestion.repository_count != generation.expected_repository_count
            or ingestion.file_count != generation.expected_file_count
            or hashlib.sha256(
                ingestion.canonical_manifest_json.encode("utf-8")
            ).hexdigest()
            != ingestion.canonical_manifest_hash
        ):
            return CorpusReadiness(False, "corpus_ingestion_incomplete", **common)
        files = cls.ordered_files(session, generation.generation_id)
        if (
            cls._generation_count_error(
                generation,
                files,
                repository_counts=_ingestion_repository_counts(ingestion),
            )
            is not None
            or generation.manifest_hash is None
            or _manifest_hash(files) != generation.manifest_hash
        ):
            return CorpusReadiness(False, "active_corpus_incomplete", **common)

        supported_count = sum(
            row.file_state == CorpusFileState.SUPPORTED.value for row in files
        )
        if supported_count == 0:
            return CorpusReadiness(False, "eligible_corpus_empty", **common)
        state = session.get(RetrievalIndexState, generation.generation_id)
        if state is None:
            return CorpusReadiness(False, "index_state_absent", **common)
        if state.status != IndexStateStatus.COMPATIBLE.value:
            return CorpusReadiness(False, f"index_{state.status}", **common)
        if state.corpus_manifest_hash != generation.manifest_hash:
            return CorpusReadiness(False, "index_corpus_stale", **common)
        if state.indexed_file_count != supported_count:
            return CorpusReadiness(False, "index_file_count_mismatch", **common)
        comparisons = (
            (state.tokenizer_version, requirements.tokenizer_version, "index_tokenizer_mismatch"),
            (state.embedding_provider, requirements.embedding_provider, "index_provider_mismatch"),
            (state.embedding_model, requirements.embedding_model, "index_model_mismatch"),
            (state.embedding_revision, requirements.embedding_revision, "index_revision_mismatch"),
            (state.embedding_dimension, requirements.embedding_dimension, "index_dimension_mismatch"),
            (state.embedding_fingerprint, requirements.embedding_fingerprint, "index_embedding_fingerprint_mismatch"),
            (state.engine_config_fingerprint, requirements.engine_config_fingerprint, "index_config_mismatch"),
        )
        for actual, expected, code in comparisons:
            if actual != expected:
                return CorpusReadiness(False, code, **common)
        return CorpusReadiness(True, "ready", **common)
