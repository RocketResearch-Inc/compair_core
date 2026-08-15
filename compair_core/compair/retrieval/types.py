"""Versioned internal types for dependency-injected retrieval.

These dataclasses deliberately have no dependency on the API, database, task,
or legacy selector.  Phase 1B can adapt them at a single integration seam
without changing the frozen baseline algorithm.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

REQUEST_SCHEMA_VERSION = "retrieval-request.v2"
RESULT_SCHEMA_VERSION = "retrieval-result.v2"


class RetrievalStatus(str, Enum):
    """Terminal state of a retrieval request."""

    OK = "ok"
    INSUFFICIENT = "insufficient"
    ERROR = "error"


class RetrievalQueryOrigin(str, Enum):
    """How the in-memory retrieval query was supplied."""

    EXPLICIT = "explicit"
    LEGACY_DERIVED = "legacy_derived"
    ABSENT = "absent"


@dataclass(frozen=True, slots=True)
class RetrievalQueryProvenance:
    """Trace-safe metadata for a query; raw text is intentionally excluded."""

    sha256: str | None
    length: int
    origin: RetrievalQueryOrigin

    def trace_fields(self) -> dict[str, str | int | None]:
        return {
            "retrieval_query_sha256": self.sha256,
            "retrieval_query_length": self.length,
            "retrieval_query_origin": self.origin.value,
        }


def retrieval_query_provenance(
    query: str | None,
    origin: RetrievalQueryOrigin,
) -> RetrievalQueryProvenance:
    """Describe ``query`` without retaining it in trace metadata.

    Length is measured in Python characters and the digest covers its exact
    UTF-8 representation. ``absent`` is represented without a digest.
    """

    if origin is RetrievalQueryOrigin.ABSENT:
        if query is not None:
            raise ValueError("an absent retrieval query cannot contain text")
        return RetrievalQueryProvenance(sha256=None, length=0, origin=origin)
    if query is None:
        raise ValueError(f"{origin.value} retrieval query text is required")
    return RetrievalQueryProvenance(
        sha256=hashlib.sha256(query.encode("utf-8")).hexdigest(),
        length=len(query),
        origin=origin,
    )


@dataclass(frozen=True, slots=True)
class RetrievalError:
    """Machine-readable explanation for an insufficient or error result."""

    code: str
    message: str


@runtime_checkable
class DenseEmbeddingProvider(Protocol):
    """The only dense dependency required by ``baseline_v1``.

    Implementations must return one finite, fixed-width vector for every input
    text.  The baseline never substitutes another provider or hash vectors.
    """

    model: str
    revision: str
    dimension: int

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Embed ``texts`` in input order."""


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    """Versioned internal request shared by every retrieval engine.

    Query text remains available only in memory on this request. Callers must
    use ``query_provenance`` for tracing instead of serializing or logging the
    raw ``retrieval_query`` field.
    """

    request_id: str
    changed_repository: Path | None
    repository_roots: tuple[Path, ...]
    corpus_version: str
    retrieval_query: str | None = None
    retrieval_query_origin: RetrievalQueryOrigin = RetrievalQueryOrigin.ABSENT
    query_kind: str = "raw_git_diff_v1"
    corpus_complete: bool = True
    corpus_scope_key: str | None = None
    changed_repository_id: str | None = None
    schema_version: str = REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        retrieval_query_provenance(
            self.retrieval_query,
            self.retrieval_query_origin,
        )

    @property
    def query_provenance(self) -> RetrievalQueryProvenance:
        return retrieval_query_provenance(
            self.retrieval_query,
            self.retrieval_query_origin,
        )

    @property
    def has_usable_explicit_query(self) -> bool:
        return (
            self.retrieval_query_origin is RetrievalQueryOrigin.EXPLICIT
            and self.retrieval_query is not None
            and bool(self.retrieval_query.strip())
        )


@dataclass(frozen=True, slots=True)
class FileCandidate:
    """An eligible UTF-8 whole file before scoring."""

    repository: str
    relative_path: str
    content: str
    content_hash: str
    byte_size: int

    @property
    def path(self) -> str:
        return f"{self.repository}/{self.relative_path}"


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """A whole-file candidate with its complete lane provenance."""

    repository: str
    relative_path: str
    content: str
    content_hash: str
    byte_size: int
    bm25_score: float
    bm25_rank: int
    dense_score: float
    dense_rank: int
    rrf_score: float
    fused_rank: int
    document_id: str | None = None

    @property
    def path(self) -> str:
        return f"{self.repository}/{self.relative_path}"


@dataclass(frozen=True, slots=True)
class RetrievalEvidence:
    """One normalized evidence item delivered within the common budget."""

    repository: str
    relative_path: str
    content: str
    content_hash: str
    fused_rank: int
    render_truncated: bool
    bm25_score: float | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None
    rrf_score: float | None = None
    document_id: str | None = None

    @property
    def path(self) -> str:
        return f"{self.repository}/{self.relative_path}"


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Deterministic result from the pure ``baseline_v1`` engine."""

    request_id: str
    status: RetrievalStatus
    corpus_version: str
    config_fingerprint: str
    embedding_model: str
    embedding_revision: str
    embedding_dimension: int
    candidates: tuple[RetrievalCandidate, ...] = ()
    evidence: tuple[RetrievalEvidence, ...] = ()
    candidate_count: int = 0
    retrieved_count: int = 0
    filtered_count: int = 0
    duplicate_count: int = 0
    refill_count: int = 0
    evidence_characters: int = 0
    underfilled: bool = True
    error: RetrievalError | None = None
    fallback_engine: str | None = None
    engine: str = "baseline_v1"
    engine_version: str = "baseline_v1"
    corpus_id: str | None = None
    corpus_manifest_hash: str | None = None
    corpus_scope_key: str | None = None
    index_id: str | None = None
    index_version: str | None = None
    index_schema_version: str | None = None
    index_fingerprint: str | None = None
    embedding_provider: str | None = None
    embedding_fingerprint: str | None = None
    query_provenance: RetrievalQueryProvenance | None = None
    schema_version: str = RESULT_SCHEMA_VERSION
