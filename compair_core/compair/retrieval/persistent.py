"""Read-only baseline_v1 retrieval over one compatible published index.

This adapter never scans a filesystem, writes References, invokes generation,
or calls the legacy selector. The explicit retrieval query remains in memory.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sqlalchemy import select

from .baseline import (
    BM25_B,
    BM25_K1,
    MAX_EVIDENCE_ITEMS,
    RANKING_CONTENT_CHARACTERS,
    RETRIEVAL_LIMIT,
    frozen_tokens,
    normalize_retrieved_candidates,
    reciprocal_rank_fusion,
)
from .corpus import (
    BaselineIndexBuildStatus,
    CorpusGenerationStatus,
    CorpusIngestionStatus,
    IndexStateStatus,
    RetrievalBaselineIndexBuild,
    RetrievalBaselineIndexDocument,
    RetrievalBaselineIndexPublication,
    RetrievalBaselineIndexTerm,
    RetrievalBaselineIndexVector,
    RetrievalCorpus,
    RetrievalCorpusFile,
    RetrievalCorpusGeneration,
    RetrievalCorpusIngestion,
    RetrievalIndexState,
)
from .indexing import (
    BaselineEmbeddingIdentity,
    BaselineIndexEmbeddingProvider,
    assess_baseline_index,
    baseline_embedding_identity_from_build,
    embedding_provider_mismatch_code,
)
from .types import (
    REQUEST_SCHEMA_VERSION,
    RetrievalCandidate,
    RetrievalError,
    RetrievalEvidence,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStatus,
)

PERSISTENT_BASELINE_ENGINE_VERSION = "baseline_v1.persistent.v1"

EvidenceFilter = Callable[[RetrievalCandidate], bool]


@dataclass(frozen=True, slots=True)
class _PersistentDocument:
    index_document_id: str
    corpus_file_id: str
    repository_id: str
    repository_name: str
    relative_path: str
    content: str
    content_hash: str
    byte_size: int
    token_count: int
    term_frequencies: Mapping[str, int]
    vector_bytes: bytes

    @property
    def path(self) -> str:
        return f"{self.repository_name}/{self.relative_path}"


@dataclass(frozen=True, slots=True)
class _PublicationSnapshot:
    corpus_id: str
    corpus_scope_key: str
    changed_repository_id: str
    generation_id: str
    generation_version: str
    corpus_manifest_hash: str
    index_id: str
    index_version: str
    index_schema_version: str
    index_fingerprint: str
    config_fingerprint: str
    embedding: BaselineEmbeddingIdentity
    documents: tuple[_PersistentDocument, ...]


class _PersistentResolutionError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: RetrievalStatus,
        snapshot: _PublicationSnapshot | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.snapshot = snapshot


def published_index_fingerprint(build: RetrievalBaselineIndexBuild) -> str:
    """Identify the immutable published artifacts independently of row IDs."""

    payload = {
        "config_fingerprint": build.engine_config_fingerprint,
        "corpus_manifest_hash": build.corpus_manifest_hash,
        "dense_manifest_hash": build.dense_manifest_hash,
        "document_manifest_hash": build.document_manifest_hash,
        "embedding_fingerprint": build.embedding_fingerprint,
        "index_schema_version": build.index_schema_version,
        "index_version": build.index_version,
        "lexical_manifest_hash": build.lexical_manifest_hash,
        "tokenizer_version": build.tokenizer_version,
    }
    if any(value is None for value in payload.values()):
        raise ValueError("published index artifact manifests are incomplete")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def bm25_scores_from_persisted_statistics(
    query: str,
    document_lengths: Sequence[int],
    document_term_frequencies: Sequence[Mapping[str, int]],
) -> list[float]:
    """Compute exact comparator BM25 from durable document statistics."""

    if len(document_lengths) != len(document_term_frequencies):
        raise ValueError("document lengths and term-frequency rows differ")
    if any(length < 0 for length in document_lengths):
        raise ValueError("document token lengths must be non-negative")
    if any(
        frequency <= 0
        for terms in document_term_frequencies
        for frequency in terms.values()
    ):
        raise ValueError("persisted term frequencies must be positive")

    query_counts = Counter(frozen_tokens(query))
    document_count = len(document_lengths)
    average_length = sum(document_lengths) / max(1, document_count)
    document_frequency: Counter[str] = Counter()
    for terms in document_term_frequencies:
        document_frequency.update(terms.keys())

    scores: list[float] = []
    for length, terms in zip(document_lengths, document_term_frequencies):
        score = 0.0
        normalization = BM25_K1 * (
            1.0 - BM25_B + BM25_B * length / max(1.0, average_length)
        )
        for term, query_frequency in query_counts.items():
            frequency = terms.get(term, 0)
            if not frequency:
                continue
            frequency_in_documents = document_frequency[term]
            inverse_frequency = math.log(
                1.0
                + (document_count - frequency_in_documents + 0.5)
                / (frequency_in_documents + 0.5)
            )
            score += (
                query_frequency
                * inverse_frequency
                * (frequency * (BM25_K1 + 1.0) / (frequency + normalization))
            )
        scores.append(score)
    return scores


def _state_status(code: str) -> RetrievalStatus:
    if code in {
        "active_corpus_absent",
        "active_generation_incomplete",
        "compatible_index_absent",
        "eligible_corpus_empty",
        "published_index_corpus_stale",
        "corpus_version_mismatch",
        "corpus_generation_stale",
        "corpus_ingestion_incomplete",
        "index_incomplete",
        "index_staging",
        "index_validated",
        "index_state_absent",
        "index_state_incomplete",
        "index_state_stale",
    }:
        return RetrievalStatus.INSUFFICIENT
    return RetrievalStatus.ERROR


def _provider_identity(
    provider: BaselineIndexEmbeddingProvider | None,
) -> BaselineEmbeddingIdentity | None:
    if provider is None:
        return None
    try:
        return BaselineEmbeddingIdentity(
            provider=provider.provider,
            model=provider.model,
            revision=provider.revision,
            dimension=provider.dimension,
            fingerprint=provider.fingerprint,
        )
    except (AttributeError, TypeError, ValueError):
        return None


class PersistentBaselineV1Retriever:
    """Exact read-only retrieval from a compatible immutable publication."""

    name = "baseline_v1"

    def __init__(
        self,
        session_factory: Any,
        provider: BaselineIndexEmbeddingProvider | None,
        *,
        evidence_filter: EvidenceFilter | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._provider = provider
        self._evidence_filter = evidence_filter

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        precondition = self._precondition_error(request)
        if precondition is not None:
            return self._result(
                request,
                status=RetrievalStatus.INSUFFICIENT,
                error=precondition,
            )

        try:
            snapshot = self._load_publication(request)
        except _PersistentResolutionError as exc:
            return self._result(
                request,
                status=exc.status,
                snapshot=exc.snapshot,
                error=RetrievalError(exc.code, str(exc)),
            )
        except Exception as exc:  # noqa: BLE001 - database/provider boundary
            return self._result(
                request,
                status=RetrievalStatus.ERROR,
                error=RetrievalError(
                    "persistent_index_read_failed",
                    f"persistent baseline index read failed: {type(exc).__name__}",
                ),
            )

        if not snapshot.documents:
            return self._result(
                request,
                status=RetrievalStatus.INSUFFICIENT,
                snapshot=snapshot,
                error=RetrievalError(
                    "eligible_corpus_empty",
                    "compatible publication contains no eligible documents",
                ),
            )

        query = request.retrieval_query
        assert query is not None
        lexical_scores = bm25_scores_from_persisted_statistics(
            query,
            [document.token_count for document in snapshot.documents],
            [document.term_frequencies for document in snapshot.documents],
        )
        try:
            dense_scores = self._dense_scores(snapshot, query)
        except _PersistentResolutionError as exc:
            return self._result(
                request,
                status=exc.status,
                snapshot=snapshot,
                candidate_count=len(snapshot.documents),
                error=RetrievalError(exc.code, str(exc)),
            )

        ranked = self._rank(snapshot.documents, lexical_scores, dense_scores)
        retrieved = ranked[:RETRIEVAL_LIMIT]
        evidence, filtered, duplicates, refills = normalize_retrieved_candidates(
            retrieved,
            evidence_filter=self._evidence_filter,
        )
        try:
            self._recheck_publication(snapshot)
        except _PersistentResolutionError as exc:
            return self._result(
                request,
                status=exc.status,
                snapshot=snapshot,
                candidate_count=len(ranked),
                retrieved_count=len(retrieved),
                error=RetrievalError(exc.code, str(exc)),
            )

        evidence_characters = sum(len(item.content) for item in evidence)
        return self._result(
            request,
            status=(RetrievalStatus.OK if evidence else RetrievalStatus.INSUFFICIENT),
            snapshot=snapshot,
            candidates=tuple(ranked),
            evidence=tuple(evidence),
            candidate_count=len(ranked),
            retrieved_count=len(retrieved),
            filtered_count=filtered,
            duplicate_count=duplicates,
            refill_count=refills,
            evidence_characters=evidence_characters,
            underfilled=len(evidence) < MAX_EVIDENCE_ITEMS,
            error=(
                None
                if evidence
                else RetrievalError(
                    "eligible_evidence_empty",
                    "the top-six retrieval cut contains no eligible unique evidence",
                )
            ),
        )

    @staticmethod
    def _precondition_error(request: RetrievalRequest) -> RetrievalError | None:
        if request.schema_version != REQUEST_SCHEMA_VERSION:
            return RetrievalError(
                "unsupported_request_schema",
                f"expected {REQUEST_SCHEMA_VERSION}, got {request.schema_version}",
            )
        if not request.has_usable_explicit_query:
            return RetrievalError(
                "explicit_retrieval_query_absent",
                "baseline_v1 requires a non-empty explicit retrieval query",
            )
        if request.query_kind != "raw_git_diff_v1":
            return RetrievalError(
                "unsupported_query_kind",
                "persistent baseline_v1 requires raw_git_diff_v1",
            )
        if not request.corpus_complete:
            return RetrievalError(
                "file_corpus_incomplete",
                "the requested sibling corpus is incomplete",
            )
        if not request.corpus_scope_key:
            return RetrievalError(
                "corpus_scope_absent",
                "persistent baseline_v1 requires a corpus scope key",
            )
        if not request.changed_repository_id:
            return RetrievalError(
                "changed_repository_identity_absent",
                "persistent baseline_v1 requires changed repository identity",
            )
        return None

    def _load_publication(self, request: RetrievalRequest) -> _PublicationSnapshot:
        with self._session_factory() as session:
            corpus = session.scalar(
                select(RetrievalCorpus).where(
                    RetrievalCorpus.scope_key == request.corpus_scope_key
                )
            )
            if corpus is None or corpus.active_generation_id is None:
                raise _PersistentResolutionError(
                    "active_corpus_absent",
                    "no active trusted corpus exists for the requested scope",
                    status=RetrievalStatus.INSUFFICIENT,
                )
            if corpus.changed_repository_id != request.changed_repository_id:
                raise _PersistentResolutionError(
                    "changed_repository_mismatch",
                    "request changed repository does not match corpus identity",
                    status=RetrievalStatus.INSUFFICIENT,
                )
            generation = session.get(
                RetrievalCorpusGeneration,
                corpus.active_generation_id,
            )
            if (
                generation is None
                or generation.status != CorpusGenerationStatus.ACTIVE.value
            ):
                raise _PersistentResolutionError(
                    "active_generation_incomplete",
                    "active corpus generation is absent or incomplete",
                    status=RetrievalStatus.INSUFFICIENT,
                )
            if request.corpus_version and (
                request.corpus_version != generation.generation_version
            ):
                raise _PersistentResolutionError(
                    "corpus_version_mismatch",
                    "request corpus version is not the active generation",
                    status=RetrievalStatus.INSUFFICIENT,
                )
            ingestion = session.get(
                RetrievalCorpusIngestion,
                generation.generation_id,
            )
            if (
                ingestion is None
                or ingestion.status != CorpusIngestionStatus.ACTIVE.value
            ):
                raise _PersistentResolutionError(
                    "corpus_ingestion_incomplete",
                    "active corpus lacks complete trusted ingestion provenance",
                    status=RetrievalStatus.INSUFFICIENT,
                )
            publication = session.get(
                RetrievalBaselineIndexPublication,
                corpus.corpus_id,
            )
            if publication is None or publication.index_id is None:
                raise _PersistentResolutionError(
                    "compatible_index_absent",
                    "active corpus has no compatible published baseline index",
                    status=RetrievalStatus.INSUFFICIENT,
                )
            build = session.get(RetrievalBaselineIndexBuild, publication.index_id)
            if build is None:
                raise _PersistentResolutionError(
                    "published_index_absent",
                    "published baseline index metadata is absent",
                    status=RetrievalStatus.ERROR,
                )
            if build.generation_id != generation.generation_id:
                raise _PersistentResolutionError(
                    "published_index_corpus_stale",
                    "published baseline index targets a stale corpus generation",
                    status=RetrievalStatus.INSUFFICIENT,
                )
            if build.status != BaselineIndexBuildStatus.COMPATIBLE.value:
                code = f"index_{build.status}"
                raise _PersistentResolutionError(
                    code,
                    "published baseline index is not compatible",
                    status=_state_status(code),
                )

            documents = tuple(
                session.scalars(
                    select(RetrievalBaselineIndexDocument)
                    .where(RetrievalBaselineIndexDocument.index_id == build.index_id)
                    .order_by(RetrievalBaselineIndexDocument.ordinal)
                )
            )
            if any(
                document.repository_id == corpus.changed_repository_id
                for document in documents
            ):
                raise _PersistentResolutionError(
                    "changed_repository_in_published_index",
                    "published index contains the changed repository",
                    status=RetrievalStatus.ERROR,
                )

            index_state = session.get(
                RetrievalIndexState,
                generation.generation_id,
            )
            if index_state is None:
                raise _PersistentResolutionError(
                    "index_state_absent",
                    "published index compatibility state is absent",
                    status=RetrievalStatus.INSUFFICIENT,
                )
            if index_state.status != IndexStateStatus.COMPATIBLE.value:
                code = f"index_state_{index_state.status}"
                raise _PersistentResolutionError(
                    code,
                    "published index compatibility state is not compatible",
                    status=_state_status(code),
                )

            embedding = baseline_embedding_identity_from_build(build)
            mismatch = embedding_provider_mismatch_code(
                self._provider,
                embedding,
            )
            if mismatch is not None:
                raise _PersistentResolutionError(
                    mismatch,
                    "query embedding adapter does not match the published index",
                    status=RetrievalStatus.ERROR,
                )
            provider_identity = _provider_identity(self._provider)
            if provider_identity is None:
                raise _PersistentResolutionError(
                    "embedding_adapter_identity_invalid",
                    "query embedding adapter identity is invalid",
                    status=RetrievalStatus.ERROR,
                )
            readiness = assess_baseline_index(
                session,
                scope_key=corpus.scope_key,
                embedding=provider_identity,
            )
            if not readiness.ready:
                raise _PersistentResolutionError(
                    readiness.code,
                    "published baseline index failed compatibility validation",
                    status=_state_status(readiness.code),
                )

            terms = tuple(
                session.scalars(
                    select(RetrievalBaselineIndexTerm).where(
                        RetrievalBaselineIndexTerm.index_id == build.index_id
                    )
                )
            )
            terms_by_document: dict[str, dict[str, int]] = {}
            for row in terms:
                terms_by_document.setdefault(row.index_document_id, {})[
                    row.term
                ] = row.term_frequency
            vectors = tuple(
                session.scalars(
                    select(RetrievalBaselineIndexVector).where(
                        RetrievalBaselineIndexVector.index_id == build.index_id
                    )
                )
            )
            vectors_by_document = {
                row.index_document_id: row.vector_bytes for row in vectors
            }
            loaded: list[_PersistentDocument] = []
            for document in documents:
                source = session.get(RetrievalCorpusFile, document.corpus_file_id)
                vector_bytes = vectors_by_document.get(document.index_document_id)
                if (
                    source is None
                    or source.content is None
                    or vector_bytes is None
                    or source.repository_id == corpus.changed_repository_id
                ):
                    raise _PersistentResolutionError(
                        "published_index_artifact_incomplete",
                        "published index document artifacts are incomplete",
                        status=RetrievalStatus.ERROR,
                    )
                loaded.append(
                    _PersistentDocument(
                        index_document_id=document.index_document_id,
                        corpus_file_id=document.corpus_file_id,
                        repository_id=document.repository_id,
                        repository_name=document.repository_name,
                        relative_path=document.relative_path,
                        content=source.content[:RANKING_CONTENT_CHARACTERS],
                        content_hash=source.content_hash,
                        byte_size=source.byte_size,
                        token_count=document.token_count,
                        term_frequencies=terms_by_document.get(
                            document.index_document_id,
                            {},
                        ),
                        vector_bytes=vector_bytes,
                    )
                )
            assert generation.manifest_hash is not None
            return _PublicationSnapshot(
                corpus_id=corpus.corpus_id,
                corpus_scope_key=corpus.scope_key,
                changed_repository_id=corpus.changed_repository_id,
                generation_id=generation.generation_id,
                generation_version=generation.generation_version,
                corpus_manifest_hash=generation.manifest_hash,
                index_id=build.index_id,
                index_version=build.index_version,
                index_schema_version=build.index_schema_version,
                index_fingerprint=published_index_fingerprint(build),
                config_fingerprint=build.engine_config_fingerprint,
                embedding=embedding,
                documents=tuple(loaded),
            )

    def _dense_scores(
        self,
        snapshot: _PublicationSnapshot,
        query: str,
    ) -> list[float]:
        provider = self._provider
        if provider is None:
            raise _PersistentResolutionError(
                "embedding_adapter_unavailable",
                "query embedding adapter is unavailable",
                status=RetrievalStatus.ERROR,
                snapshot=snapshot,
            )
        try:
            raw_query = list(provider.embed([query]))
        except Exception as exc:
            raise _PersistentResolutionError(
                "query_embedding_failed",
                f"query embedding failed: {type(exc).__name__}",
                status=RetrievalStatus.ERROR,
                snapshot=snapshot,
            ) from exc
        if len(raw_query) != 1:
            raise _PersistentResolutionError(
                "query_embedding_count_mismatch",
                "embedding adapter did not return exactly one query vector",
                status=RetrievalStatus.ERROR,
                snapshot=snapshot,
            )
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                query_vector = np.asarray(raw_query[0], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise _PersistentResolutionError(
                "query_embedding_invalid",
                "embedding adapter returned an invalid query vector",
                status=RetrievalStatus.ERROR,
                snapshot=snapshot,
            ) from exc
        if query_vector.shape != (snapshot.embedding.dimension,):
            raise _PersistentResolutionError(
                "query_embedding_dimension_mismatch",
                "query vector does not match the published index dimension",
                status=RetrievalStatus.ERROR,
                snapshot=snapshot,
            )
        if not np.isfinite(query_vector).all():
            raise _PersistentResolutionError(
                "query_embedding_nonfinite",
                "query vector contains NaN or infinity",
                status=RetrievalStatus.ERROR,
                snapshot=snapshot,
            )
        document_vectors = np.stack(
            [
                np.frombuffer(document.vector_bytes, dtype="<f4")
                for document in snapshot.documents
            ]
        ).astype(np.float32, copy=False)
        if document_vectors.shape != (
            len(snapshot.documents),
            snapshot.embedding.dimension,
        ):
            raise _PersistentResolutionError(
                "published_vector_dimension_mismatch",
                "published document vectors have an invalid shape",
                status=RetrievalStatus.ERROR,
                snapshot=snapshot,
            )
        dense_scores = document_vectors @ query_vector
        if not np.isfinite(dense_scores).all():
            raise _PersistentResolutionError(
                "dense_score_nonfinite",
                "float32 dot product produced a non-finite score",
                status=RetrievalStatus.ERROR,
                snapshot=snapshot,
            )
        return [float(score) for score in dense_scores]

    @staticmethod
    def _rank(
        documents: Sequence[_PersistentDocument],
        lexical_scores: Sequence[float],
        dense_scores: Sequence[float],
    ) -> list[RetrievalCandidate]:
        indices = range(len(documents))

        def tie_key(index: int) -> tuple[str, str]:
            return (
                documents[index].path,
                documents[index].index_document_id,
            )

        bm25_order = sorted(
            indices,
            key=lambda index: (-lexical_scores[index], *tie_key(index)),
        )
        dense_order = sorted(
            indices,
            key=lambda index: (-dense_scores[index], *tie_key(index)),
        )
        bm25_rank = {index: rank for rank, index in enumerate(bm25_order, start=1)}
        dense_rank = {index: rank for rank, index in enumerate(dense_order, start=1)}
        fused_scores = reciprocal_rank_fusion(
            [bm25_rank[index] for index in indices],
            [dense_rank[index] for index in indices],
        )
        fused_order = sorted(
            indices,
            key=lambda index: (-fused_scores[index], *tie_key(index)),
        )
        fused_rank = {index: rank for rank, index in enumerate(fused_order, start=1)}
        by_index = {
            index: RetrievalCandidate(
                repository=documents[index].repository_name,
                relative_path=documents[index].relative_path,
                content=documents[index].content,
                content_hash=documents[index].content_hash,
                byte_size=documents[index].byte_size,
                bm25_score=lexical_scores[index],
                bm25_rank=bm25_rank[index],
                dense_score=dense_scores[index],
                dense_rank=dense_rank[index],
                rrf_score=fused_scores[index],
                fused_rank=fused_rank[index],
                document_id=documents[index].index_document_id,
            )
            for index in indices
        }
        return [by_index[index] for index in fused_order]

    def _recheck_publication(self, snapshot: _PublicationSnapshot) -> None:
        with self._session_factory() as session:
            corpus = session.get(RetrievalCorpus, snapshot.corpus_id)
            publication = session.get(
                RetrievalBaselineIndexPublication,
                snapshot.corpus_id,
            )
            build = session.get(RetrievalBaselineIndexBuild, snapshot.index_id)
            if (
                corpus is None
                or corpus.active_generation_id != snapshot.generation_id
                or publication is None
                or publication.index_id != snapshot.index_id
                or build is None
                or build.status != BaselineIndexBuildStatus.COMPATIBLE.value
            ):
                raise _PersistentResolutionError(
                    "publication_changed_during_retrieval",
                    "active compatible publication changed during retrieval",
                    status=RetrievalStatus.INSUFFICIENT,
                    snapshot=snapshot,
                )
            readiness = assess_baseline_index(
                session,
                scope_key=snapshot.corpus_scope_key,
                embedding=snapshot.embedding,
            )
            if not readiness.ready or readiness.index_id != snapshot.index_id:
                raise _PersistentResolutionError(
                    readiness.code,
                    "published baseline index became incompatible during retrieval",
                    status=_state_status(readiness.code),
                    snapshot=snapshot,
                )

    def _result(
        self,
        request: RetrievalRequest,
        *,
        status: RetrievalStatus,
        snapshot: _PublicationSnapshot | None = None,
        candidates: tuple[RetrievalCandidate, ...] = (),
        evidence: tuple[RetrievalEvidence, ...] = (),
        candidate_count: int = 0,
        retrieved_count: int = 0,
        filtered_count: int = 0,
        duplicate_count: int = 0,
        refill_count: int = 0,
        evidence_characters: int = 0,
        underfilled: bool = True,
        error: RetrievalError | None = None,
    ) -> RetrievalResult:
        provider_identity = _provider_identity(self._provider)
        fallback_fingerprint = hashlib.sha256(
            f"{PERSISTENT_BASELINE_ENGINE_VERSION}:{error.code if error else status.value}".encode()
        ).hexdigest()
        embedding = snapshot.embedding if snapshot is not None else provider_identity
        return RetrievalResult(
            request_id=request.request_id,
            status=status,
            corpus_version=(
                snapshot.generation_version
                if snapshot is not None
                else request.corpus_version
            ),
            config_fingerprint=(
                snapshot.config_fingerprint
                if snapshot is not None
                else fallback_fingerprint
            ),
            embedding_model=embedding.model if embedding is not None else "unavailable",
            embedding_revision=(
                embedding.revision if embedding is not None else "unavailable"
            ),
            embedding_dimension=embedding.dimension if embedding is not None else 0,
            candidates=candidates,
            evidence=evidence,
            candidate_count=candidate_count,
            retrieved_count=retrieved_count,
            filtered_count=filtered_count,
            duplicate_count=duplicate_count,
            refill_count=refill_count,
            evidence_characters=evidence_characters,
            underfilled=underfilled,
            error=error,
            fallback_engine=None,
            engine="baseline_v1",
            engine_version=PERSISTENT_BASELINE_ENGINE_VERSION,
            corpus_id=snapshot.corpus_id if snapshot is not None else None,
            corpus_manifest_hash=(
                snapshot.corpus_manifest_hash if snapshot is not None else None
            ),
            corpus_scope_key=(
                snapshot.corpus_scope_key
                if snapshot is not None
                else request.corpus_scope_key
            ),
            index_id=snapshot.index_id if snapshot is not None else None,
            index_version=snapshot.index_version if snapshot is not None else None,
            index_schema_version=(
                snapshot.index_schema_version if snapshot is not None else None
            ),
            index_fingerprint=(
                snapshot.index_fingerprint if snapshot is not None else None
            ),
            embedding_provider=(
                embedding.provider if embedding is not None else None
            ),
            embedding_fingerprint=(
                embedding.fingerprint if embedding is not None else None
            ),
            query_provenance=request.query_provenance,
        )
