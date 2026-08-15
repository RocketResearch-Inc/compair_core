"""Frozen, pure BM25 + dense dot-product + RRF baseline retrieval.

The implementation is intentionally dependency-injected and filesystem-only.
It neither imports nor falls back to Core's legacy selector, storage, FTS, or
embedding helpers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

from .types import (
    REQUEST_SCHEMA_VERSION,
    DenseEmbeddingProvider,
    FileCandidate,
    RetrievalCandidate,
    RetrievalError,
    RetrievalEvidence,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStatus,
)

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/-]{1,}|[0-9]+")
STOPWORDS = frozenset(
    {
        "and",
        "are",
        "class",
        "const",
        "false",
        "for",
        "from",
        "function",
        "import",
        "into",
        "return",
        "that",
        "the",
        "this",
        "true",
        "with",
    }
)
EXCLUDED_PATH_PARTS = frozenset({".git", ".compair", "build", "dist", "node_modules"})
MAX_FILE_BYTES = 200_000
RANKING_CONTENT_CHARACTERS = 12_000
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60
RETRIEVAL_LIMIT = 6
MAX_EVIDENCE_ITEMS = 4
MAX_EVIDENCE_CHARACTERS = 16_000
SYMLINK_POLICY = "reject_all_v1"
BASELINE_TOKENIZER_VERSION = "baseline_v1_frozen_tokenizer.v1"
BASELINE_DOCUMENT_FORMAT_VERSION = "baseline_v1_whole_file_12000.v1"


def frozen_tokens(text: str) -> list[str]:
    """Tokenize with the frozen comparator regex, splits, and stoplist."""

    output: list[str] = []
    for raw in TOKEN_RE.findall(text):
        lowered = raw.lower().strip("./:-")
        candidates = [lowered, *re.split(r"[_.:/-]+", lowered)]
        for value in candidates:
            if len(value) >= 2 and value not in STOPWORDS:
                output.append(value)
    return output


def baseline_ranking_document(
    repository: str,
    relative_path: str,
    content: str,
) -> str:
    """Return the frozen whole-file document embedded and scored by baseline_v1."""

    return (
        f"Repository file: {repository}/{relative_path}\n\n"
        f"{content[:RANKING_CONTENT_CHARACTERS]}"
    )


def bm25_scores(query: str, documents: Sequence[str]) -> list[float]:
    """Compute the comparator's exact whole-corpus BM25 formula."""

    tokenized = [frozen_tokens(document) for document in documents]
    query_counts = Counter(frozen_tokens(query))
    document_counts = [Counter(tokens) for tokens in tokenized]
    lengths = [len(tokens) for tokens in tokenized]
    average_length = sum(lengths) / max(1, len(lengths))
    document_frequency: Counter[str] = Counter()
    for counts in document_counts:
        document_frequency.update(counts.keys())

    document_count = len(documents)
    scores: list[float] = []
    for counts, length in zip(document_counts, lengths):
        score = 0.0
        normalization = BM25_K1 * (
            1.0 - BM25_B + BM25_B * length / max(1.0, average_length)
        )
        for term, query_frequency in query_counts.items():
            frequency = counts.get(term, 0)
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


def enumerate_file_candidates(
    repository_roots: Iterable[Path],
    *,
    changed_repository: Path,
) -> list[FileCandidate]:
    """Enumerate eligible UTF-8 whole files in stable repository/path order.

    Unlike the vendored comparator, production baseline_v1 deliberately rejects
    every symlink, including links whose targets remain inside the repository.
    This prevents an escaping link from expanding candidate scope. Expected
    comparator exclusions are skipped; other I/O errors become explicit errors.
    """

    changed_resolved = changed_repository.resolve()
    supplied_roots = {Path(root) for root in repository_roots}
    if any(root.is_symlink() for root in supplied_roots):
        raise OSError("candidate repository roots must not be symlinks")
    roots = sorted(
        {root.resolve() for root in supplied_roots},
        key=lambda root: (root.name, root.as_posix()),
    )
    rows: list[FileCandidate] = []
    for repository in roots:
        if repository == changed_resolved:
            continue
        if not repository.is_dir():
            raise OSError(f"candidate repository is not a directory: {repository}")
        for path in repository.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(repository)
            if any(part in EXCLUDED_PATH_PARTS for part in relative.parts):
                continue
            stat = path.stat()
            if stat.st_size > MAX_FILE_BYTES:
                continue
            raw_content = path.read_bytes()
            if len(raw_content) > MAX_FILE_BYTES:
                continue
            try:
                content = raw_content.decode("utf-8")
            except UnicodeDecodeError:
                continue
            rows.append(
                FileCandidate(
                    repository=repository.name,
                    relative_path=relative.as_posix(),
                    content=content,
                    content_hash=hashlib.sha256(raw_content).hexdigest(),
                    byte_size=len(raw_content),
                )
            )
    return sorted(rows, key=lambda candidate: candidate.path)


def reciprocal_rank_fusion(
    bm25_ranks: Sequence[int],
    dense_ranks: Sequence[int],
) -> list[float]:
    """Fuse equal-weight one-based ranks with frozen ``k=60``."""

    if len(bm25_ranks) != len(dense_ranks):
        raise ValueError("BM25 and dense rank counts differ")
    if any(rank < 1 for rank in (*bm25_ranks, *dense_ranks)):
        raise ValueError("RRF ranks must be one-based positive integers")
    return [
        1.0 / (RRF_K + bm25_rank) + 1.0 / (RRF_K + dense_rank)
        for bm25_rank, dense_rank in zip(bm25_ranks, dense_ranks)
    ]


EvidenceFilter = Callable[[RetrievalCandidate], bool]


def normalize_retrieved_candidates(
    retrieved: Sequence[RetrievalCandidate],
    *,
    evidence_filter: EvidenceFilter | None = None,
) -> tuple[list[RetrievalEvidence], int, int, int]:
    """Normalize only the supplied retrieval cut; never scan beyond it."""

    candidate_filter = evidence_filter or (lambda candidate: True)
    evidence: list[RetrievalEvidence] = []
    seen_contents: set[str] = set()
    filtered_count = 0
    duplicate_count = 0
    refill_count = 0
    remaining = MAX_EVIDENCE_CHARACTERS

    for position, candidate in enumerate(retrieved, start=1):
        content = candidate.content.strip()
        if (
            not candidate.path.strip()
            or not content
            or not candidate_filter(candidate)
        ):
            filtered_count += 1
            continue
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if content_hash in seen_contents:
            duplicate_count += 1
            continue
        seen_contents.add(content_hash)
        if remaining <= 0 or len(evidence) >= MAX_EVIDENCE_ITEMS:
            break
        clipped = content[:remaining]
        if not clipped:
            break
        if position > MAX_EVIDENCE_ITEMS:
            refill_count += 1
        evidence.append(
            RetrievalEvidence(
                repository=candidate.repository,
                relative_path=candidate.relative_path,
                content=clipped,
                content_hash=content_hash,
                fused_rank=candidate.fused_rank,
                render_truncated=len(clipped) < len(content),
                bm25_score=candidate.bm25_score,
                bm25_rank=candidate.bm25_rank,
                dense_score=candidate.dense_score,
                dense_rank=candidate.dense_rank,
                rrf_score=candidate.rrf_score,
                document_id=candidate.document_id,
            )
        )
        remaining -= len(clipped)
    return evidence, filtered_count, duplicate_count, refill_count


class BaselineV1Retriever:
    """Pure frozen comparator with an injected dense embedding provider."""

    def __init__(
        self,
        dense_provider: DenseEmbeddingProvider,
        *,
        evidence_filter: EvidenceFilter | None = None,
    ) -> None:
        self._dense_provider = dense_provider
        self._evidence_filter = evidence_filter or (lambda candidate: True)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        precondition = self._precondition_error(request)
        if precondition is not None:
            return self._result(
                request,
                status=RetrievalStatus.INSUFFICIENT,
                error=precondition,
            )

        try:
            files = enumerate_file_candidates(
                request.repository_roots,
                changed_repository=request.changed_repository,
            )
        except OSError as exc:
            return self._result(
                request,
                status=RetrievalStatus.ERROR,
                error=RetrievalError("candidate_enumeration_failed", str(exc)),
            )

        if not files:
            return self._result(
                request,
                status=RetrievalStatus.INSUFFICIENT,
                error=RetrievalError(
                    "eligible_corpus_empty", "no eligible sibling files were found"
                ),
            )

        texts = [
            baseline_ranking_document(
                candidate.repository,
                candidate.relative_path,
                candidate.content,
            )
            for candidate in files
        ]
        query = request.retrieval_query
        assert query is not None  # Established by the explicit-query precondition.
        lexical_scores = bm25_scores(query, texts)
        try:
            dense_scores = self._dense_scores(texts, query)
        except (ArithmeticError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return self._result(
                request,
                status=RetrievalStatus.ERROR,
                candidate_count=len(files),
                error=RetrievalError("dense_embedding_failed", str(exc)),
            )

        ranked = self._rank(files, lexical_scores, dense_scores)
        retrieved = ranked[:RETRIEVAL_LIMIT]
        evidence, filtered, duplicates, refills = self._normalize(retrieved)
        evidence_characters = sum(len(item.content) for item in evidence)
        return self._result(
            request,
            status=(RetrievalStatus.OK if evidence else RetrievalStatus.INSUFFICIENT),
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
                    "retrieved candidates produced no eligible unique evidence",
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
                "baseline_v1 requires a non-empty explicit change-set query",
            )
        if request.query_kind != "raw_git_diff_v1":
            return RetrievalError(
                "unsupported_query_kind", "baseline_v1 requires raw_git_diff_v1"
            )
        if not request.corpus_complete:
            return RetrievalError(
                "file_corpus_incomplete", "the sibling file corpus is incomplete"
            )
        if not request.repository_roots:
            return RetrievalError(
                "file_corpus_absent", "no sibling repository roots were supplied"
            )
        if request.changed_repository is None:
            return RetrievalError(
                "changed_repository_absent", "no changed repository was supplied"
            )
        return None

    def _dense_scores(self, documents: Sequence[str], query: str) -> list[float]:
        dimension = self._dense_provider.dimension
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dense provider dimension must be a positive integer")

        raw_documents = list(self._dense_provider.embed(documents))
        raw_query = list(self._dense_provider.embed([query]))
        if len(raw_documents) != len(documents):
            raise ValueError(
                f"dense provider returned {len(raw_documents)} vectors for "
                f"{len(documents)} documents"
            )
        if len(raw_query) != 1:
            raise ValueError("dense provider did not return exactly one query vector")

        document_vectors = np.asarray(raw_documents, dtype=np.float32)
        query_vector = np.asarray(raw_query[0], dtype=np.float32)
        if document_vectors.shape != (len(documents), dimension):
            raise ValueError(
                "dense document vectors do not match the expected count and dimension"
            )
        if query_vector.shape != (dimension,):
            raise ValueError(
                f"dense query vector shape {query_vector.shape} does not match "
                f"expected ({dimension},)"
            )
        if (
            not np.isfinite(document_vectors).all()
            or not np.isfinite(query_vector).all()
        ):
            raise ValueError("dense vector contains a non-finite value")
        dense_scores = document_vectors @ query_vector
        return [float(score) for score in dense_scores]

    @staticmethod
    def _rank(
        files: Sequence[FileCandidate],
        lexical_scores: Sequence[float],
        dense_scores: Sequence[float],
    ) -> list[RetrievalCandidate]:
        indices = range(len(files))
        bm25_order = sorted(
            indices, key=lambda index: (-lexical_scores[index], files[index].path)
        )
        dense_order = sorted(
            indices, key=lambda index: (-dense_scores[index], files[index].path)
        )
        bm25_rank = {index: rank for rank, index in enumerate(bm25_order, start=1)}
        dense_rank = {index: rank for rank, index in enumerate(dense_order, start=1)}
        fused_scores = reciprocal_rank_fusion(
            [bm25_rank[index] for index in indices],
            [dense_rank[index] for index in indices],
        )
        fused_order = sorted(
            indices,
            key=lambda index: (-fused_scores[index], files[index].path),
        )
        fused_rank = {index: rank for rank, index in enumerate(fused_order, start=1)}
        # Preserve ranking precision internally. A future external serializer
        # may reproduce the vendor's display rounding without affecting order.
        by_index = {
            index: RetrievalCandidate(
                repository=files[index].repository,
                relative_path=files[index].relative_path,
                content=files[index].content[:RANKING_CONTENT_CHARACTERS],
                content_hash=files[index].content_hash,
                byte_size=files[index].byte_size,
                bm25_score=lexical_scores[index],
                bm25_rank=bm25_rank[index],
                dense_score=dense_scores[index],
                dense_rank=dense_rank[index],
                rrf_score=fused_scores[index],
                fused_rank=fused_rank[index],
            )
            for index in indices
        }
        return [by_index[index] for index in fused_order]

    def _normalize(
        self, ranked: Sequence[RetrievalCandidate]
    ) -> tuple[list[RetrievalEvidence], int, int, int]:
        return normalize_retrieved_candidates(
            ranked,
            evidence_filter=self._evidence_filter,
        )

    def _result(
        self,
        request: RetrievalRequest,
        *,
        status: RetrievalStatus,
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
        provider = self._dense_provider
        fingerprint_payload = {
            "bm25_b": BM25_B,
            "bm25_k1": BM25_K1,
            "dense_dimension": provider.dimension,
            "dense_model": provider.model,
            "dense_revision": provider.revision,
            "evidence_characters": MAX_EVIDENCE_CHARACTERS,
            "evidence_items": MAX_EVIDENCE_ITEMS,
            "ranking_characters": RANKING_CONTENT_CHARACTERS,
            "retrieval_limit": RETRIEVAL_LIMIT,
            "rrf_k": RRF_K,
            "tokenizer": BASELINE_TOKENIZER_VERSION,
            "symlink_policy": SYMLINK_POLICY,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                fingerprint_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        return RetrievalResult(
            request_id=request.request_id,
            status=status,
            corpus_version=request.corpus_version,
            config_fingerprint=fingerprint,
            embedding_model=provider.model,
            embedding_revision=provider.revision,
            embedding_dimension=provider.dimension,
            embedding_provider=getattr(provider, "provider", None),
            embedding_fingerprint=getattr(provider, "fingerprint", None),
            query_provenance=request.query_provenance,
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
        )
