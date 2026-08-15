"""Durable, fail-closed baseline_v1 index construction and publication.

This module builds only whole-file lexical and dense artifacts. It does not
accept a retrieval query, score candidates, fuse ranks, select evidence, or
invoke legacy retrieval.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Protocol, runtime_checkable

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from .baseline import (
    BASELINE_DOCUMENT_FORMAT_VERSION,
    BASELINE_TOKENIZER_VERSION,
    BM25_B,
    BM25_K1,
    EXCLUDED_PATH_PARTS,
    MAX_EVIDENCE_CHARACTERS,
    MAX_EVIDENCE_ITEMS,
    MAX_FILE_BYTES,
    RANKING_CONTENT_CHARACTERS,
    RETRIEVAL_LIMIT,
    RRF_K,
    STOPWORDS,
    SYMLINK_POLICY,
    TOKEN_RE,
    baseline_ranking_document,
    frozen_tokens,
)
from .corpus import (
    BaselineIndexBuildStatus,
    CorpusFileState,
    CorpusGenerationStatus,
    CorpusIngestionStatus,
    CorpusLifecycle,
    IndexRequirements,
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

BASELINE_INDEX_SCHEMA_VERSION = "baseline-index.v1"
BASELINE_VECTOR_FORMAT = "float32-le.v1"


@runtime_checkable
class BaselineIndexEmbeddingProvider(Protocol):
    """Pinned embedding adapter used only to construct baseline index vectors."""

    provider: str
    model: str
    revision: str
    dimension: int
    fingerprint: str

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one document vector for each supplied ranking document."""


def _canonical_identifier(value: str, label: str, max_length: int) -> str:
    if not value or value != value.strip() or len(value) > max_length:
        raise ValueError(f"{label} must be a canonical non-empty identifier")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{label} contains a control character")
    return value


def _sha256(value: str, label: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    return normalized


@dataclass(frozen=True, slots=True)
class BaselineEmbeddingIdentity:
    """Exact dense adapter identity required for one compatible build."""

    provider: str
    model: str
    revision: str
    dimension: int
    fingerprint: str

    def __post_init__(self) -> None:
        _canonical_identifier(self.provider, "embedding provider", 128)
        _canonical_identifier(self.model, "embedding model", 256)
        _canonical_identifier(self.revision, "embedding revision", 256)
        if not isinstance(self.dimension, int) or self.dimension <= 0:
            raise ValueError("embedding dimension must be a positive integer")
        object.__setattr__(
            self,
            "fingerprint",
            _sha256(self.fingerprint, "embedding fingerprint"),
        )


@dataclass(frozen=True, slots=True)
class BaselineIndexBuildResult:
    index_id: str
    generation_id: str
    index_version: str
    document_count: int
    total_token_count: int
    document_manifest_hash: str
    lexical_manifest_hash: str
    dense_manifest_hash: str
    status: BaselineIndexBuildStatus


@dataclass(frozen=True, slots=True)
class BaselineIndexReadiness:
    ready: bool
    code: str
    corpus_id: str | None = None
    generation_id: str | None = None
    index_id: str | None = None


class BaselineIndexBuildError(RuntimeError):
    """Machine-readable fail-closed index construction error."""

    def __init__(self, code: str, message: str, *, index_id: str | None = None):
        super().__init__(message)
        self.code = code
        self.index_id = index_id


@dataclass(frozen=True, slots=True)
class _SourceFile:
    file_id: str
    repository_id: str
    repository_name: str
    relative_path: str
    content: str
    content_hash: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class _DocumentArtifact:
    source: _SourceFile
    ordinal: int
    ranking_text: str
    indexed_document_hash: str
    term_frequencies: tuple[tuple[str, int], ...]
    token_count: int
    vector_bytes: bytes
    vector_hash: str


def baseline_engine_config_fingerprint(
    embedding: BaselineEmbeddingIdentity,
) -> str:
    """Hash every frozen representation choice relevant to baseline_v1."""

    payload = {
        "bm25_b": BM25_B,
        "bm25_k1": BM25_K1,
        "document_format": BASELINE_DOCUMENT_FORMAT_VERSION,
        "embedding": {
            "dimension": embedding.dimension,
            "fingerprint": embedding.fingerprint,
            "model": embedding.model,
            "provider": embedding.provider,
            "revision": embedding.revision,
        },
        "evidence_characters": MAX_EVIDENCE_CHARACTERS,
        "evidence_items": MAX_EVIDENCE_ITEMS,
        "index_schema": BASELINE_INDEX_SCHEMA_VERSION,
        "max_file_bytes": MAX_FILE_BYTES,
        "ranking_characters": RANKING_CONTENT_CHARACTERS,
        "retrieval_limit": RETRIEVAL_LIMIT,
        "rrf_k": RRF_K,
        "stopwords": sorted(STOPWORDS),
        "symlink_policy": SYMLINK_POLICY,
        "token_regex": TOKEN_RE.pattern,
        "tokenizer": BASELINE_TOKENIZER_VERSION,
        "vector_format": BASELINE_VECTOR_FORMAT,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def baseline_index_requirements(
    embedding: BaselineEmbeddingIdentity,
) -> IndexRequirements:
    return IndexRequirements(
        tokenizer_version=BASELINE_TOKENIZER_VERSION,
        embedding_provider=embedding.provider,
        embedding_model=embedding.model,
        embedding_revision=embedding.revision,
        embedding_dimension=embedding.dimension,
        embedding_fingerprint=embedding.fingerprint,
        engine_config_fingerprint=baseline_engine_config_fingerprint(embedding),
    )


def lexical_term_frequencies(text: str) -> tuple[tuple[str, int], ...]:
    """Return exact frozen tokenizer frequencies in deterministic term order."""

    return tuple(sorted(Counter(frozen_tokens(text)).items()))


def _digest_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _eligible_file(row: RetrievalCorpusFile) -> bool:
    return (
        row.file_state == CorpusFileState.SUPPORTED.value
        and row.content is not None
        and row.byte_size <= MAX_FILE_BYTES
        and not any(
            part in EXCLUDED_PATH_PARTS
            for part in PurePosixPath(row.relative_path).parts
        )
    )


def _source_files(
    session: Session,
    generation_id: str,
) -> tuple[_SourceFile, ...]:
    generation = session.get(RetrievalCorpusGeneration, generation_id)
    if generation is None:
        raise BaselineIndexBuildError(
            "corpus_generation_absent",
            "corpus generation does not exist",
        )
    corpus = session.get(RetrievalCorpus, generation.corpus_id)
    if corpus is None:
        raise BaselineIndexBuildError("corpus_absent", "corpus does not exist")
    rows = CorpusLifecycle.ordered_files(session, generation_id)
    sources: list[_SourceFile] = []
    for row in rows:
        if row.repository_id == corpus.changed_repository_id:
            raise BaselineIndexBuildError(
                "changed_repository_in_candidate_corpus",
                "changed repository must not appear in baseline candidate corpus",
            )
        if not _eligible_file(row):
            continue
        assert row.content is not None
        raw = row.content.encode("utf-8")
        if len(raw) != row.byte_size:
            raise BaselineIndexBuildError(
                "corpus_file_size_mismatch",
                "eligible corpus file byte size changed after ingestion",
            )
        if hashlib.sha256(raw).hexdigest() != row.content_hash:
            raise BaselineIndexBuildError(
                "corpus_file_hash_mismatch",
                "eligible corpus file content changed after ingestion",
            )
        sources.append(
            _SourceFile(
                file_id=row.file_id,
                repository_id=row.repository_id,
                repository_name=row.repository_name,
                relative_path=row.relative_path,
                content=row.content,
                content_hash=row.content_hash,
                byte_size=row.byte_size,
            )
        )
    return tuple(sources)


def baseline_embedding_identity_from_build(
    build: RetrievalBaselineIndexBuild,
) -> BaselineEmbeddingIdentity:
    return BaselineEmbeddingIdentity(
        provider=build.embedding_provider,
        model=build.embedding_model,
        revision=build.embedding_revision,
        dimension=build.embedding_dimension,
        fingerprint=build.embedding_fingerprint,
    )


def embedding_provider_mismatch_code(
    provider: BaselineIndexEmbeddingProvider | None,
    expected: BaselineEmbeddingIdentity,
) -> str | None:
    if provider is None:
        return "embedding_adapter_unavailable"
    try:
        actual = BaselineEmbeddingIdentity(
            provider=provider.provider,
            model=provider.model,
            revision=provider.revision,
            dimension=provider.dimension,
            fingerprint=provider.fingerprint,
        )
    except (AttributeError, TypeError, ValueError):
        return "embedding_adapter_identity_invalid"
    if actual.fingerprint != expected.fingerprint:
        return "embedding_fingerprint_mismatch"
    if actual != expected:
        return "embedding_identity_mismatch"
    return None


def _manifest_hashes(
    artifacts: Sequence[_DocumentArtifact],
) -> tuple[str, str, str]:
    documents = [
        {
            "content_hash": artifact.source.content_hash,
            "document_hash": artifact.indexed_document_hash,
            "file_id": artifact.source.file_id,
            "ordinal": artifact.ordinal,
            "path": (
                f"{artifact.source.repository_name}/"
                f"{artifact.source.relative_path}"
            ),
            "repository_id": artifact.source.repository_id,
            "token_count": artifact.token_count,
        }
        for artifact in artifacts
    ]
    lexical = [
        {
            "document_hash": artifact.indexed_document_hash,
            "frequency": frequency,
            "ordinal": artifact.ordinal,
            "term": term,
            "term_hash": hashlib.sha256(term.encode("utf-8")).hexdigest(),
        }
        for artifact in artifacts
        for term, frequency in artifact.term_frequencies
    ]
    dense = [
        {
            "document_hash": artifact.indexed_document_hash,
            "ordinal": artifact.ordinal,
            "vector_hash": artifact.vector_hash,
        }
        for artifact in artifacts
    ]
    return _digest_json(documents), _digest_json(lexical), _digest_json(dense)


class BaselineIndexLifecycle:
    """Validation and publication operations over staged baseline artifacts."""

    @staticmethod
    def _active_context_error(
        session: Session,
        generation: RetrievalCorpusGeneration,
    ) -> tuple[str, RetrievalCorpus | None]:
        corpus = session.get(RetrievalCorpus, generation.corpus_id)
        if corpus is None:
            return "corpus_absent", None
        if (
            generation.status != CorpusGenerationStatus.ACTIVE.value
            or corpus.active_generation_id != generation.generation_id
        ):
            return "corpus_generation_stale", corpus
        ingestion = session.get(RetrievalCorpusIngestion, generation.generation_id)
        if (
            ingestion is None
            or ingestion.status != CorpusIngestionStatus.ACTIVE.value
        ):
            return "corpus_ingestion_incomplete", corpus
        if generation.manifest_hash is None:
            return "corpus_manifest_absent", corpus
        if ingestion.file_count != generation.expected_file_count:
            return "corpus_file_count_mismatch", corpus
        if hashlib.sha256(
            ingestion.canonical_manifest_json.encode("utf-8")
        ).hexdigest() != ingestion.canonical_manifest_hash:
            return "corpus_ingestion_manifest_mismatch", corpus
        integrity_error = CorpusLifecycle.generation_integrity_error(
            session,
            generation.generation_id,
        )
        if integrity_error is not None:
            return integrity_error, corpus
        return "", corpus

    @classmethod
    def validation_error(cls, session: Session, index_id: str) -> str | None:
        build = session.get(RetrievalBaselineIndexBuild, index_id)
        if build is None:
            return "index_build_absent"
        generation = session.get(RetrievalCorpusGeneration, build.generation_id)
        if generation is None:
            return "corpus_generation_absent"
        context_error, _ = cls._active_context_error(session, generation)
        if context_error:
            return context_error
        if generation.manifest_hash != build.corpus_manifest_hash:
            return "index_corpus_manifest_mismatch"

        identity = baseline_embedding_identity_from_build(build)
        if build.index_schema_version != BASELINE_INDEX_SCHEMA_VERSION:
            return "index_schema_mismatch"
        if build.document_format_version != BASELINE_DOCUMENT_FORMAT_VERSION:
            return "index_document_format_mismatch"
        if build.tokenizer_version != BASELINE_TOKENIZER_VERSION:
            return "index_tokenizer_mismatch"
        if build.engine_config_fingerprint != baseline_engine_config_fingerprint(
            identity
        ):
            return "index_config_fingerprint_mismatch"

        try:
            sources = _source_files(session, build.generation_id)
        except BaselineIndexBuildError as exc:
            return exc.code
        documents = tuple(
            session.scalars(
                select(RetrievalBaselineIndexDocument)
                .where(RetrievalBaselineIndexDocument.index_id == index_id)
                .order_by(RetrievalBaselineIndexDocument.ordinal)
            )
        )
        if (
            len(sources) != build.expected_document_count
            or len(documents) != build.indexed_document_count
            or len(documents) != len(sources)
        ):
            return "index_document_count_mismatch"

        terms = tuple(
            session.scalars(
                select(RetrievalBaselineIndexTerm)
                .where(RetrievalBaselineIndexTerm.index_id == index_id)
                .order_by(
                    RetrievalBaselineIndexTerm.index_document_id,
                    RetrievalBaselineIndexTerm.term,
                )
            )
        )
        terms_by_document: dict[str, list[RetrievalBaselineIndexTerm]] = {}
        for term in terms:
            terms_by_document.setdefault(term.index_document_id, []).append(term)
        document_ids = {document.index_document_id for document in documents}
        if set(terms_by_document) - document_ids:
            return "index_lexical_orphan_term"
        vectors = tuple(
            session.scalars(
                select(RetrievalBaselineIndexVector).where(
                    RetrievalBaselineIndexVector.index_id == index_id
                )
            )
        )
        vectors_by_document = {row.index_document_id: row for row in vectors}
        if len(vectors) != len(documents):
            return "index_vector_count_mismatch"

        reconstructed: list[_DocumentArtifact] = []
        total_tokens = 0
        expected_term_rows = 0
        for ordinal, (source, document) in enumerate(zip(sources, documents)):
            expected_text = baseline_ranking_document(
                source.repository_name,
                source.relative_path,
                source.content,
            )
            expected_hash = hashlib.sha256(expected_text.encode("utf-8")).hexdigest()
            expected_terms = lexical_term_frequencies(expected_text)
            expected_count = sum(frequency for _, frequency in expected_terms)
            if (
                document.ordinal != ordinal
                or document.corpus_file_id != source.file_id
                or document.repository_id != source.repository_id
                or document.repository_name != source.repository_name
                or document.relative_path != source.relative_path
                or document.ranking_text != expected_text
                or document.source_content_hash != source.content_hash
                or document.indexed_document_hash != expected_hash
                or document.token_count != expected_count
            ):
                return "index_document_mismatch"
            actual_terms = tuple(
                sorted(
                    (row.term, row.term_frequency)
                    for row in terms_by_document.get(document.index_document_id, [])
                )
            )
            for row in terms_by_document.get(document.index_document_id, []):
                if (
                    row.term_frequency <= 0
                    or row.term_hash
                    != hashlib.sha256(row.term.encode("utf-8")).hexdigest()
                ):
                    return "index_lexical_term_invalid"
            if actual_terms != expected_terms:
                return "index_lexical_coverage_mismatch"
            expected_term_rows += len(expected_terms)

            vector = vectors_by_document.get(document.index_document_id)
            if vector is None or vector.dimension != identity.dimension:
                return "index_vector_dimension_mismatch"
            if len(vector.vector_bytes) != identity.dimension * 4:
                return "index_vector_size_mismatch"
            if hashlib.sha256(vector.vector_bytes).hexdigest() != vector.vector_hash:
                return "index_vector_hash_mismatch"
            values = np.frombuffer(vector.vector_bytes, dtype="<f4")
            if values.shape != (identity.dimension,) or not np.isfinite(values).all():
                return "index_vector_nonfinite"

            total_tokens += expected_count
            reconstructed.append(
                _DocumentArtifact(
                    source=source,
                    ordinal=ordinal,
                    ranking_text=expected_text,
                    indexed_document_hash=expected_hash,
                    term_frequencies=expected_terms,
                    token_count=expected_count,
                    vector_bytes=vector.vector_bytes,
                    vector_hash=vector.vector_hash,
                )
            )
        if total_tokens != build.total_token_count:
            return "index_token_count_mismatch"
        if len(terms) != expected_term_rows:
            return "index_lexical_term_count_mismatch"
        document_hash, lexical_hash, dense_hash = _manifest_hashes(reconstructed)
        if document_hash != build.document_manifest_hash:
            return "index_document_manifest_mismatch"
        if lexical_hash != build.lexical_manifest_hash:
            return "index_lexical_manifest_mismatch"
        if dense_hash != build.dense_manifest_hash:
            return "index_dense_manifest_mismatch"
        return None

    @classmethod
    def publish(cls, session: Session, index_id: str) -> None:
        build = session.get(RetrievalBaselineIndexBuild, index_id)
        if build is None:
            raise BaselineIndexBuildError("index_build_absent", "index build is absent")
        if build.status != BaselineIndexBuildStatus.VALIDATED.value:
            raise BaselineIndexBuildError(
                "index_build_not_validated",
                "only a fully validated index build can be published",
                index_id=index_id,
            )
        error = cls.validation_error(session, index_id)
        if error is not None:
            raise BaselineIndexBuildError(
                error,
                "staged baseline index failed publication validation",
                index_id=index_id,
            )
        generation = session.get(RetrievalCorpusGeneration, build.generation_id)
        assert generation is not None
        statement = select(RetrievalCorpus).where(
            RetrievalCorpus.corpus_id == generation.corpus_id
        )
        if session.get_bind().dialect.name == "postgresql":
            statement = statement.with_for_update()
        corpus = session.scalar(statement)
        assert corpus is not None
        publication = session.get(RetrievalBaselineIndexPublication, corpus.corpus_id)
        prior_index_id = publication.index_id if publication is not None else None
        if prior_index_id is not None and prior_index_id != index_id:
            prior = session.get(RetrievalBaselineIndexBuild, prior_index_id)
            if (
                prior is not None
                and prior.status == BaselineIndexBuildStatus.COMPATIBLE.value
            ):
                prior.status = BaselineIndexBuildStatus.STALE.value

        now = datetime.now(timezone.utc)
        if publication is None:
            publication = RetrievalBaselineIndexPublication(
                corpus_id=corpus.corpus_id,
                index_id=index_id,
            )
            session.add(publication)
        else:
            publication.index_id = index_id
            publication.published_at = now
        build.status = BaselineIndexBuildStatus.COMPATIBLE.value
        build.failure_code = None
        build.published_at = now
        CorpusLifecycle.record_index_state(
            session,
            build.generation_id,
            status=IndexStateStatus.COMPATIBLE,
            tokenizer_version=build.tokenizer_version,
            embedding_provider=build.embedding_provider,
            embedding_model=build.embedding_model,
            embedding_revision=build.embedding_revision,
            embedding_dimension=build.embedding_dimension,
            embedding_fingerprint=build.embedding_fingerprint,
            engine_config_fingerprint=build.engine_config_fingerprint,
            indexed_file_count=build.indexed_document_count,
        )


def assess_baseline_index(
    session: Session,
    *,
    scope_key: str,
    embedding: BaselineEmbeddingIdentity,
) -> BaselineIndexReadiness:
    """Fail-closed readiness check for the published durable index only."""

    corpus = session.scalar(
        select(RetrievalCorpus).where(RetrievalCorpus.scope_key == scope_key)
    )
    if corpus is None or corpus.active_generation_id is None:
        return BaselineIndexReadiness(False, "active_corpus_absent")
    common = {
        "corpus_id": corpus.corpus_id,
        "generation_id": corpus.active_generation_id,
    }
    publication = session.get(RetrievalBaselineIndexPublication, corpus.corpus_id)
    if publication is None or publication.index_id is None:
        return BaselineIndexReadiness(False, "compatible_index_absent", **common)
    build = session.get(RetrievalBaselineIndexBuild, publication.index_id)
    if build is None:
        return BaselineIndexReadiness(
            False,
            "published_index_absent",
            index_id=publication.index_id,
            **common,
        )
    if build.generation_id != corpus.active_generation_id:
        return BaselineIndexReadiness(
            False,
            "published_index_corpus_stale",
            index_id=build.index_id,
            **common,
        )
    if build.status != BaselineIndexBuildStatus.COMPATIBLE.value:
        return BaselineIndexReadiness(
            False,
            f"index_{build.status}",
            index_id=build.index_id,
            **common,
        )
    requirements = baseline_index_requirements(embedding)
    comparisons = (
        (build.tokenizer_version, requirements.tokenizer_version, "index_tokenizer_mismatch"),
        (build.embedding_provider, requirements.embedding_provider, "index_provider_mismatch"),
        (build.embedding_model, requirements.embedding_model, "index_model_mismatch"),
        (build.embedding_revision, requirements.embedding_revision, "index_revision_mismatch"),
        (build.embedding_dimension, requirements.embedding_dimension, "index_dimension_mismatch"),
        (build.embedding_fingerprint, requirements.embedding_fingerprint, "index_embedding_fingerprint_mismatch"),
        (build.engine_config_fingerprint, requirements.engine_config_fingerprint, "index_config_mismatch"),
    )
    for actual, expected, code in comparisons:
        if actual != expected:
            return BaselineIndexReadiness(
                False,
                code,
                index_id=build.index_id,
                **common,
            )
    error = BaselineIndexLifecycle.validation_error(session, build.index_id)
    if error is not None:
        return BaselineIndexReadiness(
            False,
            error,
            index_id=build.index_id,
            **common,
        )
    state = session.get(RetrievalIndexState, build.generation_id)
    if state is None or state.status != IndexStateStatus.COMPATIBLE.value:
        return BaselineIndexReadiness(
            False,
            "index_state_incompatible",
            index_id=build.index_id,
            **common,
        )
    state_comparisons = (
        (state.corpus_manifest_hash, build.corpus_manifest_hash),
        (state.tokenizer_version, build.tokenizer_version),
        (state.embedding_provider, build.embedding_provider),
        (state.embedding_model, build.embedding_model),
        (state.embedding_revision, build.embedding_revision),
        (state.embedding_dimension, build.embedding_dimension),
        (state.embedding_fingerprint, build.embedding_fingerprint),
        (state.engine_config_fingerprint, build.engine_config_fingerprint),
        (state.indexed_file_count, build.indexed_document_count),
    )
    if any(actual != expected for actual, expected in state_comparisons):
        return BaselineIndexReadiness(
            False,
            "index_state_metadata_mismatch",
            index_id=build.index_id,
            **common,
        )
    return BaselineIndexReadiness(
        True,
        "ready",
        index_id=build.index_id,
        **common,
    )


class BaselineIndexBuilder:
    """Construct, validate, and atomically publish one durable baseline index."""

    def __init__(
        self,
        session_factory: Any,
        *,
        publish_index: Callable[[Session, str], None] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._publish_index = publish_index or BaselineIndexLifecycle.publish

    def _mark(
        self,
        index_id: str,
        *,
        status: BaselineIndexBuildStatus,
        failure_code: str,
    ) -> None:
        with self._session_factory.begin() as session:
            build = session.get(RetrievalBaselineIndexBuild, index_id)
            if build is not None and build.status != BaselineIndexBuildStatus.COMPATIBLE.value:
                build.status = status.value
                build.failure_code = failure_code

    def _raise_recorded(
        self,
        index_id: str,
        code: str,
        message: str,
        *,
        status: BaselineIndexBuildStatus = BaselineIndexBuildStatus.FAILED,
    ) -> None:
        self._mark(index_id, status=status, failure_code=code)
        raise BaselineIndexBuildError(code, message, index_id=index_id)

    def build(
        self,
        *,
        generation_id: str,
        index_version: str,
        embedding: BaselineEmbeddingIdentity,
        provider: BaselineIndexEmbeddingProvider | None,
    ) -> BaselineIndexBuildResult:
        index_version = _canonical_identifier(index_version, "index version", 128)
        requirements = baseline_index_requirements(embedding)
        preflight_error: str | None = None
        sources: tuple[_SourceFile, ...] = ()
        with self._session_factory.begin() as session:
            generation = session.get(RetrievalCorpusGeneration, generation_id)
            if generation is None:
                raise BaselineIndexBuildError(
                    "corpus_generation_absent",
                    "baseline index requires an existing corpus generation",
                )
            context_error, _ = BaselineIndexLifecycle._active_context_error(
                session,
                generation,
            )
            try:
                sources = _source_files(session, generation_id)
            except BaselineIndexBuildError as exc:
                context_error = context_error or exc.code
            build = RetrievalBaselineIndexBuild(
                generation_id=generation_id,
                index_version=index_version,
                index_schema_version=BASELINE_INDEX_SCHEMA_VERSION,
                document_format_version=BASELINE_DOCUMENT_FORMAT_VERSION,
                corpus_manifest_hash=generation.manifest_hash or "0" * 64,
                tokenizer_version=BASELINE_TOKENIZER_VERSION,
                embedding_provider=embedding.provider,
                embedding_model=embedding.model,
                embedding_revision=embedding.revision,
                embedding_dimension=embedding.dimension,
                embedding_fingerprint=embedding.fingerprint,
                engine_config_fingerprint=requirements.engine_config_fingerprint,
                expected_document_count=len(sources),
            )
            session.add(build)
            session.flush()
            index_id = build.index_id
            if context_error:
                build.status = (
                    BaselineIndexBuildStatus.STALE.value
                    if context_error == "corpus_generation_stale"
                    else BaselineIndexBuildStatus.INCOMPATIBLE.value
                )
                build.failure_code = context_error
                preflight_error = context_error

        if preflight_error is not None:
            raise BaselineIndexBuildError(
                preflight_error,
                "corpus generation is not complete and active",
                index_id=index_id,
            )

        mismatch = embedding_provider_mismatch_code(provider, embedding)
        if mismatch is not None:
            self._raise_recorded(
                index_id,
                mismatch,
                "embedding adapter does not match the pinned baseline identity",
            )
        assert provider is not None

        lexical_documents: list[
            tuple[_SourceFile, int, str, str, tuple[tuple[str, int], ...], int]
        ] = []
        for ordinal, source in enumerate(sources):
            ranking_text = baseline_ranking_document(
                source.repository_name,
                source.relative_path,
                source.content,
            )
            term_frequencies = lexical_term_frequencies(ranking_text)
            lexical_documents.append(
                (
                    source,
                    ordinal,
                    ranking_text,
                    hashlib.sha256(ranking_text.encode("utf-8")).hexdigest(),
                    term_frequencies,
                    sum(frequency for _, frequency in term_frequencies),
                )
            )

        try:
            raw_vectors = list(
                provider.embed([document[2] for document in lexical_documents])
            )
        except Exception as exc:  # noqa: BLE001 - external adapter boundary
            self._raise_recorded(
                index_id,
                "embedding_adapter_failed",
                f"embedding adapter failed: {type(exc).__name__}",
            )
        if len(raw_vectors) != len(lexical_documents):
            self._raise_recorded(
                index_id,
                "embedding_vector_count_mismatch",
                "embedding adapter returned the wrong vector count",
            )

        artifacts: list[_DocumentArtifact] = []
        for document, raw_vector in zip(lexical_documents, raw_vectors):
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    vector = np.asarray(raw_vector, dtype="<f4")
            except (TypeError, ValueError):
                self._raise_recorded(
                    index_id,
                    "embedding_vector_invalid",
                    "embedding adapter returned an invalid vector",
                )
            if vector.shape != (embedding.dimension,):
                self._raise_recorded(
                    index_id,
                    "embedding_dimension_mismatch",
                    "embedding vector has the wrong dimension",
                )
            if not np.isfinite(vector).all():
                self._raise_recorded(
                    index_id,
                    "embedding_vector_nonfinite",
                    "embedding vector contains NaN or infinity",
                )
            vector_bytes = vector.tobytes(order="C")
            artifacts.append(
                _DocumentArtifact(
                    source=document[0],
                    ordinal=document[1],
                    ranking_text=document[2],
                    indexed_document_hash=document[3],
                    term_frequencies=document[4],
                    token_count=document[5],
                    vector_bytes=vector_bytes,
                    vector_hash=hashlib.sha256(vector_bytes).hexdigest(),
                )
            )

        document_hash, lexical_hash, dense_hash = _manifest_hashes(artifacts)
        try:
            with self._session_factory.begin() as session:
                build = session.get(RetrievalBaselineIndexBuild, index_id)
                assert build is not None
                generation = session.get(RetrievalCorpusGeneration, generation_id)
                assert generation is not None
                context_error, _ = BaselineIndexLifecycle._active_context_error(
                    session,
                    generation,
                )
                current_sources = _source_files(session, generation_id)
                if context_error or current_sources != sources:
                    raise BaselineIndexBuildError(
                        context_error or "corpus_changed_during_index_build",
                        "corpus changed while baseline index artifacts were built",
                        index_id=index_id,
                    )
                for artifact in artifacts:
                    document = RetrievalBaselineIndexDocument(
                        index_id=index_id,
                        corpus_file_id=artifact.source.file_id,
                        ordinal=artifact.ordinal,
                        repository_id=artifact.source.repository_id,
                        repository_name=artifact.source.repository_name,
                        relative_path=artifact.source.relative_path,
                        ranking_text=artifact.ranking_text,
                        source_content_hash=artifact.source.content_hash,
                        indexed_document_hash=artifact.indexed_document_hash,
                        token_count=artifact.token_count,
                    )
                    session.add(document)
                    session.flush()
                    for term, frequency in artifact.term_frequencies:
                        session.add(
                            RetrievalBaselineIndexTerm(
                                index_id=index_id,
                                index_document_id=document.index_document_id,
                                term_hash=hashlib.sha256(
                                    term.encode("utf-8")
                                ).hexdigest(),
                                term=term,
                                term_frequency=frequency,
                            )
                        )
                    session.add(
                        RetrievalBaselineIndexVector(
                            index_document_id=document.index_document_id,
                            index_id=index_id,
                            dimension=embedding.dimension,
                            vector_bytes=artifact.vector_bytes,
                            vector_hash=artifact.vector_hash,
                        )
                    )
                build.indexed_document_count = len(artifacts)
                build.total_token_count = sum(
                    artifact.token_count for artifact in artifacts
                )
                build.document_manifest_hash = document_hash
                build.lexical_manifest_hash = lexical_hash
                build.dense_manifest_hash = dense_hash
                build.status = BaselineIndexBuildStatus.VALIDATED.value
                build.failure_code = None
                build.validated_at = datetime.now(timezone.utc)
        except BaselineIndexBuildError as exc:
            self._raise_recorded(
                index_id,
                exc.code,
                str(exc),
                status=(
                    BaselineIndexBuildStatus.STALE
                    if exc.code == "corpus_generation_stale"
                    else BaselineIndexBuildStatus.INCOMPATIBLE
                ),
            )
        except Exception as exc:  # noqa: BLE001 - persist any staging failure
            self._raise_recorded(
                index_id,
                "artifact_staging_failed",
                f"artifact staging failed: {type(exc).__name__}",
            )

        try:
            with self._session_factory.begin() as session:
                self._publish_index(session, index_id)
        except BaselineIndexBuildError as exc:
            self._raise_recorded(
                index_id,
                exc.code,
                str(exc),
                status=(
                    BaselineIndexBuildStatus.STALE
                    if exc.code == "corpus_generation_stale"
                    else BaselineIndexBuildStatus.INCOMPATIBLE
                ),
            )
        except Exception as exc:  # noqa: BLE001 - injected publication boundary
            self._raise_recorded(
                index_id,
                "index_publication_failed",
                f"index publication failed: {type(exc).__name__}",
            )

        return BaselineIndexBuildResult(
            index_id=index_id,
            generation_id=generation_id,
            index_version=index_version,
            document_count=len(artifacts),
            total_token_count=sum(artifact.token_count for artifact in artifacts),
            document_manifest_hash=document_hash,
            lexical_manifest_hash=lexical_hash,
            dense_manifest_hash=dense_hash,
            status=BaselineIndexBuildStatus.COMPATIBLE,
        )
