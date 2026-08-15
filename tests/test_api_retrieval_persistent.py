from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import Engine, select

from compair_core import db as core_db
from compair_core.compair.retrieval.baseline import BM25_B, BM25_K1
from compair_core.compair.retrieval.corpus import (
    BaselineIndexBuildStatus,
    CorpusFileInput,
    IndexStateStatus,
    RetrievalBaselineIndexBuild,
    RetrievalBaselineIndexDocument,
    RetrievalIndexState,
    ensure_retrieval_corpus_schema,
)
from compair_core.compair.retrieval.factory import retrieve_reference_evidence
from compair_core.compair.retrieval.indexing import (
    BaselineEmbeddingIdentity,
    BaselineIndexBuilder,
)
from compair_core.compair.retrieval.ingestion import (
    CorpusIngestionService,
    CorpusRepositoryInput,
    CorpusSnapshotInput,
)
from compair_core.compair.retrieval.persistent import (
    PersistentBaselineV1Retriever,
    _PersistentDocument,
)
from compair_core.compair.retrieval.types import (
    RESULT_SCHEMA_VERSION,
    RetrievalQueryOrigin,
    RetrievalRequest,
    RetrievalStatus,
)

SCOPE_KEY = "group:persistent-retrieval"
CHANGED_REPOSITORY_ID = "repo-changed"
_TEST_ENGINES: list[Engine] = []


@pytest.fixture(autouse=True)
def _dispose_test_engines():
    try:
        yield
    finally:
        while _TEST_ENGINES:
            _TEST_ENGINES.pop().dispose()


def _sessions(path: Path):
    engine = core_db.create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False},
    )
    _TEST_ENGINES.append(engine)
    ensure_retrieval_corpus_schema(engine)
    return engine, core_db.sessionmaker(engine, expire_on_commit=False)


def _identity(*, fingerprint: str | None = None) -> BaselineEmbeddingIdentity:
    return BaselineEmbeddingIdentity(
        provider="fixture-baseline-adapter",
        model="fixture/dense-two",
        revision="fixture-revision-1",
        dimension=2,
        fingerprint=fingerprint
        or hashlib.sha256(b"fixture-dense-two-revision-1").hexdigest(),
    )


class FixtureEmbeddingProvider:
    def __init__(
        self,
        identity: BaselineEmbeddingIdentity,
        *,
        document_vectors: dict[str, tuple[float, ...]] | None = None,
        query_vector: tuple[float, ...] = (1.0, 0.0),
        query_mode: str = "ok",
    ) -> None:
        self.provider = identity.provider
        self.model = identity.model
        self.revision = identity.revision
        self.dimension = identity.dimension
        self.fingerprint = identity.fingerprint
        self.document_vectors = document_vectors or {}
        self.query_vector = query_vector
        self.query_mode = query_mode
        self.query_calls: list[str] = []

    def embed(self, texts):
        output = []
        for text in texts:
            if text.startswith("Repository file: "):
                path = text.splitlines()[0].removeprefix("Repository file: ")
                output.append(self.document_vectors.get(path, (1.0, 0.0)))
                continue
            self.query_calls.append(text)
            if self.query_mode == "failure":
                raise RuntimeError("fixture query adapter failure")
            if self.query_mode == "dimension":
                output.append((1.0, 0.0, 0.0))
            elif self.query_mode == "nan":
                output.append((math.nan, 0.0))
            elif self.query_mode == "infinite":
                output.append((math.inf, 0.0))
            else:
                output.append(self.query_vector)
        return output


def _repository(file_count: int, *, revision: str) -> CorpusRepositoryInput:
    return CorpusRepositoryInput(
        repository_id="repo-peer",
        repository_name="peer",
        expected_file_count=file_count,
        repository_revision=revision,
        document_id="document-peer",
        document_revision=f"document-{revision}",
    )


def _changed(version: str) -> CorpusRepositoryInput:
    return CorpusRepositoryInput(
        repository_id=CHANGED_REPOSITORY_ID,
        repository_name="changed",
        expected_file_count=0,
        repository_revision=f"changed-{version}",
        document_id="document-changed",
        document_revision=f"document-changed-{version}",
    )


def _file(path: str, content: str) -> CorpusFileInput:
    return CorpusFileInput.supported_text(
        repository_id="repo-peer",
        repository_name="peer",
        relative_path=path,
        content=content,
    )


def _ingest(
    SessionMaker,
    *,
    version: str = "generation-1",
    files: tuple[CorpusFileInput, ...],
):
    snapshot = CorpusSnapshotInput.create(
        scope_key=SCOPE_KEY,
        generation_version=version,
        changed_repository=_changed(version),
        sibling_repositories=(
            _repository(len(files), revision=f"peer-{version}"),
        ),
        files=files,
        producer_id="trusted-persistent-test-producer",
        producer_version="1.0",
        snapshot_id=f"snapshot-{version}",
        source_revision=f"changed-{version}",
        source_manifest_hash=hashlib.sha256(
            f"source-{version}".encode()
        ).hexdigest(),
    )
    return CorpusIngestionService(SessionMaker).ingest(snapshot)


def _build(
    SessionMaker,
    generation_id: str,
    provider: FixtureEmbeddingProvider,
    *,
    version: str = "index-1",
):
    identity = BaselineEmbeddingIdentity(
        provider=provider.provider,
        model=provider.model,
        revision=provider.revision,
        dimension=provider.dimension,
        fingerprint=provider.fingerprint,
    )
    return BaselineIndexBuilder(SessionMaker).build(
        generation_id=generation_id,
        index_version=version,
        embedding=identity,
        provider=provider,
    )


def _request(
    *,
    query: str = "alpha alpha",
    version: str = "generation-1",
    changed_repository_id: str = CHANGED_REPOSITORY_ID,
) -> RetrievalRequest:
    return RetrievalRequest(
        request_id="persistent-request",
        changed_repository=None,
        repository_roots=(),
        corpus_version=version,
        retrieval_query=query,
        retrieval_query_origin=RetrievalQueryOrigin.EXPLICIT,
        corpus_complete=True,
        corpus_scope_key=SCOPE_KEY,
        changed_repository_id=changed_repository_id,
    )


def _published_retriever(
    tmp_path: Path,
    *,
    files: tuple[CorpusFileInput, ...],
    document_vectors: dict[str, tuple[float, ...]] | None = None,
    query_vector: tuple[float, ...] = (1.0, 0.0),
    evidence_filter=None,
):
    engine, SessionMaker = _sessions(tmp_path / "persistent.db")
    generation = _ingest(SessionMaker, files=files)
    identity = _identity()
    build_provider = FixtureEmbeddingProvider(
        identity,
        document_vectors=document_vectors,
    )
    build = _build(SessionMaker, generation.generation_id, build_provider)
    query_provider = FixtureEmbeddingProvider(
        identity,
        document_vectors=document_vectors,
        query_vector=query_vector,
    )
    retriever = PersistentBaselineV1Retriever(
        SessionMaker,
        query_provider,
        evidence_filter=evidence_filter,
    )
    return engine, SessionMaker, build, query_provider, retriever


def test_persistent_baseline_exact_bm25_rrf_and_result_provenance(
    tmp_path: Path,
) -> None:
    query = "alpha alpha retrieval-secret-marker"
    _, _, build, provider, retriever = _published_retriever(
        tmp_path,
        files=(
            _file("src/one.txt", "alpha alpha beta"),
            _file("src/two.txt", "beta gamma"),
        ),
        document_vectors={
            "peer/src/one.txt": (1.0, 0.0),
            "peer/src/two.txt": (0.0, 1.0),
        },
        query_vector=(1.0, 0.0),
    )

    result = retrieve_reference_evidence(
        engine_name="baseline_v1",
        baseline_retriever=retriever,
        request=_request(query=query),
    )

    normalization = BM25_K1 * (1.0 - BM25_B + BM25_B * 15 / 14)
    # The frozen tokenizer emits both the full token and its split token, so
    # two raw ``alpha`` occurrences produce a query frequency of four; the
    # additional trace-leak sentinel is absent from every document.
    expected_bm25 = 4 * math.log(2) * (4 * (BM25_K1 + 1) / (4 + normalization))
    assert result.status is RetrievalStatus.OK
    assert [candidate.path for candidate in result.candidates] == [
        "peer/src/one.txt",
        "peer/src/two.txt",
    ]
    assert result.candidates[0].bm25_score == pytest.approx(expected_bm25)
    assert result.candidates[1].bm25_score == 0.0
    assert [candidate.bm25_rank for candidate in result.candidates] == [1, 2]
    assert [candidate.dense_rank for candidate in result.candidates] == [1, 2]
    assert result.candidates[0].dense_score == 1.0
    assert result.candidates[1].dense_score == 0.0
    assert result.candidates[0].rrf_score == pytest.approx(2 / 61)
    assert result.candidates[1].rrf_score == pytest.approx(2 / 62)
    assert all(candidate.document_id for candidate in result.candidates)
    assert result.evidence[0].document_id == result.candidates[0].document_id
    assert result.evidence[0].bm25_score == result.candidates[0].bm25_score
    assert result.evidence[0].dense_rank == result.candidates[0].dense_rank
    assert result.evidence[0].rrf_score == result.candidates[0].rrf_score
    assert result.schema_version == RESULT_SCHEMA_VERSION
    assert result.engine == "baseline_v1"
    assert result.engine_version == "baseline_v1.persistent.v1"
    assert result.corpus_scope_key == SCOPE_KEY
    assert result.corpus_manifest_hash is not None
    assert result.index_id == build.index_id
    assert result.index_version == "index-1"
    assert result.index_schema_version == "baseline-index.v1"
    assert result.index_fingerprint is not None
    assert len(result.index_fingerprint) == 64
    assert result.embedding_provider == provider.provider
    assert result.embedding_fingerprint == provider.fingerprint
    assert result.query_provenance.sha256 == hashlib.sha256(query.encode()).hexdigest()
    assert result.query_provenance.length == len(query)
    assert result.query_provenance.origin is RetrievalQueryOrigin.EXPLICIT
    assert query not in repr(result)
    assert provider.query_calls == [query]
    assert result.fallback_engine is None


def test_float32_near_tie_uses_path_order_not_python_double_order(
    tmp_path: Path,
) -> None:
    _, _, _, _, retriever = _published_retriever(
        tmp_path,
        files=(
            _file("src/a.txt", "first"),
            _file("src/z.txt", "second"),
        ),
        document_vectors={
            "peer/src/a.txt": (1.0, 0.0),
            "peer/src/z.txt": (1.0, 1e-8),
        },
        query_vector=(1.0, 1.0),
    )
    assert sum(a * b for a, b in zip((1.0, 1e-8), (1.0, 1.0))) > 1.0

    first = retriever.retrieve(_request(query="no lexical match"))
    second = retriever.retrieve(_request(query="no lexical match"))

    assert np.float32(1.0) + np.float32(1e-8) == np.float32(1.0)
    assert [candidate.dense_score for candidate in first.candidates] == [1.0, 1.0]
    assert [candidate.path for candidate in first.candidates] == [
        "peer/src/a.txt",
        "peer/src/z.txt",
    ]
    assert first.candidates == second.candidates


def test_stable_document_identity_breaks_an_impossible_valid_path_tie() -> None:
    def document(document_id: str) -> _PersistentDocument:
        return _PersistentDocument(
            index_document_id=document_id,
            corpus_file_id=f"file-{document_id}",
            repository_id="repo-peer",
            repository_name="peer",
            relative_path="same.txt",
            content=document_id,
            content_hash=hashlib.sha256(document_id.encode()).hexdigest(),
            byte_size=len(document_id),
            token_count=1,
            term_frequencies={"token": 1},
            vector_bytes=b"",
        )

    ranked = PersistentBaselineV1Retriever._rank(
        (document("document-b"), document("document-a")),
        (0.0, 0.0),
        (0.0, 0.0),
    )

    assert [candidate.document_id for candidate in ranked] == [
        "document-a",
        "document-b",
    ]


def test_changed_repository_identity_is_verified_before_query_or_ranking(
    tmp_path: Path,
) -> None:
    _, SessionMaker, build, provider, retriever = _published_retriever(
        tmp_path,
        files=(_file("src/one.txt", "alpha"),),
    )

    mismatch = retriever.retrieve(
        _request(changed_repository_id="different-changed-repository")
    )
    assert mismatch.status is RetrievalStatus.INSUFFICIENT
    assert mismatch.error.code == "changed_repository_mismatch"
    assert mismatch.candidates == ()
    assert provider.query_calls == []

    with SessionMaker() as session:
        document = session.scalar(
            select(RetrievalBaselineIndexDocument).where(
                RetrievalBaselineIndexDocument.index_id == build.index_id
            )
        )
        document.repository_id = CHANGED_REPOSITORY_ID
        session.commit()

    contaminated = retriever.retrieve(_request())
    assert contaminated.status is RetrievalStatus.ERROR
    assert contaminated.error.code == "changed_repository_in_published_index"
    assert contaminated.candidates == ()
    assert provider.query_calls == []


def test_top_six_is_the_only_filter_dedupe_and_budget_refill_window(
    tmp_path: Path,
) -> None:
    files = (
        _file("01-filtered.txt", "excluded content"),
        _file("02-original.txt", " duplicate body "),
        _file("03-duplicate.txt", "duplicate body"),
        _file("04-long.txt", "a" * 9_000),
        _file("05-clipped.txt", "b" * 9_000),
        _file("06-unused.txt", "sixth candidate"),
        _file("07-outside.txt", "must not refill"),
    )
    _, _, _, _, retriever = _published_retriever(
        tmp_path,
        files=files,
        evidence_filter=lambda candidate: candidate.relative_path != "01-filtered.txt",
    )

    result = retriever.retrieve(_request(query="unmatched-query-token"))

    assert result.status is RetrievalStatus.OK
    assert result.candidate_count == 7
    assert result.retrieved_count == 6
    assert result.filtered_count == 1
    assert result.duplicate_count == 1
    assert result.refill_count == 1
    assert [evidence.relative_path for evidence in result.evidence] == [
        "02-original.txt",
        "04-long.txt",
        "05-clipped.txt",
    ]
    assert result.evidence[2].render_truncated is True
    assert len(result.evidence[2].content) == 6_986
    assert result.evidence_characters == 16_000
    assert all(
        candidate.relative_path != "07-outside.txt"
        for candidate in result.candidates[: result.retrieved_count]
    )
    assert all(evidence.relative_path != "07-outside.txt" for evidence in result.evidence)


def test_evidence_item_budget_stops_at_four(tmp_path: Path) -> None:
    _, _, _, _, retriever = _published_retriever(
        tmp_path,
        files=tuple(_file(f"0{number}.txt", f"content {number}") for number in range(1, 6)),
    )

    result = retriever.retrieve(_request(query="unmatched-query-token"))

    assert result.status is RetrievalStatus.OK
    assert len(result.evidence) == 4
    assert [item.relative_path for item in result.evidence] == [
        "01.txt",
        "02.txt",
        "03.txt",
        "04.txt",
    ]


def test_evidence_uses_the_same_twelve_k_content_as_the_ranking_document(
    tmp_path: Path,
) -> None:
    content = "z" * 20_000
    _, _, _, _, retriever = _published_retriever(
        tmp_path,
        files=(_file("large.txt", content),),
    )

    result = retriever.retrieve(_request(query="unmatched-query-token"))

    assert result.status is RetrievalStatus.OK
    assert result.candidates[0].content == content[:12_000]
    assert result.evidence[0].content == content[:12_000]
    assert result.evidence[0].render_truncated is False
    assert result.evidence_characters == 12_000


@pytest.mark.parametrize(
    ("provider_factory", "expected_code"),
    (
        (lambda identity: None, "embedding_adapter_unavailable"),
        (
            lambda identity: FixtureEmbeddingProvider(
                _identity(fingerprint="b" * 64)
            ),
            "embedding_fingerprint_mismatch",
        ),
        (
            lambda identity: FixtureEmbeddingProvider(identity, query_mode="dimension"),
            "query_embedding_dimension_mismatch",
        ),
        (
            lambda identity: FixtureEmbeddingProvider(identity, query_mode="nan"),
            "query_embedding_nonfinite",
        ),
        (
            lambda identity: FixtureEmbeddingProvider(identity, query_mode="infinite"),
            "query_embedding_nonfinite",
        ),
        (
            lambda identity: FixtureEmbeddingProvider(identity, query_mode="failure"),
            "query_embedding_failed",
        ),
    ),
)
def test_query_embedding_failures_are_explicit_and_never_fall_back(
    tmp_path: Path,
    provider_factory,
    expected_code: str,
) -> None:
    _, SessionMaker = _sessions(tmp_path / f"{expected_code}.db")
    generation = _ingest(
        SessionMaker,
        files=(_file("src/one.txt", "alpha"),),
    )
    identity = _identity()
    _build(
        SessionMaker,
        generation.generation_id,
        FixtureEmbeddingProvider(identity),
    )
    retriever = PersistentBaselineV1Retriever(
        SessionMaker,
        provider_factory(identity),
    )

    result = retriever.retrieve(_request())

    assert result.status is RetrievalStatus.ERROR
    assert result.error.code == expected_code
    assert result.evidence == ()
    assert result.fallback_engine is None


@pytest.mark.parametrize(
    ("status", "expected_status", "expected_code"),
    (
        (
            BaselineIndexBuildStatus.VALIDATED,
            RetrievalStatus.INSUFFICIENT,
            "index_validated",
        ),
        (
            BaselineIndexBuildStatus.INCOMPATIBLE,
            RetrievalStatus.ERROR,
            "index_incompatible",
        ),
        (
            BaselineIndexBuildStatus.FAILED,
            RetrievalStatus.ERROR,
            "index_failed",
        ),
    ),
)
def test_noncompatible_publication_states_fail_closed(
    tmp_path: Path,
    status: BaselineIndexBuildStatus,
    expected_status: RetrievalStatus,
    expected_code: str,
) -> None:
    _, SessionMaker, build, provider, retriever = _published_retriever(
        tmp_path,
        files=(_file("src/one.txt", "alpha"),),
    )
    with SessionMaker() as session:
        row = session.get(RetrievalBaselineIndexBuild, build.index_id)
        row.status = status.value
        session.commit()

    result = retriever.retrieve(_request())

    assert result.status is expected_status
    assert result.error.code == expected_code
    assert result.evidence == ()
    assert provider.query_calls == []


def test_published_config_fingerprint_mismatch_fails_closed(
    tmp_path: Path,
) -> None:
    _, SessionMaker, build, provider, retriever = _published_retriever(
        tmp_path,
        files=(_file("src/one.txt", "alpha"),),
    )
    with SessionMaker() as session:
        row = session.get(RetrievalBaselineIndexBuild, build.index_id)
        row.engine_config_fingerprint = "c" * 64
        session.commit()

    result = retriever.retrieve(_request())

    assert result.status is RetrievalStatus.ERROR
    assert result.error.code == "index_config_mismatch"
    assert result.evidence == ()
    assert provider.query_calls == []


def test_incomplete_compatible_index_state_is_an_explicit_insufficient_result(
    tmp_path: Path,
) -> None:
    _, SessionMaker, build, provider, retriever = _published_retriever(
        tmp_path,
        files=(_file("src/one.txt", "alpha"),),
    )
    with SessionMaker() as session:
        row = session.get(RetrievalBaselineIndexBuild, build.index_id)
        state = session.get(RetrievalIndexState, row.generation_id)
        state.status = IndexStateStatus.INCOMPLETE.value
        session.commit()

    result = retriever.retrieve(_request())

    assert result.status is RetrievalStatus.INSUFFICIENT
    assert result.error.code == "index_state_incomplete"
    assert result.evidence == ()
    assert provider.query_calls == []


def test_absent_and_stale_publications_have_distinct_insufficient_reasons(
    tmp_path: Path,
) -> None:
    _, SessionMaker = _sessions(tmp_path / "states.db")
    first = _ingest(
        SessionMaker,
        files=(_file("src/one.txt", "alpha"),),
    )
    identity = _identity()
    provider = FixtureEmbeddingProvider(identity)
    retriever = PersistentBaselineV1Retriever(SessionMaker, provider)

    absent = retriever.retrieve(_request())
    assert absent.status is RetrievalStatus.INSUFFICIENT
    assert absent.error.code == "compatible_index_absent"
    assert provider.query_calls == []

    _build(SessionMaker, first.generation_id, FixtureEmbeddingProvider(identity))
    _ingest(
        SessionMaker,
        version="generation-2",
        files=(_file("src/two.txt", "beta"),),
    )
    stale = retriever.retrieve(_request(version="generation-2"))
    assert stale.status is RetrievalStatus.INSUFFICIENT
    assert stale.error.code == "published_index_corpus_stale"
    assert provider.query_calls == []


def test_restart_reads_the_same_compatible_publication(tmp_path: Path) -> None:
    database_path = tmp_path / "restart.db"
    engine, SessionMaker = _sessions(database_path)
    generation = _ingest(
        SessionMaker,
        files=(
            _file("src/a.txt", "alpha"),
            _file("src/b.txt", "beta"),
        ),
    )
    identity = _identity()
    _build(
        SessionMaker,
        generation.generation_id,
        FixtureEmbeddingProvider(identity),
    )
    engine.dispose()

    restarted_engine, RestartedSession = _sessions(database_path)
    result = PersistentBaselineV1Retriever(
        RestartedSession,
        FixtureEmbeddingProvider(identity),
    ).retrieve(_request())
    restarted_engine.dispose()

    assert result.status is RetrievalStatus.OK
    assert [candidate.path for candidate in result.candidates] == [
        "peer/src/a.txt",
        "peer/src/b.txt",
    ]
    assert result.index_id is not None
    assert result.corpus_manifest_hash is not None
