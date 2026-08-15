from __future__ import annotations

import hashlib
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from sqlalchemy import select

from compair_core import db as core_db
from compair_core.compair.retrieval.baseline import (
    BASELINE_TOKENIZER_VERSION,
    RANKING_CONTENT_CHARACTERS,
    baseline_ranking_document,
    frozen_tokens,
)
from compair_core.compair.retrieval.corpus import (
    BaselineIndexBuildStatus,
    CorpusFileInput,
    CorpusIngestionStatus,
    IndexStateStatus,
    RetrievalBaselineIndexBuild,
    RetrievalBaselineIndexDocument,
    RetrievalBaselineIndexPublication,
    RetrievalBaselineIndexTerm,
    RetrievalBaselineIndexVector,
    RetrievalCorpusIngestion,
    RetrievalIndexState,
    ensure_retrieval_corpus_schema,
)
from compair_core.compair.retrieval.indexing import (
    BASELINE_INDEX_SCHEMA_VERSION,
    BASELINE_VECTOR_FORMAT,
    BaselineEmbeddingIdentity,
    BaselineIndexBuilder,
    BaselineIndexBuildError,
    BaselineIndexLifecycle,
    assess_baseline_index,
    lexical_term_frequencies,
)
from compair_core.compair.retrieval.ingestion import (
    CorpusIngestionService,
    CorpusRepositoryInput,
    CorpusSnapshotInput,
)


def _database_url(path: Path) -> str:
    return f"sqlite:///{path}"


def _sessions(path: Path):
    engine = core_db.create_engine(
        _database_url(path),
        connect_args={"check_same_thread": False},
    )
    ensure_retrieval_corpus_schema(engine)
    return engine, core_db.sessionmaker(engine, expire_on_commit=False)


def _identity(*, fingerprint: str | None = None) -> BaselineEmbeddingIdentity:
    return BaselineEmbeddingIdentity(
        provider="fixture-fastembed-adapter",
        model="fixture/bge-small",
        revision="fixture-revision-1",
        dimension=3,
        fingerprint=fingerprint
        or hashlib.sha256(b"fixture-bge-small-revision-1").hexdigest(),
    )


class FixtureIndexProvider:
    def __init__(
        self,
        identity: BaselineEmbeddingIdentity,
        *,
        mode: str = "ok",
    ) -> None:
        self.provider = identity.provider
        self.model = identity.model
        self.revision = identity.revision
        self.dimension = identity.dimension
        self.fingerprint = identity.fingerprint
        self.mode = mode

    def embed(self, texts):
        if self.mode == "failure":
            raise RuntimeError("fixture adapter unavailable during embedding")
        vectors = [
            [float(position + 1), 0.5, -0.25]
            for position, _text in enumerate(texts)
        ]
        if vectors and self.mode == "dimension":
            vectors[0] = [1.0, 2.0]
        if vectors and self.mode == "nan":
            vectors[0][1] = math.nan
        if vectors and self.mode == "infinite":
            vectors[0][1] = math.inf
        return vectors


def _repository(file_count: int, *, revision: str = "peer-revision-1"):
    return CorpusRepositoryInput(
        repository_id="repo-peer",
        repository_name="peer",
        expected_file_count=file_count,
        repository_revision=revision,
        document_id="document-peer",
        document_revision=f"document-{revision}",
    )


def _changed(*, revision: str = "changed-revision-1"):
    return CorpusRepositoryInput(
        repository_id="repo-changed",
        repository_name="changed",
        expected_file_count=0,
        repository_revision=revision,
        document_id="document-changed",
        document_revision=f"document-{revision}",
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
    version: str,
    files: tuple[CorpusFileInput, ...],
    scope_key: str = "group:indexing",
    revision: str = "peer-revision-1",
):
    snapshot = CorpusSnapshotInput.create(
        scope_key=scope_key,
        generation_version=version,
        changed_repository=_changed(revision=f"changed-{version}"),
        sibling_repositories=(_repository(len(files), revision=revision),),
        files=files,
        producer_id="trusted-index-test-producer",
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
    *,
    version: str = "index-1",
    identity: BaselineEmbeddingIdentity | None = None,
    provider=None,
    publish_index=None,
):
    identity = identity or _identity()
    if provider is None:
        provider = FixtureIndexProvider(identity)
    return BaselineIndexBuilder(
        SessionMaker,
        publish_index=publish_index,
    ).build(
        generation_id=generation_id,
        index_version=version,
        embedding=identity,
        provider=provider,
    )


def test_canonical_document_and_frozen_token_statistics_are_exact() -> None:
    content = "x" * (RANKING_CONTENT_CHARACTERS + 5)
    document = baseline_ranking_document("peer", "src/value.py", content)

    assert document == (
        "Repository file: peer/src/value.py\n\n"
        + "x" * RANKING_CONTENT_CHARACTERS
    )
    assert len(document.rsplit("\n\n", 1)[1]) == RANKING_CONTENT_CHARACTERS
    assert baseline_ranking_document("peer", "empty.txt", "") == (
        "Repository file: peer/empty.txt\n\n"
    )
    assert lexical_term_frequencies("") == ()
    assert lexical_term_frequencies("Foo.Bar and alpha alpha") == (
        ("alpha", 4),
        ("bar", 1),
        ("foo", 1),
        ("foo.bar", 1),
    )
    assert [term for term, frequency in lexical_term_frequencies(document) for _ in range(frequency)] == sorted(
        frozen_tokens(document)
    )


def test_persisted_lexical_dense_statistics_and_restart_readiness(tmp_path: Path) -> None:
    database_path = tmp_path / "persistent-index.db"
    engine, SessionMaker = _sessions(database_path)
    corpus = _ingest(
        SessionMaker,
        version="generation-1",
        files=(
            _file("src/one.txt", "alpha alpha beta"),
            _file("src/two.txt", "beta gamma"),
        ),
    )
    identity = _identity()
    result = _build(SessionMaker, corpus.generation_id, identity=identity)

    with SessionMaker() as session:
        build = session.get(RetrievalBaselineIndexBuild, result.index_id)
        documents = tuple(
            session.scalars(
                select(RetrievalBaselineIndexDocument)
                .where(RetrievalBaselineIndexDocument.index_id == result.index_id)
                .order_by(RetrievalBaselineIndexDocument.ordinal)
            )
        )
        terms = tuple(
            session.scalars(
                select(RetrievalBaselineIndexTerm).where(
                    RetrievalBaselineIndexTerm.index_id == result.index_id
                )
            )
        )
        vectors = tuple(
            session.scalars(
                select(RetrievalBaselineIndexVector).where(
                    RetrievalBaselineIndexVector.index_id == result.index_id
                )
            )
        )
        readiness = assess_baseline_index(
            session,
            scope_key="group:indexing",
            embedding=identity,
        )

    assert build.status == BaselineIndexBuildStatus.COMPATIBLE.value
    assert build.index_schema_version == BASELINE_INDEX_SCHEMA_VERSION
    assert build.tokenizer_version == BASELINE_TOKENIZER_VERSION
    assert build.indexed_document_count == 2
    assert build.total_token_count == 28
    assert [document.relative_path for document in documents] == [
        "src/one.txt",
        "src/two.txt",
    ]
    assert [document.token_count for document in documents] == [15, 13]
    assert all(
        document.indexed_document_hash
        == hashlib.sha256(document.ranking_text.encode("utf-8")).hexdigest()
        for document in documents
    )
    alpha_rows = [row for row in terms if row.term == "alpha"]
    beta_rows = [row for row in terms if row.term == "beta"]
    assert [row.term_frequency for row in alpha_rows] == [4]
    assert sorted(row.term_frequency for row in beta_rows) == [2, 2]
    assert len(vectors) == 2
    assert all(row.dimension == 3 and len(row.vector_bytes) == 12 for row in vectors)
    assert readiness.ready is True
    assert readiness.index_id == result.index_id

    engine.dispose()
    restarted_engine, RestartedSession = _sessions(database_path)
    with RestartedSession() as session:
        restarted = assess_baseline_index(
            session,
            scope_key="group:indexing",
            embedding=identity,
        )
        persisted_document = session.get(
            RetrievalBaselineIndexDocument,
            documents[0].index_document_id,
        )
    restarted_engine.dispose()

    assert restarted.ready is True
    assert restarted.index_id == result.index_id
    assert persisted_document.ranking_text == documents[0].ranking_text


def test_empty_file_is_indexed_as_canonical_header_document(tmp_path: Path) -> None:
    _, SessionMaker = _sessions(tmp_path / "empty-file.db")
    corpus = _ingest(
        SessionMaker,
        version="generation-1",
        files=(_file("empty.txt", ""),),
    )
    result = _build(SessionMaker, corpus.generation_id)

    with SessionMaker() as session:
        document = session.scalar(
            select(RetrievalBaselineIndexDocument).where(
                RetrievalBaselineIndexDocument.index_id == result.index_id
            )
        )
        terms = tuple(
            session.scalars(
                select(RetrievalBaselineIndexTerm).where(
                    RetrievalBaselineIndexTerm.index_id == result.index_id
                )
            )
        )

    assert document.ranking_text == "Repository file: peer/empty.txt\n\n"
    assert document.token_count == len(frozen_tokens(document.ranking_text))
    assert document.token_count > 0
    assert sum(term.term_frequency for term in terms) == document.token_count


def test_stale_corpus_and_requested_fingerprint_mismatch_fail_closed(
    tmp_path: Path,
) -> None:
    _, SessionMaker = _sessions(tmp_path / "stale.db")
    first = _ingest(
        SessionMaker,
        version="generation-1",
        files=(_file("src/one.py", "one"),),
        revision="peer-revision-1",
    )
    identity = _identity()
    compatible = _build(SessionMaker, first.generation_id, identity=identity)

    with SessionMaker() as session:
        mismatch = assess_baseline_index(
            session,
            scope_key="group:indexing",
            embedding=_identity(fingerprint="b" * 64),
        )
    assert mismatch.ready is False
    assert mismatch.code == "index_embedding_fingerprint_mismatch"

    _ingest(
        SessionMaker,
        version="generation-2",
        files=(_file("src/two.py", "two"),),
        revision="peer-revision-2",
    )
    with SessionMaker() as session:
        stale = assess_baseline_index(
            session,
            scope_key="group:indexing",
            embedding=identity,
        )
        old_build = session.get(RetrievalBaselineIndexBuild, compatible.index_id)
        old_state = session.get(RetrievalIndexState, first.generation_id)
    assert stale.ready is False
    assert stale.code == "published_index_corpus_stale"
    assert old_build.status == BaselineIndexBuildStatus.STALE.value
    assert old_state.status == IndexStateStatus.STALE.value

    with pytest.raises(BaselineIndexBuildError) as caught:
        _build(
            SessionMaker,
            first.generation_id,
            version="stale-build",
            identity=identity,
        )
    assert caught.value.code == "corpus_generation_stale"
    with SessionMaker() as session:
        attempted = session.get(RetrievalBaselineIndexBuild, caught.value.index_id)
    assert attempted.status == BaselineIndexBuildStatus.STALE.value


@pytest.mark.parametrize(
    ("mode", "provider_factory", "expected_code"),
    (
        (
            "unavailable",
            lambda identity: None,
            "embedding_adapter_unavailable",
        ),
        (
            "fingerprint",
            lambda identity: FixtureIndexProvider(
                _identity(fingerprint="c" * 64)
            ),
            "embedding_fingerprint_mismatch",
        ),
        (
            "dimension",
            lambda identity: FixtureIndexProvider(identity, mode="dimension"),
            "embedding_dimension_mismatch",
        ),
        (
            "nan",
            lambda identity: FixtureIndexProvider(identity, mode="nan"),
            "embedding_vector_nonfinite",
        ),
        (
            "infinite",
            lambda identity: FixtureIndexProvider(identity, mode="infinite"),
            "embedding_vector_nonfinite",
        ),
        (
            "failure",
            lambda identity: FixtureIndexProvider(identity, mode="failure"),
            "embedding_adapter_failed",
        ),
    ),
)
def test_embedding_contract_failures_never_publish(
    tmp_path: Path,
    mode: str,
    provider_factory,
    expected_code: str,
) -> None:
    _, SessionMaker = _sessions(tmp_path / f"{mode}.db")
    corpus = _ingest(
        SessionMaker,
        version="generation-1",
        files=(_file("src/value.py", "value"),),
    )
    identity = _identity()

    with pytest.raises(BaselineIndexBuildError) as caught:
        BaselineIndexBuilder(SessionMaker).build(
            generation_id=corpus.generation_id,
            index_version=f"index-{mode}",
            embedding=identity,
            provider=provider_factory(identity),
        )

    assert caught.value.code == expected_code
    with SessionMaker() as session:
        publication = session.get(RetrievalBaselineIndexPublication, corpus.corpus_id)
        build = session.get(RetrievalBaselineIndexBuild, caught.value.index_id)
        state = session.get(RetrievalIndexState, corpus.generation_id)
    assert publication is None
    assert build.status == BaselineIndexBuildStatus.FAILED.value
    assert build.failure_code == expected_code
    assert state.status == IndexStateStatus.INCOMPLETE.value


def test_failed_publication_retains_previous_compatible_index(tmp_path: Path) -> None:
    _, SessionMaker = _sessions(tmp_path / "rollback.db")
    corpus = _ingest(
        SessionMaker,
        version="generation-1",
        files=(_file("src/value.py", "value"),),
    )
    identity = _identity()
    first = _build(SessionMaker, corpus.generation_id, identity=identity)

    def fail_after_publication(session, index_id):
        BaselineIndexLifecycle.publish(session, index_id)
        raise RuntimeError("simulated index publication failure")

    with pytest.raises(BaselineIndexBuildError) as caught:
        _build(
            SessionMaker,
            corpus.generation_id,
            version="index-2",
            identity=identity,
            publish_index=fail_after_publication,
        )
    assert caught.value.code == "index_publication_failed"

    with SessionMaker() as session:
        publication = session.get(RetrievalBaselineIndexPublication, corpus.corpus_id)
        prior = session.get(RetrievalBaselineIndexBuild, first.index_id)
        failed = session.get(RetrievalBaselineIndexBuild, caught.value.index_id)
        readiness = assess_baseline_index(
            session,
            scope_key="group:indexing",
            embedding=identity,
        )
    assert publication.index_id == first.index_id
    assert prior.status == BaselineIndexBuildStatus.COMPATIBLE.value
    assert failed.status == BaselineIndexBuildStatus.FAILED.value
    assert readiness.ready is True
    assert readiness.index_id == first.index_id


def test_sqlite_reader_observes_old_or_new_complete_publication(
    tmp_path: Path,
) -> None:
    _, SessionMaker = _sessions(tmp_path / "concurrent.db")
    corpus = _ingest(
        SessionMaker,
        version="generation-1",
        files=(_file("src/value.py", "value"),),
    )
    identity = _identity()
    first = _build(SessionMaker, corpus.generation_id, identity=identity)
    publication_uncommitted = Event()
    allow_commit = Event()

    def pause_after_publication(session, index_id):
        BaselineIndexLifecycle.publish(session, index_id)
        publication_uncommitted.set()
        if not allow_commit.wait(timeout=5):
            raise RuntimeError("timed out waiting for concurrent reader")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _build,
            SessionMaker,
            corpus.generation_id,
            version="index-2",
            identity=identity,
            publish_index=pause_after_publication,
        )
        assert publication_uncommitted.wait(timeout=5)
        with SessionMaker() as reader:
            during = reader.get(
                RetrievalBaselineIndexPublication,
                corpus.corpus_id,
            )
            during_readiness = assess_baseline_index(
                reader,
                scope_key="group:indexing",
                embedding=identity,
            )
        assert during.index_id == first.index_id
        assert during_readiness.ready is True
        allow_commit.set()
        second = future.result(timeout=5)

    with SessionMaker() as reader:
        after = reader.get(RetrievalBaselineIndexPublication, corpus.corpus_id)
        first_build = reader.get(RetrievalBaselineIndexBuild, first.index_id)
        after_readiness = assess_baseline_index(
            reader,
            scope_key="group:indexing",
            embedding=identity,
        )
    assert after.index_id == second.index_id
    assert first_build.status == BaselineIndexBuildStatus.STALE.value
    assert after_readiness.ready is True
    assert after_readiness.index_id == second.index_id


def test_index_schema_has_no_query_or_legacy_embedding_columns(tmp_path: Path) -> None:
    engine, _ = _sessions(tmp_path / "schema.db")
    table_names = {
        "retrieval_baseline_index_build",
        "retrieval_baseline_index_document",
        "retrieval_baseline_index_term",
        "retrieval_baseline_index_vector",
        "retrieval_baseline_index_publication",
    }
    with engine.connect() as connection:
        assert table_names <= set(engine.dialect.get_table_names(connection))
        for table_name in table_names:
            columns = {
                column["name"]
                for column in engine.dialect.get_columns(connection, table_name)
            }
            assert "retrieval_query" not in columns
            assert "query_text" not in columns
            assert "chunk_id" not in columns
    assert BASELINE_VECTOR_FORMAT == "float32-le.v1"


def test_active_ingestion_provenance_remains_query_free(tmp_path: Path) -> None:
    _, SessionMaker = _sessions(tmp_path / "provenance.db")
    corpus = _ingest(
        SessionMaker,
        version="generation-1",
        files=(_file("src/value.py", "value"),),
    )
    _build(SessionMaker, corpus.generation_id)

    with SessionMaker() as session:
        ingestion = session.get(
            RetrievalCorpusIngestion,
            corpus.generation_id,
        )
    assert ingestion.status == CorpusIngestionStatus.ACTIVE.value
    assert "retrieval_query" not in ingestion.canonical_manifest_json
