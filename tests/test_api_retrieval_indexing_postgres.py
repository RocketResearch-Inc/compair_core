"""Real PostgreSQL publication/rollback integration for baseline_v1 indexes.

Set COMPAIR_TEST_POSTGRES_URL to a dedicated PostgreSQL database to run it.
The test is skipped, never emulated, when the variable is absent.
"""

from __future__ import annotations

import hashlib
import os
from uuid import uuid4

import pytest

from compair_core import db as core_db
from compair_core.compair.retrieval.corpus import (
    CorpusFileInput,
    RetrievalBaselineIndexBuild,
    RetrievalBaselineIndexPublication,
    RetrievalCorpus,
    ensure_retrieval_corpus_schema,
)
from compair_core.compair.retrieval.indexing import (
    BaselineEmbeddingIdentity,
    BaselineIndexBuilder,
    BaselineIndexBuildError,
    BaselineIndexLifecycle,
    assess_baseline_index,
)
from compair_core.compair.retrieval.ingestion import (
    CorpusIngestionService,
    CorpusRepositoryInput,
    CorpusSnapshotInput,
)

POSTGRES_URL = os.getenv("COMPAIR_TEST_POSTGRES_URL")


class PostgresFixtureProvider:
    def __init__(self, identity: BaselineEmbeddingIdentity) -> None:
        self.provider = identity.provider
        self.model = identity.model
        self.revision = identity.revision
        self.dimension = identity.dimension
        self.fingerprint = identity.fingerprint

    def embed(self, texts):
        return [[1.0, float(position + 1), -0.5] for position, _ in enumerate(texts)]


@pytest.mark.skipif(
    not POSTGRES_URL,
    reason="set COMPAIR_TEST_POSTGRES_URL to run real PostgreSQL index publication",
)
def test_postgres_index_publication_rollback_retains_prior_compatible_build() -> None:
    assert POSTGRES_URL is not None
    engine = core_db.create_engine(POSTGRES_URL, pool_pre_ping=True)
    if engine.dialect.name != "postgresql":
        pytest.fail("COMPAIR_TEST_POSTGRES_URL must select PostgreSQL")
    ensure_retrieval_corpus_schema(engine)
    SessionMaker = core_db.sessionmaker(engine, expire_on_commit=False)
    suffix = uuid4().hex
    scope_key = f"postgres-index-{suffix}"
    sibling = CorpusRepositoryInput(
        repository_id=f"peer-{suffix}",
        repository_name=f"peer-{suffix}",
        expected_file_count=1,
        repository_revision="peer-revision-1",
    )
    changed = CorpusRepositoryInput(
        repository_id=f"changed-{suffix}",
        repository_name=f"changed-{suffix}",
        expected_file_count=0,
        repository_revision="changed-revision-1",
    )
    snapshot = CorpusSnapshotInput.create(
        scope_key=scope_key,
        generation_version="generation-1",
        changed_repository=changed,
        sibling_repositories=(sibling,),
        files=(
            CorpusFileInput.supported_text(
                repository_id=sibling.repository_id,
                repository_name=sibling.repository_name,
                relative_path="src/value.py",
                content="value = 1\n",
            ),
        ),
        producer_id="postgres-index-ci",
    )
    identity = BaselineEmbeddingIdentity(
        provider="fixture-fastembed-adapter",
        model="fixture/bge-small",
        revision="fixture-revision-1",
        dimension=3,
        fingerprint=hashlib.sha256(b"postgres-fixture-embedding").hexdigest(),
    )
    provider = PostgresFixtureProvider(identity)

    try:
        corpus = CorpusIngestionService(SessionMaker).ingest(snapshot)
        first = BaselineIndexBuilder(SessionMaker).build(
            generation_id=corpus.generation_id,
            index_version="index-1",
            embedding=identity,
            provider=provider,
        )

        def fail_after_publication(session, index_id):
            BaselineIndexLifecycle.publish(session, index_id)
            raise RuntimeError("postgres index publication rollback")

        with pytest.raises(BaselineIndexBuildError) as caught:
            BaselineIndexBuilder(
                SessionMaker,
                publish_index=fail_after_publication,
            ).build(
                generation_id=corpus.generation_id,
                index_version="index-2",
                embedding=identity,
                provider=provider,
            )
        assert caught.value.code == "index_publication_failed"

        with SessionMaker() as session:
            publication = session.get(
                RetrievalBaselineIndexPublication,
                corpus.corpus_id,
            )
            prior = session.get(RetrievalBaselineIndexBuild, first.index_id)
            failed = session.get(
                RetrievalBaselineIndexBuild,
                caught.value.index_id,
            )
            readiness = assess_baseline_index(
                session,
                scope_key=scope_key,
                embedding=identity,
            )
        assert publication.index_id == first.index_id
        assert prior.status == "compatible"
        assert failed.status == "failed"
        assert readiness.ready is True
    finally:
        with SessionMaker.begin() as session:
            persisted = session.query(RetrievalCorpus).filter_by(
                scope_key=scope_key
            ).one_or_none()
            if persisted is not None:
                session.delete(persisted)
        engine.dispose()
