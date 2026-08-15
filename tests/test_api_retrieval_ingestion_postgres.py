"""Runnable PostgreSQL integration for the corpus publication transaction.

CI can execute this test by setting COMPAIR_TEST_POSTGRES_URL to a dedicated
PostgreSQL database. It is skipped, not emulated, when that variable is absent.
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from compair_core import db as core_db
from compair_core.compair.retrieval.corpus import (
    CorpusFileInput,
    CorpusLifecycle,
    RetrievalCorpus,
    ensure_retrieval_corpus_schema,
)
from compair_core.compair.retrieval.ingestion import (
    CorpusGenerationFreshness,
    CorpusIngestionService,
    CorpusRepositoryInput,
    CorpusSnapshotInput,
    corpus_generation_freshness,
)

POSTGRES_URL = os.getenv("COMPAIR_TEST_POSTGRES_URL")


@pytest.mark.skipif(
    not POSTGRES_URL,
    reason="set COMPAIR_TEST_POSTGRES_URL to run the real PostgreSQL integration",
)
def test_postgres_activation_rollback_retains_prior_generation() -> None:
    assert POSTGRES_URL is not None
    engine = core_db.create_engine(POSTGRES_URL, pool_pre_ping=True)
    if engine.dialect.name != "postgresql":
        pytest.fail("COMPAIR_TEST_POSTGRES_URL must select PostgreSQL")
    ensure_retrieval_corpus_schema(engine)
    SessionMaker = core_db.sessionmaker(engine, expire_on_commit=False)
    suffix = uuid4().hex
    scope_key = f"postgres-ingestion-{suffix}"
    repository = CorpusRepositoryInput(
        repository_id=f"peer-{suffix}",
        repository_name=f"peer-{suffix}",
        expected_file_count=1,
        repository_revision="revision-1",
    )
    changed = CorpusRepositoryInput(
        repository_id=f"changed-{suffix}",
        repository_name=f"changed-{suffix}",
        expected_file_count=0,
        repository_revision="changed-revision",
    )

    def snapshot(version: str, content: str) -> CorpusSnapshotInput:
        return CorpusSnapshotInput.create(
            scope_key=scope_key,
            generation_version=version,
            changed_repository=changed,
            sibling_repositories=(repository,),
            files=(
                CorpusFileInput.supported_text(
                    repository_id=repository.repository_id,
                    repository_name=repository.repository_name,
                    relative_path="src/value.py",
                    content=content,
                ),
            ),
            producer_id="postgres-ci-test",
        )

    try:
        first = CorpusIngestionService(SessionMaker).ingest(
            snapshot("generation-1", "value = 1\n")
        )

        def fail_after_activation(session, generation_id):
            CorpusLifecycle.activate_generation(session, generation_id)
            raise RuntimeError("postgres publication rollback")

        with pytest.raises(RuntimeError, match="publication rollback"):
            CorpusIngestionService(
                SessionMaker,
                activate_generation=fail_after_activation,
            ).ingest(snapshot("generation-2", "value = 2\n"))

        with SessionMaker() as session:
            corpus = session.query(RetrievalCorpus).filter_by(
                scope_key=scope_key
            ).one()
            assert corpus.active_generation_id == first.generation_id
            assert (
                corpus_generation_freshness(session, first.generation_id)
                is CorpusGenerationFreshness.ACTIVE
            )
    finally:
        with SessionMaker.begin() as session:
            corpus = session.query(RetrievalCorpus).filter_by(
                scope_key=scope_key
            ).one_or_none()
            if corpus is not None:
                session.delete(corpus)
        engine.dispose()
