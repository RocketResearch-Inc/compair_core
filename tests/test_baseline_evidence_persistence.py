from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import Engine, text

from compair_core import db as core_db
from compair_core.compair import models
from compair_core.compair.retrieval.corpus import (
    CorpusFileInput,
    RetrievalBaselineIndexPublication,
    ensure_retrieval_corpus_schema,
)
from compair_core.compair.retrieval.evidence_persistence import (
    BaselineEvidencePersistenceCommand,
    BaselineEvidencePersistenceError,
    BaselineEvidencePersistenceService,
    PersistenceWriteStage,
)
from compair_core.compair.retrieval.indexing import (
    BaselineEmbeddingIdentity,
    BaselineIndexBuilder,
)
from compair_core.compair.retrieval.ingestion import (
    CorpusIngestionService,
    CorpusRepositoryInput,
    CorpusSnapshotInput,
)
from compair_core.compair.retrieval.persistent import PersistentBaselineV1Retriever
from compair_core.compair.retrieval.types import (
    RetrievalQueryOrigin,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStatus,
)
from compair_core.schema_migrations import run_schema_migrations


class FixtureEmbeddingProvider:
    provider = "fixture-baseline-adapter"
    model = "fixture/bge-small"
    revision = "fixture-revision-1"
    dimension = 2
    fingerprint = hashlib.sha256(b"fixture/bge-small@fixture-revision-1").hexdigest()

    def embed(self, texts):
        vectors = []
        for ordinal, value in enumerate(texts):
            if value.startswith("Repository file: "):
                vectors.append((1.0, float(ordinal + 1)))
            else:
                vectors.append((1.0, 0.0))
        return vectors


@dataclass(frozen=True)
class PersistenceEnvironment:
    engine: Engine
    sessions: object
    group_id: str
    source_document_id: str
    source_chunk_id: str
    peer_document_id: str
    peer_chunk_id: str
    result: RetrievalResult

    def command(self, key: str = "caller-retry-token-1"):
        return BaselineEvidencePersistenceCommand(
            group_id=self.group_id,
            source_chunk_id=self.source_chunk_id,
            source_document_id=self.source_document_id,
            idempotency_key=key,
            retrieval_result=self.result,
        )


def _seed_core_scope(engine: Engine) -> tuple[str, str, str, str, str]:
    group_id = str(uuid4())
    user_id = str(uuid4())
    source_document_id = str(uuid4())
    source_chunk_id = str(uuid4())
    peer_document_id = str(uuid4())
    peer_chunk_id = str(uuid4())
    now = datetime.now(timezone.utc)
    with engine.begin() as connection:
        connection.execute(
            text(
                'INSERT INTO "user" '
                "(user_id, username, name, datetime_registered, password_hash, "
                "password_salt, status, include_own_documents_in_feedback, "
                "default_publish, preferred_feedback_length, hide_affiliations) "
                "VALUES (:user_id, :username, :name, :now, 'hash', 'salt', "
                "'active', false, true, 'Brief', false)"
            ),
            {
                "user_id": user_id,
                "username": f"user-{user_id}",
                "name": "Persistence Test User",
                "now": now,
            },
        )
        connection.execute(
            text(
                'INSERT INTO "group" '
                "(group_id, name, datetime_created, category, description, visibility) "
                "VALUES (:group_id, 'Evidence test', :now, 'Other', '', 'private')"
            ),
            {"group_id": group_id, "now": now},
        )
        for document_id, title in (
            (source_document_id, "Changed repository"),
            (peer_document_id, "Legacy peer"),
        ):
            connection.execute(
                text(
                    "INSERT INTO document "
                    "(document_id, user_id, author_id, title, content, doc_type, "
                    "datetime_created, datetime_modified, is_published) "
                    "VALUES (:document_id, :user_id, :user_id, :title, :content, "
                    "'text', :now, :now, true)"
                ),
                {
                    "document_id": document_id,
                    "user_id": user_id,
                    "title": title,
                    "content": f"content for {title}",
                    "now": now,
                },
            )
            connection.execute(
                text(
                    "INSERT INTO document_to_group (document_id, group_id) "
                    "VALUES (:document_id, :group_id)"
                ),
                {"document_id": document_id, "group_id": group_id},
            )
        for chunk_id, document_id, content in (
            (source_chunk_id, source_document_id, "authoritative source chunk"),
            (peer_chunk_id, peer_document_id, "legacy peer chunk"),
        ):
            connection.execute(
                text(
                    "INSERT INTO chunk "
                    "(chunk_id, hash, content, document_id, note_id, chunk_type) "
                    "VALUES (:chunk_id, :hash, :content, :document_id, NULL, 'document')"
                ),
                {
                    "chunk_id": chunk_id,
                    "hash": hashlib.sha256(content.encode()).hexdigest(),
                    "content": content,
                    "document_id": document_id,
                },
            )
        connection.execute(
            text(
                "INSERT INTO reference "
                "(reference_id, source_chunk_id, reference_chunk_id, "
                "reference_document_id, reference_note_id, reference_type, "
                "baseline_selected_evidence_id) "
                "VALUES (:reference_id, :source, :peer, :peer_document, NULL, "
                "'document', NULL)"
            ),
            {
                "reference_id": "legacy-reference-" + uuid4().hex[:16],
                "source": source_chunk_id,
                "peer": peer_chunk_id,
                "peer_document": peer_document_id,
            },
        )
    return (
        group_id,
        source_document_id,
        source_chunk_id,
        peer_document_id,
        peer_chunk_id,
    )


def make_persistence_environment(engine: Engine) -> PersistenceEnvironment:
    models.Base.metadata.create_all(engine)
    ensure_retrieval_corpus_schema(engine)
    run_schema_migrations(engine)
    sessions = core_db.sessionmaker(engine, expire_on_commit=False)
    (
        group_id,
        source_document_id,
        source_chunk_id,
        peer_document_id,
        peer_chunk_id,
    ) = _seed_core_scope(engine)

    files = tuple(
        CorpusFileInput.supported_text(
            repository_id="repository-peer",
            repository_name="peer",
            relative_path=f"src/file_{ordinal}.py",
            content=f"alpha evidence file {ordinal}\nvalue = {ordinal}\n",
        )
        for ordinal in range(1, 5)
    )
    snapshot = CorpusSnapshotInput.create(
        scope_key=f"group:{group_id}",
        generation_version="generation-1",
        changed_repository=CorpusRepositoryInput(
            repository_id="repository-changed",
            repository_name="changed",
            expected_file_count=0,
            repository_revision="changed-revision-1",
            document_id=source_document_id,
            document_revision="changed-document-revision-1",
        ),
        sibling_repositories=(
            CorpusRepositoryInput(
                repository_id="repository-peer",
                repository_name="peer",
                expected_file_count=len(files),
                repository_revision="peer-revision-1",
            ),
        ),
        files=files,
        producer_id="trusted-persistence-fixture",
        producer_version="1",
        snapshot_id="trusted-snapshot-1",
    )
    corpus = CorpusIngestionService(sessions).ingest(snapshot)
    provider = FixtureEmbeddingProvider()
    identity = BaselineEmbeddingIdentity(
        provider=provider.provider,
        model=provider.model,
        revision=provider.revision,
        dimension=provider.dimension,
        fingerprint=provider.fingerprint,
    )
    BaselineIndexBuilder(sessions).build(
        generation_id=corpus.generation_id,
        index_version="index-1",
        embedding=identity,
        provider=provider,
    )
    request = RetrievalRequest(
        request_id="persistence-request-1",
        changed_repository=None,
        repository_roots=(),
        corpus_version="generation-1",
        retrieval_query="alpha persistence query",
        retrieval_query_origin=RetrievalQueryOrigin.EXPLICIT,
        corpus_complete=True,
        corpus_scope_key=f"group:{group_id}",
        changed_repository_id="repository-changed",
    )
    result = PersistentBaselineV1Retriever(sessions, provider).retrieve(request)
    assert result.status is RetrievalStatus.OK
    assert len(result.evidence) == 4
    return PersistenceEnvironment(
        engine=engine,
        sessions=sessions,
        group_id=group_id,
        source_document_id=source_document_id,
        source_chunk_id=source_chunk_id,
        peer_document_id=peer_document_id,
        peer_chunk_id=peer_chunk_id,
        result=result,
    )


def _sqlite_environment(tmp_path: Path, name: str = "evidence.db"):
    engine = core_db.create_engine(
        f"sqlite:///{tmp_path / name}",
        connect_args={"check_same_thread": False, "timeout": 15},
    )
    return make_persistence_environment(engine)


def persistence_counts(engine: Engine) -> tuple[int, int, int, int, int]:
    with engine.connect() as connection:
        return tuple(
            int(
                connection.execute(
                    text(
                        f"SELECT count(*) FROM {table}"
                        + (
                            " WHERE baseline_selected_evidence_id IS NOT NULL"
                            if table == "reference"
                            else ""
                        )
                    )
                ).scalar_one()
            )
            for table in (
                "baseline_retrieval_run",
                "baseline_evidence_artifact",
                "baseline_selected_evidence",
                "reference",
                "feedback",
            )
        )


def test_success_persists_explicit_order_and_survives_restart(tmp_path: Path) -> None:
    environment = _sqlite_environment(tmp_path)
    try:
        receipt = BaselineEvidencePersistenceService(environment.sessions).persist(
            environment.command()
        )
        assert receipt.replayed is False
        assert persistence_counts(environment.engine) == (1, 4, 4, 4, 0)
        with environment.engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT s.ordinal, s.fused_rank, a.repository_name, "
                    "a.relative_path, r.reference_type, r.reference_chunk_id, "
                    "r.baseline_selected_evidence_id, s.renderer_output "
                    "FROM baseline_selected_evidence s "
                    "JOIN baseline_evidence_artifact a ON a.artifact_id = s.artifact_id "
                    "JOIN reference r ON r.baseline_selected_evidence_id = s.selected_evidence_id "
                    "ORDER BY s.ordinal"
                )
            ).all()
            legacy = connection.execute(
                text(
                    "SELECT reference_chunk_id, reference_document_id, "
                    "baseline_selected_evidence_id FROM reference "
                    "WHERE baseline_selected_evidence_id IS NULL"
                )
            ).one()
        assert [row.ordinal for row in rows] == [1, 2, 3, 4]
        assert [row.fused_rank for row in rows] == [1, 2, 3, 4]
        assert [row.relative_path for row in rows] == [
            item.relative_path for item in environment.result.evidence
        ]
        assert all(row.reference_type == "baseline_file" for row in rows)
        assert all(row.reference_chunk_id is None for row in rows)
        assert tuple(row.baseline_selected_evidence_id for row in rows) == (
            receipt.selected_evidence_ids
        )
        assert rows[0].renderer_output.startswith("Repository file: peer/src/")
        assert legacy == (
            environment.peer_chunk_id,
            environment.peer_document_id,
            None,
        )

        database_url = str(environment.engine.url)
        environment.engine.dispose()
        restarted = core_db.create_engine(
            database_url,
            connect_args={"check_same_thread": False, "timeout": 15},
        )
        restarted_sessions = core_db.sessionmaker(restarted, expire_on_commit=False)
        replay = BaselineEvidencePersistenceService(restarted_sessions).persist(
            environment.command()
        )
        assert replay.replayed is True
        assert replay.run_id == receipt.run_id
        assert replay.selected_evidence_ids == receipt.selected_evidence_ids
        assert replay.reference_ids == receipt.reference_ids
        assert persistence_counts(restarted) == (1, 4, 4, 4, 0)
        restarted.dispose()
    finally:
        environment.engine.dispose()


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        (
            lambda result: replace(result, status=RetrievalStatus.INSUFFICIENT),
            "unsupported_result",
        ),
        (lambda result: replace(result, evidence=()), "malformed_result"),
        (
            lambda result: replace(result, index_fingerprint="0" * 64),
            "result_publication_mismatch",
        ),
        (
            lambda result: replace(result, corpus_scope_key="group:somewhere-else"),
            "result_publication_mismatch",
        ),
    ],
)
def test_invalid_results_write_nothing(
    tmp_path: Path, mutation, error_code: str
) -> None:
    environment = _sqlite_environment(tmp_path, f"invalid-{error_code}.db")
    try:
        command = replace(
            environment.command(), retrieval_result=mutation(environment.result)
        )
        with pytest.raises(BaselineEvidencePersistenceError) as caught:
            BaselineEvidencePersistenceService(environment.sessions).persist(command)
        assert caught.value.code == error_code
        assert persistence_counts(environment.engine) == (0, 0, 0, 0, 0)
    finally:
        environment.engine.dispose()


def test_authorization_and_stale_publication_fail_before_writes(tmp_path: Path) -> None:
    unauthorized = _sqlite_environment(tmp_path, "unauthorized.db")
    try:
        with unauthorized.engine.begin() as connection:
            connection.execute(
                text(
                    "DELETE FROM document_to_group "
                    "WHERE document_id = :document_id AND group_id = :group_id"
                ),
                {
                    "document_id": unauthorized.source_document_id,
                    "group_id": unauthorized.group_id,
                },
            )
        with pytest.raises(BaselineEvidencePersistenceError) as caught:
            BaselineEvidencePersistenceService(unauthorized.sessions).persist(
                unauthorized.command()
            )
        assert caught.value.code == "source_unauthorized"
        assert persistence_counts(unauthorized.engine) == (0, 0, 0, 0, 0)
    finally:
        unauthorized.engine.dispose()

    stale = _sqlite_environment(tmp_path, "stale.db")
    try:
        with stale.sessions.begin() as session:
            corpus = session.scalar(
                text("SELECT corpus_id FROM retrieval_corpus WHERE scope_key = :scope"),
                {"scope": f"group:{stale.group_id}"},
            )
            publication = session.get(RetrievalBaselineIndexPublication, corpus)
            publication.index_id = None
        with pytest.raises(BaselineEvidencePersistenceError) as caught:
            BaselineEvidencePersistenceService(stale.sessions).persist(stale.command())
        assert caught.value.code == "compatible_publication_absent"
        assert persistence_counts(stale.engine) == (0, 0, 0, 0, 0)
    finally:
        stale.engine.dispose()


def test_replay_conflict_and_reauthorization(tmp_path: Path) -> None:
    environment = _sqlite_environment(tmp_path)
    try:
        service = BaselineEvidencePersistenceService(environment.sessions)
        first = service.persist(environment.command())
        replay = service.persist(environment.command())
        assert replay.replayed is True
        assert replay.run_id == first.run_id
        assert persistence_counts(environment.engine) == (1, 4, 4, 4, 0)

        conflicting_result = replace(
            environment.result, request_id="a-different-request-intent"
        )
        with pytest.raises(BaselineEvidencePersistenceError) as caught:
            service.persist(
                replace(environment.command(), retrieval_result=conflicting_result)
            )
        assert caught.value.code == "idempotency_conflict"
        assert persistence_counts(environment.engine) == (1, 4, 4, 4, 0)

        with environment.engine.begin() as connection:
            connection.execute(
                text(
                    "DELETE FROM document_to_group "
                    "WHERE document_id = :document_id AND group_id = :group_id"
                ),
                {
                    "document_id": environment.source_document_id,
                    "group_id": environment.group_id,
                },
            )
        with pytest.raises(BaselineEvidencePersistenceError) as caught:
            service.persist(environment.command())
        assert caught.value.code == "source_unauthorized"
        assert persistence_counts(environment.engine) == (1, 4, 4, 4, 0)
    finally:
        environment.engine.dispose()


def test_concurrent_identical_retries_create_one_run(tmp_path: Path) -> None:
    environment = _sqlite_environment(tmp_path)
    try:

        def persist_once(_ordinal: int):
            return BaselineEvidencePersistenceService(environment.sessions).persist(
                environment.command()
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            receipts = tuple(executor.map(persist_once, range(4)))
        assert len({receipt.run_id for receipt in receipts}) == 1
        assert sum(not receipt.replayed for receipt in receipts) == 1
        assert persistence_counts(environment.engine) == (1, 4, 4, 4, 0)
    finally:
        environment.engine.dispose()


@pytest.mark.parametrize("stage", tuple(PersistenceWriteStage))
def test_failure_at_every_write_stage_rolls_back_fully(
    tmp_path: Path,
    stage: PersistenceWriteStage,
) -> None:
    environment = _sqlite_environment(tmp_path, f"rollback-{stage.value}.db")

    def fail_at(actual: PersistenceWriteStage) -> None:
        if actual is stage:
            raise RuntimeError(f"injected-{stage.value}")

    try:
        with pytest.raises(RuntimeError, match=f"injected-{stage.value}"):
            BaselineEvidencePersistenceService(
                environment.sessions, stage_hook=fail_at
            ).persist(environment.command())
        assert persistence_counts(environment.engine) == (0, 0, 0, 0, 0)
    finally:
        environment.engine.dispose()


def test_source_deletion_preserves_auditable_baseline_evidence(tmp_path: Path) -> None:
    environment = _sqlite_environment(tmp_path)
    try:
        receipt = BaselineEvidencePersistenceService(environment.sessions).persist(
            environment.command()
        )
        with environment.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM document WHERE document_id = :document_id"),
                {"document_id": environment.source_document_id},
            )
        assert persistence_counts(environment.engine) == (1, 4, 4, 4, 0)
        with environment.engine.connect() as connection:
            run = connection.execute(
                text(
                    "SELECT source_chunk_id, source_document_id "
                    "FROM baseline_retrieval_run WHERE run_id = :run_id"
                ),
                {"run_id": receipt.run_id},
            ).one()
            references = connection.execute(
                text(
                    "SELECT source_chunk_id FROM reference "
                    "WHERE baseline_selected_evidence_id IS NOT NULL"
                )
            ).all()
        assert run == (None, None)
        assert references == [(None,)] * 4

        with pytest.raises(BaselineEvidencePersistenceError) as caught:
            BaselineEvidencePersistenceService(environment.sessions).persist(
                environment.command()
            )
        assert caught.value.code == "source_unauthorized"
        assert persistence_counts(environment.engine) == (1, 4, 4, 4, 0)
    finally:
        environment.engine.dispose()
