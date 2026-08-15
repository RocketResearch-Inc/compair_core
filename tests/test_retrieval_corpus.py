from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from compair_core import db as core_db
from compair_core.compair.retrieval.corpus import (
    CORPUS_SNAPSHOT_SCHEMA_VERSION,
    RETRIEVAL_CORPUS_TABLES,
    TOKENIZER_VERSION_PLACEHOLDER,
    CorpusFileInput,
    CorpusFileSkipReason,
    CorpusFileState,
    CorpusGenerationStatus,
    CorpusLifecycle,
    IndexRequirements,
    IndexStateStatus,
    RetrievalCorpus,
    RetrievalCorpusGeneration,
    RetrievalCorpusIngestion,
    RetrievalIndexState,
    compile_retrieval_corpus_ddl,
    ensure_retrieval_corpus_schema,
)


def _sessions(tmp_path: Path):
    engine = core_db.create_engine(f"sqlite:///{tmp_path / 'corpus.db'}")
    ensure_retrieval_corpus_schema(engine)
    return engine, core_db.sessionmaker(engine, expire_on_commit=False)


def _text(repository: str, path: str, content: str) -> CorpusFileInput:
    return CorpusFileInput.supported_text(
        repository_id=f"repo-id-{repository}",
        repository_name=repository,
        relative_path=path,
        content=content,
        document_id=f"doc-{repository}",
        source_snapshot_id="snapshot-1",
    )


def _stage_validate_activate(
    SessionMaker,
    *,
    scope_key: str,
    version: str,
    files: list[CorpusFileInput],
    changed_repository_id: str = "repo-id-changed",
) -> tuple[str, str]:
    with SessionMaker.begin() as session:
        corpus = CorpusLifecycle.get_or_create_corpus(
            session,
            scope_key=scope_key,
            changed_repository_id=changed_repository_id,
            source_document_id="source-document",
        )
        generation = CorpusLifecycle.stage_generation(
            session,
            corpus=corpus,
            generation_version=version,
            files=files,
            expected_repository_count=len({row.repository_id for row in files}),
            expected_file_count=len(files),
            source_revision="source-revision-1",
        )
        generation_id = generation.generation_id
        corpus_id = corpus.corpus_id
        repository_counts = {
            row.repository_id: sum(
                candidate.repository_id == row.repository_id for candidate in files
            )
            for row in files
        }
        manifest_json = json.dumps(
            {
                "sibling_repositories": [
                    {
                        "repository_id": repository_id,
                        "expected_file_count": file_count,
                    }
                    for repository_id, file_count in sorted(
                        repository_counts.items()
                    )
                ]
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        session.add(
            RetrievalCorpusIngestion(
                generation_id=generation_id,
                snapshot_schema_version=CORPUS_SNAPSHOT_SCHEMA_VERSION,
                ingestion_source="trusted_snapshot_v1",
                producer_id="phase-2b1-test",
                canonical_manifest_hash=hashlib.sha256(
                    manifest_json.encode("utf-8")
                ).hexdigest(),
                canonical_manifest_json=manifest_json,
                repository_count=len(repository_counts),
                file_count=len(files),
            )
        )
    with SessionMaker.begin() as session:
        assert CorpusLifecycle.validate_generation(session, generation_id).complete
    with SessionMaker.begin() as session:
        CorpusLifecycle.activate_generation(session, generation_id)
    return corpus_id, generation_id


def _requirements(*, dimension: int = 3) -> IndexRequirements:
    return IndexRequirements(
        tokenizer_version="tokenizer-v1",
        embedding_provider="fixture-provider",
        embedding_model="fixture-model",
        embedding_revision="fixture-revision",
        embedding_dimension=dimension,
        embedding_fingerprint="a" * 64,
        engine_config_fingerprint="b" * 64,
    )


def _record_compatible_index(SessionMaker, generation_id: str, *, dimension: int = 3):
    requirements = _requirements(dimension=dimension)
    with SessionMaker.begin() as session:
        CorpusLifecycle.record_index_state(
            session,
            generation_id,
            status=IndexStateStatus.COMPATIBLE,
            tokenizer_version=requirements.tokenizer_version,
            embedding_provider=requirements.embedding_provider,
            embedding_model=requirements.embedding_model,
            embedding_revision=requirements.embedding_revision,
            embedding_dimension=requirements.embedding_dimension,
            embedding_fingerprint=requirements.embedding_fingerprint,
            engine_config_fingerprint=requirements.engine_config_fingerprint,
            indexed_file_count=1,
        )
    return requirements


def test_activation_is_atomic_and_failed_build_retains_prior_generation(
    tmp_path: Path,
) -> None:
    _, SessionMaker = _sessions(tmp_path)
    corpus_id, first_id = _stage_validate_activate(
        SessionMaker,
        scope_key="group-a:changed",
        version="generation-1",
        files=[_text("peer", "src/one.py", "value = 1\n")],
    )

    with SessionMaker.begin() as session:
        corpus = session.get(RetrievalCorpus, corpus_id)
        failed = CorpusLifecycle.stage_generation(
            session,
            corpus=corpus,
            generation_version="generation-failed",
            files=[_text("peer", "src/two.py", "value = 2\n")],
            expected_repository_count=1,
            expected_file_count=2,
        )
        failed_id = failed.generation_id
    with SessionMaker.begin() as session:
        validation = CorpusLifecycle.validate_generation(session, failed_id)
        assert validation.complete is False
        assert validation.error_code == "file_count_mismatch"

    with SessionMaker() as session:
        corpus = session.get(RetrievalCorpus, corpus_id)
        failed = session.get(RetrievalCorpusGeneration, failed_id)
        assert corpus.active_generation_id == first_id
        assert failed.status == CorpusGenerationStatus.FAILED.value

    with SessionMaker.begin() as session:
        corpus = session.get(RetrievalCorpus, corpus_id)
        pending = CorpusLifecycle.stage_generation(
            session,
            corpus=corpus,
            generation_version="generation-pending",
            files=[_text("peer", "src/three.py", "value = 3\n")],
            expected_repository_count=1,
            expected_file_count=1,
        )
        pending_id = pending.generation_id
    with SessionMaker.begin() as session:
        assert CorpusLifecycle.validate_generation(session, pending_id).complete

    with (
        pytest.raises(RuntimeError, match="simulated publication failure"),
        SessionMaker.begin() as session,
    ):
        CorpusLifecycle.activate_generation(session, pending_id)
        raise RuntimeError("simulated publication failure")

    with SessionMaker() as session:
        corpus = session.get(RetrievalCorpus, corpus_id)
        first = session.get(RetrievalCorpusGeneration, first_id)
        pending = session.get(RetrievalCorpusGeneration, pending_id)
        assert corpus.active_generation_id == first_id
        assert first.status == CorpusGenerationStatus.ACTIVE.value
        assert pending.status == CorpusGenerationStatus.VALIDATED.value


def test_full_snapshot_update_delete_and_rename_are_deterministic(tmp_path: Path) -> None:
    _, SessionMaker = _sessions(tmp_path)
    corpus_id, first_id = _stage_validate_activate(
        SessionMaker,
        scope_key="group-b:changed",
        version="generation-1",
        files=[
            _text("zeta", "src/deleted.py", "remove me\n"),
            _text("alpha", "src/original.py", "old\n"),
            _text("alpha", "src/keep.py", "before\n"),
        ],
    )
    _, second_id = _stage_validate_activate(
        SessionMaker,
        scope_key="group-b:changed",
        version="generation-2",
        files=[
            _text("alpha", "src/renamed.py", "old\n"),
            _text("alpha", "src/keep.py", "after\n"),
        ],
    )

    with SessionMaker() as session:
        corpus = session.get(RetrievalCorpus, corpus_id)
        active = CorpusLifecycle.ordered_files(session, corpus.active_generation_id)
        prior = CorpusLifecycle.ordered_files(session, first_id)

    assert corpus.active_generation_id == second_id
    assert [(row.repository_name, row.relative_path) for row in active] == [
        ("alpha", "src/keep.py"),
        ("alpha", "src/renamed.py"),
    ]
    assert active[0].content == "after\n"
    assert {row.relative_path for row in prior} == {
        "src/deleted.py",
        "src/keep.py",
        "src/original.py",
    }


def test_corpus_records_unsupported_files_and_changed_repository_identity(
    tmp_path: Path,
) -> None:
    _, SessionMaker = _sessions(tmp_path)
    unsupported_hash = hashlib.sha256(b"\xff\xfe").hexdigest()
    files = [
        _text("peer", "src/peer.py", "peer\n"),
        CorpusFileInput(
            repository_id="repo-id-peer",
            repository_name="peer",
            relative_path="assets/blob.bin",
            file_state=CorpusFileState.UNSUPPORTED_UTF8,
            content_hash=unsupported_hash,
            byte_size=2,
            skip_reason=CorpusFileSkipReason.NON_UTF8,
        ),
    ]
    corpus_id, generation_id = _stage_validate_activate(
        SessionMaker,
        scope_key="group-c:changed",
        version="generation-1",
        files=files,
    )

    with SessionMaker() as session:
        corpus = session.get(RetrievalCorpus, corpus_id)
        rows = CorpusLifecycle.ordered_files(session, generation_id)

    assert corpus.changed_repository_id == "repo-id-changed"
    assert all(row.repository_id != corpus.changed_repository_id for row in rows)
    assert rows[0].relative_path == "assets/blob.bin"
    assert rows[0].content is None
    assert rows[0].file_state == CorpusFileState.UNSUPPORTED_UTF8.value


def test_incomplete_stale_and_dimension_mismatched_indexes_fail_closed(
    tmp_path: Path,
) -> None:
    _, SessionMaker = _sessions(tmp_path)
    _, generation_id = _stage_validate_activate(
        SessionMaker,
        scope_key="group-d:changed",
        version="generation-1",
        files=[_text("peer", "src/file.py", "content\n")],
    )

    with SessionMaker() as session:
        state = session.get(RetrievalIndexState, generation_id)
        readiness = CorpusLifecycle.assess_active_corpus(
            session,
            scope_key="group-d:changed",
            requirements=_requirements(),
        )
    assert state.tokenizer_version == TOKENIZER_VERSION_PLACEHOLDER
    assert readiness.ready is False
    assert readiness.code == "index_incomplete"

    matching = _record_compatible_index(SessionMaker, generation_id, dimension=3)
    with SessionMaker() as session:
        ready = CorpusLifecycle.assess_active_corpus(
            session,
            scope_key="group-d:changed",
            requirements=matching,
        )
    assert ready.ready is True
    assert ready.code == "ready"

    with SessionMaker() as session:
        mismatch = CorpusLifecycle.assess_active_corpus(
            session,
            scope_key="group-d:changed",
            requirements=_requirements(dimension=4),
        )
    assert mismatch.ready is False
    assert mismatch.code == "index_dimension_mismatch"

    with SessionMaker.begin() as session:
        CorpusLifecycle.record_index_state(
            session,
            generation_id,
            status=IndexStateStatus.STALE,
            tokenizer_version=matching.tokenizer_version,
            embedding_provider=matching.embedding_provider,
            embedding_model=matching.embedding_model,
            embedding_revision=matching.embedding_revision,
            embedding_dimension=matching.embedding_dimension,
            embedding_fingerprint=matching.embedding_fingerprint,
            engine_config_fingerprint=matching.engine_config_fingerprint,
            indexed_file_count=1,
        )
    with SessionMaker() as session:
        stale = CorpusLifecycle.assess_active_corpus(
            session,
            scope_key="group-d:changed",
            requirements=matching,
        )
    assert stale.ready is False
    assert stale.code == "index_stale"


def test_schema_is_additive_and_portable_between_sqlite_and_postgres(
    tmp_path: Path,
) -> None:
    engine, _ = _sessions(tmp_path)
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE legacy_sentinel (value INTEGER)")
        connection.exec_driver_sql("INSERT INTO legacy_sentinel (value) VALUES (7)")
    ensure_retrieval_corpus_schema(engine)
    expected = {
        "retrieval_baseline_index_build",
        "retrieval_baseline_index_document",
        "retrieval_baseline_index_publication",
        "retrieval_baseline_index_term",
        "retrieval_baseline_index_vector",
        "retrieval_corpus",
        "retrieval_corpus_generation",
        "retrieval_corpus_ingestion",
        "retrieval_corpus_file",
        "retrieval_index_state",
    }
    with engine.connect() as connection:
        assert expected <= set(engine.dialect.get_table_names(connection))
        assert connection.exec_driver_sql(
            "SELECT value FROM legacy_sentinel"
        ).scalar_one() == 7
        for table_name in expected:
            column_names = {
                column["name"]
                for column in engine.dialect.get_columns(connection, table_name)
            }
            assert "retrieval_query" not in column_names
            assert "query_text" not in column_names

    sqlite_ddl = compile_retrieval_corpus_ddl("sqlite")
    postgres_ddl = compile_retrieval_corpus_ddl("postgresql")
    for table, sqlite_statement, postgres_statement in zip(
        RETRIEVAL_CORPUS_TABLES,
        sqlite_ddl,
        postgres_ddl,
    ):
        assert f"CREATE TABLE {table.name}" in sqlite_statement
        assert f"CREATE TABLE {table.name}" in postgres_statement


def test_path_normalization_rejects_escape_and_duplicates(tmp_path: Path) -> None:
    _, SessionMaker = _sessions(tmp_path)
    with SessionMaker.begin() as session:
        corpus = CorpusLifecycle.get_or_create_corpus(
            session,
            scope_key="group-e:changed",
            changed_repository_id="changed",
        )
        with pytest.raises(ValueError, match="unsafe"):
            CorpusLifecycle.stage_generation(
                session,
                corpus=corpus,
                generation_version="escape",
                files=[_text("peer", "../outside.py", "bad\n")],
                expected_repository_count=1,
                expected_file_count=1,
            )
        with pytest.raises(ValueError, match="duplicate"):
            CorpusLifecycle.stage_generation(
                session,
                corpus=corpus,
                generation_version="duplicate",
                files=[
                    _text("peer", "src/same.py", "one\n"),
                    _text("peer", "src/same.py", "two\n"),
                ],
                expected_repository_count=1,
                expected_file_count=2,
            )
