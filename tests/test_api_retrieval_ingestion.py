from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from compair_core import db as core_db
from compair_core.compair.retrieval.corpus import (
    CorpusFileInput,
    CorpusFileSkipReason,
    CorpusFileState,
    CorpusLifecycle,
    IndexRequirements,
    RetrievalCorpus,
    RetrievalCorpusFile,
    RetrievalCorpusGeneration,
    RetrievalCorpusIngestion,
    RetrievalIndexState,
    ensure_retrieval_corpus_schema,
)
from compair_core.compair.retrieval.ingestion import (
    CorpusGenerationFreshness,
    CorpusIngestionService,
    CorpusRepositoryInput,
    CorpusSnapshotInput,
    canonical_snapshot_manifest_json,
    corpus_generation_freshness,
    validate_snapshot_input,
)


def _sessions(tmp_path: Path):
    engine = core_db.create_engine(f"sqlite:///{tmp_path / 'ingestion.db'}")
    ensure_retrieval_corpus_schema(engine)
    return engine, core_db.sessionmaker(engine, expire_on_commit=False)


def _repository(name: str, file_count: int, *, revision: str = "repo-revision-1"):
    return CorpusRepositoryInput(
        repository_id=f"repo-{name}",
        repository_name=name,
        expected_file_count=file_count,
        repository_revision=revision,
        document_id=f"document-{name}",
        document_revision=f"document-revision-{revision}",
    )


def _changed(*, revision: str = "changed-revision-1"):
    return CorpusRepositoryInput(
        repository_id="repo-changed",
        repository_name="changed",
        expected_file_count=0,
        repository_revision=revision,
        document_id="document-changed",
        document_revision=f"document-revision-{revision}",
    )


def _text(repository: str, path: str, content: str) -> CorpusFileInput:
    return CorpusFileInput.supported_text(
        repository_id=f"repo-{repository}",
        repository_name=repository,
        relative_path=path,
        content=content,
    )


def _snapshot(
    *,
    version: str,
    files: tuple[CorpusFileInput, ...],
    repositories: tuple[CorpusRepositoryInput, ...],
    scope_key: str = "group:changed",
    changed_revision: str = "changed-revision-1",
) -> CorpusSnapshotInput:
    return CorpusSnapshotInput.create(
        scope_key=scope_key,
        generation_version=version,
        changed_repository=_changed(revision=changed_revision),
        sibling_repositories=repositories,
        files=files,
        producer_id="trusted-test-producer",
        producer_version="1.0",
        snapshot_id=f"snapshot-{version}",
        source_revision=changed_revision,
        source_manifest_hash=hashlib.sha256(
            f"source-{version}".encode()
        ).hexdigest(),
    )


@pytest.mark.parametrize(
    "path",
    (
        "/absolute/file.py",
        "../outside.py",
        "src/../../outside.py",
        "C:\\outside.py",
        "src\\ambiguous.py",
        "src//ambiguous.py",
    ),
)
def test_snapshot_rejects_absolute_traversal_and_ambiguous_paths(path: str) -> None:
    snapshot = _snapshot(
        version="bad-path",
        files=(_text("peer", path, "value = 1\n"),),
        repositories=(_repository("peer", 1),),
    )

    with pytest.raises(ValueError):
        validate_snapshot_input(snapshot)


def test_snapshot_rejects_duplicate_normalized_paths() -> None:
    snapshot = _snapshot(
        version="duplicate",
        files=(
            _text("peer", "src/value.py", "value = 1\n"),
            _text("peer", "src/value.py", "value = 2\n"),
        ),
        repositories=(_repository("peer", 2),),
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_snapshot_input(snapshot)


def test_symlink_derived_content_is_rejected_and_unfollowed_skip_is_recordable(
    tmp_path: Path,
) -> None:
    linked = replace(
        _text("peer", "src/link.py", "outside content\n"),
        derived_from_symlink=True,
    )
    rejected = _snapshot(
        version="derived-link",
        files=(linked,),
        repositories=(_repository("peer", 1),),
    )
    with pytest.raises(ValueError, match="symlink"):
        validate_snapshot_input(rejected)

    skipped = CorpusFileInput(
        repository_id="repo-peer",
        repository_name="peer",
        relative_path="src/link.py",
        file_state=CorpusFileState.SYMLINK_REJECTED,
        content_hash=hashlib.sha256(b"src/target.py").hexdigest(),
        byte_size=len(b"src/target.py"),
        skip_reason=CorpusFileSkipReason.SYMLINK,
    )
    snapshot = _snapshot(
        version="unfollowed-link",
        files=(skipped,),
        repositories=(_repository("peer", 1),),
    )
    _, SessionMaker = _sessions(tmp_path)

    result = CorpusIngestionService(SessionMaker).ingest(snapshot)

    with SessionMaker() as session:
        row = session.query(RetrievalCorpusFile).filter_by(
            generation_id=result.generation_id
        ).one()
    assert row.file_state == CorpusFileState.SYMLINK_REJECTED.value
    assert row.content is None


def test_deterministic_manifest_activation_identity_and_no_query_leakage(
    tmp_path: Path,
) -> None:
    alpha = _repository("alpha", 1, revision="alpha-revision")
    zeta = _repository("zeta", 1, revision="zeta-revision")
    files = (
        _text("zeta", "src/z.py", "zeta\n"),
        _text("alpha", "src/a.py", "alpha\n"),
    )
    first = _snapshot(
        version="generation-1",
        files=files,
        repositories=(zeta, alpha),
    )
    reordered = _snapshot(
        version="generation-1",
        files=tuple(reversed(files)),
        repositories=(alpha, zeta),
    )
    assert first.declared_manifest_hash == reordered.declared_manifest_hash
    assert canonical_snapshot_manifest_json(first) == canonical_snapshot_manifest_json(
        reordered
    )

    engine, SessionMaker = _sessions(tmp_path)
    result = CorpusIngestionService(SessionMaker).ingest(first)
    retrieval_query = "diff --git a/private.py b/private.py\n+private sentinel\n"

    with SessionMaker() as session:
        corpus = session.get(RetrievalCorpus, result.corpus_id)
        ingestion = session.get(RetrievalCorpusIngestion, result.generation_id)
        index = session.get(RetrievalIndexState, result.generation_id)
        rows = CorpusLifecycle.ordered_files(session, result.generation_id)
        readiness = CorpusLifecycle.assess_active_corpus(
            session,
            scope_key="group:changed",
            requirements=IndexRequirements(
                tokenizer_version="not-built",
                embedding_provider="not-built",
                embedding_model="not-built",
                embedding_revision="not-built",
                embedding_dimension=1,
                embedding_fingerprint="a" * 64,
                engine_config_fingerprint="b" * 64,
            ),
        )

    assert result.status is CorpusGenerationFreshness.ACTIVE
    assert corpus.changed_repository_id == "repo-changed"
    assert corpus.active_generation_id == result.generation_id
    assert ingestion.status == "active"
    assert ingestion.canonical_manifest_hash == first.declared_manifest_hash
    assert [(row.repository_name, row.relative_path) for row in rows] == [
        ("alpha", "src/a.py"),
        ("zeta", "src/z.py"),
    ]
    assert index.status == "incomplete"
    assert readiness.ready is False
    assert readiness.code == "index_incomplete"
    assert retrieval_query not in ingestion.canonical_manifest_json
    assert retrieval_query not in repr(ingestion)

    with engine.connect() as connection:
        for table_name in (
            "retrieval_corpus_ingestion",
            "retrieval_corpus_generation",
            "retrieval_corpus_file",
        ):
            columns = {
                column["name"]
                for column in engine.dialect.get_columns(connection, table_name)
            }
            assert "retrieval_query" not in columns
            assert "query_text" not in columns


def test_delete_and_rename_create_new_immutable_generation(tmp_path: Path) -> None:
    _, SessionMaker = _sessions(tmp_path)
    service = CorpusIngestionService(SessionMaker)
    first = service.ingest(
        _snapshot(
            version="generation-1",
            files=(
                _text("peer", "src/deleted.py", "delete\n"),
                _text("peer", "src/original.py", "rename\n"),
            ),
            repositories=(_repository("peer", 2, revision="peer-revision-1"),),
        )
    )
    second = service.ingest(
        _snapshot(
            version="generation-2",
            files=(_text("peer", "src/renamed.py", "rename\n"),),
            repositories=(_repository("peer", 1, revision="peer-revision-2"),),
            changed_revision="changed-revision-2",
        )
    )

    with SessionMaker() as session:
        prior_paths = {
            row.relative_path
            for row in CorpusLifecycle.ordered_files(session, first.generation_id)
        }
        active_paths = {
            row.relative_path
            for row in CorpusLifecycle.ordered_files(session, second.generation_id)
        }
        first_freshness = corpus_generation_freshness(session, first.generation_id)
        second_freshness = corpus_generation_freshness(session, second.generation_id)

    assert prior_paths == {"src/deleted.py", "src/original.py"}
    assert active_paths == {"src/renamed.py"}
    assert first_freshness is CorpusGenerationFreshness.STALE
    assert second_freshness is CorpusGenerationFreshness.ACTIVE


def test_activation_failure_rolls_back_and_retains_prior_active_generation(
    tmp_path: Path,
) -> None:
    _, SessionMaker = _sessions(tmp_path)
    first = CorpusIngestionService(SessionMaker).ingest(
        _snapshot(
            version="generation-1",
            files=(_text("peer", "src/one.py", "one\n"),),
            repositories=(_repository("peer", 1),),
        )
    )

    def fail_after_activation(session, generation_id):
        CorpusLifecycle.activate_generation(session, generation_id)
        raise RuntimeError("simulated activation publication failure")

    failing_service = CorpusIngestionService(
        SessionMaker,
        activate_generation=fail_after_activation,
    )
    with pytest.raises(RuntimeError, match="publication failure"):
        failing_service.ingest(
            _snapshot(
                version="generation-2",
                files=(_text("peer", "src/two.py", "two\n"),),
                repositories=(_repository("peer", 1, revision="repo-revision-2"),),
                changed_revision="changed-revision-2",
            )
        )

    with SessionMaker() as session:
        corpus = session.get(RetrievalCorpus, first.corpus_id)
        pending = session.query(RetrievalCorpusGeneration).filter_by(
            corpus_id=first.corpus_id,
            generation_version="generation-2",
        ).one()
        prior_freshness = corpus_generation_freshness(session, first.generation_id)
        pending_freshness = corpus_generation_freshness(
            session,
            pending.generation_id,
        )

    assert corpus.active_generation_id == first.generation_id
    assert prior_freshness is CorpusGenerationFreshness.ACTIVE
    assert pending_freshness is CorpusGenerationFreshness.COMPLETE


def test_declared_manifest_hash_tampering_fails_before_staging(tmp_path: Path) -> None:
    _, SessionMaker = _sessions(tmp_path)
    snapshot = _snapshot(
        version="tampered",
        files=(_text("peer", "src/value.py", "value\n"),),
        repositories=(_repository("peer", 1),),
    )
    tampered = replace(snapshot, declared_manifest_hash="0" * 64)

    with pytest.raises(ValueError, match="manifest hash"):
        CorpusIngestionService(SessionMaker).ingest(tampered)

    with SessionMaker() as session:
        assert session.query(RetrievalCorpusGeneration).count() == 0


def test_file_content_hash_tampering_fails_before_staging(tmp_path: Path) -> None:
    _, SessionMaker = _sessions(tmp_path)
    tampered_file = replace(
        _text("peer", "src/value.py", "value\n"),
        content_hash="0" * 64,
    )
    snapshot = _snapshot(
        version="tampered-file",
        files=(tampered_file,),
        repositories=(_repository("peer", 1),),
    )

    with pytest.raises(ValueError, match="content hash"):
        CorpusIngestionService(SessionMaker).ingest(snapshot)

    with SessionMaker() as session:
        assert session.query(RetrievalCorpusGeneration).count() == 0


def test_freshness_distinguishes_incomplete_and_failed_generations(
    tmp_path: Path,
) -> None:
    _, SessionMaker = _sessions(tmp_path)
    with SessionMaker.begin() as session:
        corpus = CorpusLifecycle.get_or_create_corpus(
            session,
            scope_key="group:freshness",
            changed_repository_id="repo-changed",
        )
        generation = CorpusLifecycle.stage_generation(
            session,
            corpus=corpus,
            generation_version="incomplete-generation",
            files=(_text("peer", "src/value.py", "value\n"),),
            expected_repository_count=1,
            expected_file_count=2,
        )
        generation_id = generation.generation_id

    with SessionMaker() as session:
        assert (
            corpus_generation_freshness(session, generation_id)
            is CorpusGenerationFreshness.INCOMPLETE
        )

    with SessionMaker.begin() as session:
        validation = CorpusLifecycle.validate_generation(session, generation_id)
        assert validation.complete is False

    with SessionMaker() as session:
        assert (
            corpus_generation_freshness(session, generation_id)
            is CorpusGenerationFreshness.FAILED
        )
