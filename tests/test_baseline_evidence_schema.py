from __future__ import annotations

import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from compair_core import db as core_db
from compair_core.baseline_evidence_schema import (
    BRIDGE_SCHEMA_VERSION,
    PROVENANCE_SCHEMA_VERSION,
    RENDERER_VERSION,
    baseline_evidence_artifact,
    baseline_retrieval_run,
    baseline_selected_evidence,
)
from compair_core.compair.retrieval.corpus import ensure_retrieval_corpus_schema
from compair_core.schema_migrations import (
    CORE_SCHEMA_MIGRATIONS,
    SchemaMigrationError,
    read_schema_migration_state,
    run_schema_migrations,
    schema_migration_table,
)


def _engine(path: Path):
    return core_db.create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False, "timeout": 10},
    )


def _create_legacy_database(path: Path) -> None:
    """Create the bridge-relevant pre-2B2F.1 schema and real legacy rows."""

    engine = _engine(path)
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                'CREATE TABLE "group" (group_id VARCHAR(36) PRIMARY KEY)'
            )
            connection.exec_driver_sql(
                'CREATE TABLE "user" (user_id VARCHAR(36) PRIMARY KEY)'
            )
            connection.exec_driver_sql(
                "CREATE TABLE document ("
                "document_id VARCHAR(36) PRIMARY KEY, user_id VARCHAR(36) NOT NULL, "
                "content TEXT NOT NULL)"
            )
            connection.exec_driver_sql(
                "CREATE TABLE chunk ("
                "chunk_id VARCHAR(36) PRIMARY KEY, hash VARCHAR(64) NOT NULL, "
                "content TEXT NOT NULL, document_id VARCHAR(36), note_id VARCHAR(36), "
                "chunk_type VARCHAR(16) NOT NULL, "
                "FOREIGN KEY(document_id) REFERENCES document(document_id) ON DELETE CASCADE)"
            )
            connection.exec_driver_sql(
                "CREATE TABLE reference ("
                "reference_id VARCHAR(36) PRIMARY KEY, source_chunk_id VARCHAR(36) NOT NULL, "
                "reference_chunk_id VARCHAR(36), reference_document_id VARCHAR(36), "
                "reference_note_id VARCHAR(36), reference_type VARCHAR(16) NOT NULL, "
                "FOREIGN KEY(source_chunk_id) REFERENCES chunk(chunk_id) ON DELETE CASCADE, "
                "FOREIGN KEY(reference_chunk_id) REFERENCES chunk(chunk_id) ON DELETE CASCADE)"
            )
            connection.exec_driver_sql(
                "CREATE TABLE feedback ("
                "feedback_id VARCHAR(36) PRIMARY KEY, source_chunk_id VARCHAR(36) NOT NULL, "
                "feedback TEXT NOT NULL, model TEXT NOT NULL, timestamp DATETIME, "
                "user_feedback VARCHAR(16), is_hidden BOOLEAN NOT NULL DEFAULT 0, "
                "FOREIGN KEY(source_chunk_id) REFERENCES chunk(chunk_id) ON DELETE CASCADE)"
            )
            connection.exec_driver_sql("INSERT INTO \"group\" VALUES ('group-a')")
            connection.exec_driver_sql("INSERT INTO \"user\" VALUES ('user-a')")
            connection.exec_driver_sql(
                "INSERT INTO document VALUES ('doc-source', 'user-a', 'source document')"
            )
            connection.exec_driver_sql(
                "INSERT INTO document VALUES ('doc-peer', 'user-a', 'peer document')"
            )
            connection.exec_driver_sql(
                "INSERT INTO chunk VALUES "
                "('chunk-source', 'hash-source', 'source chunk', 'doc-source', NULL, 'document')"
            )
            connection.exec_driver_sql(
                "INSERT INTO chunk VALUES "
                "('chunk-peer', 'hash-peer', 'peer chunk', 'doc-peer', NULL, 'document')"
            )
            connection.exec_driver_sql(
                "INSERT INTO reference VALUES "
                "('legacy-reference', 'chunk-source', 'chunk-peer', 'doc-peer', NULL, 'document')"
            )
            connection.exec_driver_sql(
                "INSERT INTO reference VALUES "
                "('legacy-document-reference', 'chunk-source', NULL, "
                "'doc-peer', NULL, 'document')"
            )
            connection.exec_driver_sql(
                "INSERT INTO feedback "
                "(feedback_id, source_chunk_id, feedback, model, is_hidden) VALUES "
                "('legacy-feedback', 'chunk-source', 'legacy feedback', 'legacy-model', 0)"
            )
        ensure_retrieval_corpus_schema(engine)
    finally:
        engine.dispose()


def _migrated_legacy_engine(path: Path):
    _create_legacy_database(path)
    engine = _engine(path)
    run_schema_migrations(engine)
    return engine


def _add_scope(connection, suffix: str) -> tuple[str, str, str]:
    group_id = f"group-{suffix}"
    document_id = f"doc-{suffix}"
    chunk_id = f"chunk-{suffix}"
    connection.execute(
        text('INSERT INTO "group" (group_id) VALUES (:group_id)'),
        {"group_id": group_id},
    )
    connection.execute(
        text(
            "INSERT INTO document (document_id, user_id, content) "
            "VALUES (:document_id, 'user-a', :content)"
        ),
        {"document_id": document_id, "content": f"source {suffix}"},
    )
    connection.execute(
        text(
            "INSERT INTO chunk "
            "(chunk_id, hash, content, document_id, note_id, chunk_type) "
            "VALUES (:chunk_id, :hash, :content, :document_id, NULL, 'document')"
        ),
        {
            "chunk_id": chunk_id,
            "hash": f"hash-{suffix}",
            "content": f"chunk {suffix}",
            "document_id": document_id,
        },
    )
    return group_id, document_id, chunk_id


def _run_values(
    suffix: str,
    *,
    group_id: str,
    document_id: str,
    chunk_id: str,
    idempotency_key: str = "caller-intent-1",
    selected_count: int = 1,
) -> dict[str, object]:
    digest = "a" * 64
    return {
        "run_id": f"run-{suffix}",
        "group_id": group_id,
        "source_chunk_id": chunk_id,
        "source_document_id": document_id,
        "idempotency_key": idempotency_key,
        "bridge_schema_version": BRIDGE_SCHEMA_VERSION,
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "renderer_version": RENDERER_VERSION,
        "request_id": f"request-{suffix}",
        "result_schema_version": "retrieval-result.v2",
        "retrieval_status": "ok",
        "engine": "baseline_v1",
        "engine_version": "baseline_v1",
        "config_fingerprint": digest,
        "query_kind": "raw_git_diff_v1",
        "query_sha256": "b" * 64,
        "query_length": 19,
        "query_origin": "explicit",
        "corpus_scope_key": f"scope-{suffix}",
        "corpus_id": f"corpus-{suffix}",
        "corpus_generation_id": f"generation-{suffix}",
        "corpus_generation_version": "generation-v1",
        "corpus_manifest_hash": "c" * 64,
        "index_publication_fingerprint": "d" * 64,
        "index_published_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "index_id": f"index-{suffix}",
        "index_version": "baseline-index-v1",
        "index_schema_version": "baseline-index-schema.v1",
        "index_fingerprint": "e" * 64,
        "embedding_provider": "baseline_http",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_revision": "immutable-revision",
        "embedding_dimension": 384,
        "embedding_fingerprint": "f" * 64,
        "authorization_scope_version": "group-scope.v1",
        "authorization_scope_hash": "1" * 64,
        "candidate_count": 8,
        "retrieved_count": 6,
        "filtered_count": 0,
        "duplicate_count": 0,
        "refill_count": 0,
        "selected_count": selected_count,
        "evidence_character_count": 20 * selected_count,
        "underfilled": selected_count < 4,
        "generation_state": "pending",
        "generation_attempt_count": 0,
    }


def _artifact_values(
    suffix: str,
    *,
    group_id: str,
    artifact_key: str,
    content: str,
    source_document_id: str | None = None,
) -> dict[str, object]:
    return {
        "artifact_id": f"artifact-{suffix}",
        "group_id": group_id,
        "artifact_key": artifact_key,
        "bridge_schema_version": BRIDGE_SCHEMA_VERSION,
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "repository_id": f"repository-{suffix}",
        "repository_name": f"repo-{suffix}",
        "relative_path": f"src/{suffix}.py",
        "corpus_id": f"corpus-{suffix}",
        "corpus_file_id": f"file-{suffix}",
        "corpus_generation_id": f"generation-{suffix}",
        "corpus_generation_version": "generation-v1",
        "corpus_manifest_hash": "2" * 64,
        "index_publication_fingerprint": "3" * 64,
        "index_published_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "index_id": f"index-{suffix}",
        "index_document_id": f"index-document-{suffix}",
        "index_fingerprint": "4" * 64,
        "indexed_document_hash": "5" * 64,
        "source_document_id": source_document_id,
        "source_snapshot_id": f"snapshot-{suffix}",
        "complete_content": content,
        "whole_file_content_hash": "6" * 64,
        "byte_size": len(content.encode("utf-8")),
        "character_count": len(content),
    }


def _selected_values(
    suffix: str,
    *,
    group_id: str,
    run_id: str,
    artifact_id: str,
    ordinal: int,
    content: str,
) -> dict[str, object]:
    renderer_output = f"Repository file: repo-{suffix}/src/{suffix}.py\n\n{content}"
    return {
        "selected_evidence_id": f"selected-{suffix}",
        "group_id": group_id,
        "run_id": run_id,
        "artifact_id": artifact_id,
        "ordinal": ordinal,
        "fused_rank": ordinal,
        "selected_content": content,
        "selected_content_hash": (str(ordinal % 10) or "7") * 64,
        "selected_character_count": len(content),
        "ranking_truncated": False,
        "budget_truncated": False,
        "bm25_score": 1.25 + ordinal,
        "bm25_rank": ordinal,
        "dense_score": 0.125 + ordinal,
        "dense_rank": ordinal,
        "rrf_score": 0.03 + ordinal / 1000,
        "renderer_version": RENDERER_VERSION,
        "renderer_output": renderer_output,
        "renderer_output_hash": "8" * 64,
        "renderer_output_character_count": len(renderer_output),
    }


def test_sqlite_copied_legacy_database_upgrade_preserves_reference_order_and_bytes(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "legacy.db"
    upgraded_path = tmp_path / "upgraded-copy.db"
    _create_legacy_database(legacy_path)
    shutil.copy2(legacy_path, upgraded_path)

    engine = _engine(upgraded_path)
    try:
        before = None
        with engine.connect() as connection:
            before = connection.execute(
                text(
                    "SELECT reference_id, source_chunk_id, reference_chunk_id, "
                    "reference_document_id, reference_note_id, reference_type "
                    "FROM reference WHERE reference_id = 'legacy-reference'"
                )
            ).one()

        report = run_schema_migrations(engine)
        assert report.applied == (
            "0000_core_schema_baseline",
            "0001_baseline_evidence_bridge_v1",
        )
        with engine.connect() as connection:
            after = connection.execute(
                text(
                    "SELECT reference_id, source_chunk_id, reference_chunk_id, "
                    "reference_document_id, reference_note_id, reference_type "
                    "FROM reference WHERE reference_id = 'legacy-reference'"
                )
            ).one()
            assert connection.execute(
                text(
                    "SELECT baseline_selected_evidence_id FROM reference "
                    "WHERE reference_id = 'legacy-reference'"
                )
            ).scalar_one() is None
            assert connection.execute(
                text(
                    "SELECT reference_chunk_id, reference_document_id, "
                    "baseline_selected_evidence_id FROM reference "
                    "WHERE reference_id = 'legacy-document-reference'"
                )
            ).one() == (None, "doc-peer", None)
        assert after == before
        assert [row.migration_id for row in read_schema_migration_state(engine)] == [
            "0000_core_schema_baseline",
            "0001_baseline_evidence_bridge_v1",
        ]

        forbidden = {"retrieval_query", "query_text", "raw_query", "document_id"}
        for table_name in ("baseline_retrieval_run", "baseline_evidence_artifact"):
            columns = {column["name"] for column in inspect(engine).get_columns(table_name)}
            assert "source_document_id" in columns
            assert not forbidden & columns

        engine.dispose()
        engine = _engine(upgraded_path)
        restarted = run_schema_migrations(engine)
        assert restarted.applied == ()
        assert restarted.already_applied == (
            "0000_core_schema_baseline",
            "0001_baseline_evidence_bridge_v1",
        )
        with engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
    finally:
        engine.dispose()


def test_sqlite_group_scoped_idempotency_constraints_order_and_renderer(
    tmp_path: Path,
) -> None:
    engine = _migrated_legacy_engine(tmp_path / "constraints.db")
    try:
        with engine.begin() as connection:
            group_b, document_b, chunk_b = _add_scope(connection, "b")
            group_c, document_c, chunk_c = _add_scope(connection, "c")
            connection.execute(
                baseline_retrieval_run.insert(),
                _run_values(
                    "b",
                    group_id=group_b,
                    document_id=document_b,
                    chunk_id=chunk_b,
                    selected_count=2,
                ),
            )
            # The same caller-provided opaque intent is valid in another group.
            connection.execute(
                baseline_retrieval_run.insert(),
                _run_values(
                    "c",
                    group_id=group_c,
                    document_id=document_c,
                    chunk_id=chunk_c,
                ),
            )
            connection.execute(
                baseline_evidence_artifact.insert(),
                _artifact_values(
                    "b-one",
                    group_id=group_b,
                    artifact_key="9" * 64,
                    content="first selected body",
                ),
            )
            connection.execute(
                baseline_evidence_artifact.insert(),
                _artifact_values(
                    "b-two",
                    group_id=group_b,
                    artifact_key="0" * 64,
                    content="second selected body",
                ),
            )
            connection.execute(
                baseline_evidence_artifact.insert(),
                _artifact_values(
                    "c-one",
                    group_id=group_c,
                    artifact_key="9" * 64,
                    content="other group body",
                ),
            )
            # Insert in reverse ordinal order; durable order is explicit.
            connection.execute(
                baseline_selected_evidence.insert(),
                _selected_values(
                    "b-two",
                    group_id=group_b,
                    run_id="run-b",
                    artifact_id="artifact-b-two",
                    ordinal=2,
                    content="second selected body",
                ),
            )
            connection.execute(
                baseline_selected_evidence.insert(),
                _selected_values(
                    "b-one",
                    group_id=group_b,
                    run_id="run-b",
                    artifact_id="artifact-b-one",
                    ordinal=1,
                    content="first selected body",
                ),
            )

        with pytest.raises(IntegrityError), engine.begin() as connection:
            values = _run_values(
                "b-duplicate",
                group_id=group_b,
                document_id=document_b,
                chunk_id=chunk_b,
            )
            connection.execute(baseline_retrieval_run.insert(), values)
        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(
                baseline_evidence_artifact.insert(),
                _artifact_values(
                    "b-duplicate",
                    group_id=group_b,
                    artifact_key="9" * 64,
                    content="duplicate key",
                ),
            )
        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(
                baseline_selected_evidence.insert(),
                _selected_values(
                    "cross-scope",
                    group_id=group_c,
                    run_id="run-b",
                    artifact_id="artifact-c-one",
                    ordinal=3,
                    content="cross scope",
                ),
            )
        with pytest.raises(IntegrityError), engine.begin() as connection:
            bad = _selected_values(
                "bad-ordinal",
                group_id=group_c,
                run_id="run-c",
                artifact_id="artifact-c-one",
                ordinal=0,
                content="bad ordinal",
            )
            connection.execute(baseline_selected_evidence.insert(), bad)

        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT ordinal, renderer_version, renderer_output, "
                    "bm25_rank, dense_rank, rrf_score "
                    "FROM baseline_selected_evidence WHERE run_id = 'run-b' "
                    "ORDER BY ordinal"
                )
            ).all()
        assert [row.ordinal for row in rows] == [1, 2]
        assert rows[0].renderer_version == RENDERER_VERSION
        assert rows[0].renderer_output.endswith("first selected body")
        assert (rows[0].bm25_rank, rows[0].dense_rank, rows[0].rrf_score) == (
            1,
            1,
            pytest.approx(0.031),
        )
    finally:
        engine.dispose()


def test_sqlite_reference_feedback_and_lifecycle_constraints(tmp_path: Path) -> None:
    engine = _migrated_legacy_engine(tmp_path / "lifecycle.db")
    try:
        with engine.begin() as connection:
            group_id, document_id, chunk_id = _add_scope(connection, "life")
            connection.execute(
                baseline_retrieval_run.insert(),
                _run_values(
                    "life",
                    group_id=group_id,
                    document_id=document_id,
                    chunk_id=chunk_id,
                ),
            )
            connection.execute(
                baseline_evidence_artifact.insert(),
                _artifact_values(
                    "life",
                    group_id=group_id,
                    artifact_key="7" * 64,
                    content="auditable evidence",
                ),
            )
            connection.execute(
                baseline_selected_evidence.insert(),
                _selected_values(
                    "life",
                    group_id=group_id,
                    run_id="run-life",
                    artifact_id="artifact-life",
                    ordinal=1,
                    content="auditable evidence",
                ),
            )
            connection.execute(
                text(
                    "INSERT INTO reference "
                    "(reference_id, source_chunk_id, reference_chunk_id, "
                    "reference_document_id, reference_note_id, reference_type, "
                    "baseline_selected_evidence_id) VALUES "
                    "('baseline-reference', :chunk_id, NULL, NULL, NULL, "
                    "'baseline_file', 'selected-life')"
                ),
                {"chunk_id": chunk_id},
            )
            connection.execute(
                text(
                    "INSERT INTO feedback "
                    "(feedback_id, source_chunk_id, feedback, model, is_hidden, "
                    "baseline_retrieval_run_id, baseline_finding_ordinal) VALUES "
                    "('baseline-feedback', :chunk_id, 'finding', 'model', 0, "
                    "'run-life', 1)"
                ),
                {"chunk_id": chunk_id},
            )

        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO reference "
                    "(reference_id, source_chunk_id, reference_chunk_id, "
                    "reference_document_id, reference_note_id, reference_type, "
                    "baseline_selected_evidence_id) VALUES "
                    "('invalid-target', :source, 'chunk-peer', NULL, NULL, "
                    "'document', 'selected-life')"
                ),
                {"source": chunk_id},
            )
        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO feedback "
                    "(feedback_id, source_chunk_id, feedback, model, is_hidden, "
                    "baseline_retrieval_run_id, baseline_finding_ordinal) VALUES "
                    "('invalid-feedback', :chunk_id, 'bad', 'model', 0, "
                    "'run-life', NULL)"
                ),
                {"chunk_id": chunk_id},
            )
        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(
                baseline_evidence_artifact.delete().where(
                    baseline_evidence_artifact.c.artifact_id == "artifact-life"
                )
            )

        # Corpus/index lifecycle has no retention FK into immutable evidence.
        artifact_foreign_tables = {
            row["referred_table"]
            for row in inspect(engine).get_foreign_keys("baseline_evidence_artifact")
        }
        assert not {
            "retrieval_corpus",
            "retrieval_corpus_generation",
            "retrieval_baseline_index_build",
            "retrieval_baseline_index_publication",
        } & artifact_foreign_tables

        with engine.begin() as connection:
            connection.execute(
                baseline_retrieval_run.delete().where(
                    baseline_retrieval_run.c.run_id == "run-life"
                )
            )
        with engine.connect() as connection:
            assert connection.execute(
                text("SELECT count(*) FROM baseline_selected_evidence")
            ).scalar_one() == 0
            assert connection.execute(
                text("SELECT count(*) FROM reference WHERE reference_id='baseline-reference'")
            ).scalar_one() == 0
            assert connection.execute(
                text("SELECT count(*) FROM feedback WHERE feedback_id='baseline-feedback'")
            ).scalar_one() == 0
            assert connection.execute(
                text("SELECT count(*) FROM baseline_evidence_artifact")
            ).scalar_one() == 1

        # Recreate a run/selection, then one group deletion removes the scope.
        with engine.begin() as connection:
            connection.execute(
                baseline_retrieval_run.insert(),
                _run_values(
                    "life-2",
                    group_id=group_id,
                    document_id=document_id,
                    chunk_id=chunk_id,
                ),
            )
            selected = _selected_values(
                "life-2",
                group_id=group_id,
                run_id="run-life-2",
                artifact_id="artifact-life",
                ordinal=1,
                content="auditable evidence",
            )
            connection.execute(baseline_selected_evidence.insert(), selected)
            connection.execute(
                text('DELETE FROM "group" WHERE group_id = :group_id'),
                {"group_id": group_id},
            )
        with engine.connect() as connection:
            for table_name in (
                "baseline_retrieval_run",
                "baseline_evidence_artifact",
                "baseline_selected_evidence",
            ):
                assert connection.execute(
                    text(f"SELECT count(*) FROM {table_name}")
                ).scalar_one() == 0

        # Artifact audit content survives an independently deleted source
        # document, but its optional Core pointer is cleared.
        with engine.begin() as connection:
            group_id, document_id, _chunk_id = _add_scope(connection, "source-delete")
            connection.execute(
                baseline_evidence_artifact.insert(),
                _artifact_values(
                    "source-delete",
                    group_id=group_id,
                    artifact_key="4" * 64,
                    content="retained source snapshot",
                    source_document_id=document_id,
                ),
            )
            connection.execute(
                text("DELETE FROM document WHERE document_id = :document_id"),
                {"document_id": document_id},
            )
        with engine.connect() as connection:
            assert connection.execute(
                text(
                    "SELECT source_document_id FROM baseline_evidence_artifact "
                    "WHERE artifact_id = 'artifact-source-delete'"
                )
            ).scalar_one() is None
        with engine.begin() as connection:
            connection.execute(
                text('DELETE FROM "group" WHERE group_id = :group_id'),
                {"group_id": group_id},
            )
    finally:
        engine.dispose()


def test_sqlite_failed_bridge_migration_requires_reviewed_recovery(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "failed-recovery.db"
    _create_legacy_database(database_path)
    engine = _engine(database_path)
    production = CORE_SCHEMA_MIGRATIONS[1]

    def fail_after_real_upgrade(connection) -> None:
        production.upgrade(connection)
        raise RuntimeError("injected bridge migration failure")

    failing = replace(production, upgrade=fail_after_real_upgrade)
    try:
        with pytest.raises(SchemaMigrationError) as error:
            run_schema_migrations(engine, (CORE_SCHEMA_MIGRATIONS[0], failing))
        assert error.value.migration_id == "0001_baseline_evidence_bridge_v1"
        assert error.value.code == "upgrade_failed"
        assert "baseline_retrieval_run" not in inspect(engine).get_table_names()
        assert "baseline_selected_evidence_id" not in {
            column["name"] for column in inspect(engine).get_columns("reference")
        }
        assert [(row.migration_id, row.state) for row in read_schema_migration_state(engine)] == [
            ("0001_baseline_evidence_bridge_v1", "failed")
        ]

        # Reviewed recovery: diagnose/backup first, then clear only the failed
        # marker and rerun the immutable production definition.
        with engine.begin() as connection:
            connection.execute(
                schema_migration_table.delete().where(
                    schema_migration_table.c.migration_id
                    == "0001_baseline_evidence_bridge_v1"
                )
            )
        recovered = run_schema_migrations(engine)
        assert recovered.applied == (
            "0000_core_schema_baseline",
            "0001_baseline_evidence_bridge_v1",
        )
    finally:
        engine.dispose()
