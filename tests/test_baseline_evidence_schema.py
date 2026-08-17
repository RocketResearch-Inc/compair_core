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


def _add_mutable_corpus_lifecycle(connection, suffix: str) -> None:
    digest = "9" * 64
    connection.execute(
        text(
            "INSERT INTO retrieval_corpus "
            "(corpus_id, scope_key, changed_repository_id, source_document_id, "
            "active_generation_id, created_at, updated_at) VALUES "
            "(:corpus, :scope, :changed, NULL, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ),
        {
            "corpus": f"corpus-{suffix}",
            "scope": f"scope-{suffix}",
            "changed": f"changed-{suffix}",
        },
    )
    connection.execute(
        text(
            "INSERT INTO retrieval_corpus_generation "
            "(generation_id, corpus_id, generation_version, expected_repository_count, "
            "expected_file_count, status, manifest_hash, created_at, validated_at, activated_at) "
            "VALUES (:generation, :corpus, 'generation-v1', 1, 1, 'active', :hash, "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ),
        {
            "generation": f"generation-{suffix}",
            "corpus": f"corpus-{suffix}",
            "hash": digest,
        },
    )
    connection.execute(
        text(
            "INSERT INTO retrieval_corpus_file "
            "(file_id, generation_id, repository_id, repository_name, relative_path, "
            "file_state, content_hash, byte_size, content) VALUES "
            "(:file, :generation, :repository, :name, :path, 'supported', :hash, 5, 'first')"
        ),
        {
            "file": f"file-{suffix}",
            "generation": f"generation-{suffix}",
            "repository": f"repository-{suffix}",
            "name": f"repo-{suffix}",
            "path": f"src/{suffix}.py",
            "hash": digest,
        },
    )
    connection.execute(
        text(
            "INSERT INTO retrieval_baseline_index_build "
            "(index_id, generation_id, index_version, index_schema_version, "
            "document_format_version, corpus_manifest_hash, tokenizer_version, "
            "embedding_provider, embedding_model, embedding_revision, embedding_dimension, "
            "embedding_fingerprint, engine_config_fingerprint, expected_document_count, "
            "status, indexed_document_count, total_token_count, document_manifest_hash, "
            "lexical_manifest_hash, dense_manifest_hash, created_at, validated_at, published_at) "
            "VALUES (:index, :generation, 'baseline-index-v1', 'baseline-index-schema.v1', "
            "'whole-file-v1', :hash, 'baseline-tokenizer-v1', 'baseline_http', "
            "'BAAI/bge-small-en-v1.5', 'immutable-revision', 384, :hash, :hash, 1, "
            "'compatible', 1, 1, :hash, :hash, :hash, CURRENT_TIMESTAMP, "
            "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
        ),
        {
            "index": f"index-{suffix}",
            "generation": f"generation-{suffix}",
            "hash": digest,
        },
    )
    connection.execute(
        text(
            "INSERT INTO retrieval_baseline_index_publication (corpus_id, index_id, published_at) "
            "VALUES (:corpus, :index, CURRENT_TIMESTAMP)"
        ),
        {"corpus": f"corpus-{suffix}", "index": f"index-{suffix}"},
    )
    connection.execute(
        text(
            "UPDATE retrieval_corpus SET active_generation_id=:generation "
            "WHERE corpus_id=:corpus"
        ),
        {"generation": f"generation-{suffix}", "corpus": f"corpus-{suffix}"},
    )


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
            "0002_baseline_evidence_retention_v1",
            "0003_baseline_generation_state_v1",
            "0004_baseline_notification_outbox_v1",
            "0005_baseline_control_plane_staging_v1",
            "0006_baseline_control_plane_continuation_v1",
            "0007_baseline_control_plane_ingestion_worker_v1",
            "0008_baseline_compatible_index_job_v1",
            "0009_baseline_run_job_v1",
            "0010_baseline_document_source_scope_v1",
            "0011_baseline_run_executor_v1",
            "0012_baseline_control_generation_v1",
            "0013_baseline_database_worker_v1",
        )
        with engine.connect() as connection:
            after = connection.execute(
                text(
                    "SELECT reference_id, source_chunk_id, reference_chunk_id, "
                    "reference_document_id, reference_note_id, reference_type "
                    "FROM reference WHERE reference_id = 'legacy-reference'"
                )
            ).one()
            assert (
                connection.execute(
                    text(
                        "SELECT baseline_selected_evidence_id FROM reference "
                        "WHERE reference_id = 'legacy-reference'"
                    )
                ).scalar_one()
                is None
            )
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
            "0002_baseline_evidence_retention_v1",
            "0003_baseline_generation_state_v1",
            "0004_baseline_notification_outbox_v1",
            "0005_baseline_control_plane_staging_v1",
            "0006_baseline_control_plane_continuation_v1",
            "0007_baseline_control_plane_ingestion_worker_v1",
            "0008_baseline_compatible_index_job_v1",
            "0009_baseline_run_job_v1",
            "0010_baseline_document_source_scope_v1",
            "0011_baseline_run_executor_v1",
            "0012_baseline_control_generation_v1",
            "0013_baseline_database_worker_v1",
        ]

        forbidden = {"retrieval_query", "query_text", "raw_query", "document_id"}
        for table_name in ("baseline_retrieval_run", "baseline_evidence_artifact"):
            columns = {
                column["name"] for column in inspect(engine).get_columns(table_name)
            }
            assert "source_document_id" in columns
            assert not forbidden & columns

        engine.dispose()
        engine = _engine(upgraded_path)
        restarted = run_schema_migrations(engine)
        assert restarted.applied == ()
        assert restarted.already_applied == (
            "0000_core_schema_baseline",
            "0001_baseline_evidence_bridge_v1",
            "0002_baseline_evidence_retention_v1",
            "0003_baseline_generation_state_v1",
            "0004_baseline_notification_outbox_v1",
            "0005_baseline_control_plane_staging_v1",
            "0006_baseline_control_plane_continuation_v1",
            "0007_baseline_control_plane_ingestion_worker_v1",
            "0008_baseline_compatible_index_job_v1",
            "0009_baseline_run_job_v1",
            "0010_baseline_document_source_scope_v1",
            "0011_baseline_run_executor_v1",
            "0012_baseline_control_generation_v1",
            "0013_baseline_database_worker_v1",
        )
        with engine.connect() as connection:
            assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
    finally:
        engine.dispose()


def test_existing_retrieval_runs_migrate_deterministically_to_legacy_chunk(
    tmp_path: Path,
) -> None:
    engine = _engine(tmp_path / "legacy-scope-backfill.db")
    try:
        _create_legacy_database(tmp_path / "legacy-scope-backfill.db")
        run_schema_migrations(engine, CORE_SCHEMA_MIGRATIONS[:10])
        with engine.begin() as connection:
            group_id, document_id, chunk_id = _add_scope(connection, "scope-backfill")
            original = _run_values(
                "scope-backfill",
                group_id=group_id,
                document_id=document_id,
                chunk_id=chunk_id,
            )
            connection.execute(baseline_retrieval_run.insert().values(**original))

        report = run_schema_migrations(engine, CORE_SCHEMA_MIGRATIONS[:11])
        assert report.applied == ("0010_baseline_document_source_scope_v1",)
        with engine.connect() as connection:
            row = (
                connection.execute(
                    text(
                        "SELECT source_scope_version, source_scope, source_chunk_id, "
                        "source_document_id FROM baseline_retrieval_run "
                        "WHERE run_id = :run_id"
                    ),
                    {"run_id": original["run_id"]},
                )
                .mappings()
                .one()
            )
        assert dict(row) == {
            "source_scope_version": "baseline-source-scope.v1",
            "source_scope": "legacy_chunk",
            "source_chunk_id": chunk_id,
            "source_document_id": document_id,
        }
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
                    source_document_id=document_id,
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
        assert (
            not {
                "retrieval_corpus",
                "retrieval_corpus_generation",
                "retrieval_baseline_index_build",
                "retrieval_baseline_index_publication",
            }
            & artifact_foreign_tables
        )

        # A run with selected evidence cannot become a hidden non-group purge.
        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(
                baseline_retrieval_run.delete().where(
                    baseline_retrieval_run.c.run_id == "run-life"
                )
            )

        # Normal source deletion preserves immutable evidence, Reference, and
        # Feedback, clearing only source provenance pointers.
        with engine.begin() as connection:
            connection.execute(
                text("DELETE FROM document WHERE document_id = :document_id"),
                {"document_id": document_id},
            )
        with engine.connect() as connection:
            assert (
                connection.execute(
                    text("SELECT count(*) FROM baseline_selected_evidence")
                ).scalar_one()
                == 1
            )
            assert (
                connection.execute(
                    text(
                        "SELECT count(*) FROM reference WHERE reference_id='baseline-reference'"
                    )
                ).scalar_one()
                == 1
            )
            assert (
                connection.execute(
                    text(
                        "SELECT count(*) FROM feedback WHERE feedback_id='baseline-feedback'"
                    )
                ).scalar_one()
                == 1
            )
            assert (
                connection.execute(
                    text("SELECT count(*) FROM baseline_evidence_artifact")
                ).scalar_one()
                == 1
            )
            assert connection.execute(
                text(
                    "SELECT source_chunk_id, source_document_id "
                    "FROM baseline_retrieval_run WHERE run_id='run-life'"
                )
            ).one() == (None, None)
            assert (
                connection.execute(
                    text(
                        "SELECT source_document_id FROM baseline_evidence_artifact "
                        "WHERE artifact_id='artifact-life'"
                    )
                ).scalar_one()
                is None
            )
            assert (
                connection.execute(
                    text(
                        "SELECT source_chunk_id FROM reference "
                        "WHERE reference_id='baseline-reference'"
                    )
                ).scalar_one()
                is None
            )
            assert (
                connection.execute(
                    text(
                        "SELECT source_chunk_id FROM feedback "
                        "WHERE feedback_id='baseline-feedback'"
                    )
                ).scalar_one()
                is None
            )

        # Restart persistence does not depend on the removed source objects.
        engine.dispose()
        engine = _engine(tmp_path / "lifecycle.db")
        assert run_schema_migrations(engine).applied == ()
        with engine.connect() as connection:
            assert (
                connection.execute(
                    text("SELECT renderer_output FROM baseline_selected_evidence")
                )
                .scalar_one()
                .endswith("auditable evidence")
            )

        # Legacy source-owned rows retain their historical cascade behavior.
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM chunk WHERE chunk_id='chunk-source'"))
        with engine.connect() as connection:
            assert (
                connection.execute(
                    text(
                        "SELECT count(*) FROM reference WHERE reference_id LIKE 'legacy-%'"
                    )
                ).scalar_one()
                == 0
            )
            assert (
                connection.execute(
                    text(
                        "SELECT count(*) FROM feedback WHERE feedback_id='legacy-feedback'"
                    )
                ).scalar_one()
                == 0
            )

        # Group deletion is the privacy boundary and removes the entire scope.
        with engine.begin() as connection:
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
                assert (
                    connection.execute(
                        text(f"SELECT count(*) FROM {table_name}")
                    ).scalar_one()
                    == 0
                )
            assert (
                connection.execute(
                    text(
                        "SELECT count(*) FROM reference "
                        "WHERE reference_id='baseline-reference'"
                    )
                ).scalar_one()
                == 0
            )
            assert (
                connection.execute(
                    text(
                        "SELECT count(*) FROM feedback "
                        "WHERE feedback_id='baseline-feedback'"
                    )
                ).scalar_one()
                == 0
            )

        # A direct Chunk delete (without deleting its Document) clears only
        # chunk provenance and leaves document provenance intact.
        with engine.begin() as connection:
            group_id, document_id, chunk_id = _add_scope(connection, "chunk-only")
            connection.execute(
                baseline_retrieval_run.insert(),
                _run_values(
                    "chunk-only",
                    group_id=group_id,
                    document_id=document_id,
                    chunk_id=chunk_id,
                ),
            )
            connection.execute(
                baseline_evidence_artifact.insert(),
                _artifact_values(
                    "chunk-only",
                    group_id=group_id,
                    artifact_key="5" * 64,
                    content="chunk retained evidence",
                ),
            )
            connection.execute(
                baseline_selected_evidence.insert(),
                _selected_values(
                    "chunk-only",
                    group_id=group_id,
                    run_id="run-chunk-only",
                    artifact_id="artifact-chunk-only",
                    ordinal=1,
                    content="chunk retained evidence",
                ),
            )
            connection.execute(
                text(
                    "INSERT INTO reference "
                    "(reference_id, source_chunk_id, reference_type, "
                    "baseline_selected_evidence_id) VALUES "
                    "('chunk-only-reference', :source, 'baseline_file', "
                    "'selected-chunk-only')"
                ),
                {"source": chunk_id},
            )
            connection.execute(
                text(
                    "INSERT INTO feedback "
                    "(feedback_id, source_chunk_id, feedback, model, is_hidden, "
                    "baseline_retrieval_run_id, baseline_finding_ordinal) VALUES "
                    "('chunk-only-feedback', :source, 'retained', 'model', 0, "
                    "'run-chunk-only', 1)"
                ),
                {"source": chunk_id},
            )
            connection.execute(
                text("DELETE FROM chunk WHERE chunk_id=:chunk_id"),
                {"chunk_id": chunk_id},
            )
        with engine.connect() as connection:
            assert connection.execute(
                text(
                    "SELECT source_chunk_id, source_document_id "
                    "FROM baseline_retrieval_run WHERE run_id='run-chunk-only'"
                )
            ).one() == (None, document_id)
            assert (
                connection.execute(
                    text(
                        "SELECT source_chunk_id FROM reference "
                        "WHERE reference_id='chunk-only-reference'"
                    )
                ).scalar_one()
                is None
            )
            assert (
                connection.execute(
                    text(
                        "SELECT source_chunk_id FROM feedback "
                        "WHERE feedback_id='chunk-only-feedback'"
                    )
                ).scalar_one()
                is None
            )
        with engine.begin() as connection:
            connection.execute(
                text('DELETE FROM "group" WHERE group_id=:group_id'),
                {"group_id": group_id},
            )
    finally:
        engine.dispose()


def test_sqlite_corpus_index_rename_and_explicit_retention_purge(
    tmp_path: Path,
) -> None:
    engine = _migrated_legacy_engine(tmp_path / "mutable-provenance.db")
    try:
        with engine.begin() as connection:
            group_id, document_id, chunk_id = _add_scope(connection, "mutable")
            _add_mutable_corpus_lifecycle(connection, "mutable")
            connection.execute(
                baseline_retrieval_run.insert(),
                _run_values(
                    "mutable",
                    group_id=group_id,
                    document_id=document_id,
                    chunk_id=chunk_id,
                ),
            )
            connection.execute(
                baseline_evidence_artifact.insert(),
                _artifact_values(
                    "mutable",
                    group_id=group_id,
                    artifact_key="4" * 64,
                    content="immutable retained bytes",
                ),
            )
            connection.execute(
                baseline_selected_evidence.insert(),
                _selected_values(
                    "mutable",
                    group_id=group_id,
                    run_id="run-mutable",
                    artifact_id="artifact-mutable",
                    ordinal=1,
                    content="immutable retained bytes",
                ),
            )
            connection.execute(
                text(
                    "INSERT INTO reference "
                    "(reference_id, source_chunk_id, reference_type, "
                    "baseline_selected_evidence_id) VALUES "
                    "('mutable-reference', :source, 'baseline_file', 'selected-mutable')"
                ),
                {"source": chunk_id},
            )
            connection.execute(
                text(
                    "INSERT INTO feedback "
                    "(feedback_id, source_chunk_id, feedback, model, is_hidden, "
                    "baseline_retrieval_run_id, baseline_finding_ordinal) VALUES "
                    "('mutable-feedback', :source, 'retained', 'model', 0, "
                    "'run-mutable', 1)"
                ),
                {"source": chunk_id},
            )

        with engine.connect() as connection:
            before = connection.execute(
                text(
                    "SELECT repository_id, repository_name, relative_path, complete_content, "
                    "corpus_generation_id, index_id FROM baseline_evidence_artifact "
                    "WHERE artifact_id='artifact-mutable'"
                )
            ).one()

        # A later immutable generation can rename the repository/path. Deleting
        # the former publication and generation cannot reach copied evidence.
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO retrieval_corpus_generation "
                    "(generation_id, corpus_id, generation_version, expected_repository_count, "
                    "expected_file_count, status, manifest_hash, created_at, validated_at, activated_at) "
                    "VALUES ('generation-mutable-v2', 'corpus-mutable', 'generation-v2', "
                    "1, 1, 'active', :hash, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {"hash": "8" * 64},
            )
            connection.execute(
                text(
                    "INSERT INTO retrieval_corpus_file "
                    "(file_id, generation_id, repository_id, repository_name, relative_path, "
                    "file_state, content_hash, byte_size, content) VALUES "
                    "('file-mutable-v2', 'generation-mutable-v2', 'repository-mutable', "
                    "'renamed-repo', 'renamed/location.py', 'supported', :hash, 6, 'second')"
                ),
                {"hash": "8" * 64},
            )
            connection.execute(
                text(
                    "UPDATE retrieval_corpus SET active_generation_id='generation-mutable-v2' "
                    "WHERE corpus_id='corpus-mutable'"
                )
            )
            connection.execute(
                text(
                    "DELETE FROM retrieval_baseline_index_publication "
                    "WHERE corpus_id='corpus-mutable'"
                )
            )
            connection.execute(
                text(
                    "DELETE FROM retrieval_corpus_generation "
                    "WHERE generation_id='generation-mutable'"
                )
            )
        with engine.connect() as connection:
            after = connection.execute(
                text(
                    "SELECT repository_id, repository_name, relative_path, complete_content, "
                    "corpus_generation_id, index_id FROM baseline_evidence_artifact "
                    "WHERE artifact_id='artifact-mutable'"
                )
            ).one()
            assert after == before
            assert (
                connection.execute(
                    text(
                        "SELECT count(*) FROM reference WHERE reference_id='mutable-reference'"
                    )
                ).scalar_one()
                == 1
            )
            assert (
                connection.execute(
                    text(
                        "SELECT count(*) FROM feedback WHERE feedback_id='mutable-feedback'"
                    )
                ).scalar_one()
                == 1
            )

        # This is the schema-level ordering an authorized audited retention
        # purge must use. No ordinary lifecycle delete can substitute for it.
        with engine.begin() as connection:
            connection.execute(
                baseline_selected_evidence.delete().where(
                    baseline_selected_evidence.c.selected_evidence_id
                    == "selected-mutable"
                )
            )
            connection.execute(
                baseline_retrieval_run.delete().where(
                    baseline_retrieval_run.c.run_id == "run-mutable"
                )
            )
            connection.execute(
                baseline_evidence_artifact.delete().where(
                    baseline_evidence_artifact.c.artifact_id == "artifact-mutable"
                )
            )
        with engine.connect() as connection:
            for table_name in (
                "baseline_retrieval_run",
                "baseline_evidence_artifact",
                "baseline_selected_evidence",
            ):
                assert (
                    connection.execute(
                        text(f"SELECT count(*) FROM {table_name}")
                    ).scalar_one()
                    == 0
                )
            assert (
                connection.execute(
                    text(
                        "SELECT count(*) FROM reference WHERE reference_id='mutable-reference'"
                    )
                ).scalar_one()
                == 0
            )
            assert (
                connection.execute(
                    text(
                        "SELECT count(*) FROM feedback WHERE feedback_id='mutable-feedback'"
                    )
                ).scalar_one()
                == 0
            )
    finally:
        engine.dispose()


def test_core_bulk_document_delete_builds_legacy_only_predicates(monkeypatch) -> None:
    from types import SimpleNamespace

    from compair_core.api import _delete_document_records

    class FakeColumn:
        def __init__(self, name: str) -> None:
            self.name = name

        def in_(self, values):
            return (self.name, tuple(values))

    class FakeQuery:
        def __init__(self, session, entity) -> None:
            self.session = session
            self.entity = entity
            self.filters = ()

        def filter(self, *filters):
            self.filters = filters
            return self

        def all(self):
            if getattr(self.entity, "name", None) == "chunk_id":
                return [("chunk-bulk",)]
            return []

        def delete(self, *, synchronize_session):
            assert synchronize_session is False
            self.session.deletions.append((self.entity, self.filters))

    class FakeSession:
        def __init__(self) -> None:
            self.deletions = []

        def query(self, entity):
            return FakeQuery(self, entity)

        def execute(self, _statement):
            raise AssertionError("no association table is configured")

    document_model = SimpleNamespace(document_id=FakeColumn("document_id"))
    note_model = SimpleNamespace(
        note_id=FakeColumn("note_id"), document_id=FakeColumn("note_document_id")
    )
    chunk_model = SimpleNamespace(
        chunk_id=FakeColumn("chunk_id"), document_id=FakeColumn("chunk_document_id")
    )
    feedback_model = SimpleNamespace(source_chunk_id=FakeColumn("feedback_source"))
    reference_model = SimpleNamespace(
        source_chunk_id=FakeColumn("reference_source"),
        reference_document_id=FakeColumn("reference_document_id"),
    )
    fake_models = SimpleNamespace(
        Document=document_model,
        Note=note_model,
        Chunk=chunk_model,
        Feedback=feedback_model,
        Reference=reference_model,
    )
    monkeypatch.setattr("compair_core.api.models", fake_models)
    session = FakeSession()

    _delete_document_records(session, ["doc-bulk"])

    feedback_delete = next(row for row in session.deletions if row[0] is feedback_model)
    source_reference_delete = next(
        row
        for row in session.deletions
        if row[0] is reference_model
        and any(
            str(value) == "baseline_selected_evidence_id IS NULL" for value in row[1]
        )
    )
    assert any(
        str(value) == "baseline_retrieval_run_id IS NULL"
        for value in feedback_delete[1]
    )
    assert any(
        str(value) == "baseline_selected_evidence_id IS NULL"
        for value in source_reference_delete[1]
    )


def test_sqlite_failed_retention_copy_swap_rolls_back_and_recovers(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "failed-retention.db"
    _create_legacy_database(database_path)
    engine = _engine(database_path)
    run_schema_migrations(engine, CORE_SCHEMA_MIGRATIONS[:2])
    production = CORE_SCHEMA_MIGRATIONS[2]

    def fail_after_real_upgrade(connection) -> None:
        production.upgrade(connection)
        raise RuntimeError("injected retention migration failure")

    failing = replace(production, upgrade=fail_after_real_upgrade)
    try:
        with pytest.raises(SchemaMigrationError) as error:
            run_schema_migrations(engine, (*CORE_SCHEMA_MIGRATIONS[:2], failing))
        assert error.value.migration_id == "0002_baseline_evidence_retention_v1"
        assert error.value.code == "upgrade_failed"
        assert {
            column["name"]: column["nullable"]
            for column in inspect(engine).get_columns("reference")
        }["source_chunk_id"] is False
        assert [
            (row.migration_id, row.state) for row in read_schema_migration_state(engine)
        ] == [
            ("0000_core_schema_baseline", "applied"),
            ("0001_baseline_evidence_bridge_v1", "applied"),
            ("0002_baseline_evidence_retention_v1", "failed"),
        ]
        with engine.begin() as connection:
            connection.execute(
                schema_migration_table.delete().where(
                    schema_migration_table.c.migration_id
                    == "0002_baseline_evidence_retention_v1"
                )
            )
        assert run_schema_migrations(engine).applied == (
            "0002_baseline_evidence_retention_v1",
            "0003_baseline_generation_state_v1",
            "0004_baseline_notification_outbox_v1",
            "0005_baseline_control_plane_staging_v1",
            "0006_baseline_control_plane_continuation_v1",
            "0007_baseline_control_plane_ingestion_worker_v1",
            "0008_baseline_compatible_index_job_v1",
            "0009_baseline_run_job_v1",
            "0010_baseline_document_source_scope_v1",
            "0011_baseline_run_executor_v1",
            "0012_baseline_control_generation_v1",
            "0013_baseline_database_worker_v1",
        )
        assert {
            column["name"]: column["nullable"]
            for column in inspect(engine).get_columns("reference")
        }["source_chunk_id"] is True
    finally:
        engine.dispose()


def test_sqlite_failed_generation_state_copy_swap_rolls_back_and_recovers(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "failed-generation-state.db"
    _create_legacy_database(database_path)
    engine = _engine(database_path)
    run_schema_migrations(engine, CORE_SCHEMA_MIGRATIONS[:3])
    production = CORE_SCHEMA_MIGRATIONS[3]

    def fail_after_real_upgrade(connection) -> None:
        production.upgrade(connection)
        raise RuntimeError("injected generation migration failure")

    failing = replace(production, upgrade=fail_after_real_upgrade)
    try:
        with pytest.raises(SchemaMigrationError) as error:
            run_schema_migrations(engine, (*CORE_SCHEMA_MIGRATIONS[:3], failing))
        assert error.value.migration_id == "0003_baseline_generation_state_v1"
        assert error.value.code == "upgrade_failed"
        assert "generation_lease_token" not in {
            column["name"]
            for column in inspect(engine).get_columns("baseline_retrieval_run")
        }
        assert [
            (row.migration_id, row.state) for row in read_schema_migration_state(engine)
        ] == [
            ("0000_core_schema_baseline", "applied"),
            ("0001_baseline_evidence_bridge_v1", "applied"),
            ("0002_baseline_evidence_retention_v1", "applied"),
            ("0003_baseline_generation_state_v1", "failed"),
        ]
        with engine.begin() as connection:
            connection.execute(
                schema_migration_table.delete().where(
                    schema_migration_table.c.migration_id
                    == "0003_baseline_generation_state_v1"
                )
            )
        assert run_schema_migrations(engine).applied == (
            "0003_baseline_generation_state_v1",
            "0004_baseline_notification_outbox_v1",
            "0005_baseline_control_plane_staging_v1",
            "0006_baseline_control_plane_continuation_v1",
            "0007_baseline_control_plane_ingestion_worker_v1",
            "0008_baseline_compatible_index_job_v1",
            "0009_baseline_run_job_v1",
            "0010_baseline_document_source_scope_v1",
            "0011_baseline_run_executor_v1",
            "0012_baseline_control_generation_v1",
            "0013_baseline_database_worker_v1",
        )
        assert "generation_lease_token" in {
            column["name"]
            for column in inspect(engine).get_columns("baseline_retrieval_run")
        }
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
        assert [
            (row.migration_id, row.state) for row in read_schema_migration_state(engine)
        ] == [("0001_baseline_evidence_bridge_v1", "failed")]

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
            "0002_baseline_evidence_retention_v1",
            "0003_baseline_generation_state_v1",
            "0004_baseline_notification_outbox_v1",
            "0005_baseline_control_plane_staging_v1",
            "0006_baseline_control_plane_continuation_v1",
            "0007_baseline_control_plane_ingestion_worker_v1",
            "0008_baseline_compatible_index_job_v1",
            "0009_baseline_run_job_v1",
            "0010_baseline_document_source_scope_v1",
            "0011_baseline_run_executor_v1",
            "0012_baseline_control_generation_v1",
            "0013_baseline_database_worker_v1",
        )
    finally:
        engine.dispose()
