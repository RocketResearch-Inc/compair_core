"""Real PostgreSQL coverage for the Core migration foundation.

Run against a disposable database with::

    COMPAIR_TEST_POSTGRES_URL=postgresql+psycopg2://user:pass@127.0.0.1:5432/db \
      pytest -q tests/test_schema_migrations_postgres.py
"""

from __future__ import annotations

import os
from dataclasses import replace
from uuid import uuid4

import pytest
from test_baseline_evidence_schema import (
    IntegrityError,
    _add_mutable_corpus_lifecycle,
    _add_scope,
    _artifact_values,
    _run_values,
    _selected_values,
    text,
)

from compair_core import db as core_db
from compair_core import schema_migrations as migration_registry
from compair_core.baseline_evidence_schema import (
    baseline_evidence_artifact,
    baseline_retrieval_run,
    baseline_selected_evidence,
)
from compair_core.schema_migrations import (
    CORE_SCHEMA_MIGRATIONS,
    SchemaMigration,
    SchemaMigrationError,
    read_schema_migration_state,
    run_schema_migrations,
)

POSTGRES_URL = os.getenv("COMPAIR_TEST_POSTGRES_URL")


def _create_legacy_core_schema(engine) -> None:
    """Create the bridge-relevant schema from before migration 0001."""

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
            "feedback TEXT NOT NULL, model TEXT NOT NULL, timestamp TIMESTAMPTZ, "
            "user_feedback VARCHAR(16), is_hidden BOOLEAN NOT NULL DEFAULT FALSE, "
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
            "('legacy-feedback', 'chunk-source', 'legacy feedback', 'legacy-model', FALSE)"
        )


@pytest.mark.skipif(
    not POSTGRES_URL,
    reason="set COMPAIR_TEST_POSTGRES_URL to run the real PostgreSQL migration test",
)
def test_postgres_publication_lock_restart_and_transactional_rollback() -> None:
    assert POSTGRES_URL is not None
    schema_name = f"migration_{uuid4().hex}"
    admin_engine = core_db.create_engine(POSTGRES_URL, pool_pre_ping=True)
    assert admin_engine.dialect.name == "postgresql"
    scoped_engine = None
    try:
        with admin_engine.begin() as connection:
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema_name}"')

        scoped_engine = core_db.create_engine(
            POSTGRES_URL,
            pool_pre_ping=True,
            connect_args={"options": f"-csearch_path={schema_name}"},
        )
        from compair_core.compair.retrieval.corpus import (
            ensure_retrieval_corpus_schema,
        )

        # Reproduce a populated pre-0001 database, then apply the real bridge
        # migration rather than relying on create_all's current schema.
        _create_legacy_core_schema(scoped_engine)
        ensure_retrieval_corpus_schema(scoped_engine)
        baseline = run_schema_migrations(scoped_engine)
        assert baseline.applied == (
            "0000_core_schema_baseline",
            "0001_baseline_evidence_bridge_v1",
            "0002_baseline_evidence_retention_v1",
            "0003_baseline_generation_state_v1",
        )
        with scoped_engine.connect() as connection:
            assert connection.execute(
                text(
                    "SELECT reference_id, source_chunk_id, reference_chunk_id, "
                    "reference_document_id, reference_note_id, reference_type, "
                    "baseline_selected_evidence_id FROM reference "
                    "WHERE reference_id = 'legacy-reference'"
                )
            ).one() == (
                "legacy-reference",
                "chunk-source",
                "chunk-peer",
                "doc-peer",
                None,
                "document",
                None,
            )
            assert connection.execute(
                text(
                    "SELECT reference_chunk_id, reference_document_id, "
                    "baseline_selected_evidence_id FROM reference "
                    "WHERE reference_id = 'legacy-document-reference'"
                )
            ).one() == (None, "doc-peer", None)
            constraint_state = dict(
                connection.execute(
                    text(
                        "SELECT conname, convalidated FROM pg_constraint "
                        "WHERE conname IN "
                        "('ck_reference_exactly_one_target', "
                        "'ck_feedback_baseline_finding_pair')"
                    )
                ).all()
            )
            assert constraint_state == {
                "ck_feedback_baseline_finding_pair": True,
                "ck_reference_exactly_one_target": False,
            }

        with scoped_engine.begin() as connection:
            group_id, document_id, chunk_id = _add_scope(connection, "postgres")
            _add_mutable_corpus_lifecycle(connection, "postgres")
            connection.execute(
                baseline_retrieval_run.insert(),
                _run_values(
                    "postgres",
                    group_id=group_id,
                    document_id=document_id,
                    chunk_id=chunk_id,
                ),
            )
            connection.execute(
                baseline_evidence_artifact.insert(),
                _artifact_values(
                    "postgres",
                    group_id=group_id,
                    artifact_key="7" * 64,
                    content="postgres evidence",
                ),
            )
            connection.execute(
                baseline_selected_evidence.insert(),
                _selected_values(
                    "postgres",
                    group_id=group_id,
                    run_id="run-postgres",
                    artifact_id="artifact-postgres",
                    ordinal=1,
                    content="postgres evidence",
                ),
            )
            connection.execute(
                text(
                    "INSERT INTO reference "
                    "(reference_id, source_chunk_id, reference_chunk_id, "
                    "reference_document_id, reference_note_id, reference_type, "
                    "baseline_selected_evidence_id) VALUES "
                    "('baseline-reference', :source, NULL, NULL, NULL, "
                    "'baseline_file', 'selected-postgres')"
                ),
                {"source": chunk_id},
            )
            connection.execute(
                text(
                    "INSERT INTO feedback "
                    "(feedback_id, source_chunk_id, feedback, model, is_hidden, "
                    "baseline_retrieval_run_id, baseline_finding_ordinal) VALUES "
                    "('baseline-feedback', :source, 'finding', 'model', FALSE, "
                    "'run-postgres', 1)"
                ),
                {"source": chunk_id},
            )

        with pytest.raises(IntegrityError), scoped_engine.begin() as connection:
            connection.execute(
                baseline_retrieval_run.insert(),
                _run_values(
                    "postgres-duplicate-intent",
                    group_id=group_id,
                    document_id=document_id,
                    chunk_id=chunk_id,
                ),
            )
        with pytest.raises(IntegrityError), scoped_engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO reference "
                    "(reference_id, source_chunk_id, reference_chunk_id, "
                    "reference_document_id, reference_note_id, reference_type, "
                    "baseline_selected_evidence_id) VALUES "
                    "('invalid-reference', :source, 'chunk-peer', NULL, NULL, "
                    "'document', 'selected-postgres')"
                ),
                {"source": chunk_id},
            )
        with pytest.raises(IntegrityError), scoped_engine.begin() as connection:
            connection.execute(
                baseline_evidence_artifact.delete().where(
                    baseline_evidence_artifact.c.artifact_id
                    == "artifact-postgres"
                )
            )

        with pytest.raises(IntegrityError), scoped_engine.begin() as connection:
            connection.execute(
                baseline_retrieval_run.delete().where(
                    baseline_retrieval_run.c.run_id == "run-postgres"
                )
            )

        # Source Document -> Chunk deletion clears only provenance pointers for
        # baseline audit rows. The chunk trigger still deletes legacy rows.
        with scoped_engine.begin() as connection:
            connection.execute(
                text("DELETE FROM document WHERE document_id=:document_id"),
                {"document_id": document_id},
            )
            connection.execute(text("DELETE FROM chunk WHERE chunk_id='chunk-source'"))
        with scoped_engine.connect() as connection:
            assert connection.execute(
                text(
                    "SELECT source_chunk_id, source_document_id "
                    "FROM baseline_retrieval_run WHERE run_id='run-postgres'"
                )
            ).one() == (None, None)
            assert connection.execute(
                text(
                    "SELECT source_chunk_id FROM reference "
                    "WHERE reference_id='baseline-reference'"
                )
            ).scalar_one() is None
            assert connection.execute(
                text(
                    "SELECT source_chunk_id FROM feedback "
                    "WHERE feedback_id='baseline-feedback'"
                )
            ).scalar_one() is None
            assert connection.execute(
                text("SELECT count(*) FROM reference WHERE reference_id LIKE 'legacy-%'")
            ).scalar_one() == 0
            assert connection.execute(
                text("SELECT count(*) FROM feedback WHERE feedback_id='legacy-feedback'")
            ).scalar_one() == 0

        # Re-ingestion/repository rename and publication/generation deletion
        # cannot reach the immutable copied evidence.
        with scoped_engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO retrieval_corpus_generation "
                    "(generation_id, corpus_id, generation_version, expected_repository_count, "
                    "expected_file_count, status, manifest_hash, created_at, validated_at, activated_at) "
                    "VALUES ('generation-postgres-v2', 'corpus-postgres', 'generation-v2', "
                    "1, 1, 'active', :hash, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {"hash": "8" * 64},
            )
            connection.execute(
                text(
                    "INSERT INTO retrieval_corpus_file "
                    "(file_id, generation_id, repository_id, repository_name, relative_path, "
                    "file_state, content_hash, byte_size, content) VALUES "
                    "('file-postgres-v2', 'generation-postgres-v2', 'repository-postgres', "
                    "'renamed-repo', 'renamed/location.py', 'supported', :hash, 6, 'second')"
                ),
                {"hash": "8" * 64},
            )
            connection.execute(
                text(
                    "UPDATE retrieval_corpus SET active_generation_id='generation-postgres-v2' "
                    "WHERE corpus_id='corpus-postgres'"
                )
            )
            connection.execute(
                text(
                    "DELETE FROM retrieval_baseline_index_publication "
                    "WHERE corpus_id='corpus-postgres'"
                )
            )
            connection.execute(
                text(
                    "DELETE FROM retrieval_corpus_generation "
                    "WHERE generation_id='generation-postgres'"
                )
            )
        scoped_engine.dispose()
        scoped_engine = core_db.create_engine(
            POSTGRES_URL,
            pool_pre_ping=True,
            connect_args={"options": f"-csearch_path={schema_name}"},
        )
        assert run_schema_migrations(scoped_engine).applied == ()
        with scoped_engine.connect() as connection:
            assert connection.execute(
                text(
                    "SELECT repository_name, relative_path, complete_content, "
                    "corpus_generation_id, index_id FROM baseline_evidence_artifact "
                    "WHERE artifact_id='artifact-postgres'"
                )
            ).one() == (
                "repo-postgres",
                "src/postgres.py",
                "postgres evidence",
                "generation-postgres",
                "index-postgres",
            )
            assert connection.execute(
                text("SELECT count(*) FROM reference WHERE reference_id='baseline-reference'")
            ).scalar_one() == 1
            assert connection.execute(
                text("SELECT count(*) FROM feedback WHERE feedback_id='baseline-feedback'")
            ).scalar_one() == 1

        # One scope deletion cascades run, selection, Reference, Feedback, and
        # artifact through the direct group privacy boundary.
        with scoped_engine.begin() as connection:
            connection.execute(
                text('DELETE FROM "group" WHERE group_id = :group_id'),
                {"group_id": group_id},
            )
        with scoped_engine.connect() as connection:
            for table_name in (
                "baseline_retrieval_run",
                "baseline_evidence_artifact",
                "baseline_selected_evidence",
            ):
                assert connection.execute(
                    text(f"SELECT count(*) FROM {table_name}")
                ).scalar_one() == 0
            assert connection.execute(
                text(
                    "SELECT count(*) FROM reference "
                    "WHERE reference_id = 'baseline-reference'"
                )
            ).scalar_one() == 0
            assert connection.execute(
                text(
                    "SELECT count(*) FROM feedback "
                    "WHERE feedback_id = 'baseline-feedback'"
                )
            ).scalar_one() == 0

        def create_relationship(connection) -> None:
            connection.exec_driver_sql(
                "CREATE TABLE migration_evidence ("
                "evidence_id VARCHAR(36) PRIMARY KEY, content_hash VARCHAR(64) NOT NULL)"
            )
            connection.exec_driver_sql(
                "CREATE TABLE migration_owner (owner_id VARCHAR(36) PRIMARY KEY)"
            )
            connection.exec_driver_sql(
                "INSERT INTO migration_owner (owner_id) VALUES ('legacy')"
            )
            connection.exec_driver_sql(
                "ALTER TABLE migration_owner ADD COLUMN evidence_id VARCHAR(36)"
            )
            connection.exec_driver_sql(
                "ALTER TABLE migration_owner ADD CONSTRAINT "
                "fk_migration_owner_evidence FOREIGN KEY (evidence_id) "
                "REFERENCES migration_evidence(evidence_id) NOT VALID"
            )
            connection.exec_driver_sql(
                "ALTER TABLE migration_owner VALIDATE CONSTRAINT "
                "fk_migration_owner_evidence"
            )
            connection.exec_driver_sql(
                "CREATE INDEX ix_migration_owner_evidence_id "
                "ON migration_owner (evidence_id)"
            )

        first = SchemaMigration(
            migration_id="1000_postgres_relationship",
            description="PostgreSQL staged nullable relationship",
            checksum_material=(
                "add table and populated owner, nullable column, NOT VALID FK, "
                "validate, index v2"
            ),
            upgrade=create_relationship,
        )
        registry = (*CORE_SCHEMA_MIGRATIONS, first)
        report = run_schema_migrations(scoped_engine, registry)
        assert report.applied == (first.migration_id,)
        assert migration_registry.inspect(scoped_engine).get_foreign_keys("migration_owner")[0][
            "name"
        ] == "fk_migration_owner_evidence"
        with scoped_engine.connect() as connection:
            legacy_value = connection.exec_driver_sql(
                "SELECT evidence_id FROM migration_owner WHERE owner_id = 'legacy'"
            ).scalar_one()
        assert legacy_value is None

        # A fresh pool sees the durable registry marker.
        scoped_engine.dispose()
        scoped_engine = core_db.create_engine(
            POSTGRES_URL,
            pool_pre_ping=True,
            connect_args={"options": f"-csearch_path={schema_name}"},
        )
        assert run_schema_migrations(scoped_engine, registry).already_applied == (
            "0000_core_schema_baseline",
            "0001_baseline_evidence_bridge_v1",
            "0002_baseline_evidence_retention_v1",
            "0003_baseline_generation_state_v1",
            first.migration_id,
        )

        def fail_after_ddl(connection) -> None:
            connection.exec_driver_sql(
                "CREATE TABLE should_rollback (id INTEGER PRIMARY KEY)"
            )
            raise RuntimeError("deliberate PostgreSQL migration failure")

        second = SchemaMigration(
            migration_id="1001_postgres_failure",
            description="Verify PostgreSQL DDL rollback",
            checksum_material="create should_rollback then fail v1",
            upgrade=fail_after_ddl,
        )
        with pytest.raises(SchemaMigrationError) as error:
            run_schema_migrations(scoped_engine, (*registry, second))
        assert error.value.code == "upgrade_failed"
        assert "should_rollback" not in migration_registry.inspect(
            scoped_engine
        ).get_table_names()
        assert [(row.migration_id, row.state) for row in read_schema_migration_state(scoped_engine)] == [
            ("0000_core_schema_baseline", "applied"),
            ("0001_baseline_evidence_bridge_v1", "applied"),
            ("0002_baseline_evidence_retention_v1", "applied"),
            ("0003_baseline_generation_state_v1", "applied"),
            (first.migration_id, "applied"),
            (second.migration_id, "failed"),
        ]

        # Model the documented reviewed recovery: back up and diagnose first,
        # then clear only the failed marker and rerun the corrected immutable
        # definition. The failed DDL itself was fully rolled back.
        with scoped_engine.begin() as connection:
            connection.execute(
                migration_registry.schema_migration_table.delete().where(
                    migration_registry.schema_migration_table.c.migration_id
                    == second.migration_id
                )
            )

        def corrected_ddl(connection) -> None:
            connection.exec_driver_sql(
                "CREATE TABLE should_rollback (id INTEGER PRIMARY KEY)"
            )

        recovered = replace(second, upgrade=corrected_ddl)
        assert run_schema_migrations(
            scoped_engine, (*registry, recovered)
        ).applied == (second.migration_id,)
        assert "should_rollback" in migration_registry.inspect(
            scoped_engine
        ).get_table_names()
    finally:
        if scoped_engine is not None:
            scoped_engine.dispose()
        with admin_engine.begin() as connection:
            connection.exec_driver_sql(
                f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'
            )
        admin_engine.dispose()
