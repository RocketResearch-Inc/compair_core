"""Real PostgreSQL coverage for the Core migration foundation.

Run against a disposable database with::

    COMPAIR_TEST_POSTGRES_URL=postgresql+psycopg2://user:pass@127.0.0.1:5432/db \
      pytest -q tests/test_schema_migrations_postgres.py
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

from compair_core import db as core_db
from compair_core import schema_migrations as migration_registry
from compair_core.schema_migrations import (
    CORE_SCHEMA_MIGRATIONS,
    SchemaMigration,
    SchemaMigrationError,
    read_schema_migration_state,
    run_schema_migrations,
)

POSTGRES_URL = os.getenv("COMPAIR_TEST_POSTGRES_URL")


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
            connect_args={"options": f"-csearch_path={schema_name},public"},
        )
        from compair_core.compair import models
        from compair_core.compair.retrieval.corpus import (
            ensure_retrieval_corpus_schema,
        )

        # Reproduce the pre-registry startup schema, then recognize it through
        # the real production baseline migration.
        models.Base.metadata.create_all(scoped_engine)
        ensure_retrieval_corpus_schema(scoped_engine)
        baseline = run_schema_migrations(scoped_engine)
        assert baseline.applied == ("0000_core_schema_baseline",)

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
            connect_args={"options": f"-csearch_path={schema_name},public"},
        )
        assert run_schema_migrations(scoped_engine, registry).already_applied == (
            "0000_core_schema_baseline",
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
            (first.migration_id, "applied"),
            (second.migration_id, "failed"),
        ]
    finally:
        if scoped_engine is not None:
            scoped_engine.dispose()
        with admin_engine.begin() as connection:
            connection.exec_driver_sql(
                f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'
            )
        admin_engine.dispose()
