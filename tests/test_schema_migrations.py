from __future__ import annotations

import threading
from pathlib import Path

import pytest

from compair_core import db as core_db
from compair_core import schema_migrations as migration_registry
from compair_core.schema_migrations import (
    CORE_SCHEMA_MIGRATIONS,
    MIGRATION_RUNNER_VERSION,
    SchemaInvariantError,
    SchemaMigration,
    SchemaMigrationError,
    read_schema_migration_state,
    run_schema_migrations,
)


def _sqlite_engine(path: Path):
    return core_db.create_engine(
        f"sqlite:///{path}",
        connect_args={"check_same_thread": False, "timeout": 10},
    )


def _create_current_startup_schema(engine) -> None:
    # Import only when needed: importing the application package intentionally
    # exercises its real startup initialization as well.
    from compair_core.compair import models
    from compair_core.compair.retrieval.corpus import ensure_retrieval_corpus_schema

    models.Base.metadata.create_all(engine)
    ensure_retrieval_corpus_schema(engine)


def _toy_relationship_migration(
    identifier: str = "1000_toy_relationship",
) -> SchemaMigration:
    def upgrade(connection) -> None:
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
            "ALTER TABLE migration_owner ADD COLUMN evidence_id VARCHAR(36) "
            "REFERENCES migration_evidence(evidence_id)"
        )
        connection.exec_driver_sql(
            "CREATE INDEX ix_migration_owner_evidence_id "
            "ON migration_owner (evidence_id)"
        )

    def validate(connection) -> None:
        inspector = migration_registry.inspect(connection)
        if {"migration_evidence", "migration_owner"} - set(inspector.get_table_names()):
            raise SchemaInvariantError("toy_table_missing")
        owner_columns = {
            column["name"] for column in inspector.get_columns("migration_owner")
        }
        if "evidence_id" not in owner_columns:
            raise SchemaInvariantError("toy_column_missing")
        indexes = {index["name"] for index in inspector.get_indexes("migration_owner")}
        if "ix_migration_owner_evidence_id" not in indexes:
            raise SchemaInvariantError("toy_index_missing")

    return SchemaMigration(
        migration_id=identifier,
        description="Create a toy additive evidence relationship",
        checksum_material=(
            "create migration_evidence and populated migration_owner; add nullable "
            "inline foreign key and index; toy-v2"
        ),
        upgrade=upgrade,
        validate=validate,
    )


def test_existing_startup_schema_gets_transactional_baseline_marker(
    tmp_path: Path,
) -> None:
    engine = _sqlite_engine(tmp_path / "existing-core.db")
    try:
        _create_current_startup_schema(engine)

        staging_only = run_schema_migrations(engine, CORE_SCHEMA_MIGRATIONS[:9])
        continuation = run_schema_migrations(engine)
        second = run_schema_migrations(engine)
        state = read_schema_migration_state(engine)

        assert staging_only.applied == (
            "0000_core_schema_baseline",
            "0001_baseline_evidence_bridge_v1",
            "0002_baseline_evidence_retention_v1",
            "0003_baseline_generation_state_v1",
            "0004_baseline_notification_outbox_v1",
            "0005_baseline_control_plane_staging_v1",
            "0006_baseline_control_plane_continuation_v1",
            "0007_baseline_control_plane_ingestion_worker_v1",
            "0008_baseline_compatible_index_job_v1",
        )
        assert continuation.applied == (
            "0009_baseline_run_job_v1",
            "0010_baseline_document_source_scope_v1",
            "0011_baseline_run_executor_v1",
            "0012_baseline_control_generation_v1",
            "0013_baseline_database_worker_v1",
            "0014_baseline_worker_runtime_attestation_v1",
        )
        assert staging_only.already_applied == ()
        assert continuation.already_applied == staging_only.applied
        assert second.applied == ()
        assert second.already_applied == (
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
            "0014_baseline_worker_runtime_attestation_v1",
        )
        assert [(row.migration_id, row.state) for row in state] == [
            ("0000_core_schema_baseline", "applied"),
            ("0001_baseline_evidence_bridge_v1", "applied"),
            ("0002_baseline_evidence_retention_v1", "applied"),
            ("0003_baseline_generation_state_v1", "applied"),
            ("0004_baseline_notification_outbox_v1", "applied"),
            ("0005_baseline_control_plane_staging_v1", "applied"),
            ("0006_baseline_control_plane_continuation_v1", "applied"),
            ("0007_baseline_control_plane_ingestion_worker_v1", "applied"),
            ("0008_baseline_compatible_index_job_v1", "applied"),
            ("0009_baseline_run_job_v1", "applied"),
            ("0010_baseline_document_source_scope_v1", "applied"),
            ("0011_baseline_run_executor_v1", "applied"),
            ("0012_baseline_control_generation_v1", "applied"),
            ("0013_baseline_database_worker_v1", "applied"),
            ("0014_baseline_worker_runtime_attestation_v1", "applied"),
        ]
        assert [row.checksum for row in state] == [
            migration.checksum for migration in CORE_SCHEMA_MIGRATIONS
        ]
        assert all(row.runner_version == MIGRATION_RUNNER_VERSION for row in state)
        assert all(row.error_code is None for row in state)
    finally:
        engine.dispose()


def test_sqlite_additive_table_nullable_foreign_key_and_index(
    tmp_path: Path,
) -> None:
    engine = _sqlite_engine(tmp_path / "additive.db")
    migration = _toy_relationship_migration()
    try:
        report = run_schema_migrations(engine, (migration,))
        assert report.applied == (migration.migration_id,)

        inspector = migration_registry.inspect(engine)
        foreign_keys = inspector.get_foreign_keys("migration_owner")
        assert foreign_keys[0]["referred_table"] == "migration_evidence"
        assert foreign_keys[0]["constrained_columns"] == ["evidence_id"]
        with engine.connect() as connection:
            legacy_value = connection.exec_driver_sql(
                "SELECT evidence_id FROM migration_owner WHERE owner_id = 'legacy'"
            ).scalar_one()
        assert legacy_value is None

        with engine.begin() as connection:
            connection.exec_driver_sql(
                "INSERT INTO migration_owner (owner_id, evidence_id) VALUES ('empty', NULL)"
            )
    finally:
        engine.dispose()


def test_failed_sqlite_batch_rolls_back_all_pending_ddl_and_records_failure(
    tmp_path: Path,
) -> None:
    engine = _sqlite_engine(tmp_path / "rollback.db")

    first = SchemaMigration(
        migration_id="1000_first_pending",
        description="First pending DDL",
        checksum_material="create first_pending v1",
        upgrade=lambda connection: connection.exec_driver_sql(
            "CREATE TABLE first_pending (id INTEGER PRIMARY KEY)"
        ),
    )

    def fail_after_ddl(connection) -> None:
        connection.exec_driver_sql(
            "CREATE TABLE second_pending (id INTEGER PRIMARY KEY)"
        )
        raise RuntimeError("deliberate migration failure")

    second = SchemaMigration(
        migration_id="1001_second_pending",
        description="Fail after DDL",
        checksum_material="create second_pending then fail v1",
        upgrade=fail_after_ddl,
    )
    try:
        with pytest.raises(SchemaMigrationError) as error:
            run_schema_migrations(engine, (first, second))

        assert error.value.migration_id == second.migration_id
        assert error.value.code == "upgrade_failed"
        assert (
            "first_pending" not in migration_registry.inspect(engine).get_table_names()
        )
        assert (
            "second_pending" not in migration_registry.inspect(engine).get_table_names()
        )
        state = read_schema_migration_state(engine)
        assert [(row.migration_id, row.state, row.error_code) for row in state] == [
            (second.migration_id, "failed", "upgrade_failed")
        ]

        with pytest.raises(SchemaMigrationError) as retry_error:
            run_schema_migrations(engine, (first, second))
        assert retry_error.value.code == "previous_failure"
    finally:
        engine.dispose()


def test_partial_legacy_schema_is_rejected_without_application_ddl(
    tmp_path: Path,
) -> None:
    engine = _sqlite_engine(tmp_path / "partial.db")
    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                'CREATE TABLE "user" (user_id VARCHAR(36) PRIMARY KEY)'
            )

        with pytest.raises(SchemaMigrationError) as error:
            run_schema_migrations(engine)

        assert error.value.code == "missing_table:document"
        application_tables = set(
            migration_registry.inspect(engine).get_table_names()
        ) - {
            "core_schema_migration",
            "user",
        }
        assert application_tables == set()
        state = read_schema_migration_state(engine)
        assert state[0].state == "failed"
        assert state[0].error_code == "missing_table:document"
    finally:
        engine.dispose()


def test_applied_checksum_drift_fails_closed(tmp_path: Path) -> None:
    engine = _sqlite_engine(tmp_path / "checksum.db")
    original = SchemaMigration(
        migration_id="1000_checksum",
        description="Stable migration",
        checksum_material="stable-v1",
        upgrade=lambda _connection: None,
    )
    changed = SchemaMigration(
        migration_id=original.migration_id,
        description=original.description,
        checksum_material="mutated-v2",
        upgrade=lambda _connection: None,
    )
    try:
        run_schema_migrations(engine, (original,))
        with pytest.raises(SchemaMigrationError) as error:
            run_schema_migrations(engine, (changed,))
        assert error.value.code == "checksum_mismatch"
        assert read_schema_migration_state(engine)[0].checksum == original.checksum
    finally:
        engine.dispose()


def test_sqlite_lock_serializes_concurrent_runners(tmp_path: Path) -> None:
    database_path = tmp_path / "concurrent.db"
    counter = 0
    counter_lock = threading.Lock()
    barrier = threading.Barrier(2)
    errors: list[Exception] = []
    reports = []

    def upgrade(connection) -> None:
        nonlocal counter
        connection.exec_driver_sql("CREATE TABLE once_only (id INTEGER PRIMARY KEY)")
        with counter_lock:
            counter += 1

    migration = SchemaMigration(
        migration_id="1000_concurrent",
        description="Run once under concurrent startup",
        checksum_material="create once_only v1",
        upgrade=upgrade,
    )

    def runner() -> None:
        engine = _sqlite_engine(database_path)
        try:
            barrier.wait(timeout=5)
            reports.append(run_schema_migrations(engine, (migration,)))
        except Exception as exc:  # noqa: BLE001 - transfer worker failures for assertion
            errors.append(exc)
        finally:
            engine.dispose()

    threads = [threading.Thread(target=runner), threading.Thread(target=runner)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert counter == 1
    assert sorted(report.applied for report in reports) == [
        (),
        (migration.migration_id,),
    ]


def test_initialize_database_propagates_registry_failure(monkeypatch) -> None:
    import compair_core.compair as application

    monkeypatch.setattr(
        application.models.Base.metadata, "create_all", lambda _engine: None
    )
    monkeypatch.setattr(application, "_ensure_pgvector_extension", lambda: None)
    monkeypatch.setattr(application, "_ensure_retrieval_corpus_tables", lambda: None)
    monkeypatch.setattr(application, "_ensure_user_retrial_count_default", lambda: None)
    monkeypatch.setattr(
        application,
        "_ensure_notification_preferences_delivery_columns",
        lambda: None,
    )
    monkeypatch.setattr(application, "_ensure_reference_chunk_id_column", lambda: None)
    monkeypatch.setattr(
        application,
        "_ensure_notification_event_fingerprint_columns",
        lambda: None,
    )
    monkeypatch.setattr(application, "_ensure_topic_tags_column", lambda: None)

    def fail_startup(_engine) -> None:
        raise SchemaMigrationError("1000_future", "upgrade_failed")

    monkeypatch.setattr(application, "run_schema_migrations", fail_startup)
    with pytest.raises(SchemaMigrationError, match="1000_future"):
        application.initialize_database()
