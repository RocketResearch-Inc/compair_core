"""Forward-only, transactional schema migrations for Core.

This module intentionally has no dependency on ``compair_core.compair`` so it
can be imported and tested without triggering application startup.  The first
registered migration records the schema produced by the pre-registry startup
path; future additive migrations can build on that checked baseline.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    Engine,
    MetaData,
    String,
    Table,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Connection
from sqlalchemy.schema import CreateIndex, CreateTable

from compair_core.baseline_evidence_schema import (
    BASELINE_EVIDENCE_TABLES,
    BASELINE_RETRIEVAL_RUN_TABLE,
    BASELINE_SELECTED_EVIDENCE_TABLE,
)

MIGRATION_RUNNER_VERSION = "core-forward-v1"
MIGRATION_TABLE_NAME = "core_schema_migration"
SUPPORTED_DIALECTS = frozenset({"sqlite", "postgresql"})

_metadata = MetaData()
schema_migration_table = Table(
    MIGRATION_TABLE_NAME,
    _metadata,
    Column("migration_id", String(128), primary_key=True),
    Column("checksum", String(64), nullable=False),
    Column("state", String(16), nullable=False),
    Column("runner_version", String(32), nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True), nullable=False),
    Column("error_code", String(128), nullable=True),
    CheckConstraint(
        "state IN ('applied', 'failed')",
        name="ck_core_schema_migration_state",
    ),
)


class SchemaInvariantError(RuntimeError):
    """A stable, non-sensitive schema validation failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class SchemaMigrationError(RuntimeError):
    """A migration failure safe to surface during application startup."""

    def __init__(self, migration_id: str | None, code: str) -> None:
        self.migration_id = migration_id
        self.code = code
        target = migration_id or "registry"
        super().__init__(f"schema migration {target} failed: {code}")


MigrationOperation = Callable[[Connection], None]


@dataclass(frozen=True, slots=True)
class SchemaMigration:
    """One immutable forward migration definition.

    ``checksum_material`` is an explicit, reviewable description of the DDL
    and validation contract.  Changing it after publication is detected as
    drift; a correction must be a new migration.
    """

    migration_id: str
    description: str
    checksum_material: str
    upgrade: MigrationOperation
    validate: MigrationOperation | None = None

    @property
    def checksum(self) -> str:
        payload = json.dumps(
            {
                "migration_id": self.migration_id,
                "description": self.description,
                "checksum_material": self.checksum_material,
                "runner_version": MIGRATION_RUNNER_VERSION,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MigrationReport:
    applied: tuple[str, ...]
    already_applied: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MigrationState:
    migration_id: str
    checksum: str
    state: str
    runner_version: str
    error_code: str | None


class _ExecutionFailure(Exception):
    def __init__(self, migration_id: str, code: str, cause: Exception) -> None:
        self.migration_id = migration_id
        self.code = code
        self.cause = cause
        super().__init__(migration_id, code)


# These are the bridge-relevant invariants already established by the current
# create/ensure startup path.  The baseline marker does not create or alter any
# application table.
_CURRENT_CORE_COLUMNS: dict[str, frozenset[str]] = {
    "user": frozenset({"user_id"}),
    "document": frozenset({"document_id", "user_id", "content"}),
    "chunk": frozenset(
        {"chunk_id", "hash", "content", "document_id", "note_id", "chunk_type"}
    ),
    "reference": frozenset(
        {
            "reference_id",
            "source_chunk_id",
            "reference_chunk_id",
            "reference_document_id",
            "reference_note_id",
            "reference_type",
        }
    ),
    "feedback": frozenset({"feedback_id", "source_chunk_id", "feedback", "model"}),
    "retrieval_corpus": frozenset(
        {"corpus_id", "scope_key", "source_document_id", "changed_repository_id"}
    ),
    "retrieval_corpus_generation": frozenset(
        {"generation_id", "corpus_id", "status", "manifest_hash"}
    ),
    "retrieval_corpus_file": frozenset(
        {"file_id", "generation_id", "repository_id", "relative_path", "content_hash"}
    ),
    "retrieval_baseline_index_build": frozenset(
        {"index_id", "generation_id", "status", "corpus_manifest_hash"}
    ),
    "retrieval_baseline_index_document": frozenset(
        {
            "index_document_id",
            "index_id",
            "corpus_file_id",
            "indexed_document_hash",
        }
    ),
    "retrieval_baseline_index_publication": frozenset(
        {"corpus_id", "index_id", "published_at"}
    ),
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _no_op(_connection: Connection) -> None:
    return None


def _validate_current_core_schema(connection: Connection) -> None:
    inspector = inspect(connection)
    tables = set(inspector.get_table_names())
    for table_name, required_columns in _CURRENT_CORE_COLUMNS.items():
        if table_name not in tables:
            raise SchemaInvariantError(f"missing_table:{table_name}")
        columns = {column["name"] for column in inspector.get_columns(table_name)}
        missing = sorted(required_columns - columns)
        if missing:
            raise SchemaInvariantError(f"missing_column:{table_name}:{missing[0]}")


_REFERENCE_TARGET_PREDICATE = (
    "(baseline_selected_evidence_id IS NULL AND reference_chunk_id IS NOT NULL) "
    "OR (baseline_selected_evidence_id IS NOT NULL "
    "AND reference_chunk_id IS NULL "
    "AND reference_document_id IS NULL "
    "AND reference_note_id IS NULL "
    "AND reference_type = 'baseline_file')"
)

_FEEDBACK_BASELINE_PREDICATE = (
    "(baseline_retrieval_run_id IS NULL AND baseline_finding_ordinal IS NULL) "
    "OR (baseline_retrieval_run_id IS NOT NULL AND baseline_finding_ordinal > 0)"
)

_SQLITE_REFERENCE_TARGET_PREDICATE = (
    "(NEW.baseline_selected_evidence_id IS NULL "
    "AND NEW.reference_chunk_id IS NOT NULL) "
    "OR (NEW.baseline_selected_evidence_id IS NOT NULL "
    "AND NEW.reference_chunk_id IS NULL "
    "AND NEW.reference_document_id IS NULL "
    "AND NEW.reference_note_id IS NULL "
    "AND NEW.reference_type = 'baseline_file')"
)

_SQLITE_FEEDBACK_BASELINE_PREDICATE = (
    "(NEW.baseline_retrieval_run_id IS NULL "
    "AND NEW.baseline_finding_ordinal IS NULL) "
    "OR (NEW.baseline_retrieval_run_id IS NOT NULL "
    "AND NEW.baseline_finding_ordinal > 0)"
)

_REFERENCE_INSERT_TRIGGER = "trg_reference_baseline_target_insert"
_REFERENCE_UPDATE_TRIGGER = "trg_reference_baseline_target_update"
_FEEDBACK_INSERT_TRIGGER = "trg_feedback_baseline_pair_insert"
_FEEDBACK_UPDATE_TRIGGER = "trg_feedback_baseline_pair_update"


def _column_names(connection: Connection, table_name: str) -> set[str]:
    return {column["name"] for column in inspect(connection).get_columns(table_name)}


def _constraint_names(
    connection: Connection,
    table_name: str,
    kind: str,
) -> set[str]:
    inspector = inspect(connection)
    if kind == "foreign_key":
        rows = inspector.get_foreign_keys(table_name)
    elif kind == "check":
        rows = inspector.get_check_constraints(table_name)
    else:  # pragma: no cover - private caller contract
        raise ValueError(kind)
    return {str(row["name"]) for row in rows if row.get("name")}


def _postgres_add_constraint(
    connection: Connection,
    *,
    table_name: str,
    constraint_name: str,
    definition: str,
    kind: str,
    validate: bool = True,
) -> None:
    if constraint_name not in _constraint_names(connection, table_name, kind):
        connection.exec_driver_sql(
            f"ALTER TABLE {table_name} ADD CONSTRAINT {constraint_name} "
            f"{definition} NOT VALID"
        )
    if validate:
        connection.exec_driver_sql(
            f"ALTER TABLE {table_name} VALIDATE CONSTRAINT {constraint_name}"
        )


def _create_baseline_evidence_tables(connection: Connection) -> None:
    for table in BASELINE_EVIDENCE_TABLES:
        connection.execute(CreateTable(table, if_not_exists=True))
        for index in sorted(table.indexes, key=lambda candidate: candidate.name or ""):
            connection.execute(CreateIndex(index, if_not_exists=True))


def _upgrade_baseline_evidence_bridge(connection: Connection) -> None:
    _create_baseline_evidence_tables(connection)
    reference_columns = _column_names(connection, "reference")
    feedback_columns = _column_names(connection, "feedback")

    if connection.dialect.name == "sqlite":
        if "baseline_selected_evidence_id" not in reference_columns:
            connection.exec_driver_sql(
                "ALTER TABLE reference ADD COLUMN baseline_selected_evidence_id "
                "VARCHAR(36) REFERENCES baseline_selected_evidence(selected_evidence_id) "
                "ON DELETE CASCADE"
            )
        if "baseline_retrieval_run_id" not in feedback_columns:
            connection.exec_driver_sql(
                "ALTER TABLE feedback ADD COLUMN baseline_retrieval_run_id "
                "VARCHAR(36) REFERENCES baseline_retrieval_run(run_id) ON DELETE CASCADE"
            )
        if "baseline_finding_ordinal" not in feedback_columns:
            connection.exec_driver_sql(
                "ALTER TABLE feedback ADD COLUMN baseline_finding_ordinal INTEGER"
            )
        connection.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "uq_reference_baseline_selected_evidence "
            "ON reference (baseline_selected_evidence_id) "
            "WHERE baseline_selected_evidence_id IS NOT NULL"
        )
        connection.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_feedback_baseline_run_finding "
            "ON feedback (baseline_retrieval_run_id, baseline_finding_ordinal) "
            "WHERE baseline_retrieval_run_id IS NOT NULL"
        )
        connection.exec_driver_sql(
            f"CREATE TRIGGER IF NOT EXISTS {_REFERENCE_INSERT_TRIGGER} "
            "BEFORE INSERT ON reference "
            f"WHEN ({_SQLITE_REFERENCE_TARGET_PREDICATE}) IS NOT TRUE "
            "BEGIN SELECT RAISE(ABORT, 'reference_target_invalid'); END"
        )
        connection.exec_driver_sql(
            f"CREATE TRIGGER IF NOT EXISTS {_REFERENCE_UPDATE_TRIGGER} "
            "BEFORE UPDATE OF reference_chunk_id, reference_document_id, "
            "reference_note_id, reference_type, baseline_selected_evidence_id "
            "ON reference "
            f"WHEN ({_SQLITE_REFERENCE_TARGET_PREDICATE}) IS NOT TRUE "
            "BEGIN SELECT RAISE(ABORT, 'reference_target_invalid'); END"
        )
        connection.exec_driver_sql(
            f"CREATE TRIGGER IF NOT EXISTS {_FEEDBACK_INSERT_TRIGGER} "
            "BEFORE INSERT ON feedback "
            f"WHEN ({_SQLITE_FEEDBACK_BASELINE_PREDICATE}) IS NOT TRUE "
            "BEGIN SELECT RAISE(ABORT, 'feedback_baseline_pair_invalid'); END"
        )
        connection.exec_driver_sql(
            f"CREATE TRIGGER IF NOT EXISTS {_FEEDBACK_UPDATE_TRIGGER} "
            "BEFORE UPDATE OF baseline_retrieval_run_id, baseline_finding_ordinal "
            "ON feedback "
            f"WHEN ({_SQLITE_FEEDBACK_BASELINE_PREDICATE}) IS NOT TRUE "
            "BEGIN SELECT RAISE(ABORT, 'feedback_baseline_pair_invalid'); END"
        )
        return

    if "baseline_selected_evidence_id" not in reference_columns:
        connection.exec_driver_sql(
            "ALTER TABLE reference ADD COLUMN baseline_selected_evidence_id VARCHAR(36)"
        )
    if "baseline_retrieval_run_id" not in feedback_columns:
        connection.exec_driver_sql(
            "ALTER TABLE feedback ADD COLUMN baseline_retrieval_run_id VARCHAR(36)"
        )
    if "baseline_finding_ordinal" not in feedback_columns:
        connection.exec_driver_sql(
            "ALTER TABLE feedback ADD COLUMN baseline_finding_ordinal INTEGER"
        )

    _postgres_add_constraint(
        connection,
        table_name="reference",
        constraint_name="fk_reference_baseline_selected_evidence",
        definition=(
            "FOREIGN KEY (baseline_selected_evidence_id) "
            "REFERENCES baseline_selected_evidence(selected_evidence_id) "
            "ON DELETE CASCADE"
        ),
        kind="foreign_key",
    )
    _postgres_add_constraint(
        connection,
        table_name="reference",
        constraint_name="ck_reference_exactly_one_target",
        definition=f"CHECK ({_REFERENCE_TARGET_PREDICATE})",
        kind="check",
        # Historical rows can predate reference_chunk_id and remain valid
        # audit data. PostgreSQL NOT VALID checks still enforce all new and
        # updated rows without rewriting or rejecting that history.
        validate=False,
    )
    _postgres_add_constraint(
        connection,
        table_name="feedback",
        constraint_name="fk_feedback_baseline_retrieval_run",
        definition=(
            "FOREIGN KEY (baseline_retrieval_run_id) "
            "REFERENCES baseline_retrieval_run(run_id) ON DELETE CASCADE"
        ),
        kind="foreign_key",
    )
    _postgres_add_constraint(
        connection,
        table_name="feedback",
        constraint_name="ck_feedback_baseline_finding_pair",
        definition=f"CHECK ({_FEEDBACK_BASELINE_PREDICATE})",
        kind="check",
    )
    connection.exec_driver_sql(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_reference_baseline_selected_evidence "
        "ON reference (baseline_selected_evidence_id) "
        "WHERE baseline_selected_evidence_id IS NOT NULL"
    )
    connection.exec_driver_sql(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_feedback_baseline_run_finding "
        "ON feedback (baseline_retrieval_run_id, baseline_finding_ordinal) "
        "WHERE baseline_retrieval_run_id IS NOT NULL"
    )


def _foreign_key_targets(connection: Connection, table_name: str) -> set[tuple]:
    if connection.dialect.name == "sqlite":
        # SQLAlchemy's SQLite inspector omits ``ondelete`` for a foreign key
        # introduced by ``ALTER TABLE ... ADD COLUMN ... REFERENCES`` even
        # though SQLite records and enforces it.  The pragma is the backend's
        # authoritative representation and also preserves composite ordering.
        rows = connection.exec_driver_sql(
            f'PRAGMA foreign_key_list("{table_name}")'
        ).mappings()
        grouped: dict[int, list[dict[str, object]]] = {}
        for row in rows:
            grouped.setdefault(int(row["id"]), []).append(dict(row))
        return {
            (
                tuple(str(row["from"]) for row in sorted(parts, key=lambda item: int(item["seq"]))),
                str(parts[0]["table"]),
                tuple(str(row["to"]) for row in sorted(parts, key=lambda item: int(item["seq"]))),
                str(parts[0]["on_delete"] or "NO ACTION").upper(),
            )
            for parts in grouped.values()
        }
    return {
        (
            tuple(row.get("constrained_columns") or ()),
            str(row.get("referred_table") or ""),
            tuple(row.get("referred_columns") or ()),
            str((row.get("options") or {}).get("ondelete") or "NO ACTION").upper(),
        )
        for row in inspect(connection).get_foreign_keys(table_name)
    }


def _index_names(connection: Connection, table_name: str) -> set[str]:
    return {
        str(row["name"])
        for row in inspect(connection).get_indexes(table_name)
        if row.get("name")
    }


def _unique_constraint_names(connection: Connection, table_name: str) -> set[str]:
    return {
        str(row["name"])
        for row in inspect(connection).get_unique_constraints(table_name)
        if row.get("name")
    }


def _validate_baseline_evidence_bridge(connection: Connection) -> None:
    inspector = inspect(connection)
    tables = set(inspector.get_table_names())
    for table in BASELINE_EVIDENCE_TABLES:
        if table.name not in tables:
            raise SchemaInvariantError(f"missing_table:{table.name}")
        expected = {column.name for column in table.columns}
        actual = _column_names(connection, table.name)
        missing = sorted(expected - actual)
        if missing:
            raise SchemaInvariantError(f"missing_column:{table.name}:{missing[0]}")

    for table_name in (BASELINE_RETRIEVAL_RUN_TABLE, "baseline_evidence_artifact"):
        columns = _column_names(connection, table_name)
        if "source_document_id" not in columns or "document_id" in columns:
            raise SchemaInvariantError(f"ambiguous_source_document:{table_name}")
        if {"retrieval_query", "query_text", "raw_query"} & columns:
            raise SchemaInvariantError(f"raw_query_column:{table_name}")

    reference_columns = _column_names(connection, "reference")
    feedback_columns = _column_names(connection, "feedback")
    if "baseline_selected_evidence_id" not in reference_columns:
        raise SchemaInvariantError("missing_column:reference:baseline_selected_evidence_id")
    if not {"baseline_retrieval_run_id", "baseline_finding_ordinal"} <= feedback_columns:
        raise SchemaInvariantError("missing_column:feedback:baseline_retrieval_run_id")

    selected_fks = _foreign_key_targets(connection, BASELINE_SELECTED_EVIDENCE_TABLE)
    required_selected_fks = {
        (
            ("run_id", "group_id"),
            BASELINE_RETRIEVAL_RUN_TABLE,
            ("run_id", "group_id"),
            "CASCADE",
        ),
        (
            ("artifact_id", "group_id"),
            "baseline_evidence_artifact",
            ("artifact_id", "group_id"),
            "NO ACTION",
        ),
    }
    if not required_selected_fks <= selected_fks:
        raise SchemaInvariantError("selected_scope_foreign_key_invalid")

    run_fks = _foreign_key_targets(connection, BASELINE_RETRIEVAL_RUN_TABLE)
    required_run_fks = {
        (("group_id",), "group", ("group_id",), "CASCADE"),
        (("source_chunk_id",), "chunk", ("chunk_id",), "CASCADE"),
        (("source_document_id",), "document", ("document_id",), "CASCADE"),
    }
    if not required_run_fks <= run_fks:
        raise SchemaInvariantError("run_lifecycle_foreign_key_invalid")

    artifact_fks = _foreign_key_targets(connection, "baseline_evidence_artifact")
    required_artifact_fks = {
        (("group_id",), "group", ("group_id",), "CASCADE"),
        (("source_document_id",), "document", ("document_id",), "SET NULL"),
    }
    if not required_artifact_fks <= artifact_fks:
        raise SchemaInvariantError("artifact_lifecycle_foreign_key_invalid")

    reference_fks = _foreign_key_targets(connection, "reference")
    if (
        ("baseline_selected_evidence_id",),
        BASELINE_SELECTED_EVIDENCE_TABLE,
        ("selected_evidence_id",),
        "CASCADE",
    ) not in reference_fks:
        raise SchemaInvariantError("reference_baseline_foreign_key_invalid")
    feedback_fks = _foreign_key_targets(connection, "feedback")
    if (
        ("baseline_retrieval_run_id",),
        BASELINE_RETRIEVAL_RUN_TABLE,
        ("run_id",),
        "CASCADE",
    ) not in feedback_fks:
        raise SchemaInvariantError("feedback_baseline_foreign_key_invalid")

    required_indexes = {
        "reference": {"uq_reference_baseline_selected_evidence"},
        "feedback": {"uq_feedback_baseline_run_finding"},
    }
    required_indexes.update(
        {
            table.name: {
                str(index.name) for index in table.indexes if index.name is not None
            }
            for table in BASELINE_EVIDENCE_TABLES
        }
    )
    for table_name, expected_indexes in required_indexes.items():
        if not expected_indexes <= _index_names(connection, table_name):
            raise SchemaInvariantError(f"missing_index:{table_name}")

    for table_name, expected_unique in {
        BASELINE_RETRIEVAL_RUN_TABLE: {
            "uq_bl_run_group_intent",
            "uq_bl_run_id_group",
        },
        "baseline_evidence_artifact": {
            "uq_bl_artifact_group_key",
            "uq_bl_artifact_id_group",
        },
        BASELINE_SELECTED_EVIDENCE_TABLE: {
            "uq_bl_selected_run_ordinal",
            "uq_bl_selected_run_artifact",
            "uq_bl_selected_run_content",
        },
    }.items():
        if not expected_unique <= _unique_constraint_names(connection, table_name):
            raise SchemaInvariantError(f"missing_unique_constraint:{table_name}")

    for table in BASELINE_EVIDENCE_TABLES:
        expected_checks = {
            str(constraint.name)
            for constraint in table.constraints
            if isinstance(constraint, CheckConstraint) and constraint.name is not None
        }
        if not expected_checks <= _constraint_names(connection, table.name, "check"):
            raise SchemaInvariantError(f"missing_check_constraint:{table.name}")

    if connection.dialect.name == "sqlite":
        if connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() != 1:
            raise SchemaInvariantError("sqlite_foreign_keys_disabled")
        trigger_names = {
            str(row[0])
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).all()
        }
        required_triggers = {
            _REFERENCE_INSERT_TRIGGER,
            _REFERENCE_UPDATE_TRIGGER,
            _FEEDBACK_INSERT_TRIGGER,
            _FEEDBACK_UPDATE_TRIGGER,
        }
        if not required_triggers <= trigger_names:
            raise SchemaInvariantError("sqlite_bridge_trigger_missing")
    else:
        if "ck_reference_exactly_one_target" not in _constraint_names(
            connection, "reference", "check"
        ):
            raise SchemaInvariantError("reference_target_check_missing")
        if "ck_feedback_baseline_finding_pair" not in _constraint_names(
            connection, "feedback", "check"
        ):
            raise SchemaInvariantError("feedback_pair_check_missing")


CORE_SCHEMA_MIGRATIONS: tuple[SchemaMigration, ...] = (
    SchemaMigration(
        migration_id="0000_core_schema_baseline",
        description="Recognize the bridge-ready schema produced by the legacy startup path",
        checksum_material=(
            "no-ddl; validate required Core document/chunk/reference/feedback and "
            "persistent retrieval corpus/index tables; baseline-contract-v1"
        ),
        upgrade=_no_op,
        validate=_validate_current_core_schema,
    ),
    SchemaMigration(
        migration_id="0001_baseline_evidence_bridge_v1",
        description="Create immutable group-scoped baseline evidence schema",
        checksum_material=(
            "baseline-reference-bridge.v1; create baseline_retrieval_run, "
            "baseline_evidence_artifact, baseline_selected_evidence; group-scoped "
            "idempotency and artifact keys; immutable corpus/index/query provenance; "
            "versioned exact renderer output; selected ordinal 1..4; add exclusive "
            "Reference baseline target and idempotent Feedback run ordinal; SQLite "
            "FK/cascade triggers; PostgreSQL validated nullable FKs and Feedback "
            "pair plus NOT VALID new-write Reference target check; ddl-v1"
        ),
        upgrade=_upgrade_baseline_evidence_bridge,
        validate=_validate_baseline_evidence_bridge,
    ),
)


def _validate_registry(migrations: Sequence[SchemaMigration]) -> tuple[SchemaMigration, ...]:
    ordered = tuple(migrations)
    identifiers = [migration.migration_id for migration in ordered]
    if identifiers != sorted(identifiers):
        raise SchemaMigrationError(None, "migration_order_invalid")
    if len(identifiers) != len(set(identifiers)):
        raise SchemaMigrationError(None, "migration_id_duplicate")
    for identifier in identifiers:
        if not identifier or len(identifier) > 128:
            raise SchemaMigrationError(None, "migration_id_invalid")
    return ordered


def _require_supported_dialect(engine: Engine) -> str:
    dialect = engine.dialect.name
    if dialect not in SUPPORTED_DIALECTS:
        raise SchemaMigrationError(None, f"unsupported_dialect:{dialect}")
    return dialect


def _bootstrap_migration_table(engine: Engine, dialect: str) -> None:
    """Create the registry under a backend-appropriate bootstrap lock."""

    with engine.connect() as connection:
        if dialect == "sqlite":
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                connection.execute(CreateTable(schema_migration_table, if_not_exists=True))
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            return

        with connection.begin():
            # The advisory lock covers the moment before the registry table
            # exists.  The regular table lock is used for all later work.
            connection.execute(text("SELECT pg_advisory_xact_lock(291682902411)"))
            connection.execute(CreateTable(schema_migration_table, if_not_exists=True))


@contextmanager
def _locked_migration_connection(engine: Engine, dialect: str) -> Iterator[Connection]:
    connection = engine.connect()
    try:
        if dialect == "sqlite":
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            return

        transaction = connection.begin()
        try:
            connection.execute(text(f"LOCK TABLE {MIGRATION_TABLE_NAME} IN EXCLUSIVE MODE"))
            yield connection
            transaction.commit()
        except BaseException:
            transaction.rollback()
            raise
    finally:
        connection.close()


def _rows_by_id(connection: Connection) -> dict[str, dict[str, object]]:
    rows = connection.execute(select(schema_migration_table)).mappings()
    return {str(row["migration_id"]): dict(row) for row in rows}


def _failure_code(exc: Exception, phase: str) -> str:
    if isinstance(exc, SchemaInvariantError):
        return exc.code
    return f"{phase}_failed"


def _record_failure(
    engine: Engine,
    dialect: str,
    *,
    migration: SchemaMigration,
    started_at: datetime,
    error_code: str,
) -> None:
    """Record a sanitized failure after the migration transaction rolls back."""

    with _locked_migration_connection(engine, dialect) as connection:
        existing = connection.execute(
            select(schema_migration_table).where(
                schema_migration_table.c.migration_id == migration.migration_id
            )
        ).mappings().one_or_none()
        if existing is not None and existing["state"] == "applied":
            return
        values = {
            "checksum": migration.checksum,
            "state": "failed",
            "runner_version": MIGRATION_RUNNER_VERSION,
            "started_at": started_at,
            "finished_at": _utcnow(),
            "error_code": error_code,
        }
        if existing is None:
            connection.execute(
                schema_migration_table.insert().values(
                    migration_id=migration.migration_id,
                    **values,
                )
            )
        else:
            connection.execute(
                schema_migration_table.update()
                .where(schema_migration_table.c.migration_id == migration.migration_id)
                .values(**values)
            )


def run_schema_migrations(
    engine: Engine,
    migrations: Sequence[SchemaMigration] = CORE_SCHEMA_MIGRATIONS,
) -> MigrationReport:
    """Apply one ordered migration batch atomically and fail closed on drift.

    All pending migrations share one transaction.  On an execution or
    validation failure the batch is rolled back, then only a sanitized failed
    registry row is committed in a separate transaction.
    """

    ordered = _validate_registry(migrations)
    dialect = _require_supported_dialect(engine)
    _bootstrap_migration_table(engine, dialect)

    applied: list[str] = []
    already_applied: list[str] = []
    try:
        with _locked_migration_connection(engine, dialect) as connection:
            existing = _rows_by_id(connection)
            pending: list[SchemaMigration] = []
            for migration in ordered:
                row = existing.get(migration.migration_id)
                if row is None:
                    pending.append(migration)
                    continue
                if row["state"] == "failed":
                    raise SchemaMigrationError(migration.migration_id, "previous_failure")
                if row["state"] != "applied":
                    raise SchemaMigrationError(migration.migration_id, "state_invalid")
                if row["checksum"] != migration.checksum:
                    raise SchemaMigrationError(migration.migration_id, "checksum_mismatch")
                if migration.validate is not None:
                    try:
                        migration.validate(connection)
                    except Exception as exc:
                        code = _failure_code(exc, "validation")
                        raise SchemaMigrationError(migration.migration_id, code) from exc
                already_applied.append(migration.migration_id)

            for migration in pending:
                started_at = _utcnow()
                try:
                    migration.upgrade(connection)
                except Exception as exc:
                    raise _ExecutionFailure(
                        migration.migration_id,
                        _failure_code(exc, "upgrade"),
                        exc,
                    ) from exc
                if migration.validate is not None:
                    try:
                        migration.validate(connection)
                    except Exception as exc:
                        raise _ExecutionFailure(
                            migration.migration_id,
                            _failure_code(exc, "validation"),
                            exc,
                        ) from exc
                connection.execute(
                    schema_migration_table.insert().values(
                        migration_id=migration.migration_id,
                        checksum=migration.checksum,
                        state="applied",
                        runner_version=MIGRATION_RUNNER_VERSION,
                        started_at=started_at,
                        finished_at=_utcnow(),
                        error_code=None,
                    )
                )
                applied.append(migration.migration_id)
    except _ExecutionFailure as failure:
        migration = next(
            migration
            for migration in ordered
            if migration.migration_id == failure.migration_id
        )
        try:
            _record_failure(
                engine,
                dialect,
                migration=migration,
                started_at=started_at,
                error_code=failure.code,
            )
        except Exception as record_exc:
            raise SchemaMigrationError(
                failure.migration_id,
                "failure_state_record_failed",
            ) from record_exc
        raise SchemaMigrationError(failure.migration_id, failure.code) from failure.cause

    return MigrationReport(tuple(applied), tuple(already_applied))


def read_schema_migration_state(engine: Engine) -> tuple[MigrationState, ...]:
    """Return ordered migration metadata for health checks and operations."""

    dialect = _require_supported_dialect(engine)
    _bootstrap_migration_table(engine, dialect)
    with engine.connect() as connection:
        rows = connection.execute(
            select(schema_migration_table).order_by(schema_migration_table.c.migration_id)
        ).mappings()
        return tuple(
            MigrationState(
                migration_id=str(row["migration_id"]),
                checksum=str(row["checksum"]),
                state=str(row["state"]),
                runner_version=str(row["runner_version"]),
                error_code=str(row["error_code"]) if row["error_code"] else None,
            )
            for row in rows
        )


__all__ = [
    "CORE_SCHEMA_MIGRATIONS",
    "MIGRATION_RUNNER_VERSION",
    "MIGRATION_TABLE_NAME",
    "MigrationReport",
    "MigrationState",
    "SchemaInvariantError",
    "SchemaMigration",
    "SchemaMigrationError",
    "read_schema_migration_state",
    "run_schema_migrations",
]
