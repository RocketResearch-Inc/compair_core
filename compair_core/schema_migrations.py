"""Forward-only, transactional schema migrations for Core.

This module intentionally has no dependency on ``compair_core.compair`` so it
can be imported and tested without triggering application startup.  The first
registered migration records the schema produced by the pre-registry startup
path; future additive migrations can build on that checked baseline.
"""

from __future__ import annotations

import hashlib
import json
import re
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
_CHUNK_RETENTION_TRIGGER = "trg_chunk_baseline_retention_before_delete"
_CHUNK_RETENTION_FUNCTION = "core_chunk_baseline_retention_v1"
_GENERATION_STATES = (
    "pending",
    "running",
    "succeeded",
    "retryable_failed",
    "terminal_failed",
    "blocked",
)

_GENERATION_RUN_COLUMNS: tuple[tuple[str, str], ...] = (
    ("generation_lease_token", "VARCHAR(128)"),
    ("generation_started_at", "TIMESTAMP"),
    ("generation_input_fingerprint", "VARCHAR(64)"),
    ("generation_provider", "VARCHAR(128)"),
    ("generation_model", "VARCHAR(256)"),
    ("generation_model_version", "VARCHAR(256)"),
    ("generation_output_fingerprint", "VARCHAR(64)"),
    ("generation_error_fingerprint", "VARCHAR(64)"),
    ("generation_updated_at", "TIMESTAMP"),
)

_GENERATION_FEEDBACK_COLUMNS: tuple[tuple[str, str], ...] = (
    ("generation_provider", "VARCHAR(128)"),
    ("generation_model", "VARCHAR(256)"),
    ("generation_model_version", "VARCHAR(256)"),
    ("generation_input_fingerprint", "VARCHAR(64)"),
    ("generation_output_fingerprint", "VARCHAR(64)"),
)


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
            ("artifact_id", "group_id"),
            "baseline_evidence_artifact",
            ("artifact_id", "group_id"),
            "NO ACTION",
        ),
    }
    if not required_selected_fks <= selected_fks:
        raise SchemaInvariantError("selected_scope_foreign_key_invalid")
    if not any(
        candidate in selected_fks
        for candidate in (
            (
                ("run_id", "group_id"),
                BASELINE_RETRIEVAL_RUN_TABLE,
                ("run_id", "group_id"),
                "CASCADE",
            ),
            (
                ("run_id", "group_id"),
                BASELINE_RETRIEVAL_RUN_TABLE,
                ("run_id", "group_id"),
                "NO ACTION",
            ),
        )
    ):
        raise SchemaInvariantError("selected_run_foreign_key_invalid")

    run_fks = _foreign_key_targets(connection, BASELINE_RETRIEVAL_RUN_TABLE)
    if (("group_id",), "group", ("group_id",), "CASCADE") not in run_fks:
        raise SchemaInvariantError("run_lifecycle_foreign_key_invalid")
    for constrained, target, referred in (
        (("source_chunk_id",), "chunk", ("chunk_id",)),
        (("source_document_id",), "document", ("document_id",)),
    ):
        if not any(
            (constrained, target, referred, action) in run_fks
            for action in ("CASCADE", "SET NULL")
        ):
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
    if not any(
        candidate in feedback_fks
        for candidate in (
            (
                ("baseline_retrieval_run_id",),
                BASELINE_RETRIEVAL_RUN_TABLE,
                ("run_id",),
                "CASCADE",
            ),
            (
                ("baseline_retrieval_run_id", "baseline_finding_ordinal"),
                BASELINE_SELECTED_EVIDENCE_TABLE,
                ("run_id", "ordinal"),
                "CASCADE",
            ),
        )
    ):
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
        actual_checks = _constraint_names(connection, table.name, "check")
        if (
            table.name == BASELINE_RETRIEVAL_RUN_TABLE
            and "ck_bl_run_generation_v1" in actual_checks
        ):
            expected_checks.discard("ck_bl_run_generation")
        if not expected_checks <= actual_checks:
            raise SchemaInvariantError(f"missing_check_constraint:{table.name}")

    if connection.dialect.name == "sqlite":
        foreign_keys_enabled = connection.exec_driver_sql(
            "PRAGMA foreign_keys"
        ).scalar_one()
        suspended_by_runner = bool(
            connection.info.get("core_migration_foreign_keys_suspended")
        )
        if foreign_keys_enabled != 1 and not suspended_by_runner:
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


def _sqlite_create_sql(connection: Connection, table_name: str) -> str:
    sql = connection.execute(
        text(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = :table_name"
        ),
        {"table_name": table_name},
    ).scalar_one_or_none()
    if not sql:
        raise SchemaInvariantError(f"sqlite_create_sql_missing:{table_name}")
    return str(sql)


def _sqlite_rewrite_once(
    sql: str,
    pattern: str,
    replacement: str,
    error_code: str,
) -> str:
    rewritten, count = re.subn(pattern, replacement, sql, count=1, flags=re.IGNORECASE)
    if count != 1:
        raise SchemaInvariantError(error_code)
    return rewritten


def _sqlite_nullable_source_chunk(sql: str, table_name: str) -> str:
    return _sqlite_rewrite_once(
        sql,
        r"(\bsource_chunk_id\s+[A-Z]+(?:\s*\(\s*\d+\s*\))?)\s+NOT\s+NULL",
        r"\1",
        f"sqlite_source_chunk_nullability_rewrite_failed:{table_name}",
    )


def _sqlite_retarget_table(sql: str, table_name: str, temporary_name: str) -> str:
    return _sqlite_rewrite_once(
        sql,
        rf"^(CREATE\s+TABLE\s+)(?:\"{re.escape(table_name)}\"|{re.escape(table_name)})",
        rf'\1"{temporary_name}"',
        f"sqlite_table_rewrite_failed:{table_name}",
    )


def _sqlite_append_constraint(sql: str, definition: str, table_name: str) -> str:
    closing = sql.rfind(")")
    if closing < 0:
        raise SchemaInvariantError(f"sqlite_constraint_append_failed:{table_name}")
    return f"{sql[:closing]}, {definition}{sql[closing:]}"


def _sqlite_retention_table_sql(
    connection: Connection,
    table_name: str,
    temporary_name: str,
) -> str:
    sql = _sqlite_retarget_table(
        _sqlite_create_sql(connection, table_name), table_name, temporary_name
    )
    chunk_fk = (
        r"((?:CONSTRAINT\s+[^\s,]+\s+)?FOREIGN\s+KEY\s*\(\s*source_chunk_id\s*\)"
        r"\s+REFERENCES\s+(?:\"?chunk\"?)\s*\(\s*chunk_id\s*\)\s+ON\s+DELETE\s+)CASCADE"
    )
    if table_name == BASELINE_RETRIEVAL_RUN_TABLE:
        sql = _sqlite_nullable_source_chunk(sql, table_name)
        sql = _sqlite_rewrite_once(
            sql,
            chunk_fk,
            r"\1SET NULL",
            "sqlite_run_chunk_fk_rewrite_failed",
        )
        sql = _sqlite_rewrite_once(
            sql,
            r"((?:CONSTRAINT\s+[^\s,]+\s+)?FOREIGN\s+KEY\s*\(\s*source_document_id\s*\)"
            r"\s+REFERENCES\s+(?:\"?document\"?)\s*\(\s*document_id\s*\)\s+ON\s+DELETE\s+)CASCADE",
            r"\1SET NULL",
            "sqlite_run_document_fk_rewrite_failed",
        )
    elif table_name == BASELINE_SELECTED_EVIDENCE_TABLE:
        sql = _sqlite_rewrite_once(
            sql,
            r"((?:CONSTRAINT\s+[^\s,]+\s+)?FOREIGN\s+KEY\s*\(\s*run_id\s*,\s*group_id\s*\)"
            r"\s+REFERENCES\s+(?:\"?baseline_retrieval_run\"?)"
            r"\s*\(\s*run_id\s*,\s*group_id\s*\)\s+ON\s+DELETE\s+)CASCADE",
            r"\1NO ACTION DEFERRABLE INITIALLY DEFERRED",
            "sqlite_selected_run_fk_rewrite_failed",
        )
        sql = _sqlite_append_constraint(
            sql,
            'CONSTRAINT fk_bl_selected_group_retention FOREIGN KEY(group_id) '
            'REFERENCES "group"(group_id) ON DELETE CASCADE',
            table_name,
        )
    elif table_name == "reference":
        sql = _sqlite_nullable_source_chunk(sql, table_name)
        sql = _sqlite_rewrite_once(
            sql,
            chunk_fk,
            r"\1SET NULL",
            "sqlite_reference_chunk_fk_rewrite_failed",
        )
    elif table_name == "feedback":
        sql = _sqlite_nullable_source_chunk(sql, table_name)
        sql = _sqlite_rewrite_once(
            sql,
            chunk_fk,
            r"\1SET NULL",
            "sqlite_feedback_chunk_fk_rewrite_failed",
        )
        sql = _sqlite_rewrite_once(
            sql,
            r"(baseline_retrieval_run_id\s+[A-Z]+(?:\s*\(\s*\d+\s*\))?)"
            r"\s+REFERENCES\s+(?:\"?baseline_retrieval_run\"?)"
            r"\s*\(\s*run_id\s*\)\s+ON\s+DELETE\s+CASCADE",
            r"\1",
            "sqlite_feedback_run_fk_rewrite_failed",
        )
        sql = _sqlite_append_constraint(
            sql,
            "CONSTRAINT fk_feedback_baseline_selected_evidence "
            "FOREIGN KEY(baseline_retrieval_run_id, baseline_finding_ordinal) "
            "REFERENCES baseline_selected_evidence(run_id, ordinal) "
            "ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED",
            table_name,
        )
    else:  # pragma: no cover - private caller contract
        raise ValueError(table_name)
    return sql


def _sqlite_schema_objects(
    connection: Connection, table_names: tuple[str, ...]
) -> tuple[str, ...]:
    rows = connection.execute(
        text(
            "SELECT sql FROM sqlite_master "
            "WHERE tbl_name IN (:run, :selected, :reference, :feedback) "
            "AND type IN ('index', 'trigger') AND sql IS NOT NULL "
            "ORDER BY type, name"
        ),
        {
            "run": table_names[0],
            "selected": table_names[1],
            "reference": table_names[2],
            "feedback": table_names[3],
        },
    ).scalars()
    return tuple(str(row) for row in rows)


def _sqlite_copy_table(
    connection: Connection, source_name: str, target_name: str
) -> None:
    columns = [
        str(row[1])
        for row in connection.exec_driver_sql(
            f'PRAGMA table_info("{source_name}")'
        ).all()
    ]
    if not columns:
        raise SchemaInvariantError(f"sqlite_copy_columns_missing:{source_name}")
    quoted = ", ".join(f'"{column}"' for column in columns)
    connection.exec_driver_sql(
        f'INSERT INTO "{target_name}" ({quoted}) '
        f'SELECT {quoted} FROM "{source_name}"'
    )


def _create_sqlite_chunk_retention_trigger(connection: Connection) -> None:
    connection.exec_driver_sql(
        f"CREATE TRIGGER {_CHUNK_RETENTION_TRIGGER} "
        "BEFORE DELETE ON chunk BEGIN "
        "DELETE FROM reference WHERE source_chunk_id = OLD.chunk_id "
        "AND baseline_selected_evidence_id IS NULL; "
        "DELETE FROM feedback WHERE source_chunk_id = OLD.chunk_id "
        "AND baseline_retrieval_run_id IS NULL; END"
    )


def _upgrade_sqlite_baseline_evidence_retention(connection: Connection) -> None:
    table_names = (
        BASELINE_RETRIEVAL_RUN_TABLE,
        BASELINE_SELECTED_EVIDENCE_TABLE,
        "reference",
        "feedback",
    )
    preserved_objects = _sqlite_schema_objects(connection, table_names)
    temporary_names = {
        table_name: f"__retention_v2_{table_name}" for table_name in table_names
    }
    for table_name in table_names:
        connection.exec_driver_sql(
            _sqlite_retention_table_sql(
                connection, table_name, temporary_names[table_name]
            )
        )
        _sqlite_copy_table(connection, table_name, temporary_names[table_name])

    for table_name in ("reference", "feedback", BASELINE_SELECTED_EVIDENCE_TABLE, BASELINE_RETRIEVAL_RUN_TABLE):
        connection.exec_driver_sql(f'DROP TABLE "{table_name}"')
    for table_name in table_names:
        connection.exec_driver_sql(
            f'ALTER TABLE "{temporary_names[table_name]}" RENAME TO "{table_name}"'
        )
    for ddl in preserved_objects:
        connection.exec_driver_sql(ddl)
    connection.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS ix_bl_selected_group_retention "
        "ON baseline_selected_evidence(group_id)"
    )
    _create_sqlite_chunk_retention_trigger(connection)


def _postgres_drop_foreign_key(
    connection: Connection, table_name: str, constrained_columns: tuple[str, ...]
) -> None:
    preparer = connection.dialect.identifier_preparer
    for row in inspect(connection).get_foreign_keys(table_name):
        if tuple(row.get("constrained_columns") or ()) != constrained_columns:
            continue
        name = row.get("name")
        if not name:
            raise SchemaInvariantError(f"postgres_unnamed_fk:{table_name}")
        connection.exec_driver_sql(
            f"ALTER TABLE {preparer.quote(table_name)} "
            f"DROP CONSTRAINT {preparer.quote(str(name))}"
        )
        return
    raise SchemaInvariantError(
        f"postgres_foreign_key_missing:{table_name}:{','.join(constrained_columns)}"
    )


def _upgrade_postgres_baseline_evidence_retention(connection: Connection) -> None:
    for column_name in ("source_chunk_id", "source_document_id"):
        _postgres_drop_foreign_key(
            connection, BASELINE_RETRIEVAL_RUN_TABLE, (column_name,)
        )
    connection.exec_driver_sql(
        "ALTER TABLE baseline_retrieval_run ALTER COLUMN source_chunk_id DROP NOT NULL"
    )
    _postgres_add_constraint(
        connection,
        table_name=BASELINE_RETRIEVAL_RUN_TABLE,
        constraint_name="fk_bl_run_source_chunk_retention",
        definition="FOREIGN KEY (source_chunk_id) REFERENCES chunk(chunk_id) ON DELETE SET NULL",
        kind="foreign_key",
    )
    _postgres_add_constraint(
        connection,
        table_name=BASELINE_RETRIEVAL_RUN_TABLE,
        constraint_name="fk_bl_run_source_document_retention",
        definition="FOREIGN KEY (source_document_id) REFERENCES document(document_id) ON DELETE SET NULL",
        kind="foreign_key",
    )

    _postgres_drop_foreign_key(
        connection, BASELINE_SELECTED_EVIDENCE_TABLE, ("run_id", "group_id")
    )
    _postgres_add_constraint(
        connection,
        table_name=BASELINE_SELECTED_EVIDENCE_TABLE,
        constraint_name="fk_bl_selected_run_scope_retention",
        definition=(
            "FOREIGN KEY (run_id, group_id) "
            "REFERENCES baseline_retrieval_run(run_id, group_id) "
            "ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED"
        ),
        kind="foreign_key",
    )
    _postgres_add_constraint(
        connection,
        table_name=BASELINE_SELECTED_EVIDENCE_TABLE,
        constraint_name="fk_bl_selected_group_retention",
        definition='FOREIGN KEY (group_id) REFERENCES "group"(group_id) ON DELETE CASCADE',
        kind="foreign_key",
    )

    _postgres_drop_foreign_key(connection, "reference", ("source_chunk_id",))
    connection.exec_driver_sql(
        "ALTER TABLE reference ALTER COLUMN source_chunk_id DROP NOT NULL"
    )
    _postgres_add_constraint(
        connection,
        table_name="reference",
        constraint_name="fk_reference_source_chunk_retention",
        definition="FOREIGN KEY (source_chunk_id) REFERENCES chunk(chunk_id) ON DELETE SET NULL",
        kind="foreign_key",
    )

    _postgres_drop_foreign_key(connection, "feedback", ("source_chunk_id",))
    _postgres_drop_foreign_key(
        connection, "feedback", ("baseline_retrieval_run_id",)
    )
    connection.exec_driver_sql(
        "ALTER TABLE feedback ALTER COLUMN source_chunk_id DROP NOT NULL"
    )
    _postgres_add_constraint(
        connection,
        table_name="feedback",
        constraint_name="fk_feedback_source_chunk_retention",
        definition="FOREIGN KEY (source_chunk_id) REFERENCES chunk(chunk_id) ON DELETE SET NULL",
        kind="foreign_key",
    )
    _postgres_add_constraint(
        connection,
        table_name="feedback",
        constraint_name="fk_feedback_baseline_selected_evidence",
        definition=(
            "FOREIGN KEY (baseline_retrieval_run_id, baseline_finding_ordinal) "
            "REFERENCES baseline_selected_evidence(run_id, ordinal) "
            "ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED"
        ),
        kind="foreign_key",
    )
    connection.exec_driver_sql(
        f"CREATE OR REPLACE FUNCTION {_CHUNK_RETENTION_FUNCTION}() "
        "RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
        "DELETE FROM reference WHERE source_chunk_id = OLD.chunk_id "
        "AND baseline_selected_evidence_id IS NULL; "
        "DELETE FROM feedback WHERE source_chunk_id = OLD.chunk_id "
        "AND baseline_retrieval_run_id IS NULL; RETURN OLD; END $$"
    )
    connection.exec_driver_sql(
        f"DROP TRIGGER IF EXISTS {_CHUNK_RETENTION_TRIGGER} ON chunk"
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER {_CHUNK_RETENTION_TRIGGER} BEFORE DELETE ON chunk "
        f"FOR EACH ROW EXECUTE FUNCTION {_CHUNK_RETENTION_FUNCTION}()"
    )


def _upgrade_baseline_evidence_retention(connection: Connection) -> None:
    if connection.dialect.name == "sqlite":
        _upgrade_sqlite_baseline_evidence_retention(connection)
    else:
        _upgrade_postgres_baseline_evidence_retention(connection)
        connection.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS ix_bl_selected_group_retention "
            "ON baseline_selected_evidence(group_id)"
        )


def _column_nullable(connection: Connection, table_name: str, column_name: str) -> bool:
    for column in inspect(connection).get_columns(table_name):
        if column["name"] == column_name:
            return bool(column["nullable"])
    raise SchemaInvariantError(f"missing_column:{table_name}:{column_name}")


def _validate_baseline_evidence_retention(connection: Connection) -> None:
    expected_fks = {
        BASELINE_RETRIEVAL_RUN_TABLE: {
            (("group_id",), "group", ("group_id",), "CASCADE"),
            (("source_chunk_id",), "chunk", ("chunk_id",), "SET NULL"),
            (("source_document_id",), "document", ("document_id",), "SET NULL"),
        },
        BASELINE_SELECTED_EVIDENCE_TABLE: {
            (("group_id",), "group", ("group_id",), "CASCADE"),
            (
                ("run_id", "group_id"),
                BASELINE_RETRIEVAL_RUN_TABLE,
                ("run_id", "group_id"),
                "NO ACTION",
            ),
            (
                ("artifact_id", "group_id"),
                "baseline_evidence_artifact",
                ("artifact_id", "group_id"),
                "NO ACTION",
            ),
        },
        "reference": {
            (("source_chunk_id",), "chunk", ("chunk_id",), "SET NULL"),
            (
                ("baseline_selected_evidence_id",),
                BASELINE_SELECTED_EVIDENCE_TABLE,
                ("selected_evidence_id",),
                "CASCADE",
            ),
        },
        "feedback": {
            (("source_chunk_id",), "chunk", ("chunk_id",), "SET NULL"),
            (
                ("baseline_retrieval_run_id", "baseline_finding_ordinal"),
                BASELINE_SELECTED_EVIDENCE_TABLE,
                ("run_id", "ordinal"),
                "CASCADE",
            ),
        },
    }
    for table_name, required in expected_fks.items():
        if not required <= _foreign_key_targets(connection, table_name):
            raise SchemaInvariantError(f"retention_foreign_key_invalid:{table_name}")
    for table_name in (BASELINE_RETRIEVAL_RUN_TABLE, "reference", "feedback"):
        if not _column_nullable(connection, table_name, "source_chunk_id"):
            raise SchemaInvariantError(f"retention_source_not_nullable:{table_name}")
    if not _column_nullable(
        connection, BASELINE_RETRIEVAL_RUN_TABLE, "source_document_id"
    ):
        raise SchemaInvariantError("retention_source_not_nullable:baseline_retrieval_run")
    if "ix_bl_selected_group_retention" not in _index_names(
        connection, BASELINE_SELECTED_EVIDENCE_TABLE
    ):
        raise SchemaInvariantError("retention_group_index_missing")

    if connection.dialect.name == "sqlite":
        trigger_names = {
            str(row[0])
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).all()
        }
    else:
        trigger_names = {
            str(row[0])
            for row in connection.execute(
                text(
                    "SELECT tgname FROM pg_trigger "
                    "WHERE NOT tgisinternal AND tgrelid = 'chunk'::regclass"
                )
            ).all()
        }
    if _CHUNK_RETENTION_TRIGGER not in trigger_names:
        raise SchemaInvariantError("chunk_retention_trigger_missing")

    # Audit provenance is copied by value. It must never acquire a retention
    # foreign key to mutable corpus/index lifecycle tables.
    forbidden_targets = {
        "retrieval_corpus",
        "retrieval_corpus_generation",
        "retrieval_baseline_index_build",
        "retrieval_baseline_index_publication",
    }
    for table_name in (BASELINE_RETRIEVAL_RUN_TABLE, "baseline_evidence_artifact"):
        targets = {target for _, target, _, _ in _foreign_key_targets(connection, table_name)}
        if forbidden_targets & targets:
            raise SchemaInvariantError(f"retention_mutable_provenance_fk:{table_name}")


def _sqlite_clone_table_sql(
    connection: Connection,
    table_name: str,
    target_name: str,
) -> str:
    source = connection.execute(
        text(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = :table_name"
        ),
        {"table_name": table_name},
    ).scalar_one_or_none()
    if not source:
        raise SchemaInvariantError(f"sqlite_table_sql_missing:{table_name}")
    return _sqlite_retarget_table(str(source), table_name, target_name)


def _sqlite_append_columns(
    sql: str,
    columns: Sequence[str],
    table_name: str,
) -> str:
    if not columns:
        return sql
    constraint = re.search(
        r",\s*(?=(?:PRIMARY\s+KEY|UNIQUE\s*\(|CHECK\s*\(|"
        r"CONSTRAINT\s+|FOREIGN\s+KEY\s*\())",
        sql,
        flags=re.IGNORECASE,
    )
    if constraint is None:
        raise SchemaInvariantError(f"sqlite_column_append_failed:{table_name}")
    insertion = ", " + ", ".join(columns)
    return f"{sql[:constraint.start()]}{insertion}{sql[constraint.start():]}"


def _sqlite_generation_run_sql(
    connection: Connection,
    target_name: str,
) -> str:
    sql = _sqlite_clone_table_sql(
        connection,
        BASELINE_RETRIEVAL_RUN_TABLE,
        target_name,
    )
    states = ", ".join(f"'{state}'" for state in _GENERATION_STATES)
    sql, count = re.subn(
        r"generation_state\s+IN\s*\(\s*'pending'\s*,\s*'generating'\s*,"
        r"\s*'completed'\s*,\s*'failed'\s*\)",
        f"generation_state IN ({states})",
        sql,
        count=1,
        flags=re.IGNORECASE,
    )
    if count != 1:
        raise SchemaInvariantError("sqlite_generation_check_rewrite_failed")
    existing = _column_names(connection, BASELINE_RETRIEVAL_RUN_TABLE)
    additions = [
        f'"{column_name}" {ddl}'
        for column_name, ddl in _GENERATION_RUN_COLUMNS
        if column_name not in existing
    ]
    return _sqlite_append_columns(sql, additions, target_name)


def _sqlite_generation_feedback_sql(
    connection: Connection,
    target_name: str,
) -> str:
    sql = _sqlite_clone_table_sql(connection, "feedback", target_name)
    existing = _column_names(connection, "feedback")
    additions = [
        f'"{column_name}" {ddl}'
        for column_name, ddl in _GENERATION_FEEDBACK_COLUMNS
        if column_name not in existing
    ]
    return _sqlite_append_columns(sql, additions, target_name)


def _sqlite_copy_generation_run(
    connection: Connection,
    target_name: str,
) -> None:
    columns = [
        str(row[1])
        for row in connection.exec_driver_sql(
            f'PRAGMA table_info("{BASELINE_RETRIEVAL_RUN_TABLE}")'
        ).all()
    ]
    quoted = ", ".join(f'"{column}"' for column in columns)
    selected = [
        (
            "CASE generation_state "
            "WHEN 'generating' THEN 'running' "
            "WHEN 'completed' THEN 'succeeded' "
            "WHEN 'failed' THEN 'retryable_failed' "
            "ELSE generation_state END"
            if column == "generation_state"
            else f'"{column}"'
        )
        for column in columns
    ]
    connection.exec_driver_sql(
        f'INSERT INTO "{target_name}" ({quoted}) '
        f"SELECT {', '.join(selected)} "
        f'FROM "{BASELINE_RETRIEVAL_RUN_TABLE}"'
    )


def _upgrade_sqlite_baseline_generation_state(connection: Connection) -> None:
    table_names = (
        BASELINE_RETRIEVAL_RUN_TABLE,
        BASELINE_SELECTED_EVIDENCE_TABLE,
        "reference",
        "feedback",
    )
    preserved_objects = _sqlite_schema_objects(connection, table_names)
    temporary = {
        table_name: f"__generation_v1_{table_name}" for table_name in table_names
    }
    connection.exec_driver_sql(
        _sqlite_generation_run_sql(
            connection,
            temporary[BASELINE_RETRIEVAL_RUN_TABLE],
        )
    )
    _sqlite_copy_generation_run(
        connection,
        temporary[BASELINE_RETRIEVAL_RUN_TABLE],
    )
    for table_name in (BASELINE_SELECTED_EVIDENCE_TABLE, "reference"):
        connection.exec_driver_sql(
            _sqlite_clone_table_sql(connection, table_name, temporary[table_name])
        )
        _sqlite_copy_table(connection, table_name, temporary[table_name])
    connection.exec_driver_sql(
        _sqlite_generation_feedback_sql(connection, temporary["feedback"])
    )
    _sqlite_copy_table(connection, "feedback", temporary["feedback"])

    connection.exec_driver_sql(
        f"DROP TRIGGER IF EXISTS {_CHUNK_RETENTION_TRIGGER}"
    )
    for table_name in (
        "reference",
        "feedback",
        BASELINE_SELECTED_EVIDENCE_TABLE,
        BASELINE_RETRIEVAL_RUN_TABLE,
    ):
        connection.exec_driver_sql(f'DROP TABLE "{table_name}"')
    for table_name in table_names:
        connection.exec_driver_sql(
            f'ALTER TABLE "{temporary[table_name]}" RENAME TO "{table_name}"'
        )
    for ddl in preserved_objects:
        connection.exec_driver_sql(ddl)
    _create_sqlite_chunk_retention_trigger(connection)
    connection.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS ix_bl_run_generation_state "
        "ON baseline_retrieval_run(generation_state, generation_lease_expires_at)"
    )


def _upgrade_postgres_baseline_generation_state(connection: Connection) -> None:
    run_columns = _column_names(connection, BASELINE_RETRIEVAL_RUN_TABLE)
    for column_name, ddl in _GENERATION_RUN_COLUMNS:
        if column_name not in run_columns:
            connection.exec_driver_sql(
                f"ALTER TABLE baseline_retrieval_run ADD COLUMN {column_name} {ddl}"
            )
    feedback_columns = _column_names(connection, "feedback")
    for column_name, ddl in _GENERATION_FEEDBACK_COLUMNS:
        if column_name not in feedback_columns:
            connection.exec_driver_sql(
                f"ALTER TABLE feedback ADD COLUMN {column_name} {ddl}"
            )
    connection.exec_driver_sql(
        "ALTER TABLE baseline_retrieval_run "
        "DROP CONSTRAINT IF EXISTS ck_bl_run_generation"
    )
    connection.exec_driver_sql(
        "UPDATE baseline_retrieval_run SET generation_state = CASE generation_state "
        "WHEN 'generating' THEN 'running' WHEN 'completed' THEN 'succeeded' "
        "WHEN 'failed' THEN 'retryable_failed' ELSE generation_state END"
    )
    states = ", ".join(f"'{state}'" for state in _GENERATION_STATES)
    _postgres_add_constraint(
        connection,
        table_name=BASELINE_RETRIEVAL_RUN_TABLE,
        constraint_name="ck_bl_run_generation_v1",
        definition=(
            f"CHECK (generation_state IN ({states}) "
            "AND generation_attempt_count >= 0)"
        ),
        kind="check",
    )
    connection.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS ix_bl_run_generation_state "
        "ON baseline_retrieval_run(generation_state, generation_lease_expires_at)"
    )


def _upgrade_baseline_generation_state(connection: Connection) -> None:
    if connection.dialect.name == "sqlite":
        _upgrade_sqlite_baseline_generation_state(connection)
    else:
        _upgrade_postgres_baseline_generation_state(connection)


def _validate_baseline_generation_state(connection: Connection) -> None:
    required_run = {name for name, _ddl in _GENERATION_RUN_COLUMNS}
    required_feedback = {name for name, _ddl in _GENERATION_FEEDBACK_COLUMNS}
    if not required_run <= _column_names(connection, BASELINE_RETRIEVAL_RUN_TABLE):
        raise SchemaInvariantError("baseline_generation_run_columns_missing")
    if not required_feedback <= _column_names(connection, "feedback"):
        raise SchemaInvariantError("baseline_generation_feedback_columns_missing")
    if "ix_bl_run_generation_state" not in _index_names(
        connection, BASELINE_RETRIEVAL_RUN_TABLE
    ):
        raise SchemaInvariantError("baseline_generation_state_index_missing")
    if connection.dialect.name == "sqlite":
        schema_sql = connection.execute(
            text(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'table' AND name = 'baseline_retrieval_run'"
            )
        ).scalar_one()
        for state in _GENERATION_STATES:
            if f"'{state}'" not in str(schema_sql):
                raise SchemaInvariantError(
                    f"baseline_generation_state_missing:{state}"
                )
    elif "ck_bl_run_generation_v1" not in _constraint_names(
        connection, BASELINE_RETRIEVAL_RUN_TABLE, "check"
    ):
        raise SchemaInvariantError("baseline_generation_state_check_missing")


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
    SchemaMigration(
        migration_id="0002_baseline_evidence_retention_v1",
        description="Preserve immutable baseline evidence across source lifecycle deletion",
        checksum_material=(
            "baseline-evidence-retention.v1; source chunk/document provenance uses "
            "nullable SET NULL; legacy chunk-owned Reference and Feedback deletion "
            "preserved by scoped trigger; selected evidence directly group-cascades; "
            "selected-to-run and selected-to-artifact restrict historical deletion; "
            "baseline Feedback targets selected run ordinal; mutable corpus/index "
            "provenance remains value-only; SQLite transactional copy/swap with "
            "foreign_key_check; PostgreSQL ALTER constraints; ddl-v1"
        ),
        upgrade=_upgrade_baseline_evidence_retention,
        validate=_validate_baseline_evidence_retention,
    ),
    SchemaMigration(
        migration_id="0003_baseline_generation_state_v1",
        description="Add leased baseline generation and idempotent Feedback provenance",
        checksum_material=(
            "baseline-generation-state.v1; pending running succeeded "
            "retryable_failed terminal_failed blocked; lease token and expiry; "
            "attempt count; input output and sanitized error fingerprints; "
            "provider model revision provenance on run and Feedback; SQLite "
            "transactional copy/swap preserving retention constraints; PostgreSQL "
            "additive columns and validated state check; ddl-v1"
        ),
        upgrade=_upgrade_baseline_generation_state,
        validate=_validate_baseline_generation_state,
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
            # SQLite cannot alter FK actions or column nullability in place.
            # Suspend enforcement only for the locked migration transaction;
            # a full foreign_key_check gates commit and the connection is
            # restored before it returns to the pool.
            connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
            if connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() != 0:
                raise SchemaMigrationError(None, "sqlite_foreign_key_suspend_failed")
            connection.info["core_migration_foreign_keys_suspended"] = True
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            try:
                yield connection
                violations = connection.exec_driver_sql(
                    "PRAGMA foreign_key_check"
                ).first()
                if violations is not None:
                    raise SchemaInvariantError("sqlite_foreign_key_check_failed")
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.info.pop("core_migration_foreign_keys_suspended", None)
                connection.exec_driver_sql("PRAGMA foreign_keys=ON")
                if connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() != 1:
                    raise SchemaMigrationError(None, "sqlite_foreign_key_restore_failed")
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
