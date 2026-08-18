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

from compair_core.baseline_control_plane_schema import (
    BASELINE_RUN_JOB_TABLE,
    BASELINE_RUN_PAYLOAD_TABLE,
    BASELINE_RUN_WORKER_CONTRACT_VERSION,
    BASELINE_RUN_WORKER_SERVICE_ID,
    BASELINE_WORKER_ATTESTATION_TABLE,
    BASELINE_WORKER_INSTANCE_TABLE,
    COMPATIBLE_INDEX_JOB_TABLE,
    CONTINUATION_CONTROL_PLANE_TABLES,
    CONTROL_JOB_TABLE,
    DATABASE_WORKER_ATTESTATION_TABLES,
    DATABASE_WORKER_TABLES,
    INDEX_CONTROL_PLANE_TABLES,
    REPOSITORY_APPROVAL_TABLE,
    REPOSITORY_REGISTRATION_TABLE,
    RUN_CONTROL_PLANE_TABLES,
    SNAPSHOT_CONTENT_PART_TABLE,
    SNAPSHOT_CONTINUATION_JOB_TABLE,
    SNAPSHOT_STAGING_TABLE,
    STAGING_CONTROL_PLANE_TABLES,
)
from compair_core.baseline_evidence_schema import (
    BASELINE_EVIDENCE_TABLES,
    BASELINE_RETRIEVAL_RUN_TABLE,
    BASELINE_SELECTED_EVIDENCE_TABLE,
    SOURCE_SCOPE_CONTROL_DOCUMENT,
    SOURCE_SCOPE_LEGACY_CHUNK,
    SOURCE_SCOPE_VERSION,
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

_BASELINE_NOTIFICATION_OUTBOX_TABLE = "baseline_notification_outbox"
_BASELINE_NOTIFICATION_STATES = (
    "pending",
    "running",
    "delivered",
    "retryable_failed",
    "terminal_failed",
    "suppressed",
    "cancelled",
)
_BASELINE_NOTIFICATION_INSERT_TRIGGER = "trg_bl_notify_succeeded_insert"
_BASELINE_NOTIFICATION_IMMUTABLE_TRIGGER = "trg_bl_notify_payload_immutable"
_BASELINE_NOTIFICATION_INSERT_FUNCTION = "core_bl_notify_succeeded_insert_v1"
_BASELINE_NOTIFICATION_IMMUTABLE_FUNCTION = "core_bl_notify_payload_immutable_v1"
_CONTROL_PART_IMMUTABLE_TRIGGER = "trg_bl_ctl_part_immutable"
_CONTROL_PART_IMMUTABLE_FUNCTION = "core_bl_ctl_part_immutable_v1"
_CONTROL_REGISTRATION_IMMUTABLE_TRIGGER = "trg_bl_ctl_registration_immutable"
_CONTROL_REGISTRATION_IMMUTABLE_FUNCTION = "core_bl_ctl_registration_immutable_v1"
_CONTROL_APPROVAL_IMMUTABLE_TRIGGER = "trg_bl_ctl_approval_immutable"
_CONTROL_APPROVAL_IMMUTABLE_FUNCTION = "core_bl_ctl_approval_immutable_v1"
_CONTROL_STAGING_IDENTITY_IMMUTABLE_TRIGGER = "trg_bl_ctl_staging_identity_immutable"
_CONTROL_STAGING_IDENTITY_IMMUTABLE_FUNCTION = (
    "core_bl_ctl_staging_identity_immutable_v1"
)
_CONTROL_CONTINUATION_IMMUTABLE_TRIGGER = "trg_bl_ctl_continuation_immutable"
_CONTROL_CONTINUATION_IMMUTABLE_FUNCTION = "core_bl_ctl_continuation_immutable_v1"
_CONTROL_CONTINUATION_RESULT_INSERT_TRIGGER = "trg_bl_ctl_continuation_result_insert"
_CONTROL_CONTINUATION_RESULT_UPDATE_TRIGGER = "trg_bl_ctl_continuation_result_update"
_CONTROL_CONTINUATION_RESULT_FUNCTION = "core_bl_ctl_continuation_result_v1"
_CONTROL_SEALED_PART_INSERT_TRIGGER = "trg_bl_ctl_sealed_part_insert"
_CONTROL_SEALED_PART_DELETE_TRIGGER = "trg_bl_ctl_sealed_part_delete"
_CONTROL_SEALED_PART_FUNCTION = "core_bl_ctl_sealed_part_guard_v1"
_CONTROL_INDEX_JOB_IMMUTABLE_TRIGGER = "trg_bl_idx_job_immutable"
_CONTROL_INDEX_JOB_IMMUTABLE_FUNCTION = "core_bl_idx_job_immutable_v1"
_CONTROL_INDEX_JOB_STATE_TRIGGER = "trg_bl_idx_job_state_result"
_CONTROL_INDEX_JOB_STATE_FUNCTION = "core_bl_idx_job_state_result_v1"

_CONTINUATION_RESULT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("result_corpus_id", "VARCHAR(36)"),
    ("result_generation_id", "VARCHAR(36)"),
    ("result_generation_version", "VARCHAR(128)"),
    ("result_manifest_hash", "VARCHAR(64)"),
    ("result_provenance_fingerprint", "VARCHAR(64)"),
    ("result_worker_contract_version", "VARCHAR(64)"),
    ("result_published_at", "TIMESTAMP"),
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
                tuple(
                    str(row["from"])
                    for row in sorted(parts, key=lambda item: int(item["seq"]))
                ),
                str(parts[0]["table"]),
                tuple(
                    str(row["to"])
                    for row in sorted(parts, key=lambda item: int(item["seq"]))
                ),
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
        if table.name == BASELINE_RETRIEVAL_RUN_TABLE:
            # These columns belong to migration 0010.  Keeping the earlier
            # validator frozen lets an existing 0001 database advance to it.
            expected -= {"source_scope_version", "source_scope"}
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
        raise SchemaInvariantError(
            "missing_column:reference:baseline_selected_evidence_id"
        )
    if (
        not {"baseline_retrieval_run_id", "baseline_finding_ordinal"}
        <= feedback_columns
    ):
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
        if table.name == BASELINE_RETRIEVAL_RUN_TABLE:
            expected_checks.discard("ck_bl_run_source_scope")
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
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = :table_name"
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
    pattern = r"(\bsource_chunk_id\s+[A-Z]+(?:\s*\(\s*\d+\s*\))?)\s+NOT\s+NULL"
    if re.search(pattern, sql, flags=re.IGNORECASE):
        return _sqlite_rewrite_once(
            sql,
            pattern,
            r"\1",
            f"sqlite_source_chunk_nullability_rewrite_failed:{table_name}",
        )
    if re.search(r"\bsource_chunk_id\s+[A-Z]+", sql, flags=re.IGNORECASE):
        return sql
    raise SchemaInvariantError(
        f"sqlite_source_chunk_nullability_rewrite_failed:{table_name}"
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
        if re.search(chunk_fk, sql, flags=re.IGNORECASE):
            sql = _sqlite_rewrite_once(
                sql,
                chunk_fk,
                r"\1SET NULL",
                "sqlite_run_chunk_fk_rewrite_failed",
            )
        elif not re.search(
            r"FOREIGN\s+KEY\s*\(\s*source_chunk_id\s*\).*?ON\s+DELETE\s+SET\s+NULL",
            sql,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            raise SchemaInvariantError("sqlite_run_chunk_fk_rewrite_failed")
        document_fk = (
            r"((?:CONSTRAINT\s+[^\s,]+\s+)?FOREIGN\s+KEY\s*\(\s*source_document_id\s*\)"
            r"\s+REFERENCES\s+(?:\"?document\"?)\s*\(\s*document_id\s*\)\s+ON\s+DELETE\s+)CASCADE"
        )
        if re.search(document_fk, sql, flags=re.IGNORECASE):
            sql = _sqlite_rewrite_once(
                sql,
                document_fk,
                r"\1SET NULL",
                "sqlite_run_document_fk_rewrite_failed",
            )
        elif not re.search(
            r"FOREIGN\s+KEY\s*\(\s*source_document_id\s*\).*?ON\s+DELETE\s+SET\s+NULL",
            sql,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            raise SchemaInvariantError("sqlite_run_document_fk_rewrite_failed")
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
            "CONSTRAINT fk_bl_selected_group_retention FOREIGN KEY(group_id) "
            'REFERENCES "group"(group_id) ON DELETE CASCADE',
            table_name,
        )
    elif table_name == "reference":
        sql = _sqlite_nullable_source_chunk(sql, table_name)
        if re.search(chunk_fk, sql, flags=re.IGNORECASE):
            sql = _sqlite_rewrite_once(
                sql,
                chunk_fk,
                r"\1SET NULL",
                "sqlite_reference_chunk_fk_rewrite_failed",
            )
        elif not re.search(
            r"FOREIGN\s+KEY\s*\(\s*source_chunk_id\s*\).*?ON\s+DELETE\s+SET\s+NULL",
            sql,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            raise SchemaInvariantError("sqlite_reference_chunk_fk_rewrite_failed")
    elif table_name == "feedback":
        sql = _sqlite_nullable_source_chunk(sql, table_name)
        if re.search(chunk_fk, sql, flags=re.IGNORECASE):
            sql = _sqlite_rewrite_once(
                sql,
                chunk_fk,
                r"\1SET NULL",
                "sqlite_feedback_chunk_fk_rewrite_failed",
            )
        elif not re.search(
            r"FOREIGN\s+KEY\s*\(\s*source_chunk_id\s*\).*?ON\s+DELETE\s+SET\s+NULL",
            sql,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            raise SchemaInvariantError("sqlite_feedback_chunk_fk_rewrite_failed")
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
        f'INSERT INTO "{target_name}" ({quoted}) SELECT {quoted} FROM "{source_name}"'
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

    for table_name in (
        "reference",
        "feedback",
        BASELINE_SELECTED_EVIDENCE_TABLE,
        BASELINE_RETRIEVAL_RUN_TABLE,
    ):
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
    _postgres_drop_foreign_key(connection, "feedback", ("baseline_retrieval_run_id",))
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
        raise SchemaInvariantError(
            "retention_source_not_nullable:baseline_retrieval_run"
        )
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
        targets = {
            target for _, target, _, _ in _foreign_key_targets(connection, table_name)
        }
        if forbidden_targets & targets:
            raise SchemaInvariantError(f"retention_mutable_provenance_fk:{table_name}")


def _sqlite_clone_table_sql(
    connection: Connection,
    table_name: str,
    target_name: str,
) -> str:
    source = connection.execute(
        text(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = :table_name"
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
    return f"{sql[: constraint.start()]}{insertion}{sql[constraint.start() :]}"


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

    connection.exec_driver_sql(f"DROP TRIGGER IF EXISTS {_CHUNK_RETENTION_TRIGGER}")
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
            f"CHECK (generation_state IN ({states}) AND generation_attempt_count >= 0)"
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
                raise SchemaInvariantError(f"baseline_generation_state_missing:{state}")
    elif "ck_bl_run_generation_v1" not in _constraint_names(
        connection, BASELINE_RETRIEVAL_RUN_TABLE, "check"
    ):
        raise SchemaInvariantError("baseline_generation_state_check_missing")


def _baseline_notification_outbox_ddl(dialect: str) -> str:
    timestamp_type = (
        "TIMESTAMP WITH TIME ZONE" if dialect == "postgresql" else "TIMESTAMP"
    )
    states = ", ".join(f"'{state}'" for state in _BASELINE_NOTIFICATION_STATES)
    return f"""
CREATE TABLE IF NOT EXISTS {_BASELINE_NOTIFICATION_OUTBOX_TABLE} (
    outbox_id VARCHAR(36) NOT NULL,
    run_id VARCHAR(36) NOT NULL,
    group_id VARCHAR(36) NOT NULL,
    recipient_user_id VARCHAR(36),
    channel VARCHAR(32) NOT NULL,
    digest_key VARCHAR(64) NOT NULL,
    payload_schema_version VARCHAR(64) NOT NULL,
    finding_count INTEGER NOT NULL,
    finding_manifest TEXT NOT NULL,
    finding_manifest_hash VARCHAR(64) NOT NULL,
    state VARCHAR(32) NOT NULL,
    lease_token VARCHAR(128),
    lease_expires_at {timestamp_type},
    attempt_count INTEGER NOT NULL DEFAULT 0,
    error_code VARCHAR(128),
    error_fingerprint VARCHAR(64),
    created_at {timestamp_type} NOT NULL,
    updated_at {timestamp_type} NOT NULL,
    delivered_at {timestamp_type},
    suppressed_at {timestamp_type},
    cancelled_at {timestamp_type},
    CONSTRAINT pk_baseline_notification_outbox PRIMARY KEY (outbox_id),
    CONSTRAINT uq_bl_notify_run_recipient_channel UNIQUE (run_id, recipient_user_id, channel),
    CONSTRAINT uq_bl_notify_digest_key UNIQUE (digest_key),
    CONSTRAINT fk_bl_notify_run_group FOREIGN KEY (run_id, group_id)
        REFERENCES {BASELINE_RETRIEVAL_RUN_TABLE}(run_id, group_id)
        ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
    CONSTRAINT fk_bl_notify_group FOREIGN KEY (group_id)
        REFERENCES "group"(group_id) ON DELETE CASCADE,
    CONSTRAINT fk_bl_notify_recipient FOREIGN KEY (recipient_user_id)
        REFERENCES "user"(user_id) ON DELETE SET NULL,
    CONSTRAINT ck_bl_notify_contract CHECK (
        channel = 'in_app'
        AND payload_schema_version = 'baseline-notification-digest.v1'
        AND finding_count BETWEEN 1 AND 4
        AND length(digest_key) = 64
        AND length(finding_manifest_hash) = 64
        AND attempt_count >= 0
    ),
    CONSTRAINT ck_bl_notify_state CHECK (state IN ({states})),
    CONSTRAINT ck_bl_notify_lease CHECK (
        (state = 'running' AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL)
        OR (state <> 'running' AND lease_token IS NULL AND lease_expires_at IS NULL)
    )
)
"""


def _create_sqlite_baseline_notification_triggers(connection: Connection) -> None:
    connection.exec_driver_sql(
        f"""
CREATE TRIGGER IF NOT EXISTS {_BASELINE_NOTIFICATION_INSERT_TRIGGER}
BEFORE INSERT ON {_BASELINE_NOTIFICATION_OUTBOX_TABLE}
FOR EACH ROW
WHEN NEW.recipient_user_id IS NULL OR NOT EXISTS (
    SELECT 1 FROM {BASELINE_RETRIEVAL_RUN_TABLE} r
    WHERE r.run_id = NEW.run_id
      AND r.group_id = NEW.group_id
      AND r.generation_state = 'succeeded'
)
BEGIN
    SELECT RAISE(ABORT, 'baseline_notification_requires_succeeded_run');
END
"""
    )
    connection.exec_driver_sql(
        f"""
CREATE TRIGGER IF NOT EXISTS {_BASELINE_NOTIFICATION_IMMUTABLE_TRIGGER}
BEFORE UPDATE ON {_BASELINE_NOTIFICATION_OUTBOX_TABLE}
FOR EACH ROW
WHEN NEW.run_id <> OLD.run_id
  OR NEW.group_id <> OLD.group_id
  OR NEW.channel <> OLD.channel
  OR NEW.digest_key <> OLD.digest_key
  OR NEW.payload_schema_version <> OLD.payload_schema_version
  OR NEW.finding_count <> OLD.finding_count
  OR NEW.finding_manifest <> OLD.finding_manifest
  OR NEW.finding_manifest_hash <> OLD.finding_manifest_hash
  OR NEW.created_at <> OLD.created_at
BEGIN
    SELECT RAISE(ABORT, 'baseline_notification_payload_immutable');
END
"""
    )


def _create_postgres_baseline_notification_triggers(connection: Connection) -> None:
    connection.exec_driver_sql(
        f"""
CREATE OR REPLACE FUNCTION {_BASELINE_NOTIFICATION_INSERT_FUNCTION}()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.recipient_user_id IS NULL OR NOT EXISTS (
        SELECT 1 FROM {BASELINE_RETRIEVAL_RUN_TABLE} r
        WHERE r.run_id = NEW.run_id
          AND r.group_id = NEW.group_id
          AND r.generation_state = 'succeeded'
    ) THEN
        RAISE EXCEPTION 'baseline_notification_requires_succeeded_run';
    END IF;
    RETURN NEW;
END
$$
"""
    )
    connection.exec_driver_sql(
        f"""
CREATE OR REPLACE FUNCTION {_BASELINE_NOTIFICATION_IMMUTABLE_FUNCTION}()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.run_id IS DISTINCT FROM OLD.run_id
       OR NEW.group_id IS DISTINCT FROM OLD.group_id
       OR NEW.channel IS DISTINCT FROM OLD.channel
       OR NEW.digest_key IS DISTINCT FROM OLD.digest_key
       OR NEW.payload_schema_version IS DISTINCT FROM OLD.payload_schema_version
       OR NEW.finding_count IS DISTINCT FROM OLD.finding_count
       OR NEW.finding_manifest IS DISTINCT FROM OLD.finding_manifest
       OR NEW.finding_manifest_hash IS DISTINCT FROM OLD.finding_manifest_hash
       OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
        RAISE EXCEPTION 'baseline_notification_payload_immutable';
    END IF;
    RETURN NEW;
END
$$
"""
    )
    connection.exec_driver_sql(
        f"DROP TRIGGER IF EXISTS {_BASELINE_NOTIFICATION_INSERT_TRIGGER} "
        f"ON {_BASELINE_NOTIFICATION_OUTBOX_TABLE}"
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER {_BASELINE_NOTIFICATION_INSERT_TRIGGER} BEFORE INSERT "
        f"ON {_BASELINE_NOTIFICATION_OUTBOX_TABLE} FOR EACH ROW "
        f"EXECUTE FUNCTION {_BASELINE_NOTIFICATION_INSERT_FUNCTION}()"
    )
    connection.exec_driver_sql(
        f"DROP TRIGGER IF EXISTS {_BASELINE_NOTIFICATION_IMMUTABLE_TRIGGER} "
        f"ON {_BASELINE_NOTIFICATION_OUTBOX_TABLE}"
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER {_BASELINE_NOTIFICATION_IMMUTABLE_TRIGGER} BEFORE UPDATE "
        f"ON {_BASELINE_NOTIFICATION_OUTBOX_TABLE} FOR EACH ROW "
        f"EXECUTE FUNCTION {_BASELINE_NOTIFICATION_IMMUTABLE_FUNCTION}()"
    )


def _upgrade_baseline_notification_outbox(connection: Connection) -> None:
    connection.exec_driver_sql(
        _baseline_notification_outbox_ddl(connection.dialect.name)
    )
    connection.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS ix_bl_notify_dispatch "
        "ON baseline_notification_outbox(state, lease_expires_at, created_at, outbox_id)"
    )
    connection.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS ix_bl_notify_recipient "
        "ON baseline_notification_outbox(recipient_user_id, state, created_at)"
    )
    connection.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS ix_bl_notify_group_created "
        "ON baseline_notification_outbox(group_id, created_at)"
    )
    if connection.dialect.name == "sqlite":
        _create_sqlite_baseline_notification_triggers(connection)
    else:
        _create_postgres_baseline_notification_triggers(connection)


def _validate_baseline_notification_outbox(connection: Connection) -> None:
    table_name = _BASELINE_NOTIFICATION_OUTBOX_TABLE
    if table_name not in inspect(connection).get_table_names():
        raise SchemaInvariantError("baseline_notification_outbox_missing")
    required_columns = {
        "outbox_id",
        "run_id",
        "group_id",
        "recipient_user_id",
        "channel",
        "digest_key",
        "payload_schema_version",
        "finding_count",
        "finding_manifest",
        "finding_manifest_hash",
        "state",
        "lease_token",
        "lease_expires_at",
        "attempt_count",
        "error_code",
        "error_fingerprint",
        "created_at",
        "updated_at",
        "delivered_at",
        "suppressed_at",
        "cancelled_at",
    }
    if not required_columns <= _column_names(connection, table_name):
        raise SchemaInvariantError("baseline_notification_outbox_columns_missing")
    forbidden = {
        "retrieval_query",
        "query_text",
        "raw_query",
        "source_text",
        "evidence_content",
        "finding_text",
        "feedback",
    }
    if forbidden & _column_names(connection, table_name):
        raise SchemaInvariantError("baseline_notification_outbox_private_column")
    required_fks = {
        (
            ("run_id", "group_id"),
            BASELINE_RETRIEVAL_RUN_TABLE,
            ("run_id", "group_id"),
            "CASCADE",
        ),
        (("group_id",), "group", ("group_id",), "CASCADE"),
        (("recipient_user_id",), "user", ("user_id",), "SET NULL"),
    }
    if not required_fks <= _foreign_key_targets(connection, table_name):
        raise SchemaInvariantError("baseline_notification_outbox_foreign_key_invalid")
    required_indexes = {
        "ix_bl_notify_dispatch",
        "ix_bl_notify_recipient",
        "ix_bl_notify_group_created",
    }
    if not required_indexes <= _index_names(connection, table_name):
        raise SchemaInvariantError("baseline_notification_outbox_index_missing")
    required_unique = {
        "uq_bl_notify_run_recipient_channel",
        "uq_bl_notify_digest_key",
    }
    if not required_unique <= _unique_constraint_names(connection, table_name):
        raise SchemaInvariantError("baseline_notification_outbox_unique_missing")
    required_checks = {
        "ck_bl_notify_contract",
        "ck_bl_notify_state",
        "ck_bl_notify_lease",
    }
    if not required_checks <= _constraint_names(connection, table_name, "check"):
        raise SchemaInvariantError("baseline_notification_outbox_check_missing")
    if connection.dialect.name == "sqlite":
        triggers = {
            str(row[0])
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).all()
        }
    else:
        triggers = {
            str(row[0])
            for row in connection.execute(
                text(
                    "SELECT tgname FROM pg_trigger t "
                    "JOIN pg_class c ON c.oid = t.tgrelid "
                    "WHERE c.relname = :table_name AND NOT t.tgisinternal"
                ),
                {"table_name": table_name},
            ).all()
        }
    if (
        not {
            _BASELINE_NOTIFICATION_INSERT_TRIGGER,
            _BASELINE_NOTIFICATION_IMMUTABLE_TRIGGER,
        }
        <= triggers
    ):
        raise SchemaInvariantError("baseline_notification_outbox_trigger_missing")


def _create_sqlite_control_part_trigger(connection: Connection) -> None:
    connection.exec_driver_sql(
        f"""
CREATE TRIGGER IF NOT EXISTS {_CONTROL_PART_IMMUTABLE_TRIGGER}
BEFORE UPDATE ON {SNAPSHOT_CONTENT_PART_TABLE}
FOR EACH ROW
WHEN NEW.staging_id <> OLD.staging_id
  OR NEW.group_id <> OLD.group_id
  OR NEW.part_ordinal <> OLD.part_ordinal
  OR NEW.part_sha256 <> OLD.part_sha256
  OR NEW.request_body_sha256 <> OLD.request_body_sha256
  OR NEW.item_count <> OLD.item_count
  OR NEW.content_bytes <> OLD.content_bytes
  OR NEW.canonical_content_items_json <> OLD.canonical_content_items_json
  OR NEW.created_at <> OLD.created_at
BEGIN
    SELECT RAISE(ABORT, 'baseline_control_part_immutable');
END
"""
    )


def _create_postgres_control_part_trigger(connection: Connection) -> None:
    connection.exec_driver_sql(
        f"""
CREATE OR REPLACE FUNCTION {_CONTROL_PART_IMMUTABLE_FUNCTION}()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.staging_id IS DISTINCT FROM OLD.staging_id
       OR NEW.group_id IS DISTINCT FROM OLD.group_id
       OR NEW.part_ordinal IS DISTINCT FROM OLD.part_ordinal
       OR NEW.part_sha256 IS DISTINCT FROM OLD.part_sha256
       OR NEW.request_body_sha256 IS DISTINCT FROM OLD.request_body_sha256
       OR NEW.item_count IS DISTINCT FROM OLD.item_count
       OR NEW.content_bytes IS DISTINCT FROM OLD.content_bytes
       OR NEW.canonical_content_items_json IS DISTINCT FROM OLD.canonical_content_items_json
       OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
        RAISE EXCEPTION 'baseline_control_part_immutable';
    END IF;
    RETURN NEW;
END
$$
"""
    )
    connection.exec_driver_sql(
        f"DROP TRIGGER IF EXISTS {_CONTROL_PART_IMMUTABLE_TRIGGER} "
        f"ON {SNAPSHOT_CONTENT_PART_TABLE}"
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER {_CONTROL_PART_IMMUTABLE_TRIGGER} BEFORE UPDATE "
        f"ON {SNAPSHOT_CONTENT_PART_TABLE} FOR EACH ROW "
        f"EXECUTE FUNCTION {_CONTROL_PART_IMMUTABLE_FUNCTION}()"
    )


def _upgrade_baseline_control_plane_staging(connection: Connection) -> None:
    for table in STAGING_CONTROL_PLANE_TABLES:
        connection.execute(CreateTable(table, if_not_exists=True))
        for index in sorted(table.indexes, key=lambda candidate: candidate.name or ""):
            connection.execute(CreateIndex(index, if_not_exists=True))
    if connection.dialect.name == "sqlite":
        _create_sqlite_control_part_trigger(connection)
    else:
        _create_postgres_control_part_trigger(connection)


def _validate_baseline_control_plane_staging(connection: Connection) -> None:
    tables = set(inspect(connection).get_table_names())
    for table in STAGING_CONTROL_PLANE_TABLES:
        if table.name not in tables:
            raise SchemaInvariantError(f"missing_table:{table.name}")
        missing = sorted(
            {column.name for column in table.columns}
            - _column_names(connection, table.name)
        )
        if missing:
            raise SchemaInvariantError(f"missing_column:{table.name}:{missing[0]}")

    forbidden_status_columns = {
        "retrieval_query",
        "query_text",
        "raw_query",
        "raw_diff",
        "source_text",
        "file_content",
        "content_utf8",
        "local_path",
        "endpoint_url",
        "credentials",
    }
    for table_name in (
        REPOSITORY_REGISTRATION_TABLE,
        CONTROL_JOB_TABLE,
        SNAPSHOT_STAGING_TABLE,
    ):
        if forbidden_status_columns & _column_names(connection, table_name):
            raise SchemaInvariantError(f"baseline_control_private_column:{table_name}")

    required_foreign_keys = {
        REPOSITORY_REGISTRATION_TABLE: {
            (("group_id",), "group", ("group_id",), "CASCADE"),
            (("source_document_id",), "document", ("document_id",), "SET NULL"),
            (("created_by_user_id",), "user", ("user_id",), "SET NULL"),
        },
        CONTROL_JOB_TABLE: {
            (("group_id",), "group", ("group_id",), "CASCADE"),
        },
        SNAPSHOT_STAGING_TABLE: {
            (("source_document_id",), "document", ("document_id",), "SET NULL"),
            (
                ("job_id", "group_id"),
                CONTROL_JOB_TABLE,
                ("job_id", "group_id"),
                "CASCADE",
            ),
        },
        SNAPSHOT_CONTENT_PART_TABLE: {
            (
                ("staging_id", "group_id"),
                SNAPSHOT_STAGING_TABLE,
                ("staging_id", "group_id"),
                "CASCADE",
            ),
        },
    }
    for table_name, expected in required_foreign_keys.items():
        if not expected <= _foreign_key_targets(connection, table_name):
            raise SchemaInvariantError(f"baseline_control_foreign_key:{table_name}")

    required_indexes = {
        REPOSITORY_REGISTRATION_TABLE: {
            "ix_bl_ctl_repository_group_enabled",
            "ix_bl_ctl_repository_document",
        },
        CONTROL_JOB_TABLE: {"ix_bl_ctl_job_group_state", "ix_bl_ctl_job_lease"},
        SNAPSHOT_STAGING_TABLE: {
            "ix_bl_ctl_staging_expiry",
            "ix_bl_ctl_staging_group_status",
        },
        SNAPSHOT_CONTENT_PART_TABLE: {"ix_bl_ctl_part_order"},
    }
    for table_name, expected in required_indexes.items():
        if not expected <= _index_names(connection, table_name):
            raise SchemaInvariantError(f"baseline_control_index:{table_name}")

    required_uniques = {
        REPOSITORY_REGISTRATION_TABLE: {
            "uq_bl_ctl_repository_group_id",
            "uq_bl_ctl_repository_group_name",
            "uq_bl_ctl_repository_registration_group",
        },
        CONTROL_JOB_TABLE: {
            "uq_bl_ctl_job_group_operation_intent",
            "uq_bl_ctl_job_id_group",
        },
        SNAPSHOT_STAGING_TABLE: {
            "uq_bl_ctl_staging_id_group",
            "uq_bl_ctl_staging_group_snapshot",
        },
        SNAPSHOT_CONTENT_PART_TABLE: {"uq_bl_ctl_part_staging_ordinal"},
    }
    for table_name, expected in required_uniques.items():
        if not expected <= _unique_constraint_names(connection, table_name):
            raise SchemaInvariantError(f"baseline_control_unique:{table_name}")

    required_checks = {
        REPOSITORY_REGISTRATION_TABLE: {"ck_bl_ctl_repository_identity"},
        CONTROL_JOB_TABLE: {
            "ck_bl_ctl_job_operation",
            "ck_bl_ctl_job_state",
            "ck_bl_ctl_job_contract",
            "ck_bl_ctl_job_counts",
            "ck_bl_ctl_job_lease",
        },
        SNAPSHOT_STAGING_TABLE: {
            "ck_bl_ctl_staging_status",
            "ck_bl_ctl_staging_manifest",
            "ck_bl_ctl_staging_counts",
        },
        SNAPSHOT_CONTENT_PART_TABLE: {
            "ck_bl_ctl_part_identity",
            "ck_bl_ctl_part_limits",
        },
    }
    for table_name, expected in required_checks.items():
        if not expected <= _constraint_names(connection, table_name, "check"):
            raise SchemaInvariantError(f"baseline_control_check:{table_name}")

    if connection.dialect.name == "sqlite":
        triggers = {
            str(row[0])
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).all()
        }
    else:
        triggers = {
            str(row[0])
            for row in connection.execute(
                text(
                    "SELECT tgname FROM pg_trigger t "
                    "JOIN pg_class c ON c.oid = t.tgrelid "
                    "WHERE c.relname = :table_name AND NOT t.tgisinternal"
                ),
                {"table_name": SNAPSHOT_CONTENT_PART_TABLE},
            ).all()
        }
    if _CONTROL_PART_IMMUTABLE_TRIGGER not in triggers:
        raise SchemaInvariantError("baseline_control_part_trigger_missing")


def _create_sqlite_control_continuation_triggers(connection: Connection) -> None:
    connection.exec_driver_sql(
        f"""
CREATE TRIGGER IF NOT EXISTS {_CONTROL_REGISTRATION_IMMUTABLE_TRIGGER}
BEFORE UPDATE ON {REPOSITORY_REGISTRATION_TABLE}
FOR EACH ROW
WHEN NEW.registration_id <> OLD.registration_id
  OR NEW.group_id <> OLD.group_id
  OR NEW.repository_id <> OLD.repository_id
  OR NEW.repository_name <> OLD.repository_name
  OR NEW.created_at <> OLD.created_at
BEGIN
    SELECT RAISE(ABORT, 'baseline_control_registration_immutable');
END
"""
    )
    connection.exec_driver_sql(
        f"""
CREATE TRIGGER IF NOT EXISTS {_CONTROL_APPROVAL_IMMUTABLE_TRIGGER}
BEFORE UPDATE ON {REPOSITORY_APPROVAL_TABLE}
FOR EACH ROW
WHEN NEW.registration_id <> OLD.registration_id
  OR NEW.group_id <> OLD.group_id
  OR NEW.descriptor_version <> OLD.descriptor_version
  OR NEW.repository_authority <> OLD.repository_authority
  OR NEW.repository_uid <> OLD.repository_uid
  OR NEW.descriptor_hash <> OLD.descriptor_hash
  OR NEW.created_at <> OLD.created_at
BEGIN
    SELECT RAISE(ABORT, 'baseline_control_approval_immutable');
END
"""
    )
    connection.exec_driver_sql(
        f"""
CREATE TRIGGER IF NOT EXISTS {_CONTROL_STAGING_IDENTITY_IMMUTABLE_TRIGGER}
BEFORE UPDATE ON {SNAPSHOT_STAGING_TABLE}
FOR EACH ROW
WHEN OLD.status = 'sealed' AND (
  NEW.staging_id <> OLD.staging_id
  OR NEW.group_id <> OLD.group_id
  OR NEW.job_id <> OLD.job_id
  OR NEW.snapshot_id <> OLD.snapshot_id
  OR NEW.manifest_schema_version <> OLD.manifest_schema_version
  OR NEW.canonical_manifest_hash <> OLD.canonical_manifest_hash
  OR NEW.canonical_manifest_json <> OLD.canonical_manifest_json
  OR NEW.changed_repository_id <> OLD.changed_repository_id
  OR NEW.expected_repository_count <> OLD.expected_repository_count
  OR NEW.expected_file_count <> OLD.expected_file_count
  OR NEW.expected_supported_file_count <> OLD.expected_supported_file_count
  OR NEW.expected_supported_content_bytes <> OLD.expected_supported_content_bytes
  OR NEW.expected_part_count <> OLD.expected_part_count
  OR NEW.status <> OLD.status
  OR NEW.received_part_count <> OLD.received_part_count
  OR NEW.received_file_count <> OLD.received_file_count
  OR NEW.received_content_bytes <> OLD.received_content_bytes
  OR NEW.content_manifest_hash <> OLD.content_manifest_hash
  OR NEW.expires_at <> OLD.expires_at
  OR NEW.created_at <> OLD.created_at
  OR NEW.updated_at <> OLD.updated_at
  OR NEW.sealed_at <> OLD.sealed_at
)
BEGIN
    SELECT RAISE(ABORT, 'baseline_control_staging_identity_immutable');
END
"""
    )
    connection.exec_driver_sql(
        f"""
CREATE TRIGGER IF NOT EXISTS {_CONTROL_CONTINUATION_IMMUTABLE_TRIGGER}
BEFORE UPDATE ON {SNAPSHOT_CONTINUATION_JOB_TABLE}
FOR EACH ROW
WHEN NEW.continuation_job_id <> OLD.continuation_job_id
  OR NEW.group_id <> OLD.group_id
  OR NEW.staging_id <> OLD.staging_id
  OR NEW.request_id <> OLD.request_id
  OR NEW.contract_version <> OLD.contract_version
  OR NEW.idempotency_key <> OLD.idempotency_key
  OR NEW.sealed_intent_hash <> OLD.sealed_intent_hash
  OR NEW.snapshot_id <> OLD.snapshot_id
  OR NEW.canonical_manifest_hash <> OLD.canonical_manifest_hash
  OR NEW.content_manifest_hash <> OLD.content_manifest_hash
  OR NEW.repository_set_hash <> OLD.repository_set_hash
  OR NEW.expected_repository_count <> OLD.expected_repository_count
  OR NEW.expected_file_count <> OLD.expected_file_count
  OR NEW.expected_supported_file_count <> OLD.expected_supported_file_count
  OR NEW.expected_supported_content_bytes <> OLD.expected_supported_content_bytes
  OR NEW.expected_part_count <> OLD.expected_part_count
  OR NEW.created_at <> OLD.created_at
BEGIN
    SELECT RAISE(ABORT, 'baseline_control_continuation_immutable');
END
"""
    )


def _create_postgres_control_continuation_triggers(connection: Connection) -> None:
    definitions = (
        (
            _CONTROL_REGISTRATION_IMMUTABLE_FUNCTION,
            _CONTROL_REGISTRATION_IMMUTABLE_TRIGGER,
            REPOSITORY_REGISTRATION_TABLE,
            (
                "registration_id",
                "group_id",
                "repository_id",
                "repository_name",
                "created_at",
            ),
            "baseline_control_registration_immutable",
        ),
        (
            _CONTROL_APPROVAL_IMMUTABLE_FUNCTION,
            _CONTROL_APPROVAL_IMMUTABLE_TRIGGER,
            REPOSITORY_APPROVAL_TABLE,
            (
                "registration_id",
                "group_id",
                "descriptor_version",
                "repository_authority",
                "repository_uid",
                "descriptor_hash",
                "created_at",
            ),
            "baseline_control_approval_immutable",
        ),
        (
            _CONTROL_CONTINUATION_IMMUTABLE_FUNCTION,
            _CONTROL_CONTINUATION_IMMUTABLE_TRIGGER,
            SNAPSHOT_CONTINUATION_JOB_TABLE,
            (
                "continuation_job_id",
                "group_id",
                "staging_id",
                "request_id",
                "contract_version",
                "idempotency_key",
                "sealed_intent_hash",
                "snapshot_id",
                "canonical_manifest_hash",
                "content_manifest_hash",
                "repository_set_hash",
                "expected_repository_count",
                "expected_file_count",
                "expected_supported_file_count",
                "expected_supported_content_bytes",
                "expected_part_count",
                "created_at",
            ),
            "baseline_control_continuation_immutable",
        ),
    )
    for function_name, trigger_name, table_name, columns, error_code in definitions:
        comparisons = "\n       OR ".join(
            f"NEW.{column} IS DISTINCT FROM OLD.{column}" for column in columns
        )
        connection.exec_driver_sql(
            f"""
CREATE OR REPLACE FUNCTION {function_name}()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF {comparisons} THEN
        RAISE EXCEPTION '{error_code}';
    END IF;
    RETURN NEW;
END
$$
"""
        )
        connection.exec_driver_sql(
            f"DROP TRIGGER IF EXISTS {trigger_name} ON {table_name}"
        )
        connection.exec_driver_sql(
            f"CREATE TRIGGER {trigger_name} BEFORE UPDATE ON {table_name} "
            f"FOR EACH ROW EXECUTE FUNCTION {function_name}()"
        )
    staging_columns = (
        "staging_id",
        "group_id",
        "job_id",
        "snapshot_id",
        "manifest_schema_version",
        "canonical_manifest_hash",
        "canonical_manifest_json",
        "changed_repository_id",
        "expected_repository_count",
        "expected_file_count",
        "expected_supported_file_count",
        "expected_supported_content_bytes",
        "expected_part_count",
        "status",
        "received_part_count",
        "received_file_count",
        "received_content_bytes",
        "content_manifest_hash",
        "expires_at",
        "created_at",
        "updated_at",
        "sealed_at",
    )
    staging_comparisons = "\n       OR ".join(
        f"NEW.{column} IS DISTINCT FROM OLD.{column}" for column in staging_columns
    )
    connection.exec_driver_sql(
        f"""
CREATE OR REPLACE FUNCTION {_CONTROL_STAGING_IDENTITY_IMMUTABLE_FUNCTION}()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.status = 'sealed' AND ({staging_comparisons}) THEN
        RAISE EXCEPTION 'baseline_control_staging_identity_immutable';
    END IF;
    RETURN NEW;
END
$$
"""
    )
    connection.exec_driver_sql(
        f"DROP TRIGGER IF EXISTS {_CONTROL_STAGING_IDENTITY_IMMUTABLE_TRIGGER} "
        f"ON {SNAPSHOT_STAGING_TABLE}"
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER {_CONTROL_STAGING_IDENTITY_IMMUTABLE_TRIGGER} "
        f"BEFORE UPDATE ON {SNAPSHOT_STAGING_TABLE} FOR EACH ROW EXECUTE FUNCTION "
        f"{_CONTROL_STAGING_IDENTITY_IMMUTABLE_FUNCTION}()"
    )


def _upgrade_baseline_control_plane_continuation(connection: Connection) -> None:
    for table in CONTINUATION_CONTROL_PLANE_TABLES:
        connection.execute(CreateTable(table, if_not_exists=True))
        for index in sorted(table.indexes, key=lambda candidate: candidate.name or ""):
            connection.execute(CreateIndex(index, if_not_exists=True))
    if connection.dialect.name == "sqlite":
        _create_sqlite_control_continuation_triggers(connection)
    else:
        _create_postgres_control_continuation_triggers(connection)


def _validate_baseline_control_plane_continuation(connection: Connection) -> None:
    tables = set(inspect(connection).get_table_names())
    for table in CONTINUATION_CONTROL_PLANE_TABLES:
        if table.name not in tables:
            raise SchemaInvariantError(f"missing_table:{table.name}")
        missing = sorted(
            {column.name for column in table.columns}
            - _column_names(connection, table.name)
        )
        if missing:
            raise SchemaInvariantError(f"missing_column:{table.name}:{missing[0]}")

    required_foreign_keys = {
        REPOSITORY_APPROVAL_TABLE: {
            (
                ("registration_id", "group_id"),
                REPOSITORY_REGISTRATION_TABLE,
                ("registration_id", "group_id"),
                "CASCADE",
            ),
            (("approved_by_user_id",), "user", ("user_id",), "SET NULL"),
            (("disabled_by_user_id",), "user", ("user_id",), "SET NULL"),
        },
        SNAPSHOT_CONTINUATION_JOB_TABLE: {
            (
                ("staging_id", "group_id"),
                SNAPSHOT_STAGING_TABLE,
                ("staging_id", "group_id"),
                "CASCADE",
            ),
            (("group_id",), "group", ("group_id",), "CASCADE"),
            (("created_by_user_id",), "user", ("user_id",), "SET NULL"),
        },
    }
    for table_name, expected in required_foreign_keys.items():
        if not expected <= _foreign_key_targets(connection, table_name):
            raise SchemaInvariantError(
                f"baseline_continuation_foreign_key:{table_name}"
            )

    required_indexes = {
        REPOSITORY_APPROVAL_TABLE: {"ix_bl_ctl_repository_approval_group_state"},
        SNAPSHOT_CONTINUATION_JOB_TABLE: {
            "ix_bl_ctl_continuation_group_state",
            "ix_bl_ctl_continuation_lease",
        },
    }
    for table_name, expected in required_indexes.items():
        if not expected <= _index_names(connection, table_name):
            raise SchemaInvariantError(f"baseline_continuation_index:{table_name}")

    required_uniques = {
        REPOSITORY_APPROVAL_TABLE: {"uq_bl_ctl_repository_approval_identity"},
        SNAPSHOT_CONTINUATION_JOB_TABLE: {
            "uq_bl_ctl_continuation_group_intent",
            "uq_bl_ctl_continuation_staging",
            "uq_bl_ctl_continuation_job_group",
        },
    }
    for table_name, expected in required_uniques.items():
        if not expected <= _unique_constraint_names(connection, table_name):
            raise SchemaInvariantError(f"baseline_continuation_unique:{table_name}")

    required_checks = {
        REPOSITORY_APPROVAL_TABLE: {
            "ck_bl_ctl_repository_approval_descriptor",
            "ck_bl_ctl_repository_approval_state",
            "ck_bl_ctl_repository_approval_disabled",
        },
        SNAPSHOT_CONTINUATION_JOB_TABLE: {
            "ck_bl_ctl_continuation_state",
            "ck_bl_ctl_continuation_contract",
            "ck_bl_ctl_continuation_counts",
            "ck_bl_ctl_continuation_lease",
        },
    }
    for table_name, expected in required_checks.items():
        if not expected <= _constraint_names(connection, table_name, "check"):
            raise SchemaInvariantError(f"baseline_continuation_check:{table_name}")

    if connection.dialect.name == "sqlite":
        triggers = {
            str(row[0])
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).all()
        }
    else:
        triggers = {
            str(row[0])
            for row in connection.execute(
                text(
                    "SELECT tgname FROM pg_trigger t "
                    "JOIN pg_class c ON c.oid = t.tgrelid "
                    "WHERE NOT t.tgisinternal AND c.relname IN "
                    "(:registration, :approval, :staging, :continuation)"
                ),
                {
                    "registration": REPOSITORY_REGISTRATION_TABLE,
                    "approval": REPOSITORY_APPROVAL_TABLE,
                    "staging": SNAPSHOT_STAGING_TABLE,
                    "continuation": SNAPSHOT_CONTINUATION_JOB_TABLE,
                },
            ).all()
        }
    required_triggers = {
        _CONTROL_REGISTRATION_IMMUTABLE_TRIGGER,
        _CONTROL_APPROVAL_IMMUTABLE_TRIGGER,
        _CONTROL_STAGING_IDENTITY_IMMUTABLE_TRIGGER,
        _CONTROL_CONTINUATION_IMMUTABLE_TRIGGER,
    }
    if not required_triggers <= triggers:
        raise SchemaInvariantError("baseline_continuation_trigger_missing")


def _create_sqlite_continuation_ingestion_triggers(connection: Connection) -> None:
    complete = (
        "NEW.result_corpus_id IS NOT NULL "
        "AND NEW.result_generation_id IS NOT NULL "
        "AND NEW.result_generation_version IS NOT NULL "
        "AND NEW.result_manifest_hash IS NOT NULL "
        "AND NEW.result_provenance_fingerprint IS NOT NULL "
        "AND NEW.result_worker_contract_version IS NOT NULL "
        "AND NEW.result_published_at IS NOT NULL "
        "AND length(NEW.result_corpus_id) = 36 "
        "AND length(NEW.result_generation_id) = 36 "
        "AND length(NEW.result_generation_version) BETWEEN 1 AND 128 "
        "AND length(NEW.result_manifest_hash) = 64 "
        "AND length(NEW.result_provenance_fingerprint) = 64 "
        "AND NEW.result_worker_contract_version = "
        "'baseline-continuation-worker.v1'"
    )
    absent = (
        "NEW.result_corpus_id IS NULL "
        "AND NEW.result_generation_id IS NULL "
        "AND NEW.result_generation_version IS NULL "
        "AND NEW.result_manifest_hash IS NULL "
        "AND NEW.result_provenance_fingerprint IS NULL "
        "AND NEW.result_worker_contract_version IS NULL "
        "AND NEW.result_published_at IS NULL"
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER IF NOT EXISTS {_CONTROL_CONTINUATION_RESULT_INSERT_TRIGGER} "
        f"BEFORE INSERT ON {SNAPSHOT_CONTINUATION_JOB_TABLE} FOR EACH ROW "
        f"WHEN NOT ((NEW.state = 'succeeded' AND {complete}) "
        f"OR (NEW.state <> 'succeeded' AND {absent})) "
        "BEGIN SELECT RAISE(ABORT, "
        "'baseline_control_continuation_result_invalid'); END"
    )
    immutable_result = " OR ".join(
        f"NEW.{column_name} IS NOT OLD.{column_name}"
        for column_name, _ddl in _CONTINUATION_RESULT_COLUMNS
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER IF NOT EXISTS {_CONTROL_CONTINUATION_RESULT_UPDATE_TRIGGER} "
        f"BEFORE UPDATE ON {SNAPSHOT_CONTINUATION_JOB_TABLE} FOR EACH ROW "
        f"WHEN NOT ((NEW.state = 'succeeded' AND {complete}) "
        f"OR (NEW.state <> 'succeeded' AND {absent})) "
        f"OR (OLD.state = 'succeeded' AND "
        f"(NEW.state IS NOT OLD.state OR {immutable_result})) "
        "BEGIN SELECT RAISE(ABORT, "
        "'baseline_control_continuation_result_invalid'); END"
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER IF NOT EXISTS {_CONTROL_SEALED_PART_INSERT_TRIGGER} "
        f"BEFORE INSERT ON {SNAPSHOT_CONTENT_PART_TABLE} FOR EACH ROW "
        f"WHEN EXISTS (SELECT 1 FROM {SNAPSHOT_STAGING_TABLE} "
        "WHERE staging_id = NEW.staging_id AND group_id = NEW.group_id "
        "AND status = 'sealed') "
        "BEGIN SELECT RAISE(ABORT, 'baseline_control_sealed_part_immutable'); END"
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER IF NOT EXISTS {_CONTROL_SEALED_PART_DELETE_TRIGGER} "
        f"BEFORE DELETE ON {SNAPSHOT_CONTENT_PART_TABLE} FOR EACH ROW "
        f"WHEN EXISTS (SELECT 1 FROM {SNAPSHOT_STAGING_TABLE} "
        "WHERE staging_id = OLD.staging_id AND group_id = OLD.group_id "
        "AND status = 'sealed') "
        "BEGIN SELECT RAISE(ABORT, 'baseline_control_sealed_part_immutable'); END"
    )


def _create_postgres_continuation_ingestion_triggers(connection: Connection) -> None:
    connection.exec_driver_sql(
        f"""
CREATE OR REPLACE FUNCTION {_CONTROL_CONTINUATION_RESULT_FUNCTION}()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    complete boolean;
    absent boolean;
BEGIN
    complete := NEW.result_corpus_id IS NOT NULL
        AND NEW.result_generation_id IS NOT NULL
        AND NEW.result_generation_version IS NOT NULL
        AND NEW.result_manifest_hash IS NOT NULL
        AND NEW.result_provenance_fingerprint IS NOT NULL
        AND NEW.result_worker_contract_version IS NOT NULL
        AND NEW.result_published_at IS NOT NULL
        AND length(NEW.result_corpus_id) = 36
        AND length(NEW.result_generation_id) = 36
        AND length(NEW.result_generation_version) BETWEEN 1 AND 128
        AND length(NEW.result_manifest_hash) = 64
        AND length(NEW.result_provenance_fingerprint) = 64
        AND NEW.result_worker_contract_version = 'baseline-continuation-worker.v1';
    absent := NEW.result_corpus_id IS NULL
        AND NEW.result_generation_id IS NULL
        AND NEW.result_generation_version IS NULL
        AND NEW.result_manifest_hash IS NULL
        AND NEW.result_provenance_fingerprint IS NULL
        AND NEW.result_worker_contract_version IS NULL
        AND NEW.result_published_at IS NULL;
    IF NOT ((NEW.state = 'succeeded' AND complete)
            OR (NEW.state <> 'succeeded' AND absent)) THEN
        RAISE EXCEPTION 'baseline_control_continuation_result_invalid';
    END IF;
    IF TG_OP = 'UPDATE' AND OLD.state = 'succeeded' AND (
        NEW.state IS DISTINCT FROM OLD.state
        OR NEW.result_corpus_id IS DISTINCT FROM OLD.result_corpus_id
        OR NEW.result_generation_id IS DISTINCT FROM OLD.result_generation_id
        OR NEW.result_generation_version IS DISTINCT FROM OLD.result_generation_version
        OR NEW.result_manifest_hash IS DISTINCT FROM OLD.result_manifest_hash
        OR NEW.result_provenance_fingerprint IS DISTINCT FROM OLD.result_provenance_fingerprint
        OR NEW.result_worker_contract_version IS DISTINCT FROM OLD.result_worker_contract_version
        OR NEW.result_published_at IS DISTINCT FROM OLD.result_published_at
    ) THEN
        RAISE EXCEPTION 'baseline_control_continuation_result_immutable';
    END IF;
    RETURN NEW;
END
$$
"""
    )
    connection.exec_driver_sql(
        f"DROP TRIGGER IF EXISTS {_CONTROL_CONTINUATION_RESULT_INSERT_TRIGGER} "
        f"ON {SNAPSHOT_CONTINUATION_JOB_TABLE}"
    )
    connection.exec_driver_sql(
        f"DROP TRIGGER IF EXISTS {_CONTROL_CONTINUATION_RESULT_UPDATE_TRIGGER} "
        f"ON {SNAPSHOT_CONTINUATION_JOB_TABLE}"
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER {_CONTROL_CONTINUATION_RESULT_INSERT_TRIGGER} BEFORE INSERT "
        f"ON {SNAPSHOT_CONTINUATION_JOB_TABLE} FOR EACH ROW EXECUTE FUNCTION "
        f"{_CONTROL_CONTINUATION_RESULT_FUNCTION}()"
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER {_CONTROL_CONTINUATION_RESULT_UPDATE_TRIGGER} BEFORE UPDATE "
        f"ON {SNAPSHOT_CONTINUATION_JOB_TABLE} FOR EACH ROW EXECUTE FUNCTION "
        f"{_CONTROL_CONTINUATION_RESULT_FUNCTION}()"
    )
    connection.exec_driver_sql(
        f"""
CREATE OR REPLACE FUNCTION {_CONTROL_SEALED_PART_FUNCTION}()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    target_staging_id varchar(36);
    target_group_id varchar(36);
BEGIN
    target_staging_id := CASE WHEN TG_OP = 'DELETE' THEN OLD.staging_id ELSE NEW.staging_id END;
    target_group_id := CASE WHEN TG_OP = 'DELETE' THEN OLD.group_id ELSE NEW.group_id END;
    IF EXISTS (
        SELECT 1 FROM {SNAPSHOT_STAGING_TABLE}
        WHERE staging_id = target_staging_id
          AND group_id = target_group_id
          AND status = 'sealed'
    ) THEN
        RAISE EXCEPTION 'baseline_control_sealed_part_immutable';
    END IF;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END
$$
"""
    )
    for trigger_name, operation in (
        (_CONTROL_SEALED_PART_INSERT_TRIGGER, "INSERT"),
        (_CONTROL_SEALED_PART_DELETE_TRIGGER, "DELETE"),
    ):
        connection.exec_driver_sql(
            f"DROP TRIGGER IF EXISTS {trigger_name} ON {SNAPSHOT_CONTENT_PART_TABLE}"
        )
        connection.exec_driver_sql(
            f"CREATE TRIGGER {trigger_name} BEFORE {operation} "
            f"ON {SNAPSHOT_CONTENT_PART_TABLE} FOR EACH ROW EXECUTE FUNCTION "
            f"{_CONTROL_SEALED_PART_FUNCTION}()"
        )


def _upgrade_baseline_control_plane_ingestion_worker(connection: Connection) -> None:
    existing = _column_names(connection, SNAPSHOT_CONTINUATION_JOB_TABLE)
    for column_name, ddl in _CONTINUATION_RESULT_COLUMNS:
        if column_name not in existing:
            connection.exec_driver_sql(
                f"ALTER TABLE {SNAPSHOT_CONTINUATION_JOB_TABLE} "
                f"ADD COLUMN {column_name} {ddl}"
            )
    connection.exec_driver_sql(
        "CREATE INDEX IF NOT EXISTS ix_bl_ctl_continuation_result_generation "
        f"ON {SNAPSHOT_CONTINUATION_JOB_TABLE}(result_generation_id)"
    )
    if connection.dialect.name == "sqlite":
        _create_sqlite_continuation_ingestion_triggers(connection)
    else:
        _create_postgres_continuation_ingestion_triggers(connection)


def _validate_baseline_control_plane_ingestion_worker(connection: Connection) -> None:
    required_columns = {name for name, _ddl in _CONTINUATION_RESULT_COLUMNS}
    if not required_columns <= _column_names(
        connection, SNAPSHOT_CONTINUATION_JOB_TABLE
    ):
        raise SchemaInvariantError("baseline_continuation_worker_columns_missing")
    if "ix_bl_ctl_continuation_result_generation" not in _index_names(
        connection, SNAPSHOT_CONTINUATION_JOB_TABLE
    ):
        raise SchemaInvariantError("baseline_continuation_worker_index_missing")
    if connection.dialect.name == "sqlite":
        triggers = {
            str(row[0])
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).all()
        }
    else:
        triggers = {
            str(row[0])
            for row in connection.execute(
                text(
                    "SELECT tgname FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
                    "WHERE NOT t.tgisinternal AND c.relname IN "
                    "(:continuation, :part)"
                ),
                {
                    "continuation": SNAPSHOT_CONTINUATION_JOB_TABLE,
                    "part": SNAPSHOT_CONTENT_PART_TABLE,
                },
            ).all()
        }
    required_triggers = {
        _CONTROL_CONTINUATION_RESULT_INSERT_TRIGGER,
        _CONTROL_CONTINUATION_RESULT_UPDATE_TRIGGER,
        _CONTROL_SEALED_PART_INSERT_TRIGGER,
        _CONTROL_SEALED_PART_DELETE_TRIGGER,
    }
    if not required_triggers <= triggers:
        raise SchemaInvariantError("baseline_continuation_worker_trigger_missing")


_INDEX_JOB_IMMUTABLE_COLUMNS = (
    "job_id",
    "group_id",
    "continuation_job_id",
    "submitted_by_user_id",
    "contract_version",
    "index_intent_hash",
    "snapshot_id",
    "corpus_id",
    "generation_id",
    "generation_version",
    "control_manifest_hash",
    "corpus_manifest_hash",
    "corpus_file_manifest_hash",
    "ingestion_provenance_fingerprint",
    "index_format_version",
    "tokenizer_version",
    "retrieval_config_fingerprint",
    "embedding_contract_version",
    "embedding_provider",
    "embedding_model",
    "embedding_revision",
    "embedding_dimension",
    "embedding_dtype",
    "embedding_fingerprint",
    "created_at",
)
_INDEX_JOB_RESULT_COLUMNS = (
    "result_index_id",
    "result_document_count",
    "result_total_token_count",
    "result_document_manifest_hash",
    "result_lexical_manifest_hash",
    "result_dense_manifest_hash",
    "result_published_at",
)


def _create_sqlite_index_job_triggers(connection: Connection) -> None:
    immutable = " OR ".join(
        f"NEW.{column} IS NOT OLD.{column}" for column in _INDEX_JOB_IMMUTABLE_COLUMNS
    )
    result_changed = " OR ".join(
        f"NEW.{column} IS NOT OLD.{column}" for column in _INDEX_JOB_RESULT_COLUMNS
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER IF NOT EXISTS {_CONTROL_INDEX_JOB_IMMUTABLE_TRIGGER} "
        f"BEFORE UPDATE ON {COMPATIBLE_INDEX_JOB_TABLE} FOR EACH ROW WHEN "
        f"({immutable}) OR (OLD.result_index_id IS NOT NULL AND ({result_changed})) "
        "OR (OLD.result_index_id IS NULL AND NEW.result_index_id IS NOT NULL AND "
        f"NOT EXISTS (SELECT 1 FROM {CONTROL_JOB_TABLE} j "
        "WHERE j.job_id = OLD.job_id AND j.group_id = OLD.group_id "
        "AND j.operation = 'index_build' AND j.state = 'running')) "
        "BEGIN SELECT RAISE(ABORT, 'baseline_index_job_immutable'); END"
    )
    complete = (
        f"EXISTS (SELECT 1 FROM {COMPATIBLE_INDEX_JOB_TABLE} x "
        "WHERE x.job_id = NEW.job_id AND x.group_id = NEW.group_id "
        "AND x.result_index_id IS NOT NULL "
        "AND x.result_document_count IS NOT NULL "
        "AND x.result_total_token_count IS NOT NULL "
        "AND x.result_document_manifest_hash IS NOT NULL "
        "AND x.result_lexical_manifest_hash IS NOT NULL "
        "AND x.result_dense_manifest_hash IS NOT NULL "
        "AND x.result_published_at IS NOT NULL)"
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER IF NOT EXISTS {_CONTROL_INDEX_JOB_STATE_TRIGGER} "
        f"BEFORE UPDATE OF state ON {CONTROL_JOB_TABLE} FOR EACH ROW "
        "WHEN NEW.operation = 'index_build' AND ("
        f"(NEW.state = 'succeeded' AND NOT ({complete})) OR "
        f"(NEW.state <> 'succeeded' AND ({complete})) OR "
        "(OLD.state = 'succeeded' AND NEW.state <> OLD.state)) "
        "BEGIN SELECT RAISE(ABORT, 'baseline_index_job_result_invalid'); END"
    )


def _create_postgres_index_job_triggers(connection: Connection) -> None:
    immutable = "\n       OR ".join(
        f"NEW.{column} IS DISTINCT FROM OLD.{column}"
        for column in _INDEX_JOB_IMMUTABLE_COLUMNS
    )
    result_changed = "\n       OR ".join(
        f"NEW.{column} IS DISTINCT FROM OLD.{column}"
        for column in _INDEX_JOB_RESULT_COLUMNS
    )
    connection.exec_driver_sql(
        f"""
CREATE OR REPLACE FUNCTION {_CONTROL_INDEX_JOB_IMMUTABLE_FUNCTION}()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF {immutable}
       OR (OLD.result_index_id IS NOT NULL AND ({result_changed}))
       OR (OLD.result_index_id IS NULL AND NEW.result_index_id IS NOT NULL
           AND NOT EXISTS (
               SELECT 1 FROM {CONTROL_JOB_TABLE} j
               WHERE j.job_id = OLD.job_id AND j.group_id = OLD.group_id
                 AND j.operation = 'index_build' AND j.state = 'running'
           )) THEN
        RAISE EXCEPTION 'baseline_index_job_immutable';
    END IF;
    RETURN NEW;
END
$$
"""
    )
    connection.exec_driver_sql(
        f"DROP TRIGGER IF EXISTS {_CONTROL_INDEX_JOB_IMMUTABLE_TRIGGER} "
        f"ON {COMPATIBLE_INDEX_JOB_TABLE}"
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER {_CONTROL_INDEX_JOB_IMMUTABLE_TRIGGER} BEFORE UPDATE "
        f"ON {COMPATIBLE_INDEX_JOB_TABLE} FOR EACH ROW EXECUTE FUNCTION "
        f"{_CONTROL_INDEX_JOB_IMMUTABLE_FUNCTION}()"
    )
    connection.exec_driver_sql(
        f"""
CREATE OR REPLACE FUNCTION {_CONTROL_INDEX_JOB_STATE_FUNCTION}()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    complete boolean;
BEGIN
    IF NEW.operation <> 'index_build' THEN
        RETURN NEW;
    END IF;
    complete := EXISTS (
        SELECT 1 FROM {COMPATIBLE_INDEX_JOB_TABLE} x
        WHERE x.job_id = NEW.job_id AND x.group_id = NEW.group_id
          AND x.result_index_id IS NOT NULL
          AND x.result_document_count IS NOT NULL
          AND x.result_total_token_count IS NOT NULL
          AND x.result_document_manifest_hash IS NOT NULL
          AND x.result_lexical_manifest_hash IS NOT NULL
          AND x.result_dense_manifest_hash IS NOT NULL
          AND x.result_published_at IS NOT NULL
    );
    IF (NEW.state = 'succeeded' AND NOT complete)
       OR (NEW.state <> 'succeeded' AND complete)
       OR (OLD.state = 'succeeded' AND NEW.state IS DISTINCT FROM OLD.state) THEN
        RAISE EXCEPTION 'baseline_index_job_result_invalid';
    END IF;
    RETURN NEW;
END
$$
"""
    )
    connection.exec_driver_sql(
        f"DROP TRIGGER IF EXISTS {_CONTROL_INDEX_JOB_STATE_TRIGGER} "
        f"ON {CONTROL_JOB_TABLE}"
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER {_CONTROL_INDEX_JOB_STATE_TRIGGER} BEFORE UPDATE OF state "
        f"ON {CONTROL_JOB_TABLE} FOR EACH ROW EXECUTE FUNCTION "
        f"{_CONTROL_INDEX_JOB_STATE_FUNCTION}()"
    )


def _upgrade_baseline_compatible_index_job(connection: Connection) -> None:
    for table in INDEX_CONTROL_PLANE_TABLES:
        connection.execute(CreateTable(table, if_not_exists=True))
        for index in sorted(table.indexes, key=lambda candidate: candidate.name or ""):
            connection.execute(CreateIndex(index, if_not_exists=True))
    if connection.dialect.name == "sqlite":
        _create_sqlite_index_job_triggers(connection)
    else:
        _create_postgres_index_job_triggers(connection)


def _validate_baseline_compatible_index_job(connection: Connection) -> None:
    if COMPATIBLE_INDEX_JOB_TABLE not in set(inspect(connection).get_table_names()):
        raise SchemaInvariantError("baseline_index_job_table_missing")
    required_columns = {column.name for column in INDEX_CONTROL_PLANE_TABLES[0].columns}
    if not required_columns <= _column_names(connection, COMPATIBLE_INDEX_JOB_TABLE):
        raise SchemaInvariantError("baseline_index_job_column_missing")
    required_fks = {
        (
            ("job_id", "group_id"),
            CONTROL_JOB_TABLE,
            ("job_id", "group_id"),
            "CASCADE",
        ),
        (
            ("continuation_job_id", "group_id"),
            SNAPSHOT_CONTINUATION_JOB_TABLE,
            ("continuation_job_id", "group_id"),
            "CASCADE",
        ),
        (("submitted_by_user_id",), "user", ("user_id",), "SET NULL"),
    }
    if not required_fks <= _foreign_key_targets(connection, COMPATIBLE_INDEX_JOB_TABLE):
        raise SchemaInvariantError("baseline_index_job_foreign_key_missing")
    if not {
        "uq_bl_idx_job_scope",
        "uq_bl_idx_job_generation_intent",
    } <= _unique_constraint_names(connection, COMPATIBLE_INDEX_JOB_TABLE):
        raise SchemaInvariantError("baseline_index_job_unique_missing")
    if not {"ck_bl_idx_job_contract", "ck_bl_idx_job_result"} <= _constraint_names(
        connection, COMPATIBLE_INDEX_JOB_TABLE, "check"
    ):
        raise SchemaInvariantError("baseline_index_job_check_missing")
    if not {
        "ix_bl_idx_job_continuation",
        "ix_bl_idx_job_generation",
        "ix_bl_idx_job_result",
    } <= _index_names(connection, COMPATIBLE_INDEX_JOB_TABLE):
        raise SchemaInvariantError("baseline_index_job_index_missing")
    if connection.dialect.name == "sqlite":
        triggers = {
            str(row[0])
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).all()
        }
    else:
        triggers = {
            str(row[0])
            for row in connection.execute(
                text(
                    "SELECT tgname FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
                    "WHERE NOT t.tgisinternal AND c.relname IN (:job, :extension)"
                ),
                {"job": CONTROL_JOB_TABLE, "extension": COMPATIBLE_INDEX_JOB_TABLE},
            ).all()
        }
    if (
        not {
            _CONTROL_INDEX_JOB_IMMUTABLE_TRIGGER,
            _CONTROL_INDEX_JOB_STATE_TRIGGER,
        }
        <= triggers
    ):
        raise SchemaInvariantError("baseline_index_job_trigger_missing")


def _upgrade_baseline_run_job(connection: Connection) -> None:
    for table in RUN_CONTROL_PLANE_TABLES:
        connection.execute(CreateTable(table, if_not_exists=True))
        for index in sorted(table.indexes, key=lambda candidate: candidate.name or ""):
            connection.execute(CreateIndex(index, if_not_exists=True))


def _validate_baseline_run_job(connection: Connection) -> None:
    table_names = set(inspect(connection).get_table_names())
    if {BASELINE_RUN_JOB_TABLE, BASELINE_RUN_PAYLOAD_TABLE} - table_names:
        raise SchemaInvariantError("baseline_run_job_table_missing")
    executor_columns = {
        "retrieval_result_fingerprint",
        "started_at",
        "worker_contract_version",
        "worker_service_id",
    }
    for table in RUN_CONTROL_PLANE_TABLES:
        expected_columns = {column.name for column in table.columns}
        if table.name == BASELINE_RUN_JOB_TABLE:
            expected_columns -= executor_columns
        actual_columns = _column_names(connection, table.name)
        if not expected_columns <= actual_columns or not actual_columns <= (
            expected_columns | executor_columns
        ):
            raise SchemaInvariantError(f"baseline_run_job_column_missing:{table.name}")
    job_fks = _foreign_key_targets(connection, BASELINE_RUN_JOB_TABLE)
    required_job_fks = {
        (("group_id",), "group", ("group_id",), "CASCADE"),
        (("submitted_by_user_id",), "user", ("user_id",), "SET NULL"),
        (("source_document_id",), "document", ("document_id",), "SET NULL"),
    }
    if not required_job_fks <= job_fks:
        raise SchemaInvariantError("baseline_run_job_foreign_key_missing")
    payload_fks = _foreign_key_targets(connection, BASELINE_RUN_PAYLOAD_TABLE)
    if (
        ("job_id", "group_id"),
        BASELINE_RUN_JOB_TABLE,
        ("job_id", "group_id"),
        "CASCADE",
    ) not in payload_fks:
        raise SchemaInvariantError("baseline_run_payload_foreign_key_missing")
    if not {"uq_bl_run_job_group_idempotency", "uq_bl_run_job_scope"} <= (
        _unique_constraint_names(connection, BASELINE_RUN_JOB_TABLE)
    ):
        raise SchemaInvariantError("baseline_run_job_unique_missing")
    if "uq_bl_run_payload_key_nonce" not in _unique_constraint_names(
        connection, BASELINE_RUN_PAYLOAD_TABLE
    ):
        raise SchemaInvariantError("baseline_run_payload_unique_missing")
    if not {
        "ck_bl_run_job_contract",
        "ck_bl_run_job_query",
        "ck_bl_run_job_index",
        "ck_bl_run_job_state",
        "ck_bl_run_job_lease",
        "ck_bl_run_job_counts",
    } <= _constraint_names(connection, BASELINE_RUN_JOB_TABLE, "check"):
        raise SchemaInvariantError("baseline_run_job_check_missing")
    if "ck_bl_run_payload_contract" not in _constraint_names(
        connection, BASELINE_RUN_PAYLOAD_TABLE, "check"
    ):
        raise SchemaInvariantError("baseline_run_payload_check_missing")
    if not {
        "ix_bl_run_job_group_state",
        "ix_bl_run_job_expiry",
        "ix_bl_run_job_source",
        "ix_bl_run_job_publication",
    } <= _index_names(connection, BASELINE_RUN_JOB_TABLE):
        raise SchemaInvariantError("baseline_run_job_index_missing")
    if "ix_bl_run_payload_expiry" not in _index_names(
        connection, BASELINE_RUN_PAYLOAD_TABLE
    ):
        raise SchemaInvariantError("baseline_run_payload_index_missing")


_SOURCE_SCOPE_RUN_INSERT_TRIGGER = "trg_bl_run_source_scope_insert_v1"
_SOURCE_SCOPE_RUN_UPDATE_TRIGGER = "trg_bl_run_source_scope_update_v1"
_SOURCE_SCOPE_JOB_LINK_TRIGGER = "trg_bl_run_job_persisted_link_v1"


def _sqlite_document_scope_job_sql(
    connection: Connection,
    temporary_name: str,
) -> str:
    sql = _sqlite_retarget_table(
        _sqlite_create_sql(connection, BASELINE_RUN_JOB_TABLE),
        BASELINE_RUN_JOB_TABLE,
        temporary_name,
    )
    if "uq_bl_run_job_persisted_run" not in sql:
        sql = _sqlite_append_constraint(
            sql,
            "CONSTRAINT uq_bl_run_job_persisted_run UNIQUE (persisted_run_id)",
            BASELINE_RUN_JOB_TABLE,
        )
    if "fk_bl_run_job_persisted_run_scope" not in sql:
        sql = _sqlite_append_constraint(
            sql,
            "CONSTRAINT fk_bl_run_job_persisted_run_scope "
            "FOREIGN KEY (persisted_run_id, group_id) "
            "REFERENCES baseline_retrieval_run(run_id, group_id) "
            "ON DELETE NO ACTION DEFERRABLE INITIALLY DEFERRED",
            BASELINE_RUN_JOB_TABLE,
        )
    return sql


def _sqlite_rebuild_document_scope_job(connection: Connection) -> None:
    required_fk = (
        ("persisted_run_id", "group_id"),
        BASELINE_RETRIEVAL_RUN_TABLE,
        ("run_id", "group_id"),
        "NO ACTION",
    )
    if required_fk in _foreign_key_targets(
        connection, BASELINE_RUN_JOB_TABLE
    ) and "uq_bl_run_job_persisted_run" in _unique_constraint_names(
        connection, BASELINE_RUN_JOB_TABLE
    ):
        return
    preserved = tuple(
        str(value)
        for value in connection.execute(
            text(
                "SELECT sql FROM sqlite_master WHERE tbl_name = :table_name "
                "AND type IN ('index', 'trigger') AND sql IS NOT NULL "
                "ORDER BY type, name"
            ),
            {"table_name": BASELINE_RUN_JOB_TABLE},
        ).scalars()
    )
    temporary_name = "__document_scope_v1_baseline_control_run_job"
    connection.exec_driver_sql(
        _sqlite_document_scope_job_sql(connection, temporary_name)
    )
    _sqlite_copy_table(connection, BASELINE_RUN_JOB_TABLE, temporary_name)
    connection.exec_driver_sql(f'DROP TABLE "{BASELINE_RUN_JOB_TABLE}"')
    connection.exec_driver_sql(
        f'ALTER TABLE "{temporary_name}" RENAME TO "{BASELINE_RUN_JOB_TABLE}"'
    )
    for ddl in preserved:
        connection.exec_driver_sql(ddl)


def _create_sqlite_document_scope_triggers(connection: Connection) -> None:
    for trigger_name in (
        _SOURCE_SCOPE_RUN_INSERT_TRIGGER,
        _SOURCE_SCOPE_RUN_UPDATE_TRIGGER,
        _SOURCE_SCOPE_JOB_LINK_TRIGGER,
    ):
        connection.exec_driver_sql(f"DROP TRIGGER IF EXISTS {trigger_name}")
    connection.exec_driver_sql(
        f"CREATE TRIGGER {_SOURCE_SCOPE_RUN_INSERT_TRIGGER} "
        "BEFORE INSERT ON baseline_retrieval_run WHEN "
        "NEW.source_scope_version <> 'baseline-source-scope.v1' OR "
        "NEW.source_scope NOT IN ('legacy_chunk', 'control_document') OR "
        "(NEW.source_scope = 'legacy_chunk' AND "
        "(NEW.source_chunk_id IS NULL OR NEW.source_document_id IS NULL)) OR "
        "(NEW.source_scope = 'control_document' AND "
        "(NEW.source_chunk_id IS NOT NULL OR NEW.source_document_id IS NULL)) OR "
        "NEW.evidence_character_count NOT BETWEEN 1 AND 16000 "
        "BEGIN SELECT RAISE(ABORT, 'baseline_source_scope_invalid'); END"
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER {_SOURCE_SCOPE_RUN_UPDATE_TRIGGER} "
        "BEFORE UPDATE OF source_scope_version, source_scope, source_chunk_id, "
        "source_document_id, evidence_character_count ON baseline_retrieval_run WHEN "
        "NEW.source_scope_version <> OLD.source_scope_version OR "
        "NEW.source_scope <> OLD.source_scope OR "
        "(OLD.source_chunk_id IS NULL AND NEW.source_chunk_id IS NOT NULL) OR "
        "(OLD.source_chunk_id IS NOT NULL AND NEW.source_chunk_id IS NOT NULL "
        "AND NEW.source_chunk_id <> OLD.source_chunk_id) OR "
        "(OLD.source_document_id IS NULL AND NEW.source_document_id IS NOT NULL) OR "
        "(OLD.source_document_id IS NOT NULL AND NEW.source_document_id IS NOT NULL "
        "AND NEW.source_document_id <> OLD.source_document_id) OR "
        "(NEW.source_scope = 'control_document' AND NEW.source_chunk_id IS NOT NULL) OR "
        "NEW.evidence_character_count NOT BETWEEN 1 AND 16000 "
        "BEGIN SELECT RAISE(ABORT, 'baseline_source_scope_immutable'); END"
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER {_SOURCE_SCOPE_JOB_LINK_TRIGGER} "
        "BEFORE UPDATE OF persisted_run_id, state, evidence_count, reference_count "
        "ON baseline_control_run_job BEGIN "
        "SELECT CASE WHEN NEW.persisted_run_id IS NOT NULL AND NOT EXISTS ("
        "SELECT 1 FROM baseline_retrieval_run r WHERE "
        "r.run_id = NEW.persisted_run_id AND r.group_id = NEW.group_id "
        "AND r.source_scope_version = 'baseline-source-scope.v1' "
        "AND r.source_scope = 'control_document') "
        "THEN RAISE(ABORT, 'baseline_control_run_link_invalid') END; "
        "SELECT CASE WHEN OLD.persisted_run_id IS NULL "
        "AND NEW.persisted_run_id IS NOT NULL AND NOT ("
        "OLD.state = 'running' AND NEW.state = 'references_persisted' "
        "AND OLD.source_document_id IS NOT NULL "
        "AND OLD.source_document_id = (SELECT r.source_document_id "
        "FROM baseline_retrieval_run r WHERE r.run_id = NEW.persisted_run_id) "
        "AND NEW.evidence_count BETWEEN 1 AND 4 "
        "AND NEW.reference_count = NEW.evidence_count) "
        "THEN RAISE(ABORT, 'baseline_control_run_attachment_invalid') END; "
        "SELECT CASE WHEN OLD.persisted_run_id IS NOT NULL "
        "AND NEW.persisted_run_id IS NOT OLD.persisted_run_id "
        "THEN RAISE(ABORT, 'baseline_control_run_link_immutable') END; "
        "SELECT CASE WHEN NEW.state IN ('references_persisted', 'feedback_persisted') "
        "AND NEW.persisted_run_id IS NULL "
        "THEN RAISE(ABORT, 'baseline_control_run_link_required') END; END"
    )


def _upgrade_sqlite_baseline_document_source_scope(connection: Connection) -> None:
    run_columns = _column_names(connection, BASELINE_RETRIEVAL_RUN_TABLE)
    if "source_scope_version" not in run_columns:
        connection.exec_driver_sql(
            "ALTER TABLE baseline_retrieval_run ADD COLUMN source_scope_version "
            "VARCHAR(64) NOT NULL DEFAULT 'baseline-source-scope.v1'"
        )
    if "source_scope" not in run_columns:
        connection.exec_driver_sql(
            "ALTER TABLE baseline_retrieval_run ADD COLUMN source_scope "
            "VARCHAR(32) NOT NULL DEFAULT 'legacy_chunk'"
        )
    connection.exec_driver_sql(
        "UPDATE baseline_retrieval_run SET "
        "source_scope_version = 'baseline-source-scope.v1', "
        "source_scope = 'legacy_chunk' WHERE source_scope_version IS NULL "
        "OR source_scope IS NULL"
    )
    _sqlite_rebuild_document_scope_job(connection)
    _create_sqlite_document_scope_triggers(connection)


def _upgrade_postgres_baseline_document_source_scope(connection: Connection) -> None:
    run_columns = _column_names(connection, BASELINE_RETRIEVAL_RUN_TABLE)
    if "source_scope_version" not in run_columns:
        connection.exec_driver_sql(
            "ALTER TABLE baseline_retrieval_run ADD COLUMN source_scope_version "
            "VARCHAR(64) NOT NULL DEFAULT 'baseline-source-scope.v1'"
        )
    if "source_scope" not in run_columns:
        connection.exec_driver_sql(
            "ALTER TABLE baseline_retrieval_run ADD COLUMN source_scope "
            "VARCHAR(32) NOT NULL DEFAULT 'legacy_chunk'"
        )
    connection.exec_driver_sql(
        "ALTER TABLE baseline_retrieval_run "
        "DROP CONSTRAINT IF EXISTS ck_bl_run_source_scope"
    )
    connection.exec_driver_sql(
        "ALTER TABLE baseline_retrieval_run ADD CONSTRAINT ck_bl_run_source_scope "
        "CHECK (source_scope_version = 'baseline-source-scope.v1' AND "
        "source_scope IN ('legacy_chunk', 'control_document'))"
    )
    connection.exec_driver_sql(
        "ALTER TABLE baseline_retrieval_run DROP CONSTRAINT IF EXISTS ck_bl_run_counts"
    )
    connection.exec_driver_sql(
        "ALTER TABLE baseline_retrieval_run ADD CONSTRAINT ck_bl_run_counts "
        "CHECK (candidate_count >= 0 AND retrieved_count >= 0 "
        "AND filtered_count >= 0 AND duplicate_count >= 0 AND refill_count >= 0 "
        "AND selected_count BETWEEN 1 AND 4 "
        "AND evidence_character_count BETWEEN 1 AND 16000)"
    )
    if "uq_bl_run_job_persisted_run" not in _unique_constraint_names(
        connection, BASELINE_RUN_JOB_TABLE
    ):
        connection.exec_driver_sql(
            "ALTER TABLE baseline_control_run_job ADD CONSTRAINT "
            "uq_bl_run_job_persisted_run UNIQUE (persisted_run_id)"
        )
    _postgres_add_constraint(
        connection,
        table_name=BASELINE_RUN_JOB_TABLE,
        constraint_name="fk_bl_run_job_persisted_run_scope",
        definition=(
            "FOREIGN KEY (persisted_run_id, group_id) REFERENCES "
            "baseline_retrieval_run(run_id, group_id) ON DELETE NO ACTION "
            "DEFERRABLE INITIALLY DEFERRED"
        ),
        kind="foreign_key",
    )
    connection.exec_driver_sql(
        "CREATE OR REPLACE FUNCTION compair_bl_source_scope_guard_v1() "
        "RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
        "IF TG_OP = 'INSERT' THEN "
        "IF NEW.source_scope_version <> 'baseline-source-scope.v1' OR "
        "NEW.source_scope NOT IN ('legacy_chunk', 'control_document') OR "
        "(NEW.source_scope = 'legacy_chunk' AND "
        "(NEW.source_chunk_id IS NULL OR NEW.source_document_id IS NULL)) OR "
        "(NEW.source_scope = 'control_document' AND "
        "(NEW.source_chunk_id IS NOT NULL OR NEW.source_document_id IS NULL)) THEN "
        "RAISE EXCEPTION 'baseline_source_scope_invalid'; END IF; "
        "ELSE IF NEW.source_scope_version <> OLD.source_scope_version OR "
        "NEW.source_scope <> OLD.source_scope OR "
        "(OLD.source_chunk_id IS NULL AND NEW.source_chunk_id IS NOT NULL) OR "
        "(OLD.source_chunk_id IS NOT NULL AND NEW.source_chunk_id IS NOT NULL "
        "AND NEW.source_chunk_id <> OLD.source_chunk_id) OR "
        "(OLD.source_document_id IS NULL AND NEW.source_document_id IS NOT NULL) OR "
        "(OLD.source_document_id IS NOT NULL AND NEW.source_document_id IS NOT NULL "
        "AND NEW.source_document_id <> OLD.source_document_id) OR "
        "(NEW.source_scope = 'control_document' AND NEW.source_chunk_id IS NOT NULL) "
        "THEN RAISE EXCEPTION 'baseline_source_scope_immutable'; END IF; END IF; "
        "RETURN NEW; END $$"
    )
    connection.exec_driver_sql(
        f"DROP TRIGGER IF EXISTS {_SOURCE_SCOPE_RUN_INSERT_TRIGGER} "
        "ON baseline_retrieval_run"
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER {_SOURCE_SCOPE_RUN_INSERT_TRIGGER} BEFORE INSERT "
        "ON baseline_retrieval_run FOR EACH ROW EXECUTE FUNCTION "
        "compair_bl_source_scope_guard_v1()"
    )
    connection.exec_driver_sql(
        f"DROP TRIGGER IF EXISTS {_SOURCE_SCOPE_RUN_UPDATE_TRIGGER} "
        "ON baseline_retrieval_run"
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER {_SOURCE_SCOPE_RUN_UPDATE_TRIGGER} BEFORE UPDATE OF "
        "source_scope_version, source_scope, source_chunk_id, source_document_id "
        "ON baseline_retrieval_run FOR EACH ROW EXECUTE FUNCTION "
        "compair_bl_source_scope_guard_v1()"
    )
    connection.exec_driver_sql(
        "CREATE OR REPLACE FUNCTION compair_bl_run_job_link_guard_v1() "
        "RETURNS trigger LANGUAGE plpgsql AS $$ DECLARE linked_document VARCHAR(36); "
        "BEGIN IF NEW.persisted_run_id IS NOT NULL THEN "
        "SELECT source_document_id INTO linked_document FROM baseline_retrieval_run "
        "WHERE run_id = NEW.persisted_run_id AND group_id = NEW.group_id "
        "AND source_scope_version = 'baseline-source-scope.v1' "
        "AND source_scope = 'control_document'; "
        "IF NOT FOUND THEN RAISE EXCEPTION 'baseline_control_run_link_invalid'; END IF; "
        "END IF; IF OLD.persisted_run_id IS NULL AND NEW.persisted_run_id IS NOT NULL "
        "AND NOT (OLD.state = 'running' AND NEW.state = 'references_persisted' "
        "AND OLD.source_document_id IS NOT NULL "
        "AND OLD.source_document_id = linked_document "
        "AND NEW.evidence_count BETWEEN 1 AND 4 "
        "AND NEW.reference_count = NEW.evidence_count) THEN "
        "RAISE EXCEPTION 'baseline_control_run_attachment_invalid'; END IF; "
        "IF OLD.persisted_run_id IS NOT NULL "
        "AND NEW.persisted_run_id IS DISTINCT FROM OLD.persisted_run_id THEN "
        "RAISE EXCEPTION 'baseline_control_run_link_immutable'; END IF; "
        "IF NEW.state IN ('references_persisted', 'feedback_persisted') "
        "AND NEW.persisted_run_id IS NULL THEN "
        "RAISE EXCEPTION 'baseline_control_run_link_required'; END IF; "
        "RETURN NEW; END $$"
    )
    connection.exec_driver_sql(
        f"DROP TRIGGER IF EXISTS {_SOURCE_SCOPE_JOB_LINK_TRIGGER} "
        "ON baseline_control_run_job"
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER {_SOURCE_SCOPE_JOB_LINK_TRIGGER} BEFORE UPDATE OF "
        "persisted_run_id, state, evidence_count, reference_count "
        "ON baseline_control_run_job FOR EACH ROW EXECUTE FUNCTION "
        "compair_bl_run_job_link_guard_v1()"
    )


def _upgrade_baseline_document_source_scope(connection: Connection) -> None:
    if connection.dialect.name == "sqlite":
        _upgrade_sqlite_baseline_document_source_scope(connection)
    else:
        _upgrade_postgres_baseline_document_source_scope(connection)


def _validate_baseline_document_source_scope(connection: Connection) -> None:
    required_columns = {"source_scope_version", "source_scope"}
    if not required_columns <= _column_names(connection, BASELINE_RETRIEVAL_RUN_TABLE):
        raise SchemaInvariantError("baseline_source_scope_columns_missing")
    invalid = connection.execute(
        text(
            "SELECT count(*) FROM baseline_retrieval_run WHERE "
            "source_scope_version <> :version OR source_scope NOT IN "
            "(:legacy_scope, :control_scope)"
        ),
        {
            "version": SOURCE_SCOPE_VERSION,
            "legacy_scope": SOURCE_SCOPE_LEGACY_CHUNK,
            "control_scope": SOURCE_SCOPE_CONTROL_DOCUMENT,
        },
    ).scalar_one()
    if int(invalid) != 0:
        raise SchemaInvariantError("baseline_source_scope_row_invalid")
    required_fk = (
        ("persisted_run_id", "group_id"),
        BASELINE_RETRIEVAL_RUN_TABLE,
        ("run_id", "group_id"),
        "NO ACTION",
    )
    if required_fk not in _foreign_key_targets(connection, BASELINE_RUN_JOB_TABLE):
        raise SchemaInvariantError("baseline_control_run_relationship_fk_missing")
    if "uq_bl_run_job_persisted_run" not in _unique_constraint_names(
        connection, BASELINE_RUN_JOB_TABLE
    ):
        raise SchemaInvariantError("baseline_control_run_relationship_unique_missing")
    linked_invalid = connection.execute(
        text(
            "SELECT count(*) FROM baseline_control_run_job j LEFT JOIN "
            "baseline_retrieval_run r ON r.run_id = j.persisted_run_id "
            "AND r.group_id = j.group_id WHERE j.persisted_run_id IS NOT NULL "
            "AND (r.run_id IS NULL OR r.source_scope <> 'control_document')"
        )
    ).scalar_one()
    if int(linked_invalid) != 0:
        raise SchemaInvariantError("baseline_control_run_relationship_invalid")
    if connection.dialect.name == "sqlite":
        triggers = {
            str(row[0])
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).all()
        }
    else:
        triggers = {
            str(row[0])
            for row in connection.execute(
                text(
                    "SELECT tgname FROM pg_trigger WHERE NOT tgisinternal AND "
                    "tgrelid IN ('baseline_retrieval_run'::regclass, "
                    "'baseline_control_run_job'::regclass)"
                )
            ).all()
        }
    required_triggers = {
        _SOURCE_SCOPE_RUN_INSERT_TRIGGER,
        _SOURCE_SCOPE_RUN_UPDATE_TRIGGER,
        _SOURCE_SCOPE_JOB_LINK_TRIGGER,
    }
    if not required_triggers <= triggers:
        raise SchemaInvariantError("baseline_source_scope_trigger_missing")


_RUN_EXECUTOR_INSERT_TRIGGER = "trg_bl_run_executor_insert_v1"
_RUN_EXECUTOR_METADATA_TRIGGER = "trg_bl_run_executor_metadata_v1"
_CONTROL_GENERATION_INSERT_TRIGGER = "trg_bl_control_generation_insert_v1"
_CONTROL_GENERATION_METADATA_TRIGGER = "trg_bl_control_generation_update_v1"


def _upgrade_baseline_run_executor(connection: Connection) -> None:
    columns = _column_names(connection, BASELINE_RUN_JOB_TABLE)
    additions = (
        ("worker_service_id", "VARCHAR(128)"),
        ("worker_contract_version", "VARCHAR(64)"),
        (
            "started_at",
            "TIMESTAMP WITH TIME ZONE"
            if connection.dialect.name == "postgresql"
            else "DATETIME",
        ),
        ("retrieval_result_fingerprint", "VARCHAR(64)"),
    )
    for column_name, column_type in additions:
        if column_name not in columns:
            connection.exec_driver_sql(
                f"ALTER TABLE {BASELINE_RUN_JOB_TABLE} "
                f"ADD COLUMN {column_name} {column_type}"
            )

    if connection.dialect.name == "sqlite":
        connection.exec_driver_sql(
            f"DROP TRIGGER IF EXISTS {_RUN_EXECUTOR_INSERT_TRIGGER}"
        )
        connection.exec_driver_sql(
            f"DROP TRIGGER IF EXISTS {_RUN_EXECUTOR_METADATA_TRIGGER}"
        )
        invalid = (
            "((NEW.worker_service_id IS NULL) <> "
            "(NEW.worker_contract_version IS NULL)) OR "
            "((NEW.worker_service_id IS NULL) <> (NEW.started_at IS NULL)) OR "
            "(NEW.worker_service_id IS NOT NULL AND "
            "(NEW.worker_service_id <> 'compair-core-baseline-runner' OR "
            "NEW.worker_contract_version <> 'baseline-run-worker.v1')) OR "
            "(NEW.retrieval_result_fingerprint IS NOT NULL AND "
            "(length(NEW.retrieval_result_fingerprint) <> 64 OR "
            "NEW.retrieval_result_fingerprint GLOB '*[^0-9a-f]*')) "
        )
        connection.exec_driver_sql(
            f"CREATE TRIGGER {_RUN_EXECUTOR_INSERT_TRIGGER} BEFORE INSERT ON "
            f"baseline_control_run_job WHEN {invalid}"
            "BEGIN SELECT RAISE(ABORT, 'baseline_run_executor_metadata_invalid'); END"
        )
        connection.exec_driver_sql(
            f"CREATE TRIGGER {_RUN_EXECUTOR_METADATA_TRIGGER} BEFORE UPDATE OF "
            "worker_service_id, worker_contract_version, started_at, "
            "retrieval_result_fingerprint ON baseline_control_run_job WHEN "
            f"{invalid}OR "
            "(OLD.started_at IS NOT NULL AND NEW.started_at IS NOT OLD.started_at) OR "
            "(OLD.retrieval_result_fingerprint IS NOT NULL AND "
            "NEW.retrieval_result_fingerprint IS NOT "
            "OLD.retrieval_result_fingerprint) "
            "BEGIN SELECT RAISE(ABORT, 'baseline_run_executor_metadata_invalid'); END"
        )
        return

    connection.exec_driver_sql(
        "ALTER TABLE baseline_control_run_job DROP CONSTRAINT IF EXISTS "
        "ck_bl_run_job_worker"
    )
    connection.exec_driver_sql(
        "ALTER TABLE baseline_control_run_job ADD CONSTRAINT ck_bl_run_job_worker "
        "CHECK ((worker_service_id IS NULL AND worker_contract_version IS NULL "
        "AND started_at IS NULL) OR "
        "(worker_service_id = 'compair-core-baseline-runner' "
        "AND worker_contract_version = 'baseline-run-worker.v1' "
        "AND started_at IS NOT NULL))"
    )
    connection.exec_driver_sql(
        "ALTER TABLE baseline_control_run_job DROP CONSTRAINT IF EXISTS "
        "ck_bl_run_job_result_fingerprint"
    )
    connection.exec_driver_sql(
        "ALTER TABLE baseline_control_run_job ADD CONSTRAINT "
        "ck_bl_run_job_result_fingerprint CHECK "
        "(retrieval_result_fingerprint IS NULL OR "
        "retrieval_result_fingerprint ~ '^[0-9a-f]{64}$')"
    )
    connection.exec_driver_sql(
        "CREATE OR REPLACE FUNCTION compair_bl_run_executor_metadata_guard_v1() "
        "RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
        "IF OLD.started_at IS NOT NULL AND NEW.started_at IS DISTINCT FROM "
        "OLD.started_at THEN RAISE EXCEPTION 'baseline_run_started_at_immutable'; "
        "END IF; IF OLD.retrieval_result_fingerprint IS NOT NULL AND "
        "NEW.retrieval_result_fingerprint IS DISTINCT FROM "
        "OLD.retrieval_result_fingerprint THEN "
        "RAISE EXCEPTION 'baseline_run_result_fingerprint_immutable'; END IF; "
        "RETURN NEW; END $$"
    )
    connection.exec_driver_sql(
        f"DROP TRIGGER IF EXISTS {_RUN_EXECUTOR_METADATA_TRIGGER} "
        "ON baseline_control_run_job"
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER {_RUN_EXECUTOR_METADATA_TRIGGER} BEFORE UPDATE OF "
        "worker_service_id, worker_contract_version, started_at, "
        "retrieval_result_fingerprint ON baseline_control_run_job FOR EACH ROW "
        "EXECUTE FUNCTION compair_bl_run_executor_metadata_guard_v1()"
    )


def _validate_baseline_run_executor(connection: Connection) -> None:
    required_columns = {
        "retrieval_result_fingerprint",
        "started_at",
        "worker_contract_version",
        "worker_service_id",
    }
    if not required_columns <= _column_names(connection, BASELINE_RUN_JOB_TABLE):
        raise SchemaInvariantError("baseline_run_executor_columns_missing")
    fingerprint_invalid = (
        "(length(retrieval_result_fingerprint) <> 64 OR "
        "retrieval_result_fingerprint GLOB '*[^0-9a-f]*')"
        if connection.dialect.name == "sqlite"
        else "retrieval_result_fingerprint !~ '^[0-9a-f]{64}$'"
    )
    invalid = connection.execute(
        text(
            "SELECT count(*) FROM baseline_control_run_job WHERE "
            "((worker_service_id IS NULL) <> (worker_contract_version IS NULL)) OR "
            "((worker_service_id IS NULL) <> (started_at IS NULL)) OR "
            "(worker_service_id IS NOT NULL AND "
            "(worker_service_id <> :service OR worker_contract_version <> :contract)) OR "
            "(retrieval_result_fingerprint IS NOT NULL AND "
            f"{fingerprint_invalid})"
        ),
        {
            "service": BASELINE_RUN_WORKER_SERVICE_ID,
            "contract": BASELINE_RUN_WORKER_CONTRACT_VERSION,
        },
    ).scalar_one()
    if int(invalid) != 0:
        raise SchemaInvariantError("baseline_run_executor_row_invalid")
    if connection.dialect.name == "sqlite":
        triggers = {
            str(row[0])
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).all()
        }
    else:
        triggers = {
            str(row[0])
            for row in connection.execute(
                text(
                    "SELECT tgname FROM pg_trigger WHERE NOT tgisinternal AND "
                    "tgrelid = 'baseline_control_run_job'::regclass"
                )
            ).all()
        }
    required_triggers = {_RUN_EXECUTOR_METADATA_TRIGGER}
    if connection.dialect.name == "sqlite":
        required_triggers.add(_RUN_EXECUTOR_INSERT_TRIGGER)
    if not required_triggers <= triggers:
        raise SchemaInvariantError("baseline_run_executor_trigger_missing")


_CONTROL_GENERATION_COLUMN_TYPES = (
    ("generation_attempt_count", "INTEGER NOT NULL DEFAULT 0"),
    ("generation_contract_version", "VARCHAR(64)"),
    ("generation_started_at", "TIMESTAMP WITH TIME ZONE", "DATETIME"),
    ("generation_provider", "VARCHAR(128)"),
    ("generation_model", "VARCHAR(256)"),
    ("generation_model_version", "VARCHAR(256)"),
    ("generation_provider_fingerprint", "VARCHAR(64)"),
    ("generation_provider_idempotency_supported", "BOOLEAN"),
    ("generation_output_schema_version", "VARCHAR(64)"),
    ("generation_output_schema_sha256", "VARCHAR(64)"),
    ("generation_input_fingerprint", "VARCHAR(64)"),
    ("generation_output_fingerprint", "VARCHAR(64)"),
    ("generation_completed_at", "TIMESTAMP WITH TIME ZONE", "DATETIME"),
)


def _control_generation_invalid_sql(*, sqlite: bool, prefix: str = "NEW.") -> str:
    hash_invalid = (
        f"(length({prefix}{{column}}) <> 64 OR {prefix}{{column}} GLOB '*[^0-9a-f]*')"
        if sqlite
        else f"{prefix}{{column}} !~ '^[0-9a-f]{{64}}$'"
    )
    hashes = (
        "generation_provider_fingerprint",
        "generation_output_schema_sha256",
        "generation_input_fingerprint",
        "generation_output_fingerprint",
    )
    hash_checks = " OR ".join(
        f"({prefix}{column} IS NOT NULL AND {hash_invalid.replace('{column}', column)})"
        for column in hashes
    )
    metadata = (
        "generation_contract_version",
        "generation_started_at",
        "generation_provider",
        "generation_model",
        "generation_model_version",
        "generation_provider_fingerprint",
        "generation_provider_idempotency_supported",
        "generation_output_schema_version",
        "generation_output_schema_sha256",
        "generation_input_fingerprint",
    )
    missing = " OR ".join(f"{prefix}{column} IS NULL" for column in metadata)
    present = " OR ".join(f"{prefix}{column} IS NOT NULL" for column in metadata)
    return (
        f"{prefix}generation_attempt_count < 0 OR "
        f"({prefix}generation_attempt_count = 0 AND ({present})) OR "
        f"({prefix}generation_attempt_count > 0 AND ({missing})) OR "
        f"({prefix}generation_contract_version IS NOT NULL AND "
        f"{prefix}generation_contract_version <> 'baseline-control-generation.v1') OR "
        f"({prefix}generation_output_schema_version IS NOT NULL AND "
        f"{prefix}generation_output_schema_version <> 'baseline-generation-output.v2') OR "
        f"({hash_checks}) OR "
        f"({prefix}state = 'feedback_persisted' AND ("
        f"{prefix}generation_invoked IS NOT TRUE OR "
        f"{prefix}generation_attempt_count < 1 OR "
        f"{prefix}generation_output_fingerprint IS NULL OR "
        f"{prefix}generation_completed_at IS NULL OR "
        f"{prefix}feedback_count > {prefix}reference_count OR "
        f"({prefix}feedback_count = 0 AND {prefix}notification_outbox_count <> 0)))"
    )


def _upgrade_baseline_control_generation(connection: Connection) -> None:
    columns = _column_names(connection, BASELINE_RUN_JOB_TABLE)
    for definition in _CONTROL_GENERATION_COLUMN_TYPES:
        column_name = definition[0]
        if column_name in columns:
            continue
        column_type = (
            definition[2]
            if connection.dialect.name == "sqlite" and len(definition) == 3
            else definition[1]
        )
        connection.exec_driver_sql(
            f"ALTER TABLE {BASELINE_RUN_JOB_TABLE} ADD COLUMN "
            f"{column_name} {column_type}"
        )

    invalid = _control_generation_invalid_sql(
        sqlite=connection.dialect.name == "sqlite"
    )
    if connection.dialect.name == "sqlite":
        connection.exec_driver_sql(
            f"DROP TRIGGER IF EXISTS {_CONTROL_GENERATION_INSERT_TRIGGER}"
        )
        connection.exec_driver_sql(
            f"DROP TRIGGER IF EXISTS {_CONTROL_GENERATION_METADATA_TRIGGER}"
        )
        connection.exec_driver_sql(
            f"CREATE TRIGGER {_CONTROL_GENERATION_INSERT_TRIGGER} BEFORE INSERT ON "
            f"{BASELINE_RUN_JOB_TABLE} WHEN {invalid} BEGIN SELECT RAISE(ABORT, "
            "'baseline_control_generation_metadata_invalid'); END"
        )
        immutable = (
            "OLD.state = 'feedback_persisted' AND ("
            "NEW.state IS NOT OLD.state OR "
            "NEW.generation_attempt_count IS NOT OLD.generation_attempt_count OR "
            "NEW.generation_contract_version IS NOT OLD.generation_contract_version OR "
            "NEW.generation_provider IS NOT OLD.generation_provider OR "
            "NEW.generation_model IS NOT OLD.generation_model OR "
            "NEW.generation_model_version IS NOT OLD.generation_model_version OR "
            "NEW.generation_provider_fingerprint IS NOT OLD.generation_provider_fingerprint OR "
            "NEW.generation_provider_idempotency_supported IS NOT "
            "OLD.generation_provider_idempotency_supported OR "
            "NEW.generation_input_fingerprint IS NOT OLD.generation_input_fingerprint OR "
            "NEW.generation_output_fingerprint IS NOT OLD.generation_output_fingerprint OR "
            "NEW.generation_completed_at IS NOT OLD.generation_completed_at OR "
            "NEW.feedback_count IS NOT OLD.feedback_count OR "
            "NEW.notification_outbox_count IS NOT OLD.notification_outbox_count)"
        )
        connection.exec_driver_sql(
            f"CREATE TRIGGER {_CONTROL_GENERATION_METADATA_TRIGGER} BEFORE UPDATE ON "
            f"{BASELINE_RUN_JOB_TABLE} WHEN ({invalid}) OR ({immutable}) "
            "BEGIN SELECT RAISE(ABORT, "
            "'baseline_control_generation_metadata_invalid'); END"
        )
        return

    connection.exec_driver_sql(
        "ALTER TABLE baseline_control_run_job DROP CONSTRAINT IF EXISTS "
        "ck_bl_run_job_generation_contract"
    )
    connection.exec_driver_sql(
        "ALTER TABLE baseline_control_run_job ADD CONSTRAINT "
        "ck_bl_run_job_generation_contract CHECK (NOT ("
        + invalid.replace("NEW.", "")
        + "))"
    )
    connection.exec_driver_sql(
        "CREATE OR REPLACE FUNCTION compair_bl_control_generation_guard_v1() "
        "RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
        "IF OLD.state = 'feedback_persisted' AND ("
        "NEW.state IS DISTINCT FROM OLD.state OR "
        "NEW.generation_attempt_count IS DISTINCT FROM OLD.generation_attempt_count OR "
        "NEW.generation_contract_version IS DISTINCT FROM OLD.generation_contract_version OR "
        "NEW.generation_provider IS DISTINCT FROM OLD.generation_provider OR "
        "NEW.generation_model IS DISTINCT FROM OLD.generation_model OR "
        "NEW.generation_model_version IS DISTINCT FROM OLD.generation_model_version OR "
        "NEW.generation_provider_fingerprint IS DISTINCT FROM OLD.generation_provider_fingerprint OR "
        "NEW.generation_provider_idempotency_supported IS DISTINCT FROM "
        "OLD.generation_provider_idempotency_supported OR "
        "NEW.generation_input_fingerprint IS DISTINCT FROM OLD.generation_input_fingerprint OR "
        "NEW.generation_output_fingerprint IS DISTINCT FROM OLD.generation_output_fingerprint OR "
        "NEW.generation_completed_at IS DISTINCT FROM OLD.generation_completed_at OR "
        "NEW.feedback_count IS DISTINCT FROM OLD.feedback_count OR "
        "NEW.notification_outbox_count IS DISTINCT FROM OLD.notification_outbox_count) "
        "THEN RAISE EXCEPTION 'baseline_control_generation_metadata_immutable'; END IF; "
        "RETURN NEW; END $$"
    )
    connection.exec_driver_sql(
        f"DROP TRIGGER IF EXISTS {_CONTROL_GENERATION_METADATA_TRIGGER} "
        "ON baseline_control_run_job"
    )
    connection.exec_driver_sql(
        f"CREATE TRIGGER {_CONTROL_GENERATION_METADATA_TRIGGER} BEFORE UPDATE ON "
        "baseline_control_run_job FOR EACH ROW EXECUTE FUNCTION "
        "compair_bl_control_generation_guard_v1()"
    )


def _validate_baseline_control_generation(connection: Connection) -> None:
    required = {definition[0] for definition in _CONTROL_GENERATION_COLUMN_TYPES}
    if not required <= _column_names(connection, BASELINE_RUN_JOB_TABLE):
        raise SchemaInvariantError("baseline_control_generation_columns_missing")
    invalid = _control_generation_invalid_sql(
        sqlite=connection.dialect.name == "sqlite", prefix=""
    )
    count = connection.execute(
        text(f"SELECT count(*) FROM {BASELINE_RUN_JOB_TABLE} WHERE {invalid}")
    ).scalar_one()
    if int(count):
        raise SchemaInvariantError("baseline_control_generation_row_invalid")
    if connection.dialect.name == "sqlite":
        trigger_names = {
            str(row[0])
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            ).all()
        }
        required_triggers = {
            _CONTROL_GENERATION_INSERT_TRIGGER,
            _CONTROL_GENERATION_METADATA_TRIGGER,
        }
    else:
        trigger_names = {
            str(row[0])
            for row in connection.execute(
                text(
                    "SELECT tgname FROM pg_trigger WHERE NOT tgisinternal AND "
                    "tgrelid = 'baseline_control_run_job'::regclass"
                )
            ).all()
        }
        required_triggers = {_CONTROL_GENERATION_METADATA_TRIGGER}
    if not required_triggers <= trigger_names:
        raise SchemaInvariantError("baseline_control_generation_trigger_missing")


def _upgrade_baseline_database_worker(connection: Connection) -> None:
    for table in DATABASE_WORKER_TABLES:
        connection.execute(CreateTable(table, if_not_exists=True))
        for index in sorted(table.indexes, key=lambda candidate: candidate.name or ""):
            connection.execute(CreateIndex(index, if_not_exists=True))


def _validate_baseline_database_worker(connection: Connection) -> None:
    if BASELINE_WORKER_INSTANCE_TABLE not in set(
        inspect(connection).get_table_names()
    ):
        raise SchemaInvariantError("baseline_database_worker_table_missing")
    table = DATABASE_WORKER_TABLES[0]
    if not {column.name for column in table.columns} <= _column_names(
        connection, BASELINE_WORKER_INSTANCE_TABLE
    ):
        raise SchemaInvariantError("baseline_database_worker_column_missing")
    if not {
        "ck_bl_db_worker_contract",
        "ck_bl_db_worker_supported_jobs",
        "ck_bl_db_worker_capacity",
    } <= _constraint_names(connection, BASELINE_WORKER_INSTANCE_TABLE, "check"):
        raise SchemaInvariantError("baseline_database_worker_check_missing")
    if "ix_bl_db_worker_heartbeat" not in _index_names(
        connection, BASELINE_WORKER_INSTANCE_TABLE
    ):
        raise SchemaInvariantError("baseline_database_worker_index_missing")


def _upgrade_baseline_worker_runtime_attestation(connection: Connection) -> None:
    for table in DATABASE_WORKER_ATTESTATION_TABLES:
        connection.execute(CreateTable(table, if_not_exists=True))
        for index in sorted(table.indexes, key=lambda candidate: candidate.name or ""):
            connection.execute(CreateIndex(index, if_not_exists=True))


def _validate_baseline_worker_runtime_attestation(connection: Connection) -> None:
    if BASELINE_WORKER_ATTESTATION_TABLE not in set(
        inspect(connection).get_table_names()
    ):
        raise SchemaInvariantError("baseline_worker_attestation_table_missing")
    table = DATABASE_WORKER_ATTESTATION_TABLES[0]
    if not {column.name for column in table.columns} <= _column_names(
        connection, BASELINE_WORKER_ATTESTATION_TABLE
    ):
        raise SchemaInvariantError("baseline_worker_attestation_column_missing")
    if "ck_bl_db_worker_attestation_contract" not in _constraint_names(
        connection,
        BASELINE_WORKER_ATTESTATION_TABLE,
        "check",
    ):
        raise SchemaInvariantError("baseline_worker_attestation_check_missing")
    if "ix_bl_db_worker_attestation_runtime" not in _index_names(
        connection,
        BASELINE_WORKER_ATTESTATION_TABLE,
    ):
        raise SchemaInvariantError("baseline_worker_attestation_index_missing")
    foreign_keys = {
        str(item.get("name"))
        for item in inspect(connection).get_foreign_keys(
            BASELINE_WORKER_ATTESTATION_TABLE
        )
    }
    if "fk_bl_db_worker_attestation_instance" not in foreign_keys:
        raise SchemaInvariantError("baseline_worker_attestation_foreign_key_missing")


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
    SchemaMigration(
        migration_id="0004_baseline_notification_outbox_v1",
        description="Add privacy-safe idempotent baseline notification digest outbox",
        checksum_material=(
            "baseline-notification-digest.v1; one group-scoped in_app digest per "
            "succeeded baseline run recipient channel; ordered Feedback identifiers "
            "only; pending running delivered retryable_failed terminal_failed "
            "suppressed cancelled; leased at-least-once dispatch; recipient SET NULL, "
            "run/group CASCADE; succeeded-run insert and immutable-payload triggers; "
            "SQLite and PostgreSQL DDL; no legacy notification changes; ddl-v1"
        ),
        upgrade=_upgrade_baseline_notification_outbox,
        validate=_validate_baseline_notification_outbox,
    ),
    SchemaMigration(
        migration_id="0005_baseline_control_plane_staging_v1",
        description="Add authenticated baseline control-plane staging and durable jobs",
        checksum_material=(
            "baseline-control-plane-staging.v1; group-scoped authorized repository "
            "registrations; durable snapshot_ingest jobs with opaque idempotency "
            "intent hashes, safe states and expiring leases; immutable bounded content "
            "parts; open sealed expired failed staging sessions; source provenance "
            "SET NULL, group lifecycle CASCADE; SQLite and PostgreSQL DDL and immutable "
            "part triggers; no corpus/index eligibility or worker execution; ddl-v1"
        ),
        upgrade=_upgrade_baseline_control_plane_staging,
        validate=_validate_baseline_control_plane_staging,
    ),
    SchemaMigration(
        migration_id="0006_baseline_control_plane_continuation_v1",
        description=(
            "Add administrator repository approvals and sealed snapshot continuation jobs"
        ),
        checksum_material=(
            "baseline-control-plane-continuation.v1; immutable group-admin-approved "
            "repository identity descriptors linked to opaque registration IDs; active "
            "disabled lifecycle; distinct sealed snapshot continuation jobs; queued "
            "running succeeded retryable_failed terminal_failed cancelled; leases, "
            "attempts, sanitized errors, immutable sealed hashes and counts; staging "
            "and continuation group CASCADE, user provenance SET NULL; SQLite and "
            "PostgreSQL additive DDL plus registration, descriptor, sealed staging "
            "identity, and continuation immutability triggers; no corpus ingestion "
            "or eligibility; ddl-v2"
        ),
        upgrade=_upgrade_baseline_control_plane_continuation,
        validate=_validate_baseline_control_plane_continuation,
    ),
    SchemaMigration(
        migration_id="0007_baseline_control_plane_ingestion_worker_v1",
        description=(
            "Add immutable continuation corpus-publication outcomes and sealed-part guards"
        ),
        checksum_material=(
            "baseline-continuation-worker.v1; additive value-only corpus and generation "
            "result identifiers, generation version, trusted manifest and provenance "
            "fingerprints, worker contract and publication timestamp; result fields "
            "required only for succeeded state and immutable thereafter; sealed snapshot "
            "parts reject insert and delete after commit; SQLite and PostgreSQL triggers "
            "and result lookup index; corpus/index tables remain lifecycle-independent; "
            "ddl-v1"
        ),
        upgrade=_upgrade_baseline_control_plane_ingestion_worker,
        validate=_validate_baseline_control_plane_ingestion_worker,
    ),
    SchemaMigration(
        migration_id="0008_baseline_compatible_index_job_v1",
        description=(
            "Add leased compatible-index build continuation and safe publication result"
        ),
        checksum_material=(
            "baseline-index-build-continuation.v1; group-scoped extension of durable "
            "baseline_control_job linked to one succeeded sealed-snapshot continuation; "
            "immutable active corpus generation and trusted ingestion hash provenance; "
            "exact baseline-index, tokenizer, retrieval config, embedding HTTP contract, "
            "provider, BGE model, revision, dimension, float32 dtype and fingerprint "
            "intent; unique generation intent and job scope; complete safe index result "
            "required atomically for succeeded; immutable result and intent triggers; "
            "SQLite and PostgreSQL additive DDL; no baseline run enablement; ddl-v1"
        ),
        upgrade=_upgrade_baseline_compatible_index_job,
        validate=_validate_baseline_compatible_index_job,
    ),
    SchemaMigration(
        migration_id="0009_baseline_run_job_v1",
        description="Add protected transient baseline-run jobs and encrypted payloads",
        checksum_material=(
            "baseline-run-job.v1; group-scoped safe run intent and frozen status "
            "metadata; SHA-256 protected caller idempotency; authoritative source, "
            "changed registration, active corpus, compatible index job and publication "
            "identities; random processing identity fingerprint; raw query provenance "
            "hash, character and byte lengths only; separate AES-256-GCM payload with "
            "external key ID, database-unique per-key 96-bit nonce, authenticated-data "
            "version and bounded "
            "expiry; source and submitter SET NULL, group and payload CASCADE; SQLite "
            "and PostgreSQL additive DDL; no run endpoint or worker; ddl-v1"
        ),
        upgrade=_upgrade_baseline_run_job,
        validate=_validate_baseline_run_job,
    ),
    SchemaMigration(
        migration_id="0010_baseline_document_source_scope_v1",
        description=(
            "Align immutable baseline evidence with document-scoped control runs"
        ),
        checksum_material=(
            "baseline-source-scope.v1; deterministic legacy_chunk backfill for all "
            "existing retrieval runs; explicit legacy_chunk and control_document "
            "creation contracts with retained nullable source provenance; one unique "
            "group-consistent persisted_run_id foreign key from control job to "
            "retrieval run; atomic running-to-references_persisted attachment guard; "
            "job-wide one-to-four evidence and 16000-character limit; SQLite table "
            "copy plus triggers and PostgreSQL additive constraints plus triggers; "
            "no synthetic chunks, executor, retrieval, or generation dispatch; ddl-v1"
        ),
        upgrade=_upgrade_baseline_document_source_scope,
        validate=_validate_baseline_document_source_scope,
    ),
    SchemaMigration(
        migration_id="0011_baseline_run_executor_v1",
        description=(
            "Add lease-safe internal document-level baseline-run executor metadata"
        ),
        checksum_material=(
            "baseline-run-worker.v1; fixed compair-core-baseline-runner service; "
            "immutable first-start timestamp and safe retrieval-result SHA-256; reuse "
            "existing opaque lease, expiry, attempt, state, count and sanitized-error "
            "fields; SQLite additive columns plus guard trigger and PostgreSQL additive "
            "columns, checks and immutable guard trigger; no public run endpoint, "
            "generation, feedback, notification, preview or capability enablement; ddl-v1"
        ),
        upgrade=_upgrade_baseline_run_executor,
        validate=_validate_baseline_run_executor,
    ),
    SchemaMigration(
        migration_id="0012_baseline_control_generation_v1",
        description=(
            "Add coordinated structured generation metadata to document run jobs"
        ),
        checksum_material=(
            "baseline-control-generation.v1; reuse control-job and retrieval-run "
            "opaque generation lease; additive attempt, provider identity, provider "
            "idempotency attestation, frozen baseline-generation-output.v2 schema, "
            "input/output fingerprints and completion timestamp; feedback_persisted "
            "allows zero through four ordered findings and requires resolved generation; "
            "terminal result metadata immutable; SQLite and PostgreSQL guards; no public "
            "run endpoint, query access, retrieval, preview or capability enablement; ddl-v1"
        ),
        upgrade=_upgrade_baseline_control_generation,
        validate=_validate_baseline_control_generation,
    ),
    SchemaMigration(
        migration_id="0013_baseline_database_worker_v1",
        description=(
            "Add privacy-safe database worker instance and heartbeat readiness state"
        ),
        checksum_material=(
            "baseline-database-worker.v1; opaque worker instance UUID; fixed worker "
            "contract and safe supported corpus-ingestion, index-build, baseline-run "
            "and cleanup flags; start and recent heartbeat timestamps; draining, "
            "bounded concurrency and active capacity only; no host, path, endpoint, "
            "credential, lease, payload or job-content fields; heartbeat lookup index; "
            "SQLite and PostgreSQL additive portable DDL; ddl-v1"
        ),
        upgrade=_upgrade_baseline_database_worker,
        validate=_validate_baseline_database_worker,
    ),
    SchemaMigration(
        migration_id="0014_baseline_worker_runtime_attestation_v1",
        description=("Add privacy-safe worker runtime configuration attestation"),
        checksum_material=(
            "baseline-runtime-config.v1; one-to-one cascade extension of "
            "baseline_database_worker_instance; exact runtime, embedding and "
            "generation SHA-256 fingerprints; no endpoint, DSN, path, secret, "
            "payload or job identity; runtime fingerprint lookup index; SQLite "
            "and PostgreSQL additive portable DDL; ddl-v1"
        ),
        upgrade=_upgrade_baseline_worker_runtime_attestation,
        validate=_validate_baseline_worker_runtime_attestation,
    ),
)


def _validate_registry(
    migrations: Sequence[SchemaMigration],
) -> tuple[SchemaMigration, ...]:
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
                connection.execute(
                    CreateTable(schema_migration_table, if_not_exists=True)
                )
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
                    raise SchemaMigrationError(
                        None, "sqlite_foreign_key_restore_failed"
                    )
            return

        transaction = connection.begin()
        try:
            connection.execute(
                text(f"LOCK TABLE {MIGRATION_TABLE_NAME} IN EXCLUSIVE MODE")
            )
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
        existing = (
            connection.execute(
                select(schema_migration_table).where(
                    schema_migration_table.c.migration_id == migration.migration_id
                )
            )
            .mappings()
            .one_or_none()
        )
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
                    raise SchemaMigrationError(
                        migration.migration_id, "previous_failure"
                    )
                if row["state"] != "applied":
                    raise SchemaMigrationError(migration.migration_id, "state_invalid")
                if row["checksum"] != migration.checksum:
                    raise SchemaMigrationError(
                        migration.migration_id, "checksum_mismatch"
                    )
                if migration.validate is not None:
                    try:
                        migration.validate(connection)
                    except Exception as exc:
                        code = _failure_code(exc, "validation")
                        raise SchemaMigrationError(
                            migration.migration_id, code
                        ) from exc
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
        raise SchemaMigrationError(
            failure.migration_id, failure.code
        ) from failure.cause

    return MigrationReport(tuple(applied), tuple(already_applied))


def read_schema_migration_state(engine: Engine) -> tuple[MigrationState, ...]:
    """Return ordered migration metadata for health checks and operations."""

    dialect = _require_supported_dialect(engine)
    _bootstrap_migration_table(engine, dialect)
    with engine.connect() as connection:
        rows = connection.execute(
            select(schema_migration_table).order_by(
                schema_migration_table.c.migration_id
            )
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
