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
from sqlalchemy.schema import CreateTable

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
