from __future__ import annotations

import os
import sys

from . import embeddings, feedback, logger, main, models, tasks, utils
from compair_core.db import SessionLocal as Session
from compair_core.db import engine
from .default_groups import initialize_default_groups

edition = os.getenv("COMPAIR_EDITION", "core").lower()

initialize_database_override = None

if edition == "cloud":
    try:  # Import cloud overrides if the private package is installed
        from compair_cloud import (  # type: ignore
            bootstrap as cloud_bootstrap,
            embeddings as cloud_embeddings,
            feedback as cloud_feedback,
            logger as cloud_logger,
            main as cloud_main,
            models as cloud_models,
            tasks as cloud_tasks,
            utils as cloud_utils,
        )

        embeddings = cloud_embeddings
        feedback = cloud_feedback
        logger = cloud_logger
        main = cloud_main
        models = cloud_models
        tasks = cloud_tasks
        utils = cloud_utils
        initialize_database_override = getattr(cloud_bootstrap, "initialize_database", None)
    except Exception as exc:
        print(f"[compair_core] Failed to import compair_cloud: {exc}", file=sys.stderr)
        import traceback; traceback.print_exc()


def _ensure_topic_tags_column() -> None:
    try:
        from sqlalchemy import inspect, text

        insp = inspect(engine)
        if "document" not in insp.get_table_names():
            return
        cols = {c["name"] for c in insp.get_columns("document")}
        if "topic_tags" in cols:
            return
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE document ADD COLUMN topic_tags JSON"))
    except Exception as exc:
        print(f"[compair_core] topic_tags migration skipped: {exc}", file=sys.stderr)


def _ensure_user_retrial_count_default() -> None:
    """Keep core inserts compatible with Cloud-shaped user tables."""
    try:
        from sqlalchemy import inspect, text

        insp = inspect(engine)
        if "user" not in insp.get_table_names():
            return
        cols = {c["name"] for c in insp.get_columns("user")}
        if "retrial_count" not in cols:
            return
        with engine.begin() as conn:
            conn.execute(text('UPDATE "user" SET retrial_count = 0 WHERE retrial_count IS NULL'))
            if conn.dialect.name == "postgresql":
                conn.execute(text('ALTER TABLE "user" ALTER COLUMN retrial_count SET DEFAULT 0'))
    except Exception as exc:
        print(f"[compair_core] retrial_count migration skipped: {exc}", file=sys.stderr)


def _ensure_notification_preferences_delivery_columns() -> None:
    try:
        from sqlalchemy import inspect, text

        insp = inspect(engine)
        if "notification_preferences" not in insp.get_table_names():
            return
        cols = {c["name"] for c in insp.get_columns("notification_preferences")}
        statements: list[str] = []
        if "notification_delivery_email" not in cols:
            statements.append("ALTER TABLE notification_preferences ADD COLUMN notification_delivery_email VARCHAR(256)")
        if "notification_delivery_email_pending" not in cols:
            statements.append("ALTER TABLE notification_preferences ADD COLUMN notification_delivery_email_pending VARCHAR(256)")
        if "notification_delivery_email_verified_at" not in cols:
            statements.append("ALTER TABLE notification_preferences ADD COLUMN notification_delivery_email_verified_at DATETIME")
        if not statements:
            return
        with engine.begin() as conn:
            for statement in statements:
                conn.execute(text(statement))
    except Exception as exc:
        print(f"[compair_core] notification delivery email migration skipped: {exc}", file=sys.stderr)


def _ensure_reference_chunk_id_column() -> None:
    try:
        from sqlalchemy import inspect, text

        insp = inspect(engine)
        if "reference" not in insp.get_table_names():
            return
        cols = {c["name"] for c in insp.get_columns("reference")}
        if "reference_chunk_id" in cols:
            return
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE reference ADD COLUMN reference_chunk_id VARCHAR(36)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS ix_reference_reference_chunk_id ON reference (reference_chunk_id)"))
    except Exception as exc:
        print(f"[compair_core] reference_chunk_id migration skipped: {exc}", file=sys.stderr)


def _ensure_pgvector_extension() -> None:
    try:
        from sqlalchemy import text

        if engine.dialect.name != "postgresql":
            return
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception as exc:
        print(f"[compair_core] pgvector extension setup skipped: {exc}", file=sys.stderr)


def initialize_database() -> None:
    _ensure_pgvector_extension()
    if edition == "cloud" and initialize_database_override:
        initialize_database_override(engine)
    else:
        models.Base.metadata.create_all(engine)
    _ensure_user_retrial_count_default()
    _ensure_notification_preferences_delivery_columns()
    _ensure_reference_chunk_id_column()
    if edition == "core":
        _ensure_topic_tags_column()
    elif not initialize_database_override:
        _ensure_topic_tags_column()
    if initialize_database_override and edition != "cloud":
        initialize_database_override(engine)


def _initialize_defaults() -> None:
    with Session() as session:
        initialize_default_groups(session)


initialize_database()
embedder = embeddings.Embedder()
reviewer = feedback.Reviewer()
_initialize_defaults()

__all__ = ["embeddings", "feedback", "main", "models", "utils", "Session"]
