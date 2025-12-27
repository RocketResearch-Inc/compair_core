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


def initialize_database() -> None:
    models.Base.metadata.create_all(engine)
    if edition == "core":
        _ensure_topic_tags_column()
    if initialize_database_override:
        initialize_database_override(engine)


def _initialize_defaults() -> None:
    with Session() as session:
        initialize_default_groups(session)


initialize_database()
embedder = embeddings.Embedder()
reviewer = feedback.Reviewer()
_initialize_defaults()

__all__ = ["embeddings", "feedback", "main", "models", "utils", "Session"]
