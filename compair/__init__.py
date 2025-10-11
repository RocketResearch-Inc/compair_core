from __future__ import annotations

import logging
import os

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker

edition = os.getenv("COMPAIR_EDITION", "core").lower()

_cloud_available = False
if edition == "cloud":
    try:  # Import cloud overrides if the private package is installed
        from compair_cloud import (
            bootstrap as cloud_bootstrap,
            celery_app as cloud_celery_app,
            default_groups as cloud_default_groups,
            embeddings as cloud_embeddings,
            feedback as cloud_feedback,
            logger as cloud_logger,
            main as cloud_main,
            models as cloud_models,
            tasks as cloud_tasks,
            utils as cloud_utils,
        )  # type: ignore

        _cloud_available = True
    except ImportError:
        _cloud_available = False

if _cloud_available:
    from compair_cloud.default_groups import initialize_default_groups  # type: ignore
    embeddings = cloud_embeddings
    feedback = cloud_feedback
    logger = cloud_logger
    main = cloud_main
    models = cloud_models
    tasks = cloud_tasks
    utils = cloud_utils
    initialize_database_override = getattr(cloud_bootstrap, "initialize_database", None)
else:
    from . import embeddings, feedback, logger, main, models, tasks, utils
    from .default_groups import initialize_default_groups
    initialize_database_override = None


logging.basicConfig(level=logging.INFO)


def _handle_engine() -> Engine:
    db = os.getenv("DB")
    db_user = os.getenv("DB_USER")
    db_passw = os.getenv("DB_PASSW")
    db_url = os.getenv("DB_URL")

    if all([db, db_user, db_passw, db_url]):
        return create_engine(
            f"postgresql+psycopg2://{db_user}:{db_passw}@{db_url}/{db}",
            pool_size=10,
            max_overflow=0,
        )

    sqlite_dir = os.getenv("COMPAIR_SQLITE_DIR", "/data")
    os.makedirs(sqlite_dir, exist_ok=True)
    sqlite_path = os.path.join(sqlite_dir, os.getenv("COMPAIR_SQLITE_NAME", "compair.db"))
    return create_engine(f"sqlite:///{sqlite_path}", connect_args={"check_same_thread": False})


engine = _handle_engine()


def initialize_database() -> None:
    models.Base.metadata.create_all(engine)
    if initialize_database_override:
        initialize_database_override(engine)


initialize_database()

Session = sessionmaker(engine)
embedder = embeddings.Embedder()
reviewer = feedback.Reviewer()

with Session() as session:
    initialize_default_groups(session)

__all__ = ["embeddings", "feedback", "main", "models", "utils", "Session"]
