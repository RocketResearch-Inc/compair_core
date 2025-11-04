from __future__ import annotations

import os
from pathlib import Path
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker

from . import embeddings, feedback, logger, main, models, tasks, utils
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
    except ImportError:
        pass


def _handle_engine() -> Engine:
    # Preferred configuration: explicit database URL
    explicit_url = (
        os.getenv("COMPAIR_DATABASE_URL")
        or os.getenv("COMPAIR_DB_URL")
        or os.getenv("DATABASE_URL")
    )
    if explicit_url:
        if explicit_url.startswith("sqlite:"):
            return create_engine(explicit_url, connect_args={"check_same_thread": False})
        return create_engine(explicit_url)

    # Backwards compatibility with legacy Postgres env variables
    db = os.getenv("DB")
    db_user = os.getenv("DB_USER")
    db_passw = os.getenv("DB_PASSW")
    db_host = os.getenv("DB_URL")

    if all([db, db_user, db_passw, db_host]):
        return create_engine(
            f"postgresql+psycopg2://{db_user}:{db_passw}@{db_host}/{db}",
            pool_size=10,
            max_overflow=0,
        )

    # Local default: place an SQLite database inside COMPAIR_DB_DIR
    db_dir = (
        os.getenv("COMPAIR_DB_DIR")
        or os.getenv("COMPAIR_SQLITE_DIR")
        or os.path.join(Path.home(), ".compair-core", "data")
    )
    db_name = os.getenv("COMPAIR_DB_NAME") or os.getenv("COMPAIR_SQLITE_NAME") or "compair.db"

    db_path = Path(db_dir).expanduser()
    try:
        db_path.mkdir(parents=True, exist_ok=True)
    except OSError:
        fallback_dir = Path(os.getcwd()) / "compair_data"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        db_path = fallback_dir

    sqlite_path = db_path / db_name
    return create_engine(
        f"sqlite:///{sqlite_path}",
        connect_args={"check_same_thread": False},
    )


def initialize_database() -> None:
    models.Base.metadata.create_all(engine)
    if initialize_database_override:
        initialize_database_override(engine)


def _initialize_defaults() -> None:
    with Session() as session:
        initialize_default_groups(session)


engine = _handle_engine()
initialize_database()
Session = sessionmaker(engine)
embedder = embeddings.Embedder()
reviewer = feedback.Reviewer()
_initialize_defaults()

__all__ = ["embeddings", "feedback", "main", "models", "utils", "Session"]
