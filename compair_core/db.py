from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker


def _build_engine() -> Engine:
    """Create the SQLAlchemy engine using the same precedence as the core package."""
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


engine = _build_engine()

# Keep behavior identical to previous `Session = sessionmaker(engine)`
SessionLocal = sessionmaker(engine)
Session = SessionLocal

__all__ = ["engine", "SessionLocal", "Session"]
