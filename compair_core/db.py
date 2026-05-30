from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import sessionmaker


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _optional_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value >= 0 else None


def _pooled_engine_kwargs(*, default_pool_size: int | None = None, default_max_overflow: int | None = None) -> dict[str, object]:
    """Return conservative SQLAlchemy pool settings for hosted Postgres.

    Celery workers can run for hours between DB checkouts while Render/Postgres
    may close idle SSL sockets. `pool_pre_ping` makes checkout discard dead
    sockets instead of failing the first query in the next task.
    """
    kwargs: dict[str, object] = {}
    if _bool_env("COMPAIR_DB_POOL_PRE_PING", True):
        kwargs["pool_pre_ping"] = True
    recycle_sec = _optional_int_env("COMPAIR_DB_POOL_RECYCLE_SEC")
    if recycle_sec is None:
        recycle_sec = 1800
    if recycle_sec > 0:
        kwargs["pool_recycle"] = recycle_sec
    pool_size = _optional_int_env("COMPAIR_DB_POOL_SIZE")
    if pool_size is None:
        pool_size = default_pool_size
    if pool_size is not None:
        kwargs["pool_size"] = pool_size
    max_overflow = _optional_int_env("COMPAIR_DB_MAX_OVERFLOW")
    if max_overflow is None:
        max_overflow = default_max_overflow
    if max_overflow is not None:
        kwargs["max_overflow"] = max_overflow
    pool_timeout = _optional_int_env("COMPAIR_DB_POOL_TIMEOUT_SEC")
    if pool_timeout is not None and pool_timeout > 0:
        kwargs["pool_timeout"] = pool_timeout
    return kwargs


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
        return create_engine(explicit_url, **_pooled_engine_kwargs())

    # Backwards compatibility with legacy Postgres env variables
    db = os.getenv("DB")
    db_user = os.getenv("DB_USER")
    db_passw = os.getenv("DB_PASSW")
    db_host = os.getenv("DB_URL")

    if all([db, db_user, db_passw, db_host]):
        return create_engine(
            f"postgresql+psycopg2://{db_user}:{db_passw}@{db_host}/{db}",
            **_pooled_engine_kwargs(default_pool_size=10, default_max_overflow=0),
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


def dispose_engine(*, close: bool = True) -> None:
    """Dispose pooled DB connections.

    Celery prefork children call this with `close=False` immediately after fork
    so they do not reuse sockets opened by the parent process.
    """
    try:
        engine.dispose(close=close)
    except TypeError:  # SQLAlchemy <1.4.33 compatibility.
        engine.dispose()


__all__ = ["engine", "SessionLocal", "Session", "dispose_engine"]
