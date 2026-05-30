from __future__ import annotations

from compair_core import db


def test_pooled_engine_kwargs_defaults_to_pre_ping_and_recycle(monkeypatch) -> None:
    monkeypatch.delenv("COMPAIR_DB_POOL_PRE_PING", raising=False)
    monkeypatch.delenv("COMPAIR_DB_POOL_RECYCLE_SEC", raising=False)

    assert db._pooled_engine_kwargs() == {"pool_pre_ping": True, "pool_recycle": 1800}


def test_pooled_engine_kwargs_allows_overrides(monkeypatch) -> None:
    monkeypatch.setenv("COMPAIR_DB_POOL_PRE_PING", "false")
    monkeypatch.setenv("COMPAIR_DB_POOL_RECYCLE_SEC", "0")
    monkeypatch.setenv("COMPAIR_DB_POOL_SIZE", "2")
    monkeypatch.setenv("COMPAIR_DB_MAX_OVERFLOW", "1")
    monkeypatch.setenv("COMPAIR_DB_POOL_TIMEOUT_SEC", "7")

    assert db._pooled_engine_kwargs(default_pool_size=10, default_max_overflow=0) == {
        "pool_size": 2,
        "max_overflow": 1,
        "pool_timeout": 7,
    }
