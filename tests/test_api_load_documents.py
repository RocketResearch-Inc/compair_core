from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi.routing import APIRoute

from compair_core import api


def _route_endpoint(path: str):
    for route in api.router.routes:
        if isinstance(route, APIRoute) and route.path == path:
            return route.endpoint
    raise AssertionError(f"Route not found: {path}")


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def unique(self):
        return self

    def fetchall(self):
        return self._rows


class _FakeQuery:
    def __init__(self, total_count: int):
        self.total_count = total_count
        self.limited = False
        self.limit_value = None

    def filter(self, *args, **kwargs):
        return self

    def join(self, *args, **kwargs):
        return self

    def outerjoin(self, *args, **kwargs):
        return self

    def options(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def count(self):
        return self.total_count

    def offset(self, *args, **kwargs):
        return self

    def limit(self, limit_value):
        self.limited = True
        self.limit_value = limit_value
        return self


class _FakeSession:
    def __init__(self, rows):
        self.rows = rows
        self.executed_query = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def query(self, *args, **kwargs):
        return _FakeQuery(total_count=len(self.rows))

    def execute(self, query):
        if not getattr(query, "limited", False):
            raise AssertionError("load_documents executed an unbounded document query")
        self.executed_query = query
        return _FakeResult(self.rows)


def _user(user_id: str = "user-1"):
    return SimpleNamespace(
        user_id=user_id,
        username="user@example.com",
        name="User",
        datetime_registered=datetime(2024, 1, 1, tzinfo=timezone.utc),
        status="active",
        groups=[],
    )


def _document(user):
    return SimpleNamespace(
        document_id="doc-1",
        user_id=user.user_id,
        author_id=user.user_id,
        groups=[],
        user=user,
        title="Large snapshot",
        content="large content",
        doc_type="document",
        datetime_created=datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime_modified=datetime(2024, 1, 2, tzinfo=timezone.utc),
        is_published=True,
        file_key=None,
        image_key=None,
        topic_tags=None,
    )


def test_load_documents_executes_only_paginated_query(monkeypatch):
    user = _user()
    fake_session = _FakeSession(rows=[(_document(user),)])
    monkeypatch.setattr(api.compair, "Session", lambda: fake_session)

    endpoint = _route_endpoint("/load_documents")
    result = endpoint(page=1, page_size=1000, current_user=user)

    assert result["total_count"] == 1
    assert len(result["documents"]) == 1
    assert fake_session.executed_query is not None
    assert fake_session.executed_query.limit_value == 50
