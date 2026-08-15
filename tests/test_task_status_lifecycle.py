import asyncio
from datetime import datetime, timedelta, timezone

from compair_core import api
from compair_core.compair import tasks


def test_process_doc_propagates_query_and_traces_only_provenance(monkeypatch):
    query = "diff --git a/source.py b/source.py\n+private change\n"
    dispatched = {}
    events = []

    async def read_payload(request):
        return {
            "doc_id": "document",
            "doc_text": "body",
            "retrieval_query": query,
        }

    class Query:
        def filter(self, *args, **kwargs):
            return self

        def first(self):
            return type("Document", (), {"author_id": "user"})()

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def query(self, model):
            return Query()

    monkeypatch.setattr(api, "_read_process_doc_payload", read_payload)
    monkeypatch.setattr(api, "_stage_process_doc_payload", lambda **kwargs: "storage-key")
    monkeypatch.setattr(api.compair, "Session", Session)
    monkeypatch.setattr(
        api,
        "_dispatch_process_document_task",
        lambda **kwargs: dispatched.update(kwargs) or type("Task", (), {"id": "task"})(),
    )
    monkeypatch.setattr(
        api,
        "log_event",
        lambda name, **values: events.append((name, values)),
    )
    user = type("User", (), {"user_id": "user", "status": "active"})()
    analytics = type("Analytics", (), {"track": lambda self, *args: None})()
    settings = type("Settings", (), {"edition": "core"})()

    asyncio.run(
        api.process_doc(
            request=object(),
            current_user=user,
            analytics=analytics,
            storage=object(),
            settings=settings,
        )
    )

    assert dispatched["retrieval_query"] == query
    received = next(values for name, values in events if name == "process_doc_request_received")
    assert received["retrieval_query_sha256"]
    assert received["retrieval_query_length"] == len(query)
    assert received["retrieval_query_origin"] == "explicit"
    assert query not in repr(received)


def test_api_dispatch_preserves_explicit_retrieval_query(monkeypatch):
    captured = {}

    class Conf:
        broker_url = (
            "rediss://worker:secret@redis.example/0?ssl_cert_reqs=required"
        )
        broker_use_ssl = False
        result_extended = False
        task_always_eager = False
        task_protocol = 2

    class App:
        conf = Conf()

    class Task:
        app = App()

        @staticmethod
        def apply_async(*args, **kwargs):
            captured.update(kwargs)
            return object()

    query = "diff --git a/source.py b/source.py\n+new_value = 2\n"
    monkeypatch.setattr(api, "process_document_celery", Task())

    api._dispatch_process_document_task(
        "user",
        "document",
        "body",
        True,
        retrieval_query=query,
    )

    assert captured["kwargs"]["retrieval_query"] == query
    assert query not in captured["argsrepr"]
    assert query not in captured["kwargsrepr"]


def test_api_dispatch_without_query_keeps_legacy_task_kwargs(monkeypatch):
    captured = {}

    class Task:
        @staticmethod
        def delay(*args, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setattr(api, "process_document_celery", Task())

    api._dispatch_process_document_task("user", "document", "body", True)

    assert "retrieval_query" not in captured


def test_core_task_preserves_retrieval_query_for_process_document(monkeypatch):
    captured = {}
    events = []
    user = type("UserRow", (), {"user_id": "user"})()
    document = type("DocumentRow", (), {"document_id": "document", "groups": []})()

    class Query:
        def __init__(self, row):
            self.row = row

        def options(self, *args, **kwargs):
            return self

        def filter(self, *args, **kwargs):
            return self

        def first(self):
            return self.row

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def query(self, model):
            return Query(user if model is UserModel else document)

        def add(self, value):
            return None

    class SessionMaker:
        def __new__(cls):
            return Session()

    class UserModel:
        user_id = object()

    class DocumentModel:
        document_id = object()
        groups = object()

    def process_document(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        tasks,
        "_lazy_components",
        lambda: (
            SessionMaker,
            object,
            object,
            lambda name, **kwargs: events.append((name, kwargs)),
            process_document,
            DocumentModel,
            UserModel,
            lambda text: [],
            lambda text: text,
        ),
    )
    monkeypatch.setattr(tasks, "joinedload", lambda value: value)
    query = "diff --git a/a.py b/a.py\n-old\n+new\n"

    task_status = tasks.process_document_task(
        "user",
        "document",
        "body",
        retrieval_query=query,
    )

    assert captured["retrieval_query"] == query
    assert task_status == {"chunk_task_ids": []}
    assert query not in repr(task_status)
    assert query not in repr(events)


def test_stale_prechunk_parent_recommends_smaller_resubmit(monkeypatch):
    monkeypatch.setenv("COMPAIR_STATUS_STALE_AFTER_SEC", "900")
    stale_progress = (datetime.now(timezone.utc) - timedelta(seconds=901)).isoformat()

    lifecycle, health, retryable, terminal, recommended_action = api._task_lifecycle(
        "PROGRESS",
        {
            "stage": "preparing",
            "last_progress_at": stale_progress,
        },
        None,
    )

    assert lifecycle == "running"
    assert health == "stale_prechunk"
    assert retryable is False
    assert terminal is False
    assert recommended_action == "resubmit_smaller_or_inspect_worker"


def test_stale_chunked_parent_keeps_worker_inspection_guidance(monkeypatch):
    monkeypatch.setenv("COMPAIR_STATUS_STALE_AFTER_SEC", "900")
    stale_progress = (datetime.now(timezone.utc) - timedelta(seconds=901)).isoformat()

    lifecycle, health, retryable, terminal, recommended_action = api._task_lifecycle(
        "PROGRESS",
        {
            "stage": "indexing",
            "last_progress_at": stale_progress,
        },
        None,
    )

    assert lifecycle == "running"
    assert health == "stale"
    assert retryable is False
    assert terminal is False
    assert recommended_action == "inspect_worker"
