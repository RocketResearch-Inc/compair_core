from __future__ import annotations

from types import SimpleNamespace

import pytest

from compair_core import api
from compair_core.compair.retrieval.transport import (
    REDACTED_TASK_ARGS_REPR,
    REDACTED_TASK_KWARGS_REPR,
    RetrievalQueryTransportPolicyError,
    RetrievalQueryTransportStatus,
    assess_retrieval_query_transport,
)


class CapturingTask:
    def __init__(self, **conf):
        defaults = {
            "broker_url": "redis://worker:secret@redis.example/0",
            "broker_write_url": None,
            "broker_password": None,
            "broker_use_ssl": False,
            "result_extended": False,
            "task_always_eager": False,
            "task_protocol": 2,
            "task_send_sent_event": True,
            "worker_send_task_events": True,
        }
        defaults.update(conf)
        self.app = SimpleNamespace(conf=SimpleNamespace(**defaults))
        self.calls: list[dict[str, object]] = []
        self.delay_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def apply_async(self, **options):
        self.calls.append(options)
        return SimpleNamespace(id="task-id")

    def delay(self, *args, **kwargs):
        self.delay_calls.append((args, kwargs))
        return SimpleNamespace(id="legacy-task-id")


def test_production_insecure_broker_rejects_before_dispatch(monkeypatch):
    task = CapturingTask(
        broker_url="redis://worker:secret@redis.production.internal/0"
    )
    monkeypatch.setattr(api, "process_document_celery", task)

    with pytest.raises(RetrievalQueryTransportPolicyError) as exc_info:
        api._dispatch_process_document_task(
            "user",
            "document",
            "body",
            True,
            retrieval_query="private query sentinel",
        )

    assert exc_info.value.capability.status is RetrievalQueryTransportStatus.UNAVAILABLE
    assert exc_info.value.capability.reason == "broker_encryption_required"
    assert task.calls == []
    assert task.delay_calls == []
    assert "private query sentinel" not in str(exc_info.value)


def test_authenticated_verified_tls_dispatches_with_redacted_representations(
    monkeypatch,
):
    task = CapturingTask(
        broker_url=(
            "rediss://worker:secret@redis.production.internal/0"
            "?ssl_cert_reqs=required"
        )
    )
    monkeypatch.setattr(api, "process_document_celery", task)
    query = "diff --git a/private.py b/private.py\n+secret = True\n"

    result = api._dispatch_process_document_task(
        "user",
        "document",
        "body",
        True,
        retrieval_query=query,
    )

    assert result.id == "task-id"
    assert len(task.calls) == 1
    call = task.calls[0]
    assert call["kwargs"]["retrieval_query"] == query
    assert call["argsrepr"] == REDACTED_TASK_ARGS_REPR
    assert call["kwargsrepr"] == REDACTED_TASK_KWARGS_REPR
    assert query not in call["argsrepr"]
    assert query not in call["kwargsrepr"]
    assert task.delay_calls == []


def test_local_exception_is_explicit_and_rejects_remote_insecure_brokers():
    loopback = CapturingTask(broker_url="redis://localhost:6379/0")

    disabled = assess_retrieval_query_transport(loopback)
    enabled = assess_retrieval_query_transport(
        loopback,
        allow_insecure_local_transport=True,
    )
    remote = assess_retrieval_query_transport(
        CapturingTask(broker_url="redis://redis.internal:6379/0"),
        allow_insecure_local_transport=True,
    )
    memory = assess_retrieval_query_transport(
        CapturingTask(broker_url="memory://"),
        allow_insecure_local_transport=True,
    )

    assert disabled.status is RetrievalQueryTransportStatus.UNAVAILABLE
    assert enabled.status is RetrievalQueryTransportStatus.LOCALLY_OVERRIDDEN
    assert enabled.reason == "explicit_insecure_local_transport_override"
    assert remote.status is RetrievalQueryTransportStatus.UNAVAILABLE
    assert memory.status is RetrievalQueryTransportStatus.LOCALLY_OVERRIDDEN


def test_extended_results_and_unredactable_protocols_fail_closed():
    secure_url = "rediss://worker:secret@redis.example/0?ssl_cert_reqs=required"

    extended = assess_retrieval_query_transport(
        CapturingTask(broker_url=secure_url, result_extended=True)
    )
    protocol_v1 = assess_retrieval_query_transport(
        CapturingTask(broker_url=secure_url, task_protocol=1)
    )

    assert extended.reason == "celery_result_extended_must_be_disabled"
    assert protocol_v1.reason == "celery_task_protocol_v2_required"


def test_effective_write_url_and_separate_celery_password_are_enforced():
    task = CapturingTask(
        broker_url="rediss://worker:secret@safe-read.example/0?ssl_cert_reqs=required",
        broker_write_url=(
            "rediss://worker@safe-write.example/0?ssl_cert_reqs=required"
        ),
        broker_password="configured-separately",
    )

    capability = assess_retrieval_query_transport(task)

    assert capability.status is RetrievalQueryTransportStatus.SAFE


def test_capability_indicator_is_credential_and_query_free():
    query = "private query sentinel"
    task = CapturingTask(
        broker_url="rediss://worker:secret@redis.example/0?ssl_cert_reqs=required"
    )

    payload = assess_retrieval_query_transport(task).as_dict()

    assert payload["status"] == "safe"
    assert payload["broker_scheme"] == "rediss"
    assert payload["task_arguments"] == "redacted"
    assert payload["task_sent_events_enabled"] is True
    assert payload["worker_task_events_enabled"] is True
    assert "worker:secret" not in repr(payload)
    assert "secret" not in repr(payload)
    assert query not in repr(payload)


def test_events_logs_status_and_result_observables_use_no_raw_query(
    monkeypatch,
    caplog,
):
    task = CapturingTask(
        broker_url="rediss://worker:secret@redis.example/0?ssl_cert_reqs=required",
        task_send_sent_event=True,
        worker_send_task_events=True,
    )
    monkeypatch.setattr(api, "process_document_celery", task)
    query = "private event and log sentinel"

    api._dispatch_process_document_task(
        "user",
        "document",
        "body",
        True,
        retrieval_query=query,
    )
    call = task.calls[0]
    simulated_task_sent_event = {
        "args": call["argsrepr"],
        "kwargs": call["kwargsrepr"],
    }
    simulated_worker_received_log = (
        f"args={call['argsrepr']} kwargs={call['kwargsrepr']}"
    )
    status_payload = {
        "status": "SUCCESS",
        "result": {"chunk_task_ids": []},
    }

    observables = (
        simulated_task_sent_event,
        simulated_worker_received_log,
        status_payload,
        caplog.text,
    )
    assert all(query not in repr(observable) for observable in observables)


def test_no_query_uses_unchanged_legacy_delay_path(monkeypatch):
    task = CapturingTask(
        broker_url="redis://remote-insecure.internal:6379/0",
        result_extended=True,
        task_protocol=1,
    )
    monkeypatch.setattr(api, "process_document_celery", task)

    result = api._dispatch_process_document_task(
        "user",
        "document",
        "body",
        True,
    )

    assert result.id == "legacy-task-id"
    assert task.calls == []
    assert len(task.delay_calls) == 1
    assert "retrieval_query" not in task.delay_calls[0][1]
