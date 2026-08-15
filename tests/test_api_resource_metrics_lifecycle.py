from __future__ import annotations

import threading

from compair_core import api
from compair_core.server.resource_metrics import (
    ServiceResourceMetricsReporter,
    attach_service_resource_metrics,
)


def _resource_metric_threads() -> list[threading.Thread]:
    return [
        thread
        for thread in threading.enumerate()
        if thread.name.startswith("compair-core-resource-metrics-")
    ]


def test_api_import_does_not_start_resource_metrics_thread() -> None:
    reporter = api.app.state.service_resource_metrics_reporter

    assert reporter.running is False
    assert _resource_metric_threads() == []


def test_reporter_closes_successful_worker_deterministically() -> None:
    sampled = threading.Event()
    reporter = ServiceResourceMetricsReporter(
        interval_seconds=60,
        sampler=lambda service: sampled.set(),
    )

    reporter.start()
    assert sampled.wait(timeout=1)
    thread = reporter.thread
    assert thread is not None
    assert thread.daemon is False
    assert reporter.running is True

    reporter.close()
    reporter.close()

    assert thread.is_alive() is False
    assert reporter.running is False
    assert reporter.thread is None
    assert _resource_metric_threads() == []


def test_reporter_closes_after_sampler_failure(caplog) -> None:
    sampled = threading.Event()

    def failing_sampler(service_name: str) -> None:
        sampled.set()
        raise RuntimeError("fixture sampling failure")

    reporter = ServiceResourceMetricsReporter(
        interval_seconds=60,
        sampler=failing_sampler,
    )
    reporter.start()
    assert sampled.wait(timeout=1)

    reporter.close()

    assert reporter.running is False
    assert _resource_metric_threads() == []
    assert "resource log failed: fixture sampling failure" in caplog.text


def test_reporter_rolls_back_failed_thread_start() -> None:
    class FailingThread:
        daemon = False

        def start(self) -> None:
            raise RuntimeError("fixture thread start failure")

        def is_alive(self) -> bool:
            return False

    reporter = ServiceResourceMetricsReporter(
        sampler=lambda service: None,
        thread_factory=lambda **kwargs: FailingThread(),
    )

    try:
        reporter.start()
    except RuntimeError as exc:
        assert str(exc) == "fixture thread start failure"
    else:  # pragma: no cover - protects the failure-path assertion
        raise AssertionError("thread startup unexpectedly succeeded")

    reporter.close()
    assert reporter.running is False
    assert reporter.thread is None
    assert _resource_metric_threads() == []


def test_app_lifecycle_owns_reporter_thread() -> None:
    from fastapi import FastAPI

    app = FastAPI()
    sampled = threading.Event()
    reporter = ServiceResourceMetricsReporter(
        interval_seconds=60,
        sampler=lambda service: sampled.set(),
    )

    attached = attach_service_resource_metrics(app, reporter=reporter)
    startup = app.router.on_startup[-1]
    shutdown = app.router.on_shutdown[-1]

    assert attached is reporter
    assert reporter.running is False

    startup()
    assert sampled.wait(timeout=1)
    assert reporter.running is True

    shutdown()
    assert reporter.running is False
    assert _resource_metric_threads() == []
