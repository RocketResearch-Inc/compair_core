"""Lifecycle-owned periodic service resource metrics."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from typing import Any

import psutil

from ..compair.logger import log_event

logger = logging.getLogger("compair.core.resource_metrics")

MetricSampler = Callable[[str], None]
ThreadFactory = Callable[..., threading.Thread]


def _sample_service_resources(service_name: str) -> None:
    process = psutil.Process(os.getpid())
    memory_mb = round(process.memory_info().rss / 1024 / 1024, 2)
    cpu_percent = process.cpu_percent(interval=1)
    log_event(
        "service_resource",
        service=service_name,
        memory_mb=memory_mb,
        cpu_percent=cpu_percent,
    )


class ServiceResourceMetricsReporter:
    """One periodic metrics worker with deterministic start/close semantics."""

    def __init__(
        self,
        service_name: str = "backend",
        *,
        interval_seconds: float = 300,
        sampler: MetricSampler = _sample_service_resources,
        thread_factory: ThreadFactory = threading.Thread,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("resource metrics interval must be positive")
        self._service_name = service_name
        self._interval_seconds = interval_seconds
        self._sampler = sampler
        self._thread_factory = thread_factory
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def thread(self) -> threading.Thread | None:
        with self._lock:
            return self._thread

    def start(self) -> None:
        """Start exactly one worker, rolling back state if startup fails."""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            thread = self._thread_factory(
                target=self._run,
                name=f"compair-core-resource-metrics-{self._service_name}",
                daemon=False,
            )
            self._thread = thread
            try:
                thread.start()
            except BaseException:
                self._thread = None
                self._stop.set()
                raise

    def close(self) -> None:
        """Signal and join the worker; repeated closes are safe."""

        with self._lock:
            self._stop.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._lock:
            if self._thread is thread:
                self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._sampler(self._service_name)
            except Exception as exc:  # noqa: BLE001 - monitoring must stay isolated
                logger.warning("resource log failed: %s", exc)
            if self._stop.wait(self._interval_seconds):
                return


def attach_service_resource_metrics(
    app: Any,
    *,
    reporter: ServiceResourceMetricsReporter | None = None,
) -> ServiceResourceMetricsReporter:
    """Bind one reporter to an ASGI application's startup and shutdown."""

    state_name = "service_resource_metrics_reporter"
    existing = getattr(app.state, state_name, None)
    if existing is not None:
        return existing
    owned = reporter or ServiceResourceMetricsReporter()
    setattr(app.state, state_name, owned)
    app.router.add_event_handler("startup", owned.start)
    app.router.add_event_handler("shutdown", owned.close)
    return owned
