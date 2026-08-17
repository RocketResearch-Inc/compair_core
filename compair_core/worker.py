"""Executable database-backed baseline worker.

The command line deliberately accepts only lifecycle flags. Configuration,
credentials, and provider identity come from the existing typed environment
settings; durable jobs are selected by opaque database identity.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import sessionmaker

from .compair.retrieval.continuation_worker import BaselineContinuationWorker
from .compair.retrieval.control_plane import BaselineControlPlaneService
from .compair.retrieval.database_worker import (
    BaselineDatabaseWorker,
    DatabaseJobScheduler,
    DatabaseWorkerError,
    DatabaseWorkerRegistry,
    ExistingServiceDispatcher,
)
from .compair.retrieval.embedding import (
    require_configured_baseline_embedding_adapter,
)
from .compair.retrieval.index_continuation import (
    BaselineCompatibleIndexJobService,
)
from .compair.retrieval.ingestion import CorpusIngestionService
from .compair.retrieval.run_operator import BaselineRunRuntime
from .db import engine
from .server.settings import Settings, get_settings

_LOGGER = logging.getLogger("compair_core.worker")


def build_database_worker(
    *,
    settings: Settings | Any | None = None,
) -> BaselineDatabaseWorker:
    selected = settings or get_settings()
    if selected.baseline_worker_mode != "database":
        raise DatabaseWorkerError("worker_mode_manual")
    sessions = sessionmaker(engine, expire_on_commit=False)
    continuation = BaselineContinuationWorker(
        engine,
        CorpusIngestionService(sessions),
    )
    index_service = BaselineCompatibleIndexJobService(
        engine,
        lambda: require_configured_baseline_embedding_adapter(selected),
    )
    runtime = BaselineRunRuntime(engine, selected)
    registry = DatabaseWorkerRegistry(
        engine,
        heartbeat_ttl=timedelta(
            seconds=int(selected.baseline_worker_heartbeat_ttl_seconds)
        ),
    )
    scheduler = DatabaseJobScheduler(
        engine,
        poll_interval_seconds=float(selected.baseline_worker_poll_interval_seconds),
        maximum_backoff_seconds=float(selected.baseline_worker_max_backoff_seconds),
    )
    worker_instance_id = str(uuid4())
    worker = BaselineDatabaseWorker(
        registry=registry,
        scheduler=scheduler,
        dispatcher=ExistingServiceDispatcher(
            engine,
            worker_instance_id=worker_instance_id,
            continuation_worker=continuation,
            index_service=index_service,
            run_operator=runtime.operator,
        ),
        cleanup_callbacks=(
            BaselineControlPlaneService(engine).expire_staging_sessions,
            runtime.jobs.cleanup_protected_payloads,
        ),
        worker_instance_id=worker_instance_id,
        heartbeat_interval_seconds=float(
            selected.baseline_worker_heartbeat_interval_seconds
        ),
        cleanup_interval_seconds=float(
            selected.baseline_worker_cleanup_interval_seconds
        ),
        maximum_attempts=int(selected.baseline_worker_max_attempts),
        poll_interval_seconds=float(selected.baseline_worker_poll_interval_seconds),
        maximum_backoff_seconds=float(selected.baseline_worker_max_backoff_seconds),
    )
    return worker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="compair-core-worker")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--once",
        action="store_true",
        help="process at most one eligible durable job and exit",
    )
    mode.add_argument(
        "--poll",
        action="store_true",
        help="poll the durable database queue until signalled",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    worker_factory: Callable[[], BaselineDatabaseWorker] = build_database_worker,
) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    try:
        worker = worker_factory()
        worker.start()
    except DatabaseWorkerError as exc:
        _LOGGER.error("baseline_worker event=start_failed reason=%s", exc.code)
        return 2

    stop_event = threading.Event()

    def drain(_signum: int, _frame: object) -> None:
        worker.request_draining()
        stop_event.set()

    previous: dict[int, Any] = {}
    try:
        if args.poll:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous[signum] = signal.signal(signum, drain)
            worker.poll(stop_event)
            return 0
        result = worker.run_once()
        if result is None:
            return 0
        return 1 if result.state in {"failed", "retryable_failed"} else 0
    finally:
        worker.close()
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":  # pragma: no cover - exercised by entry-point smoke
    raise SystemExit(main())
