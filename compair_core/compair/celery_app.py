from __future__ import annotations

from types import SimpleNamespace

try:
    from compair_cloud.celery_app import celery_app  # type: ignore
except (ImportError, ModuleNotFoundError):

    class _NoopCelery:
        def __init__(self) -> None:
            self.conf = SimpleNamespace(beat_schedule={})

        def task(self, func=None, *args, **kwargs):
            def decorator(fn):
                return fn

            return decorator(func) if func else decorator

        def send_task(self, *args, **kwargs):
            raise RuntimeError("Celery is not available in the Compair Core edition.")

    celery_app = _NoopCelery()
