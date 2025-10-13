from __future__ import annotations

import json
import logging
from typing import Any, Callable

try:
    from compair_cloud.logger import log_event as cloud_log_event  # type: ignore
except (ImportError, ModuleNotFoundError):
    cloud_log_event: Callable[..., None] | None = None


if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

_LOGGER = logging.getLogger("compair.core")


def log_event(message: str, **fields: Any) -> None:
    """Emit a structured log entry for the core edition."""
    if cloud_log_event:
        cloud_log_event(message, **fields)
        return

    try:
        payload = json.dumps({"message": message, **fields}, default=str)
    except TypeError:
        payload = json.dumps({"message": message}, default=str)
    _LOGGER.info(payload)
