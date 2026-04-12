"""Lightweight feature-flag helpers shared by Core and Cloud."""

from __future__ import annotations

import os


_TRUE_VALUES = {"1", "true", "yes", "on", "y"}


def _env_flag(*names: str, default: bool = False) -> bool:
    for name in names:
        if not name:
            continue
        value = os.getenv(name)
        if value is None:
            continue
        return value.strip().lower() in _TRUE_VALUES
    return default


def review_now_backend_enabled(*, edition: str | None = None) -> bool:
    resolved_edition = (edition or os.getenv("COMPAIR_EDITION", "core")).strip().lower() or "core"
    if resolved_edition != "cloud":
        return True
    return _env_flag("COMPAIR_REVIEW_NOW_ENABLED", "REVIEW_NOW_ENABLED", default=False)


def review_now_disabled_detail(*, edition: str | None = None) -> str | None:
    resolved_edition = (edition or os.getenv("COMPAIR_EDITION", "core")).strip().lower() or "core"
    if review_now_backend_enabled(edition=resolved_edition):
        return None
    return "Now review is disabled for this Cloud deployment. Set COMPAIR_REVIEW_NOW_ENABLED=1 on the backend to enable it."
