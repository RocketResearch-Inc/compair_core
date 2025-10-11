from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.orm import Session

try:
    from compair_cloud.utils import log_activity as cloud_log_activity  # type: ignore
except (ImportError, ModuleNotFoundError):
    cloud_log_activity = None


def chunk_text(text: str) -> list[str]:
    chunks = text.split("\n\n")
    chunks = [c.strip() for c in chunks]
    return [c for c in chunks if c]


def generate_verification_token() -> tuple[str, datetime]:
    token = secrets.token_urlsafe(32)
    expiration = datetime.now(timezone.utc) + timedelta(hours=24)
    return token, expiration


def log_activity(
    session: Session,
    user_id: str,
    group_id: str,
    action: str,
    object_id: str,
    object_name: str,
    object_type: str,
) -> None:
    if cloud_log_activity:
        cloud_log_activity(
            session=session,
            user_id=user_id,
            group_id=group_id,
            action=action,
            object_id=object_id,
            object_name=object_name,
            object_type=object_type,
        )


def aggregate_usage_by_user() -> dict[str, Any]:
    if cloud_log_activity:
        from compair_cloud.utils import aggregate_usage_by_user as cloud_usage  # type: ignore

        return cloud_usage()
    return {}


def aggregate_service_resources() -> dict[str, Any]:
    if cloud_log_activity:
        from compair_cloud.utils import aggregate_service_resources as cloud_resources  # type: ignore

        return cloud_resources()
    return {}
