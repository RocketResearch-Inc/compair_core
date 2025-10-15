from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from .models import Activity


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
    activity = Activity(
        user_id=user_id,
        group_id=group_id,
        action=action,
        object_id=object_id,
        object_name=object_name,
        object_type=object_type,
        timestamp=datetime.now(timezone.utc),
    )
    session.add(activity)
    session.commit()
