from __future__ import annotations

from sqlalchemy.orm import Session

try:
    from compair_cloud.default_groups import initialize_default_groups as cloud_initialize_default_groups  # type: ignore
except (ImportError, ModuleNotFoundError):
    cloud_initialize_default_groups = None


def initialize_default_groups(session: Session) -> None:
    """Core builds do not seed any default groups by default."""
    if cloud_initialize_default_groups:
        cloud_initialize_default_groups(session)
