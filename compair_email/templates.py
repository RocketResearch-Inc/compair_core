from __future__ import annotations

try:
    from compair_cloud.compair_email.templates import *  # type: ignore
except (ImportError, ModuleNotFoundError):
    from .templates_core import *  # type: ignore
