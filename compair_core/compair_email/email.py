from __future__ import annotations

try:  # Cloud build ships the premium mailer
    from compair_cloud.compair_email.email import *  # type: ignore
except (ImportError, ModuleNotFoundError):
    from .email_core import *  # type: ignore
