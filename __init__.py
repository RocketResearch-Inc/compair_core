"""
Top-level package for compair-core.
Exposes the core FastAPI stack so users can import `compair_core.server.*`.
"""

from . import compair, server, compair_email  # noqa: F401

__all__ = ["compair", "server", "compair_email"]
