"""
Top-level package for compair-core.
Exposes the core FastAPI stack so users can import `compair_core.server.*`.
"""

import sys

from . import compair as _compair
from . import server as _server
from . import compair_email as _compair_email

sys.modules.setdefault("compair", _compair)
sys.modules.setdefault("compair.server", _server)
sys.modules.setdefault("compair.compair_email", _compair_email)
