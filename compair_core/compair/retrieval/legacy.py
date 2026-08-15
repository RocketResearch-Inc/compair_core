"""Thin adapter for invoking the existing legacy selector unchanged."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from .types import RetrievalRequest

ChunkT = TypeVar("ChunkT")


@dataclass(frozen=True, slots=True)
class LegacyRetriever(Generic[ChunkT]):
    """Delegate to Core's existing selector without copying or reordering."""

    selector: Callable[[], list[ChunkT]]
    name: str = "legacy"

    def retrieve(self, request: RetrievalRequest | None = None) -> list[ChunkT]:
        # The typed request reaches the selected engine, but Phase 2A keeps the
        # legacy selector's own query construction authoritative.
        del request
        return self.selector()
