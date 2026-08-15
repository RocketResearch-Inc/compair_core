"""Explicit retrieval-engine construction and the internal invocation seam."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from .legacy import LegacyRetriever
from .types import RetrievalError, RetrievalRequest, RetrievalResult, RetrievalStatus

DEFAULT_RETRIEVAL_ENGINE = "legacy"
BASELINE_RETRIEVAL_ENGINE = "baseline_v1"
ChunkT = TypeVar("ChunkT")


class UnknownRetrievalEngineError(ValueError):
    """Raised when an engine name is not explicitly registered."""


def validate_retrieval_engine_name(engine_name: object) -> str:
    """Validate explicit configuration without trimming or fallback."""

    if not isinstance(engine_name, str) or not engine_name:
        raise UnknownRetrievalEngineError(
            "unknown retrieval engine: configuration must be a non-empty string"
        )
    if engine_name not in {DEFAULT_RETRIEVAL_ENGINE, BASELINE_RETRIEVAL_ENGINE}:
        raise UnknownRetrievalEngineError(f"unknown retrieval engine: {engine_name}")
    return engine_name


class BaselineRetriever(Protocol):
    def retrieve(self, request: RetrievalRequest) -> RetrievalResult: ...


@dataclass(frozen=True, slots=True)
class BaselineV1Invocation:
    """Phase-aware wrapper around the pure baseline implementation."""

    retriever: BaselineRetriever | None = None
    name: str = BASELINE_RETRIEVAL_ENGINE

    def retrieve(self, request: RetrievalRequest | None = None) -> RetrievalResult:
        if request is None or not request.has_usable_explicit_query:
            return _baseline_query_unavailable(request)
        if request is not None and self.retriever is not None:
            return self.retriever.retrieve(request)
        return _baseline_inputs_unavailable(request)


def _baseline_query_unavailable(
    request: RetrievalRequest | None,
) -> RetrievalResult:
    """Reject absent/blank/non-explicit queries without invoking or falling back."""

    return RetrievalResult(
        request_id=request.request_id if request is not None else "",
        status=RetrievalStatus.INSUFFICIENT,
        corpus_version=request.corpus_version if request is not None else "",
        config_fingerprint=hashlib.sha256(
            b"baseline_v1:explicit_retrieval_query_absent"
        ).hexdigest(),
        embedding_model="unavailable",
        embedding_revision="unavailable",
        embedding_dimension=0,
        error=RetrievalError(
            code="explicit_retrieval_query_absent",
            message="baseline_v1 requires a non-empty explicit change-set query",
        ),
        query_provenance=(
            request.query_provenance if request is not None else None
        ),
    )


def _baseline_inputs_unavailable(
    request: RetrievalRequest | None,
) -> RetrievalResult:
    """Return an explicit Phase 1B result without invoking another engine."""

    return RetrievalResult(
        request_id=request.request_id if request is not None else "",
        status=RetrievalStatus.INSUFFICIENT,
        corpus_version=request.corpus_version if request is not None else "",
        config_fingerprint=hashlib.sha256(
            b"baseline_v1:phase_1b_inputs_unavailable"
        ).hexdigest(),
        embedding_model="unavailable",
        embedding_revision="unavailable",
        embedding_dimension=0,
        error=RetrievalError(
            code="baseline_inputs_unavailable",
            message=(
                "baseline_v1 requires a raw_git_diff_v1 request, a complete "
                "sibling-file corpus, and an injected dense provider"
            ),
        ),
        query_provenance=(
            request.query_provenance if request is not None else None
        ),
    )


def create_retrieval_engine(
    engine_name: str | None = None,
    *,
    legacy_selector: Callable[[], list[ChunkT]] | None = None,
    baseline_retriever: BaselineRetriever | None = None,
) -> LegacyRetriever[ChunkT] | BaselineV1Invocation:
    """Construct one registered engine; omitted names resolve to legacy."""

    selected_name = (
        DEFAULT_RETRIEVAL_ENGINE
        if engine_name is None
        else validate_retrieval_engine_name(engine_name)
    )
    if selected_name == DEFAULT_RETRIEVAL_ENGINE:
        if legacy_selector is None:
            raise ValueError("legacy retrieval requires the existing selector")
        return LegacyRetriever(legacy_selector)
    if selected_name == BASELINE_RETRIEVAL_ENGINE:
        return BaselineV1Invocation(baseline_retriever)
    raise UnknownRetrievalEngineError(f"unknown retrieval engine: {selected_name}")


def retrieve_reference_evidence(
    *,
    legacy_selector: Callable[[], list[ChunkT]] | None = None,
    engine_name: str | None = None,
    baseline_retriever: BaselineRetriever | None = None,
    request: RetrievalRequest | None = None,
) -> list[ChunkT] | RetrievalResult:
    """Invoke exactly one engine without implicit fallback."""

    engine = create_retrieval_engine(
        engine_name,
        legacy_selector=legacy_selector,
        baseline_retriever=baseline_retriever,
    )
    if isinstance(engine, LegacyRetriever):
        return engine.retrieve(request)
    return engine.retrieve(request)


# Phase 1B compatibility name. New integration code uses the evidence-oriented
# facade whose boundary includes discovery, ranking, and final selection.
invoke_retrieval = retrieve_reference_evidence
