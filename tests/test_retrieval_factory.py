from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from compair_core.compair.retrieval import (
    BaselineV1Retriever,
    LegacyRetriever,
    RetrievalQueryOrigin,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStatus,
    UnknownRetrievalEngineError,
    create_retrieval_engine,
    retrieval_query_provenance,
    retrieve_reference_evidence,
)


def test_factory_defaults_to_legacy_and_preserves_list_identity_and_order() -> None:
    chunks = [object(), object(), object()]

    engine = create_retrieval_engine(legacy_selector=lambda: chunks)
    selected = retrieve_reference_evidence(legacy_selector=lambda: chunks)

    assert isinstance(engine, LegacyRetriever)
    assert engine.name == "legacy"
    assert selected is chunks
    assert selected == chunks


def test_explicit_baseline_without_phase_2_inputs_is_insufficient_not_legacy() -> None:
    legacy_called = False

    def legacy_selector() -> list[object]:
        nonlocal legacy_called
        legacy_called = True
        return [object()]

    result = retrieve_reference_evidence(
        engine_name="baseline_v1",
        legacy_selector=legacy_selector,
    )

    assert isinstance(result, RetrievalResult)
    assert result.status is RetrievalStatus.INSUFFICIENT
    assert result.error is not None
    assert result.error.code == "explicit_retrieval_query_absent"
    assert result.fallback_engine is None
    assert legacy_called is False


class NeverCalledDenseProvider:
    model = "fixture/not-called"
    revision = "toy-v1"
    dimension = 2

    def embed(self, texts):
        raise AssertionError("precondition failure must occur before embedding")


def test_explicit_baseline_propagates_its_own_insufficient_precondition() -> None:
    request = RetrievalRequest(
        request_id="phase-1b-request",
        changed_repository=Path("changed"),
        repository_roots=(),
        corpus_version="",
        retrieval_query=None,
        retrieval_query_origin=RetrievalQueryOrigin.ABSENT,
        corpus_complete=False,
    )

    result = retrieve_reference_evidence(
        engine_name="baseline_v1",
        baseline_retriever=BaselineV1Retriever(NeverCalledDenseProvider()),
        request=request,
    )

    assert isinstance(result, RetrievalResult)
    assert result.status is RetrievalStatus.INSUFFICIENT
    assert result.error is not None
    assert result.error.code == "explicit_retrieval_query_absent"
    assert result.fallback_engine is None


def test_query_provenance_contains_only_hash_length_and_origin() -> None:
    query = "diff --git a/old.py b/new.py\n+secret marker\n"

    provenance = retrieval_query_provenance(query, RetrievalQueryOrigin.EXPLICIT)
    fields = provenance.trace_fields()

    assert fields == {
        "retrieval_query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "retrieval_query_length": len(query),
        "retrieval_query_origin": "explicit",
    }
    assert query not in repr(fields)

    assert retrieval_query_provenance(
        None, RetrievalQueryOrigin.ABSENT
    ).trace_fields() == {
        "retrieval_query_sha256": None,
        "retrieval_query_length": 0,
        "retrieval_query_origin": "absent",
    }


def test_missing_query_never_calls_baseline_or_legacy() -> None:
    calls = []

    class BaselineSpy:
        def retrieve(self, request):
            calls.append(("baseline", request))
            raise AssertionError("missing query must fail before baseline retrieval")

    request = RetrievalRequest(
        request_id="missing-query",
        changed_repository=None,
        repository_roots=(),
        corpus_version="",
        corpus_complete=False,
    )
    result = retrieve_reference_evidence(
        engine_name="baseline_v1",
        legacy_selector=lambda: calls.append(("legacy", None)),
        baseline_retriever=BaselineSpy(),
        request=request,
    )

    assert result.status is RetrievalStatus.INSUFFICIENT
    assert result.error is not None
    assert result.error.code == "explicit_retrieval_query_absent"
    assert calls == []


def test_phase_2b1_does_not_enable_baseline_findings() -> None:
    request = RetrievalRequest(
        request_id="phase-2b1",
        changed_repository=None,
        repository_roots=(),
        corpus_version="durable-generation",
        retrieval_query="diff --git a/a.py b/a.py\n-old\n+new\n",
        retrieval_query_origin=RetrievalQueryOrigin.EXPLICIT,
        corpus_complete=True,
    )

    result = retrieve_reference_evidence(
        engine_name="baseline_v1",
        request=request,
    )

    assert result.status is RetrievalStatus.INSUFFICIENT
    assert result.error is not None
    assert result.error.code == "baseline_inputs_unavailable"
    assert result.evidence == ()
    assert result.fallback_engine is None


@pytest.mark.parametrize("engine_name", ["", "unregistered"])
def test_unknown_engine_fails_instead_of_falling_back(engine_name: str) -> None:
    with pytest.raises(UnknownRetrievalEngineError, match="unknown retrieval engine"):
        retrieve_reference_evidence(
            engine_name=engine_name,
            legacy_selector=lambda: [object()],
        )
