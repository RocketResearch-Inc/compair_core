from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from compair_core.compair.retrieval import (
    BaselineV1Retriever,
    RetrievalQueryOrigin,
    RetrievalRequest,
    RetrievalStatus,
    bm25_scores,
    enumerate_file_candidates,
    frozen_tokens,
    reciprocal_rank_fusion,
)


class FixtureDenseProvider:
    model = "fixture/dense-dot"
    revision = "toy-v1"

    def __init__(
        self,
        *,
        vectors_by_path: dict[str, Sequence[float]] | None = None,
        query_vector: Sequence[float] = (1.0, 0.0),
        document_vector: Sequence[float] = (1.0, 0.0),
    ) -> None:
        self.vectors_by_path = vectors_by_path or {}
        self.query_vector = tuple(query_vector)
        self.document_vector = tuple(document_vector)
        self.dimension = len(self.query_vector)

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors: list[Sequence[float]] = []
        for text in texts:
            if text.startswith("Repository file: "):
                path = text.splitlines()[0].removeprefix("Repository file: ")
                vectors.append(self.vectors_by_path.get(path, self.document_vector))
            else:
                vectors.append(self.query_vector)
        return vectors


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _request(
    workspace: Path,
    *,
    query: str = "unmatched-change-token",
    corpus_complete: bool = True,
) -> RetrievalRequest:
    repositories = tuple(path for path in workspace.iterdir() if path.is_dir())
    return RetrievalRequest(
        request_id="toy-request",
        changed_repository=workspace / "changed",
        repository_roots=repositories,
        corpus_version="toy-corpus-v1",
        retrieval_query=query,
        retrieval_query_origin=RetrievalQueryOrigin.EXPLICIT,
        corpus_complete=corpus_complete,
    )


def test_frozen_tokenizer_and_stopwords() -> None:
    assert frozen_tokens("Foo.Bar and X 7 /API/v1-test snake_case") == [
        "foo.bar",
        "foo",
        "bar",
        "api/v1-test",
        "api",
        "v1",
        "test",
        "snake_case",
        "snake",
        "case",
    ]


def test_explicit_retrieval_query_reaches_baseline_dense_provider_unchanged(
    tmp_path: Path,
) -> None:
    query = "diff --git a/widget.py b/widget.py\n-old_widget\n+new_widget\n"
    _write(tmp_path / "changed" / "widget.py", "new_widget")
    _write(tmp_path / "peer" / "widget.py", "def new_widget(): pass")

    class CapturingProvider(FixtureDenseProvider):
        def __init__(self) -> None:
            super().__init__()
            self.calls = []

        def embed(self, texts):
            self.calls.append(list(texts))
            return super().embed(texts)

    provider = CapturingProvider()
    BaselineV1Retriever(provider).retrieve(_request(tmp_path, query=query))

    assert provider.calls[-1] == [query]


def test_exact_bm25_formula_uses_frozen_constants() -> None:
    scores = bm25_scores("alpha", ["alpha alpha beta", "beta gamma"])
    expected = 2.0 * math.log(2.0) * (4.0 * 2.5 / (4.0 + 1.725))

    assert scores[0] == pytest.approx(expected, rel=1e-15)
    assert scores[1] == 0.0


def test_rrf_is_equal_weight_one_based_and_k_60() -> None:
    scores = reciprocal_rank_fusion([1, 2], [2, 1])
    assert scores == [
        1.0 / 61.0 + 1.0 / 62.0,
        1.0 / 62.0 + 1.0 / 61.0,
    ]
    assert scores[0] != round(scores[0], 8)
    with pytest.raises(ValueError, match="one-based"):
        reciprocal_rank_fusion([0], [1])


def test_candidate_enumeration_is_stable_and_applies_all_exclusions(
    tmp_path: Path,
) -> None:
    changed = tmp_path / "changed"
    alpha = tmp_path / "alpha_repo"
    zeta = tmp_path / "zeta_repo"
    _write(changed / "own.txt", "not a peer")
    _write(zeta / "src" / "later.txt", "zeta")
    _write(alpha / "src" / "first.txt", "alpha")
    for excluded in (".git", ".compair", "build", "dist", "node_modules"):
        _write(alpha / excluded / "ignored.txt", "ignored")
    (alpha / "binary.dat").write_bytes(b"\xff\xfe")
    (alpha / "too-large.txt").write_bytes(b"x" * 200_001)
    outside = tmp_path / "outside.txt"
    _write(outside, "outside")
    internal_link = alpha / "internal-link.txt"
    internal_link.symlink_to(alpha / "src" / "first.txt")
    escaping_link = alpha / "escape.txt"
    escaping_link.symlink_to(outside)

    candidates = enumerate_file_candidates(
        (zeta, changed, alpha), changed_repository=changed
    )

    assert [candidate.path for candidate in candidates] == [
        "alpha_repo/src/first.txt",
        "zeta_repo/src/later.txt",
    ]
    assert [candidate.content for candidate in candidates] == ["alpha", "zeta"]
    assert internal_link.is_symlink()
    assert escaping_link.is_symlink()
    assert all(
        candidate.relative_path not in {internal_link.name, escaping_link.name}
        for candidate in candidates
    )


def test_dense_lane_is_unnormalized_dot_product_and_full_path_breaks_ties(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "changed" / "diff.txt", "changed")
    _write(tmp_path / "peer" / "a.txt", "plain one")
    _write(tmp_path / "peer" / "b.txt", "plain two")
    _write(tmp_path / "peer" / "c.txt", "plain three")
    provider = FixtureDenseProvider(
        vectors_by_path={
            "peer/a.txt": (10.0, 0.0),
            "peer/b.txt": (1.0, 1.0),
            "peer/c.txt": (1.0, 1.0),
        },
        query_vector=(2.0, 1.0),
    )

    result = BaselineV1Retriever(provider).retrieve(_request(tmp_path))
    by_path = {candidate.path: candidate for candidate in result.candidates}

    assert result.status is RetrievalStatus.OK
    assert by_path["peer/a.txt"].dense_score == 20.0
    assert by_path["peer/b.txt"].dense_score == 3.0
    assert by_path["peer/b.txt"].dense_rank < by_path["peer/c.txt"].dense_rank
    assert [candidate.path for candidate in result.candidates] == [
        "peer/a.txt",
        "peer/b.txt",
        "peer/c.txt",
    ]


def test_float32_dense_near_tie_matches_vendor_selected_order(tmp_path: Path) -> None:
    _write(tmp_path / "changed" / "diff.txt", "changed")
    _write(tmp_path / "peer" / "a.txt", "needle")
    _write(tmp_path / "peer" / "x.txt", "needle needle")
    _write(tmp_path / "peer" / "z.txt", "needle")
    provider = FixtureDenseProvider(
        vectors_by_path={
            "peer/a.txt": (1.0, 0.0),
            "peer/x.txt": (0.0, 0.0),
            "peer/z.txt": (1.0, 1e-8),
        },
        query_vector=(1.0, 1.0),
    )

    result = BaselineV1Retriever(provider).retrieve(_request(tmp_path, query="needle"))
    by_path = {candidate.path: candidate for candidate in result.candidates}

    assert sum(a * b for a, b in zip((1.0, 1e-8), (1.0, 1.0))) > 1.0
    assert by_path["peer/a.txt"].dense_score == 1.0
    assert by_path["peer/z.txt"].dense_score == 1.0
    assert by_path["peer/a.txt"].dense_rank == 1
    assert by_path["peer/z.txt"].dense_rank == 2
    assert [candidate.path for candidate in result.candidates] == [
        "peer/a.txt",
        "peer/x.txt",
        "peer/z.txt",
    ]


def test_retrieve_six_normalize_four_filters_deduplicates_refills_and_budgets(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "changed" / "diff.txt", "changed")
    contents = {
        "01-filtered.txt": "filtered",
        "02-original.txt": " duplicate body ",
        "03-duplicate.txt": "duplicate body",
        "04-long.txt": "a" * 9_000,
        "05-clipped.txt": "b" * 9_000,
        "06-unused.txt": "unused",
        "07-outside-cut.txt": "outside retrieval cut",
    }
    for name, content in contents.items():
        _write(tmp_path / "peer" / name, content)

    retriever = BaselineV1Retriever(
        FixtureDenseProvider(),
        evidence_filter=lambda candidate: candidate.relative_path != "01-filtered.txt",
    )
    result = retriever.retrieve(_request(tmp_path))

    assert result.status is RetrievalStatus.OK
    assert result.candidate_count == 7
    assert result.retrieved_count == 6
    assert result.filtered_count == 1
    assert result.duplicate_count == 1
    assert result.refill_count == 1
    assert result.evidence_characters == 16_000
    assert result.underfilled is True
    assert [item.path for item in result.evidence] == [
        "peer/02-original.txt",
        "peer/04-long.txt",
        "peer/05-clipped.txt",
    ]
    assert len(result.evidence[-1].content) == 6_986
    assert result.evidence[-1].render_truncated is True
    assert all(item.path != "peer/07-outside-cut.txt" for item in result.evidence)


def test_four_item_cap_and_repeated_runs_are_identical(tmp_path: Path) -> None:
    _write(tmp_path / "changed" / "diff.txt", "changed")
    for index in range(1, 7):
        _write(tmp_path / "peer" / f"{index}.txt", f"unique content {index}")
    retriever = BaselineV1Retriever(FixtureDenseProvider())
    request = _request(tmp_path)

    first = retriever.retrieve(request)
    second = retriever.retrieve(request)

    assert first == second
    assert first.status is RetrievalStatus.OK
    assert [item.path for item in first.evidence] == [
        "peer/1.txt",
        "peer/2.txt",
        "peer/3.txt",
        "peer/4.txt",
    ]
    assert first.underfilled is False
    assert first.fallback_engine is None


@pytest.mark.parametrize(
    ("query", "corpus_complete", "expected_code"),
    [
        ("", True, "explicit_retrieval_query_absent"),
        ("diff", False, "file_corpus_incomplete"),
    ],
)
def test_missing_preconditions_are_explicitly_insufficient(
    tmp_path: Path,
    query: str,
    corpus_complete: bool,
    expected_code: str,
) -> None:
    _write(tmp_path / "changed" / "diff.txt", "changed")
    _write(tmp_path / "peer" / "file.txt", "peer")

    result = BaselineV1Retriever(FixtureDenseProvider()).retrieve(
        _request(tmp_path, query=query, corpus_complete=corpus_complete)
    )

    assert result.status is RetrievalStatus.INSUFFICIENT
    assert result.error is not None
    assert result.error.code == expected_code
    assert result.fallback_engine is None


def test_empty_eligible_corpus_is_insufficient(tmp_path: Path) -> None:
    _write(tmp_path / "changed" / "diff.txt", "changed")
    _write(tmp_path / "peer" / "build" / "ignored.txt", "ignored")

    result = BaselineV1Retriever(FixtureDenseProvider()).retrieve(_request(tmp_path))

    assert result.status is RetrievalStatus.INSUFFICIENT
    assert result.error is not None
    assert result.error.code == "eligible_corpus_empty"


class BrokenDenseProvider(FixtureDenseProvider):
    def __init__(self, failure: str) -> None:
        super().__init__()
        self.failure = failure
        if failure == "dimension":
            self.dimension = 3

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        if self.failure == "timeout":
            raise TimeoutError("fixture provider timed out")
        vectors = list(super().embed(texts))
        if texts and texts[0].startswith("Repository file: "):
            if self.failure == "missing":
                return vectors[:-1]
            if self.failure == "nan":
                vectors[0] = (math.nan, 0.0)
            if self.failure == "infinite":
                vectors[0] = (math.inf, 0.0)
        return vectors


@pytest.mark.parametrize(
    "failure", ["dimension", "missing", "nan", "infinite", "timeout"]
)
def test_dense_contract_failures_are_explicit_errors(
    tmp_path: Path, failure: str
) -> None:
    _write(tmp_path / "changed" / "diff.txt", "changed")
    _write(tmp_path / "peer" / "file.txt", "peer")

    result = BaselineV1Retriever(BrokenDenseProvider(failure)).retrieve(
        _request(tmp_path)
    )

    assert result.status is RetrievalStatus.ERROR
    assert result.error is not None
    assert result.error.code == "dense_embedding_failed"
    assert result.fallback_engine is None
