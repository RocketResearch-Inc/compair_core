from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


def _load_local_summary_module():
    module_path = pathlib.Path(__file__).resolve().parents[1] / "compair_core" / "compair" / "local_summary.py"
    spec = importlib.util.spec_from_file_location("test_local_summary_module", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


local_summary = _load_local_summary_module()


class LocalSummaryTests(unittest.TestCase):
    def test_rename_relation(self) -> None:
        relation = local_summary.assess_relation(
            'return payload.items.map((item) => render(item.priority, item.type));',
            'return payload.reviews.map((review) => render(review.severity, review.category));',
        )
        self.assertEqual(relation.kind, "rename")
        self.assertIn("payload.items", relation.target_artifact)
        self.assertIn("payload.reviews", relation.peer_artifact)

    def test_value_mismatch_relation(self) -> None:
        relation = local_summary.assess_relation("publish = true", "publish = false")
        self.assertEqual(relation.kind, "value mismatch")
        summary = local_summary.summarize_comparison("publish = true", "publish = false", "ref", relation)
        self.assertEqual(summary, 'The changed content says "publish = true", while ref says "publish = false".')

    def test_presence_absence_relation(self) -> None:
        relation = local_summary.assess_relation(
            "enable_cache = true\nexperimental_mode = true",
            "enable_cache = true",
        )
        self.assertEqual(relation.kind, "presence/absence")
        summary = local_summary.summarize_comparison(
            "enable_cache = true experimental_mode = true",
            "enable_cache = true",
            "ref",
            relation,
        )
        self.assertIn('introduces "experimental_mode"', summary)

    def test_route_path_mismatch_relation(self) -> None:
        relation = local_summary.assess_relation("GET /v2/reviews", "GET /reviews")
        self.assertEqual(relation.kind, "route/path mismatch")

    def test_docs_vs_impl_relation(self) -> None:
        relation = local_summary.assess_relation(
            "The endpoint is /reviews and returns severity with rationale.",
            "paths:\n  /items:\n    get:\n      summary: Returns priority and rationale",
        )
        self.assertEqual(relation.kind, "docs-vs-impl mismatch")

    def test_generic_divergence_fallback(self) -> None:
        relation = local_summary.assess_relation(
            "The changed note mentions summary output for review items.",
            "Reference note mentions review output and item summaries.",
        )
        self.assertEqual(relation.kind, "generic divergence")

    def test_snapshot_header_is_not_selected_as_peer_excerpt(self) -> None:
        reference = local_summary.ReferenceText(
            label="demo.local:compair/demo-api",
            text=(
                "# Compair baseline snapshot\n"
                "Generated: 2026-03-28T00:00:00Z\n"
                "### File: api/openapi.yaml\n"
                "paths:\n"
                "  /reviews:\n"
                "    get:\n"
                "      summary: Returns reviews\n"
                "### File: README.md\n"
                "This repo documents the API.\n"
            ),
        )
        match = local_summary.best_reference_match("GET /reviews", [reference])
        self.assertIsNotNone(match)
        assert match is not None
        self.assertNotIn("baseline snapshot", match.peer_excerpt.lower())
        self.assertIn("/reviews", match.peer_excerpt)

    def test_weak_evidence_returns_none(self) -> None:
        summary = local_summary.summarize_reference_feedback(
            "Updated wording only.",
            [local_summary.ReferenceText(label="ref", text="Misc notes.")],
        )
        self.assertIsNone(summary)


if __name__ == "__main__":
    unittest.main()
