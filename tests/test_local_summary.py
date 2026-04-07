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

    def test_numeric_fraction_is_not_treated_as_path_artifact(self) -> None:
        profile = local_summary.extract_artifacts("1/2 OAuth Core capabilities")
        self.assertEqual(profile.paths, ())

    def test_numeric_fraction_does_not_drive_presence_absence_summary(self) -> None:
        relation = local_summary.assess_relation(
            "1/2 OAuth Core capabilities",
            "OAuth Core capabilities",
        )
        self.assertNotEqual(relation.kind, "presence/absence")
        summary = local_summary.summarize_comparison(
            "1/2 OAuth Core capabilities",
            "OAuth Core capabilities",
            "ref",
            relation,
        )
        self.assertNotIn('introduces "1/2"', summary or "")

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

    def test_changed_lines_are_preferred_for_target_excerpt(self) -> None:
        target = (
            "@@ -1,5 +1,5 @@\n"
            ' title = "Compair"\n'
            '-payload.reviews.map((review) => render(review.severity, review.category));\n'
            '+payload.items.map((item) => render(item.priority, item.type));\n'
            ' footer = "done"\n'
        )
        reference = local_summary.ReferenceText(
            label="demo.local:compair/demo-api",
            text="payload.reviews.map((review) => render(review.severity, review.category));",
        )
        match = local_summary.best_reference_match(target, [reference])
        self.assertIsNotNone(match)
        assert match is not None
        self.assertIn("payload.items", match.target_excerpt)

    def test_focus_text_guides_full_chunk_summary(self) -> None:
        full_chunk = (
            "### File: internal/api/capabilities.go\n"
            "package api\n\n"
            "func capabilitiesCachePath() string {\n"
            '  return filepath.Join(dir, ".compair", "cache", "capabilities.json")\n'
            "}\n\n"
            "type authCaps struct {\n"
            '  SingleUser bool `json:"singleUser"`\n'
            '  ActivityFeed bool `json:"activityFeed"`\n'
            "}\n"
        )
        focus_text = (
            'type authCaps struct {\n'
            '  SingleUser bool `json:"singleUser"`\n'
            '  ActivityFeed bool `json:"activityFeed"`\n'
            "}\n"
        )
        reference = local_summary.ReferenceText(
            label="demo.local:compair/demo-api",
            text=(
                'type authCaps struct {\n'
                '  SingleUser bool `json:"single_user"`\n'
                '  ActivityFeed bool `json:"activity_feed"`\n'
                "}\n"
            ),
        )
        match = local_summary.best_reference_match(full_chunk, [reference], focus_text=focus_text)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertIn("singleUser", match.target_excerpt)
        summary = local_summary.summarize_reference_feedback(full_chunk, [reference], focus_text=focus_text)
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertIn("singleUser", summary)

    def test_snapshot_header_is_stripped_from_summary_excerpt(self) -> None:
        full_chunk = (
            "### File: docs/core_quickstart.md (part 1/2, lang markdown)\n"
            "Google OAuth is available on Core and should appear in /capabilities when client credentials are configured.\n"
        )
        reference = local_summary.ReferenceText(
            label="ref",
            text=(
                "### File: compair_core/server/routers/capabilities.py\n"
                'google_oauth_configured = edition == "cloud"\n'
            ),
        )
        summary = local_summary.summarize_reference_feedback(full_chunk, [reference])
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertNotIn("### File:", summary)
        self.assertNotIn("part 1/2", summary)
        self.assertIn("Google OAuth is available on Core", summary)

    def test_structured_config_fragments_remain_eligible(self) -> None:
        summary = local_summary.summarize_reference_feedback(
            'feature_flags = ["sync", "reviews"]\napi_base = "https://api.example.com/v2"',
            [
                local_summary.ReferenceText(
                    label="ref",
                    text='feature_flags = ["sync"]\napi_base = "https://api.example.com/v1"',
                )
            ],
        )
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertIn("changed content", summary)

    def test_generic_prose_peer_is_rejected_when_overlap_is_weak(self) -> None:
        summary = local_summary.summarize_reference_feedback(
            'payload.items.map((item) => render(item.priority, item.type));',
            [
                local_summary.ReferenceText(
                    label="ref",
                    text="This page includes general notes for contributors and general project guidance.",
                )
            ],
        )
        self.assertIsNone(summary)

    def test_weak_evidence_returns_none(self) -> None:
        summary = local_summary.summarize_reference_feedback(
            "Updated wording only.",
            [local_summary.ReferenceText(label="ref", text="Misc notes.")],
        )
        self.assertIsNone(summary)


if __name__ == "__main__":
    unittest.main()
