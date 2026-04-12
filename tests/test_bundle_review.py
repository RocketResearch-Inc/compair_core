from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
from types import SimpleNamespace
import unittest


def _load_bundle_review_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    module_path = root / "compair_core" / "compair" / "bundle_review.py"
    spec = importlib.util.spec_from_file_location("test_compair_bundle_review", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


bundle_review = _load_bundle_review_module()


class BundleReviewTests(unittest.TestCase):
    def test_build_document_bundle_prefers_code_repo_docs_first(self) -> None:
        docs = [
            SimpleNamespace(document_id="doc_b", title="Notes", doc_type="note", content="hello"),
            SimpleNamespace(document_id="doc_a", title="Core", doc_type="code-repo", content="### File: README.md\ntext"),
        ]

        stats, bundle_text = bundle_review.build_document_bundle(docs)

        self.assertEqual(stats[0]["document_id"], "doc_a")
        self.assertIn("===== DOCUMENT Core [doc_a] =====", bundle_text)
        self.assertIn("<<<DOC TYPE code-repo TITLE Core ID doc_a>>>", bundle_text)

    def test_extract_json_object_handles_fenced_payload(self) -> None:
        payload = """```json
{"findings":[]}
```"""

        parsed = bundle_review.extract_json_object(payload)

        self.assertEqual(parsed, {"findings": []})

    def test_normalize_findings_payload_fills_missing_lists(self) -> None:
        payload = {"findings": [{"title": "x", "summary": "y"}]}

        normalized = bundle_review.normalize_findings_payload(payload)

        self.assertEqual(len(normalized["findings"]), 1)
        self.assertEqual(normalized["findings"][0]["target_files"], [])
        self.assertEqual(normalized["findings"][0]["intent"], "relevant_update")

    def test_render_now_review_markdown_includes_findings_and_bundle(self) -> None:
        markdown = bundle_review.render_now_review_markdown(
            group_name="demo",
            findings=[
                {
                    "title": "Mismatch",
                    "intent": "potential_conflict",
                    "severity": "high",
                    "certainty": "medium",
                    "summary": "A summary",
                    "why_it_matters": "It matters",
                    "evidence_target": "target",
                    "evidence_peer": "peer",
                    "follow_up": "check it",
                    "target_files": ["docs/readme.md"],
                    "peer_files": ["src/main.py"],
                }
            ],
            meta={"model": "gpt-5.4", "prompt_estimated_tokens": 1234, "duration_sec": 5.2},
            document_stats=[{"title": "Core", "doc_type": "code-repo", "document_id": "doc_a", "estimated_tokens": 300}],
        )

        self.assertIn("# Compair Now Review: demo", markdown)
        self.assertIn("### 1. Mismatch", markdown)
        self.assertIn("Target files:", markdown)
        self.assertIn("## Bundle", markdown)

    def test_estimate_usage_cost_uses_env_rates(self) -> None:
        old_input = os.environ.get("COMPAIR_NOW_REVIEW_INPUT_COST_PER_1M_USD")
        old_output = os.environ.get("COMPAIR_NOW_REVIEW_OUTPUT_COST_PER_1M_USD")
        try:
            os.environ["COMPAIR_NOW_REVIEW_INPUT_COST_PER_1M_USD"] = "2.5"
            os.environ["COMPAIR_NOW_REVIEW_OUTPUT_COST_PER_1M_USD"] = "10"
            estimate = bundle_review.estimate_usage_cost(
                model="gpt-5.4",
                input_tokens=2000,
                output_tokens=300,
                prompt_estimated_tokens=1234,
            )
        finally:
            if old_input is None:
                os.environ.pop("COMPAIR_NOW_REVIEW_INPUT_COST_PER_1M_USD", None)
            else:
                os.environ["COMPAIR_NOW_REVIEW_INPUT_COST_PER_1M_USD"] = old_input
            if old_output is None:
                os.environ.pop("COMPAIR_NOW_REVIEW_OUTPUT_COST_PER_1M_USD", None)
            else:
                os.environ["COMPAIR_NOW_REVIEW_OUTPUT_COST_PER_1M_USD"] = old_output

        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertEqual(estimate["pricing_source"], "env")
        self.assertAlmostEqual(estimate["input_cost_usd"], 0.005)
        self.assertAlmostEqual(estimate["output_cost_usd"], 0.003)
        self.assertAlmostEqual(estimate["total_cost_usd"], 0.008)


if __name__ == "__main__":
    unittest.main()
