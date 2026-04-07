from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace
import unittest


def _load_feedback_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    package_name = "test_compair_feedback_module"
    feedback_path = root / "compair_core" / "compair" / "feedback.py"
    local_summary_path = root / "compair_core" / "compair" / "local_summary.py"

    package = types.ModuleType(package_name)
    package.__path__ = [str(feedback_path.parent)]
    sys.modules[package_name] = package

    requests_module = types.ModuleType("requests")
    requests_module.post = lambda *args, **kwargs: None
    sys.modules["requests"] = requests_module

    logger_module = types.ModuleType(f"{package_name}.logger")
    logger_module.log_event = lambda *args, **kwargs: None
    sys.modules[logger_module.__name__] = logger_module

    models_module = types.ModuleType(f"{package_name}.models")
    models_module.Document = type("Document", (), {})
    models_module.User = type("User", (), {})
    sys.modules[models_module.__name__] = models_module

    local_summary_spec = importlib.util.spec_from_file_location(
        f"{package_name}.local_summary",
        local_summary_path,
    )
    local_summary_module = importlib.util.module_from_spec(local_summary_spec)
    sys.modules[local_summary_spec.name] = local_summary_module
    assert local_summary_spec.loader is not None
    local_summary_spec.loader.exec_module(local_summary_module)

    spec = importlib.util.spec_from_file_location(f"{package_name}.feedback", feedback_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


feedback = _load_feedback_module()


class FeedbackPromptTests(unittest.TestCase):
    def test_format_changed_chunk_prompt_prefers_focus_excerpt(self) -> None:
        full_chunk = (
            "### File: docs/api_mapping.md\n"
            "| `docs list` | `GET /load_documents` |\n"
            "| `activity` | `GET /activity_feed` |\n"
            "| `notifications` | `GET /notification_events` |\n"
        )
        focus_text = "| `activity` | `GET /activity_feed` |"

        prompt = feedback._format_changed_chunk_prompt(full_chunk, focus_text)

        self.assertIn("Primary changed excerpt:\n| `activity` | `GET /activity_feed` |", prompt)
        self.assertIn("Surrounding chunk context (secondary):\n### File: docs/api_mapping.md", prompt)
        self.assertLess(prompt.index(focus_text), prompt.index("### File: docs/api_mapping.md"))

    def test_format_changed_chunk_prompt_omits_secondary_when_focus_matches_full_chunk(self) -> None:
        full_chunk = "Google OAuth is available on Core and should appear in /capabilities when configured."

        prompt = feedback._format_changed_chunk_prompt(full_chunk, full_chunk)

        self.assertEqual(
            prompt,
            "Primary changed excerpt:\nGoogle OAuth is available on Core and should appear in /capabilities when configured.",
        )
        self.assertNotIn("Surrounding chunk context (secondary):", prompt)

    def test_openai_prompt_can_include_change_context_block(self) -> None:
        changed_chunk_prompt = feedback._format_changed_chunk_prompt(
            "### File: docs/api_mapping.md\n| `activity` | `GET /activity_feed` |",
            "| `activity` | `GET /activity_feed` |",
        )
        change_context = (
            "### File: docs/api_mapping.md\n"
            "- | `activity` | `GET /get_activity_feed` |\n"
            "+ | `activity` | `GET /activity_feed` |"
        )

        self.assertIn("Primary changed excerpt:\n| `activity` | `GET /activity_feed` |", changed_chunk_prompt)
        self.assertIn("- | `activity` | `GET /get_activity_feed` |", change_context)
        self.assertIn("+ | `activity` | `GET /activity_feed` |", change_context)

    def test_split_feedback_items_supports_multiple_findings(self) -> None:
        raw = (
            "The docs claim /activity_feed.\n"
            "<<<FINDING>>>\n"
            "The desktop client still calls /get_activity_feed.\n"
        )

        items = feedback.split_feedback_items(raw)

        self.assertEqual(
            items,
            [
                "The docs claim /activity_feed.",
                "The desktop client still calls /get_activity_feed.",
            ],
        )

    def test_split_feedback_items_dedupes_repeated_findings(self) -> None:
        raw = (
            "The workflow skips the artifact audit.\n"
            "<<<FINDING>>>\n"
            "The workflow skips the artifact audit.\n"
        )

        items = feedback.split_feedback_items(raw)

        self.assertEqual(items, ["The workflow skips the artifact audit."])

    def test_reasoning_retry_only_triggers_on_unsupported_reasoning_errors(self) -> None:
        unsupported = Exception("Unknown parameter: reasoning.effort")
        transient = Exception("Read timeout while calling /responses")

        self.assertTrue(feedback._should_retry_without_reasoning(unsupported))
        self.assertFalse(feedback._should_retry_without_reasoning(transient))

    def test_render_local_reference_match_prefers_concise_route_message(self) -> None:
        relation = SimpleNamespace(
            kind="route/path mismatch",
            confidence=6,
            target_artifact="/activity_feed",
            peer_artifact="/get_activity_feed",
        )
        match = SimpleNamespace(
            relation=relation,
            reference_label="RocketResearch-Inc/compair_core",
            target_excerpt='| `activity` | `GET /activity_feed` |',
            peer_excerpt='| `activity` | `GET /get_activity_feed` |',
        )

        summary = feedback._render_local_reference_match(match)

        self.assertEqual(
            summary,
            'Possible route/path drift: the changed excerpt uses "/activity_feed", while RocketResearch-Inc/compair_core uses "/get_activity_feed".',
        )

    def test_render_local_reference_match_genericizes_weak_match(self) -> None:
        relation = SimpleNamespace(kind="generic divergence", confidence=1, target_artifact="", peer_artifact="")
        match = SimpleNamespace(
            relation=relation,
            reference_label="RocketResearch-Inc/compair_core",
            target_excerpt="WinGet is live today.",
            peer_excerpt="Homebrew remains pending.",
        )

        summary = feedback._render_local_reference_match(match)

        self.assertIn("Possible cross-repo drift detected", summary)
        self.assertIn("bundled local review path", summary)

    def test_render_local_reference_match_avoids_specific_path_when_excerpt_has_many_paths(self) -> None:
        relation = SimpleNamespace(
            kind="presence/absence",
            confidence=3,
            target_artifact="/notification_events",
            peer_artifact="",
        )
        match = SimpleNamespace(
            relation=relation,
            reference_label="RocketResearch-Inc/compair_core",
            target_excerpt=(
                "| `docs list` | `GET /load_documents` |\n"
                "| `activity` | `GET /activity_feed` |\n"
                "| `notifications` | `GET /notification_events` |"
            ),
            peer_excerpt='| `activity` | `GET /get_activity_feed` |',
        )

        summary = feedback._render_local_reference_match(match)

        self.assertIn("Possible cross-repo drift detected", summary)
        self.assertNotIn('/notification_events', summary)


if __name__ == "__main__":
    unittest.main()
