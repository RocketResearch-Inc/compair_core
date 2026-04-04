from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
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


if __name__ == "__main__":
    unittest.main()
