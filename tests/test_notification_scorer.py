from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
import types
import unittest
from unittest import mock


def _load_notification_scorer_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    package_name = "test_compair_notification_scorer"
    scorer_path = root / "compair_core" / "compair" / "notifications" / "llm_notification_scorer.py"
    parse_path = root / "compair_core" / "compair" / "notifications" / "parse_llm_structured_output.py"

    package = types.ModuleType(package_name)
    package.__path__ = [str(scorer_path.parent)]
    sys.modules[package_name] = package

    parse_spec = importlib.util.spec_from_file_location(
        f"{package_name}.parse_llm_structured_output",
        parse_path,
    )
    parse_module = importlib.util.module_from_spec(parse_spec)
    sys.modules[parse_spec.name] = parse_module
    assert parse_spec.loader is not None
    parse_spec.loader.exec_module(parse_module)

    spec = importlib.util.spec_from_file_location(f"{package_name}.llm_notification_scorer", scorer_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


scorer_module = _load_notification_scorer_module()


class _FakeResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.output = []


class _FakeResponses:
    def __init__(self, planned: list[object]) -> None:
        self._planned = list(planned)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self._planned:
            raise AssertionError("unexpected extra responses.create call")
        current = self._planned.pop(0)
        if isinstance(current, Exception):
            raise current
        return current


class _FakeClient:
    def __init__(self, planned: list[object]) -> None:
        self.responses = _FakeResponses(planned)


def _payload() -> dict:
    return {
        "target": {
            "chunk_excerpt": "| `activity` | `GET /activity_feed` |",
            "chunk_text": "### File: docs/api_mapping.md\n| `activity` | `GET /activity_feed` |",
        },
        "candidates": [
            {
                "peer_excerpt": "| `activity` | `GET /get_activity_feed` |",
                "peer_chunk_text": "### File: desktop/api_mapping.md\n| `activity` | `GET /get_activity_feed` |",
            }
        ],
        "generated_feedback": {
            "summary": "There is a concrete route/path drift between /activity_feed and /get_activity_feed.",
        },
    }


def _assessment(parse_mode: str) -> scorer_module.ParsedLLMNotificationAssessment:
    return scorer_module.ParsedLLMNotificationAssessment(
        intent="potential_conflict",
        relevance="HIGH",
        novelty="HIGH",
        severity="HIGH",
        certainty="HIGH",
        delivery="push",
        rationale=["Target and peer disagree on the route path."],
        evidence_target="| `activity` | `GET /activity_feed` |",
        evidence_peer="| `activity` | `GET /get_activity_feed` |",
        parse_mode=parse_mode,
        raw_extracted=None,
        errors=[],
    )


class NotificationScorerTests(unittest.TestCase):
    def test_score_prefers_json_schema_output(self) -> None:
        client = _FakeClient(
            [
                _FakeResponse(
                    json.dumps(
                        {
                            "same_surface_area": "yes",
                            "direct_contradiction": "yes",
                            "docs_vs_impl_drift": "yes",
                            "user_or_runtime_impact": "yes",
                            "policy_or_release_risk": "no",
                            "duplication_or_overlap": "no",
                            "alignment_or_confirmation": "no",
                            "novel_for_user": "yes",
                            "rationale": ["Target and peer disagree on the route path."],
                            "evidence_target": "| `activity` | `GET /activity_feed` |",
                            "evidence_peer": "| `activity` | `GET /get_activity_feed` |",
                        }
                    )
                )
            ]
        )
        scorer = scorer_module.NotificationScorer(
            config=scorer_module.NotificationScorerConfig(max_retries=1),
            client=client,
        )

        result = scorer.score(_payload())

        self.assertEqual(result.parse_mode, "json_schema_rubric")
        self.assertEqual(result.intent, "potential_conflict")
        self.assertEqual(result.severity, "HIGH")
        self.assertEqual(result.delivery, "push")
        self.assertEqual(len(client.responses.requests), 1)
        request = client.responses.requests[0]
        self.assertIn("text", request)
        self.assertEqual(request["text"]["format"]["type"], "json_schema")
        self.assertTrue(request["text"]["format"]["strict"])
        self.assertEqual(request["text"]["format"]["name"], "notification_score")

    def test_structured_request_uses_configured_timeout(self) -> None:
        client = _FakeClient(
            [
                _FakeResponse(
                    json.dumps(
                        {
                            "same_surface_area": "yes",
                            "direct_contradiction": "yes",
                            "docs_vs_impl_drift": "yes",
                            "user_or_runtime_impact": "yes",
                            "policy_or_release_risk": "no",
                            "duplication_or_overlap": "no",
                            "alignment_or_confirmation": "no",
                            "novel_for_user": "yes",
                            "rationale": ["Target and peer disagree on the route path."],
                            "evidence_target": "| `activity` | `GET /activity_feed` |",
                            "evidence_peer": "| `activity` | `GET /get_activity_feed` |",
                        }
                    )
                )
            ]
        )
        scorer = scorer_module.NotificationScorer(
            config=scorer_module.NotificationScorerConfig(max_retries=1, timeout_s=75.0),
            client=client,
        )

        scorer.score(_payload())

        self.assertEqual(client.responses.requests[0]["timeout"], 75.0)

    def test_scorer_model_inherits_primary_openai_model_when_notif_model_unset(self) -> None:
        with mock.patch.dict(os.environ, {"COMPAIR_OPENAI_MODEL": "gpt-5-nano"}, clear=False):
            scorer = scorer_module.NotificationScorer(
                config=scorer_module.NotificationScorerConfig(),
                client=object(),
            )

        self.assertEqual(scorer.config.model, "gpt-5-nano")

    def test_rubric_mapping_yields_low_overlap_digest(self) -> None:
        parsed, errors = scorer_module._rubric_assessment(
            {
                "same_surface_area": "yes",
                "direct_contradiction": "no",
                "docs_vs_impl_drift": "no",
                "user_or_runtime_impact": "no",
                "policy_or_release_risk": "no",
                "duplication_or_overlap": "yes",
                "alignment_or_confirmation": "no",
                "novel_for_user": "no",
                "rationale": ["Both snippets cover the same endpoint mapping without disagreement."],
                "evidence_target": "| `notifications` | `GET /notification_events` |",
                "evidence_peer": "| `notifications` | `GET /notification_events` |",
            }
        )

        self.assertEqual(errors, [])
        assert parsed is not None
        self.assertEqual(parsed.intent, "hidden_overlap")
        self.assertEqual(parsed.severity, "LOW")
        self.assertEqual(parsed.delivery, "digest")

    def test_rubric_mapping_caps_relationless_novelty(self) -> None:
        parsed, errors = scorer_module._rubric_assessment(
            {
                "same_surface_area": "no",
                "direct_contradiction": "no",
                "docs_vs_impl_drift": "no",
                "user_or_runtime_impact": "no",
                "policy_or_release_risk": "no",
                "duplication_or_overlap": "no",
                "alignment_or_confirmation": "no",
                "novel_for_user": "yes",
                "rationale": ["The peer mentions a different topic."],
                "evidence_target": "target excerpt",
                "evidence_peer": "peer excerpt",
            }
        )

        self.assertEqual(errors, [])
        assert parsed is not None
        self.assertEqual(parsed.intent, "relevant_update")
        self.assertEqual(parsed.relevance, "LOW")
        self.assertEqual(parsed.severity, "LOW")
        self.assertEqual(parsed.novelty, "LOW")

    def test_score_stops_after_repair_without_kv_fallback(self) -> None:
        scorer = scorer_module.NotificationScorer(
            config=scorer_module.NotificationScorerConfig(max_retries=1),
            client=object(),
        )
        calls: list[str] = []

        def fake_structured(system_prompt: str, user_prompt: str):
            calls.append("structured")
            return scorer_module.conservative_default("failed_default", errors=["schema path unavailable"])

        legacy_results = iter(
            [
                scorer_module.conservative_default("failed_default", errors=["legacy parse failed"]),
                _assessment("json_repaired"),
            ]
        )

        def fake_score_once(system_prompt: str, user_prompt: str):
            calls.append("legacy" if len(calls) == 1 else "repair")
            return next(legacy_results)

        scorer._score_once_structured = fake_structured  # type: ignore[method-assign]
        scorer._score_once = fake_score_once  # type: ignore[method-assign]

        result = scorer.score(_payload())

        self.assertEqual(result.parse_mode, "json_repaired")
        self.assertEqual(calls, ["structured", "legacy", "repair"])

    def test_score_short_circuits_to_heuristic_on_transport_failure(self) -> None:
        scorer = scorer_module.NotificationScorer(
            config=scorer_module.NotificationScorerConfig(max_retries=1),
            client=object(),
        )
        calls: list[str] = []

        def fake_structured(system_prompt: str, user_prompt: str):
            calls.append("structured")
            return scorer_module.conservative_default(
                "failed_default",
                errors=["structured_transport_error: APITimeoutError('Request timed out.')"],
            )

        def fake_score_once(system_prompt: str, user_prompt: str):
            calls.append("legacy")
            return _assessment("json")

        scorer._score_once_structured = fake_structured  # type: ignore[method-assign]
        scorer._score_once = fake_score_once  # type: ignore[method-assign]

        result = scorer.score(_payload())

        self.assertEqual(result.parse_mode, "heuristic")
        self.assertEqual(result.intent, "potential_conflict")
        self.assertEqual(calls, ["structured"])


if __name__ == "__main__":
    unittest.main()
