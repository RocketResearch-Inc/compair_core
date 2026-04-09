from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import pathlib
import sys
import types
import unittest


def _load_service_module():
    root = pathlib.Path(__file__).resolve().parents[1]
    root_package_name = "test_compair_notification_service"
    compair_package_name = f"{root_package_name}.compair"
    notifications_package_name = f"{compair_package_name}.notifications"
    service_path = root / "compair_core" / "compair" / "notifications" / "service.py"

    root_package = types.ModuleType(root_package_name)
    root_package.__path__ = [str(service_path.parents[2])]
    sys.modules[root_package_name] = root_package

    compair_package = types.ModuleType(compair_package_name)
    compair_package.__path__ = [str(service_path.parents[1])]
    sys.modules[compair_package_name] = compair_package

    notifications_package = types.ModuleType(notifications_package_name)
    notifications_package.__path__ = [str(service_path.parent)]
    sys.modules[notifications_package_name] = notifications_package

    sqlalchemy = types.ModuleType("sqlalchemy")
    sys.modules["sqlalchemy"] = sqlalchemy

    sqlalchemy_orm = types.ModuleType("sqlalchemy.orm")
    sqlalchemy_orm.Session = object
    sys.modules["sqlalchemy.orm"] = sqlalchemy_orm

    local_summary = types.ModuleType(f"{compair_package_name}.local_summary")
    local_summary.assess_relation = lambda target, peer: types.SimpleNamespace(kind="route/path mismatch")
    local_summary.best_grounded_excerpt = lambda text, signal_texts, preferred_text="": preferred_text or text
    local_summary.excerpt_tokens = lambda *values: set()
    sys.modules[local_summary.__name__] = local_summary

    models = types.ModuleType(f"{compair_package_name}.models")
    for name in ("Chunk", "Document", "NotificationEvent"):
        setattr(models, name, type(name, (), {}))
    sys.modules[models.__name__] = models

    delivery_logic = types.ModuleType(f"{notifications_package_name}.delivery_logic")
    for name in ("CandidateContext", "DeliveryDecision", "DeliveryPolicy", "DigestItem"):
        setattr(delivery_logic, name, type(name, (), {}))
    delivery_logic.compute_dedupe_key = lambda *args, **kwargs: "dedupe"
    delivery_logic.decide_and_queue = lambda *args, **kwargs: None
    delivery_logic.maybe_escalate_conflict = lambda *args, **kwargs: None
    sys.modules[delivery_logic.__name__] = delivery_logic

    scorer = types.ModuleType(f"{notifications_package_name}.llm_notification_scorer")
    scorer.NotificationScorer = type("NotificationScorer", (), {})
    scorer.NotificationScorerConfig = type("NotificationScorerConfig", (), {})
    sys.modules[scorer.__name__] = scorer

    parse = types.ModuleType(f"{notifications_package_name}.parse_llm_structured_output")

    @dataclass(frozen=True)
    class ParsedLLMNotificationAssessment:
        intent: str
        relevance: str
        novelty: str
        severity: str
        certainty: str
        delivery: str
        rationale: list[str]
        evidence_target: str
        evidence_peer: str
        parse_mode: str
        raw_extracted: object | None
        errors: list[str]

    parse.ParsedLLMNotificationAssessment = ParsedLLMNotificationAssessment
    sys.modules[parse.__name__] = parse

    spec = importlib.util.spec_from_file_location(f"{notifications_package_name}.service", service_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


service = _load_service_module()


class NotificationServiceTests(unittest.TestCase):
    def test_consistency_guard_downgrades_conflict_when_rationale_denies_explicit_contradiction(self) -> None:
        candidate = service.NotificationCandidate(
            user_id="u1",
            group_id="g1",
            target_doc_id="d1",
            target_chunk_id="c1",
            target_text="target excerpt",
            peer_candidates=(),
            generated_feedback={"summary": "There is a concrete mismatch/drift in the same surface."},
        )
        assessment = service.ParsedLLMNotificationAssessment(
            intent="potential_conflict",
            relevance="HIGH",
            novelty="HIGH",
            severity="HIGH",
            certainty="HIGH",
            delivery="push",
            rationale=[
                "Both excerpts concern the same configuration surface.",
                "There is no explicit contradiction in the provided excerpts.",
            ],
            evidence_target="target excerpt",
            evidence_peer="peer excerpt",
            parse_mode="json_schema_rubric",
            raw_extracted=None,
            errors=[],
        )

        normalized = service._enforce_assessment_consistency(candidate, assessment)

        self.assertEqual(normalized.intent, "relevant_update")
        self.assertEqual(normalized.severity, "MEDIUM")
        self.assertEqual(normalized.delivery, "digest")

    def test_consistency_guard_downgrades_truncated_conflict_without_strong_summary_signal(self) -> None:
        candidate = service.NotificationCandidate(
            user_id="u1",
            group_id="g1",
            target_doc_id="d1",
            target_chunk_id="c1",
            target_text="target excerpt",
            peer_candidates=(),
            generated_feedback={"summary": "The peer discusses a related topic."},
        )
        assessment = service.ParsedLLMNotificationAssessment(
            intent="potential_conflict",
            relevance="MEDIUM",
            novelty="MEDIUM",
            severity="MEDIUM",
            certainty="HIGH",
            delivery="digest",
            rationale=[
                "The changed excerpt appears truncated.",
                "The evidence is incomplete.",
            ],
            evidence_target="target excerpt",
            evidence_peer="peer excerpt",
            parse_mode="json_schema_rubric",
            raw_extracted=None,
            errors=[],
        )

        normalized = service._enforce_assessment_consistency(candidate, assessment)

        self.assertEqual(normalized.intent, "relevant_update")
        self.assertEqual(normalized.severity, "LOW")
        self.assertEqual(normalized.relevance, "LOW")

    def test_calibration_does_not_promote_when_rationale_explicitly_denies_disagreement(self) -> None:
        candidate = service.NotificationCandidate(
            user_id="u1",
            group_id="g1",
            target_doc_id="d1",
            target_chunk_id="c1",
            target_text="target excerpt",
            peer_candidates=(),
            generated_feedback={"summary": "There is a concrete mismatch/drift between the target and peer."},
        )
        assessment = service.ParsedLLMNotificationAssessment(
            intent="quiet_validation",
            relevance="MEDIUM",
            novelty="LOW",
            severity="LOW",
            certainty="HIGH",
            delivery="digest",
            rationale=[
                "Both excerpts concern the same product surface.",
                "There is no explicit disagreement; the peer is consistent with the target.",
            ],
            evidence_target="target excerpt",
            evidence_peer="peer excerpt",
            parse_mode="json_schema_rubric",
            raw_extracted=None,
            errors=[],
        )

        calibrated = service._calibrate_assessment_from_feedback(candidate, assessment)

        self.assertEqual(calibrated.intent, "quiet_validation")
        self.assertEqual(calibrated.rationale, assessment.rationale)

    def test_calibration_can_promote_hidden_overlap_when_rationale_is_neutral(self) -> None:
        candidate = service.NotificationCandidate(
            user_id="u1",
            group_id="g1",
            target_doc_id="d1",
            target_chunk_id="c1",
            target_text="target excerpt",
            peer_candidates=(),
            generated_feedback={"summary": "There is a concrete route mismatch in the same API surface."},
        )
        assessment = service.ParsedLLMNotificationAssessment(
            intent="hidden_overlap",
            relevance="LOW",
            novelty="LOW",
            severity="LOW",
            certainty="HIGH",
            delivery="digest",
            rationale=["Both excerpts refer to the same API surface."],
            evidence_target="target excerpt",
            evidence_peer="peer excerpt",
            parse_mode="json_schema_rubric",
            raw_extracted=None,
            errors=[],
        )

        calibrated = service._calibrate_assessment_from_feedback(candidate, assessment)

        self.assertEqual(calibrated.intent, "potential_conflict")
        self.assertIn("Generated feedback describes a mismatch/drift", calibrated.rationale[-1])
