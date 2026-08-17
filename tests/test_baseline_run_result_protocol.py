from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLI_ROOT = ROOT.parent / "compair-cli"
V2_HASH = "b278abe007779f05e92509db068f555701c03cba5cf236151e8df231a9b44091"
OBSOLETE_V2_HASH = (
    "c9486b3deb1a494781513109df17d8e8df1281fbc9687960ace711485b50d174"
)
ARTIFACTS = {
    "baseline-run-result.v1.md": (
        "f1a9456cf1c9ed20f706a85e47e9ae03fe4f9b776cab0c12b680b05578fae5b9"
    ),
    "baseline-run-result.v1.schema.json": (
        "d1681bb22b63e0e6c56499bf7a24131e4bf8e5f1babacc91f2c165fb094c3b96"
    ),
    "fixtures/baseline-run-result.v1.valid.json": (
        "3148f4fae5ac197288c3d8b868eb64cabe67ec6dc0d4756e8766272ae385bf29"
    ),
    "fixtures/baseline-run-result.v1.invalid.json": (
        "785a3eb6b00d9cfe7564ea29e0d00b444b82af1a3a23badbb69ec8852d53deaa"
    ),
}
FORBIDDEN = {
    "retrieval_query",
    "raw_diff",
    "idempotency_key",
    "lease_token",
    "ciphertext",
    "nonce",
    "key_id",
    "parent_processing_secret",
    "provider_input",
    "provider_output",
    "evidence_content",
    "feedback",
    "feedback_text",
    "credentials",
    "endpoint_url",
    "child_runs",
}


def _artifact(root: Path, relative: str) -> bytes:
    return (root / "protocol" / relative).read_bytes()


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for nested in value.values():
            result.update(_keys(nested))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for nested in value:
            result.update(_keys(nested))
        return result
    return set()


def _semantic_errors(result: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(result) != set(schema["required"]):
        errors.append("fields")
    if result.get("schema_version") != "baseline-run-result.v1":
        errors.append("schema_version")
    if (
        result.get("protocol_version") != "baseline-control-plane.v2"
        or result.get("protocol_sha256") != V2_HASH
    ):
        errors.append("protocol")
    reason = result.get("reason_code")
    if reason is not None and re.fullmatch(r"[a-z0-9_]{1,64}", reason) is None:
        errors.append("reason")
    evidence = result.get("evidence_count")
    references = result.get("reference_count")
    feedback = result.get("feedback_count")
    outbox = result.get("notification_outbox_count")
    if not all(isinstance(value, int) for value in (evidence, references, feedback)):
        errors.append("counts")
    elif not (0 <= evidence <= 4 and 0 <= references <= 4 and 0 <= feedback <= 4):
        errors.append("counts")
    state = result.get("state")
    if state == "feedback_persisted":
        if (
            result.get("exit_classification") != "success"
            or result.get("persisted_retrieval_run_id") is None
            or evidence is None
            or evidence < 1
            or evidence != references
            or result.get("generation_invoked") is not True
            or reason is not None
            or (feedback == 0 and outbox != 0)
        ):
            errors.append("feedback_persisted")
    elif state == "references_persisted":
        if (
            result.get("exit_classification") != "pending"
            or result.get("persisted_retrieval_run_id") is None
            or evidence is None
            or evidence < 1
            or evidence != references
            or feedback != 0
            or outbox != 0
            or result.get("generation_invoked") is not False
            or reason is not None
        ):
            errors.append("references_persisted")
    elif state in {"queued", "running"}:
        if any((evidence, references, feedback, outbox)) or result.get(
            "generation_invoked"
        ):
            errors.append("pending")
    elif state == "insufficient":
        if (
            result.get("exit_classification") != "insufficient"
            or result.get("persisted_retrieval_run_id") is not None
            or any((evidence, references, feedback, outbox))
            or result.get("generation_invoked")
            or reason != "retrieval_insufficient"
        ):
            errors.append("insufficient")
    elif state in {"retryable_failed", "retryable_incomplete"}:
        if reason is None:
            errors.append("retryable")
    elif state in {"terminal_failed", "blocked", "cancelled", "failed"}:
        if reason is None:
            errors.append("terminal")
    else:
        errors.append("state")
    if _keys(result) & FORBIDDEN:
        errors.append("forbidden")
    return errors


def test_baseline_run_result_artifacts_are_frozen_and_byte_identical() -> None:
    for relative, expected in ARTIFACTS.items():
        core = _artifact(ROOT, relative)
        assert core == _artifact(CLI_ROOT, relative)
        assert hashlib.sha256(core).hexdigest() == expected


def test_baseline_run_results_freeze_job_wide_effects_and_zero_findings() -> None:
    schema = json.loads(_artifact(ROOT, "baseline-run-result.v1.schema.json"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    results = json.loads(
        _artifact(ROOT, "fixtures/baseline-run-result.v1.valid.json")
    )["results"]
    assert len(results) == 4
    for result in results:
        assert _semantic_errors(result, schema) == []
        assert result["evidence_count"] == result["reference_count"]
        assert result["reference_count"] <= 4
        assert "persisted_retrieval_run_ids" not in result

    positive, zero, references, insufficient = results
    assert positive["feedback_count"] == 2
    assert zero["state"] == "feedback_persisted"
    assert zero["feedback_count"] == 0
    assert zero["generation_invoked"] is True
    assert zero["notification_outbox_count"] == 0
    assert references["state"] == "references_persisted"
    assert insufficient["state"] == "insufficient"
    assert insufficient["persisted_retrieval_run_id"] is None


def test_baseline_run_result_invalid_fixtures_cover_protected_contract() -> None:
    schema = json.loads(_artifact(ROOT, "baseline-run-result.v1.schema.json"))
    results = json.loads(
        _artifact(ROOT, "fixtures/baseline-run-result.v1.valid.json")
    )["results"]
    cases = json.loads(
        _artifact(ROOT, "fixtures/baseline-run-result.v1.invalid.json")
    )["cases"]
    assert {case["case_id"] for case in cases} == {
        "obsolete_protocol_hash",
        "raw_query_forbidden",
        "feedback_text_forbidden",
        "zero_finding_outbox",
        "zero_finding_generation_not_invoked",
        "references_are_per_chunk",
        "insufficient_has_reference",
        "too_many_job_wide_references",
        "mismatched_reference_evidence_counts",
    }
    for case in cases:
        result = copy.deepcopy(results[case["base_result"]])
        mutation = case["mutation"]
        operation, details = next(iter(mutation.items()))
        path = details["path"].removeprefix("/")
        if operation in {"replace", "add"}:
            result[path] = details["value"]
        assert _semantic_errors(result, schema), case["case_id"]

    obsolete = copy.deepcopy(results[0])
    obsolete["protocol_sha256"] = OBSOLETE_V2_HASH
    assert "protocol" in _semantic_errors(obsolete, schema)
