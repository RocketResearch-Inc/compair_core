from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLI_ROOT = ROOT.parent / "compair-cli"
ARTIFACTS = {
    "baseline-index-result.v1.md": (
        "3686c6533a149a588613bb9ff53c8a8a9ffd5b035affc491466cb1f5d337857a"
    ),
    "baseline-index-result.v1.schema.json": (
        "49a67fc7a79f31136b51858a3ad75ae662b89bb4b66d0cf7330be9aa4f051cbe"
    ),
    "fixtures/baseline-index-result.v1.valid.json": (
        "2c5696c122880069122f0ae43904b88f5c43a013542b7d3f8c49d3e860789034"
    ),
    "fixtures/baseline-index-result.v1.invalid.json": (
        "627df61845ad318f27b7a32028bc2ca27dfb87cff30e4fc72b3aaf67e4f0dc9c"
    ),
}
FORBIDDEN_KEYS = {
    "credentials",
    "endpoint_url",
    "file_content",
    "idempotency_key",
    "lease_token",
    "raw_diff",
    "repository_path",
    "retrieval_query",
    "vector",
}


def _bytes(root: Path, relative: str) -> bytes:
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


def _errors(result: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if set(result) != set(schema["required"]):
        errors.append("fields")
    if result.get("schema_version") != "baseline-index-result.v1":
        errors.append("schema_version")
    if result.get("protocol_version") != "baseline-control-plane.v2" or result.get(
        "protocol_sha256"
    ) != "b278abe007779f05e92509db068f555701c03cba5cf236151e8df231a9b44091":
        errors.append("protocol")
    reason = result.get("reason_code")
    if reason is not None and re.fullmatch(r"[a-z0-9_]{1,64}", reason) is None:
        errors.append("reason")
    state = result.get("state")
    if state == "succeeded":
        if (
            result.get("exit_classification") != "success"
            or result.get("compatible_publication_id") is None
            or result.get("index_fingerprint") is None
            or result.get("index_intent_fingerprint") is None
            or reason is not None
            or result.get("indexed_document_count")
            != result.get("vector_count")
        ):
            errors.append("success")
    elif state in {"queued", "running", "retryable_failed"}:
        if (
            result.get("exit_classification") != "pending"
            or result.get("compatible_publication_id") is not None
            or result.get("index_fingerprint") is not None
            or (state == "retryable_failed" and reason is None)
        ):
            errors.append("pending")
    elif state == "retryable_incomplete":
        if result.get("exit_classification") != "retryable" or reason is None:
            errors.append("retryable")
    elif state in {"terminal_failed", "blocked", "failed"}:
        if result.get("exit_classification") != "failed" or reason is None:
            errors.append("failed")
    elif state == "cancelled":
        if result.get("exit_classification") != "cancelled" or reason is None:
            errors.append("cancelled")
    else:
        errors.append("state")
    if _keys(result) & FORBIDDEN_KEYS:
        errors.append("forbidden")
    return errors


def test_baseline_index_result_artifacts_are_frozen_and_byte_identical() -> None:
    for relative, expected in ARTIFACTS.items():
        core = _bytes(ROOT, relative)
        assert core == _bytes(CLI_ROOT, relative)
        assert hashlib.sha256(core).hexdigest() == expected


def test_baseline_index_result_valid_and_invalid_fixtures() -> None:
    schema = json.loads(_bytes(ROOT, "baseline-index-result.v1.schema.json"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    valid = json.loads(
        _bytes(ROOT, "fixtures/baseline-index-result.v1.valid.json")
    )["results"]
    for result in valid:
        assert _errors(result, schema) == []

    invalid = json.loads(
        _bytes(ROOT, "fixtures/baseline-index-result.v1.invalid.json")
    )["cases"]
    assert invalid
    for case in invalid:
        assert _errors(case["value"], schema), case["case_id"]


def test_baseline_index_result_keeps_v2_pin_and_v1_bytes_unchanged() -> None:
    valid = json.loads(
        _bytes(ROOT, "fixtures/baseline-index-result.v1.valid.json")
    )["results"]
    assert {
        (item["protocol_version"], item["protocol_sha256"]) for item in valid
    } == {
        (
            "baseline-control-plane.v2",
            "b278abe007779f05e92509db068f555701c03cba5cf236151e8df231a9b44091",
        )
    }
    assert hashlib.sha256(
        _bytes(ROOT, "baseline-control-plane.v1.md")
    ).hexdigest() == (
        "3b45287a54d04cea751e9cc3209c5f0783192de13062e682eadcae40af322650"
    )
