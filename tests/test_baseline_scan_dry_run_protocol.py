from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CLI_ROOT = ROOT.parent / "compair-cli"
ARTIFACTS = {
    "baseline-scan-dry-run.v1.md": (
        "080633b7af37a7dfed4998527a1e7d1877bee364385e55c9027a53cd81e66ca4"
    ),
    "baseline-scan-dry-run.v1.schema.json": (
        "9dc19feca68ee5aa655a397b7001c1d675592d6f146049c7469ebe6befe636fd"
    ),
    "fixtures/baseline-scan-dry-run.v1.valid.json": (
        "35ef126001808d4b6e9ebb1072dd6e9b12772775bb35f867441876221b7719f4"
    ),
    "fixtures/baseline-scan-dry-run.v1.invalid.json": (
        "cf1e52d90d552f0b91d737ea38556ab439962733166476c31600888d497ce683"
    ),
}


def _protocol(root: Path, relative: str) -> bytes:
    return (root / "protocol" / relative).read_bytes()


def _semantic_errors(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    counts = report.get("counts", {})
    siblings = report.get("sibling_repositories", [])
    if counts.get("repository_count") != len(siblings):
        errors.append("repository_count")
    if counts.get("file_count") != (
        counts.get("supported_file_count", -1) + counts.get("skipped_file_count", -1)
    ):
        errors.append("file_count")
    skips = report.get("skip_reason_counts", {})
    if sum(skips.values()) != counts.get("skipped_file_count"):
        errors.append("skip_reason_counts")
    parts = report.get("parts", [])
    if [part.get("part_ordinal") for part in parts] != list(range(1, len(parts) + 1)):
        errors.append("part_order")
    if sum(part.get("file_count", -1) for part in parts) != counts.get(
        "supported_file_count"
    ):
        errors.append("part_file_count")
    if sum(part.get("decoded_content_bytes", -1) for part in parts) != counts.get(
        "supported_content_bytes"
    ):
        errors.append("part_content_bytes")
    planned = (
        report.get("manifest_request_bytes", -1)
        + report.get("commit_request_bytes", -1)
        + sum(part.get("request_bytes", -1) for part in parts)
    )
    if planned != report.get("maximum_planned_upload_bytes"):
        errors.append("maximum_planned_upload_bytes")
    return errors


def _shape_errors(report: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = set(schema["required"])
    if set(report) != required:
        errors.append("top_level_fields")
    for field, expected in (
        ("schema_version", "baseline-scan-dry-run.v1"),
        ("protocol_version", "baseline-control-plane.v1"),
        (
            "protocol_sha256",
            "3b45287a54d04cea751e9cc3209c5f0783192de13062e682eadcae40af322650",
        ),
    ):
        if report.get(field) != expected:
            errors.append(field)
    nested_fields = {
        "changed_repository": {
            "repository_registration_id",
            "base_revision",
            "head_revision",
            "source_document_id",
        },
        "counts": {
            "repository_count",
            "file_count",
            "supported_file_count",
            "skipped_file_count",
            "supported_content_bytes",
        },
        "skip_reason_counts": {
            "non_utf8",
            "oversized",
            "symlink",
            "excluded_directory",
            "unsupported_file_type",
            "unreadable",
        },
        "raw_diff": {
            "representation",
            "base_revision",
            "head_revision",
            "byte_size",
            "sha256",
        },
    }
    for field, expected in nested_fields.items():
        if not isinstance(report.get(field), dict) or set(report[field]) != expected:
            errors.append(field)
    part_fields = {
        "part_ordinal",
        "part_sha256",
        "file_count",
        "decoded_content_bytes",
        "request_bytes",
    }
    if not isinstance(report.get("parts"), list) or any(
        not isinstance(part, dict) or set(part) != part_fields
        for part in report.get("parts", [])
    ):
        errors.append("parts")
    if report.get("warnings") != ["dry_run_only", "no_network_or_persistence"]:
        errors.append("warnings")
    if report.get("errors") != []:
        errors.append("errors")
    return errors


def _set_path(value: dict[str, Any], path: str, replacement: Any) -> None:
    current: Any = value
    components = path.split(".")
    for component in components[:-1]:
        current = current[int(component)] if component.isdigit() else current[component]
    final = components[-1]
    if final.isdigit():
        current[int(final)] = replacement
    else:
        current[final] = replacement


def test_dry_run_v1_artifacts_are_frozen_and_core_cli_identical() -> None:
    for relative, expected_hash in ARTIFACTS.items():
        core = _protocol(ROOT, relative)
        cli = _protocol(CLI_ROOT, relative)
        assert core == cli
        assert hashlib.sha256(core).hexdigest() == expected_hash


def test_dry_run_v1_schema_valid_and_invalid_fixtures() -> None:
    schema = json.loads(_protocol(ROOT, "baseline-scan-dry-run.v1.schema.json"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    valid = json.loads(_protocol(ROOT, "fixtures/baseline-scan-dry-run.v1.valid.json"))
    assert len(valid) == 1
    assert _shape_errors(valid[0], schema) == []
    assert _semantic_errors(valid[0]) == []

    invalid = json.loads(
        _protocol(ROOT, "fixtures/baseline-scan-dry-run.v1.invalid.json")
    )
    for case in invalid:
        candidate = copy.deepcopy(valid[0])
        mutation = case["mutation"]
        removed = mutation.get("remove")
        if removed:
            candidate.pop(removed)
        for path, replacement in mutation.get("replace", {}).items():
            _set_path(candidate, path, replacement)
        candidate.update(mutation.get("add", {}))
        assert _shape_errors(candidate, schema) or _semantic_errors(candidate), case[
            "case"
        ]


def test_dry_run_v1_privacy_and_v1_control_hash_unchanged() -> None:
    schema_text = _protocol(ROOT, "baseline-scan-dry-run.v1.schema.json").decode()
    for forbidden in (
        "content_utf8",
        "raw_diff_text",
        "local_path",
        "remote_url",
        "idempotency_key",
        "lease_token",
        "credential",
    ):
        assert forbidden not in schema_text
    assert (
        hashlib.sha256(_protocol(ROOT, "baseline-control-plane.v1.md")).hexdigest()
        == "3b45287a54d04cea751e9cc3209c5f0783192de13062e682eadcae40af322650"
    )
