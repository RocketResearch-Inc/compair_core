from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = ROOT / "protocol"
SPEC_PATH = PROTOCOL_DIR / "baseline-control-plane.v1.md"
SCHEMA_PATH = PROTOCOL_DIR / "baseline-control-plane.v1.schema.json"
FIXTURE_PATH = PROTOCOL_DIR / "fixtures" / "baseline-control-plane.v1.valid.json"
SCANNER_FIXTURE_PATH = (
    PROTOCOL_DIR / "fixtures" / "baseline-scanner-inputs.v1.valid.json"
)

PINNED_SPEC_SHA256 = "3b45287a54d04cea751e9cc3209c5f0783192de13062e682eadcae40af322650"
PINNED_SCHEMA_SHA256 = (
    "4ea2bbd09c6362b0510cf6cc43dc16f0ec3458fda2525a2409a59d299e801200"
)
PINNED_FIXTURE_SHA256 = (
    "bd89803abcdeac97a57bf0c22b9460cf61be8e0b186b58db8fc0c5cfd3dd60c4"
)
PINNED_SCANNER_FIXTURE_SHA256 = (
    "e483e017270aff1997aafce4225e4b4787e643084ffe716dfe36acb40c03c553"
)


class ContractValidationError(ValueError):
    pass


def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    raise AssertionError(f"unsupported test validator type: {expected}")


def _resolve_ref(root: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise AssertionError("fixtures use local JSON Schema references only")
    value: Any = root
    for component in reference[2:].split("/"):
        value = value[component.replace("~1", "/").replace("~0", "~")]
    assert isinstance(value, dict)
    return value


def _validate_schema(
    value: Any,
    rule: dict[str, Any],
    root: dict[str, Any],
    path: str = "$",
) -> None:
    if "$ref" in rule:
        _validate_schema(value, _resolve_ref(root, rule["$ref"]), root, path)
        return

    for keyword in ("allOf",):
        for child in rule.get(keyword, []):
            _validate_schema(value, child, root, path)

    for keyword in ("oneOf", "anyOf"):
        if keyword not in rule:
            continue
        matches = 0
        for child in rule[keyword]:
            try:
                _validate_schema(value, child, root, path)
            except ContractValidationError:
                continue
            matches += 1
        expected = 1 if keyword == "oneOf" else None
        if matches == 0 or (expected is not None and matches != expected):
            raise ContractValidationError(
                f"{path}: expected {keyword} match; found {matches}"
            )

    if "const" in rule and value != rule["const"]:
        raise ContractValidationError(f"{path}: does not match const")
    if "enum" in rule and value not in rule["enum"]:
        raise ContractValidationError(f"{path}: value is outside enum")

    declared_type = rule.get("type")
    if declared_type is not None:
        types = [declared_type] if isinstance(declared_type, str) else declared_type
        if not any(_json_type_matches(value, item) for item in types):
            raise ContractValidationError(f"{path}: wrong type")

    if isinstance(value, dict):
        properties = rule.get("properties", {})
        missing = set(rule.get("required", [])) - value.keys()
        if missing:
            raise ContractValidationError(f"{path}: missing {sorted(missing)}")
        if rule.get("additionalProperties") is False:
            extras = value.keys() - properties.keys()
            if extras:
                raise ContractValidationError(f"{path}: extra {sorted(extras)}")
        for key, child in properties.items():
            if key in value:
                _validate_schema(value[key], child, root, f"{path}.{key}")

    if isinstance(value, list):
        if len(value) < rule.get("minItems", 0):
            raise ContractValidationError(f"{path}: too few items")
        if "maxItems" in rule and len(value) > rule["maxItems"]:
            raise ContractValidationError(f"{path}: too many items")
        if "items" in rule:
            for index, item in enumerate(value):
                _validate_schema(item, rule["items"], root, f"{path}[{index}]")

    if isinstance(value, str):
        if len(value) < rule.get("minLength", 0):
            raise ContractValidationError(f"{path}: string is too short")
        if "maxLength" in rule and len(value) > rule["maxLength"]:
            raise ContractValidationError(f"{path}: string is too long")
        if "pattern" in rule and re.search(rule["pattern"], value) is None:
            raise ContractValidationError(f"{path}: pattern mismatch")
        if rule.get("format") == "uuid":
            try:
                uuid.UUID(value)
            except ValueError as error:
                raise ContractValidationError(f"{path}: invalid UUID") from error
        if rule.get("format") == "date-time":
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as error:
                raise ContractValidationError(f"{path}: invalid date-time") from error

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < rule.get("minimum", value):
            raise ContractValidationError(f"{path}: below minimum")
        if "maximum" in rule and value > rule["maximum"]:
            raise ContractValidationError(f"{path}: above maximum")


def _canonical_bytes(value: Any) -> bytes:
    # All frozen fixtures are ASCII, so this is RFC 8785-equivalent for their
    # strings, integers, nulls, booleans, arrays, and objects.
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def messages() -> dict[str, dict[str, Any]]:
    values = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return {value["message_type"]: value for value in values}


def test_shared_protocol_artifacts_are_pinned() -> None:
    assert _sha256(SPEC_PATH.read_bytes()) == PINNED_SPEC_SHA256
    assert _sha256(SCHEMA_PATH.read_bytes()) == PINNED_SCHEMA_SHA256
    assert _sha256(FIXTURE_PATH.read_bytes()) == PINNED_FIXTURE_SHA256
    assert _sha256(SCANNER_FIXTURE_PATH.read_bytes()) == PINNED_SCANNER_FIXTURE_SHA256


def test_local_scanner_input_fixture_is_explicit_and_schema_valid(
    schema: dict[str, Any], messages: dict[str, dict[str, Any]]
) -> None:
    scanner_input = json.loads(SCANNER_FIXTURE_PATH.read_text(encoding="utf-8"))
    changed = scanner_input["changed"]
    siblings = scanner_input["siblings"]
    _validate_schema(changed, schema["$defs"]["changed_repository_input"], schema)
    for sibling in siblings:
        _validate_schema(sibling, schema["$defs"]["sibling_repository_input"], schema)

    assert scanner_input["group_id"]
    assert scanner_input["dry_run"] is True
    assert scanner_input["json"] is True
    assert changed["repository_revision"] == scanner_input["head_revision"]
    assert changed["repository_id"] not in {
        sibling["repository_id"] for sibling in siblings
    }
    assert len(
        {(item["repository_name"], item["repository_id"]) for item in siblings}
    ) == len(siblings)

    scan_plan = json.dumps(messages["scan_plan"], sort_keys=True)
    assert changed["local_path"] not in scan_plan
    assert all(sibling["local_path"] not in scan_plan for sibling in siblings)

    unknown_field = copy.deepcopy(changed)
    unknown_field["remote_url"] = "not permitted"
    with pytest.raises(ContractValidationError):
        _validate_schema(
            unknown_field, schema["$defs"]["changed_repository_input"], schema
        )

    symbolic_revision = copy.deepcopy(siblings[0])
    symbolic_revision["repository_revision"] = "main"
    with pytest.raises(ContractValidationError):
        _validate_schema(
            symbolic_revision, schema["$defs"]["sibling_repository_input"], schema
        )


def test_every_versioned_message_fixture_validates(
    schema: dict[str, Any], messages: dict[str, dict[str, Any]]
) -> None:
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert set(messages) == {
        "scan_plan",
        "snapshot_begin",
        "snapshot_content_part",
        "snapshot_commit",
        "index_build_submit",
        "run_submit",
        "job_accepted",
        "job_status_request",
        "job_status",
        "error",
        "capabilities_request",
        "capabilities",
    }
    for message in messages.values():
        _validate_schema(message, schema, schema)


def test_manifest_content_parts_and_diff_are_self_consistent(
    messages: dict[str, dict[str, Any]],
) -> None:
    plan = messages["scan_plan"]
    snapshot = plan["snapshot"]
    assert messages["snapshot_begin"]["snapshot"] == snapshot
    assert snapshot["group_id"] == plan["group_id"]
    assert snapshot["repository_count"] == len(snapshot["sibling_repositories"])
    assert snapshot["total_file_count"] == len(snapshot["files"])
    assert snapshot["supported_file_count"] == 1
    assert snapshot["supported_content_bytes"] == 6

    canonical_manifest = {
        "schema_version": snapshot["schema_version"],
        "changed_repository": snapshot["changed_repository"],
        "sibling_repositories": snapshot["sibling_repositories"],
        "files": snapshot["files"],
    }
    manifest_hash = _sha256(_canonical_bytes(canonical_manifest))
    assert manifest_hash == snapshot["canonical_manifest_hash"]
    assert snapshot["snapshot_id"] == f"bsnap_{manifest_hash}"

    files = snapshot["files"]
    assert [item["ordinal"] for item in files] == [1, 2]
    assert [
        (item["repository_name"], item["relative_path"], item["repository_id"])
        for item in files
    ] == sorted(
        (item["repository_name"], item["relative_path"], item["repository_id"])
        for item in files
    )
    assert files[0]["file_state"] == "supported"
    assert files[0]["skip_reason"] is None
    assert files[0]["content_required"] is True
    assert files[1]["git_mode"] == "120000"
    assert files[1]["file_state"] == "symlink_rejected"
    assert files[1]["skip_reason"] == "symlink"
    assert files[1]["content_required"] is False

    part = messages["snapshot_content_part"]
    content = part["content_items"][0]["content_utf8"].encode("utf-8")
    assert len(content) == part["content_items"][0]["byte_size"]
    assert _sha256(content) == part["content_items"][0]["content_sha256"]
    assert _sha256(_canonical_bytes(part["content_items"])) == part["part_sha256"]

    commit = messages["snapshot_commit"]
    assert commit["parts"] == [
        {"part_ordinal": part["part_ordinal"], "part_sha256": part["part_sha256"]}
    ]
    assert _sha256(_canonical_bytes(commit["parts"])) == commit["content_manifest_hash"]

    raw_diff = messages["run_submit"]["raw_diff"]
    diff_bytes = raw_diff["text"].encode("utf-8")
    assert len(diff_bytes) == raw_diff["byte_size"] == 111
    assert _sha256(diff_bytes) == raw_diff["sha256"]
    assert {key: raw_diff[key] for key in plan["raw_diff"]} == plan["raw_diff"]
    assert raw_diff["base_revision"] == snapshot["changed_repository"]["base_revision"]
    assert raw_diff["head_revision"] == snapshot["changed_repository"]["head_revision"]


def test_schema_rejects_implicit_scope_paths_and_status_secrets(
    schema: dict[str, Any], messages: dict[str, dict[str, Any]]
) -> None:
    missing_group = copy.deepcopy(messages["snapshot_begin"])
    del missing_group["group_id"]
    with pytest.raises(ContractValidationError):
        _validate_schema(missing_group, schema, schema)

    raw_text_in_dry_run = copy.deepcopy(messages["scan_plan"])
    raw_text_in_dry_run["raw_diff"]["text"] = "not permitted"
    with pytest.raises(ContractValidationError):
        _validate_schema(raw_text_in_dry_run, schema, schema)

    leaked_status = copy.deepcopy(messages["job_status"])
    leaked_status["retrieval_query"] = "not permitted"
    with pytest.raises(ContractValidationError):
        _validate_schema(leaked_status, schema, schema)

    for invalid_path in (
        "/src/a.txt",
        "../src/a.txt",
        "src/../a.txt",
        "src//a.txt",
        "src\\a.txt",
        "C:/src/a.txt",
        "src/a.txt/",
    ):
        invalid = copy.deepcopy(messages["scan_plan"])
        invalid["snapshot"]["files"][0]["relative_path"] = invalid_path
        with pytest.raises(ContractValidationError):
            _validate_schema(invalid, schema, schema)


def test_safe_responses_exclude_source_and_query_material(
    messages: dict[str, dict[str, Any]],
) -> None:
    forbidden_keys = {
        "idempotency_key",
        "parent_processing_key",
        "raw_diff",
        "retrieval_query",
        "content_utf8",
        "relative_path",
        "repository_name",
        "endpoint_url",
        "credentials",
    }

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            assert not (forbidden_keys & value.keys())
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    raw_query = messages["run_submit"]["raw_diff"]["text"]
    raw_content = messages["snapshot_content_part"]["content_items"][0]["content_utf8"]
    for message_type in ("job_accepted", "job_status", "error", "capabilities"):
        message = messages[message_type]
        visit(message)
        serialized = json.dumps(message, sort_keys=True)
        assert raw_query not in serialized
        assert raw_content not in serialized
