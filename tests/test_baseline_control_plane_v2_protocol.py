from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import rfc8785

ROOT = Path(__file__).resolve().parents[1]
CLI_ROOT = ROOT.parent / "compair-cli"
PROTOCOL_DIR = ROOT / "protocol"
CLI_PROTOCOL_DIR = CLI_ROOT / "protocol"
SPEC_PATH = PROTOCOL_DIR / "baseline-control-plane.v2.md"
SCHEMA_PATH = PROTOCOL_DIR / "baseline-control-plane.v2.schema.json"
VALID_FIXTURE_PATH = PROTOCOL_DIR / "fixtures" / "baseline-control-plane.v2.valid.json"
INVALID_FIXTURE_PATH = (
    PROTOCOL_DIR / "fixtures" / "baseline-control-plane.v2.invalid.json"
)
GENERATION_OUTPUT_SPEC_PATH = PROTOCOL_DIR / "baseline-generation-output.v2.md"
GENERATION_OUTPUT_SCHEMA_PATH = (
    PROTOCOL_DIR / "baseline-generation-output.v2.schema.json"
)
GENERATION_OUTPUT_VALID_FIXTURE_PATH = (
    PROTOCOL_DIR / "fixtures" / "baseline-generation-output.v2.valid.json"
)
GENERATION_OUTPUT_INVALID_FIXTURE_PATH = (
    PROTOCOL_DIR / "fixtures" / "baseline-generation-output.v2.invalid.json"
)

V1_SPEC_SHA256 = "3b45287a54d04cea751e9cc3209c5f0783192de13062e682eadcae40af322650"
V1_SCHEMA_SHA256 = "4ea2bbd09c6362b0510cf6cc43dc16f0ec3458fda2525a2409a59d299e801200"
V1_FIXTURE_SHA256 = "bd89803abcdeac97a57bf0c22b9460cf61be8e0b186b58db8fc0c5cfd3dd60c4"
V1_SCANNER_FIXTURE_SHA256 = (
    "e483e017270aff1997aafce4225e4b4787e643084ffe716dfe36acb40c03c553"
)
OBSOLETE_UNRELEASED_V2_SPEC_SHA256 = (
    "c9486b3deb1a494781513109df17d8e8df1281fbc9687960ace711485b50d174"
)

# Filled only from the final raw frozen artifact bytes. Any later change is a
# protocol version change, not a fixture update.
V2_SPEC_SHA256 = "b278abe007779f05e92509db068f555701c03cba5cf236151e8df231a9b44091"
V2_SCHEMA_SHA256 = "10170faf5cecab1861a0e3c831080cbe1073f437b4c668b55c39dd3be9ca631a"
V2_VALID_FIXTURE_SHA256 = (
    "d06ea3ab7194c2ef58eea9af555835ed0f1d29eb8a431fb8d5c68976d2b76003"
)
V2_INVALID_FIXTURE_SHA256 = (
    "64f06b80f17cc4804f72f8bfd599139dc1ab7e681c9f8d37c244f55612894e3a"
)
V2_VALID_FIXTURE_JCS_SHA256 = (
    "8b43d80e15a84f2bafdfa143a0ddbaa7a9912b63f28586b93a0a7c988f1c8d34"
)
V2_INVALID_FIXTURE_JCS_SHA256 = (
    "de66f15097d0346d0f66191f91f79e79fac29cedf6bddd7186b0ad847d92f731"
)

GENERATION_OUTPUT_SPEC_SHA256 = (
    "1dccd3a11ec659a5e8705f9b8acf333a64a21f056265fcd7c96e9c6ac197bb20"
)
GENERATION_OUTPUT_SCHEMA_SHA256 = (
    "39f8e8eaf5e5a219e806d34f46af887d69268a88d5f1d06d45e6c56465e250ed"
)
GENERATION_OUTPUT_VALID_FIXTURE_SHA256 = (
    "b9781155870350dd8b72619e562ea8da6997125229f2064a39947e71a494b488"
)
GENERATION_OUTPUT_INVALID_FIXTURE_SHA256 = (
    "489164e6b5f1596134ce0a4e0092dcdc65a80d0fd173870beafa01fe73ea108f"
)
GENERATION_OUTPUT_VALID_FIXTURE_JCS_SHA256 = (
    "b428181d7fecbb4c2f6bfca00e120ec3347182fc4ef9c43a4ec50066e9d71336"
)
GENERATION_OUTPUT_INVALID_FIXTURE_JCS_SHA256 = (
    "24126307ddf2257f8cf16f2b9d30a6ed740688653fc7f80c5bf11b2b5a214ed3"
)

EXPECTED_RUN_STATES = {
    "queued": (False, "pending"),
    "running": (False, "pending"),
    "references_persisted": (False, "pending"),
    "feedback_persisted": (True, "success"),
    "insufficient": (True, "insufficient"),
    "retryable_failed": (False, "pending"),
    "terminal_failed": (True, "failed"),
    "blocked": (True, "blocked"),
    "cancelled": (True, "cancelled"),
}

FORBIDDEN_SAFE_KEYS = {
    "child_runs",
    "chunk_outcomes",
    "content",
    "content_utf8",
    "credentials",
    "endpoint_url",
    "evidence_content",
    "idempotency_key",
    "idempotency_intent_hash",
    "lease_token",
    "parent_processing_key",
    "persisted_run_ids",
    "raw_diff",
    "relative_path",
    "repository_path",
    "retrieval_query",
    "source_text",
    "source_chunk_id",
}


class ContractValidationError(ValueError):
    pass


class DuplicateJSONKeyError(ValueError):
    pass


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJSONKeyError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(_: str) -> None:
    raise ValueError("non-finite JSON number")


def _strict_loads(value: str | bytes) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    return json.loads(
        value,
        object_pairs_hook=_strict_pairs,
        parse_constant=_reject_constant,
    )


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
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    raise AssertionError(f"unsupported test validator type: {expected}")


def _resolve_ref(root: dict[str, Any], reference: str) -> dict[str, Any] | bool:
    if not reference.startswith("#/"):
        raise AssertionError("protocol schema uses local references only")
    value: Any = root
    for component in reference[2:].split("/"):
        value = value[component.replace("~1", "/").replace("~0", "~")]
    assert isinstance(value, (dict, bool))
    return value


def _validate_schema(
    value: Any,
    rule: dict[str, Any] | bool,
    root: dict[str, Any],
    path: str = "$",
) -> None:
    if rule is False:
        raise ContractValidationError(f"{path}: forbidden by schema")
    if rule is True:
        return
    if "$ref" in rule:
        _validate_schema(value, _resolve_ref(root, rule["$ref"]), root, path)
        return

    for child in rule.get("allOf", []):
        _validate_schema(value, child, root, path)

    if "if" in rule:
        try:
            _validate_schema(value, rule["if"], root, path)
        except ContractValidationError:
            if "else" in rule:
                _validate_schema(value, rule["else"], root, path)
        else:
            if "then" in rule:
                _validate_schema(value, rule["then"], root, path)

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
        if matches == 0 or (keyword == "oneOf" and matches != 1):
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
        prefix_items = rule.get("prefixItems", [])
        for index, child in enumerate(prefix_items):
            if index < len(value):
                _validate_schema(value[index], child, root, f"{path}[{index}]")
        if "items" in rule:
            start = len(prefix_items)
            for index in range(start, len(value)):
                _validate_schema(value[index], rule["items"], root, f"{path}[{index}]")

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
        if not math.isfinite(value):
            raise ContractValidationError(f"{path}: non-finite number")
        if value < rule.get("minimum", value):
            raise ContractValidationError(f"{path}: below minimum")
        if "maximum" in rule and value > rule["maximum"]:
            raise ContractValidationError(f"{path}: above maximum")


def _validate_contract(message: dict[str, Any], schema: dict[str, Any]) -> None:
    _validate_schema(message, schema, schema)
    if message["message_type"] == "run_submit":
        query = message["retrieval_query"]
        query_bytes = query["text"].encode("utf-8")
        if not query["text"].strip():
            raise ContractValidationError("retrieval query is blank")
        if len(query_bytes) > 8_000_000:
            raise ContractValidationError("retrieval query exceeds byte limit")
        if len(query_bytes) != query["byte_size"]:
            raise ContractValidationError("retrieval query byte size mismatch")
        if _sha256(query_bytes) != query["sha256"]:
            raise ContractValidationError("retrieval query hash mismatch")
    if (
        message["message_type"] == "job_status"
        and message["operation"] == "baseline_run"
    ):
        effects = message["effects"]
        if message["retrieval_status"] != "ok" and any(
            (
                effects["evidence_count"],
                effects["reference_count"],
                effects["feedback_count"],
                effects["notification_outbox_count"],
            )
        ):
            raise ContractValidationError("non-ok retrieval has durable effects")
        if message["retrieval_status"] != "ok" and effects["generation_invoked"]:
            raise ContractValidationError("non-ok retrieval invoked generation")
        if effects["evidence_count"] != effects["reference_count"]:
            raise ContractValidationError("evidence/reference count mismatch")
        if message["state"] == "feedback_persisted" and (
            effects["feedback_count"] > effects["reference_count"]
        ):
            raise ContractValidationError("feedback count exceeds references")
        if (
            message["state"] == "feedback_persisted"
            and effects["feedback_count"] == 0
            and effects["notification_outbox_count"] != 0
        ):
            raise ContractValidationError("zero findings cannot create an outbox")
    if (
        message["message_type"] == "job_status"
        and message["operation"] == "index_build"
        and message["state"] == "succeeded"
    ):
        if message["result"]["document_count"] != message["progress"]["document_count"]:
            raise ContractValidationError("index document count mismatch")
        if message["result"]["vector_count"] != message["progress"]["vector_count"]:
            raise ContractValidationError("index vector count mismatch")


def _messages(fixtures: dict[str, Any]) -> list[dict[str, Any]]:
    return fixtures["messages"]


def _find_base(
    messages: list[dict[str, Any]], selector: dict[str, Any]
) -> dict[str, Any]:
    if "message_type" in selector:
        matches = [
            value
            for value in messages
            if value["message_type"] == selector["message_type"]
        ]
        return matches[selector.get("occurrence", 1) - 1]
    matches = [
        value
        for value in messages
        if value.get("message_type") == "job_status"
        and value.get("operation") == selector["operation"]
        and value.get("state") == selector["state"]
    ]
    assert len(matches) == 1
    return matches[0]


def _pointer_parent(value: dict[str, Any], pointer: str) -> tuple[dict[str, Any], str]:
    components = [
        component.replace("~1", "/").replace("~0", "~")
        for component in pointer.lstrip("/").split("/")
    ]
    parent: dict[str, Any] = value
    for component in components[:-1]:
        parent = parent[component]
    return parent, components[-1]


def _mutate(base: dict[str, Any], mutation: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    parent, key = _pointer_parent(result, mutation["path"])
    operation = mutation["operation"]
    if operation == "delete":
        del parent[key]
    elif operation == "set":
        parent[key] = mutation["value"]
    elif operation == "repeat_ascii":
        text = mutation["character"] * mutation["count"]
        parent[key] = text
        if mutation.get("synchronize_declared_size_and_hash"):
            query = result["retrieval_query"]
            query["byte_size"] = len(text.encode("utf-8"))
            query["sha256"] = _sha256(text.encode("utf-8"))
    else:  # pragma: no cover - frozen fixture operation inventory
        raise AssertionError(f"unknown mutation operation {operation}")
    return result


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return _strict_loads(SCHEMA_PATH.read_bytes())


@pytest.fixture(scope="module")
def valid_fixtures() -> dict[str, Any]:
    return _strict_loads(VALID_FIXTURE_PATH.read_bytes())


@pytest.fixture(scope="module")
def invalid_fixtures() -> dict[str, Any]:
    return _strict_loads(INVALID_FIXTURE_PATH.read_bytes())


def test_v1_artifacts_remain_byte_frozen() -> None:
    expected = {
        "baseline-control-plane.v1.md": V1_SPEC_SHA256,
        "baseline-control-plane.v1.schema.json": V1_SCHEMA_SHA256,
        "fixtures/baseline-control-plane.v1.valid.json": V1_FIXTURE_SHA256,
        "fixtures/baseline-scanner-inputs.v1.valid.json": V1_SCANNER_FIXTURE_SHA256,
    }
    for relative, digest in expected.items():
        assert _sha256((PROTOCOL_DIR / relative).read_bytes()) == digest


def test_v2_artifacts_are_byte_frozen_and_protocol_hash_is_exact(
    schema: dict[str, Any],
) -> None:
    assert _sha256(SPEC_PATH.read_bytes()) == V2_SPEC_SHA256
    assert _sha256(SCHEMA_PATH.read_bytes()) == V2_SCHEMA_SHA256
    assert _sha256(VALID_FIXTURE_PATH.read_bytes()) == V2_VALID_FIXTURE_SHA256
    assert _sha256(INVALID_FIXTURE_PATH.read_bytes()) == V2_INVALID_FIXTURE_SHA256
    assert schema["$defs"]["protocol_sha256"]["const"] == V2_SPEC_SHA256


def test_v2_fixture_values_have_frozen_rfc8785_hashes() -> None:
    assert (
        _sha256(rfc8785.dumps(_strict_loads(VALID_FIXTURE_PATH.read_bytes())))
        == V2_VALID_FIXTURE_JCS_SHA256
    )
    assert (
        _sha256(rfc8785.dumps(_strict_loads(INVALID_FIXTURE_PATH.read_bytes())))
        == V2_INVALID_FIXTURE_JCS_SHA256
    )


def test_core_and_cli_v2_artifacts_are_byte_identical() -> None:
    for relative in (
        "baseline-control-plane.v2.md",
        "baseline-control-plane.v2.schema.json",
        "baseline-generation-output.v2.md",
        "baseline-generation-output.v2.schema.json",
        "fixtures/baseline-control-plane.v2.valid.json",
        "fixtures/baseline-control-plane.v2.invalid.json",
        "fixtures/baseline-generation-output.v2.valid.json",
        "fixtures/baseline-generation-output.v2.invalid.json",
    ):
        assert (PROTOCOL_DIR / relative).read_bytes() == (
            CLI_PROTOCOL_DIR / relative
        ).read_bytes()


def test_generation_output_v2_artifacts_are_frozen() -> None:
    expected = {
        GENERATION_OUTPUT_SPEC_PATH: GENERATION_OUTPUT_SPEC_SHA256,
        GENERATION_OUTPUT_SCHEMA_PATH: GENERATION_OUTPUT_SCHEMA_SHA256,
        GENERATION_OUTPUT_VALID_FIXTURE_PATH: (GENERATION_OUTPUT_VALID_FIXTURE_SHA256),
        GENERATION_OUTPUT_INVALID_FIXTURE_PATH: (
            GENERATION_OUTPUT_INVALID_FIXTURE_SHA256
        ),
    }
    for path, digest in expected.items():
        assert _sha256(path.read_bytes()) == digest
    assert (
        _sha256(
            rfc8785.dumps(
                _strict_loads(GENERATION_OUTPUT_VALID_FIXTURE_PATH.read_bytes())
            )
        )
        == GENERATION_OUTPUT_VALID_FIXTURE_JCS_SHA256
    )
    assert (
        _sha256(
            rfc8785.dumps(
                _strict_loads(GENERATION_OUTPUT_INVALID_FIXTURE_PATH.read_bytes())
            )
        )
        == GENERATION_OUTPUT_INVALID_FIXTURE_JCS_SHA256
    )


def test_generation_output_v2_valid_fixtures_preserve_order() -> None:
    schema = _strict_loads(GENERATION_OUTPUT_SCHEMA_PATH.read_bytes())
    fixtures = _strict_loads(GENERATION_OUTPUT_VALID_FIXTURE_PATH.read_bytes())
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    outputs = fixtures["outputs"]
    assert [output["outcome"] for output in outputs] == [
        "no_findings",
        "findings",
        "findings",
    ]
    for output in outputs:
        _validate_schema(output, schema, schema)
    assert outputs[0]["findings"] == []
    assert [item["feedback"] for item in outputs[2]["findings"]] == [
        "First ordered finding.",
        "Second ordered finding.",
        "Third ordered finding.",
        "Fourth ordered finding.",
    ]


def test_generation_output_v2_invalid_fixtures_fail_closed() -> None:
    schema = _strict_loads(GENERATION_OUTPUT_SCHEMA_PATH.read_bytes())
    fixtures = _strict_loads(GENERATION_OUTPUT_INVALID_FIXTURE_PATH.read_bytes())
    case_ids = {case["case_id"] for case in fixtures["cases"]}
    assert {
        "plain_text_invalid",
        "blank_output_invalid",
        "none_sentinel_invalid",
        "json_none_string_invalid",
        "malformed_json_invalid",
        "duplicate_outcome_key_invalid",
        "no_findings_with_finding_invalid",
        "findings_without_finding_invalid",
        "blank_feedback_invalid",
        "too_many_findings_invalid",
        "extra_output_property_invalid",
        "extra_finding_property_invalid",
        "schema_version_mismatch_invalid",
    } == case_ids
    for case in fixtures["cases"]:
        if "raw_output" in case:
            try:
                value = _strict_loads(case["raw_output"])
            except (DuplicateJSONKeyError, json.JSONDecodeError):
                continue
        else:
            value = case["value"]
        with pytest.raises(ContractValidationError):
            _validate_schema(value, schema, schema)


def test_every_valid_v2_message_validates_and_uses_exact_protocol(
    schema: dict[str, Any], valid_fixtures: dict[str, Any]
) -> None:
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    messages = _messages(valid_fixtures)
    assert {value["message_type"] for value in messages} == {
        "capabilities_request",
        "capabilities",
        "index_build_submit",
        "run_submit",
        "job_accepted",
        "job_status_request",
        "job_status",
        "error",
    }
    for message in messages:
        _validate_contract(message, schema)
        assert message["protocol_version"] == "baseline-control-plane.v2"
        assert message["protocol_sha256"] == V2_SPEC_SHA256


def test_non_ascii_query_bytes_hash_and_safe_provenance_are_exact(
    valid_fixtures: dict[str, Any],
) -> None:
    messages = _messages(valid_fixtures)
    submit = _find_base(messages, {"message_type": "run_submit"})
    query = submit["retrieval_query"]
    assert "café" in query["text"]
    assert len(query["text"]) == 145
    assert len(query["text"].encode("utf-8")) == query["byte_size"] == 149
    assert _sha256(query["text"].encode("utf-8")) == query["sha256"]
    for status in (
        value
        for value in messages
        if value.get("operation") == "baseline_run"
        and value["message_type"] == "job_status"
    ):
        assert status["query_provenance"] == {
            "sha256": query["sha256"],
            "length": 145,
            "byte_size": 149,
            "origin": "explicit",
        }


def test_run_state_exit_classification_and_effect_rules_are_frozen(
    schema: dict[str, Any], valid_fixtures: dict[str, Any]
) -> None:
    messages = _messages(valid_fixtures)
    states = {
        value["state"]: value
        for value in messages
        if value.get("operation") == "baseline_run"
        and value["message_type"] == "job_status"
    }
    running = copy.deepcopy(states["queued"])
    running["state"] = "running"
    running["attempt"] = 1
    _validate_contract(running, schema)
    states["running"] = running
    assert set(states) == set(EXPECTED_RUN_STATES)
    for state, (terminal, exit_classification) in EXPECTED_RUN_STATES.items():
        assert states[state]["terminal"] is terminal
        assert states[state]["exit_classification"] == exit_classification

    insufficient = states["insufficient"]
    assert insufficient["retrieval_status"] == "insufficient"
    assert insufficient["effects"] == {
        "evidence_count": 0,
        "reference_count": 0,
        "feedback_count": 0,
        "generation_invoked": False,
        "notification_outbox_count": 0,
        "persisted_run_id": None,
    }
    completed_without_findings = states["feedback_persisted"]
    assert completed_without_findings["terminal"] is True
    assert completed_without_findings["exit_classification"] == "success"
    assert completed_without_findings["retrieval_status"] == "ok"
    assert completed_without_findings["effects"] == {
        "evidence_count": 2,
        "reference_count": 2,
        "feedback_count": 0,
        "generation_invoked": True,
        "notification_outbox_count": 0,
        "persisted_run_id": "90000000-0000-4000-8000-000000000001",
    }

    completed_with_findings = copy.deepcopy(completed_without_findings)
    completed_with_findings["effects"]["feedback_count"] = 2
    completed_with_findings["effects"]["notification_outbox_count"] = 1
    _validate_contract(completed_with_findings, schema)

    contradictory = copy.deepcopy(completed_without_findings)
    contradictory["effects"]["notification_outbox_count"] = 1
    with pytest.raises(ContractValidationError):
        _validate_contract(contradictory, schema)


def test_document_level_run_semantics_and_job_wide_budget_are_frozen(
    schema: dict[str, Any], valid_fixtures: dict[str, Any]
) -> None:
    messages = _messages(valid_fixtures)
    submit = _find_base(messages, {"message_type": "run_submit"})
    assert "source_document_id" in submit
    assert "source_chunk_id" not in submit
    assert "child_runs" not in submit
    assert "chunk_outcomes" not in submit

    capabilities = _find_base(messages, {"message_type": "capabilities"})
    assert capabilities["limits"]["selected_evidence_items"] == 4
    assert capabilities["limits"]["selected_evidence_characters"] == 16_000

    statuses = [
        value
        for value in messages
        if value.get("message_type") == "job_status"
        and value.get("operation") == "baseline_run"
    ]
    for status in statuses:
        assert "source_chunk_id" not in status
        assert "child_runs" not in status
        assert "chunk_outcomes" not in status
        effects = status["effects"]
        assert set(effects) == {
            "evidence_count",
            "reference_count",
            "feedback_count",
            "generation_invoked",
            "notification_outbox_count",
            "persisted_run_id",
        }
        assert effects["evidence_count"] <= 4
        assert effects["reference_count"] <= 4
        assert effects["evidence_count"] == effects["reference_count"]
        _validate_contract(status, schema)

    spec = SPEC_PATH.read_text(encoding="utf-8")
    for frozen_clause in (
        "invokes `baseline_v1` retrieval exactly once",
        "per-chunk fan-out is not part",
        "does not require or manufacture a `source_chunk_id`",
        "at most 16,000 selected-content characters",
        "filtering, content deduplication, and refill only within",
    ):
        assert frozen_clause in spec


def test_document_level_idempotent_replay_preserves_all_public_identities(
    valid_fixtures: dict[str, Any],
) -> None:
    accepted = [
        value
        for value in _messages(valid_fixtures)
        if value.get("message_type") == "job_accepted"
        and value.get("operation") == "baseline_run"
    ]
    assert len(accepted) == 2
    original, replay = accepted
    assert original["replayed"] is False
    assert replay["replayed"] is True
    for field in ("group_id", "job_id", "processing_run_id"):
        assert replay[field] == original[field]

    durable = [
        value
        for value in _messages(valid_fixtures)
        if value.get("operation") == "baseline_run"
        and value.get("message_type") == "job_status"
        and value["state"] in {"references_persisted", "feedback_persisted"}
    ]
    assert {value["effects"]["persisted_run_id"] for value in durable} == {
        "90000000-0000-4000-8000-000000000001"
    }


def test_chunk_aggregation_and_multiple_retrieval_runs_are_invalid_fixtures(
    invalid_fixtures: dict[str, Any],
) -> None:
    case_ids = {case["case_id"] for case in invalid_fixtures["cases"]}
    assert {
        "job_wide_evidence_item_limit_exceeded",
        "job_wide_reference_limit_exceeded",
        "job_wide_evidence_character_limit_exceeded",
        "multiple_persisted_run_ids_forbidden",
        "per_chunk_child_manifest_forbidden",
        "aggregate_chunk_outcomes_forbidden",
        "source_chunk_authority_forbidden",
        "zero_findings_nonzero_outbox_forbidden",
        "zero_findings_generation_not_invoked",
        "zero_findings_missing_persisted_run",
        "zero_findings_without_evidence",
        "zero_findings_without_references",
        "zero_findings_feedback_payload_forbidden",
    } <= case_ids


def test_capability_truth_table_is_schema_enforced_and_d0_is_unavailable(
    schema: dict[str, Any], valid_fixtures: dict[str, Any]
) -> None:
    messages = _messages(valid_fixtures)
    capabilities = _find_base(messages, {"message_type": "capabilities"})
    for capability in capabilities["operations"].values():
        assert capability == {
            "submission": "unavailable",
            "endpoint": "unavailable",
            "dispatch": "unavailable",
            "readiness": "unavailable",
            "reason_code": "capability_unavailable",
        }

    safe = copy.deepcopy(capabilities)
    safe["operations"]["index_build"] = {
        "submission": "safe",
        "endpoint": "authenticated_post",
        "dispatch": "manual",
        "readiness": "ready",
        "reason_code": None,
    }
    _validate_contract(safe, schema)
    lying = copy.deepcopy(safe)
    lying["operations"]["index_build"]["endpoint"] = "unavailable"
    with pytest.raises(ContractValidationError):
        _validate_contract(lying, schema)

    api_source = (ROOT / "compair_core" / "api.py").read_text(encoding="utf-8")
    assert '"/baseline/control/v2/capabilities"' in api_source
    assert '"/baseline/control/v2/index-builds"' in api_source
    assert '"/baseline/control/v2/index-builds/status"' in api_source
    assert '"/baseline/control/v2/runs"' in api_source
    assert '"/baseline/control/v2/runs/status"' in api_source
    assert "baseline_runs_enabled" in api_source


def test_v1_remains_staging_only_and_does_not_advertise_v2_operations() -> None:
    fixtures = _strict_loads(
        (
            PROTOCOL_DIR / "fixtures" / "baseline-control-plane.v1.valid.json"
        ).read_bytes()
    )
    capabilities = next(
        value for value in fixtures if value["message_type"] == "capabilities"
    )
    assert capabilities["operations"]["index_build"] == "unavailable"
    assert capabilities["operations"]["baseline_run"] == "unavailable"


def test_safe_responses_never_contain_protected_fields_or_query_text(
    valid_fixtures: dict[str, Any],
) -> None:
    messages = _messages(valid_fixtures)
    raw_query = _find_base(messages, {"message_type": "run_submit"})["retrieval_query"][
        "text"
    ]

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            assert not (FORBIDDEN_SAFE_KEYS & value.keys())
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for message in messages:
        if message["message_type"] in {
            "job_accepted",
            "job_status",
            "error",
            "capabilities",
        }:
            visit(message)
            assert raw_query not in json.dumps(
                message, ensure_ascii=False, sort_keys=True
            )


def test_invalid_fixture_recipes_are_rejected_at_the_frozen_layer(
    schema: dict[str, Any],
    valid_fixtures: dict[str, Any],
    invalid_fixtures: dict[str, Any],
) -> None:
    messages = _messages(valid_fixtures)
    for case in invalid_fixtures["cases"]:
        if "raw_bytes_hex" in case:
            with pytest.raises(UnicodeDecodeError):
                _strict_loads(bytes.fromhex(case["raw_bytes_hex"]))
            continue
        if "raw_json" in case:
            expected = (
                DuplicateJSONKeyError
                if case["expected_error"] == "duplicate_json_key"
                else ValueError
            )
            with pytest.raises(expected):
                _strict_loads(case["raw_json"])
            continue
        invalid = _mutate(_find_base(messages, case["base"]), case["mutation"])
        with pytest.raises(ContractValidationError):
            _validate_contract(invalid, schema)


def test_exact_version_hash_matching_never_downgrades(
    schema: dict[str, Any], valid_fixtures: dict[str, Any]
) -> None:
    submit = copy.deepcopy(
        _find_base(_messages(valid_fixtures), {"message_type": "run_submit"})
    )
    for field, value in (
        ("protocol_version", "baseline-control-plane.v1"),
        ("protocol_sha256", V1_SPEC_SHA256),
        ("protocol_sha256", OBSOLETE_UNRELEASED_V2_SPEC_SHA256),
    ):
        invalid = copy.deepcopy(submit)
        invalid[field] = value
        with pytest.raises(ContractValidationError):
            _validate_contract(invalid, schema)


def test_public_messages_have_no_worker_lease_or_client_parent_key(
    valid_fixtures: dict[str, Any],
) -> None:
    serialized = json.dumps(valid_fixtures, ensure_ascii=False, sort_keys=True)
    assert "lease_token" not in serialized
    assert "parent_processing_key" not in serialized
    submit = _find_base(_messages(valid_fixtures), {"message_type": "run_submit"})
    assert "processing_run_id" not in submit
    accepted = next(
        value
        for value in _messages(valid_fixtures)
        if value["message_type"] == "job_accepted"
        and value["operation"] == "baseline_run"
    )
    assert accepted["processing_run_id"]
