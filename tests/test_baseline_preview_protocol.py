from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest
from test_baseline_control_plane_v2_protocol import (
    ContractValidationError,
    DuplicateJSONKeyError,
    _strict_loads,
    _validate_schema,
)

ROOT = Path(__file__).resolve().parents[1]
CLI_ROOT = ROOT.parent / "compair-cli"
PROTOCOL_DIR = ROOT / "protocol"
CLI_PROTOCOL_DIR = CLI_ROOT / "protocol"

SPEC_PATH = PROTOCOL_DIR / "baseline-preview.v1.md"
SCHEMA_PATH = PROTOCOL_DIR / "baseline-preview.v1.schema.json"
VALID_PATH = PROTOCOL_DIR / "fixtures" / "baseline-preview.v1.valid.json"
INVALID_PATH = PROTOCOL_DIR / "fixtures" / "baseline-preview.v1.invalid.json"

# These constants are calculated from the final frozen raw bytes. A later byte
# change revises the unreleased preview contract and must update the copies and
# these assertions together.
PREVIEW_SPEC_SHA256 = "3716537f88a7a9db21f83fcd032c0522823f28c13396711ed898f1d6f7756baf"
PREVIEW_SCHEMA_SHA256 = (
    "eda7f9c71a17832340c846115024fecd3401bfbd602475d72aa347bd9b8cc45b"
)
PREVIEW_VALID_FIXTURE_SHA256 = (
    "827f18cdfca62ee56a76c5bc2229c9b7e475276beb372f1e9cd3b6dd0123c3d9"
)
PREVIEW_INVALID_FIXTURE_SHA256 = (
    "a2308d43ec4b2afe1e517ed54dd6f1f1af6bca998a879c56612e28042c800035"
)

CONTROL_V1_SPEC_SHA256 = (
    "3b45287a54d04cea751e9cc3209c5f0783192de13062e682eadcae40af322650"
)
CONTROL_V1_SCHEMA_SHA256 = (
    "4ea2bbd09c6362b0510cf6cc43dc16f0ec3458fda2525a2409a59d299e801200"
)
CONTROL_V2_SPEC_SHA256 = (
    "b278abe007779f05e92509db068f555701c03cba5cf236151e8df231a9b44091"
)
CONTROL_V2_SCHEMA_SHA256 = (
    "10170faf5cecab1861a0e3c831080cbe1073f437b4c668b55c39dd3be9ca631a"
)
GENERATION_SPEC_SHA256 = (
    "e670731777b253f9d5e3984405c2d99871ba26f637a17e6221cc82d97bc8beb1"
)
GENERATION_SCHEMA_SHA256 = (
    "fc5a85d5d38c18775afe0966987ea74e7e9ac072148822c1be60a199e32cca27"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> Any:
    return _strict_loads(path.read_bytes())


def _responses() -> dict[str, dict[str, Any]]:
    fixture = _load(VALID_PATH)
    return {item["case_id"]: item["value"] for item in fixture["responses"]}


def _requests() -> dict[str, dict[str, Any]]:
    fixture = _load(VALID_PATH)
    return {item["case_id"]: item["value"] for item in fixture["requests"]}


def _replace_pointer(value: dict[str, Any], pointer: str, replacement: Any) -> None:
    components = [
        part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/")[1:]
    ]
    target: Any = value
    for component in components[:-1]:
        target = (
            target[int(component)] if isinstance(target, list) else target[component]
        )
    leaf = components[-1]
    if isinstance(target, list):
        target[int(leaf)] = replacement
    else:
        target[leaf] = replacement


def _validate_preview(value: dict[str, Any], schema: dict[str, Any]) -> None:
    _validate_schema(value, schema, schema)
    if "control_job" not in value:
        return
    control = value["control_job"]
    retrieval = value["retrieval"]
    feedback = value["feedback"]
    digest = value["digest"]
    if retrieval["evidence_count"] != retrieval["reference_count"]:
        raise ContractValidationError("evidence and Reference counts differ")
    if control["feedback_count"] != len(feedback):
        raise ContractValidationError("Feedback count differs from ordered manifest")
    if [item["ordinal"] for item in feedback] != list(range(1, len(feedback) + 1)):
        raise ContractValidationError("Feedback ordinals are not contiguous")
    if len({item["feedback_id"] for item in feedback}) != len(feedback):
        raise ContractValidationError("Feedback identities are not unique")
    if (
        value["provenance"]["index"]["publication_id"]
        != value["provenance"]["index"]["index_id"]
    ):
        raise ContractValidationError("index publication identity mismatch")
    if control["feedback_count"] == 0:
        if feedback or digest is not None or control["notification_outbox_count"]:
            raise ContractValidationError("zero-finding effects are contradictory")
    elif (
        digest is None
        or digest["finding_count"] != len(feedback)
        or control["notification_outbox_count"] != 1
    ):
        raise ContractValidationError("positive-finding digest is contradictory")


def test_frozen_preview_artifacts_and_existing_protocols() -> None:
    assert _sha256(SPEC_PATH) == PREVIEW_SPEC_SHA256
    assert _sha256(SCHEMA_PATH) == PREVIEW_SCHEMA_SHA256
    assert _sha256(VALID_PATH) == PREVIEW_VALID_FIXTURE_SHA256
    assert _sha256(INVALID_PATH) == PREVIEW_INVALID_FIXTURE_SHA256
    assert _sha256(PROTOCOL_DIR / "baseline-control-plane.v1.md") == (
        CONTROL_V1_SPEC_SHA256
    )
    assert _sha256(PROTOCOL_DIR / "baseline-control-plane.v1.schema.json") == (
        CONTROL_V1_SCHEMA_SHA256
    )
    assert _sha256(PROTOCOL_DIR / "baseline-control-plane.v2.md") == (
        CONTROL_V2_SPEC_SHA256
    )
    assert _sha256(PROTOCOL_DIR / "baseline-control-plane.v2.schema.json") == (
        CONTROL_V2_SCHEMA_SHA256
    )
    assert _sha256(PROTOCOL_DIR / "baseline-generation-output.v2.md") == (
        GENERATION_SPEC_SHA256
    )
    assert _sha256(PROTOCOL_DIR / "baseline-generation-output.v2.schema.json") == (
        GENERATION_SCHEMA_SHA256
    )


@pytest.mark.parametrize("path", [SPEC_PATH, SCHEMA_PATH, VALID_PATH, INVALID_PATH])
def test_core_and_cli_preview_artifacts_are_byte_identical(path: Path) -> None:
    cli_path = CLI_PROTOCOL_DIR / path.relative_to(PROTOCOL_DIR)
    assert cli_path.read_bytes() == path.read_bytes()


def test_valid_preview_requests_and_responses() -> None:
    schema = _load(SCHEMA_PATH)
    requests = _requests()
    responses = _responses()
    assert set(requests) == {"request_by_job", "request_by_digest"}
    assert set(responses) == {
        "zero_findings_control_document",
        "positive_findings_control_document",
        "positive_findings_legacy_chunk",
    }
    for message in (*requests.values(), *responses.values()):
        _validate_preview(message, schema)


def test_zero_positive_and_source_scope_semantics_are_frozen() -> None:
    responses = _responses()
    zero = responses["zero_findings_control_document"]
    assert zero["control_job"]["feedback_count"] == 0
    assert zero["control_job"]["notification_outbox_count"] == 0
    assert zero["feedback"] == []
    assert zero["digest"] is None
    assert zero["source"]["source_scope"] == "control_document"
    assert zero["source"]["chunk_id"] is None

    positive = responses["positive_findings_control_document"]
    assert [item["ordinal"] for item in positive["feedback"]] == [1, 2]
    assert positive["digest"]["finding_count"] == 2
    assert positive["source"]["chunk_id"] is None

    legacy = responses["positive_findings_legacy_chunk"]
    assert legacy["source"]["source_scope"] == "legacy_chunk"
    assert legacy["source"]["chunk_id"] is not None


def test_invalid_preview_fixture_mutations_and_raw_json_are_rejected() -> None:
    schema = _load(SCHEMA_PATH)
    fixture = _load(INVALID_PATH)
    bases = _requests() | _responses()
    for case in fixture["cases"]:
        if "raw_json" in case:
            with pytest.raises((DuplicateJSONKeyError, ValueError)):
                _strict_loads(case["raw_json"])
            continue
        value = copy.deepcopy(bases[case["base_case"]])
        _replace_pointer(value, case["path"], case["value"])
        with pytest.raises(ContractValidationError):
            _validate_preview(value, schema)


def test_preview_safe_response_has_no_protected_fields() -> None:
    forbidden = {
        "content",
        "credentials",
        "encryption_metadata",
        "endpoint_url",
        "evidence",
        "idempotency_key",
        "lease_token",
        "prompt",
        "provider_request",
        "provider_response",
        "renderer_output",
        "repository_path",
        "retrieval_query",
    }

    def keys(value: Any):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    for response in _responses().values():
        assert forbidden.isdisjoint(set(keys(response)))


def test_preview_schema_has_no_retrieval_run_request_alias() -> None:
    schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
    spec_text = SPEC_PATH.read_text(encoding="utf-8")
    assert '"run_id"' not in schema_text
    assert "--run-id" not in spec_text
    assert "GET /baseline/preview/v1" not in spec_text
