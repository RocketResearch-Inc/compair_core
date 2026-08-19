from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "tests/evals/baseline_generation_qualification_v1"
VALIDATOR_PATH = EVAL_ROOT / "validator.py"
SPEC = importlib.util.spec_from_file_location("qualification_validator", VALIDATOR_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def _fixture_bytes() -> bytes:
    return (
        EVAL_ROOT / "baseline-generation-qualification-examination.v1.json"
    ).read_bytes()


def _audit_bytes() -> bytes:
    return (EVAL_ROOT / "semantic-audit.v1.json").read_bytes()


def _serialized(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _strict_fixture() -> dict[str, object]:
    value = validator.strict_json_loads(_fixture_bytes())
    assert isinstance(value, dict)
    return value


def _rehash_case(case: dict[str, object]) -> None:
    case["case_sha256"] = validator.case_sha256(case)


def test_frozen_examination_and_semantic_audit_validate() -> None:
    report = validator.validate_frozen_artifacts()
    assert report == {
        "schema_version": "baseline-generation-qualification-examination.v1",
        "fixture_sha256": validator.FIXTURE_SHA256,
        "anchor_sha256": validator.ANCHOR_SHA256,
        "case_count": 120,
        "no_findings_count": 60,
        "findings_count": 60,
        "surface_count": 12,
        "semantic_audit_sha256": validator.AUDIT_SHA256,
        "audited_case_count": 120,
        "ambiguous_case_count": 0,
    }


def test_frozen_fixture_has_direct_byte_policy_and_original_anchor() -> None:
    raw = _fixture_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in raw
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert hashlib.sha256(raw).hexdigest() == validator.FIXTURE_SHA256
    fixture = validator.strict_json_loads(raw)
    anchor = base64.b64decode(
        fixture["anchor_fixture"]["payload_base64"], validate=True
    )
    assert len(anchor) == 6366
    assert hashlib.sha256(anchor).hexdigest() == validator.ANCHOR_SHA256
    assert anchor.endswith(b"\n") and not anchor.endswith(b"\n\n")


def test_invalid_duplicate_top_level_and_nested_keys_are_rejected() -> None:
    duplicate_top = b'{"cases":[],"cases":[]}\n'
    duplicate_nested = b'{"case":{"case_id":"a","case_id":"b"}}\n'
    with pytest.raises(validator.ValidationError, match="duplicate JSON object key"):
        validator.strict_json_loads(duplicate_top)
    with pytest.raises(validator.ValidationError, match="duplicate JSON object key"):
        validator.strict_json_loads(duplicate_nested)


@pytest.mark.parametrize("constant", [b"NaN", b"Infinity", b"-Infinity"])
def test_invalid_nonfinite_numbers_are_rejected(constant: bytes) -> None:
    with pytest.raises(validator.ValidationError, match="non-finite"):
        validator.strict_json_loads(b'{"value":' + constant + b"}\n")


def test_invalid_whole_fixture_hash_is_rejected_before_acceptance() -> None:
    raw = _fixture_bytes().replace(
        b"peer/contracts/widgets.md", b"peer/contracts/widgetz.md", 1
    )
    with pytest.raises(validator.ValidationError, match="whole-fixture"):
        validator.validate_fixture_bytes(raw, expected_sha256=validator.FIXTURE_SHA256)


def test_invalid_outcome_balance_is_rejected() -> None:
    fixture = _strict_fixture()
    case = fixture["cases"][16]
    assert case["expected_outcome"] == "no_findings"
    case["expected_outcome"] = "findings"
    _rehash_case(case)
    with pytest.raises(validator.ValidationError, match="outcome balance"):
        validator.validate_fixture_bytes(_serialized(fixture))


def test_invalid_ordinal_and_case_id_uniqueness_are_rejected() -> None:
    fixture = _strict_fixture()
    fixture["cases"][20]["ordinal"] = 20
    _rehash_case(fixture["cases"][20])
    with pytest.raises(validator.ValidationError, match="ordinal/array-order"):
        validator.validate_fixture_bytes(_serialized(fixture))

    fixture = _strict_fixture()
    fixture["cases"][20]["case_id"] = fixture["cases"][19]["case_id"]
    _rehash_case(fixture["cases"][20])
    with pytest.raises(validator.ValidationError, match="duplicate case ID"):
        validator.validate_fixture_bytes(_serialized(fixture))


def test_invalid_anchor_case_drift_is_rejected_even_with_new_case_hash() -> None:
    fixture = _strict_fixture()
    fixture["cases"][0]["source_text"] += "# drift\n"
    _rehash_case(fixture["cases"][0])
    with pytest.raises(validator.ValidationError, match="anchor source bytes drifted"):
        validator.validate_fixture_bytes(_serialized(fixture))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("Read /Users/example/private.txt", "filesystem absolute path"),
        ('password = "correct-horse-value"', "credential or secret"),
        ("The model response contains text", "provider response/output"),
        ("Expected outcome: no_findings", "expected label leaked"),
        ("Read sqlite:///local.db", "local runtime state"),
    ],
)
def test_invalid_provider_visible_privacy_material_is_rejected(
    text: str, message: str
) -> None:
    fixture = _strict_fixture()
    case = fixture["cases"][20]
    case["source_text"] = text
    _rehash_case(case)
    with pytest.raises(validator.ValidationError, match=message):
        validator.validate_fixture_bytes(_serialized(fixture))


def test_invalid_semantic_audit_issue_count_and_ambiguity_are_rejected() -> None:
    fixture, _ = validator.validate_fixture_bytes(_fixture_bytes())
    audit = validator.strict_json_loads(_audit_bytes())
    audit["cases"][0]["material_issue_count"] = 1
    with pytest.raises(validator.ValidationError, match="issue count contradicts"):
        validator.validate_audit_bytes(
            _serialized(audit),
            fixture=fixture,
            fixture_sha256=validator.FIXTURE_SHA256,
        )

    audit = validator.strict_json_loads(_audit_bytes())
    audit["cases"][0]["objective"] = False
    with pytest.raises(validator.ValidationError, match="ambiguous semantic audit"):
        validator.validate_audit_bytes(
            _serialized(audit),
            fixture=fixture,
            fixture_sha256=validator.FIXTURE_SHA256,
        )


@pytest.mark.parametrize(
    "raw",
    [
        b"\xef\xbb\xbf{}\n",
        b"{}\r\n",
        b"{}",
        b"{}\n\n",
        b'{"value":"\xff"}\n',
    ],
)
def test_invalid_byte_encodings_and_line_endings_are_rejected(raw: bytes) -> None:
    with pytest.raises(validator.ValidationError):
        validator.strict_json_loads(raw)
