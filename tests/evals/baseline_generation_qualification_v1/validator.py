"""Inference-free validator for the frozen 120-case qualification examination."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parent
FIXTURE_PATH = EVAL_ROOT / "baseline-generation-qualification-examination.v1.json"
FIXTURE_HASH_PATH = (
    EVAL_ROOT / "baseline-generation-qualification-examination.v1.sha256"
)
AUDIT_PATH = EVAL_ROOT / "semantic-audit.v1.json"
AUDIT_HASH_PATH = EVAL_ROOT / "semantic-audit.v1.sha256"

FIXTURE_SHA256 = "2f1d8d204de06173fbfbe7fabf00aeb5771ef9869c09cbd959b2e7b4789d5863"
AUDIT_SHA256 = "6c6778ed3e007caaa4f7d76c2efa6b37863938dcb5f43170e8230369b2eb1167"
ANCHOR_SHA256 = "886ce0e93ac0749ade3bb109e736e3ffc0a08d0893c23fc5a83430bb0b700f2a"
ANCHOR_BYTE_LENGTH = 6366
GENERATION_SCHEMA_SHA256 = (
    "fc5a85d5d38c18775afe0966987ea74e7e9ac072148822c1be60a199e32cca27"
)
GENERATION_SPECIFICATION_SHA256 = (
    "e670731777b253f9d5e3984405c2d99871ba26f637a17e6221cc82d97bc8beb1"
)

SURFACES = (
    "http_api",
    "authentication_authorization",
    "cli_flags_defaults",
    "environment_configuration",
    "database_migrations_schemas",
    "webhooks_signatures",
    "sdk_function_behavior",
    "versioning_compatibility",
    "deployment_networking_tls",
    "serialization_nulls",
    "semantic_refactors",
    "documentation_lifecycle",
)
OUTCOMES = ("no_findings", "findings")
CASE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
FILESYSTEM_PATH_RE = re.compile(
    r"(?:^|[\s'\"`])(?:/Users/|/home/|/private/|/tmp/|/var/|[A-Za-z]:[\\/])"
)
SECRET_RE = re.compile(
    r"(?:-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"\b(?:sk|ghp|github_pat)-[A-Za-z0-9_-]{12,}|\bAKIA[A-Z0-9]{16}\b|"
    r"\bpassword\s*[:=]\s*['\"][^<{$][^'\"]{5,}['\"])",
    re.IGNORECASE,
)
PROVIDER_ARTIFACT_RE = re.compile(
    r"\b(?:model|provider|ollama|assistant)\s+(?:response|output)\b",
    re.IGNORECASE,
)
VISIBLE_LABELS = (
    "hard_no_finding",
    "obvious_finding",
    "expected_outcome",
    "no_findings",
    "should return findings",
    "should return no findings",
)


class ValidationError(ValueError):
    """The frozen eval artifact violates a structural or semantic invariant."""


def _reject_constant(value: str) -> Any:
    raise ValidationError(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def strict_json_loads(raw: bytes) -> Any:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValidationError("UTF-8 BOM is forbidden")
    if b"\r" in raw:
        raise ValidationError("only LF line endings are permitted")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValidationError("artifact must end with exactly one LF")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValidationError("artifact is not strict UTF-8") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValidationError(f"invalid JSON: {exc.msg}") from exc


def canonical_case_bytes(case: dict[str, Any]) -> bytes:
    payload = {
        key: case[key]
        for key in (
            "ordinal",
            "case_id",
            "surface",
            "expected_outcome",
            "source_text",
            "evidence_renderer_input",
        )
    }
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def case_sha256(case: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_case_bytes(case)).hexdigest()


def _assert_exact_keys(value: dict[str, Any], expected: set[str], where: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValidationError(
            f"{where} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _provider_visible_text(case: dict[str, Any]) -> str:
    return "\n".join([case["source_text"], *case["evidence_renderer_input"]])


def _validate_privacy(case: dict[str, Any]) -> None:
    visible = _provider_visible_text(case)
    if FILESYSTEM_PATH_RE.search(visible):
        raise ValidationError(f"filesystem absolute path in {case['case_id']}")
    if SECRET_RE.search(visible):
        raise ValidationError(f"credential or secret material in {case['case_id']}")
    if PROVIDER_ARTIFACT_RE.search(visible):
        raise ValidationError(f"provider response/output material in {case['case_id']}")
    if ".compair" in visible or "sqlite:///" in visible or "postgresql://" in visible:
        raise ValidationError(f"local runtime state in {case['case_id']}")
    lowered = visible.casefold()
    for label in VISIBLE_LABELS:
        if label in lowered:
            raise ValidationError(
                f"expected label leaked into provider-visible fields of {case['case_id']}"
            )


def _validate_anchor(
    fixture: dict[str, Any], cases_by_id: dict[str, dict[str, Any]]
) -> None:
    anchor = fixture["anchor_fixture"]
    _assert_exact_keys(
        anchor,
        {
            "schema_version",
            "encoding",
            "byte_length",
            "sha256",
            "hashing",
            "payload_base64",
        },
        "anchor_fixture",
    )
    if anchor["schema_version"] != "baseline-generation-qualification.v1":
        raise ValidationError("anchor schema version drifted")
    if anchor["encoding"] != "base64":
        raise ValidationError("anchor encoding drifted")
    try:
        raw = base64.b64decode(anchor["payload_base64"], validate=True)
    except Exception as exc:
        raise ValidationError("anchor payload is not valid base64") from exc
    if len(raw) != ANCHOR_BYTE_LENGTH or anchor["byte_length"] != ANCHOR_BYTE_LENGTH:
        raise ValidationError("anchor byte length drifted")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != ANCHOR_SHA256 or anchor["sha256"] != ANCHOR_SHA256:
        raise ValidationError("anchor raw-byte digest drifted")
    original = strict_json_loads(raw)
    if original.get("schema_version") != "baseline-generation-qualification.v1":
        raise ValidationError("decoded anchor schema version drifted")
    original_cases = original.get("cases")
    if not isinstance(original_cases, list) or len(original_cases) != 16:
        raise ValidationError("decoded anchor must contain exactly 16 cases")
    for original_case in original_cases:
        case_id = original_case.get("case_id")
        current = cases_by_id.get(case_id)
        if current is None or current.get("anchor_case_id") != case_id:
            raise ValidationError(f"anchor case missing from examination: {case_id}")
        if current["expected_outcome"] != original_case.get("expected_outcome"):
            raise ValidationError(f"anchor outcome drifted: {case_id}")
        if current["source_text"] != original_case.get("source_text"):
            raise ValidationError(f"anchor source bytes drifted: {case_id}")
        if current["evidence_renderer_input"] != original_case.get("evidence"):
            raise ValidationError(f"anchor evidence bytes drifted: {case_id}")


def validate_fixture_bytes(
    raw: bytes,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fixture = strict_json_loads(raw)
    if not isinstance(fixture, dict):
        raise ValidationError("fixture root must be an object")
    if (
        expected_sha256 is not None
        and hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise ValidationError("whole-fixture direct-byte SHA-256 drifted")
    _assert_exact_keys(
        fixture,
        {
            "schema_version",
            "frozen_before_inference",
            "generation_output_schema_version",
            "generation_output_schema_sha256",
            "generation_output_specification_sha256",
            "ordering",
            "case_hashing",
            "anchor_fixture",
            "surfaces",
            "cases",
        },
        "fixture",
    )
    if fixture["schema_version"] != "baseline-generation-qualification-examination.v1":
        raise ValidationError("fixture schema version drifted")
    if fixture["frozen_before_inference"] is not True:
        raise ValidationError("fixture is not marked frozen before inference")
    if fixture["generation_output_schema_version"] != "baseline-generation-output.v2":
        raise ValidationError("generation output schema version drifted")
    if fixture["generation_output_schema_sha256"] != GENERATION_SCHEMA_SHA256:
        raise ValidationError("generation schema hash drifted")
    if (
        fixture["generation_output_specification_sha256"]
        != GENERATION_SPECIFICATION_SHA256
    ):
        raise ValidationError("generation specification hash drifted")
    if fixture["ordering"] != (
        "cases are evaluated in ascending ordinal order; array order is authoritative"
    ):
        raise ValidationError("ordering contract drifted")

    surface_records = fixture["surfaces"]
    if not isinstance(surface_records, list):
        raise ValidationError("surfaces must be an array")
    surface_ids = [
        item.get("surface") for item in surface_records if isinstance(item, dict)
    ]
    if surface_ids != list(SURFACES) or len(surface_records) != len(SURFACES):
        raise ValidationError("surface inventory or ordering drifted")

    cases = fixture["cases"]
    if not isinstance(cases, list) or len(cases) != 120:
        raise ValidationError("fixture must contain exactly 120 cases")
    seen_ids: set[str] = set()
    cases_by_id: dict[str, dict[str, Any]] = {}
    surface_outcomes: Counter[tuple[str, str]] = Counter()
    overall_outcomes: Counter[str] = Counter()
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValidationError(f"case {index} must be an object")
        required = {
            "ordinal",
            "case_id",
            "surface",
            "expected_outcome",
            "source_text",
            "evidence_renderer_input",
            "case_sha256",
        }
        allowed = required | {"anchor_case_id"}
        if not required.issubset(case) or not set(case).issubset(allowed):
            raise ValidationError(f"case {index} fields drifted")
        if case["ordinal"] != index:
            raise ValidationError(f"ordinal/array-order mismatch at case {index}")
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not CASE_ID_RE.fullmatch(case_id):
            raise ValidationError(f"invalid stable case ID at ordinal {index}")
        if case_id in seen_ids:
            raise ValidationError(f"duplicate case ID: {case_id}")
        seen_ids.add(case_id)
        cases_by_id[case_id] = case
        if case["surface"] not in SURFACES:
            raise ValidationError(f"unknown surface in {case_id}")
        if case["expected_outcome"] not in OUTCOMES:
            raise ValidationError(f"unknown expected outcome in {case_id}")
        source = case["source_text"]
        evidence = case["evidence_renderer_input"]
        if not isinstance(source, str) or not source.strip():
            raise ValidationError(f"blank source in {case_id}")
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 4:
            raise ValidationError(f"evidence count outside 1..4 in {case_id}")
        if any(not isinstance(item, str) or not item.strip() for item in evidence):
            raise ValidationError(f"blank evidence item in {case_id}")
        if len(source) + sum(len(item) for item in evidence) > 16_000:
            raise ValidationError(
                f"case exceeds the evidence/source character bound: {case_id}"
            )
        expected_case_hash = case_sha256(case)
        if case["case_sha256"] != expected_case_hash:
            raise ValidationError(f"per-case hash drifted: {case_id}")
        _validate_privacy(case)
        surface_outcomes[(case["surface"], case["expected_outcome"])] += 1
        overall_outcomes[case["expected_outcome"]] += 1

    if overall_outcomes != Counter({"no_findings": 60, "findings": 60}):
        raise ValidationError(f"outcome balance drifted: {dict(overall_outcomes)}")
    for surface in SURFACES:
        if surface_outcomes[(surface, "no_findings")] != 5:
            raise ValidationError(f"no_findings balance drifted for {surface}")
        if surface_outcomes[(surface, "findings")] != 5:
            raise ValidationError(f"findings balance drifted for {surface}")
    _validate_anchor(fixture, cases_by_id)
    report = {
        "schema_version": fixture["schema_version"],
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "anchor_sha256": ANCHOR_SHA256,
        "case_count": len(cases),
        "no_findings_count": overall_outcomes["no_findings"],
        "findings_count": overall_outcomes["findings"],
        "surface_count": len(SURFACES),
    }
    return fixture, report


def validate_audit_bytes(
    raw: bytes,
    *,
    fixture: dict[str, Any],
    fixture_sha256: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    audit = strict_json_loads(raw)
    if not isinstance(audit, dict):
        raise ValidationError("semantic audit root must be an object")
    if (
        expected_sha256 is not None
        and hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise ValidationError("semantic-audit direct-byte SHA-256 drifted")
    _assert_exact_keys(
        audit,
        {
            "schema_version",
            "provider_visible",
            "method",
            "case_count",
            "cases",
            "fixture_sha256",
        },
        "semantic audit",
    )
    if audit["schema_version"] != "baseline-generation-qualification-semantic-audit.v1":
        raise ValidationError("semantic audit schema version drifted")
    if audit["provider_visible"] is not False:
        raise ValidationError("semantic audit must never be provider-visible")
    if audit["fixture_sha256"] != fixture_sha256:
        raise ValidationError("semantic audit is bound to a different fixture")
    audit_cases = audit["cases"]
    if not isinstance(audit_cases, list) or len(audit_cases) != 120:
        raise ValidationError("semantic audit must cover exactly 120 cases")
    if audit["case_count"] != 120:
        raise ValidationError("semantic audit case_count drifted")
    for fixture_case, audit_case in zip(fixture["cases"], audit_cases, strict=True):
        _assert_exact_keys(
            audit_case,
            {
                "ordinal",
                "case_id",
                "case_sha256",
                "material_issue_count",
                "objective",
                "self_contained",
                "audit_basis",
            },
            f"semantic audit case {fixture_case['case_id']}",
        )
        if audit_case["ordinal"] != fixture_case["ordinal"]:
            raise ValidationError("semantic audit ordinal drifted")
        if audit_case["case_id"] != fixture_case["case_id"]:
            raise ValidationError("semantic audit case ordering drifted")
        if audit_case["case_sha256"] != fixture_case["case_sha256"]:
            raise ValidationError(
                f"semantic audit case hash drifted: {fixture_case['case_id']}"
            )
        expected_issues = 1 if fixture_case["expected_outcome"] == "findings" else 0
        if audit_case["material_issue_count"] != expected_issues:
            raise ValidationError(
                f"semantic issue count contradicts outcome: {fixture_case['case_id']}"
            )
        if (
            audit_case["objective"] is not True
            or audit_case["self_contained"] is not True
        ):
            raise ValidationError(
                f"ambiguous semantic audit: {fixture_case['case_id']}"
            )
        basis = audit_case["audit_basis"]
        if not isinstance(basis, str) or len(basis.strip()) < 20:
            raise ValidationError(
                f"missing semantic justification: {fixture_case['case_id']}"
            )
    return {
        "semantic_audit_sha256": hashlib.sha256(raw).hexdigest(),
        "audited_case_count": 120,
        "ambiguous_case_count": 0,
    }


def _read_hash_sidecar(path: Path, expected_name: str) -> str:
    parts = path.read_text("ascii").strip().split()
    if (
        len(parts) != 2
        or parts[1] != expected_name
        or not re.fullmatch(r"[0-9a-f]{64}", parts[0])
    ):
        raise ValidationError(f"malformed hash sidecar: {path.name}")
    return parts[0]


def validate_frozen_artifacts(root: Path = EVAL_ROOT) -> dict[str, Any]:
    fixture_path = root / FIXTURE_PATH.name
    fixture_hash = _read_hash_sidecar(root / FIXTURE_HASH_PATH.name, fixture_path.name)
    if fixture_hash != FIXTURE_SHA256:
        raise ValidationError("fixture sidecar does not contain the frozen hash")
    fixture, report = validate_fixture_bytes(
        fixture_path.read_bytes(), expected_sha256=FIXTURE_SHA256
    )
    audit_path = root / AUDIT_PATH.name
    audit_hash = _read_hash_sidecar(root / AUDIT_HASH_PATH.name, audit_path.name)
    if audit_hash != AUDIT_SHA256:
        raise ValidationError("semantic-audit sidecar does not contain the frozen hash")
    report.update(
        validate_audit_bytes(
            audit_path.read_bytes(),
            fixture=fixture,
            fixture_sha256=FIXTURE_SHA256,
            expected_sha256=AUDIT_SHA256,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=EVAL_ROOT)
    args = parser.parse_args()
    report = validate_frozen_artifacts(args.root)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
