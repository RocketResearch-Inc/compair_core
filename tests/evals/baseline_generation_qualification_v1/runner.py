"""Evaluation-only runner for the frozen baseline generation examination."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from compair_core.baseline_evidence_schema import (
    RENDERER_VERSION,
    SOURCE_SCOPE_CONTROL_DOCUMENT,
    SOURCE_SCOPE_VERSION,
)
from compair_core.baseline_generation.ollama import (
    OLLAMA_GENERATION_ADAPTER_CONTRACT,
    OllamaBaselineGenerationProvider,
    OllamaGenerationConfig,
)
from compair_core.baseline_generation.profile import (
    QUALIFIED_BUDGET_PROFILE_FINGERPRINT,
    QUALIFIED_OLLAMA_RUNTIME_VERSION,
    RECOMMENDED_GENERATION_MODEL,
    RECOMMENDED_GENERATION_MODEL_DIGEST,
)
from compair_core.compair.retrieval.generation import (
    GENERATION_OUTPUT_SCHEMA_SHA256,
    GENERATION_OUTPUT_SCHEMA_VERSION,
    GENERATION_OUTPUT_SPEC_SHA256,
    BaselineGenerationError,
    BaselineGenerationEvidence,
    BaselineGenerationInput,
    BaselineGenerationProviderError,
    BaselineGenerationService,
)
from compair_core.server.settings import Settings

EVAL_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EVAL_ROOT.parents[2]
RUN_SCHEMA_VERSION = "baseline-generation-qualification-run.v1"
CASE_RESULT_SCHEMA_VERSION = "baseline-generation-qualification-case-result.v1"
SUMMARY_SCHEMA_VERSION = "baseline-generation-qualification-summary.v1"
ZERO_SHA256 = "0" * 64


def _load_validator() -> Any:
    path = EVAL_ROOT / "validator.py"
    spec = importlib.util.spec_from_file_location(
        "baseline_generation_qualification_validator", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("qualification validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


class QualificationRunnerError(RuntimeError):
    """Privacy-safe runner failure with a stable command exit code."""

    def __init__(self, code: str, *, exit_code: int) -> None:
        super().__init__(code)
        self.code = code
        self.exit_code = exit_code


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _strict_object(raw: bytes, *, label: str) -> dict[str, Any]:
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or b"\r" in raw:
        raise QualificationRunnerError(f"{label}_integrity_mismatch", exit_code=4)

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise QualificationRunnerError(
            f"{label}_integrity_mismatch", exit_code=4
        ) from None
    if not isinstance(value, dict):
        raise QualificationRunnerError(f"{label}_integrity_mismatch", exit_code=4)
    return value


def _atomic_write(path: Path, raw: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


def _outside_repository(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved == REPOSITORY_ROOT or REPOSITORY_ROOT in resolved.parents:
        raise QualificationRunnerError("output_inside_repository", exit_code=2)
    return resolved


def _load_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        report = validator.validate_frozen_artifacts(EVAL_ROOT)
        fixture, _fixture_report = validator.validate_fixture_bytes(
            validator.FIXTURE_PATH.read_bytes(),
            expected_sha256=validator.FIXTURE_SHA256,
        )
    except validator.ValidationError:
        raise QualificationRunnerError("frozen_fixture_invalid", exit_code=4) from None
    return fixture, report


def _select_cases(
    fixture: dict[str, Any], selectors: Sequence[str]
) -> tuple[str, list[dict[str, Any]]]:
    cases = fixture["cases"]
    if not selectors:
        return "full", list(cases)
    by_id = {case["case_id"]: case for case in cases}
    selected: dict[int, dict[str, Any]] = {}
    for selector in selectors:
        case: dict[str, Any] | None = None
        if selector.isdecimal():
            ordinal = int(selector)
            if 1 <= ordinal <= len(cases):
                case = cases[ordinal - 1]
        else:
            case = by_id.get(selector)
        if case is None:
            raise QualificationRunnerError("case_selection_invalid", exit_code=2)
        ordinal = int(case["ordinal"])
        if ordinal in selected:
            raise QualificationRunnerError("case_selection_duplicate", exit_code=2)
        selected[ordinal] = case
    return "selected", [selected[key] for key in sorted(selected)]


def _configuration(config: OllamaGenerationConfig) -> dict[str, Any]:
    endpoint = config.endpoint or ""
    endpoint_scheme = urlsplit(endpoint).scheme
    values: dict[str, Any] = {
        "adapter_contract": OLLAMA_GENERATION_ADAPTER_CONTRACT,
        "provider": config.provider_mode,
        "model": config.model,
        "expected_digest": config.expected_digest,
        "endpoint_scheme": endpoint_scheme,
        "endpoint_sha256": _sha256(endpoint.encode("utf-8")),
        "allow_loopback_http": config.allow_loopback_http,
        "timeout_seconds": config.timeout_seconds,
        "maximum_request_bytes": config.maximum_request_bytes,
        "maximum_response_bytes": config.maximum_response_bytes,
        "context_tokens": config.context_tokens,
        "output_tokens": config.output_tokens,
        "seed": config.seed,
        "budget_profile_fingerprint": QUALIFIED_BUDGET_PROFILE_FINGERPRINT,
        "output_schema_version": GENERATION_OUTPUT_SCHEMA_VERSION,
        "output_schema_sha256": GENERATION_OUTPUT_SCHEMA_SHA256,
        "output_spec_sha256": GENERATION_OUTPUT_SPEC_SHA256,
    }
    return {**values, "fingerprint": _sha256(_canonical(values))}


def _selection(mode: str, cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    entries = [
        {
            "ordinal": case["ordinal"],
            "case_id": case["case_id"],
            "case_sha256": case["case_sha256"],
        }
        for case in cases
    ]
    values = {"mode": mode, "case_count": len(entries), "cases": entries}
    return {**values, "fingerprint": _sha256(_canonical(values))}


def _provider_identity(identity: object) -> dict[str, Any]:
    fields = (
        "provider",
        "adapter_contract",
        "model",
        "digest",
        "runtime_version",
        "output_schema_version",
        "output_spec_sha256",
        "output_schema_sha256",
        "budget_profile_fingerprint",
        "supports_idempotency",
        "fingerprint",
    )
    values = {field: getattr(identity, field, None) for field in fields}
    if (
        values["provider"] != "ollama"
        or values["adapter_contract"] != OLLAMA_GENERATION_ADAPTER_CONTRACT
        or values["model"] != RECOMMENDED_GENERATION_MODEL
        or values["digest"] != RECOMMENDED_GENERATION_MODEL_DIGEST
        or values["runtime_version"] != QUALIFIED_OLLAMA_RUNTIME_VERSION
        or values["output_schema_version"] != GENERATION_OUTPUT_SCHEMA_VERSION
        or values["output_schema_sha256"] != GENERATION_OUTPUT_SCHEMA_SHA256
        or values["output_spec_sha256"] != GENERATION_OUTPUT_SPEC_SHA256
        or values["budget_profile_fingerprint"] != QUALIFIED_BUDGET_PROFILE_FINGERPRINT
        or values["supports_idempotency"] is not False
        or not isinstance(values["fingerprint"], str)
        or len(values["fingerprint"]) != 64
        or any(
            character not in "0123456789abcdef" for character in values["fingerprint"]
        )
    ):
        raise QualificationRunnerError("provider_identity_invalid", exit_code=3)
    return values


def _new_run(
    *,
    fixture_report: dict[str, Any],
    configuration: dict[str, Any],
    selection: dict[str, Any],
    identity: dict[str, Any],
    attestation_latency_ms: int,
) -> dict[str, Any]:
    values: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "fixture_sha256": fixture_report["fixture_sha256"],
        "semantic_audit_sha256": fixture_report["semantic_audit_sha256"],
        "configuration": configuration,
        "selection": selection,
        "provider_identity": identity,
        "attestation_latency_ms": attestation_latency_ms,
        "started_at": _timestamp(),
    }
    return {**values, "run_fingerprint": _sha256(_canonical(values))}


def _validate_run(
    run: dict[str, Any],
    *,
    fixture_report: dict[str, Any],
    configuration: dict[str, Any],
    selection: dict[str, Any],
) -> None:
    expected_keys = {
        "schema_version",
        "fixture_sha256",
        "semantic_audit_sha256",
        "configuration",
        "selection",
        "provider_identity",
        "attestation_latency_ms",
        "started_at",
        "run_fingerprint",
    }
    unsigned = {key: value for key, value in run.items() if key != "run_fingerprint"}
    if (
        set(run) != expected_keys
        or run["schema_version"] != RUN_SCHEMA_VERSION
        or run["fixture_sha256"] != fixture_report["fixture_sha256"]
        or run["semantic_audit_sha256"] != fixture_report["semantic_audit_sha256"]
        or run["configuration"] != configuration
        or run["selection"] != selection
        or run["run_fingerprint"] != _sha256(_canonical(unsigned))
        or not _is_int(run["attestation_latency_ms"])
        or run["attestation_latency_ms"] < 0
        or not isinstance(run["started_at"], str)
    ):
        raise QualificationRunnerError("run_configuration_mismatch", exit_code=4)
    _provider_identity(type("Identity", (), run["provider_identity"])())


def _generation_input(case: dict[str, Any]) -> BaselineGenerationInput:
    case_hash = case["case_sha256"]
    corpus_id = f"eval-corpus-{case_hash[:20]}"
    index_id = f"eval-index-{case_hash[:20]}"
    evidence: list[BaselineGenerationEvidence] = []
    for ordinal, renderer_output in enumerate(case["evidence_renderer_input"], start=1):
        renderer_hash = _sha256(renderer_output.encode("utf-8"))
        evidence.append(
            BaselineGenerationEvidence(
                ordinal=ordinal,
                fused_rank=ordinal,
                bm25_score=0.0,
                bm25_rank=ordinal,
                dense_score=0.0,
                dense_rank=ordinal,
                rrf_score=0.0,
                selected_evidence_id=f"eval-evidence-{renderer_hash[:20]}",
                artifact_id=f"eval-artifact-{renderer_hash[:20]}",
                repository_id=f"eval-repository-{ordinal}",
                repository_name="qualification",
                relative_path=f"evidence/{ordinal}.txt",
                renderer_version=RENDERER_VERSION,
                renderer_output=renderer_output,
                renderer_output_hash=renderer_hash,
                selected_content_hash=renderer_hash,
                whole_file_content_hash=renderer_hash,
                corpus_generation_id=corpus_id,
                index_id=index_id,
                index_document_id=f"eval-index-document-{renderer_hash[:20]}",
                index_fingerprint=case_hash,
            )
        )
    input_fingerprint = _sha256(
        _canonical(
            {
                "source_text": case["source_text"],
                "evidence_renderer_input": case["evidence_renderer_input"],
            }
        )
    )
    return BaselineGenerationInput(
        run_id=f"eval-run-{case_hash[:24]}",
        group_id=f"eval-group-{case_hash[:22]}",
        source_scope_version=SOURCE_SCOPE_VERSION,
        source_scope=SOURCE_SCOPE_CONTROL_DOCUMENT,
        source_chunk_id=None,
        source_document_id=f"eval-document-{case_hash[:19]}",
        source_text=case["source_text"],
        corpus_generation_id=corpus_id,
        corpus_manifest_hash=case_hash,
        index_id=index_id,
        index_fingerprint=case_hash,
        query_sha256=case_hash,
        evidence=tuple(evidence),
        input_fingerprint=input_fingerprint,
    )


def _budget_metrics(
    provider: OllamaBaselineGenerationProvider,
    generation_input: BaselineGenerationInput,
) -> dict[str, int]:
    try:
        body = provider._prepare_chat(
            source_text=generation_input.source_text,
            evidence=[item.renderer_output for item in generation_input.evidence],
            maximum_findings=len(generation_input.evidence),
        )
        payload = json.loads(body)
        messages = payload["messages"]
        input_tokens = provider._budget_profile.count(
            messages[0]["content"], messages[1]["content"]
        )
    except BaselineGenerationProviderError:
        raise
    except Exception:  # noqa: BLE001 - adapter-private contract must fail closed
        raise QualificationRunnerError("adapter_budget_contract_invalid", exit_code=4)
    return {
        "input_tokens": input_tokens,
        "reserved_output_tokens": provider._output_tokens,
        "context_tokens": provider._context_tokens,
        "request_body_bytes": len(body),
    }


def _case_record(
    *,
    case: dict[str, Any],
    metrics: dict[str, int],
    identity_fingerprint: str,
    output: str,
    latency_ms: int,
    previous_sha256: str,
) -> dict[str, Any]:
    try:
        findings, _output_fingerprint = BaselineGenerationService._parse_output(
            output,
            maximum_findings=len(case["evidence_renderer_input"]),
        )
    except BaselineGenerationError:
        raise QualificationRunnerError(
            "result_validation_failed", exit_code=4
        ) from None
    actual_outcome = "findings" if findings else "no_findings"
    values: dict[str, Any] = {
        "schema_version": CASE_RESULT_SCHEMA_VERSION,
        "ordinal": case["ordinal"],
        "case_id": case["case_id"],
        "case_sha256": case["case_sha256"],
        "surface": case["surface"],
        "expected_outcome": case["expected_outcome"],
        "actual_outcome": actual_outcome,
        "matches_expected": actual_outcome == case["expected_outcome"],
        "finding_count": len(findings),
        **metrics,
        "provider_identity_fingerprint": identity_fingerprint,
        "latency_ms": latency_ms,
        "previous_record_sha256": previous_sha256,
    }
    return {**values, "record_sha256": _sha256(_canonical(values))}


def _load_records(
    path: Path,
    *,
    cases: Sequence[dict[str, Any]],
    run: dict[str, Any],
) -> tuple[list[dict[str, Any]], bytes]:
    if not path.exists():
        return [], b""
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n") or b"\r" in raw:
        raise QualificationRunnerError("case_results_integrity_mismatch", exit_code=4)
    records: list[dict[str, Any]] = []
    previous = ZERO_SHA256
    expected_keys = {
        "schema_version",
        "ordinal",
        "case_id",
        "case_sha256",
        "surface",
        "expected_outcome",
        "actual_outcome",
        "matches_expected",
        "finding_count",
        "input_tokens",
        "reserved_output_tokens",
        "context_tokens",
        "request_body_bytes",
        "provider_identity_fingerprint",
        "latency_ms",
        "previous_record_sha256",
        "record_sha256",
    }
    try:
        lines = raw.splitlines()
        if len(lines) > len(cases):
            raise ValueError
        for case, line in zip(cases, lines, strict=False):
            record = _strict_object(line + b"\n", label="case_results")
            unsigned = {
                key: value for key, value in record.items() if key != "record_sha256"
            }
            integer_fields = (
                "ordinal",
                "finding_count",
                "input_tokens",
                "reserved_output_tokens",
                "context_tokens",
                "request_body_bytes",
                "latency_ms",
            )
            if (
                set(record) != expected_keys
                or record["schema_version"] != CASE_RESULT_SCHEMA_VERSION
                or record["ordinal"] != case["ordinal"]
                or record["case_id"] != case["case_id"]
                or record["case_sha256"] != case["case_sha256"]
                or record["surface"] != case["surface"]
                or record["expected_outcome"] != case["expected_outcome"]
                or record["actual_outcome"] not in {"no_findings", "findings"}
                or record["matches_expected"]
                is not (record["actual_outcome"] == case["expected_outcome"])
                or any(not _is_int(record[field]) for field in integer_fields)
                or record["finding_count"] < 0
                or record["input_tokens"] <= 0
                or record["reserved_output_tokens"]
                != run["configuration"]["output_tokens"]
                or record["context_tokens"] != run["configuration"]["context_tokens"]
                or record["input_tokens"] + record["reserved_output_tokens"]
                > record["context_tokens"]
                or not 0
                < record["request_body_bytes"]
                <= run["configuration"]["maximum_request_bytes"]
                or record["latency_ms"] < 0
                or record["provider_identity_fingerprint"]
                != run["provider_identity"]["fingerprint"]
                or record["previous_record_sha256"] != previous
                or record["record_sha256"] != _sha256(_canonical(unsigned))
            ):
                raise ValueError
            if (record["actual_outcome"] == "findings") is not (
                record["finding_count"] > 0
            ):
                raise ValueError
            previous = record["record_sha256"]
            records.append(record)
    except (QualificationRunnerError, TypeError, ValueError, KeyError):
        raise QualificationRunnerError(
            "case_results_integrity_mismatch", exit_code=4
        ) from None
    return records, raw


def _summary(
    run: dict[str, Any], records: Sequence[dict[str, Any]], raw: bytes
) -> dict[str, Any]:
    outcomes = Counter(record["actual_outcome"] for record in records)
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_fingerprint": run["run_fingerprint"],
        "fixture_sha256": run["fixture_sha256"],
        "configuration_fingerprint": run["configuration"]["fingerprint"],
        "provider_identity_fingerprint": run["provider_identity"]["fingerprint"],
        "selection_fingerprint": run["selection"]["fingerprint"],
        "selected_case_count": run["selection"]["case_count"],
        "completed_case_count": len(records),
        "matching_case_count": sum(record["matches_expected"] for record in records),
        "actual_no_findings_count": outcomes["no_findings"],
        "actual_findings_count": outcomes["findings"],
        "total_input_tokens": sum(record["input_tokens"] for record in records),
        "total_latency_ms": sum(record["latency_ms"] for record in records),
        "maximum_latency_ms": max(
            (record["latency_ms"] for record in records), default=0
        ),
        "case_results_sha256": _sha256(raw),
        "status": "complete",
    }


def _validate_summary(path: Path, expected: dict[str, Any]) -> None:
    if not path.exists():
        return
    actual = _strict_object(path.read_bytes(), label="summary")
    if actual != expected:
        raise QualificationRunnerError("summary_integrity_mismatch", exit_code=4)


ProviderFactory = Callable[[OllamaGenerationConfig], OllamaBaselineGenerationProvider]


def run_qualification(
    output_dir: Path,
    *,
    selectors: Sequence[str] = (),
    resume: bool = False,
    settings: Any | None = None,
    provider_factory: ProviderFactory = OllamaBaselineGenerationProvider,
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    fixture, fixture_report = _load_fixture()
    mode, cases = _select_cases(fixture, selectors)
    selection = _selection(mode, cases)
    try:
        config = OllamaGenerationConfig.from_settings(settings or Settings())
    except (BaselineGenerationProviderError, TypeError, ValueError):
        raise QualificationRunnerError("configuration_invalid", exit_code=3) from None
    configuration = _configuration(config)
    output_dir = _outside_repository(output_dir)
    run_path = output_dir / "run.json"
    cases_path = output_dir / "cases.jsonl"
    summary_path = output_dir / "summary.json"

    if resume:
        if not run_path.is_file():
            raise QualificationRunnerError("resume_state_missing", exit_code=4)
        run = _strict_object(run_path.read_bytes(), label="run")
        _validate_run(
            run,
            fixture_report=fixture_report,
            configuration=configuration,
            selection=selection,
        )
        records, records_raw = _load_records(cases_path, cases=cases, run=run)
        if len(records) == len(cases):
            expected_summary = _summary(run, records, records_raw)
            _validate_summary(summary_path, expected_summary)
            if not summary_path.exists():
                _atomic_write(summary_path, _canonical(expected_summary) + b"\n")
            return expected_summary
        if summary_path.exists():
            raise QualificationRunnerError("summary_integrity_mismatch", exit_code=4)
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise QualificationRunnerError("output_not_empty", exit_code=2)
        output_dir.mkdir(parents=True, exist_ok=True)
        run = {}
        records = []
        records_raw = b""

    provider = provider_factory(config)
    attestation_started = monotonic_ns()
    try:
        identity = _provider_identity(provider.attest())
    except QualificationRunnerError:
        raise
    except BaselineGenerationProviderError as exc:
        raise QualificationRunnerError(f"provider_{exc.code}", exit_code=3) from None
    except Exception:  # noqa: BLE001 - provider boundary must remain sanitized
        raise QualificationRunnerError(
            "provider_attestation_failed", exit_code=3
        ) from None
    attestation_latency_ms = max(0, (monotonic_ns() - attestation_started) // 1_000_000)

    if resume:
        if identity != run["provider_identity"]:
            raise QualificationRunnerError("provider_identity_mismatch", exit_code=3)
    else:
        run = _new_run(
            fixture_report=fixture_report,
            configuration=configuration,
            selection=selection,
            identity=identity,
            attestation_latency_ms=attestation_latency_ms,
        )
        _atomic_write(run_path, _canonical(run) + b"\n")

    previous_sha256 = records[-1]["record_sha256"] if records else ZERO_SHA256
    for case in cases[len(records) :]:
        generation_input = _generation_input(case)
        try:
            metrics = _budget_metrics(provider, generation_input)
            started = monotonic_ns()
            output = provider.generate(
                generation_input,
                idempotency_key=_sha256(
                    (
                        "qualification\x00"
                        + run["run_fingerprint"]
                        + "\x00"
                        + case["case_sha256"]
                    ).encode("utf-8")
                ),
            )
            latency_ms = max(0, (monotonic_ns() - started) // 1_000_000)
            if _provider_identity(provider.identity) != run["provider_identity"]:
                raise QualificationRunnerError(
                    "provider_identity_mismatch", exit_code=3
                )
            record = _case_record(
                case=case,
                metrics=metrics,
                identity_fingerprint=identity["fingerprint"],
                output=output,
                latency_ms=latency_ms,
                previous_sha256=previous_sha256,
            )
        except KeyboardInterrupt:
            raise
        except QualificationRunnerError:
            raise
        except BaselineGenerationProviderError as exc:
            raise QualificationRunnerError(
                f"provider_{exc.code}", exit_code=3
            ) from None
        except Exception:  # noqa: BLE001 - provider boundary must remain sanitized
            raise QualificationRunnerError(
                "provider_execution_failed", exit_code=3
            ) from None
        line = _canonical(record) + b"\n"
        records_raw += line
        _atomic_write(cases_path, records_raw)
        records.append(record)
        previous_sha256 = record["record_sha256"]
        if progress is not None:
            progress(len(records), len(cases))

    summary = _summary(run, records, records_raw)
    _atomic_write(summary_path, _canonical(summary) + b"\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="result directory outside the source checkout",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="case ID or one-based ordinal; repeat to select multiple cases",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an integrity-matched run directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_qualification(
            args.output_dir,
            selectors=args.case,
            resume=args.resume,
            progress=lambda completed, total: print(
                f"completed {completed}/{total}", file=sys.stderr, flush=True
            ),
        )
    except KeyboardInterrupt:
        result = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "interrupted",
            "error_code": "execution_interrupted",
        }
        exit_code = 130
    except QualificationRunnerError as exc:
        result = {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "status": "error",
            "error_code": exc.code,
        }
        exit_code = exc.exit_code
    else:
        exit_code = 0
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
