from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from compair_core.baseline_generation.ollama import (
    OLLAMA_GENERATION_ADAPTER_CONTRACT,
    OllamaBaselineGenerationProvider,
    OllamaGenerationIdentity,
)
from compair_core.baseline_generation.profile import (
    QUALIFIED_BUDGET_PROFILE_FINGERPRINT,
    RECOMMENDED_GENERATION_MODEL,
    RECOMMENDED_GENERATION_MODEL_DIGEST,
)
from compair_core.compair.retrieval.generation import (
    GENERATION_OUTPUT_SCHEMA_SHA256,
    GENERATION_OUTPUT_SCHEMA_VERSION,
    GENERATION_OUTPUT_SPEC_SHA256,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "tests/evals/baseline_generation_qualification_v1" / "runner.py"
SPEC = importlib.util.spec_from_file_location("qualification_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _settings(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "baseline_generation_provider": "ollama",
        "baseline_generation_endpoint": "https://generation.example.test",
        "baseline_generation_model": RECOMMENDED_GENERATION_MODEL,
        "baseline_generation_model_digest": RECOMMENDED_GENERATION_MODEL_DIGEST,
        "baseline_generation_timeout_seconds": 60.0,
        "baseline_generation_allow_loopback_http": False,
        "baseline_generation_max_request_bytes": 256_000,
        "baseline_generation_max_response_bytes": 200_000,
        "baseline_generation_context_tokens": 32_768,
        "baseline_generation_output_tokens": 1_024,
        "baseline_generation_seed": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _output(outcome: str = "no_findings", feedback: str = "") -> str:
    findings = [] if outcome == "no_findings" else [{"feedback": feedback}]
    return json.dumps(
        {
            "schema_version": GENERATION_OUTPUT_SCHEMA_VERSION,
            "outcome": outcome,
            "findings": findings,
        },
        separators=(",", ":"),
    )


class _StubProvider:
    def __init__(
        self,
        config,
        *,
        outputs: list[str] | None = None,
        interrupt_at: int | None = None,
        identity_suffix: str = "qualified",
    ) -> None:
        production = OllamaBaselineGenerationProvider(config)
        self._prepare_chat = production._prepare_chat
        self._budget_profile = production._budget_profile
        self._output_tokens = production._output_tokens
        self._context_tokens = production._context_tokens
        self.outputs = list(outputs or [])
        self.interrupt_at = interrupt_at
        self.generate_count = 0
        self.attest_count = 0
        fingerprint = __import__("hashlib").sha256(identity_suffix.encode()).hexdigest()
        self._identity = OllamaGenerationIdentity(
            provider="ollama",
            adapter_contract=OLLAMA_GENERATION_ADAPTER_CONTRACT,
            model=RECOMMENDED_GENERATION_MODEL,
            digest=RECOMMENDED_GENERATION_MODEL_DIGEST,
            runtime_version="0.32.14",
            output_schema_version=GENERATION_OUTPUT_SCHEMA_VERSION,
            output_spec_sha256=GENERATION_OUTPUT_SPEC_SHA256,
            output_schema_sha256=GENERATION_OUTPUT_SCHEMA_SHA256,
            budget_profile_fingerprint=QUALIFIED_BUDGET_PROFILE_FINGERPRINT,
            supports_idempotency=False,
            fingerprint=fingerprint,
        )

    def attest(self) -> OllamaGenerationIdentity:
        self.attest_count += 1
        return self._identity

    @property
    def identity(self) -> OllamaGenerationIdentity:
        return self._identity

    def generate(self, generation_input, *, idempotency_key: str) -> str:
        assert generation_input.source_text
        assert len(idempotency_key) == 64
        self.generate_count += 1
        if self.interrupt_at == self.generate_count:
            raise KeyboardInterrupt
        if self.outputs:
            return self.outputs.pop(0)
        return _output()


def _factory(
    instances: list[_StubProvider],
    *,
    outputs: list[str] | None = None,
    interrupt_at: int | None = None,
):
    def create(config):
        provider = _StubProvider(
            config,
            outputs=outputs,
            interrupt_at=interrupt_at,
        )
        instances.append(provider)
        return provider

    return create


def _fixture_cases() -> list[dict[str, object]]:
    fixture, _report = runner._load_fixture()
    return fixture["cases"]


def test_individual_case_records_only_privacy_safe_validated_result(
    tmp_path: Path,
) -> None:
    case = _fixture_cases()[0]
    feedback = "unique synthetic feedback that must never be persisted"
    instances: list[_StubProvider] = []
    summary = runner.run_qualification(
        tmp_path / "individual",
        selectors=(str(case["ordinal"]),),
        settings=_settings(),
        provider_factory=_factory(
            instances,
            outputs=[_output("findings", feedback)],
        ),
    )

    assert summary["completed_case_count"] == 1
    assert summary["actual_findings_count"] == 1
    assert instances[0].generate_count == 1
    artifact_bytes = b"".join(
        path.read_bytes() for path in (tmp_path / "individual").iterdir()
    )
    assert case["source_text"].encode() not in artifact_bytes
    for evidence in case["evidence_renderer_input"]:
        assert evidence.encode() not in artifact_bytes
    assert feedback.encode() not in artifact_bytes
    record = json.loads((tmp_path / "individual/cases.jsonl").read_text("utf-8"))
    assert record["input_tokens"] > 0
    assert (
        record["input_tokens"] + record["reserved_output_tokens"]
        <= record["context_tokens"]
    )
    assert record["provider_identity_fingerprint"] == instances[0].identity.fingerprint


def test_resume_rejects_configuration_mismatch_before_provider_call(
    tmp_path: Path,
) -> None:
    case = _fixture_cases()[0]
    output_dir = tmp_path / "configuration"
    runner.run_qualification(
        output_dir,
        selectors=(str(case["ordinal"]),),
        settings=_settings(),
        provider_factory=_factory([]),
    )
    resumed_instances: list[_StubProvider] = []

    with pytest.raises(
        runner.QualificationRunnerError, match="run_configuration_mismatch"
    ):
        runner.run_qualification(
            output_dir,
            selectors=(str(case["ordinal"]),),
            resume=True,
            settings=_settings(baseline_generation_seed=1),
            provider_factory=_factory(resumed_instances),
        )
    assert resumed_instances == []


def test_resume_integrity_rejects_tampered_result_before_provider_call(
    tmp_path: Path,
) -> None:
    cases = _fixture_cases()[:2]
    output_dir = tmp_path / "integrity"
    with pytest.raises(KeyboardInterrupt):
        runner.run_qualification(
            output_dir,
            selectors=tuple(str(case["ordinal"]) for case in cases),
            settings=_settings(),
            provider_factory=_factory([], interrupt_at=2),
        )
    result_path = output_dir / "cases.jsonl"
    record = json.loads(result_path.read_text("utf-8"))
    record["latency_ms"] += 1
    result_path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    resumed_instances: list[_StubProvider] = []

    with pytest.raises(
        runner.QualificationRunnerError, match="case_results_integrity_mismatch"
    ):
        runner.run_qualification(
            output_dir,
            selectors=tuple(str(case["ordinal"]) for case in cases),
            resume=True,
            settings=_settings(),
            provider_factory=_factory(resumed_instances),
        )
    assert resumed_instances == []


@pytest.mark.parametrize(
    "malformed",
    [
        "not JSON",
        _output("findings", ""),
        json.dumps(
            {
                "schema_version": GENERATION_OUTPUT_SCHEMA_VERSION,
                "outcome": "no_findings",
                "findings": [{"feedback": "contradiction"}],
            }
        ),
    ],
)
def test_invalid_structured_result_is_not_checkpointed(
    tmp_path: Path, malformed: str
) -> None:
    case = _fixture_cases()[0]
    output_dir = tmp_path / __import__("hashlib").sha256(malformed.encode()).hexdigest()
    with pytest.raises(
        runner.QualificationRunnerError, match="result_validation_failed"
    ):
        runner.run_qualification(
            output_dir,
            selectors=(str(case["ordinal"]),),
            settings=_settings(),
            provider_factory=_factory([], outputs=[malformed]),
        )
    assert (output_dir / "run.json").is_file()
    assert not (output_dir / "cases.jsonl").exists()
    assert not (output_dir / "summary.json").exists()


def test_interrupted_full_execution_resumes_only_unfinished_cases(
    tmp_path: Path,
) -> None:
    cases = _fixture_cases()
    output_dir = tmp_path / "interrupted"
    first_instances: list[_StubProvider] = []
    with pytest.raises(KeyboardInterrupt):
        runner.run_qualification(
            output_dir,
            settings=_settings(),
            provider_factory=_factory(first_instances, interrupt_at=2),
        )
    assert first_instances[0].generate_count == 2
    assert len((output_dir / "cases.jsonl").read_text("utf-8").splitlines()) == 1
    assert not (output_dir / "summary.json").exists()

    resumed_instances: list[_StubProvider] = []
    summary = runner.run_qualification(
        output_dir,
        resume=True,
        settings=_settings(),
        provider_factory=_factory(resumed_instances),
    )
    assert resumed_instances[0].generate_count == len(cases) - 1
    assert summary["selected_case_count"] == len(cases)
    assert summary["completed_case_count"] == len(cases)
    assert len((output_dir / "cases.jsonl").read_text("utf-8").splitlines()) == len(
        cases
    )

    completed_instances: list[_StubProvider] = []
    replay = runner.run_qualification(
        output_dir,
        resume=True,
        settings=_settings(),
        provider_factory=_factory(completed_instances),
    )
    assert replay == summary
    assert completed_instances == []
