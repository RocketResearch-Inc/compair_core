from __future__ import annotations

import json
import re
import threading
import time
import tracemalloc
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomllib
from sqlalchemy import text
from test_baseline_control_generation import _persist_control, _state
from test_baseline_generation import _environment

from compair_core.baseline_generation.cli import main as generation_cli
from compair_core.baseline_generation.ollama import (
    OLLAMA_GENERATION_ADAPTER_CONTRACT,
    OllamaBaselineGenerationProvider,
    OllamaGenerationConfig,
    verify_ollama_generation,
)
from compair_core.baseline_generation.profile import (
    ACCELERATED_GENERATION_TIMEOUT_SECONDS,
    RECOMMENDED_GENERATION_MODEL,
    RECOMMENDED_GENERATION_MODEL_DIGEST,
)
from compair_core.compair.retrieval.generation import (
    GENERATION_OUTPUT_SCHEMA_SHA256,
    GENERATION_OUTPUT_SPEC_SHA256,
    BaselineGenerationError,
    BaselineGenerationEvidence,
    BaselineGenerationInput,
    BaselineGenerationProviderError,
    BaselineGenerationService,
)
from compair_core.compair.retrieval.run_operator import (
    BaselineRunRuntimeError,
    _configured_generation_provider,
)
from compair_core.server.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
MODEL = RECOMMENDED_GENERATION_MODEL
DIGEST = RECOMMENDED_GENERATION_MODEL_DIGEST
DIGEST_HEX = DIGEST.removeprefix("sha256:")


def _structured(outcome: str, findings: list[str]) -> str:
    return json.dumps(
        {
            "schema_version": "baseline-generation-output.v2",
            "outcome": outcome,
            "findings": [{"feedback": value} for value in findings],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


class FakeOllamaState:
    def __init__(self) -> None:
        self.version = "0.32.13"
        self.model = MODEL
        self.digest = DIGEST_HEX
        self.chat_content = _structured("findings", ["Synthetic finding"])
        self.chat_status = 200
        self.chat_delay = 0.0
        self.chat_response_bytes: bytes | None = None
        self.requests: list[tuple[str, bytes, dict[str, str]]] = []


class _FakeOllamaServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, state: FakeOllamaState) -> None:
        super().__init__(("127.0.0.1", 0), _FakeOllamaHandler)
        self.state = state


class _FakeOllamaHandler(BaseHTTPRequestHandler):
    server: _FakeOllamaServer

    def log_message(self, _format: str, *_args: object) -> None:
        return None

    def _send_json(self, status: int, value: object) -> None:
        raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        self.server.state.requests.append(
            (
                self.path,
                b"",
                {key.lower(): value for key, value in self.headers.items()},
            )
        )
        if self.path == "/api/version":
            self._send_json(200, {"version": self.server.state.version})
            return
        if self.path == "/api/tags":
            self._send_json(
                200,
                {
                    "models": [
                        {
                            "name": self.server.state.model,
                            "model": self.server.state.model,
                            "digest": self.server.state.digest,
                        }
                    ]
                },
            )
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.server.state.requests.append(
            (
                self.path,
                body,
                {key.lower(): value for key, value in self.headers.items()},
            )
        )
        if self.path != "/api/chat":
            self._send_json(404, {"error": "not_found"})
            return
        if self.server.state.chat_delay:
            time.sleep(self.server.state.chat_delay)
        if self.server.state.chat_response_bytes is not None:
            raw = self.server.state.chat_response_bytes
            self.send_response(self.server.state.chat_status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except BrokenPipeError:
                pass
            return
        self._send_json(
            self.server.state.chat_status,
            {
                "model": self.server.state.model,
                "message": {
                    "role": "assistant",
                    "content": self.server.state.chat_content,
                },
                "done": True,
                "done_reason": "stop",
            },
        )


@contextmanager
def _server():
    state = FakeOllamaState()
    server = _FakeOllamaServer(state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield state, f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def _settings(endpoint: str | None, **overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "baseline_generation_provider": "ollama",
        "baseline_generation_endpoint": endpoint,
        "baseline_generation_model": MODEL,
        "baseline_generation_model_digest": DIGEST,
        "baseline_generation_timeout_seconds": 2.0,
        "baseline_generation_allow_loopback_http": True,
        "baseline_generation_max_request_bytes": 256_000,
        "baseline_generation_max_response_bytes": 200_000,
        "baseline_generation_context_tokens": 32_768,
        "baseline_generation_output_tokens": 1_024,
        "baseline_generation_seed": 17,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _config(endpoint: str, **overrides: object) -> OllamaGenerationConfig:
    config = OllamaGenerationConfig.from_settings(_settings(endpoint, **overrides))
    return config


def _generation_input(count: int = 2, *, source_text: str = "Synthetic source"):
    evidence = []
    for ordinal in range(1, count + 1):
        renderer = f"Repository file: sibling-{ordinal}/file.txt\n\nEvidence {ordinal}"
        evidence.append(
            BaselineGenerationEvidence(
                ordinal=ordinal,
                fused_rank=ordinal,
                bm25_score=float(ordinal),
                bm25_rank=ordinal,
                dense_score=float(ordinal),
                dense_rank=ordinal,
                rrf_score=1.0 / (60 + ordinal),
                selected_evidence_id=f"selected-{ordinal}",
                artifact_id=f"artifact-{ordinal}",
                repository_id=f"repository-{ordinal}",
                repository_name=f"sibling-{ordinal}",
                relative_path="file.txt",
                renderer_version="baseline-evidence-renderer.v1",
                renderer_output=renderer,
                renderer_output_hash="a" * 64,
                selected_content_hash="b" * 64,
                whole_file_content_hash="c" * 64,
                corpus_generation_id="generation-id",
                index_id="index-id",
                index_document_id=f"document-{ordinal}",
                index_fingerprint="d" * 64,
            )
        )
    return BaselineGenerationInput(
        run_id="run-id",
        group_id="group-id",
        source_scope_version="baseline-source-scope.v1",
        source_scope="control_document",
        source_chunk_id=None,
        source_document_id="source-document-id",
        source_text=source_text,
        corpus_generation_id="generation-id",
        corpus_manifest_hash="e" * 64,
        index_id="index-id",
        index_fingerprint="d" * 64,
        query_sha256="f" * 64,
        evidence=tuple(evidence),
        input_fingerprint="0" * 64,
    )


def _provider(endpoint: str, **overrides: object) -> OllamaBaselineGenerationProvider:
    provider = OllamaBaselineGenerationProvider(_config(endpoint, **overrides))
    provider.attest()
    return provider


def test_configuration_rejects_insecure_or_ambiguous_transport() -> None:
    with pytest.raises(BaselineGenerationProviderError) as disabled:
        OllamaGenerationConfig.from_settings(
            _settings(None, baseline_generation_provider="disabled")
        )
    assert disabled.value.code == "provider_unconfigured"

    for endpoint, allowed in (
        ("http://127.0.0.1:11434", False),
        ("http://localhost:11434", True),
        ("http://192.0.2.10:11434", True),
        ("http://user:secret@127.0.0.1:11434", True),
        ("http://127.0.0.1:11434?token=secret", True),
    ):
        with pytest.raises(BaselineGenerationProviderError) as error:
            OllamaGenerationConfig.from_settings(
                _settings(
                    endpoint,
                    baseline_generation_allow_loopback_http=allowed,
                )
            )
        assert error.value.code == "insecure_transport"

    secure = OllamaGenerationConfig.from_settings(
        _settings(
            "https://generation.example.test",
            baseline_generation_allow_loopback_http=False,
        )
    )
    assert secure.endpoint == "https://generation.example.test"


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ({"version": "0.31.0"}, "unsupported_runtime"),
        ({"model": "another:tag"}, "model_absent"),
        ({"digest": "0" * 64}, "digest_mismatch"),
    ],
)
def test_readiness_distinguishes_runtime_model_and_digest(
    mutation: dict[str, str], expected: str
) -> None:
    with _server() as (state, endpoint):
        for key, value in mutation.items():
            setattr(state, key, value)
        readiness = verify_ollama_generation(_settings(endpoint))
        assert all(path != "/api/chat" for path, _body, _headers in state.requests)
    assert readiness.ready is False
    assert readiness.status == expected
    serialized = json.dumps(readiness.as_dict(), sort_keys=True)
    assert endpoint not in serialized
    assert "Synthetic" not in serialized


def test_unavailable_endpoint_is_safe_and_retryable() -> None:
    readiness = verify_ollama_generation(_settings("http://127.0.0.1:1"))
    assert readiness.ready is False
    assert readiness.status == "endpoint_unavailable"
    assert "127.0.0.1" not in json.dumps(readiness.as_dict())


def test_probe_reports_structured_output_unavailable_without_payload() -> None:
    with _server() as (state, endpoint):
        state.chat_status = 400
        readiness = verify_ollama_generation(_settings(endpoint), probe=True)
    assert readiness.ready is False
    assert readiness.status == "structured_output_unavailable"
    serialized = json.dumps(readiness.as_dict(), sort_keys=True)
    assert endpoint not in serialized
    assert state.chat_content not in serialized


def test_exact_schema_deterministic_request_and_order(monkeypatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    with _server() as (state, endpoint):
        provider = _provider(endpoint)
        generation_input = _generation_input()
        output = provider.generate(generation_input, idempotency_key="opaque-key")

    assert output == state.chat_content
    chats = [request for request in state.requests if request[0] == "/api/chat"]
    assert len(chats) == 1
    payload = json.loads(chats[0][1])
    packaged_schema = json.loads(
        (
            ROOT
            / "compair_core/baseline_generation"
            / "baseline-generation-output.v2.schema.json"
        ).read_bytes()
    )
    protocol_schema = json.loads(
        (ROOT / "protocol/baseline-generation-output.v2.schema.json").read_bytes()
    )
    assert payload["format"] == packaged_schema == protocol_schema
    assert payload["model"] == MODEL
    assert payload["stream"] is False
    assert payload["think"] is False
    assert "tools" not in payload
    assert payload["options"] == {
        "temperature": 0,
        "seed": 17,
        "num_ctx": 32_768,
        "num_predict": 1_024,
    }
    user = payload["messages"][1]["content"]
    first, second = [item.renderer_output for item in generation_input.evidence]
    assert user.index(first) < user.index(second)
    assert (
        "between one and 2 nonblank feedback strings"
        in (payload["messages"][0]["content"])
    )
    assert (
        "outcome no_findings with findings as an empty array"
        in (payload["messages"][0]["content"])
    )
    assert "Never use NONE or an empty string" in payload["messages"][0]["content"]
    assert "opaque-key" not in chats[0][1].decode("utf-8")


def test_redirects_are_not_followed() -> None:
    with _server() as (state, endpoint):
        state.chat_status = 302
        provider = _provider(endpoint)
        with pytest.raises(BaselineGenerationProviderError) as redirected:
            provider.generate(_generation_input(), idempotency_key="ignored")
    assert redirected.value.code == "endpoint_unavailable"
    assert sum(request[0] == "/api/chat" for request in state.requests) == 1


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (_structured("no_findings", []), ()),
        (_structured("findings", ["One", "Two"]), ("One", "Two")),
    ],
)
def test_positive_and_zero_outputs_use_existing_strict_parser(
    content: str, expected: tuple[str, ...]
) -> None:
    with _server() as (state, endpoint):
        state.chat_content = content
        output = _provider(endpoint).generate(
            _generation_input(), idempotency_key="ignored"
        )
    findings, fingerprint = BaselineGenerationService._parse_output(
        output, maximum_findings=2
    )
    assert findings == expected
    assert len(fingerprint) == 64


@pytest.mark.parametrize(
    "content",
    [
        "plain text",
        "NONE",
        "```json\n{}\n```",
        "",
        "{",
        _structured("findings", ["One", "Two", "Three", "Four", "Five"]),
        (
            '{"schema_version":"baseline-generation-output.v2",'
            '"outcome":"findings","findings":'
            '[{"feedback":"One","extra":true}]}'
        ),
    ],
)
def test_malformed_or_excessive_native_content_fails_closed(content: str) -> None:
    with _server() as (state, endpoint):
        state.chat_content = content
        output = _provider(endpoint).generate(
            _generation_input(), idempotency_key="ignored"
        )
    with pytest.raises(BaselineGenerationError) as error:
        BaselineGenerationService._parse_output(output, maximum_findings=2)
    assert error.value.code == "provider_malformed_output"


def test_timeout_transient_status_response_limit_and_request_limit() -> None:
    with _server() as (state, endpoint):
        state.chat_delay = 0.2
        provider = _provider(endpoint, baseline_generation_timeout_seconds=0.1)
        with pytest.raises(BaselineGenerationProviderError) as timeout:
            provider.generate(_generation_input(), idempotency_key="ignored")
        assert timeout.value.code == "endpoint_unavailable"
        assert timeout.value.retryable is True

    with _server() as (state, endpoint):
        state.chat_status = 503
        provider = _provider(endpoint)
        with pytest.raises(BaselineGenerationProviderError) as unavailable:
            provider.generate(_generation_input(), idempotency_key="ignored")
        assert unavailable.value.retryable is True

    with _server() as (state, endpoint):
        state.chat_response_bytes = b"{" + (b" " * 5_000) + b"}"
        provider = _provider(endpoint, baseline_generation_max_response_bytes=4_096)
        with pytest.raises(BaselineGenerationProviderError) as oversized:
            provider.generate(_generation_input(), idempotency_key="ignored")
        assert oversized.value.code == "provider_response_too_large"

    with _server() as (_state, endpoint):
        provider = _provider(
            endpoint,
            baseline_generation_context_tokens=2_048,
            baseline_generation_output_tokens=1_024,
        )
        with pytest.raises(BaselineGenerationProviderError) as request:
            provider.generate(
                _generation_input(source_text="x" * 2_000),
                idempotency_key="ignored",
            )
        assert request.value.code == "provider_request_too_large"


def test_no_fallback_and_safe_persistable_identity() -> None:
    settings = _settings(None, baseline_generation_provider="disabled")
    with pytest.raises(BaselineRunRuntimeError) as disabled:
        _configured_generation_provider(settings)
    assert disabled.value.code == "worker_unavailable"

    with _server() as (_state, endpoint):
        provider = _provider(endpoint)
        identity = provider.identity
        version = provider.version
    assert identity.provider == "ollama"
    assert identity.adapter_contract == OLLAMA_GENERATION_ADAPTER_CONTRACT
    assert identity.model == MODEL
    assert identity.digest == DIGEST
    assert identity.runtime_version == "0.32.13"
    assert identity.output_spec_sha256 == GENERATION_OUTPUT_SPEC_SHA256
    assert identity.output_schema_sha256 == GENERATION_OUTPUT_SCHEMA_SHA256
    assert identity.supports_idempotency is False
    assert DIGEST in version and "runtime=0.32.13" in version
    assert "127.0.0.1" not in version and "11434" not in version


def test_generic_strict_http_remains_separate_and_explicit() -> None:
    provider = _configured_generation_provider(
        _settings(
            "https://generation.example.test/v1/generate",
            baseline_generation_provider="http",
            baseline_generation_model="strict-http-model",
            baseline_generation_model_version="immutable-http-version",
        )
    )
    assert provider.provider == "http"
    assert provider.model == "strict-http-model"
    assert not isinstance(provider, OllamaBaselineGenerationProvider)


@pytest.mark.parametrize(
    ("content", "feedback_count", "outbox_count"),
    [
        (_structured("findings", ["First", "Second"]), 2, 1),
        (_structured("no_findings", []), 0, 0),
    ],
)
def test_native_provider_uses_existing_atomic_control_generation(
    tmp_path: Path,
    content: str,
    feedback_count: int,
    outbox_count: int,
) -> None:
    environment = _environment(tmp_path, f"ollama-control-{feedback_count}.db")
    try:
        job_id, _caller, persisted = _persist_control(environment)
        with _server() as (state, endpoint):
            state.chat_content = content
            provider = _provider(endpoint)
            service = BaselineGenerationService(
                environment.sessions,
                notifications_enabled=False,
            )
            receipt = service.generate_control(job_id, provider)
            replay = service.generate_control(job_id, provider)
            chat_count = sum(request[0] == "/api/chat" for request in state.requests)

        assert receipt.state == "feedback_persisted"
        assert replay.replayed is True
        assert replay.feedback_ids == receipt.feedback_ids
        assert len(receipt.feedback_ids) == feedback_count
        assert receipt.notification_outbox_count == outbox_count
        assert chat_count == 1
        job, run, feedback, outbox, notifications = _state(
            environment,
            job_id,
            persisted.run_id,
        )
        assert job["state"] == "feedback_persisted"
        assert run["generation_state"] == "succeeded"
        assert run["generation_provider"] == "ollama"
        assert run["generation_model"] == MODEL
        assert run["generation_model_version"] == provider.version
        assert bool(job["generation_provider_idempotency_supported"]) is False
        assert job["generation_output_schema_sha256"] == (
            GENERATION_OUTPUT_SCHEMA_SHA256
        )
        assert [row["feedback"] for row in feedback] == (
            ["First", "Second"] if feedback_count else []
        )
        assert [row["baseline_finding_ordinal"] for row in feedback] == list(
            range(1, feedback_count + 1)
        )
        assert len(outbox) == outbox_count
        assert notifications == 0
        with environment.engine.connect() as connection:
            renderer_order = (
                connection.execute(
                    text(
                        "SELECT renderer_output FROM baseline_selected_evidence "
                        "WHERE run_id = :run_id ORDER BY ordinal"
                    ),
                    {"run_id": persisted.run_id},
                )
                .scalars()
                .all()
            )
        chat = next(request for request in state.requests if request[0] == "/api/chat")
        user_prompt = json.loads(chat[1])["messages"][1]["content"]
        positions = [user_prompt.index(renderer) for renderer in renderer_order]
        assert positions == sorted(positions)
    finally:
        environment.engine.dispose()


def test_native_transient_failure_recovers_without_duplicate_feedback(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path, "ollama-control-retry.db")
    try:
        job_id, _caller, persisted = _persist_control(environment)
        with _server() as (state, endpoint):
            provider = _provider(endpoint)
            service = BaselineGenerationService(
                environment.sessions,
                notifications_enabled=False,
            )
            state.chat_status = 503
            failed = service.generate_control(job_id, provider)
            state.chat_status = 200
            recovered = service.generate_control(job_id, provider)
            replay = service.generate_control(job_id, provider)
        assert failed.state == "retryable_failed"
        assert failed.error_code == "endpoint_unavailable"
        assert recovered.state == "feedback_persisted"
        assert recovered.generation_attempt_count == 2
        assert replay.replayed is True
        job, run, feedback, _outbox, _notifications = _state(
            environment,
            job_id,
            persisted.run_id,
        )
        assert job["state"] == "feedback_persisted"
        assert run["generation_state"] == "succeeded"
        assert len(feedback) == 1
    finally:
        environment.engine.dispose()


def test_native_malformed_output_is_safe_terminal_and_zero_write(
    tmp_path: Path,
) -> None:
    environment = _environment(tmp_path, "ollama-control-terminal.db")
    try:
        job_id, _caller, persisted = _persist_control(environment)
        with _server() as (state, endpoint):
            state.chat_content = "NONE"
            receipt = BaselineGenerationService(
                environment.sessions,
                notifications_enabled=False,
            ).generate_control(job_id, _provider(endpoint))
        assert receipt.state == "terminal_failed"
        assert receipt.error_code == "provider_malformed_output"
        job, run, feedback, outbox, notifications = _state(
            environment,
            job_id,
            persisted.run_id,
        )
        assert job["state"] == "terminal_failed"
        assert run["generation_state"] == "terminal_failed"
        assert feedback == []
        assert outbox == []
        assert notifications == 0
    finally:
        environment.engine.dispose()


def test_verify_command_emits_one_safe_json_value(monkeypatch, capsys, caplog) -> None:
    with _server() as (state, endpoint):
        state.chat_content = _structured("no_findings", [])
        monkeypatch.setenv("COMPAIR_BASELINE_GENERATION_PROVIDER", "ollama")
        monkeypatch.setenv("COMPAIR_BASELINE_GENERATION_ENDPOINT", endpoint)
        monkeypatch.setenv("COMPAIR_BASELINE_GENERATION_MODEL", MODEL)
        monkeypatch.setenv("COMPAIR_BASELINE_GENERATION_MODEL_DIGEST", DIGEST)
        monkeypatch.setenv("COMPAIR_BASELINE_GENERATION_TIMEOUT_SECONDS", "2")
        monkeypatch.setenv("COMPAIR_BASELINE_GENERATION_ALLOW_LOOPBACK_HTTP", "true")
        assert generation_cli(["verify"]) == 0
        static_line = capsys.readouterr().out
        assert generation_cli(["verify", "--probe"]) == 0
        probe_line = capsys.readouterr().out

    static = json.loads(static_line)
    probed = json.loads(probe_line)
    assert static["status"] == "ready" and static["probe_performed"] is False
    assert probed["status"] == "ready" and probed["probe_performed"] is True
    assert probed["probe_outcome"] == "no_findings"
    assert len(static_line.strip().splitlines()) == 1
    assert len(probe_line.strip().splitlines()) == 1
    for raw in (static_line, probe_line):
        assert endpoint not in raw
        assert "Synthetic compatibility" not in raw
        assert "evidence" not in raw.lower()
    assert endpoint not in caplog.text
    assert "Synthetic compatibility" not in caplog.text


def test_frozen_regex_and_provider_path_are_linear_at_maximum_output() -> None:
    from test_baseline_control_plane_v2_protocol import (
        ContractValidationError,
        _validate_schema,
    )

    schema = json.loads(
        (ROOT / "protocol/baseline-generation-output.v2.schema.json").read_bytes()
    )
    pattern = re.compile(
        schema["$defs"]["finding"]["properties"]["feedback"]["pattern"]
    )
    invalid = " " * 100_000
    valid = ("\n" * 50_000) + "X" + ("\n" * 49_999)
    invalid_value = json.loads(_structured("findings", [invalid]))
    valid_value = json.loads(_structured("findings", [valid]))
    tracemalloc.start()
    started = time.perf_counter()
    assert pattern.search(invalid) is None
    assert pattern.search(valid) is not None
    with pytest.raises(ContractValidationError):
        _validate_schema(invalid_value, schema, schema)
    _validate_schema(valid_value, schema, schema)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert time.perf_counter() - started < 2.0
    # CPython's frozen repeated-capture pattern uses a bounded temporary mark
    # stack; cap it well below process-scale growth for a 100k input.
    assert peak < 32_000_000

    raw = _structured("findings", [invalid])
    started = time.perf_counter()
    with pytest.raises(BaselineGenerationError):
        BaselineGenerationService._parse_output(raw, maximum_findings=1)
    assert time.perf_counter() - started < 2.0

    with _server() as (state, endpoint):
        state.chat_content = _structured("findings", [" " * 90_000])
        provider = _provider(endpoint)
        started = time.perf_counter()
        output = provider.generate(_generation_input(1), idempotency_key="ignored")
        with pytest.raises(BaselineGenerationError):
            BaselineGenerationService._parse_output(output, maximum_findings=1)
        assert time.perf_counter() - started < 5.0


def test_packaged_schema_and_entry_point_are_frozen() -> None:
    protocol = (
        ROOT / "protocol/baseline-generation-output.v2.schema.json"
    ).read_bytes()
    packaged = (
        ROOT
        / "compair_core/baseline_generation"
        / "baseline-generation-output.v2.schema.json"
    ).read_bytes()
    assert packaged == protocol
    assert __import__("hashlib").sha256(packaged).hexdigest() == (
        GENERATION_OUTPUT_SCHEMA_SHA256
    )
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"]["compair-core-generation"] == (
        "compair_core.baseline_generation.cli:main"
    )


def test_default_settings_are_fail_closed(monkeypatch) -> None:
    for key in tuple(__import__("os").environ):
        if key.startswith("COMPAIR_BASELINE_GENERATION_"):
            monkeypatch.delenv(key, raising=False)
    settings = Settings()
    assert settings.baseline_generation_provider == "disabled"
    assert settings.baseline_generation_model == MODEL
    assert settings.baseline_generation_model_digest == DIGEST
    assert (
        settings.baseline_generation_timeout_seconds
        == ACCELERATED_GENERATION_TIMEOUT_SECONDS
    )
    readiness = verify_ollama_generation(settings)
    assert readiness.status == "provider_unconfigured"
    assert readiness.ready is False
