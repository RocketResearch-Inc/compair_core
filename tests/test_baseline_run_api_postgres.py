"""Real PostgreSQL public/manual baseline-run integration.

Run with::

    COMPAIR_TEST_POSTGRES_URL=postgresql+psycopg2://... \
      pytest -q tests/test_baseline_run_api_postgres.py
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import sessionmaker
from test_baseline_control_generation import _structured
from test_baseline_control_plane_postgres import (
    postgres_control_environment as _postgres_control_environment_fixture,  # noqa: F401
)
from test_baseline_generation import CapturingProvider, RawOutputProvider
from test_baseline_run_api import (
    _client,
    _keyring_json,
    _manual_operator,
    _ready_runtime,
    _settings,
    _status,
)
from test_baseline_run_jobs import _run_payload

from compair_core.baseline_control_plane_schema import baseline_run_job
from compair_core.compair.retrieval.preview import (
    BaselinePreviewCommand,
    BaselinePreviewService,
)


@pytest.fixture
def postgres_control_environment(request: pytest.FixtureRequest):
    return request.getfixturevalue("_postgres_control_environment_fixture")


@pytest.mark.parametrize("findings", [("postgres finding",), ()])
def test_postgres_public_submission_manual_completion_and_preview(
    postgres_control_environment,
    monkeypatch: pytest.MonkeyPatch,
    findings: tuple[str, ...],
) -> None:
    environment = postgres_control_environment
    payload = _run_payload(environment)
    payload["idempotency_key"] = (
        "opaque-postgres-public-positive-0000001"
        if findings
        else "opaque-postgres-public-zero-0000000001"
    )
    settings = _settings(enabled=True, keyring=_keyring_json())
    provider = (
        CapturingProvider(*findings)
        if findings
        else RawOutputProvider(_structured("no_findings", []))
    )
    operator, recording = _manual_operator(environment, provider)
    runtime = _ready_runtime(environment, settings)
    runtime.operator = operator
    client = _client(environment, monkeypatch, settings, runtime=runtime)
    with client:
        accepted = client.post("/baseline/control/v2/runs", json=payload)
        assert accepted.status_code == 202
        queued = client.post(
            "/baseline/control/v2/runs/status",
            json=_status(environment.group_id, accepted.json()["job_id"]),
        )
        assert queued.json()["state"] == "queued"
        outcome = operator.process(accepted.json()["job_id"])
        complete = client.post(
            "/baseline/control/v2/runs/status",
            json=_status(environment.group_id, accepted.json()["job_id"]),
        )
    assert outcome.state == complete.json()["state"] == "feedback_persisted"
    assert complete.json()["effects"]["feedback_count"] == len(findings)
    assert len(recording.requests) == 1

    preview = BaselinePreviewService(
        sessionmaker(environment.engine, expire_on_commit=False)
    ).load(
        BaselinePreviewCommand(
            caller_user_id=environment.user_id,
            request_id=str(uuid4()),
            group_id=environment.group_id,
            job_id=accepted.json()["job_id"],
        )
    )
    assert [item.feedback for item in preview.feedback] == list(findings)
    with environment.engine.connect() as connection:
        payload_count = connection.execute(
            text(
                "SELECT count(*) FROM baseline_control_run_payload "
                "WHERE job_id = :job_id"
            ),
            {"job_id": accepted.json()["job_id"]},
        ).scalar_one()
        notifications = connection.execute(
            text("SELECT count(*) FROM notification_event")
        ).scalar_one()
    assert payload_count == 0
    assert notifications == 0


def test_postgres_generation_lease_projects_references_boundary(
    postgres_control_environment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = postgres_control_environment
    started = threading.Event()
    release = threading.Event()

    class BlockingProvider(CapturingProvider):
        def generate(self, generation_input, *, idempotency_key: str) -> str:
            self.inputs.append(generation_input)
            self.idempotency_keys.append(idempotency_key)
            started.set()
            assert release.wait(timeout=10)
            return self.output

    payload = _run_payload(environment)
    payload["idempotency_key"] = "opaque-postgres-public-boundary-00001"
    operator, _recording = _manual_operator(
        environment, BlockingProvider("postgres finding")
    )
    settings = _settings(enabled=True, keyring=_keyring_json())
    runtime = _ready_runtime(environment, settings)
    runtime.operator = operator
    client = _client(environment, monkeypatch, settings, runtime=runtime)
    with client, ThreadPoolExecutor(max_workers=1) as pool:
        accepted = client.post("/baseline/control/v2/runs", json=payload).json()
        future = pool.submit(operator.process, accepted["job_id"])
        assert started.wait(timeout=10)
        try:
            status = client.post(
                "/baseline/control/v2/runs/status",
                json=_status(environment.group_id, accepted["job_id"]),
            )
            with environment.engine.connect() as connection:
                internal_state = connection.execute(
                    select(baseline_run_job.c.state).where(
                        baseline_run_job.c.job_id == accepted["job_id"]
                    )
                ).scalar_one()
        finally:
            release.set()
        assert future.result(timeout=10).state == "feedback_persisted"

    assert internal_state == "running"
    assert status.status_code == 200
    assert status.json()["state"] == "references_persisted"
    assert status.json()["effects"]["persisted_run_id"] is not None
