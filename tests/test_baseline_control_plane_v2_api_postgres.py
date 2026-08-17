"""Real PostgreSQL checks for the frozen v2 compatible-index endpoints.

Run with::

    COMPAIR_TEST_POSTGRES_URL=postgresql+psycopg2://... \
      pytest -q tests/test_baseline_control_plane_v2_api_postgres.py
"""

from __future__ import annotations

import pytest
from test_baseline_control_plane_postgres import (
    postgres_control_environment as _postgres_control_environment_fixture,  # noqa: F401
)
from test_baseline_control_plane_v2_api import (
    _client,
    _v2_capabilities,
    _v2_payload,
    _v2_status,
)
from test_baseline_index_continuation import _publish_corpus

from compair_core.compair.retrieval.index_continuation import (
    InternalIndexWorkerIdentity,
)


@pytest.fixture
def postgres_control_environment(request: pytest.FixtureRequest):
    return request.getfixturevalue("_postgres_control_environment_fixture")


def test_postgres_v2_capability_submission_replay_publication_and_status(
    postgres_control_environment,
    monkeypatch,
) -> None:
    environment = postgres_control_environment
    _publish_corpus(environment)
    client, service = _client(environment, monkeypatch)
    payload = _v2_payload(environment)
    with client:
        capability = client.post(
            "/baseline/control/v2/capabilities", json=_v2_capabilities(environment)
        )
        assert capability.status_code == 200
        assert capability.json()["operations"]["index_build"]["readiness"] == "ready"
        first = client.post("/baseline/control/v2/index-builds", json=payload)
        replay = client.post("/baseline/control/v2/index-builds", json=payload)
        assert first.status_code == replay.status_code == 202
        assert replay.json()["job_id"] == first.json()["job_id"]
        assert replay.json()["replayed"] is True

        service.execute(
            identity=InternalIndexWorkerIdentity.create("postgres-v2-worker"),
            group_id=environment.group_id,
            job_id=first.json()["job_id"],
        )
        status = client.post(
            "/baseline/control/v2/index-builds/status",
            json=_v2_status(environment, first.json()["job_id"]),
        )
    assert status.status_code == 200
    assert status.json()["state"] == "succeeded"
    assert status.json()["result"]["document_count"] == 1
    assert status.json()["result"]["vector_count"] == 1
