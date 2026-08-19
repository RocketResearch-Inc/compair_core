from __future__ import annotations

import hashlib
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from test_baseline_control_plane import (
    ControlEnvironment,
    _continuation_worker,
    _stage_worker_snapshot,
)
from test_baseline_control_plane import (
    environment as _environment_fixture,  # noqa: F401 - pytest fixture import
)

from compair_core.baseline_control_plane_schema import (
    compatible_index_job,
    control_job,
    repository_approval,
    snapshot_continuation_job,
)
from compair_core.compair.retrieval.baseline import BASELINE_TOKENIZER_VERSION
from compair_core.compair.retrieval.continuation_worker import (
    InternalContinuationWorkerIdentity,
)
from compair_core.compair.retrieval.control_plane import (
    PROTOCOL_SHA256,
    PROTOCOL_VERSION,
    assess_control_transport,
    capabilities_response,
)
from compair_core.compair.retrieval.corpus import (
    RetrievalBaselineIndexBuild,
    RetrievalBaselineIndexPublication,
)
from compair_core.compair.retrieval.embedding import (
    BASELINE_EMBEDDING_HTTP_CONTRACT,
    BASELINE_EMBEDDING_HTTP_PROVIDER,
    BaselineEmbeddingAdapterError,
)
from compair_core.compair.retrieval.index_continuation import (
    PINNED_BASELINE_DIMENSION,
    PINNED_BASELINE_MODEL,
    BaselineCompatibleIndexJobService,
    IndexJobError,
    IndexJobStage,
    InternalIndexWorkerIdentity,
)
from compair_core.compair.retrieval.indexing import (
    BASELINE_INDEX_SCHEMA_VERSION,
    BaselineEmbeddingIdentity,
    baseline_engine_config_fingerprint,
)

REVISION = "fixture-bge-immutable-revision"


@pytest.fixture
def environment(request: pytest.FixtureRequest):
    """Reuse the control-plane fixture without duplicating its setup."""

    return request.getfixturevalue("_environment_fixture")


def _fingerprint(revision: str = REVISION) -> str:
    return hashlib.sha256(
        (
            '{"contract_version":"baseline-embedding-http.v1","dimension":384,'
            '"model":"BAAI/bge-small-en-v1.5","provider":"baseline_http_v1",'
            f'"revision":"{revision}"}}'
        ).encode()
    ).hexdigest()


def _identity(revision: str = REVISION) -> BaselineEmbeddingIdentity:
    return BaselineEmbeddingIdentity(
        provider=BASELINE_EMBEDDING_HTTP_PROVIDER,
        model=PINNED_BASELINE_MODEL,
        revision=revision,
        dimension=PINNED_BASELINE_DIMENSION,
        fingerprint=_fingerprint(revision),
    )


class FixtureAdapter:
    def __init__(self, *, mode: str = "ok", identity=None) -> None:
        self._identity = identity or _identity()
        self.provider = self._identity.provider
        self.model = self._identity.model
        self.revision = self._identity.revision
        self.dimension = self._identity.dimension
        self.fingerprint = self._identity.fingerprint
        self.mode = mode

    @property
    def identity(self):
        return self._identity

    def attest(self):
        if self.mode == "unavailable":
            raise BaselineEmbeddingAdapterError(
                "embedding_service_unavailable", "service unavailable"
            )
        return self._identity

    def embed(self, texts):
        if self.mode == "failure":
            raise BaselineEmbeddingAdapterError(
                "embedding_service_timeout", "service timeout"
            )
        vectors = [
            [float(ordinal + 1)] + [0.0] * (PINNED_BASELINE_DIMENSION - 1)
            for ordinal, _text in enumerate(texts)
        ]
        if vectors and self.mode == "dimension":
            vectors[0] = vectors[0][:-1]
        if vectors and self.mode == "nan":
            vectors[0][0] = math.nan
        return vectors


def _publish_corpus(environment: ControlEnvironment, *, ordinal: int = 1) -> str:
    continuation_id = _stage_worker_snapshot(
        environment,
        content=f"benign index corpus {ordinal}\n",
        idempotency_key=f"opaque-index-corpus-continuation-intent-{ordinal:04d}",
    )
    _continuation_worker(environment).execute(
        identity=InternalContinuationWorkerIdentity.create(f"ingestion-{ordinal}"),
        group_id=environment.group_id,
        continuation_job_id=continuation_id,
    )
    return continuation_id


def _payload(
    environment: ControlEnvironment,
    *,
    idempotency_key: str = "opaque-index-build-intent-00000001",
    identity: BaselineEmbeddingIdentity | None = None,
    continuation_job_id: str | None = None,
) -> dict[str, object]:
    identity = identity or _identity()
    with environment.engine.connect() as connection:
        statement = select(snapshot_continuation_job).where(
            snapshot_continuation_job.c.state == "succeeded"
        )
        if continuation_job_id is not None:
            statement = statement.where(
                snapshot_continuation_job.c.continuation_job_id
                == continuation_job_id
            )
        continuation = (
            connection.execute(
                statement.order_by(snapshot_continuation_job.c.finished_at.desc())
            )
            .mappings()
            .first()
        )
    assert continuation is not None
    return {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256,
        "message_type": "index_build_submit",
        "request_id": "90000000-0000-4000-8000-000000000001",
        "group_id": environment.group_id,
        "idempotency_key": idempotency_key,
        "snapshot_id": continuation["snapshot_id"],
        "corpus_generation_id": continuation["result_generation_id"],
        "canonical_manifest_hash": continuation["canonical_manifest_hash"],
        "index_format_version": BASELINE_INDEX_SCHEMA_VERSION,
        "tokenizer_version": BASELINE_TOKENIZER_VERSION,
        "retrieval_config_fingerprint": baseline_engine_config_fingerprint(identity),
        "embedding": {
            "contract_version": BASELINE_EMBEDDING_HTTP_CONTRACT,
            "provider": identity.provider,
            "model": identity.model,
            "revision": identity.revision,
            "dimension": identity.dimension,
            "fingerprint": identity.fingerprint,
        },
    }


def _service(environment: ControlEnvironment, adapter=None, *, clock=None, stage_hook=None):
    selected = adapter or FixtureAdapter()
    options = {"stage_hook": stage_hook}
    if clock is not None:
        options["clock"] = clock
    return BaselineCompatibleIndexJobService(
        environment.engine,
        lambda: selected,
        **options,
    )


def test_index_job_success_replay_status_restart_and_no_feature_enablement(
    environment: ControlEnvironment,
) -> None:
    _publish_corpus(environment)
    payload = _payload(environment)
    service = _service(environment)
    accepted = service.submit(payload, caller_user_id=environment.user_id)
    replay = service.submit(payload, caller_user_id=environment.user_id)
    assert replay["job_id"] == accepted["job_id"]
    assert replay["replayed"] is True

    outcome = service.execute(
        identity=InternalIndexWorkerIdentity.create("index-worker-success"),
        group_id=environment.group_id,
        job_id=str(accepted["job_id"]),
    )
    assert outcome.state == "succeeded"
    assert outcome.document_count == 1
    assert len(outcome.document_manifest_hash) == 64

    status = environment.service.job_status(
        {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "message_type": "job_status_request",
            "request_id": "90000000-0000-4000-8000-000000000002",
            "group_id": environment.group_id,
            "job_id": accepted["job_id"],
        },
        caller_user_id=environment.user_id,
    )
    assert status["operation"] == "index_build"
    assert status["state"] == "succeeded"
    assert status["result"] == {
        "corpus_generation_id": outcome.generation_id,
        "index_publication_id": outcome.index_id,
    }
    serialized = str(status)
    for forbidden in (
        "benign index corpus",
        payload["idempotency_key"],
        "lease_token",
        "repository_authority",
        "retrieval_query",
    ):
        assert str(forbidden) not in serialized

    restarted = _service(environment).execute(
        identity=InternalIndexWorkerIdentity.create("index-worker-restart"),
        group_id=environment.group_id,
        job_id=outcome.job_id,
    )
    assert restarted == outcome
    with environment.engine.connect() as connection:
        assert connection.execute(
            select(func.count()).select_from(compatible_index_job)
        ).scalar_one() == 1
        assert connection.execute(
            select(func.count()).select_from(control_job).where(
                control_job.c.operation == "baseline_run"
            )
        ).scalar_one() == 0
    capability = capabilities_response(
        request_id="90000000-0000-4000-8000-000000000003",
        group_id=environment.group_id,
        transport=assess_control_transport(
            connection_scheme="https",
            peer_host="203.0.113.8",
            allow_insecure_loopback=False,
        ),
    )
    assert capability["operations"] == {
        "snapshot_staging": "safe",
        "corpus_ingestion": "unavailable",
        "index_build": "unavailable",
        "baseline_run": "unavailable",
    }


def test_exact_intent_replays_across_keys_and_conflicting_key_fails(
    environment: ControlEnvironment,
) -> None:
    _publish_corpus(environment)
    service = _service(environment)
    first = service.submit(_payload(environment), caller_user_id=environment.user_id)
    different_key = _payload(
        environment,
        idempotency_key="opaque-index-build-intent-00000002",
    )
    replay = service.submit(different_key, caller_user_id=environment.user_id)
    assert replay["job_id"] == first["job_id"]
    assert replay["replayed"] is True

    _publish_corpus(environment, ordinal=2)
    conflict = _payload(
        environment,
        idempotency_key="opaque-index-build-intent-00000001",
    )
    with pytest.raises(IndexJobError, match="index_build_conflict"):
        service.submit(conflict, caller_user_id=environment.user_id)


@pytest.mark.parametrize(
    ("mode", "code"),
    (("dimension", "embedding_dimension_mismatch"), ("nan", "embedding_vector_nonfinite")),
)
def test_vector_failures_are_terminal_and_preserve_prior_publication(
    environment: ControlEnvironment,
    mode: str,
    code: str,
) -> None:
    _publish_corpus(environment, ordinal=1)
    first_service = _service(environment)
    first_job = first_service.submit(
        _payload(environment, idempotency_key="opaque-index-first-publication-0001"),
        caller_user_id=environment.user_id,
    )
    first = first_service.execute(
        identity=InternalIndexWorkerIdentity.create("first-index-worker"),
        group_id=environment.group_id,
        job_id=str(first_job["job_id"]),
    )
    _publish_corpus(environment, ordinal=2)
    failing_service = _service(environment, FixtureAdapter(mode=mode))
    failed_job = failing_service.submit(
        _payload(
            environment,
            idempotency_key=f"opaque-index-vector-{mode}-failure-intent-0002",
        ),
        caller_user_id=environment.user_id,
    )
    with pytest.raises(IndexJobError, match=code):
        failing_service.execute(
            identity=InternalIndexWorkerIdentity.create(f"{mode}-worker"),
            group_id=environment.group_id,
            job_id=str(failed_job["job_id"]),
        )
    with environment.engine.connect() as connection:
        failed_state = connection.execute(
            select(control_job.c.state).where(control_job.c.job_id == failed_job["job_id"])
        ).scalar_one()
        corpus_id = connection.execute(
            select(compatible_index_job.c.corpus_id).where(
                compatible_index_job.c.job_id == failed_job["job_id"]
            )
        ).scalar_one()
    sessions = failing_service.sessions
    with sessions() as session:
        publication = session.get(RetrievalBaselineIndexPublication, corpus_id)
    assert failed_state == "terminal_failed"
    assert publication is not None and publication.index_id == first.index_id
    terminal_status = environment.service.job_status(
        {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "message_type": "job_status_request",
            "request_id": "90000000-0000-4000-8000-000000000007",
            "group_id": environment.group_id,
            "job_id": failed_job["job_id"],
        },
        caller_user_id=environment.user_id,
    )
    assert terminal_status["state"] == "terminal_failed"
    assert terminal_status["result"] is None


@pytest.mark.parametrize(
    "fault_stage",
    (IndexJobStage.AFTER_PUBLICATION, IndexJobStage.BEFORE_SUCCESS),
)
def test_publication_crash_rolls_back_then_retry_succeeds(
    environment: ControlEnvironment,
    fault_stage: IndexJobStage,
) -> None:
    _publish_corpus(environment, ordinal=1)
    first_service = _service(environment)
    first_job = first_service.submit(
        _payload(
            environment,
            idempotency_key="opaque-index-atomic-first-publication-0001",
        ),
        caller_user_id=environment.user_id,
    )
    first = first_service.execute(
        identity=InternalIndexWorkerIdentity.create("first-atomic-index-worker"),
        group_id=environment.group_id,
        job_id=str(first_job["job_id"]),
    )

    _publish_corpus(environment, ordinal=2)
    payload = _payload(
        environment,
        idempotency_key="opaque-index-atomic-rollback-intent-0002",
    )
    job = _service(environment).submit(payload, caller_user_id=environment.user_id)

    def crash(stage: IndexJobStage) -> None:
        if stage is fault_stage:
            raise RuntimeError("injected publication crash")

    failing = _service(environment, stage_hook=crash)
    with pytest.raises(IndexJobError, match="index_publication_failed"):
        failing.execute(
            identity=InternalIndexWorkerIdentity.create("crashing-index-worker"),
            group_id=environment.group_id,
            job_id=str(job["job_id"]),
        )
    with failing.sessions() as session:
        extension = session.execute(
            select(compatible_index_job).where(
                compatible_index_job.c.job_id == job["job_id"]
            )
        ).mappings().one()
        publication = session.get(RetrievalBaselineIndexPublication, extension["corpus_id"])
        failed_state = session.execute(
            select(control_job.c.state).where(control_job.c.job_id == job["job_id"])
        ).scalar_one()
    assert publication is not None and publication.index_id == first.index_id
    assert extension["result_index_id"] is None
    assert failed_state == "retryable_failed"

    failed_status = environment.service.job_status(
        {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "message_type": "job_status_request",
            "request_id": "90000000-0000-4000-8000-000000000005",
            "group_id": environment.group_id,
            "job_id": job["job_id"],
        },
        caller_user_id=environment.user_id,
    )
    assert failed_status["state"] == "retryable_failed"
    assert failed_status["result"] is None

    recovered = _service(environment).execute(
        identity=InternalIndexWorkerIdentity.create("recovered-index-worker"),
        group_id=environment.group_id,
        job_id=str(job["job_id"]),
    )
    assert recovered.state == "succeeded"
    assert recovered.attempt_count == 2


def test_post_commit_crash_keeps_publication_and_succeeded_job_consistent(
    environment: ControlEnvironment,
) -> None:
    _publish_corpus(environment)
    payload = _payload(
        environment,
        idempotency_key="opaque-index-post-commit-crash-intent-0001",
    )
    job = _service(environment).submit(payload, caller_user_id=environment.user_id)

    def crash_after_commit(stage: IndexJobStage) -> None:
        if stage is IndexJobStage.AFTER_COMMIT:
            raise RuntimeError("injected response crash after commit")

    crashing = _service(environment, stage_hook=crash_after_commit)
    with pytest.raises(IndexJobError, match="index_build_failed"):
        crashing.execute(
            identity=InternalIndexWorkerIdentity.create("post-commit-crash-worker"),
            group_id=environment.group_id,
            job_id=str(job["job_id"]),
        )

    with crashing.sessions() as session:
        control = session.execute(
            select(control_job).where(control_job.c.job_id == job["job_id"])
        ).mappings().one()
        extension = session.execute(
            select(compatible_index_job).where(
                compatible_index_job.c.job_id == job["job_id"]
            )
        ).mappings().one()
        publication = session.get(
            RetrievalBaselineIndexPublication,
            extension["corpus_id"],
        )
    assert control["state"] == "succeeded"
    assert control["error_code"] is None
    assert extension["result_index_id"] is not None
    assert publication is not None
    assert publication.index_id == extension["result_index_id"]

    status = environment.service.job_status(
        {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "message_type": "job_status_request",
            "request_id": "90000000-0000-4000-8000-000000000006",
            "group_id": environment.group_id,
            "job_id": job["job_id"],
        },
        caller_user_id=environment.user_id,
    )
    assert status["state"] == "succeeded"
    assert status["result"] == {
        "corpus_generation_id": extension["generation_id"],
        "index_publication_id": extension["result_index_id"],
    }

    recovered = _service(environment).execute(
        identity=InternalIndexWorkerIdentity.create("post-commit-recovery-worker"),
        group_id=environment.group_id,
        job_id=str(job["job_id"]),
    )
    assert recovered.state == "succeeded"
    assert recovered.index_id == extension["result_index_id"]


def test_revocation_stale_generation_and_embedding_attestation_fail_closed(
    environment: ControlEnvironment,
) -> None:
    _publish_corpus(environment)
    payload = _payload(environment)
    service = _service(environment)
    revoked = service.submit(payload, caller_user_id=environment.user_id)
    with environment.engine.begin() as connection:
        connection.execute(
            update(repository_approval)
            .where(repository_approval.c.registration_id == environment.sibling_repository_id)
            .values(state="disabled", disabled_at=datetime.now(timezone.utc))
        )
    with pytest.raises(IndexJobError, match="repository_not_authorized"):
        service.execute(
            identity=InternalIndexWorkerIdentity.create("revoked-index-worker"),
            group_id=environment.group_id,
            job_id=str(revoked["job_id"]),
        )
    with environment.engine.connect() as connection:
        assert connection.execute(
            select(control_job.c.state).where(control_job.c.job_id == revoked["job_id"])
        ).scalar_one() == "terminal_failed"

    with environment.engine.begin() as connection:
        connection.execute(
            update(repository_approval)
            .where(repository_approval.c.registration_id == environment.sibling_repository_id)
            .values(state="active", disabled_at=None, disabled_by_user_id=None)
        )
    _publish_corpus(environment, ordinal=2)
    unavailable = _service(environment, FixtureAdapter(mode="unavailable"))
    unavailable_job = unavailable.submit(
        _payload(environment, idempotency_key="opaque-index-unavailable-intent-0002"),
        caller_user_id=environment.user_id,
    )
    with pytest.raises(IndexJobError, match="embedding_service_unavailable"):
        unavailable.execute(
            identity=InternalIndexWorkerIdentity.create("unavailable-index-worker"),
            group_id=environment.group_id,
            job_id=str(unavailable_job["job_id"]),
        )
    with environment.engine.connect() as connection:
        assert connection.execute(
            select(control_job.c.state).where(
                control_job.c.job_id == unavailable_job["job_id"]
            )
        ).scalar_one() == "retryable_failed"


def test_stale_generation_deleted_source_and_identity_drift_are_terminal(
    environment: ControlEnvironment,
) -> None:
    _publish_corpus(environment, ordinal=1)
    stale_service = _service(environment)
    stale_job = stale_service.submit(
        _payload(environment, idempotency_key="opaque-index-stale-generation-intent-0001"),
        caller_user_id=environment.user_id,
    )
    _publish_corpus(environment, ordinal=2)
    with pytest.raises(IndexJobError, match="corpus_generation_stale"):
        stale_service.execute(
            identity=InternalIndexWorkerIdentity.create("stale-generation-worker"),
            group_id=environment.group_id,
            job_id=str(stale_job["job_id"]),
        )

    source_job = _service(environment).submit(
        _payload(environment, idempotency_key="opaque-index-source-deletion-intent-0002"),
        caller_user_id=environment.user_id,
    )
    with environment.engine.begin() as connection:
        connection.exec_driver_sql(
            "DELETE FROM document WHERE document_id = ?",
            (environment.source_document_id,),
        )
    with pytest.raises(IndexJobError, match="source_not_authorized"):
        _service(environment).execute(
            identity=InternalIndexWorkerIdentity.create("deleted-source-worker"),
            group_id=environment.group_id,
            job_id=str(source_job["job_id"]),
        )


def test_embedding_identity_drift_and_transient_build_failure_recover_safely(
    environment: ControlEnvironment,
) -> None:
    _publish_corpus(environment)
    current = [FixtureAdapter()]
    service = BaselineCompatibleIndexJobService(
        environment.engine,
        lambda: current[0],
    )
    drift_job = service.submit(
        _payload(environment, idempotency_key="opaque-index-identity-drift-intent-0001"),
        caller_user_id=environment.user_id,
    )
    current[0] = FixtureAdapter(identity=_identity("different-pinned-revision"))
    with pytest.raises(IndexJobError, match="embedding_identity_mismatch"):
        service.execute(
            identity=InternalIndexWorkerIdentity.create("identity-drift-worker"),
            group_id=environment.group_id,
            job_id=str(drift_job["job_id"]),
        )

    # A new active generation avoids reusing the terminal exact intent.
    _publish_corpus(environment, ordinal=2)
    current[0] = FixtureAdapter(mode="failure")
    retry_job = service.submit(
        _payload(environment, idempotency_key="opaque-index-transient-build-intent-0002"),
        caller_user_id=environment.user_id,
    )
    with pytest.raises(IndexJobError, match="embedding_adapter_failed"):
        service.execute(
            identity=InternalIndexWorkerIdentity.create("transient-failure-worker"),
            group_id=environment.group_id,
            job_id=str(retry_job["job_id"]),
        )
    current[0] = FixtureAdapter()
    recovered = service.execute(
        identity=InternalIndexWorkerIdentity.create("transient-recovery-worker"),
        group_id=environment.group_id,
        job_id=str(retry_job["job_id"]),
    )
    assert recovered.state == "succeeded"
    assert recovered.attempt_count == 2


def test_expired_lease_reclaim_and_concurrent_workers(
    environment: ControlEnvironment,
) -> None:
    _publish_corpus(environment)
    clock_value = [datetime(2026, 8, 16, tzinfo=timezone.utc)]
    service = _service(environment, clock=lambda: clock_value[0])
    job = service.submit(_payload(environment), caller_user_id=environment.user_id)
    first = service.claim(
        identity=InternalIndexWorkerIdentity.create("first-lease-worker"),
        group_id=environment.group_id,
        job_id=str(job["job_id"]),
        lifetime=timedelta(seconds=5),
    )
    running_status = environment.service.job_status(
        {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": PROTOCOL_SHA256,
            "message_type": "job_status_request",
            "request_id": "90000000-0000-4000-8000-000000000008",
            "group_id": environment.group_id,
            "job_id": job["job_id"],
        },
        caller_user_id=environment.user_id,
    )
    assert running_status["state"] == "running"
    assert running_status["result"] is None
    clock_value[0] += timedelta(seconds=6)
    second = service.claim(
        identity=InternalIndexWorkerIdentity.create("reclaim-worker"),
        group_id=environment.group_id,
        job_id=str(job["job_id"]),
    )
    assert second.attempt_count == first.attempt_count + 1
    service.cancel(
        identity=InternalIndexWorkerIdentity.create("reclaim-worker"),
        group_id=environment.group_id,
        job_id=str(job["job_id"]),
        lease_token=second.lease_token,
    )

    _publish_corpus(environment, ordinal=2)
    concurrent_service = _service(environment)
    concurrent_job = concurrent_service.submit(
        _payload(environment, idempotency_key="opaque-index-concurrent-intent-0002"),
        caller_user_id=environment.user_id,
    )
    barrier = threading.Barrier(2)

    def execute(ordinal: int):
        barrier.wait()
        try:
            return concurrent_service.execute(
                identity=InternalIndexWorkerIdentity.create(f"concurrent-index-{ordinal}"),
                group_id=environment.group_id,
                job_id=str(concurrent_job["job_id"]),
            )
        except IndexJobError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(execute, range(2)))
    assert sum(not isinstance(item, str) for item in outcomes) == 1
    with environment.engine.connect() as connection:
        assert connection.execute(
            select(control_job.c.state).where(
                control_job.c.job_id == concurrent_job["job_id"]
            )
        ).scalar_one() == "succeeded"


def test_index_job_schema_rejects_result_without_atomic_success(
    environment: ControlEnvironment,
) -> None:
    _publish_corpus(environment)
    job = _service(environment).submit(
        _payload(environment), caller_user_id=environment.user_id
    )
    with pytest.raises(IntegrityError), environment.engine.begin() as connection:
        connection.execute(
            update(control_job)
            .where(control_job.c.job_id == job["job_id"])
            .values(state="succeeded")
        )


def test_index_build_endpoint_and_frozen_capability_cannot_disagree(
    environment: ControlEnvironment,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from compair_core import api as api_module

    _publish_corpus(environment)
    monkeypatch.setattr(
        api_module,
        "_compatible_index_job_service",
        lambda: pytest.fail("unavailable capability constructed the index service"),
    )
    monkeypatch.setattr(api_module, "_control_plane_service", lambda: environment.service)
    app = api_module.create_fastapi_app()
    app.dependency_overrides[api_module.get_current_user] = lambda: SimpleNamespace(
        user_id=environment.user_id,
        username="index-control@example.test",
        name="Index Control User",
    )
    payload = _payload(environment)
    with TestClient(app, base_url="https://core.example.test") as client:
        capability_response = client.post(
            "/baseline/control/v1/capabilities",
            json={
                "protocol_version": PROTOCOL_VERSION,
                "protocol_sha256": PROTOCOL_SHA256,
                "message_type": "capabilities_request",
                "request_id": "90000000-0000-4000-8000-000000000004",
                "group_id": environment.group_id,
            },
        )
        response = client.post("/baseline/control/v1/index-builds", json=payload)
    assert capability_response.status_code == 200
    assert capability_response.json()["operations"]["index_build"] == "unavailable"
    assert response.status_code == 503
    assert response.json()["code"] == "capability_unavailable"
    assert response.json()["retryable"] is False
    serialized = response.text
    assert payload["idempotency_key"] not in serialized
    assert "lease_token" not in serialized
    with environment.engine.connect() as connection:
        assert connection.execute(
            select(func.count()).select_from(control_job).where(
                control_job.c.operation == "index_build"
            )
        ).scalar_one() == 0
        assert connection.execute(
            select(func.count()).select_from(RetrievalBaselineIndexBuild)
        ).scalar_one() == 0
