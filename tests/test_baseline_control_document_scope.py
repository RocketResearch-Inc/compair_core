from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

import pytest
import rfc8785
from sqlalchemy import func, select, text
from sqlalchemy.orm import sessionmaker
from test_baseline_control_generation import _structured
from test_baseline_control_plane import (
    ControlEnvironment,
    _continuation_status_payload,
    _continuation_worker,
    _stage_worker_snapshot,
)
from test_baseline_control_plane import (
    environment as _environment_fixture,  # noqa: F401
)
from test_baseline_generation import CapturingProvider, RawOutputProvider
from test_baseline_index_continuation import (
    _payload as _index_payload,
)
from test_baseline_index_continuation import (
    _service as _index_service,
)
from test_baseline_run_executor import RecordingRetriever, _executor, _persistent
from test_baseline_run_jobs import RAW_QUERY, _keyring

from compair_core.baseline_control_plane_schema import (
    baseline_run_job,
    compatible_index_job,
    repository_approval,
    repository_registration,
)
from compair_core.baseline_evidence_schema import baseline_retrieval_run
from compair_core.compair.retrieval.continuation_worker import (
    ContinuationWorkerError,
    ContinuationWorkerStage,
    InternalContinuationWorkerIdentity,
)
from compair_core.compair.retrieval.control_document_scope import (
    CONTROL_DOCUMENT_CORPUS_SCOPE_MAX_LENGTH,
    CONTROL_DOCUMENT_CORPUS_SCOPE_PREFIX,
    CONTROL_DOCUMENT_CORPUS_SCOPE_VERSION,
    ControlDocumentCorpusScopeError,
    choose_control_document_corpus_scope_key,
    control_document_corpus_identity,
)
from compair_core.compair.retrieval.control_plane import canonical_sha256
from compair_core.compair.retrieval.control_plane_v2 import (
    PROTOCOL_V2_SHA256,
    PROTOCOL_V2_VERSION,
    parse_run_submission,
)
from compair_core.compair.retrieval.corpus import (
    RetrievalBaselineIndexBuild,
    RetrievalBaselineIndexPublication,
    RetrievalCorpus,
    RetrievalCorpusGeneration,
)
from compair_core.compair.retrieval.evidence_persistence import (
    BaselineEvidencePersistenceCommand,
    BaselineEvidencePersistenceError,
    BaselineEvidencePersistenceService,
    ControlDocumentSource,
)
from compair_core.compair.retrieval.generation import (
    BaselineGenerationService,
)
from compair_core.compair.retrieval.index_continuation import (
    InternalIndexWorkerIdentity,
)
from compair_core.compair.retrieval.persistent import published_index_fingerprint
from compair_core.compair.retrieval.preview import (
    BaselinePreviewCommand,
    BaselinePreviewService,
)
from compair_core.compair.retrieval.run_jobs import (
    BaselineRunJobError,
    BaselineRunJobService,
)


@pytest.fixture
def environment(request: pytest.FixtureRequest):
    return request.getfixturevalue("_environment_fixture")


def _add_source(environment: ControlEnvironment, suffix: str) -> ControlEnvironment:
    document_id = str(uuid4())
    registration_id = str(uuid4())
    now = datetime.now(timezone.utc)
    descriptor = {
        "version": "repository-identity.v1",
        "authority": "example.test",
        "repository_uid": f"upstream-changed-{suffix}",
    }
    with environment.engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO document "
                "(document_id, user_id, author_id, title, content, doc_type, "
                "datetime_created, datetime_modified, is_published) "
                "VALUES (:document_id, :user_id, :user_id, :title, :content, "
                "'text', :now, :now, true)"
            ),
            {
                "document_id": document_id,
                "user_id": environment.user_id,
                "title": f"Changed repository {suffix}",
                "content": f"benign source {suffix}",
                "now": now,
            },
        )
        connection.execute(
            text(
                "INSERT INTO document_to_group (document_id, group_id) "
                "VALUES (:document_id, :group_id)"
            ),
            {"document_id": document_id, "group_id": environment.group_id},
        )
        connection.execute(
            repository_registration.insert().values(
                registration_id=registration_id,
                group_id=environment.group_id,
                repository_id=registration_id,
                repository_name=registration_id,
                source_document_id=document_id,
                created_by_user_id=environment.user_id,
                enabled=True,
                created_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            repository_approval.insert().values(
                registration_id=registration_id,
                group_id=environment.group_id,
                descriptor_version="repository-identity.v1",
                repository_authority="example.test",
                repository_uid=descriptor["repository_uid"],
                descriptor_hash=canonical_sha256(descriptor),
                state="active",
                approved_by_user_id=environment.user_id,
                disabled_by_user_id=None,
                created_at=now,
                updated_at=now,
                disabled_at=None,
            )
        )
    return replace(
        environment,
        source_document_id=document_id,
        changed_repository_id=registration_id,
    )


def _ingest(
    environment: ControlEnvironment,
    *,
    ordinal: int,
    content: str,
):
    continuation_id = _stage_worker_snapshot(
        environment,
        content=content,
        idempotency_key=f"opaque-multi-source-ingestion-{ordinal:08d}",
    )
    outcome = _continuation_worker(environment).execute(
        identity=InternalContinuationWorkerIdentity.create(
            f"multi-source-ingestion-{ordinal}"
        ),
        group_id=environment.group_id,
        continuation_job_id=continuation_id,
    )
    return continuation_id, outcome


def _publish_index(
    environment: ControlEnvironment,
    *,
    continuation_id: str,
    ordinal: int,
):
    service = _index_service(environment)
    payload = _index_payload(
        environment,
        idempotency_key=f"opaque-multi-source-index-{ordinal:08d}",
        continuation_job_id=continuation_id,
    )
    accepted = service.submit(payload, caller_user_id=environment.user_id)
    outcome = service.execute(
        identity=InternalIndexWorkerIdentity.create(f"multi-source-index-{ordinal}"),
        group_id=environment.group_id,
        job_id=str(accepted["job_id"]),
    )
    with service.sessions() as session:
        extension = (
            session.execute(
                select(compatible_index_job).where(
                    compatible_index_job.c.job_id == outcome.job_id
                )
            )
            .mappings()
            .one()
        )
        build = session.get(RetrievalBaselineIndexBuild, outcome.index_id)
        assert build is not None
        return outcome, dict(extension), published_index_fingerprint(build)


def _run_payload(
    environment: ControlEnvironment,
    *,
    index_result,
    extension: dict[str, object],
    index_fingerprint: str,
    ordinal: int,
):
    encoded = RAW_QUERY.encode("utf-8")
    return {
        "protocol_version": PROTOCOL_V2_VERSION,
        "protocol_sha256": PROTOCOL_V2_SHA256,
        "message_type": "run_submit",
        "request_id": str(uuid4()),
        "group_id": environment.group_id,
        "idempotency_key": f"opaque-multi-source-run-{ordinal:08d}",
        "source_document_id": environment.source_document_id,
        "changed_repository_registration_id": environment.changed_repository_id,
        "index_publication": {
            "index_publication_id": index_result.index_id,
            "corpus_generation_id": index_result.generation_id,
            "corpus_manifest_hash": extension["corpus_manifest_hash"],
            "index_format_version": extension["index_format_version"],
            "tokenizer_version": extension["tokenizer_version"],
            "retrieval_config_fingerprint": extension["retrieval_config_fingerprint"],
            "embedding_fingerprint": extension["embedding_fingerprint"],
            "index_fingerprint": index_fingerprint,
        },
        "retrieval_query": {
            "representation": "raw_git_diff_v1",
            "origin": "explicit",
            "encoding": "utf-8",
            "base_revision": "1" * 40,
            "head_revision": "2" * 40,
            "byte_size": len(encoded),
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "text": RAW_QUERY,
        },
    }


def test_document_corpus_identity_is_canonical_bounded_and_exact() -> None:
    identity = control_document_corpus_identity(
        group_id="group-fixture",
        changed_repository_registration_id="changed-registration-fixture",
        source_document_id="source-document-fixture",
    )
    expected = hashlib.sha256(rfc8785.dumps(identity.canonical_payload)).hexdigest()

    assert rfc8785.dumps(identity.canonical_payload) == (
        b'{"changed_repository_registration_id":"changed-registration-fixture",'
        b'"group_id":"group-fixture","scope_contract_version":'
        b'"baseline-control-document-corpus-scope.v1","source_document_id":'
        b'"source-document-fixture"}'
    )
    assert (
        expected == "54d79c81f813c2df70fa02c986a59b55219bc9c1b05e7e5400a4d39ffd8aa46d"
    )
    assert identity.contract_version == CONTROL_DOCUMENT_CORPUS_SCOPE_VERSION
    assert identity.identity_sha256 == expected
    assert identity.scope_key == f"{CONTROL_DOCUMENT_CORPUS_SCOPE_PREFIX}{expected}"
    assert len(identity.scope_key) <= CONTROL_DOCUMENT_CORPUS_SCOPE_MAX_LENGTH
    assert "group-fixture" not in identity.scope_key
    assert (
        identity.scope_key
        == control_document_corpus_identity(
            group_id="group-fixture",
            changed_repository_registration_id="changed-registration-fixture",
            source_document_id="source-document-fixture",
        ).scope_key
    )
    assert (
        identity.scope_key
        != control_document_corpus_identity(
            group_id="group-fixture",
            changed_repository_registration_id="other-registration",
            source_document_id="source-document-fixture",
        ).scope_key
    )
    assert (
        identity.scope_key
        != control_document_corpus_identity(
            group_id="group-fixture",
            changed_repository_registration_id="changed-registration-fixture",
            source_document_id="other-document",
        ).scope_key
    )

    exact_legacy = (
        identity.legacy_group_scope_key,
        identity.changed_repository_registration_id,
        identity.source_document_id,
    )
    assert choose_control_document_corpus_scope_key(identity, (exact_legacy,)) == (
        identity.legacy_group_scope_key
    )
    mismatched_legacy = (
        identity.legacy_group_scope_key,
        "another-registration",
        "another-document",
    )
    assert choose_control_document_corpus_scope_key(identity, (mismatched_legacy,)) == (
        identity.scope_key
    )
    with pytest.raises(ControlDocumentCorpusScopeError) as conflict:
        choose_control_document_corpus_scope_key(
            identity,
            ((identity.scope_key, "another-registration", "another-document"),),
        )
    assert conflict.value.code == "control_document_corpus_scope_conflict"


def test_exact_legacy_group_scope_is_reused_without_reassignment(
    environment: ControlEnvironment,
) -> None:
    identity = control_document_corpus_identity(
        group_id=environment.group_id,
        changed_repository_registration_id=environment.changed_repository_id,
        source_document_id=environment.source_document_id,
    )
    sessions = sessionmaker(environment.engine, expire_on_commit=False)
    with sessions.begin() as session:
        legacy = RetrievalCorpus(
            scope_key=identity.legacy_group_scope_key,
            changed_repository_id=environment.changed_repository_id,
            source_document_id=environment.source_document_id,
        )
        session.add(legacy)
        session.flush()
        legacy_corpus_id = legacy.corpus_id

    _continuation_id, outcome = _ingest(
        environment,
        ordinal=41,
        content="exact legacy corpus continuation\n",
    )
    assert outcome.corpus_id == legacy_corpus_id
    with sessions() as session:
        stored = session.get(RetrievalCorpus, legacy_corpus_id)
        assert stored is not None
        assert stored.scope_key == identity.legacy_group_scope_key
        assert stored.changed_repository_id == environment.changed_repository_id
        assert stored.source_document_id == environment.source_document_id


def test_mismatched_legacy_group_scope_is_never_mutated_or_reassigned(
    environment: ControlEnvironment,
) -> None:
    identity = control_document_corpus_identity(
        group_id=environment.group_id,
        changed_repository_registration_id=environment.changed_repository_id,
        source_document_id=environment.source_document_id,
    )
    sessions = sessionmaker(environment.engine, expire_on_commit=False)
    with sessions.begin() as session:
        legacy = RetrievalCorpus(
            scope_key=identity.legacy_group_scope_key,
            changed_repository_id="historical-changed-registration",
            source_document_id=str(uuid4()),
        )
        session.add(legacy)
        session.flush()
        legacy_corpus_id = legacy.corpus_id
        legacy_source_document_id = legacy.source_document_id

    _continuation_id, outcome = _ingest(
        environment,
        ordinal=42,
        content="new source-specific corpus\n",
    )
    assert outcome.corpus_id != legacy_corpus_id
    with sessions() as session:
        historical = session.get(RetrievalCorpus, legacy_corpus_id)
        current = session.get(RetrievalCorpus, outcome.corpus_id)
        assert historical is not None and current is not None
        assert historical.scope_key == identity.legacy_group_scope_key
        assert historical.changed_repository_id == "historical-changed-registration"
        assert historical.source_document_id == legacy_source_document_id
        assert current.scope_key == identity.scope_key
        assert current.changed_repository_id == environment.changed_repository_id
        assert current.source_document_id == environment.source_document_id


def test_source_specific_identity_conflict_has_typed_sanitized_failure(
    environment: ControlEnvironment,
) -> None:
    identity = control_document_corpus_identity(
        group_id=environment.group_id,
        changed_repository_registration_id=environment.changed_repository_id,
        source_document_id=environment.source_document_id,
    )
    sessions = sessionmaker(environment.engine, expire_on_commit=False)
    with sessions.begin() as session:
        session.add(
            RetrievalCorpus(
                scope_key=identity.scope_key,
                changed_repository_id="conflicting-registration",
                source_document_id=str(uuid4()),
            )
        )
    continuation_id = _stage_worker_snapshot(
        environment,
        content="benign conflict fixture\n",
        idempotency_key="opaque-source-scope-conflict-00000001",
    )
    with pytest.raises(ContinuationWorkerError) as conflict:
        _continuation_worker(environment).execute(
            identity=InternalContinuationWorkerIdentity.create("scope-conflict"),
            group_id=environment.group_id,
            continuation_job_id=continuation_id,
        )
    assert conflict.value.code == "control_document_corpus_scope_conflict"
    status = environment.service.continuation_status(
        _continuation_status_payload(
            environment,
            continuation_job_id=continuation_id,
        ),
        caller_user_id=environment.user_id,
    )
    assert status["state"] == "terminal_failed"
    assert status["error_code"] == "control_document_corpus_scope_conflict"
    assert status["result"]["corpus_ingestion_complete"] is False
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(RetrievalCorpusGeneration)
            ).scalar_one()
            == 0
        )


def test_two_sources_share_a_group_without_corpus_or_publication_collision(
    environment: ControlEnvironment,
) -> None:
    source_a = environment
    source_b = _add_source(environment, "b")
    job_a = _stage_worker_snapshot(
        source_a,
        content="first source A snapshot\n",
        idempotency_key="opaque-concurrent-source-a-00000001",
    )
    job_b = _stage_worker_snapshot(
        source_b,
        content="first source B snapshot\n",
        idempotency_key="opaque-concurrent-source-b-00000001",
    )

    def execute(item):
        selected, job_id, label = item
        return _continuation_worker(selected).execute(
            identity=InternalContinuationWorkerIdentity.create(label),
            group_id=selected.group_id,
            continuation_job_id=job_id,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_a, first_b = tuple(
            executor.map(
                execute,
                (
                    (source_a, job_a, "concurrent-source-a"),
                    (source_b, job_b, "concurrent-source-b"),
                ),
            )
        )
    assert first_a.corpus_id != first_b.corpus_id
    assert first_a.generation_id != first_b.generation_id

    update_job_a, update_a = _ingest(
        source_a,
        ordinal=3,
        content="updated source A snapshot\n",
    )
    replay_a = _continuation_worker(source_a).execute(
        identity=InternalContinuationWorkerIdentity.create("source-a-replay"),
        group_id=source_a.group_id,
        continuation_job_id=update_job_a,
    )
    assert replay_a == update_a
    assert update_a.corpus_id == first_a.corpus_id

    expected_a = control_document_corpus_identity(
        group_id=source_a.group_id,
        changed_repository_registration_id=source_a.changed_repository_id,
        source_document_id=source_a.source_document_id,
    )
    expected_b = control_document_corpus_identity(
        group_id=source_b.group_id,
        changed_repository_registration_id=source_b.changed_repository_id,
        source_document_id=source_b.source_document_id,
    )
    with environment.engine.connect() as connection:
        corpora = {
            row.corpus_id: row
            for row in connection.execute(
                select(RetrievalCorpus).order_by(RetrievalCorpus.corpus_id)
            )
        }
        generation_counts = {
            row.corpus_id: int(row.generation_count)
            for row in connection.execute(
                select(
                    RetrievalCorpusGeneration.corpus_id,
                    func.count(RetrievalCorpusGeneration.generation_id).label(
                        "generation_count"
                    ),
                ).group_by(RetrievalCorpusGeneration.corpus_id)
            )
        }
    assert len(corpora) == 2
    assert corpora[first_a.corpus_id].scope_key == expected_a.scope_key
    assert corpora[first_b.corpus_id].scope_key == expected_b.scope_key
    assert corpora[first_a.corpus_id].active_generation_id == update_a.generation_id
    assert corpora[first_b.corpus_id].active_generation_id == first_b.generation_id
    assert generation_counts == {first_a.corpus_id: 2, first_b.corpus_id: 1}

    index_a, extension_a, fingerprint_a = _publish_index(
        source_a,
        continuation_id=update_job_a,
        ordinal=11,
    )
    index_b, extension_b, fingerprint_b = _publish_index(
        source_b,
        continuation_id=job_b,
        ordinal=12,
    )
    assert index_a.index_id != index_b.index_id
    with environment.engine.connect() as connection:
        publications = tuple(
            connection.execute(
                select(RetrievalBaselineIndexPublication).order_by(
                    RetrievalBaselineIndexPublication.corpus_id
                )
            )
        )
    assert {(row.corpus_id, row.index_id) for row in publications} == {
        (first_a.corpus_id, index_a.index_id),
        (first_b.corpus_id, index_b.index_id),
    }

    final_job_a, final_generation_a = _ingest(
        source_a,
        ordinal=13,
        content="final source A snapshot\n",
    )
    final_index_a, extension_a, fingerprint_a = _publish_index(
        source_a,
        continuation_id=final_job_a,
        ordinal=14,
    )
    assert final_index_a.index_id != index_a.index_id
    with environment.engine.connect() as connection:
        active_a = connection.execute(
            select(RetrievalCorpus.active_generation_id).where(
                RetrievalCorpus.corpus_id == first_a.corpus_id
            )
        ).scalar_one()
        active_b = connection.execute(
            select(RetrievalCorpus.active_generation_id).where(
                RetrievalCorpus.corpus_id == first_b.corpus_id
            )
        ).scalar_one()
        publication_b = connection.execute(
            select(RetrievalBaselineIndexPublication.index_id).where(
                RetrievalBaselineIndexPublication.corpus_id == first_b.corpus_id
            )
        ).scalar_one()
    assert active_a == final_generation_a.generation_id
    assert active_b == first_b.generation_id
    assert publication_b == index_b.index_id

    failed_job_a = _stage_worker_snapshot(
        source_a,
        content="source A snapshot that must roll back\n",
        idempotency_key="opaque-failed-source-a-ingestion-000001",
    )

    def fail_before_success(stage: ContinuationWorkerStage) -> None:
        if stage is ContinuationWorkerStage.BEFORE_SUCCESS:
            raise RuntimeError("injected-safe-multi-source-failure")

    with pytest.raises(ContinuationWorkerError) as failed_ingestion:
        _continuation_worker(source_a, stage_hook=fail_before_success).execute(
            identity=InternalContinuationWorkerIdentity.create(
                "failed-source-a-ingestion"
            ),
            group_id=source_a.group_id,
            continuation_job_id=failed_job_a,
        )
    assert failed_ingestion.value.code == "corpus_ingestion_failed"
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(RetrievalCorpus.active_generation_id).where(
                    RetrievalCorpus.corpus_id == first_a.corpus_id
                )
            ).scalar_one()
            == final_generation_a.generation_id
        )
        assert (
            connection.execute(
                select(RetrievalCorpus.active_generation_id).where(
                    RetrievalCorpus.corpus_id == first_b.corpus_id
                )
            ).scalar_one()
            == first_b.generation_id
        )
        assert (
            connection.execute(
                select(RetrievalBaselineIndexPublication.index_id).where(
                    RetrievalBaselineIndexPublication.corpus_id == first_a.corpus_id
                )
            ).scalar_one()
            == final_index_a.index_id
        )
        assert (
            connection.execute(
                select(RetrievalBaselineIndexPublication.index_id).where(
                    RetrievalBaselineIndexPublication.corpus_id == first_b.corpus_id
                )
            ).scalar_one()
            == index_b.index_id
        )

    payload_a = _run_payload(
        source_a,
        index_result=final_index_a,
        extension=extension_a,
        index_fingerprint=fingerprint_a,
        ordinal=21,
    )
    payload_b = _run_payload(
        source_b,
        index_result=index_b,
        extension=extension_b,
        index_fingerprint=fingerprint_b,
        ordinal=22,
    )
    run_service = BaselineRunJobService(environment.engine, _keyring())
    before = 0
    with environment.engine.connect() as connection:
        before = int(
            connection.execute(
                select(func.count()).select_from(baseline_run_job)
            ).scalar_one()
        )
    cross_wired = dict(payload_a)
    cross_wired["request_id"] = str(uuid4())
    cross_wired["idempotency_key"] = "opaque-cross-wired-run-000000000001"
    cross_wired["source_document_id"] = source_b.source_document_id
    cross_wired["changed_repository_registration_id"] = source_b.changed_repository_id
    with pytest.raises(BaselineRunJobError):
        run_service.submit(
            parse_run_submission(cross_wired), caller_user_id=environment.user_id
        )
    with environment.engine.connect() as connection:
        assert (
            connection.execute(
                select(func.count()).select_from(baseline_run_job)
            ).scalar_one()
            == before
        )

    accepted_a = run_service.submit(
        parse_run_submission(payload_a), caller_user_id=environment.user_id
    )
    accepted_b = run_service.submit(
        parse_run_submission(payload_b), caller_user_id=environment.user_id
    )
    retriever_a = RecordingRetriever(_persistent(environment))
    retriever_b = RecordingRetriever(_persistent(environment))
    outcome_a = _executor(environment, retriever_a).execute(str(accepted_a["job_id"]))
    outcome_b = _executor(environment, retriever_b).execute(str(accepted_b["job_id"]))
    assert outcome_a.state == outcome_b.state == "references_persisted"
    assert retriever_a.requests[0].corpus_scope_key == expected_a.scope_key
    assert retriever_b.requests[0].corpus_scope_key == expected_b.scope_key
    assert retriever_a.results[0].corpus_id == first_a.corpus_id
    assert retriever_b.results[0].corpus_id == first_b.corpus_id

    with environment.engine.connect() as connection:
        before_cross_source = {
            table: int(
                connection.exec_driver_sql(f"SELECT count(*) FROM {table}").scalar_one()
            )
            for table in (
                "baseline_retrieval_run",
                "baseline_evidence_artifact",
                "baseline_selected_evidence",
                "reference",
            )
        }
    with pytest.raises(BaselineEvidencePersistenceError) as cross_source_evidence:
        BaselineEvidencePersistenceService(
            sessionmaker(environment.engine, expire_on_commit=False)
        ).persist(
            BaselineEvidencePersistenceCommand(
                group_id=source_b.group_id,
                source=ControlDocumentSource(
                    document_id=source_b.source_document_id,
                    control_job_id=str(accepted_b["job_id"]),
                    lease_token="opaque-cross-source-replay-lease-token",
                ),
                idempotency_key="opaque-cross-source-evidence-intent-000001",
                retrieval_result=retriever_a.results[0],
                caller_user_id=environment.user_id,
            )
        )
    assert cross_source_evidence.value.code == "corpus_source_mismatch"
    with environment.engine.connect() as connection:
        assert {
            table: int(
                connection.exec_driver_sql(f"SELECT count(*) FROM {table}").scalar_one()
            )
            for table in before_cross_source
        } == before_cross_source

    sessions = sessionmaker(environment.engine, expire_on_commit=False)
    generation_service = BaselineGenerationService(
        sessions, notifications_enabled=False
    )
    generated_a = generation_service.generate_control(
        str(accepted_a["job_id"]), CapturingProvider("source A finding")
    )
    generated_b = generation_service.generate_control(
        str(accepted_b["job_id"]),
        RawOutputProvider(_structured("no_findings", [])),
    )
    assert generated_a.state == generated_b.state == "feedback_persisted"
    assert len(generated_a.feedback_ids) == 1
    assert generated_b.feedback_ids == ()

    preview_service = BaselinePreviewService(sessions)
    preview_a = preview_service.load(
        BaselinePreviewCommand(
            caller_user_id=environment.user_id,
            request_id=str(uuid4()),
            group_id=environment.group_id,
            job_id=str(accepted_a["job_id"]),
        )
    )
    preview_b = preview_service.load(
        BaselinePreviewCommand(
            caller_user_id=environment.user_id,
            request_id=str(uuid4()),
            group_id=environment.group_id,
            job_id=str(accepted_b["job_id"]),
        )
    )
    assert preview_a.source.document_id == source_a.source_document_id
    assert preview_b.source.document_id == source_b.source_document_id
    assert preview_a.provenance.corpus.generation_id == final_index_a.generation_id
    assert preview_b.provenance.corpus.generation_id == index_b.generation_id
    with environment.engine.connect() as connection:
        persisted = tuple(
            connection.execute(
                select(
                    baseline_retrieval_run.c.run_id,
                    baseline_retrieval_run.c.corpus_id,
                    baseline_retrieval_run.c.corpus_scope_key,
                ).order_by(baseline_retrieval_run.c.run_id)
            )
        )
    assert {(row.corpus_id, row.corpus_scope_key) for row in persisted} == {
        (first_a.corpus_id, expected_a.scope_key),
        (first_b.corpus_id, expected_b.scope_key),
    }
