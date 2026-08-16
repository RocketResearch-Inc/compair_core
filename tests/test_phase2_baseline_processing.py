from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from conftest import REAL_SQLALCHEMY_TEXT as text
from test_baseline_evidence_persistence import (
    FixtureEmbeddingProvider,
    make_persistence_environment,
    persistence_counts,
)

from compair_core import api
from compair_core import db as core_db
from compair_core.compair import main, models, tasks
from compair_core.compair.retrieval import (
    BaselineProcessingStatus,
    ProcessingRunIdentityError,
    RetrievalError,
    RetrievalStatus,
    UnknownRetrievalEngineError,
    derive_baseline_persistence_idempotency_key,
    new_processing_run_key,
    processing_run_trace_id,
    validate_processing_run_key,
)
from compair_core.compair.retrieval.corpus import CorpusFileInput
from compair_core.compair.retrieval.evidence_persistence import (
    BaselineEvidencePersistenceService,
)
from compair_core.compair.retrieval.generation import BaselineGenerationService
from compair_core.compair.retrieval.indexing import (
    BaselineEmbeddingIdentity,
    BaselineIndexBuilder,
)
from compair_core.compair.retrieval.ingestion import (
    CorpusIngestionService,
    CorpusRepositoryInput,
    CorpusSnapshotInput,
)
from compair_core.compair.retrieval.persistent import PersistentBaselineV1Retriever
from compair_core.server.settings import Settings

QUERY = "alpha persistence query"
_ISOLATED_TEST_ENV = "COMPAIR_PHASE2_BASELINE_PROCESSING_ISOLATED"
_USE_ENVIRONMENT_GROUP = object()


def run_in_isolated_pytest_if_needed(request, tmp_path: Path) -> bool:
    """Run DB integration cases outside the legacy snapshot's module stubs."""

    if os.getenv(_ISOLATED_TEST_ENV) == "1":
        return False
    environment = os.environ.copy()
    environment[_ISOLATED_TEST_ENV] = "1"
    environment["COMPAIR_DB_DIR"] = str(tmp_path / "core-db")
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", request.node.nodeid],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return True


class _ForbiddenRuntime:
    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs
        raise AssertionError("baseline instantiated a legacy model/provider runtime")

    def __getattr__(self, name: str):
        raise AssertionError(f"baseline processing invoked forbidden runtime: {name}")


class _StatusRetriever:
    def __init__(self, template, status: RetrievalStatus) -> None:
        self.template = template
        self.status = status

    def retrieve(self, request):
        return replace(
            self.template,
            request_id=request.request_id,
            status=self.status,
            candidates=(),
            evidence=(),
            candidate_count=0,
            retrieved_count=0,
            filtered_count=0,
            duplicate_count=0,
            refill_count=0,
            evidence_characters=0,
            underfilled=True,
            error=RetrievalError(
                code=f"fixture_{self.status.value}",
                message="fixture terminal retrieval state",
            ),
            query_provenance=request.query_provenance,
        )


class _ForbiddenPersistenceService:
    def persist(self, command):
        del command
        raise AssertionError("a non-ok baseline result reached persistence")


class _ForbiddenRetriever:
    def retrieve(self, request):
        del request
        raise AssertionError("an unauthorized source reached baseline retrieval")


class _CapturingRetriever:
    def __init__(self, wrapped) -> None:
        self.wrapped = wrapped
        self.requests = []

    def retrieve(self, request):
        self.requests.append(request)
        return self.wrapped.retrieve(request)


class _FixtureGenerationProvider:
    provider = "fixture-generation"
    model = "fixture-reviewer"
    version = "fixture-reviewer-r1"
    supports_idempotency = False

    def generate(self, generation_input, *, idempotency_key):
        del idempotency_key
        assert [item.ordinal for item in generation_input.evidence] == list(
            range(1, len(generation_input.evidence) + 1)
        )
        return "fixture baseline finding"


class _RevokingRetriever:
    def __init__(self, wrapped, environment) -> None:
        self.wrapped = wrapped
        self.environment = environment

    def retrieve(self, request):
        result = self.wrapped.retrieve(request)
        with self.environment.engine.begin() as connection:
            connection.execute(
                text(
                    "DELETE FROM user_to_group WHERE user_id = :user_id "
                    "AND group_id = :group_id"
                ),
                {
                    "user_id": _source_user_id(self.environment),
                    "group_id": request.group_id,
                },
            )
        return result


def _environment(tmp_path: Path, name: str = "baseline-processing.db"):
    engine = core_db.create_engine(
        f"sqlite:///{tmp_path / name}",
        connect_args={"check_same_thread": False, "timeout": 15},
    )
    return make_persistence_environment(engine)


def _source_user_id(environment) -> str:
    with environment.engine.connect() as connection:
        return str(
            connection.execute(
                text(
                    "SELECT author_id FROM document WHERE document_id = :document_id"
                ),
                {"document_id": environment.source_document_id},
            ).scalar_one()
        )


def _runtime_components(environment):
    return (
        PersistentBaselineV1Retriever(
            environment.sessions,
            FixtureEmbeddingProvider(),
        ),
        BaselineEvidencePersistenceService(environment.sessions),
    )


def _install_actual_task_path(
    monkeypatch,
    environment,
    *,
    retriever=None,
    persistence_service=None,
    chunker=None,
):
    generation = Mock(side_effect=AssertionError("baseline invoked generation"))
    embedding = Mock(side_effect=AssertionError("baseline invoked legacy embeddings"))
    events: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        tasks,
        "_lazy_components",
        lambda: (
            environment.sessions,
            _ForbiddenRuntime,
            _ForbiddenRuntime,
            lambda name, **values: events.append((name, values)),
            main.process_document,
            models.Document,
            models.User,
            lambda value: [],
            lambda value: value,
        ),
    )
    monkeypatch.setattr(main, "get_history", lambda *args: SimpleNamespace(deleted=[]))
    monkeypatch.setattr(
        main,
        "chunk_text_with_mode",
        chunker or (lambda value, chunk_mode=None: [value]),
    )
    monkeypatch.setattr(main, "count_tokens", lambda value: 200)
    monkeypatch.setattr(
        main,
        "detect_significant_edits",
        lambda **kwargs: list(range(len(kwargs["new_chunks"]))),
    )
    monkeypatch.setattr(main, "extract_topic_tags", lambda value: [])
    monkeypatch.setattr(main, "create_embedding", embedding)
    monkeypatch.setattr(main, "create_embeddings", embedding)
    monkeypatch.setattr(main, "get_feedback", generation)
    if retriever is None or persistence_service is None:
        default_retriever, default_service = _runtime_components(environment)
        retriever = retriever or default_retriever
        persistence_service = persistence_service or default_service
    monkeypatch.setattr(
        main,
        "_baseline_runtime_components",
        lambda: (
            retriever,
            persistence_service,
            BaselineGenerationService(environment.sessions),
            _FixtureGenerationProvider(),
        ),
    )
    return generation, embedding, events


def _run_task(
    environment,
    processing_run_key: str,
    *,
    retrieval_query: str | None = QUERY,
    group_id: str | None | object = _USE_ENVIRONMENT_GROUP,
    doc_text: str = "authoritative source chunk",
):
    return tasks.process_document_task(
        _source_user_id(environment),
        environment.source_document_id,
        doc_text,
        generate_feedback=True,
        retrieval_query=retrieval_query,
        retrieval_engine="baseline_v1",
        processing_run_key=processing_run_key,
        group_id=(
            environment.group_id
            if group_id is _USE_ENVIRONMENT_GROUP
            else group_id
        ),
    )


def _baseline_reference_rows(environment, group_id: str | None = None):
    with environment.engine.connect() as connection:
        return connection.execute(
            text(
                "SELECT s.ordinal, a.relative_path, r.reference_type, "
                "r.reference_chunk_id, r.reference_document_id "
                "FROM baseline_selected_evidence s "
                "JOIN baseline_retrieval_run br ON br.run_id = s.run_id "
                "JOIN baseline_evidence_artifact a "
                "ON a.artifact_id = s.artifact_id "
                "JOIN reference r "
                "ON r.baseline_selected_evidence_id = s.selected_evidence_id "
                + ("WHERE br.group_id = :group_id " if group_id is not None else "")
                + "ORDER BY br.run_id, s.ordinal"
            ),
            ({"group_id": group_id} if group_id is not None else {}),
        ).all()


def _add_authorized_group_with_publication(
    environment,
    *,
    label: str,
) -> tuple[str, tuple[str, ...]]:
    """Create a second toy group/publication for one multiply-scoped source."""

    group_id = str(uuid4())
    user_id = _source_user_id(environment)
    now = datetime.now(timezone.utc)
    with environment.engine.begin() as connection:
        connection.execute(
            text(
                'INSERT INTO "group" '
                "(group_id, name, datetime_created, category, description, visibility) "
                "VALUES (:group_id, :name, :now, 'Other', '', 'private')"
            ),
            {"group_id": group_id, "name": f"Group {label}", "now": now},
        )
        connection.execute(
            text(
                "INSERT INTO user_to_group (user_id, group_id) "
                "VALUES (:user_id, :group_id)"
            ),
            {"user_id": user_id, "group_id": group_id},
        )
        connection.execute(
            text(
                "INSERT INTO document_to_group (document_id, group_id) "
                "VALUES (:document_id, :group_id)"
            ),
            {"document_id": environment.source_document_id, "group_id": group_id},
        )

    repository_id = f"repository-peer-{label}"
    repository_name = f"peer-{label}"
    paths = tuple(f"scope_{label}/file_{ordinal}.py" for ordinal in range(1, 5))
    files = tuple(
        CorpusFileInput.supported_text(
            repository_id=repository_id,
            repository_name=repository_name,
            relative_path=path,
            content=f"alpha {label} evidence {ordinal}\nvalue = {ordinal}\n",
        )
        for ordinal, path in enumerate(paths, start=1)
    )
    snapshot = CorpusSnapshotInput.create(
        scope_key=f"group:{group_id}",
        generation_version=f"generation-{label}",
        changed_repository=CorpusRepositoryInput(
            repository_id=f"repository-changed-{label}",
            repository_name=f"changed-{label}",
            expected_file_count=0,
            repository_revision=f"changed-revision-{label}",
            document_id=environment.source_document_id,
            document_revision=f"changed-document-revision-{label}",
        ),
        sibling_repositories=(
            CorpusRepositoryInput(
                repository_id=repository_id,
                repository_name=repository_name,
                expected_file_count=len(files),
                repository_revision=f"peer-revision-{label}",
            ),
        ),
        files=files,
        producer_id="trusted-processing-fixture",
        producer_version="1",
        snapshot_id=f"trusted-snapshot-{label}",
    )
    corpus = CorpusIngestionService(environment.sessions).ingest(snapshot)
    provider = FixtureEmbeddingProvider()
    identity = BaselineEmbeddingIdentity(
        provider=provider.provider,
        model=provider.model,
        revision=provider.revision,
        dimension=provider.dimension,
        fingerprint=provider.fingerprint,
    )
    BaselineIndexBuilder(environment.sessions).build(
        generation_id=corpus.generation_id,
        index_version=f"index-{label}",
        embedding=identity,
        provider=provider,
    )
    return group_id, paths


def test_actual_baseline_task_path_persists_order_and_generates_without_notifications(
    tmp_path: Path,
    monkeypatch,
    request,
) -> None:
    if run_in_isolated_pytest_if_needed(request, tmp_path):
        return
    environment = _environment(tmp_path)
    try:
        generation, embedding, events = _install_actual_task_path(
            monkeypatch, environment
        )
        parent_key = new_processing_run_key()

        task_result = _run_task(environment, parent_key)

        outcome = task_result["baseline_processing"]["outcomes"][0]
        assert outcome["status"] == BaselineProcessingStatus.FEEDBACK_PERSISTED.value
        assert outcome["retrieval_status"] == "ok"
        assert outcome["generation_bypassed"] is False
        assert outcome["generation_state"] == "succeeded"
        assert outcome["feedback_count"] == 1
        assert outcome["selected_reference_count"] == 4
        assert outcome["idempotent_replay"] is False
        assert outcome["group_id"] == environment.group_id
        assert outcome["parent_run_trace_id"] == processing_run_trace_id(
            parent_key,
            environment.group_id,
        )
        assert task_result["baseline_processing"]["group_id"] == environment.group_id
        assert outcome["retrieval_query_length"] == len(QUERY)
        assert outcome["retrieval_query_origin"] == "explicit"
        assert parent_key not in repr(task_result)
        assert QUERY not in repr(task_result)

        rows = _baseline_reference_rows(environment)
        assert [row.ordinal for row in rows] == [1, 2, 3, 4]
        assert [row.relative_path for row in rows] == [
            item.relative_path for item in environment.result.evidence
        ]
        assert all(row.reference_type == "baseline_file" for row in rows)
        assert all(row.reference_chunk_id is None for row in rows)
        assert all(row.reference_document_id is None for row in rows)
        assert persistence_counts(environment.engine) == (1, 4, 4, 4, 1)
        with environment.engine.connect() as connection:
            assert connection.execute(text("SELECT count(*) FROM notification_event")).scalar_one() == 0
        generation.assert_not_called()
        embedding.assert_not_called()
        assert QUERY not in repr(events)
    finally:
        environment.engine.dispose()


def test_task_retry_reuses_per_chunk_intent_without_duplicates(
    tmp_path: Path,
    monkeypatch,
    request,
) -> None:
    if run_in_isolated_pytest_if_needed(request, tmp_path):
        return
    environment = _environment(tmp_path, "baseline-retry.db")
    try:
        _install_actual_task_path(monkeypatch, environment)
        parent_key = new_processing_run_key()
        expected_intent_key = derive_baseline_persistence_idempotency_key(
            parent_key,
            environment.group_id,
            environment.source_chunk_id,
        )

        first = _run_task(environment, parent_key)
        replay = _run_task(environment, parent_key)

        assert first["baseline_processing"]["outcomes"][0]["idempotent_replay"] is False
        assert replay["baseline_processing"]["outcomes"][0]["idempotent_replay"] is True
        assert first["baseline_processing"]["group_id"] == environment.group_id
        assert replay["baseline_processing"]["group_id"] == environment.group_id
        assert first["baseline_processing"]["parent_run_trace_id"] == (
            processing_run_trace_id(parent_key, environment.group_id)
        )
        assert replay["baseline_processing"]["parent_run_trace_id"] == (
            processing_run_trace_id(parent_key, environment.group_id)
        )
        assert persistence_counts(environment.engine) == (1, 4, 4, 4, 1)
        with environment.engine.connect() as connection:
            durable_key = connection.execute(
                text("SELECT idempotency_key FROM baseline_retrieval_run")
            ).scalar_one()
        assert durable_key == expected_intent_key
        assert parent_key not in durable_key
    finally:
        environment.engine.dispose()


@pytest.mark.parametrize("status", [RetrievalStatus.INSUFFICIENT, RetrievalStatus.ERROR])
def test_non_ok_baseline_result_has_zero_writes_and_no_generation(
    tmp_path: Path,
    monkeypatch,
    request,
    status: RetrievalStatus,
) -> None:
    if run_in_isolated_pytest_if_needed(request, tmp_path):
        return
    environment = _environment(tmp_path, f"baseline-{status.value}.db")
    try:
        generation, embedding, _events = _install_actual_task_path(
            monkeypatch,
            environment,
            retriever=_StatusRetriever(environment.result, status),
            persistence_service=_ForbiddenPersistenceService(),
        )

        result = _run_task(environment, new_processing_run_key())

        outcome = result["baseline_processing"]["outcomes"][0]
        assert outcome["status"] == status.value
        assert outcome["selected_reference_count"] == 0
        assert outcome["generation_bypassed"] is True
        assert persistence_counts(environment.engine) == (0, 0, 0, 0, 0)
        generation.assert_not_called()
        embedding.assert_not_called()
    finally:
        environment.engine.dispose()


def test_missing_explicit_query_uses_real_fail_closed_retriever(
    tmp_path: Path,
    monkeypatch,
    request,
) -> None:
    if run_in_isolated_pytest_if_needed(request, tmp_path):
        return
    environment = _environment(tmp_path, "baseline-missing-query.db")
    try:
        generation, embedding, _events = _install_actual_task_path(
            monkeypatch, environment
        )

        result = _run_task(
            environment,
            new_processing_run_key(),
            retrieval_query=None,
        )

        outcome = result["baseline_processing"]["outcomes"][0]
        assert outcome["status"] == "insufficient"
        assert outcome["retrieval_status"] == "insufficient"
        assert outcome["error_code"] == "explicit_retrieval_query_absent"
        assert outcome["retrieval_query_origin"] == "absent"
        assert persistence_counts(environment.engine) == (0, 0, 0, 0, 0)
        generation.assert_not_called()
        embedding.assert_not_called()
    finally:
        environment.engine.dispose()


def test_missing_explicit_group_returns_structured_zero_write_outcome(
    tmp_path: Path,
    monkeypatch,
    request,
) -> None:
    if run_in_isolated_pytest_if_needed(request, tmp_path):
        return
    environment = _environment(tmp_path, "baseline-missing-group.db")
    try:
        generation, embedding, _events = _install_actual_task_path(
            monkeypatch,
            environment,
            retriever=_ForbiddenRetriever(),
            persistence_service=_ForbiddenPersistenceService(),
        )
        parent_key = new_processing_run_key()

        result = _run_task(environment, parent_key, group_id=None)

        processing = result["baseline_processing"]
        assert processing["schema_version"] == "baseline-document-processing.v3"
        assert processing["status"] == "error"
        assert processing["error_code"] == "explicit_group_id_absent"
        assert processing["group_id"] is None
        assert processing["parent_run_trace_id"] == processing_run_trace_id(
            parent_key,
            None,
        )
        assert processing["retrieval_query_length"] == len(QUERY)
        assert processing["retrieval_query_origin"] == "explicit"
        assert QUERY not in repr(processing)
        assert processing["outcomes"] == []
        assert persistence_counts(environment.engine) == (0, 0, 0, 0, 0)
        generation.assert_not_called()
        embedding.assert_not_called()
    finally:
        environment.engine.dispose()


def test_database_authorization_mismatch_fails_before_retrieval(
    tmp_path: Path,
    monkeypatch,
    request,
) -> None:
    if run_in_isolated_pytest_if_needed(request, tmp_path):
        return
    environment = _environment(tmp_path, "baseline-auth.db")
    try:
        with environment.engine.begin() as connection:
            connection.execute(
                text(
                    "DELETE FROM document_to_group "
                    "WHERE document_id = :document_id"
                ),
                {"document_id": environment.source_document_id},
            )
        forbidden_retriever = _ForbiddenRetriever()
        generation, embedding, _events = _install_actual_task_path(
            monkeypatch,
            environment,
            retriever=forbidden_retriever,
            persistence_service=_ForbiddenPersistenceService(),
        )

        result = _run_task(environment, new_processing_run_key())

        processing = result["baseline_processing"]
        assert processing["status"] == "error"
        assert processing["error_code"] == "source_group_unauthorized"
        assert processing["group_id"] == environment.group_id
        assert processing["outcomes"] == []
        assert persistence_counts(environment.engine) == (0, 0, 0, 0, 0)
        generation.assert_not_called()
        embedding.assert_not_called()
    finally:
        environment.engine.dispose()


def test_caller_group_membership_is_rechecked_before_retrieval(
    tmp_path: Path,
    monkeypatch,
    request,
) -> None:
    if run_in_isolated_pytest_if_needed(request, tmp_path):
        return
    environment = _environment(tmp_path, "baseline-caller-auth.db")
    try:
        user_id = _source_user_id(environment)
        with environment.engine.begin() as connection:
            connection.execute(
                text(
                    "DELETE FROM user_to_group WHERE user_id = :user_id "
                    "AND group_id = :group_id"
                ),
                {"user_id": user_id, "group_id": environment.group_id},
            )
        generation, embedding, _events = _install_actual_task_path(
            monkeypatch,
            environment,
            retriever=_ForbiddenRetriever(),
            persistence_service=_ForbiddenPersistenceService(),
        )

        result = _run_task(environment, new_processing_run_key())

        processing = result["baseline_processing"]
        assert processing["status"] == "error"
        assert processing["group_id"] == environment.group_id
        assert processing["error_code"] == "caller_group_unauthorized"
        assert processing["outcomes"] == []
        assert persistence_counts(environment.engine) == (0, 0, 0, 0, 0)
        generation.assert_not_called()
        embedding.assert_not_called()
    finally:
        environment.engine.dispose()


def test_document_in_two_groups_selects_only_explicit_group_publication(
    tmp_path: Path,
    monkeypatch,
    request,
) -> None:
    if run_in_isolated_pytest_if_needed(request, tmp_path):
        return
    environment = _environment(tmp_path, "baseline-two-groups.db")
    try:
        group_b, group_b_paths = _add_authorized_group_with_publication(
            environment,
            label="b",
        )
        capturing = _CapturingRetriever(_runtime_components(environment)[0])
        _install_actual_task_path(
            monkeypatch,
            environment,
            retriever=capturing,
            persistence_service=BaselineEvidencePersistenceService(
                environment.sessions
            ),
        )

        parent_key = new_processing_run_key()
        result_a = _run_task(
            environment,
            parent_key,
            group_id=environment.group_id,
        )
        result_b = _run_task(
            environment,
            parent_key,
            group_id=group_b,
        )

        outcome_a = result_a["baseline_processing"]["outcomes"][0]
        outcome_b = result_b["baseline_processing"]["outcomes"][0]
        assert outcome_a["group_id"] == environment.group_id
        assert outcome_b["group_id"] == group_b
        assert outcome_a["status"] == "feedback_persisted"
        assert outcome_b["status"] == "feedback_persisted"
        assert outcome_a["parent_run_trace_id"] == processing_run_trace_id(
            parent_key,
            environment.group_id,
        )
        assert outcome_b["parent_run_trace_id"] == processing_run_trace_id(
            parent_key,
            group_b,
        )
        assert outcome_a["parent_run_trace_id"] != outcome_b["parent_run_trace_id"]
        assert [request.group_id for request in capturing.requests] == [
            environment.group_id,
            group_b,
        ]
        assert [request.corpus_scope_key for request in capturing.requests] == [
            f"group:{environment.group_id}",
            f"group:{group_b}",
        ]
        assert [row.relative_path for row in _baseline_reference_rows(
            environment,
            environment.group_id,
        )] == [item.relative_path for item in environment.result.evidence]
        assert [
            row.relative_path
            for row in _baseline_reference_rows(environment, group_b)
        ] == list(group_b_paths)
        with environment.engine.connect() as connection:
            run_rows = connection.execute(
                text(
                    "SELECT group_id, idempotency_key "
                    "FROM baseline_retrieval_run"
                )
            ).all()
            run_groups = set(
                row.group_id for row in run_rows
            )
        assert run_groups == {environment.group_id, group_b}
        assert len({row.idempotency_key for row in run_rows}) == 2
        assert persistence_counts(environment.engine) == (2, 8, 8, 8, 2)
    finally:
        environment.engine.dispose()


def test_group_without_its_own_corpus_does_not_reuse_another_groups_publication(
    tmp_path: Path,
    monkeypatch,
    request,
) -> None:
    if run_in_isolated_pytest_if_needed(request, tmp_path):
        return
    environment = _environment(tmp_path, "baseline-corpus-scope.db")
    try:
        other_group_id = str(uuid4())
        user_id = _source_user_id(environment)
        now = datetime.now(timezone.utc)
        with environment.engine.begin() as connection:
            connection.execute(
                text(
                    'INSERT INTO "group" '
                    "(group_id, name, datetime_created, category, description, visibility) "
                    "VALUES (:group_id, 'No corpus', :now, 'Other', '', 'private')"
                ),
                {"group_id": other_group_id, "now": now},
            )
            connection.execute(
                text(
                    "INSERT INTO user_to_group (user_id, group_id) "
                    "VALUES (:user_id, :group_id)"
                ),
                {"user_id": user_id, "group_id": other_group_id},
            )
            connection.execute(
                text(
                    "INSERT INTO document_to_group (document_id, group_id) "
                    "VALUES (:document_id, :group_id)"
                ),
                {
                    "document_id": environment.source_document_id,
                    "group_id": other_group_id,
                },
            )
        capturing = _CapturingRetriever(_runtime_components(environment)[0])
        generation, embedding, _events = _install_actual_task_path(
            monkeypatch,
            environment,
            retriever=capturing,
            persistence_service=_ForbiddenPersistenceService(),
        )

        result = _run_task(
            environment,
            new_processing_run_key(),
            group_id=other_group_id,
        )

        outcome = result["baseline_processing"]["outcomes"][0]
        assert outcome["group_id"] == other_group_id
        assert outcome["status"] in {"insufficient", "error"}
        assert outcome["selected_reference_count"] == 0
        assert capturing.requests[0].corpus_scope_key == f"group:{other_group_id}"
        assert persistence_counts(environment.engine) == (0, 0, 0, 0, 0)
        generation.assert_not_called()
        embedding.assert_not_called()
    finally:
        environment.engine.dispose()


def test_retry_after_group_authorization_deletion_writes_nothing_new(
    tmp_path: Path,
    monkeypatch,
    request,
) -> None:
    if run_in_isolated_pytest_if_needed(request, tmp_path):
        return
    environment = _environment(tmp_path, "baseline-retry-deleted-group-auth.db")
    try:
        _install_actual_task_path(monkeypatch, environment)
        parent_key = new_processing_run_key()
        first = _run_task(environment, parent_key)
        before = persistence_counts(environment.engine)
        with environment.engine.begin() as connection:
            connection.execute(
                text(
                    "DELETE FROM user_to_group WHERE user_id = :user_id "
                    "AND group_id = :group_id"
                ),
                {
                    "user_id": _source_user_id(environment),
                    "group_id": environment.group_id,
                },
            )

        retry = _run_task(environment, parent_key)

        assert first["baseline_processing"]["outcomes"][0]["status"] == (
            "feedback_persisted"
        )
        retry_outcome = retry["baseline_processing"]
        assert retry_outcome["status"] == "error"
        assert retry_outcome["error_code"] == "caller_group_unauthorized"
        assert retry_outcome["group_id"] == environment.group_id
        assert retry_outcome["outcomes"] == []
        assert persistence_counts(environment.engine) == before
    finally:
        environment.engine.dispose()


def test_retry_after_selected_group_deletion_returns_structured_zero_write_outcome(
    tmp_path: Path,
    monkeypatch,
    request,
) -> None:
    if run_in_isolated_pytest_if_needed(request, tmp_path):
        return
    environment = _environment(tmp_path, "baseline-retry-deleted-group.db")
    try:
        _install_actual_task_path(monkeypatch, environment)
        parent_key = new_processing_run_key()
        first = _run_task(environment, parent_key)
        assert first["baseline_processing"]["status"] == "feedback_persisted"
        with environment.engine.begin() as connection:
            connection.execute(
                text('DELETE FROM "group" WHERE group_id = :group_id'),
                {"group_id": environment.group_id},
            )

        retry = _run_task(environment, parent_key)

        processing = retry["baseline_processing"]
        assert processing["status"] == "error"
        assert processing["error_code"] == "selected_group_absent"
        assert processing["group_id"] == environment.group_id
        assert processing["outcomes"] == []
        assert persistence_counts(environment.engine) == (0, 0, 0, 0, 0)
    finally:
        environment.engine.dispose()


def test_membership_revoked_between_retrieval_and_persistence_fails_transactionally(
    tmp_path: Path,
    monkeypatch,
    request,
) -> None:
    if run_in_isolated_pytest_if_needed(request, tmp_path):
        return
    environment = _environment(tmp_path, "baseline-midflight-revocation.db")
    try:
        retriever = _RevokingRetriever(_runtime_components(environment)[0], environment)
        _install_actual_task_path(
            monkeypatch,
            environment,
            retriever=retriever,
            persistence_service=BaselineEvidencePersistenceService(
                environment.sessions
            ),
        )

        result = _run_task(environment, new_processing_run_key())

        outcome = result["baseline_processing"]["outcomes"][0]
        assert outcome["status"] == "error"
        assert outcome["error_code"] == "caller_unauthorized"
        assert outcome["group_id"] == environment.group_id
        assert persistence_counts(environment.engine) == (0, 0, 0, 0, 0)
    finally:
        environment.engine.dispose()


def test_document_outcome_preserves_actual_per_chunk_order(
    tmp_path: Path,
    monkeypatch,
    request,
) -> None:
    if run_in_isolated_pytest_if_needed(request, tmp_path):
        return
    environment = _environment(tmp_path, "baseline-ordered-chunks.db")
    separator = "\n<<<TEST-CHUNK>>>\n"
    try:
        _install_actual_task_path(
            monkeypatch,
            environment,
            chunker=lambda value, chunk_mode=None: value.split(separator),
        )
        combined = f"authoritative source chunk{separator}second source chunk"

        result = _run_task(
            environment,
            new_processing_run_key(),
            doc_text=combined,
        )

        outcomes = result["baseline_processing"]["outcomes"]
        with environment.engine.connect() as connection:
            chunk_ids_by_content = dict(
                connection.execute(
                    text(
                        "SELECT content, chunk_id FROM chunk "
                        "WHERE document_id = :document_id "
                        "AND content IN ('authoritative source chunk', 'second source chunk')"
                    ),
                    {"document_id": environment.source_document_id},
                ).all()
            )
        assert [outcome["source_chunk_id"] for outcome in outcomes] == [
            chunk_ids_by_content["authoritative source chunk"],
            chunk_ids_by_content["second source chunk"],
        ]
        assert all(outcome["group_id"] == environment.group_id for outcome in outcomes)
        assert all(outcome["status"] == "feedback_persisted" for outcome in outcomes)
        assert [outcome["selected_reference_count"] for outcome in outcomes] == [4, 4]
        assert persistence_counts(environment.engine) == (2, 4, 8, 8, 2)
    finally:
        environment.engine.dispose()


def test_api_dispatch_preserves_parent_key_and_engine_across_task_retry_envelope(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class Conf:
        broker_url = "rediss://worker:secret@redis.example/0?ssl_cert_reqs=required"
        broker_use_ssl = False
        result_extended = False
        task_always_eager = False
        task_protocol = 2

    class Task:
        app = SimpleNamespace(conf=Conf())

        @staticmethod
        def apply_async(**options):
            captured.update(options)
            return SimpleNamespace(id="baseline-task")

    monkeypatch.setattr(api, "process_document_celery", Task())
    parent_key = new_processing_run_key()

    result = api._dispatch_process_document_task(
        "user",
        "document",
        "body",
        True,
        retrieval_query=QUERY,
        retrieval_engine="baseline_v1",
        processing_run_key=parent_key,
        group_id="group-authoritative",
    )

    assert result.id == "baseline-task"
    assert captured["kwargs"]["processing_run_key"] == parent_key
    assert captured["kwargs"]["retrieval_engine"] == "baseline_v1"
    assert captured["kwargs"]["retrieval_query"] == QUERY
    assert captured["kwargs"]["group_id"] == "group-authoritative"
    assert parent_key not in captured["kwargsrepr"]
    assert QUERY not in captured["kwargsrepr"]


def test_api_process_request_threads_explicit_group_to_task(monkeypatch) -> None:
    group_id = str(uuid4())
    dispatched: dict[str, object] = {}

    async def read_payload(request):
        del request
        return {
            "doc_id": "document",
            "doc_text": "body",
            "retrieval_query": QUERY,
            "group_id": group_id,
        }

    class Query:
        def filter(self, *args, **kwargs):
            del args, kwargs
            return self

        @staticmethod
        def first():
            return SimpleNamespace(author_id="user")

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args

        def query(self, model):
            del model
            return Query()

    monkeypatch.setattr(api, "_read_process_doc_payload", read_payload)
    monkeypatch.setattr(
        api,
        "_stage_process_doc_payload",
        lambda **kwargs: "protected-payload-key",
    )
    monkeypatch.setattr(api.compair, "Session", Session)
    monkeypatch.setattr(
        api,
        "_dispatch_process_document_task",
        lambda **kwargs: dispatched.update(kwargs)
        or SimpleNamespace(id="baseline-task"),
    )
    monkeypatch.setattr(api, "log_event", lambda *args, **kwargs: None)

    result = asyncio.run(
        api.process_doc(
            request=object(),
            current_user=SimpleNamespace(
                user_id="user",
                status="active",
            ),
            analytics=SimpleNamespace(track=lambda *args: None),
            storage=object(),
            settings=SimpleNamespace(
                edition="core",
                retrieval_engine="baseline_v1",
                retrieval_query_allow_insecure_local_transport=True,
            ),
        )
    )

    assert result == {"task_id": "baseline-task"}
    assert dispatched["group_id"] == group_id
    assert dispatched["retrieval_engine"] == "baseline_v1"
    assert validate_processing_run_key(dispatched["processing_run_key"])


@pytest.mark.parametrize("engine_name", ["", "unknown", " baseline_v1"])
def test_invalid_engine_configuration_fails_without_dispatch_or_fallback(
    monkeypatch,
    engine_name: str,
) -> None:
    task = Mock()
    monkeypatch.setattr(api, "process_document_celery", task)

    with pytest.raises(UnknownRetrievalEngineError):
        api._dispatch_process_document_task(
            "user",
            "document",
            "body",
            True,
            retrieval_engine=engine_name,
        )

    task.delay.assert_not_called()
    task.apply_async.assert_not_called()


def test_engine_and_processing_identity_defaults_are_explicit() -> None:
    assert Settings.model_fields["retrieval_engine"].default == "legacy"
    with pytest.raises(ProcessingRunIdentityError):
        processing_run_trace_id("")
