from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import Engine, select, text
from sqlalchemy.exc import IntegrityError

from compair_core import db as core_db
from compair_core.baseline_control_plane_schema import (
    BASELINE_RUN_WORKER_CONTRACT_VERSION,
    BASELINE_RUN_WORKER_SERVICE_ID,
    baseline_run_job,
    baseline_run_payload,
    compatible_index_job,
    control_job,
    repository_approval,
    repository_registration,
    snapshot_continuation_job,
    snapshot_staging,
)
from compair_core.compair import models
from compair_core.compair.retrieval.control_plane_v2 import (
    PROTOCOL_V2_SHA256,
    PROTOCOL_V2_VERSION,
)
from compair_core.compair.retrieval.corpus import (
    CorpusFileInput,
    RetrievalBaselineIndexPublication,
    ensure_retrieval_corpus_schema,
)
from compair_core.compair.retrieval.evidence_persistence import (
    BaselineEvidencePersistenceCommand,
    BaselineEvidencePersistenceError,
    BaselineEvidencePersistenceService,
    ControlDocumentSource,
    LegacyChunkSource,
    PersistenceWriteStage,
)
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
from compair_core.compair.retrieval.types import (
    RetrievalQueryOrigin,
    RetrievalRequest,
    RetrievalResult,
    RetrievalStatus,
)
from compair_core.schema_migrations import run_schema_migrations


class FixtureEmbeddingProvider:
    provider = "fixture-baseline-adapter"
    model = "fixture/bge-small"
    revision = "fixture-revision-1"
    dimension = 2
    fingerprint = hashlib.sha256(b"fixture/bge-small@fixture-revision-1").hexdigest()

    def embed(self, texts):
        vectors = []
        for ordinal, value in enumerate(texts):
            if value.startswith("Repository file: "):
                vectors.append((1.0, float(ordinal + 1)))
            else:
                vectors.append((1.0, 0.0))
        return vectors


@dataclass(frozen=True)
class PersistenceEnvironment:
    engine: Engine
    sessions: object
    group_id: str
    source_document_id: str
    source_chunk_id: str
    peer_document_id: str
    peer_chunk_id: str
    result: RetrievalResult

    def command(self, key: str = "caller-retry-token-1"):
        return BaselineEvidencePersistenceCommand(
            group_id=self.group_id,
            source=LegacyChunkSource(
                document_id=self.source_document_id,
                chunk_id=self.source_chunk_id,
            ),
            idempotency_key=key,
            retrieval_result=self.result,
        )


def _seed_core_scope(engine: Engine) -> tuple[str, str, str, str, str]:
    group_id = str(uuid4())
    user_id = str(uuid4())
    source_document_id = str(uuid4())
    source_chunk_id = str(uuid4())
    peer_document_id = str(uuid4())
    peer_chunk_id = str(uuid4())
    now = datetime.now(timezone.utc)
    with engine.begin() as connection:
        connection.execute(
            text(
                'INSERT INTO "user" '
                "(user_id, username, name, datetime_registered, password_hash, "
                "password_salt, status, include_own_documents_in_feedback, "
                "default_publish, preferred_feedback_length, hide_affiliations) "
                "VALUES (:user_id, :username, :name, :now, 'hash', 'salt', "
                "'active', false, true, 'Brief', false)"
            ),
            {
                "user_id": user_id,
                "username": f"user-{user_id}",
                "name": "Persistence Test User",
                "now": now,
            },
        )
        connection.execute(
            text(
                'INSERT INTO "group" '
                "(group_id, name, datetime_created, category, description, visibility) "
                "VALUES (:group_id, 'Evidence test', :now, 'Other', '', 'private')"
            ),
            {"group_id": group_id, "now": now},
        )
        connection.execute(
            text(
                "INSERT INTO user_to_group (user_id, group_id) "
                "VALUES (:user_id, :group_id)"
            ),
            {"user_id": user_id, "group_id": group_id},
        )
        for document_id, title in (
            (source_document_id, "Changed repository"),
            (peer_document_id, "Legacy peer"),
        ):
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
                    "user_id": user_id,
                    "title": title,
                    "content": f"content for {title}",
                    "now": now,
                },
            )
            connection.execute(
                text(
                    "INSERT INTO document_to_group (document_id, group_id) "
                    "VALUES (:document_id, :group_id)"
                ),
                {"document_id": document_id, "group_id": group_id},
            )
        for chunk_id, document_id, content in (
            (source_chunk_id, source_document_id, "authoritative source chunk"),
            (peer_chunk_id, peer_document_id, "legacy peer chunk"),
        ):
            connection.execute(
                text(
                    "INSERT INTO chunk "
                    "(chunk_id, hash, content, document_id, note_id, chunk_type) "
                    "VALUES (:chunk_id, :hash, :content, :document_id, NULL, 'document')"
                ),
                {
                    "chunk_id": chunk_id,
                    "hash": hashlib.sha256(content.encode()).hexdigest(),
                    "content": content,
                    "document_id": document_id,
                },
            )
        connection.execute(
            text(
                "INSERT INTO reference "
                "(reference_id, source_chunk_id, reference_chunk_id, "
                "reference_document_id, reference_note_id, reference_type, "
                "baseline_selected_evidence_id) "
                "VALUES (:reference_id, :source, :peer, :peer_document, NULL, "
                "'document', NULL)"
            ),
            {
                "reference_id": "legacy-reference-" + uuid4().hex[:16],
                "source": source_chunk_id,
                "peer": peer_chunk_id,
                "peer_document": peer_document_id,
            },
        )
    return (
        group_id,
        source_document_id,
        source_chunk_id,
        peer_document_id,
        peer_chunk_id,
    )


def make_persistence_environment(engine: Engine) -> PersistenceEnvironment:
    models.Base.metadata.create_all(engine)
    ensure_retrieval_corpus_schema(engine)
    run_schema_migrations(engine)
    sessions = core_db.sessionmaker(engine, expire_on_commit=False)
    (
        group_id,
        source_document_id,
        source_chunk_id,
        peer_document_id,
        peer_chunk_id,
    ) = _seed_core_scope(engine)

    files = tuple(
        CorpusFileInput.supported_text(
            repository_id="repository-peer",
            repository_name="peer",
            relative_path=f"src/file_{ordinal}.py",
            content=f"alpha evidence file {ordinal}\nvalue = {ordinal}\n",
        )
        for ordinal in range(1, 5)
    )
    snapshot = CorpusSnapshotInput.create(
        scope_key=f"group:{group_id}",
        generation_version="generation-1",
        changed_repository=CorpusRepositoryInput(
            repository_id="repository-changed",
            repository_name="changed",
            expected_file_count=0,
            repository_revision="changed-revision-1",
            document_id=source_document_id,
            document_revision="changed-document-revision-1",
        ),
        sibling_repositories=(
            CorpusRepositoryInput(
                repository_id="repository-peer",
                repository_name="peer",
                expected_file_count=len(files),
                repository_revision="peer-revision-1",
            ),
        ),
        files=files,
        producer_id="trusted-persistence-fixture",
        producer_version="1",
        snapshot_id="trusted-snapshot-1",
    )
    corpus = CorpusIngestionService(sessions).ingest(snapshot)
    provider = FixtureEmbeddingProvider()
    identity = BaselineEmbeddingIdentity(
        provider=provider.provider,
        model=provider.model,
        revision=provider.revision,
        dimension=provider.dimension,
        fingerprint=provider.fingerprint,
    )
    BaselineIndexBuilder(sessions).build(
        generation_id=corpus.generation_id,
        index_version="index-1",
        embedding=identity,
        provider=provider,
    )
    request = RetrievalRequest(
        request_id="persistence-request-1",
        changed_repository=None,
        repository_roots=(),
        corpus_version="generation-1",
        retrieval_query="alpha persistence query",
        retrieval_query_origin=RetrievalQueryOrigin.EXPLICIT,
        corpus_complete=True,
        corpus_scope_key=f"group:{group_id}",
        changed_repository_id="repository-changed",
    )
    result = PersistentBaselineV1Retriever(sessions, provider).retrieve(request)
    assert result.status is RetrievalStatus.OK
    assert len(result.evidence) == 4
    return PersistenceEnvironment(
        engine=engine,
        sessions=sessions,
        group_id=group_id,
        source_document_id=source_document_id,
        source_chunk_id=source_chunk_id,
        peer_document_id=peer_document_id,
        peer_chunk_id=peer_chunk_id,
        result=result,
    )


def _sqlite_environment(tmp_path: Path, name: str = "evidence.db"):
    engine = core_db.create_engine(
        f"sqlite:///{tmp_path / name}",
        connect_args={"check_same_thread": False, "timeout": 15},
    )
    return make_persistence_environment(engine)


def persistence_counts(engine: Engine) -> tuple[int, int, int, int, int]:
    with engine.connect() as connection:
        return tuple(
            int(
                connection.execute(
                    text(
                        f"SELECT count(*) FROM {table}"
                        + (
                            " WHERE baseline_selected_evidence_id IS NOT NULL"
                            if table == "reference"
                            else ""
                        )
                    )
                ).scalar_one()
            )
            for table in (
                "baseline_retrieval_run",
                "baseline_evidence_artifact",
                "baseline_selected_evidence",
                "reference",
                "feedback",
            )
        )


def seed_running_control_job(
    environment: PersistenceEnvironment,
    *,
    source_document_id: str | None = None,
    group_id: str | None = None,
) -> tuple[str, str, str]:
    selected_group_id = group_id or environment.group_id
    selected_document_id = source_document_id or environment.source_document_id
    with environment.engine.connect() as connection:
        caller_user_id = str(
            connection.execute(
                text(
                    "SELECT user_id FROM user_to_group WHERE group_id = :group_id "
                    "ORDER BY user_id"
                ),
                {"group_id": environment.group_id},
            ).scalar_one()
        )
        corpus_generation_id = str(
            connection.execute(
                text(
                    "SELECT active_generation_id FROM retrieval_corpus "
                    "WHERE corpus_id = :corpus_id"
                ),
                {"corpus_id": environment.result.corpus_id},
            ).scalar_one()
        )
        corpus = (
            connection.execute(
                text(
                    "SELECT corpus_id, changed_repository_id FROM retrieval_corpus "
                    "WHERE corpus_id = :corpus_id"
                ),
                {"corpus_id": environment.result.corpus_id},
            )
            .mappings()
            .one()
        )
        generation = (
            connection.execute(
                text(
                    "SELECT generation_version, manifest_hash FROM "
                    "retrieval_corpus_generation WHERE generation_id = :generation_id"
                ),
                {"generation_id": corpus_generation_id},
            )
            .mappings()
            .one()
        )
        build = (
            connection.execute(
                text(
                    "SELECT * FROM retrieval_baseline_index_build "
                    "WHERE index_id = :index_id"
                ),
                {"index_id": environment.result.index_id},
            )
            .mappings()
            .one()
        )
    result = environment.result
    query = result.query_provenance
    assert query is not None and query.sha256 is not None
    now = datetime.now(timezone.utc)
    job_id = str(uuid4())
    lease_token = f"lease-{uuid4().hex}"
    digest = hashlib.sha256(job_id.encode()).hexdigest()
    registration_id = str(corpus["changed_repository_id"])
    staging_job_id = str(uuid4())
    staging_id = str(uuid4())
    continuation_id = str(uuid4())
    index_job_id = str(uuid4())
    snapshot_id = f"bsnap_{digest}"
    control_manifest_hash = hashlib.sha256(f"control:{job_id}".encode()).hexdigest()
    provenance_fingerprint = hashlib.sha256(f"provenance:{job_id}".encode()).hexdigest()
    with environment.engine.begin() as connection:
        if (
            connection.execute(
                select(repository_registration.c.registration_id).where(
                    repository_registration.c.registration_id == registration_id,
                    repository_registration.c.group_id == selected_group_id,
                )
            ).first()
            is None
        ):
            connection.execute(
                repository_registration.insert().values(
                    registration_id=registration_id,
                    group_id=selected_group_id,
                    repository_id=registration_id,
                    repository_name=f"changed-{selected_group_id}",
                    source_document_id=selected_document_id,
                    created_by_user_id=caller_user_id,
                    enabled=True,
                    created_at=now,
                    updated_at=now,
                )
            )
            connection.execute(
                repository_approval.insert().values(
                    registration_id=registration_id,
                    group_id=selected_group_id,
                    descriptor_version="repository-identity.v1",
                    repository_authority="persistence.test",
                    repository_uid=f"changed-{selected_group_id}",
                    descriptor_hash=hashlib.sha256(
                        f"descriptor:{selected_group_id}".encode()
                    ).hexdigest(),
                    state="active",
                    approved_by_user_id=caller_user_id,
                    disabled_by_user_id=None,
                    created_at=now,
                    updated_at=now,
                    disabled_at=None,
                )
            )
        connection.execute(
            control_job.insert().values(
                job_id=staging_job_id,
                group_id=selected_group_id,
                request_id=str(uuid4()),
                operation="snapshot_ingest",
                idempotency_key=f"fixture-staging-{staging_job_id}",
                intent_hash=hashlib.sha256(
                    f"staging:{staging_job_id}".encode()
                ).hexdigest(),
                protocol_version="baseline-control-plane.v1",
                protocol_sha256="a" * 64,
                state="succeeded",
                attempt_count=1,
                progress_completed=1,
                progress_total=1,
                result_snapshot_id=snapshot_id,
                created_at=now,
                updated_at=now,
                finished_at=now,
            )
        )
        connection.execute(
            snapshot_staging.insert().values(
                staging_id=staging_id,
                group_id=selected_group_id,
                job_id=staging_job_id,
                snapshot_id=snapshot_id,
                status="sealed",
                manifest_schema_version="baseline-snapshot.v1",
                canonical_manifest_hash=control_manifest_hash,
                canonical_manifest_json="{}",
                changed_repository_id=registration_id,
                source_document_id=selected_document_id,
                expected_repository_count=1,
                expected_file_count=0,
                expected_supported_file_count=0,
                expected_supported_content_bytes=0,
                expected_part_count=0,
                received_part_count=0,
                received_file_count=0,
                received_content_bytes=0,
                content_manifest_hash=control_manifest_hash,
                expires_at=now + timedelta(minutes=10),
                created_at=now,
                updated_at=now,
                sealed_at=now,
            )
        )
        connection.execute(
            snapshot_continuation_job.insert().values(
                continuation_job_id=continuation_id,
                group_id=selected_group_id,
                staging_id=staging_id,
                request_id=str(uuid4()),
                created_by_user_id=caller_user_id,
                contract_version="baseline-snapshot-continuation.v1",
                idempotency_key=f"fixture-continuation-{continuation_id}",
                sealed_intent_hash=digest,
                snapshot_id=snapshot_id,
                canonical_manifest_hash=control_manifest_hash,
                content_manifest_hash=control_manifest_hash,
                repository_set_hash=hashlib.sha256(
                    f"repositories:{job_id}".encode()
                ).hexdigest(),
                expected_repository_count=1,
                expected_file_count=0,
                expected_supported_file_count=0,
                expected_supported_content_bytes=0,
                expected_part_count=0,
                state="succeeded",
                attempt_count=1,
                result_corpus_id=result.corpus_id,
                result_generation_id=corpus_generation_id,
                result_generation_version=generation["generation_version"],
                result_manifest_hash=result.corpus_manifest_hash,
                result_provenance_fingerprint=provenance_fingerprint,
                result_worker_contract_version="baseline-continuation-worker.v1",
                result_published_at=now,
                created_at=now,
                updated_at=now,
                finished_at=now,
            )
        )
        connection.execute(
            control_job.insert().values(
                job_id=index_job_id,
                group_id=selected_group_id,
                request_id=str(uuid4()),
                operation="index_build",
                idempotency_key=f"fixture-index-{index_job_id}",
                intent_hash=hashlib.sha256(
                    f"index:{index_job_id}".encode()
                ).hexdigest(),
                protocol_version="baseline-control-plane.v1",
                protocol_sha256="b" * 64,
                state="succeeded",
                attempt_count=1,
                progress_completed=int(build["indexed_document_count"]),
                progress_total=int(build["indexed_document_count"]),
                created_at=now,
                updated_at=now,
                finished_at=now,
            )
        )
        connection.execute(
            compatible_index_job.insert().values(
                job_id=index_job_id,
                group_id=selected_group_id,
                continuation_job_id=continuation_id,
                submitted_by_user_id=caller_user_id,
                contract_version="baseline-index-build-continuation.v1",
                index_intent_hash=hashlib.sha256(
                    f"index-intent:{job_id}".encode()
                ).hexdigest(),
                snapshot_id=snapshot_id,
                corpus_id=result.corpus_id,
                generation_id=corpus_generation_id,
                generation_version=generation["generation_version"],
                control_manifest_hash=control_manifest_hash,
                corpus_manifest_hash=result.corpus_manifest_hash,
                corpus_file_manifest_hash=generation["manifest_hash"],
                ingestion_provenance_fingerprint=provenance_fingerprint,
                index_format_version=build["index_schema_version"],
                tokenizer_version=build["tokenizer_version"],
                retrieval_config_fingerprint=result.config_fingerprint,
                embedding_contract_version="baseline-embedding-http.v1",
                embedding_provider=result.embedding_provider,
                embedding_model=result.embedding_model,
                embedding_revision=result.embedding_revision,
                embedding_dimension=result.embedding_dimension,
                embedding_dtype="float32",
                embedding_fingerprint=result.embedding_fingerprint,
                result_index_id=result.index_id,
                result_document_count=int(build["indexed_document_count"]),
                result_total_token_count=int(build["total_token_count"]),
                result_document_manifest_hash=build["document_manifest_hash"],
                result_lexical_manifest_hash=build["lexical_manifest_hash"],
                result_dense_manifest_hash=build["dense_manifest_hash"],
                result_published_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            baseline_run_job.insert().values(
                job_id=job_id,
                group_id=selected_group_id,
                submitted_by_user_id=caller_user_id,
                source_document_id=selected_document_id,
                changed_repository_registration_id=registration_id,
                index_job_id=index_job_id,
                corpus_id=result.corpus_id,
                corpus_generation_id=corpus_generation_id,
                index_publication_id=result.index_id,
                corpus_manifest_hash=result.corpus_manifest_hash,
                index_format_version="baseline-index.v1",
                tokenizer_version=build["tokenizer_version"],
                retrieval_config_fingerprint=result.config_fingerprint,
                embedding_fingerprint=result.embedding_fingerprint,
                index_fingerprint=result.index_fingerprint,
                contract_version="baseline-run-job.v1",
                protocol_version=PROTOCOL_V2_VERSION,
                protocol_sha256=PROTOCOL_V2_SHA256,
                request_id=str(uuid4()),
                idempotency_key_hash=digest,
                intent_hash=hashlib.sha256(f"intent:{job_id}".encode()).hexdigest(),
                processing_run_id=str(uuid4()),
                parent_processing_identity_fingerprint=hashlib.sha256(
                    f"parent:{job_id}".encode()
                ).hexdigest(),
                query_representation="raw_git_diff_v1",
                query_encoding="utf-8",
                query_base_revision="1" * 40,
                query_head_revision="2" * 40,
                query_sha256=query.sha256,
                query_length=query.length,
                query_byte_length=query.length,
                query_origin=query.origin.value,
                state="running",
                attempt_count=1,
                lease_token=lease_token,
                lease_expires_at=now + timedelta(minutes=5),
                worker_service_id=BASELINE_RUN_WORKER_SERVICE_ID,
                worker_contract_version=BASELINE_RUN_WORKER_CONTRACT_VERSION,
                started_at=now,
                payload_expires_at=now + timedelta(minutes=10),
                evidence_count=0,
                reference_count=0,
                feedback_count=0,
                generation_invoked=False,
                notification_outbox_count=0,
                created_at=now,
                updated_at=now,
            )
        )
        connection.execute(
            baseline_run_payload.insert().values(
                job_id=job_id,
                group_id=selected_group_id,
                payload_schema_version="baseline-run-protected-payload.v1",
                algorithm="AES-256-GCM",
                key_id=f"fixture-{job_id}",
                nonce=uuid4().bytes[:12],
                ciphertext=b"fixture-protected-payload",
                aad_version="baseline-run-aad.v1",
                created_at=now,
                expires_at=now + timedelta(minutes=10),
            )
        )
    return job_id, lease_token, caller_user_id


def control_command(
    environment: PersistenceEnvironment,
    *,
    job_id: str,
    lease_token: str,
    caller_user_id: str,
    key: str = "document-level-persistence-intent",
) -> BaselineEvidencePersistenceCommand:
    return BaselineEvidencePersistenceCommand(
        group_id=environment.group_id,
        source=ControlDocumentSource(
            document_id=environment.source_document_id,
            control_job_id=job_id,
            lease_token=lease_token,
        ),
        idempotency_key=key,
        retrieval_result=environment.result,
        caller_user_id=caller_user_id,
    )


def test_success_persists_explicit_order_and_survives_restart(tmp_path: Path) -> None:
    environment = _sqlite_environment(tmp_path)
    try:
        receipt = BaselineEvidencePersistenceService(environment.sessions).persist(
            environment.command()
        )
        assert receipt.replayed is False
        assert persistence_counts(environment.engine) == (1, 4, 4, 4, 0)
        with environment.engine.connect() as connection:
            rows = connection.execute(
                text(
                    "SELECT s.ordinal, s.fused_rank, a.repository_name, "
                    "a.relative_path, r.reference_type, r.reference_chunk_id, "
                    "r.baseline_selected_evidence_id, s.renderer_output "
                    "FROM baseline_selected_evidence s "
                    "JOIN baseline_evidence_artifact a ON a.artifact_id = s.artifact_id "
                    "JOIN reference r ON r.baseline_selected_evidence_id = s.selected_evidence_id "
                    "ORDER BY s.ordinal"
                )
            ).all()
            legacy = connection.execute(
                text(
                    "SELECT reference_chunk_id, reference_document_id, "
                    "baseline_selected_evidence_id FROM reference "
                    "WHERE baseline_selected_evidence_id IS NULL"
                )
            ).one()
        assert [row.ordinal for row in rows] == [1, 2, 3, 4]
        assert [row.fused_rank for row in rows] == [1, 2, 3, 4]
        assert [row.relative_path for row in rows] == [
            item.relative_path for item in environment.result.evidence
        ]
        assert all(row.reference_type == "baseline_file" for row in rows)
        assert all(row.reference_chunk_id is None for row in rows)
        assert tuple(row.baseline_selected_evidence_id for row in rows) == (
            receipt.selected_evidence_ids
        )
        assert rows[0].renderer_output.startswith("Repository file: peer/src/")
        assert legacy == (
            environment.peer_chunk_id,
            environment.peer_document_id,
            None,
        )

        database_url = str(environment.engine.url)
        environment.engine.dispose()
        restarted = core_db.create_engine(
            database_url,
            connect_args={"check_same_thread": False, "timeout": 15},
        )
        restarted_sessions = core_db.sessionmaker(restarted, expire_on_commit=False)
        replay = BaselineEvidencePersistenceService(restarted_sessions).persist(
            environment.command()
        )
        assert replay.replayed is True
        assert replay.run_id == receipt.run_id
        assert replay.selected_evidence_ids == receipt.selected_evidence_ids
        assert replay.reference_ids == receipt.reference_ids
        assert persistence_counts(restarted) == (1, 4, 4, 4, 0)
        restarted.dispose()
    finally:
        environment.engine.dispose()


@pytest.mark.parametrize(
    ("mutation", "error_code"),
    [
        (
            lambda result: replace(result, status=RetrievalStatus.INSUFFICIENT),
            "unsupported_result",
        ),
        (lambda result: replace(result, evidence=()), "malformed_result"),
        (
            lambda result: replace(result, index_fingerprint="0" * 64),
            "result_publication_mismatch",
        ),
        (
            lambda result: replace(result, corpus_scope_key="group:somewhere-else"),
            "result_publication_mismatch",
        ),
    ],
)
def test_invalid_results_write_nothing(
    tmp_path: Path, mutation, error_code: str
) -> None:
    environment = _sqlite_environment(tmp_path, f"invalid-{error_code}.db")
    try:
        command = replace(
            environment.command(), retrieval_result=mutation(environment.result)
        )
        with pytest.raises(BaselineEvidencePersistenceError) as caught:
            BaselineEvidencePersistenceService(environment.sessions).persist(command)
        assert caught.value.code == error_code
        assert persistence_counts(environment.engine) == (0, 0, 0, 0, 0)
    finally:
        environment.engine.dispose()


def test_authorization_and_stale_publication_fail_before_writes(tmp_path: Path) -> None:
    unauthorized = _sqlite_environment(tmp_path, "unauthorized.db")
    try:
        with unauthorized.engine.begin() as connection:
            connection.execute(
                text(
                    "DELETE FROM document_to_group "
                    "WHERE document_id = :document_id AND group_id = :group_id"
                ),
                {
                    "document_id": unauthorized.source_document_id,
                    "group_id": unauthorized.group_id,
                },
            )
        with pytest.raises(BaselineEvidencePersistenceError) as caught:
            BaselineEvidencePersistenceService(unauthorized.sessions).persist(
                unauthorized.command()
            )
        assert caught.value.code == "source_unauthorized"
        assert persistence_counts(unauthorized.engine) == (0, 0, 0, 0, 0)
    finally:
        unauthorized.engine.dispose()

    stale = _sqlite_environment(tmp_path, "stale.db")
    try:
        with stale.sessions.begin() as session:
            corpus = session.scalar(
                text("SELECT corpus_id FROM retrieval_corpus WHERE scope_key = :scope"),
                {"scope": f"group:{stale.group_id}"},
            )
            publication = session.get(RetrievalBaselineIndexPublication, corpus)
            publication.index_id = None
        with pytest.raises(BaselineEvidencePersistenceError) as caught:
            BaselineEvidencePersistenceService(stale.sessions).persist(stale.command())
        assert caught.value.code == "compatible_publication_absent"
        assert persistence_counts(stale.engine) == (0, 0, 0, 0, 0)
    finally:
        stale.engine.dispose()


def test_replay_conflict_and_reauthorization(tmp_path: Path) -> None:
    environment = _sqlite_environment(tmp_path)
    try:
        service = BaselineEvidencePersistenceService(environment.sessions)
        first = service.persist(environment.command())
        replay = service.persist(environment.command())
        assert replay.replayed is True
        assert replay.run_id == first.run_id
        assert persistence_counts(environment.engine) == (1, 4, 4, 4, 0)

        conflicting_result = replace(
            environment.result, request_id="a-different-request-intent"
        )
        with pytest.raises(BaselineEvidencePersistenceError) as caught:
            service.persist(
                replace(environment.command(), retrieval_result=conflicting_result)
            )
        assert caught.value.code == "idempotency_conflict"
        assert persistence_counts(environment.engine) == (1, 4, 4, 4, 0)

        with environment.engine.begin() as connection:
            connection.execute(
                text(
                    "DELETE FROM document_to_group "
                    "WHERE document_id = :document_id AND group_id = :group_id"
                ),
                {
                    "document_id": environment.source_document_id,
                    "group_id": environment.group_id,
                },
            )
        with pytest.raises(BaselineEvidencePersistenceError) as caught:
            service.persist(environment.command())
        assert caught.value.code == "source_unauthorized"
        assert persistence_counts(environment.engine) == (1, 4, 4, 4, 0)
    finally:
        environment.engine.dispose()


def test_concurrent_identical_retries_create_one_run(tmp_path: Path) -> None:
    environment = _sqlite_environment(tmp_path)
    try:

        def persist_once(_ordinal: int):
            return BaselineEvidencePersistenceService(environment.sessions).persist(
                environment.command()
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            receipts = tuple(executor.map(persist_once, range(4)))
        assert len({receipt.run_id for receipt in receipts}) == 1
        assert sum(not receipt.replayed for receipt in receipts) == 1
        assert persistence_counts(environment.engine) == (1, 4, 4, 4, 0)
    finally:
        environment.engine.dispose()


@pytest.mark.parametrize(
    "stage",
    tuple(
        stage
        for stage in PersistenceWriteStage
        if stage
        not in {
            PersistenceWriteStage.CONTROL_RELATIONSHIP,
            PersistenceWriteStage.PROTECTED_PAYLOAD,
        }
    ),
)
def test_failure_at_every_write_stage_rolls_back_fully(
    tmp_path: Path,
    stage: PersistenceWriteStage,
) -> None:
    environment = _sqlite_environment(tmp_path, f"rollback-{stage.value}.db")

    def fail_at(actual: PersistenceWriteStage) -> None:
        if actual is stage:
            raise RuntimeError(f"injected-{stage.value}")

    try:
        with pytest.raises(RuntimeError, match=f"injected-{stage.value}"):
            BaselineEvidencePersistenceService(
                environment.sessions, stage_hook=fail_at
            ).persist(environment.command())
        assert persistence_counts(environment.engine) == (0, 0, 0, 0, 0)
    finally:
        environment.engine.dispose()


def test_control_document_persists_without_chunk_and_replays_exactly(
    tmp_path: Path,
) -> None:
    environment = _sqlite_environment(tmp_path, "control-document.db")
    try:
        initial_chunk_count = None
        with environment.engine.connect() as connection:
            initial_chunk_count = connection.execute(
                text("SELECT count(*) FROM chunk")
            ).scalar_one()
        job_id, lease_token, caller = seed_running_control_job(environment)
        command = control_command(
            environment,
            job_id=job_id,
            lease_token=lease_token,
            caller_user_id=caller,
        )
        service = BaselineEvidencePersistenceService(environment.sessions)
        first = service.persist(command)
        replay = service.persist(command)
        assert replay.replayed is True
        assert replay.run_id == first.run_id
        assert replay.selected_evidence_ids == first.selected_evidence_ids
        assert replay.reference_ids == first.reference_ids
        with environment.engine.connect() as connection:
            run = (
                connection.execute(
                    text(
                        "SELECT source_scope_version, source_scope, source_chunk_id, "
                        "source_document_id FROM baseline_retrieval_run "
                        "WHERE run_id = :run_id"
                    ),
                    {"run_id": first.run_id},
                )
                .mappings()
                .one()
            )
            job = (
                connection.execute(
                    text(
                        "SELECT state, persisted_run_id, evidence_count, "
                        "reference_count, lease_token FROM baseline_control_run_job "
                        "WHERE job_id = :job_id"
                    ),
                    {"job_id": job_id},
                )
                .mappings()
                .one()
            )
            reference_chunks = (
                connection.execute(
                    text(
                        "SELECT r.source_chunk_id FROM reference r "
                        "JOIN baseline_selected_evidence s ON "
                        "s.selected_evidence_id = r.baseline_selected_evidence_id "
                        "WHERE s.run_id = :run_id ORDER BY s.ordinal"
                    ),
                    {"run_id": first.run_id},
                )
                .scalars()
                .all()
            )
            chunk_count = connection.execute(
                text("SELECT count(*) FROM chunk")
            ).scalar_one()
        assert dict(run) == {
            "source_scope_version": "baseline-source-scope.v1",
            "source_scope": "control_document",
            "source_chunk_id": None,
            "source_document_id": environment.source_document_id,
        }
        assert dict(job) == {
            "state": "references_persisted",
            "persisted_run_id": first.run_id,
            "evidence_count": len(first.selected_evidence_ids),
            "reference_count": len(first.reference_ids),
            "lease_token": None,
        }
        assert reference_chunks == [None] * len(first.reference_ids)
        assert chunk_count == initial_chunk_count
    finally:
        environment.engine.dispose()


def test_control_document_source_shape_and_scope_fail_closed(tmp_path: Path) -> None:
    environment = _sqlite_environment(tmp_path, "control-scope-errors.db")
    try:
        job_id, lease_token, caller = seed_running_control_job(environment)
        with pytest.raises(TypeError):
            ControlDocumentSource(  # type: ignore[call-arg]
                document_id=environment.source_document_id,
                control_job_id=job_id,
                lease_token=lease_token,
                chunk_id=environment.source_chunk_id,
            )
        malformed_legacy = BaselineEvidencePersistenceCommand(
            group_id=environment.group_id,
            source=LegacyChunkSource(
                document_id=environment.source_document_id,
                chunk_id="",
            ),
            idempotency_key="missing-legacy-chunk",
            retrieval_result=environment.result,
        )
        with pytest.raises(BaselineEvidencePersistenceError, match="source_chunk_id"):
            BaselineEvidencePersistenceService(environment.sessions).persist(
                malformed_legacy
            )
        wrong_document = control_command(
            environment,
            job_id=job_id,
            lease_token=lease_token,
            caller_user_id=caller,
        )
        wrong_document = replace(
            wrong_document,
            source=ControlDocumentSource(
                document_id=environment.peer_document_id,
                control_job_id=job_id,
                lease_token=lease_token,
            ),
        )
        with pytest.raises(
            BaselineEvidencePersistenceError, match="control job source"
        ):
            BaselineEvidencePersistenceService(environment.sessions).persist(
                wrong_document
            )
        other_group_id = str(uuid4())
        with environment.engine.begin() as connection:
            connection.execute(
                text(
                    'INSERT INTO "group" '
                    "(group_id, name, datetime_created, category, description, visibility) "
                    "VALUES (:group_id, 'Other scope', :now, 'Other', '', 'private')"
                ),
                {"group_id": other_group_id, "now": datetime.now(timezone.utc)},
            )
            connection.execute(
                text(
                    "INSERT INTO user_to_group (user_id, group_id) "
                    "VALUES (:user_id, :group_id)"
                ),
                {"user_id": caller, "group_id": other_group_id},
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
        wrong_group = replace(
            control_command(
                environment,
                job_id=job_id,
                lease_token=lease_token,
                caller_user_id=caller,
            ),
            group_id=other_group_id,
        )
        with pytest.raises(
            BaselineEvidencePersistenceError, match="control job source"
        ):
            BaselineEvidencePersistenceService(environment.sessions).persist(
                wrong_group
            )
        assert persistence_counts(environment.engine) == (0, 0, 0, 0, 0)
    finally:
        environment.engine.dispose()


def test_control_document_conflicting_attachment_and_rollback_are_atomic(
    tmp_path: Path,
) -> None:
    environment = _sqlite_environment(tmp_path, "control-atomic.db")
    try:
        job_id, lease_token, caller = seed_running_control_job(environment)
        command = control_command(
            environment,
            job_id=job_id,
            lease_token=lease_token,
            caller_user_id=caller,
        )

        def fail_after_link(stage: PersistenceWriteStage) -> None:
            if stage is PersistenceWriteStage.CONTROL_RELATIONSHIP:
                raise RuntimeError("injected-control-link")

        with pytest.raises(RuntimeError, match="injected-control-link"):
            BaselineEvidencePersistenceService(
                environment.sessions, stage_hook=fail_after_link
            ).persist(command)
        assert persistence_counts(environment.engine) == (0, 0, 0, 0, 0)
        with environment.engine.connect() as connection:
            rolled_back = connection.execute(
                text(
                    "SELECT state, persisted_run_id FROM baseline_control_run_job "
                    "WHERE job_id = :job_id"
                ),
                {"job_id": job_id},
            ).one()
        assert tuple(rolled_back) == ("running", None)

        service = BaselineEvidencePersistenceService(environment.sessions)
        receipt = service.persist(command)
        second_job_id, second_lease, _second_caller = seed_running_control_job(
            environment
        )
        with (
            pytest.raises(IntegrityError),
            environment.engine.begin() as connection,
        ):
            connection.execute(
                text(
                    "UPDATE baseline_control_run_job SET "
                    "persisted_run_id = :run_id, state = 'references_persisted', "
                    "evidence_count = :count, reference_count = :count, "
                    "lease_token = NULL, lease_expires_at = NULL "
                    "WHERE job_id = :job_id AND lease_token = :lease_token"
                ),
                {
                    "run_id": receipt.run_id,
                    "count": len(receipt.reference_ids),
                    "job_id": second_job_id,
                    "lease_token": second_lease,
                },
            )
        conflicting = replace(command, idempotency_key="different-document-intent")
        with pytest.raises(
            BaselineEvidencePersistenceError,
            match="control job is attached|control job changed",
        ):
            service.persist(conflicting)
        assert persistence_counts(environment.engine) == (1, 4, 4, 4, 0)
        with environment.engine.connect() as connection:
            assert (
                connection.execute(
                    text(
                        "SELECT persisted_run_id FROM baseline_control_run_job "
                        "WHERE job_id = :job_id"
                    ),
                    {"job_id": job_id},
                ).scalar_one()
                == receipt.run_id
            )
    finally:
        environment.engine.dispose()


def test_control_document_retention_and_group_cascade(tmp_path: Path) -> None:
    environment = _sqlite_environment(tmp_path, "control-retention.db")
    try:
        job_id, lease_token, caller = seed_running_control_job(environment)
        receipt = BaselineEvidencePersistenceService(environment.sessions).persist(
            control_command(
                environment,
                job_id=job_id,
                lease_token=lease_token,
                caller_user_id=caller,
            )
        )
        with environment.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM document WHERE document_id = :document_id"),
                {"document_id": environment.source_document_id},
            )
        with environment.engine.connect() as connection:
            retained = connection.execute(
                text(
                    "SELECT source_chunk_id, source_document_id FROM "
                    "baseline_retrieval_run WHERE run_id = :run_id"
                ),
                {"run_id": receipt.run_id},
            ).one()
            assert tuple(retained) == (None, None)
            assert connection.execute(
                text(
                    "SELECT count(*) FROM reference WHERE "
                    "baseline_selected_evidence_id IS NOT NULL"
                )
            ).scalar_one() == len(receipt.reference_ids)
            assert (
                connection.execute(
                    text(
                        "SELECT source_document_id FROM baseline_control_run_job "
                        "WHERE job_id = :job_id"
                    ),
                    {"job_id": job_id},
                ).scalar_one()
                is None
            )
        with environment.engine.begin() as connection:
            connection.execute(
                text('DELETE FROM "group" WHERE group_id = :group_id'),
                {"group_id": environment.group_id},
            )
        with environment.engine.connect() as connection:
            assert (
                connection.execute(
                    text(
                        "SELECT count(*) FROM baseline_retrieval_run "
                        "WHERE run_id = :run_id"
                    ),
                    {"run_id": receipt.run_id},
                ).scalar_one()
                == 0
            )
            assert (
                connection.execute(
                    text(
                        "SELECT count(*) FROM baseline_control_run_job "
                        "WHERE job_id = :job_id"
                    ),
                    {"job_id": job_id},
                ).scalar_one()
                == 0
            )
    finally:
        environment.engine.dispose()


def test_source_deletion_preserves_auditable_baseline_evidence(tmp_path: Path) -> None:
    environment = _sqlite_environment(tmp_path)
    try:
        receipt = BaselineEvidencePersistenceService(environment.sessions).persist(
            environment.command()
        )
        with environment.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM document WHERE document_id = :document_id"),
                {"document_id": environment.source_document_id},
            )
        assert persistence_counts(environment.engine) == (1, 4, 4, 4, 0)
        with environment.engine.connect() as connection:
            run = connection.execute(
                text(
                    "SELECT source_chunk_id, source_document_id "
                    "FROM baseline_retrieval_run WHERE run_id = :run_id"
                ),
                {"run_id": receipt.run_id},
            ).one()
            references = connection.execute(
                text(
                    "SELECT source_chunk_id FROM reference "
                    "WHERE baseline_selected_evidence_id IS NOT NULL"
                )
            ).all()
        assert run == (None, None)
        assert references == [(None,)] * 4

        with pytest.raises(BaselineEvidencePersistenceError) as caught:
            BaselineEvidencePersistenceService(environment.sessions).persist(
                environment.command()
            )
        assert caught.value.code == "source_unauthorized"
        assert persistence_counts(environment.engine) == (1, 4, 4, 4, 0)
    finally:
        environment.engine.dispose()
