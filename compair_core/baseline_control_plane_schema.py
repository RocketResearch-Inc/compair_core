"""Durable schema for baseline control-plane staging and continuation.

The schema stops at an immutable sealed snapshot and a non-eligible downstream
continuation job. No table in this module is a corpus generation, index
publication, retrieval run, or evidence record, so neither state can become
baseline eligible by existence alone.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    text,
)

CONTROL_PLANE_SCHEMA_VERSION = "baseline-control-plane-staging.v1"

REPOSITORY_REGISTRATION_TABLE = "baseline_control_repository_registration"
CONTROL_JOB_TABLE = "baseline_control_job"
SNAPSHOT_STAGING_TABLE = "baseline_snapshot_staging"
SNAPSHOT_CONTENT_PART_TABLE = "baseline_snapshot_content_part"
REPOSITORY_APPROVAL_TABLE = "baseline_control_repository_approval"
SNAPSHOT_CONTINUATION_JOB_TABLE = "baseline_snapshot_continuation_job"
COMPATIBLE_INDEX_JOB_TABLE = "baseline_compatible_index_job"
BASELINE_RUN_JOB_TABLE = "baseline_control_run_job"
BASELINE_RUN_PAYLOAD_TABLE = "baseline_control_run_payload"
BASELINE_WORKER_INSTANCE_TABLE = "baseline_database_worker_instance"
BASELINE_WORKER_ATTESTATION_TABLE = "baseline_database_worker_attestation"
BASELINE_RUN_WORKER_SERVICE_ID = "compair-core-baseline-runner"
BASELINE_RUN_WORKER_CONTRACT_VERSION = "baseline-run-worker.v1"
BASELINE_CONTROL_GENERATION_CONTRACT_VERSION = "baseline-control-generation.v1"
BASELINE_DATABASE_WORKER_CONTRACT_VERSION = "baseline-database-worker.v1"

_metadata = MetaData()

# Dependency stubs allow the migration registry to compile this schema without
# importing the application ORM and triggering startup side effects.
Table("group", _metadata, Column("group_id", String(36), primary_key=True))
Table("user", _metadata, Column("user_id", String(36), primary_key=True))
Table("document", _metadata, Column("document_id", String(36), primary_key=True))
Table(
    "baseline_retrieval_run",
    _metadata,
    Column("run_id", String(36), primary_key=True),
    Column("group_id", String(36), nullable=False),
    UniqueConstraint("run_id", "group_id", name="uq_bl_run_id_group_stub"),
)


repository_registration = Table(
    REPOSITORY_REGISTRATION_TABLE,
    _metadata,
    Column("registration_id", String(36), primary_key=True),
    Column(
        "group_id",
        String(36),
        ForeignKey(
            "group.group_id",
            ondelete="CASCADE",
            name="fk_bl_ctl_repository_group",
        ),
        nullable=False,
    ),
    Column("repository_id", String(128), nullable=False),
    Column("repository_name", String(128), nullable=False),
    Column(
        "source_document_id",
        String(36),
        ForeignKey(
            "document.document_id",
            ondelete="SET NULL",
            name="fk_bl_ctl_repository_document",
        ),
        nullable=True,
    ),
    Column(
        "created_by_user_id",
        String(36),
        ForeignKey(
            "user.user_id",
            ondelete="SET NULL",
            name="fk_bl_ctl_repository_creator",
        ),
        nullable=True,
    ),
    Column("enabled", Boolean, nullable=False, server_default=text("TRUE")),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    UniqueConstraint(
        "group_id",
        "repository_id",
        name="uq_bl_ctl_repository_group_id",
    ),
    UniqueConstraint(
        "group_id",
        "repository_name",
        name="uq_bl_ctl_repository_group_name",
    ),
    UniqueConstraint(
        "registration_id",
        "group_id",
        name="uq_bl_ctl_repository_registration_group",
    ),
    CheckConstraint(
        "length(trim(repository_id)) BETWEEN 1 AND 128 "
        "AND length(trim(repository_name)) BETWEEN 1 AND 128",
        name="ck_bl_ctl_repository_identity",
    ),
    Index(
        "ix_bl_ctl_repository_group_enabled",
        "group_id",
        "enabled",
        "repository_id",
    ),
    Index("ix_bl_ctl_repository_document", "source_document_id"),
)


repository_approval = Table(
    REPOSITORY_APPROVAL_TABLE,
    _metadata,
    Column("registration_id", String(36), primary_key=True),
    Column("group_id", String(36), nullable=False),
    Column("descriptor_version", String(64), nullable=False),
    Column("repository_authority", String(253), nullable=False),
    Column("repository_uid", String(256), nullable=False),
    Column("descriptor_hash", String(64), nullable=False),
    Column("state", String(16), nullable=False),
    Column(
        "approved_by_user_id",
        String(36),
        ForeignKey(
            "user.user_id",
            ondelete="SET NULL",
            name="fk_bl_ctl_repository_approval_user",
        ),
        nullable=True,
    ),
    Column(
        "disabled_by_user_id",
        String(36),
        ForeignKey(
            "user.user_id",
            ondelete="SET NULL",
            name="fk_bl_ctl_repository_disabled_user",
        ),
        nullable=True,
    ),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    Column("disabled_at", DateTime(timezone=True), nullable=True),
    ForeignKeyConstraint(
        ["registration_id", "group_id"],
        [
            "baseline_control_repository_registration.registration_id",
            "baseline_control_repository_registration.group_id",
        ],
        name="fk_bl_ctl_repository_approval_registration",
        ondelete="CASCADE",
    ),
    UniqueConstraint(
        "group_id",
        "repository_authority",
        "repository_uid",
        name="uq_bl_ctl_repository_approval_identity",
    ),
    CheckConstraint(
        "descriptor_version = 'repository-identity.v1' "
        "AND length(repository_authority) BETWEEN 1 AND 253 "
        "AND length(repository_uid) BETWEEN 1 AND 256 "
        "AND length(descriptor_hash) = 64",
        name="ck_bl_ctl_repository_approval_descriptor",
    ),
    CheckConstraint(
        "state IN ('active', 'disabled')",
        name="ck_bl_ctl_repository_approval_state",
    ),
    CheckConstraint(
        "(state = 'active' AND disabled_at IS NULL AND disabled_by_user_id IS NULL) "
        "OR (state = 'disabled' AND disabled_at IS NOT NULL)",
        name="ck_bl_ctl_repository_approval_disabled",
    ),
    Index(
        "ix_bl_ctl_repository_approval_group_state",
        "group_id",
        "state",
        "registration_id",
    ),
)


control_job = Table(
    CONTROL_JOB_TABLE,
    _metadata,
    Column("job_id", String(36), primary_key=True),
    Column(
        "group_id",
        String(36),
        ForeignKey(
            "group.group_id",
            ondelete="CASCADE",
            name="fk_bl_ctl_job_group",
        ),
        nullable=False,
    ),
    Column("request_id", String(36), nullable=False),
    Column("operation", String(32), nullable=False),
    Column("idempotency_key", String(128), nullable=False),
    Column("intent_hash", String(64), nullable=False),
    Column("protocol_version", String(64), nullable=False),
    Column("protocol_sha256", String(64), nullable=False),
    Column("state", String(32), nullable=False),
    Column("attempt_count", Integer, nullable=False, server_default="0"),
    Column("lease_token", String(128), nullable=True),
    Column("lease_expires_at", DateTime(timezone=True), nullable=True),
    Column("progress_completed", Integer, nullable=False, server_default="0"),
    Column("progress_total", Integer, nullable=False, server_default="0"),
    Column("result_snapshot_id", String(72), nullable=True),
    Column("error_code", String(128), nullable=True),
    Column("error_fingerprint", String(64), nullable=True),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    UniqueConstraint(
        "group_id",
        "operation",
        "idempotency_key",
        name="uq_bl_ctl_job_group_operation_intent",
    ),
    UniqueConstraint("job_id", "group_id", name="uq_bl_ctl_job_id_group"),
    CheckConstraint(
        "operation IN ('snapshot_ingest', 'index_build', 'baseline_run')",
        name="ck_bl_ctl_job_operation",
    ),
    CheckConstraint(
        "state IN ('queued', 'running', 'succeeded', 'retryable_failed', "
        "'terminal_failed', 'cancelled')",
        name="ck_bl_ctl_job_state",
    ),
    CheckConstraint(
        "length(idempotency_key) BETWEEN 32 AND 128 "
        "AND length(intent_hash) = 64 "
        "AND length(protocol_sha256) = 64",
        name="ck_bl_ctl_job_contract",
    ),
    CheckConstraint(
        "attempt_count >= 0 AND progress_completed >= 0 "
        "AND progress_total >= 0 AND progress_completed <= progress_total",
        name="ck_bl_ctl_job_counts",
    ),
    CheckConstraint(
        "(state = 'running' AND lease_token IS NOT NULL "
        "AND lease_expires_at IS NOT NULL) "
        "OR (state <> 'running' AND lease_token IS NULL "
        "AND lease_expires_at IS NULL)",
        name="ck_bl_ctl_job_lease",
    ),
    Index(
        "ix_bl_ctl_job_group_state",
        "group_id",
        "state",
        "created_at",
        "job_id",
    ),
    Index("ix_bl_ctl_job_lease", "state", "lease_expires_at", "job_id"),
)


snapshot_staging = Table(
    SNAPSHOT_STAGING_TABLE,
    _metadata,
    Column("staging_id", String(36), primary_key=True),
    Column("group_id", String(36), nullable=False),
    Column("job_id", String(36), nullable=False, unique=True),
    Column("snapshot_id", String(72), nullable=False),
    Column("status", String(16), nullable=False),
    Column("manifest_schema_version", String(64), nullable=False),
    Column("canonical_manifest_hash", String(64), nullable=False),
    Column("canonical_manifest_json", Text, nullable=False),
    Column("changed_repository_id", String(128), nullable=False),
    Column(
        "source_document_id",
        String(36),
        ForeignKey(
            "document.document_id",
            ondelete="SET NULL",
            name="fk_bl_ctl_staging_document",
        ),
        nullable=True,
    ),
    Column("expected_repository_count", Integer, nullable=False),
    Column("expected_file_count", Integer, nullable=False),
    Column("expected_supported_file_count", Integer, nullable=False),
    Column("expected_supported_content_bytes", Integer, nullable=False),
    Column("expected_part_count", Integer, nullable=False),
    Column("received_part_count", Integer, nullable=False, server_default="0"),
    Column("received_file_count", Integer, nullable=False, server_default="0"),
    Column("received_content_bytes", Integer, nullable=False, server_default="0"),
    Column("content_manifest_hash", String(64), nullable=True),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    Column("sealed_at", DateTime(timezone=True), nullable=True),
    ForeignKeyConstraint(
        ["job_id", "group_id"],
        ["baseline_control_job.job_id", "baseline_control_job.group_id"],
        name="fk_bl_ctl_staging_job_scope",
        ondelete="CASCADE",
    ),
    UniqueConstraint(
        "staging_id",
        "group_id",
        name="uq_bl_ctl_staging_id_group",
    ),
    UniqueConstraint(
        "group_id",
        "snapshot_id",
        name="uq_bl_ctl_staging_group_snapshot",
    ),
    CheckConstraint(
        "status IN ('open', 'sealed', 'expired', 'failed')",
        name="ck_bl_ctl_staging_status",
    ),
    CheckConstraint(
        "length(canonical_manifest_hash) = 64 AND length(canonical_manifest_json) > 0",
        name="ck_bl_ctl_staging_manifest",
    ),
    CheckConstraint(
        "expected_repository_count BETWEEN 1 AND 128 "
        "AND expected_file_count BETWEEN 0 AND 50000 "
        "AND expected_supported_file_count BETWEEN 0 AND expected_file_count "
        "AND expected_supported_content_bytes BETWEEN 0 AND 512000000 "
        "AND expected_part_count BETWEEN 0 AND 512 "
        "AND received_part_count BETWEEN 0 AND expected_part_count "
        "AND received_file_count BETWEEN 0 AND expected_supported_file_count "
        "AND received_content_bytes BETWEEN 0 AND expected_supported_content_bytes",
        name="ck_bl_ctl_staging_counts",
    ),
    Index("ix_bl_ctl_staging_expiry", "status", "expires_at", "staging_id"),
    Index("ix_bl_ctl_staging_group_status", "group_id", "status", "created_at"),
)


snapshot_content_part = Table(
    SNAPSHOT_CONTENT_PART_TABLE,
    _metadata,
    Column("part_id", String(36), primary_key=True),
    Column("staging_id", String(36), nullable=False),
    Column("group_id", String(36), nullable=False),
    Column("part_ordinal", Integer, nullable=False),
    Column("part_sha256", String(64), nullable=False),
    Column("request_body_sha256", String(64), nullable=False),
    Column("item_count", Integer, nullable=False),
    Column("content_bytes", Integer, nullable=False),
    Column("canonical_content_items_json", Text, nullable=False),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    ForeignKeyConstraint(
        ["staging_id", "group_id"],
        ["baseline_snapshot_staging.staging_id", "baseline_snapshot_staging.group_id"],
        name="fk_bl_ctl_part_staging_scope",
        ondelete="CASCADE",
    ),
    UniqueConstraint(
        "staging_id",
        "part_ordinal",
        name="uq_bl_ctl_part_staging_ordinal",
    ),
    CheckConstraint(
        "part_ordinal BETWEEN 1 AND 512 "
        "AND length(part_sha256) = 64 "
        "AND length(request_body_sha256) = 64",
        name="ck_bl_ctl_part_identity",
    ),
    CheckConstraint(
        "item_count BETWEEN 1 AND 1000 "
        "AND content_bytes BETWEEN 0 AND 1000000 "
        "AND length(canonical_content_items_json) > 0",
        name="ck_bl_ctl_part_limits",
    ),
    Index("ix_bl_ctl_part_order", "staging_id", "part_ordinal"),
)


snapshot_continuation_job = Table(
    SNAPSHOT_CONTINUATION_JOB_TABLE,
    _metadata,
    Column("continuation_job_id", String(36), primary_key=True),
    Column("group_id", String(36), nullable=False),
    Column("staging_id", String(36), nullable=False),
    Column("request_id", String(36), nullable=False),
    Column(
        "created_by_user_id",
        String(36),
        ForeignKey(
            "user.user_id",
            ondelete="SET NULL",
            name="fk_bl_ctl_continuation_user",
        ),
        nullable=True,
    ),
    Column("contract_version", String(64), nullable=False),
    Column("idempotency_key", String(128), nullable=False),
    Column("sealed_intent_hash", String(64), nullable=False),
    Column("snapshot_id", String(72), nullable=False),
    Column("canonical_manifest_hash", String(64), nullable=False),
    Column("content_manifest_hash", String(64), nullable=False),
    Column("repository_set_hash", String(64), nullable=False),
    Column("expected_repository_count", Integer, nullable=False),
    Column("expected_file_count", Integer, nullable=False),
    Column("expected_supported_file_count", Integer, nullable=False),
    Column("expected_supported_content_bytes", Integer, nullable=False),
    Column("expected_part_count", Integer, nullable=False),
    Column("state", String(32), nullable=False),
    Column("attempt_count", Integer, nullable=False, server_default="0"),
    Column("lease_token", String(128), nullable=True),
    Column("lease_expires_at", DateTime(timezone=True), nullable=True),
    Column("error_code", String(128), nullable=True),
    Column("error_fingerprint", String(64), nullable=True),
    Column("result_corpus_id", String(36), nullable=True),
    Column("result_generation_id", String(36), nullable=True),
    Column("result_generation_version", String(128), nullable=True),
    Column("result_manifest_hash", String(64), nullable=True),
    Column("result_provenance_fingerprint", String(64), nullable=True),
    Column("result_worker_contract_version", String(64), nullable=True),
    Column("result_published_at", DateTime(timezone=True), nullable=True),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    ForeignKeyConstraint(
        ["staging_id", "group_id"],
        ["baseline_snapshot_staging.staging_id", "baseline_snapshot_staging.group_id"],
        name="fk_bl_ctl_continuation_staging",
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ["group_id"],
        ["group.group_id"],
        name="fk_bl_ctl_continuation_group",
        ondelete="CASCADE",
    ),
    UniqueConstraint(
        "group_id",
        "idempotency_key",
        name="uq_bl_ctl_continuation_group_intent",
    ),
    UniqueConstraint(
        "staging_id",
        "group_id",
        name="uq_bl_ctl_continuation_staging",
    ),
    UniqueConstraint(
        "continuation_job_id",
        "group_id",
        name="uq_bl_ctl_continuation_job_group",
    ),
    CheckConstraint(
        "state IN ('queued', 'running', 'succeeded', 'retryable_failed', "
        "'terminal_failed', 'cancelled')",
        name="ck_bl_ctl_continuation_state",
    ),
    CheckConstraint(
        "contract_version = 'baseline-snapshot-continuation.v1' "
        "AND length(idempotency_key) BETWEEN 32 AND 128 "
        "AND length(sealed_intent_hash) = 64 "
        "AND length(canonical_manifest_hash) = 64 "
        "AND length(content_manifest_hash) = 64 "
        "AND length(repository_set_hash) = 64",
        name="ck_bl_ctl_continuation_contract",
    ),
    CheckConstraint(
        "expected_repository_count BETWEEN 1 AND 128 "
        "AND expected_file_count BETWEEN 0 AND 50000 "
        "AND expected_supported_file_count BETWEEN 0 AND expected_file_count "
        "AND expected_supported_content_bytes BETWEEN 0 AND 512000000 "
        "AND expected_part_count BETWEEN 0 AND 512 "
        "AND attempt_count >= 0",
        name="ck_bl_ctl_continuation_counts",
    ),
    CheckConstraint(
        "(state = 'running' AND lease_token IS NOT NULL "
        "AND lease_expires_at IS NOT NULL) "
        "OR (state <> 'running' AND lease_token IS NULL "
        "AND lease_expires_at IS NULL)",
        name="ck_bl_ctl_continuation_lease",
    ),
    Index(
        "ix_bl_ctl_continuation_group_state",
        "group_id",
        "state",
        "created_at",
        "continuation_job_id",
    ),
    Index(
        "ix_bl_ctl_continuation_lease",
        "state",
        "lease_expires_at",
        "continuation_job_id",
    ),
    Index(
        "ix_bl_ctl_continuation_result_generation",
        "result_generation_id",
    ),
)


compatible_index_job = Table(
    COMPATIBLE_INDEX_JOB_TABLE,
    _metadata,
    Column("job_id", String(36), primary_key=True),
    Column("group_id", String(36), nullable=False),
    Column("continuation_job_id", String(36), nullable=False),
    Column(
        "submitted_by_user_id",
        String(36),
        ForeignKey(
            "user.user_id",
            ondelete="SET NULL",
            name="fk_bl_idx_job_submitter",
        ),
        nullable=True,
    ),
    Column("contract_version", String(64), nullable=False),
    Column("index_intent_hash", String(64), nullable=False),
    Column("snapshot_id", String(72), nullable=False),
    Column("corpus_id", String(36), nullable=False),
    Column("generation_id", String(36), nullable=False),
    Column("generation_version", String(128), nullable=False),
    Column("control_manifest_hash", String(64), nullable=False),
    Column("corpus_manifest_hash", String(64), nullable=False),
    Column("corpus_file_manifest_hash", String(64), nullable=False),
    Column("ingestion_provenance_fingerprint", String(64), nullable=False),
    Column("index_format_version", String(64), nullable=False),
    Column("tokenizer_version", String(128), nullable=False),
    Column("retrieval_config_fingerprint", String(64), nullable=False),
    Column("embedding_contract_version", String(64), nullable=False),
    Column("embedding_provider", String(128), nullable=False),
    Column("embedding_model", String(256), nullable=False),
    Column("embedding_revision", String(256), nullable=False),
    Column("embedding_dimension", Integer, nullable=False),
    Column("embedding_dtype", String(16), nullable=False),
    Column("embedding_fingerprint", String(64), nullable=False),
    Column("result_index_id", String(36), nullable=True),
    Column("result_document_count", Integer, nullable=True),
    Column("result_total_token_count", Integer, nullable=True),
    Column("result_document_manifest_hash", String(64), nullable=True),
    Column("result_lexical_manifest_hash", String(64), nullable=True),
    Column("result_dense_manifest_hash", String(64), nullable=True),
    Column("result_published_at", DateTime(timezone=True), nullable=True),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    ForeignKeyConstraint(
        ["job_id", "group_id"],
        ["baseline_control_job.job_id", "baseline_control_job.group_id"],
        name="fk_bl_idx_job_control_scope",
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ["continuation_job_id", "group_id"],
        [
            "baseline_snapshot_continuation_job.continuation_job_id",
            "baseline_snapshot_continuation_job.group_id",
        ],
        name="fk_bl_idx_job_continuation_scope",
        ondelete="CASCADE",
    ),
    UniqueConstraint("job_id", "group_id", name="uq_bl_idx_job_scope"),
    UniqueConstraint(
        "group_id",
        "generation_id",
        "index_intent_hash",
        name="uq_bl_idx_job_generation_intent",
    ),
    CheckConstraint(
        "contract_version = 'baseline-index-build-continuation.v1' "
        "AND index_format_version = 'baseline-index.v1' "
        "AND embedding_contract_version = 'baseline-embedding-http.v1' "
        "AND embedding_dtype = 'float32' "
        "AND embedding_dimension > 0 "
        "AND length(index_intent_hash) = 64 "
        "AND length(control_manifest_hash) = 64 "
        "AND length(corpus_manifest_hash) = 64 "
        "AND length(corpus_file_manifest_hash) = 64 "
        "AND length(ingestion_provenance_fingerprint) = 64 "
        "AND length(retrieval_config_fingerprint) = 64 "
        "AND length(embedding_fingerprint) = 64",
        name="ck_bl_idx_job_contract",
    ),
    CheckConstraint(
        "(result_index_id IS NULL AND result_document_count IS NULL "
        "AND result_total_token_count IS NULL "
        "AND result_document_manifest_hash IS NULL "
        "AND result_lexical_manifest_hash IS NULL "
        "AND result_dense_manifest_hash IS NULL "
        "AND result_published_at IS NULL) OR "
        "(result_index_id IS NOT NULL AND result_document_count >= 0 "
        "AND result_total_token_count >= 0 "
        "AND length(result_document_manifest_hash) = 64 "
        "AND length(result_lexical_manifest_hash) = 64 "
        "AND length(result_dense_manifest_hash) = 64 "
        "AND result_published_at IS NOT NULL)",
        name="ck_bl_idx_job_result",
    ),
    Index("ix_bl_idx_job_continuation", "continuation_job_id", "job_id"),
    Index("ix_bl_idx_job_generation", "group_id", "generation_id", "job_id"),
    Index("ix_bl_idx_job_result", "result_index_id"),
)


baseline_run_job = Table(
    BASELINE_RUN_JOB_TABLE,
    _metadata,
    Column("job_id", String(36), primary_key=True),
    Column(
        "group_id",
        String(36),
        ForeignKey(
            "group.group_id",
            ondelete="CASCADE",
            name="fk_bl_run_job_group",
        ),
        nullable=False,
    ),
    Column(
        "submitted_by_user_id",
        String(36),
        ForeignKey(
            "user.user_id",
            ondelete="SET NULL",
            name="fk_bl_run_job_submitter",
        ),
        nullable=True,
    ),
    Column(
        "source_document_id",
        String(36),
        ForeignKey(
            "document.document_id",
            ondelete="SET NULL",
            name="fk_bl_run_job_source_document",
        ),
        nullable=True,
    ),
    Column("changed_repository_registration_id", String(36), nullable=False),
    Column("index_job_id", String(36), nullable=False),
    Column("corpus_id", String(36), nullable=False),
    Column("corpus_generation_id", String(36), nullable=False),
    Column("index_publication_id", String(36), nullable=False),
    Column("corpus_manifest_hash", String(64), nullable=False),
    Column("index_format_version", String(64), nullable=False),
    Column("tokenizer_version", String(128), nullable=False),
    Column("retrieval_config_fingerprint", String(64), nullable=False),
    Column("embedding_fingerprint", String(64), nullable=False),
    Column("index_fingerprint", String(64), nullable=False),
    Column("contract_version", String(64), nullable=False),
    Column("protocol_version", String(64), nullable=False),
    Column("protocol_sha256", String(64), nullable=False),
    Column("request_id", String(36), nullable=False),
    Column("idempotency_key_hash", String(64), nullable=False),
    Column("intent_hash", String(64), nullable=False),
    Column("processing_run_id", String(36), nullable=False, unique=True),
    Column("parent_processing_identity_fingerprint", String(64), nullable=False),
    Column("query_representation", String(64), nullable=False),
    Column("query_encoding", String(16), nullable=False),
    Column("query_base_revision", String(128), nullable=False),
    Column("query_head_revision", String(128), nullable=False),
    Column("query_sha256", String(64), nullable=False),
    Column("query_length", Integer, nullable=False),
    Column("query_byte_length", Integer, nullable=False),
    Column("query_origin", String(16), nullable=False),
    Column("state", String(32), nullable=False),
    Column("attempt_count", Integer, nullable=False, server_default="0"),
    Column("lease_token", String(128), nullable=True),
    Column("lease_expires_at", DateTime(timezone=True), nullable=True),
    Column("worker_service_id", String(128), nullable=True),
    Column("worker_contract_version", String(64), nullable=True),
    Column("started_at", DateTime(timezone=True), nullable=True),
    Column("retrieval_result_fingerprint", String(64), nullable=True),
    Column("generation_attempt_count", Integer, nullable=False, server_default="0"),
    Column("generation_contract_version", String(64), nullable=True),
    Column("generation_started_at", DateTime(timezone=True), nullable=True),
    Column("generation_provider", String(128), nullable=True),
    Column("generation_model", String(256), nullable=True),
    Column("generation_model_version", String(256), nullable=True),
    Column("generation_provider_fingerprint", String(64), nullable=True),
    Column("generation_provider_idempotency_supported", Boolean, nullable=True),
    Column("generation_output_schema_version", String(64), nullable=True),
    Column("generation_output_schema_sha256", String(64), nullable=True),
    Column("generation_input_fingerprint", String(64), nullable=True),
    Column("generation_output_fingerprint", String(64), nullable=True),
    Column("generation_completed_at", DateTime(timezone=True), nullable=True),
    Column("payload_expires_at", DateTime(timezone=True), nullable=False),
    Column("reason_code", String(128), nullable=True),
    Column("failure_stage", String(32), nullable=True),
    Column("evidence_count", Integer, nullable=False, server_default="0"),
    Column("reference_count", Integer, nullable=False, server_default="0"),
    Column("feedback_count", Integer, nullable=False, server_default="0"),
    Column("generation_invoked", Boolean, nullable=False, server_default=text("FALSE")),
    Column("notification_outbox_count", Integer, nullable=False, server_default="0"),
    Column("persisted_run_id", String(36), nullable=True),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    UniqueConstraint(
        "group_id",
        "idempotency_key_hash",
        name="uq_bl_run_job_group_idempotency",
    ),
    UniqueConstraint("job_id", "group_id", name="uq_bl_run_job_scope"),
    UniqueConstraint("persisted_run_id", name="uq_bl_run_job_persisted_run"),
    ForeignKeyConstraint(
        ["persisted_run_id", "group_id"],
        ["baseline_retrieval_run.run_id", "baseline_retrieval_run.group_id"],
        name="fk_bl_run_job_persisted_run_scope",
        ondelete="NO ACTION",
        deferrable=True,
        initially="DEFERRED",
    ),
    CheckConstraint(
        "contract_version = 'baseline-run-job.v1' "
        "AND protocol_version = 'baseline-control-plane.v2' "
        "AND length(protocol_sha256) = 64 "
        "AND length(idempotency_key_hash) = 64 "
        "AND length(intent_hash) = 64 "
        "AND length(parent_processing_identity_fingerprint) = 64",
        name="ck_bl_run_job_contract",
    ),
    CheckConstraint(
        "query_representation = 'raw_git_diff_v1' "
        "AND query_encoding = 'utf-8' "
        "AND query_origin = 'explicit' "
        "AND length(query_sha256) = 64 "
        "AND query_length BETWEEN 1 AND 8000000 "
        "AND query_byte_length BETWEEN 1 AND 8000000",
        name="ck_bl_run_job_query",
    ),
    CheckConstraint(
        "index_format_version = 'baseline-index.v1' "
        "AND length(corpus_manifest_hash) = 64 "
        "AND length(retrieval_config_fingerprint) = 64 "
        "AND length(embedding_fingerprint) = 64 "
        "AND length(index_fingerprint) = 64",
        name="ck_bl_run_job_index",
    ),
    CheckConstraint(
        "state IN ('queued', 'running', 'references_persisted', "
        "'feedback_persisted', 'insufficient', 'retryable_failed', "
        "'terminal_failed', 'blocked', 'cancelled')",
        name="ck_bl_run_job_state",
    ),
    CheckConstraint(
        "(state = 'running' AND lease_token IS NOT NULL "
        "AND lease_expires_at IS NOT NULL) "
        "OR (state <> 'running' AND lease_token IS NULL "
        "AND lease_expires_at IS NULL)",
        name="ck_bl_run_job_lease",
    ),
    CheckConstraint(
        "(worker_service_id IS NULL AND worker_contract_version IS NULL "
        "AND started_at IS NULL) OR "
        "(worker_service_id = 'compair-core-baseline-runner' "
        "AND worker_contract_version = 'baseline-run-worker.v1' "
        "AND started_at IS NOT NULL)",
        name="ck_bl_run_job_worker",
    ),
    CheckConstraint(
        "retrieval_result_fingerprint IS NULL "
        "OR length(retrieval_result_fingerprint) = 64",
        name="ck_bl_run_job_result_fingerprint",
    ),
    CheckConstraint(
        "generation_attempt_count >= 0",
        name="ck_bl_run_job_generation_attempts",
    ),
    CheckConstraint(
        "attempt_count >= 0 AND evidence_count BETWEEN 0 AND 4 "
        "AND reference_count BETWEEN 0 AND 4 "
        "AND feedback_count BETWEEN 0 AND 4 "
        "AND notification_outbox_count BETWEEN 0 AND 1024",
        name="ck_bl_run_job_counts",
    ),
    Index("ix_bl_run_job_group_state", "group_id", "state", "created_at", "job_id"),
    Index("ix_bl_run_job_expiry", "state", "payload_expires_at", "job_id"),
    Index("ix_bl_run_job_source", "source_document_id", "group_id"),
    Index("ix_bl_run_job_publication", "group_id", "index_publication_id"),
)


baseline_run_payload = Table(
    BASELINE_RUN_PAYLOAD_TABLE,
    _metadata,
    Column("job_id", String(36), primary_key=True),
    Column("group_id", String(36), nullable=False),
    Column("payload_schema_version", String(64), nullable=False),
    Column("algorithm", String(32), nullable=False),
    Column("key_id", String(128), nullable=False),
    Column("nonce", LargeBinary(12), nullable=False),
    Column("ciphertext", LargeBinary, nullable=False),
    Column("aad_version", String(64), nullable=False),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    Column("expires_at", DateTime(timezone=True), nullable=False),
    ForeignKeyConstraint(
        ["job_id", "group_id"],
        [
            "baseline_control_run_job.job_id",
            "baseline_control_run_job.group_id",
        ],
        name="fk_bl_run_payload_job_scope",
        ondelete="CASCADE",
    ),
    UniqueConstraint("key_id", "nonce", name="uq_bl_run_payload_key_nonce"),
    CheckConstraint(
        "payload_schema_version = 'baseline-run-protected-payload.v1' "
        "AND algorithm = 'AES-256-GCM' "
        "AND aad_version = 'baseline-run-aad.v1' "
        "AND length(key_id) BETWEEN 1 AND 128 "
        "AND length(nonce) = 12 "
        "AND length(ciphertext) BETWEEN 16 AND 8100000",
        name="ck_bl_run_payload_contract",
    ),
    Index("ix_bl_run_payload_expiry", "expires_at", "job_id"),
)


baseline_worker_instance = Table(
    BASELINE_WORKER_INSTANCE_TABLE,
    _metadata,
    Column("worker_instance_id", String(36), primary_key=True),
    Column("worker_contract_version", String(64), nullable=False),
    Column(
        "supports_corpus_ingestion",
        Boolean,
        nullable=False,
        server_default=text("FALSE"),
    ),
    Column(
        "supports_index_build",
        Boolean,
        nullable=False,
        server_default=text("FALSE"),
    ),
    Column(
        "supports_baseline_run",
        Boolean,
        nullable=False,
        server_default=text("FALSE"),
    ),
    Column(
        "supports_cleanup",
        Boolean,
        nullable=False,
        server_default=text("FALSE"),
    ),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("last_heartbeat_at", DateTime(timezone=True), nullable=False),
    Column("draining", Boolean, nullable=False, server_default=text("FALSE")),
    Column("concurrency_limit", Integer, nullable=False, server_default="1"),
    Column("active_count", Integer, nullable=False, server_default="0"),
    CheckConstraint(
        "worker_contract_version = 'baseline-database-worker.v1'",
        name="ck_bl_db_worker_contract",
    ),
    CheckConstraint(
        "supports_corpus_ingestion = TRUE "
        "AND supports_index_build = TRUE "
        "AND supports_baseline_run = TRUE "
        "AND supports_cleanup = TRUE",
        name="ck_bl_db_worker_supported_jobs",
    ),
    CheckConstraint(
        "concurrency_limit BETWEEN 1 AND 64 "
        "AND active_count BETWEEN 0 AND concurrency_limit",
        name="ck_bl_db_worker_capacity",
    ),
    Index(
        "ix_bl_db_worker_heartbeat",
        "draining",
        "last_heartbeat_at",
        "worker_instance_id",
    ),
)


baseline_worker_attestation = Table(
    BASELINE_WORKER_ATTESTATION_TABLE,
    _metadata,
    Column(
        "worker_instance_id",
        String(36),
        ForeignKey(
            "baseline_database_worker_instance.worker_instance_id",
            ondelete="CASCADE",
            name="fk_bl_db_worker_attestation_instance",
        ),
        primary_key=True,
    ),
    Column("runtime_config_contract_version", String(64), nullable=False),
    Column("runtime_config_fingerprint", String(64), nullable=False),
    Column("embedding_identity_fingerprint", String(64), nullable=False),
    Column("generation_identity_fingerprint", String(64), nullable=False),
    CheckConstraint(
        "runtime_config_contract_version = 'baseline-runtime-config.v1' "
        "AND length(runtime_config_fingerprint) = 64 "
        "AND length(embedding_identity_fingerprint) = 64 "
        "AND length(generation_identity_fingerprint) = 64",
        name="ck_bl_db_worker_attestation_contract",
    ),
    Index(
        "ix_bl_db_worker_attestation_runtime",
        "runtime_config_fingerprint",
        "worker_instance_id",
    ),
)


STAGING_CONTROL_PLANE_TABLES = (
    repository_registration,
    control_job,
    snapshot_staging,
    snapshot_content_part,
)

CONTINUATION_CONTROL_PLANE_TABLES = (
    repository_approval,
    snapshot_continuation_job,
)

INDEX_CONTROL_PLANE_TABLES = (compatible_index_job,)

RUN_CONTROL_PLANE_TABLES = (baseline_run_job, baseline_run_payload)

DATABASE_WORKER_TABLES = (baseline_worker_instance,)

DATABASE_WORKER_ATTESTATION_TABLES = (baseline_worker_attestation,)

CONTROL_PLANE_TABLES = (
    STAGING_CONTROL_PLANE_TABLES
    + CONTINUATION_CONTROL_PLANE_TABLES
    + INDEX_CONTROL_PLANE_TABLES
    + RUN_CONTROL_PLANE_TABLES
    + DATABASE_WORKER_TABLES
    + DATABASE_WORKER_ATTESTATION_TABLES
)


__all__ = [
    "BASELINE_CONTROL_GENERATION_CONTRACT_VERSION",
    "BASELINE_DATABASE_WORKER_CONTRACT_VERSION",
    "BASELINE_RUN_JOB_TABLE",
    "BASELINE_RUN_PAYLOAD_TABLE",
    "BASELINE_RUN_WORKER_CONTRACT_VERSION",
    "BASELINE_RUN_WORKER_SERVICE_ID",
    "BASELINE_WORKER_ATTESTATION_TABLE",
    "BASELINE_WORKER_INSTANCE_TABLE",
    "COMPATIBLE_INDEX_JOB_TABLE",
    "CONTINUATION_CONTROL_PLANE_TABLES",
    "CONTROL_JOB_TABLE",
    "CONTROL_PLANE_SCHEMA_VERSION",
    "CONTROL_PLANE_TABLES",
    "DATABASE_WORKER_ATTESTATION_TABLES",
    "DATABASE_WORKER_TABLES",
    "INDEX_CONTROL_PLANE_TABLES",
    "REPOSITORY_APPROVAL_TABLE",
    "REPOSITORY_REGISTRATION_TABLE",
    "RUN_CONTROL_PLANE_TABLES",
    "SNAPSHOT_CONTENT_PART_TABLE",
    "SNAPSHOT_CONTINUATION_JOB_TABLE",
    "SNAPSHOT_STAGING_TABLE",
    "STAGING_CONTROL_PLANE_TABLES",
    "baseline_run_job",
    "baseline_run_payload",
    "baseline_worker_attestation",
    "baseline_worker_instance",
    "compatible_index_job",
    "control_job",
    "repository_approval",
    "repository_registration",
    "snapshot_content_part",
    "snapshot_continuation_job",
    "snapshot_staging",
]
