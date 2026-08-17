"""Frozen durable schema for the immutable baseline evidence bridge.

The tables are deliberately separate from retrieval corpus/index metadata.
Corpus and publication identifiers are copied as immutable provenance rather
than retention foreign keys, so later corpus lifecycle operations cannot erase
evidence already attached to an auditable Core Reference.

This module defines schema only.  It contains no bridge repository, reads,
writes, serializers, retrieval calls, or generation behavior.
"""

from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    text,
)

BRIDGE_SCHEMA_VERSION = "baseline-reference-bridge.v1"
PROVENANCE_SCHEMA_VERSION = "baseline-evidence-provenance.v1"
RENDERER_VERSION = "baseline-evidence-renderer.v1"
SOURCE_SCOPE_VERSION = "baseline-source-scope.v1"
SOURCE_SCOPE_LEGACY_CHUNK = "legacy_chunk"
SOURCE_SCOPE_CONTROL_DOCUMENT = "control_document"

BASELINE_RETRIEVAL_RUN_TABLE = "baseline_retrieval_run"
BASELINE_EVIDENCE_ARTIFACT_TABLE = "baseline_evidence_artifact"
BASELINE_SELECTED_EVIDENCE_TABLE = "baseline_selected_evidence"

_metadata = MetaData()

# Dependency stubs let these tables compile independently of the application
# ORM metadata. Only BASELINE_EVIDENCE_TABLES are ever created from this
# metadata namespace.
Table("group", _metadata, Column("group_id", String(36), primary_key=True))
Table("chunk", _metadata, Column("chunk_id", String(36), primary_key=True))
Table("document", _metadata, Column("document_id", String(36), primary_key=True))


baseline_retrieval_run = Table(
    BASELINE_RETRIEVAL_RUN_TABLE,
    _metadata,
    Column("run_id", String(36), primary_key=True),
    Column(
        "group_id",
        String(36),
        ForeignKey("group.group_id", ondelete="CASCADE", name="fk_bl_run_group"),
        nullable=False,
    ),
    Column(
        "source_chunk_id",
        String(36),
        ForeignKey(
            "chunk.chunk_id", ondelete="SET NULL", name="fk_bl_run_source_chunk"
        ),
        nullable=True,
    ),
    Column(
        "source_document_id",
        String(36),
        ForeignKey(
            "document.document_id",
            ondelete="SET NULL",
            name="fk_bl_run_source_document",
        ),
        nullable=True,
    ),
    Column(
        "source_scope_version",
        String(64),
        nullable=False,
        server_default=SOURCE_SCOPE_VERSION,
    ),
    Column(
        "source_scope",
        String(32),
        nullable=False,
        server_default=SOURCE_SCOPE_LEGACY_CHUNK,
    ),
    Column("idempotency_key", String(256), nullable=False),
    Column("bridge_schema_version", String(64), nullable=False),
    Column("provenance_schema_version", String(64), nullable=False),
    Column("renderer_version", String(64), nullable=False),
    Column("request_id", String(128), nullable=False),
    Column("result_schema_version", String(64), nullable=False),
    Column("retrieval_status", String(16), nullable=False),
    Column("engine", String(64), nullable=False),
    Column("engine_version", String(64), nullable=False),
    Column("config_fingerprint", String(64), nullable=False),
    Column("query_kind", String(64), nullable=False),
    Column("query_sha256", String(64), nullable=False),
    Column("query_length", Integer, nullable=False),
    Column("query_origin", String(32), nullable=False),
    Column("corpus_scope_key", String(256), nullable=False),
    Column("corpus_id", String(36), nullable=False),
    Column("corpus_generation_id", String(36), nullable=False),
    Column("corpus_generation_version", String(128), nullable=False),
    Column("corpus_manifest_hash", String(64), nullable=False),
    Column("index_publication_fingerprint", String(64), nullable=False),
    Column("index_published_at", DateTime(timezone=True), nullable=False),
    Column("index_id", String(36), nullable=False),
    Column("index_version", String(128), nullable=False),
    Column("index_schema_version", String(64), nullable=False),
    Column("index_fingerprint", String(64), nullable=False),
    Column("embedding_provider", String(128), nullable=False),
    Column("embedding_model", String(256), nullable=False),
    Column("embedding_revision", String(256), nullable=False),
    Column("embedding_dimension", Integer, nullable=False),
    Column("embedding_fingerprint", String(64), nullable=False),
    Column("authorization_scope_version", String(64), nullable=False),
    Column("authorization_scope_hash", String(64), nullable=False),
    Column("candidate_count", Integer, nullable=False),
    Column("retrieved_count", Integer, nullable=False),
    Column("filtered_count", Integer, nullable=False),
    Column("duplicate_count", Integer, nullable=False),
    Column("refill_count", Integer, nullable=False),
    Column("selected_count", Integer, nullable=False),
    Column("evidence_character_count", Integer, nullable=False),
    Column("underfilled", Boolean, nullable=False),
    Column("generation_state", String(16), nullable=False, server_default="pending"),
    Column("generation_error_code", String(128), nullable=True),
    Column("generation_attempt_count", Integer, nullable=False, server_default="0"),
    Column("generation_lease_expires_at", DateTime(timezone=True), nullable=True),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    Column("generation_completed_at", DateTime(timezone=True), nullable=True),
    UniqueConstraint("group_id", "idempotency_key", name="uq_bl_run_group_intent"),
    UniqueConstraint("run_id", "group_id", name="uq_bl_run_id_group"),
    CheckConstraint(
        "length(trim(idempotency_key)) > 0 AND idempotency_key <> query_sha256",
        name="ck_bl_run_idempotency",
    ),
    CheckConstraint(
        "bridge_schema_version = 'baseline-reference-bridge.v1' "
        "AND provenance_schema_version = 'baseline-evidence-provenance.v1' "
        "AND renderer_version = 'baseline-evidence-renderer.v1'",
        name="ck_bl_run_versions",
    ),
    CheckConstraint(
        "source_scope_version = 'baseline-source-scope.v1' "
        "AND source_scope IN ('legacy_chunk', 'control_document')",
        name="ck_bl_run_source_scope",
    ),
    CheckConstraint(
        "retrieval_status = 'ok' AND query_origin = 'explicit' "
        "AND query_length > 0 AND length(query_sha256) = 64",
        name="ck_bl_run_query",
    ),
    CheckConstraint(
        "length(config_fingerprint) = 64 "
        "AND length(corpus_manifest_hash) = 64 "
        "AND length(index_publication_fingerprint) = 64 "
        "AND length(index_fingerprint) = 64 "
        "AND length(embedding_fingerprint) = 64 "
        "AND length(authorization_scope_hash) = 64",
        name="ck_bl_run_hashes",
    ),
    CheckConstraint("embedding_dimension > 0", name="ck_bl_run_embedding_dimension"),
    CheckConstraint(
        "candidate_count >= 0 AND retrieved_count >= 0 "
        "AND filtered_count >= 0 AND duplicate_count >= 0 "
        "AND refill_count >= 0 AND selected_count BETWEEN 1 AND 4 "
        "AND evidence_character_count BETWEEN 1 AND 16000",
        name="ck_bl_run_counts",
    ),
    CheckConstraint(
        "generation_state IN ('pending', 'generating', 'completed', 'failed') "
        "AND generation_attempt_count >= 0",
        name="ck_bl_run_generation",
    ),
    Index("ix_bl_run_group_created", "group_id", "created_at"),
    Index("ix_bl_run_source_chunk", "source_chunk_id"),
    Index("ix_bl_run_source_document", "source_document_id"),
    Index("ix_bl_run_corpus_generation", "corpus_generation_id"),
    Index("ix_bl_run_index_publication", "index_publication_fingerprint"),
)


baseline_evidence_artifact = Table(
    BASELINE_EVIDENCE_ARTIFACT_TABLE,
    _metadata,
    Column("artifact_id", String(36), primary_key=True),
    Column(
        "group_id",
        String(36),
        ForeignKey("group.group_id", ondelete="CASCADE", name="fk_bl_artifact_group"),
        nullable=False,
    ),
    Column("artifact_key", String(64), nullable=False),
    Column("bridge_schema_version", String(64), nullable=False),
    Column("provenance_schema_version", String(64), nullable=False),
    Column("repository_id", String(256), nullable=False),
    Column("repository_name", String(256), nullable=False),
    Column("relative_path", String(1024), nullable=False),
    Column("corpus_id", String(36), nullable=False),
    Column("corpus_file_id", String(36), nullable=False),
    Column("corpus_generation_id", String(36), nullable=False),
    Column("corpus_generation_version", String(128), nullable=False),
    Column("corpus_manifest_hash", String(64), nullable=False),
    Column("index_publication_fingerprint", String(64), nullable=False),
    Column("index_published_at", DateTime(timezone=True), nullable=False),
    Column("index_id", String(36), nullable=False),
    Column("index_document_id", String(36), nullable=False),
    Column("index_fingerprint", String(64), nullable=False),
    Column("indexed_document_hash", String(64), nullable=False),
    Column(
        "source_document_id",
        String(36),
        ForeignKey(
            "document.document_id",
            ondelete="SET NULL",
            name="fk_bl_artifact_source_document",
        ),
        nullable=True,
    ),
    Column("source_snapshot_id", String(256), nullable=True),
    Column("complete_content", Text, nullable=False),
    Column("whole_file_content_hash", String(64), nullable=False),
    Column("byte_size", Integer, nullable=False),
    Column("character_count", Integer, nullable=False),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    UniqueConstraint("group_id", "artifact_key", name="uq_bl_artifact_group_key"),
    UniqueConstraint("artifact_id", "group_id", name="uq_bl_artifact_id_group"),
    CheckConstraint(
        "length(artifact_key) = 64 AND length(corpus_manifest_hash) = 64 "
        "AND length(index_publication_fingerprint) = 64 "
        "AND length(index_fingerprint) = 64 "
        "AND length(indexed_document_hash) = 64 "
        "AND length(whole_file_content_hash) = 64",
        name="ck_bl_artifact_hashes",
    ),
    CheckConstraint(
        "bridge_schema_version = 'baseline-reference-bridge.v1' "
        "AND provenance_schema_version = 'baseline-evidence-provenance.v1'",
        name="ck_bl_artifact_versions",
    ),
    CheckConstraint(
        "length(repository_id) > 0 AND length(repository_name) > 0 "
        "AND length(relative_path) > 0",
        name="ck_bl_artifact_identity",
    ),
    CheckConstraint(
        "byte_size >= 0 AND character_count >= 0 "
        "AND length(complete_content) = character_count",
        name="ck_bl_artifact_content",
    ),
    Index("ix_bl_artifact_group_content", "group_id", "whole_file_content_hash"),
    Index(
        "ix_bl_artifact_repository_path", "group_id", "repository_id", "relative_path"
    ),
    Index("ix_bl_artifact_generation", "corpus_generation_id"),
    Index("ix_bl_artifact_index_document", "index_id", "index_document_id"),
)


baseline_selected_evidence = Table(
    BASELINE_SELECTED_EVIDENCE_TABLE,
    _metadata,
    Column("selected_evidence_id", String(36), primary_key=True),
    Column("group_id", String(36), nullable=False),
    Column("run_id", String(36), nullable=False),
    Column("artifact_id", String(36), nullable=False),
    Column("ordinal", Integer, nullable=False),
    Column("fused_rank", Integer, nullable=False),
    Column("selected_content", Text, nullable=False),
    Column("selected_content_hash", String(64), nullable=False),
    Column("selected_character_count", Integer, nullable=False),
    Column("ranking_truncated", Boolean, nullable=False),
    Column("budget_truncated", Boolean, nullable=False),
    Column("bm25_score", Float(precision=53), nullable=False),
    Column("bm25_rank", Integer, nullable=False),
    Column("dense_score", Float(precision=53), nullable=False),
    Column("dense_rank", Integer, nullable=False),
    Column("rrf_score", Float(precision=53), nullable=False),
    Column("renderer_version", String(64), nullable=False),
    Column("renderer_output", Text, nullable=False),
    Column("renderer_output_hash", String(64), nullable=False),
    Column("renderer_output_character_count", Integer, nullable=False),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    ),
    ForeignKeyConstraint(
        ["run_id", "group_id"],
        ["baseline_retrieval_run.run_id", "baseline_retrieval_run.group_id"],
        name="fk_bl_selected_run_scope",
        ondelete="CASCADE",
    ),
    ForeignKeyConstraint(
        ["artifact_id", "group_id"],
        [
            "baseline_evidence_artifact.artifact_id",
            "baseline_evidence_artifact.group_id",
        ],
        name="fk_bl_selected_artifact_scope",
        ondelete="NO ACTION",
        deferrable=True,
        initially="DEFERRED",
    ),
    UniqueConstraint("run_id", "ordinal", name="uq_bl_selected_run_ordinal"),
    UniqueConstraint("run_id", "artifact_id", name="uq_bl_selected_run_artifact"),
    UniqueConstraint(
        "run_id",
        "selected_content_hash",
        name="uq_bl_selected_run_content",
    ),
    CheckConstraint("ordinal BETWEEN 1 AND 4", name="ck_bl_selected_ordinal"),
    CheckConstraint("fused_rank > 0", name="ck_bl_selected_fused_rank"),
    CheckConstraint(
        "length(selected_content) > 0 "
        "AND length(selected_content) = selected_character_count "
        "AND length(selected_content_hash) = 64",
        name="ck_bl_selected_content",
    ),
    CheckConstraint(
        "bm25_rank > 0 AND dense_rank > 0 "
        "AND bm25_score BETWEEN 0.0 AND 1.0e308 "
        "AND dense_score BETWEEN -1.0e308 AND 1.0e308 "
        "AND rrf_score BETWEEN 0.0 AND 1.0e308",
        name="ck_bl_selected_scores",
    ),
    CheckConstraint(
        "renderer_version = 'baseline-evidence-renderer.v1' "
        "AND length(renderer_output) > 0 "
        "AND length(renderer_output) = renderer_output_character_count "
        "AND length(renderer_output_hash) = 64",
        name="ck_bl_selected_renderer",
    ),
    Index("ix_bl_selected_run_order", "run_id", "ordinal"),
    Index("ix_bl_selected_artifact", "artifact_id"),
)


BASELINE_EVIDENCE_TABLES = (
    baseline_retrieval_run,
    baseline_evidence_artifact,
    baseline_selected_evidence,
)


__all__ = [
    "BASELINE_EVIDENCE_ARTIFACT_TABLE",
    "BASELINE_EVIDENCE_TABLES",
    "BASELINE_RETRIEVAL_RUN_TABLE",
    "BASELINE_SELECTED_EVIDENCE_TABLE",
    "BRIDGE_SCHEMA_VERSION",
    "PROVENANCE_SCHEMA_VERSION",
    "RENDERER_VERSION",
    "SOURCE_SCOPE_CONTROL_DOCUMENT",
    "SOURCE_SCOPE_LEGACY_CHUNK",
    "SOURCE_SCOPE_VERSION",
    "baseline_evidence_artifact",
    "baseline_retrieval_run",
    "baseline_selected_evidence",
]
