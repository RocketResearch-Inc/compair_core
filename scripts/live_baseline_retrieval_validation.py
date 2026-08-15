"""Build and query a minimal live persistent ``baseline_v1`` publication."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from compair_core import db as core_db
from compair_core.compair.retrieval.corpus import (
    CorpusFileInput,
    ensure_retrieval_corpus_schema,
)
from compair_core.compair.retrieval.embedding import (
    BASELINE_EMBEDDING_HTTP_PROVIDER,
    assess_baseline_embedding,
    build_configured_baseline_index,
    create_configured_persistent_baseline_retriever,
)
from compair_core.compair.retrieval.factory import retrieve_reference_evidence
from compair_core.compair.retrieval.ingestion import (
    CorpusIngestionService,
    CorpusRepositoryInput,
    CorpusSnapshotInput,
)
from compair_core.compair.retrieval.types import (
    RetrievalQueryOrigin,
    RetrievalRequest,
    RetrievalStatus,
)

MODEL = "BAAI/bge-small-en-v1.5"
DIMENSION = 384
SCOPE_KEY = "group:phase2b2d1-live"
GENERATION_VERSION = "phase2b2d1-live-generation-1"
INDEX_VERSION = "phase2b2d1-live-index-1"
CHANGED_REPOSITORY_ID = "repo-live-changed"
QUERY = "live-private-query-sentinel authorization policy change"


def _settings(endpoint: str, revision: str) -> SimpleNamespace:
    return SimpleNamespace(
        baseline_embedding_provider="http",
        baseline_embedding_endpoint=endpoint,
        baseline_embedding_model=MODEL,
        baseline_embedding_revision=revision,
        baseline_embedding_dimension=DIMENSION,
        baseline_embedding_timeout_seconds=60.0,
        baseline_embedding_batch_size=2,
        baseline_embedding_allow_insecure_loopback=True,
    )


def _snapshot() -> CorpusSnapshotInput:
    files = (
        CorpusFileInput.supported_text(
            repository_id="repo-live-peer",
            repository_name="peer-library",
            relative_path="src/authorization.py",
            content="Authorization policy validates roles and denies unknown access.",
        ),
        CorpusFileInput.supported_text(
            repository_id="repo-live-peer",
            repository_name="peer-library",
            relative_path="docs/audit.md",
            content="Audit events record approved policy decisions in stable order.",
        ),
    )
    return CorpusSnapshotInput.create(
        scope_key=SCOPE_KEY,
        generation_version=GENERATION_VERSION,
        changed_repository=CorpusRepositoryInput(
            repository_id=CHANGED_REPOSITORY_ID,
            repository_name="changed-library",
            expected_file_count=0,
            repository_revision="changed-live-revision-1",
            document_id="document-live-changed",
            document_revision="document-live-changed-revision-1",
        ),
        sibling_repositories=(
            CorpusRepositoryInput(
                repository_id="repo-live-peer",
                repository_name="peer-library",
                expected_file_count=len(files),
                repository_revision="peer-live-revision-1",
                document_id="document-live-peer",
                document_revision="document-live-peer-revision-1",
            ),
        ),
        files=files,
        producer_id="phase2b2d1-live-validation",
        producer_version="1.0",
        snapshot_id="phase2b2d1-live-snapshot-1",
        source_revision="changed-live-revision-1",
        source_manifest_hash=hashlib.sha256(
            b"phase2b2d1-live-source-manifest"
        ).hexdigest(),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--database", type=Path, required=True)
    args = parser.parse_args()

    database = args.database.expanduser().resolve()
    if database.exists():
        raise SystemExit("validation database already exists; choose a fresh path")
    database.parent.mkdir(parents=True, exist_ok=True)

    settings = _settings(args.endpoint, args.revision)
    capability = assess_baseline_embedding(settings)
    if not capability.available:
        print(json.dumps({"capability": capability.as_dict()}, sort_keys=True))
        return 1

    engine = core_db.create_engine(f"sqlite:///{database}")
    ensure_retrieval_corpus_schema(engine)
    SessionMaker = core_db.sessionmaker(engine, expire_on_commit=False)
    try:
        corpus = CorpusIngestionService(SessionMaker).ingest(_snapshot())
        build = build_configured_baseline_index(
            SessionMaker,
            settings=settings,
            generation_id=corpus.generation_id,
            index_version=INDEX_VERSION,
        )
        retriever = create_configured_persistent_baseline_retriever(
            SessionMaker,
            settings=settings,
        )
        request = RetrievalRequest(
            request_id="phase2b2d1-live-request",
            changed_repository=None,
            repository_roots=(),
            corpus_version=GENERATION_VERSION,
            retrieval_query=QUERY,
            retrieval_query_origin=RetrievalQueryOrigin.EXPLICIT,
            corpus_complete=True,
            corpus_scope_key=SCOPE_KEY,
            changed_repository_id=CHANGED_REPOSITORY_ID,
        )
        result = retrieve_reference_evidence(
            engine_name="baseline_v1",
            baseline_retriever=retriever,
            request=request,
        )
    finally:
        engine.dispose()

    output = {
        "capability": capability.as_dict(),
        "corpus": {
            "generation_id": corpus.generation_id,
            "generation_version": corpus.generation_version,
            "manifest_hash": corpus.manifest_hash,
            "status": corpus.status.value,
        },
        "index": {
            "index_id": build.index_id,
            "index_version": build.index_version,
            "document_count": build.document_count,
            "status": build.status.value,
        },
        "retrieval": {
            "status": result.status.value,
            "engine": result.engine,
            "engine_version": result.engine_version,
            "index_id": result.index_id,
            "embedding_provider": result.embedding_provider,
            "embedding_model": result.embedding_model,
            "embedding_revision": result.embedding_revision,
            "embedding_dimension": result.embedding_dimension,
            "embedding_fingerprint": result.embedding_fingerprint,
            "fallback_engine": result.fallback_engine,
            "query_provenance": {
                "sha256": result.query_provenance.sha256,
                "length": result.query_provenance.length,
                "origin": result.query_provenance.origin.value,
            },
            "candidate_paths": [item.path for item in result.candidates],
            "evidence_paths": [item.path for item in result.evidence],
        },
        "checks": {
            "compatible_publication_reused": result.index_id == build.index_id,
            "no_legacy_or_hash_fallback": (
                result.embedding_provider == BASELINE_EMBEDDING_HTTP_PROVIDER
                and result.fallback_engine is None
            ),
            "query_text_absent_from_result": QUERY not in repr(result),
        },
    }
    serialized = json.dumps(output, sort_keys=True)
    assert result.status is RetrievalStatus.OK
    assert QUERY not in serialized
    assert all(output["checks"].values())
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
