"""Trusted full-snapshot ingestion for the durable baseline corpus.

This module accepts an explicit, complete source manifest. It never scans a
filesystem and its contract intentionally has no retrieval-query field.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy.orm import Session

from .corpus import (
    CORPUS_SNAPSHOT_SCHEMA_VERSION,
    CorpusFileInput,
    CorpusGenerationStatus,
    CorpusIngestionStatus,
    CorpusLifecycle,
    RetrievalCorpusGeneration,
    RetrievalCorpusIngestion,
    validate_corpus_file_input,
)


class CorpusIngestionSource(str, Enum):
    TRUSTED_SNAPSHOT_V1 = "trusted_snapshot_v1"


class CorpusGenerationFreshness(str, Enum):
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"
    ACTIVE = "active"
    STALE = "stale"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CorpusRepositoryInput:
    repository_id: str
    repository_name: str
    expected_file_count: int
    repository_revision: str | None = None
    document_id: str | None = None
    document_revision: str | None = None


@dataclass(frozen=True, slots=True)
class CorpusSnapshotInput:
    schema_version: str
    scope_key: str
    generation_version: str
    changed_repository: CorpusRepositoryInput
    sibling_repositories: tuple[CorpusRepositoryInput, ...]
    files: tuple[CorpusFileInput, ...]
    ingestion_source: CorpusIngestionSource
    producer_id: str
    declared_manifest_hash: str
    snapshot_id: str | None = None
    producer_version: str | None = None
    source_revision: str | None = None
    source_manifest_hash: str | None = None

    @classmethod
    def create(
        cls,
        *,
        scope_key: str,
        generation_version: str,
        changed_repository: CorpusRepositoryInput,
        sibling_repositories: tuple[CorpusRepositoryInput, ...],
        files: tuple[CorpusFileInput, ...],
        producer_id: str,
        snapshot_id: str | None = None,
        producer_version: str | None = None,
        source_revision: str | None = None,
        source_manifest_hash: str | None = None,
    ) -> CorpusSnapshotInput:
        """Build a canonical trusted-snapshot declaration and its hash."""

        repositories = {
            repository.repository_id: repository
            for repository in sibling_repositories
        }
        inherited_files = tuple(
            replace(
                item,
                repository_revision=(
                    repositories[item.repository_id].repository_revision
                    if item.repository_id in repositories
                    else item.repository_revision
                ),
                document_id=(
                    item.document_id
                    or (
                        repositories[item.repository_id].document_id
                        if item.repository_id in repositories
                        else None
                    )
                ),
                document_revision=(
                    repositories[item.repository_id].document_revision
                    if item.repository_id in repositories
                    else item.document_revision
                ),
                source_snapshot_id=(
                    item.source_snapshot_id
                    or (
                        repositories[item.repository_id].repository_revision
                        if item.repository_id in repositories
                        else None
                    )
                    or snapshot_id
                ),
            )
            for item in files
        )
        snapshot = cls(
            schema_version=CORPUS_SNAPSHOT_SCHEMA_VERSION,
            scope_key=scope_key,
            generation_version=generation_version,
            changed_repository=changed_repository,
            sibling_repositories=sibling_repositories,
            files=inherited_files,
            ingestion_source=CorpusIngestionSource.TRUSTED_SNAPSHOT_V1,
            producer_id=producer_id,
            declared_manifest_hash="",
            snapshot_id=snapshot_id,
            producer_version=producer_version,
            source_revision=source_revision,
            source_manifest_hash=source_manifest_hash,
        )
        return replace(snapshot, declared_manifest_hash=snapshot_manifest_hash(snapshot))


@dataclass(frozen=True, slots=True)
class ValidatedCorpusSnapshot:
    snapshot: CorpusSnapshotInput
    canonical_manifest_json: str
    canonical_manifest_hash: str


@dataclass(frozen=True, slots=True)
class CorpusIngestionResult:
    corpus_id: str
    generation_id: str
    generation_version: str
    manifest_hash: str
    status: CorpusGenerationFreshness


def _identifier(
    value: str | None,
    label: str,
    *,
    max_length: int,
    optional: bool = False,
) -> str | None:
    if value is None:
        if optional:
            return None
        raise ValueError(f"{label} is required")
    if value != value.strip() or not value:
        raise ValueError(f"{label} must be a canonical non-empty identifier")
    if len(value) > max_length or any(ord(char) < 32 for char in value):
        raise ValueError(f"{label} is invalid")
    return value


def _sha256(value: str | None, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    normalized = (value or "").lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{label} must be a SHA-256 hex digest")
    return normalized


def _repository_payload(repository: CorpusRepositoryInput) -> dict[str, object]:
    return {
        "document_id": repository.document_id,
        "document_revision": repository.document_revision,
        "expected_file_count": repository.expected_file_count,
        "repository_id": repository.repository_id,
        "repository_name": repository.repository_name,
        "repository_revision": repository.repository_revision,
    }


def _file_payload(item: CorpusFileInput) -> dict[str, object]:
    return {
        "byte_size": item.byte_size,
        "content_hash": item.content_hash,
        "document_id": item.document_id,
        "document_revision": item.document_revision,
        "file_state": item.file_state.value,
        "relative_path": item.relative_path,
        "repository_id": item.repository_id,
        "repository_name": item.repository_name,
        "repository_revision": item.repository_revision,
        "skip_reason": item.skip_reason.value if item.skip_reason else None,
        "source_snapshot_id": item.source_snapshot_id,
    }


def canonical_snapshot_manifest_json(snapshot: CorpusSnapshotInput) -> str:
    """Return the deterministic metadata-only manifest representation."""

    siblings = sorted(
        snapshot.sibling_repositories,
        key=lambda repository: (
            repository.repository_name,
            repository.repository_id,
        ),
    )
    files = sorted(
        snapshot.files,
        key=lambda item: (
            item.repository_name,
            item.relative_path,
            item.repository_id,
        ),
    )
    payload = {
        "changed_repository": _repository_payload(snapshot.changed_repository),
        "files": [_file_payload(item) for item in files],
        "schema_version": snapshot.schema_version,
        "sibling_repositories": [
            _repository_payload(repository) for repository in siblings
        ],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def snapshot_manifest_hash(snapshot: CorpusSnapshotInput) -> str:
    return hashlib.sha256(
        canonical_snapshot_manifest_json(snapshot).encode("utf-8")
    ).hexdigest()


def _validate_repository(
    repository: CorpusRepositoryInput,
    *,
    changed: bool,
) -> CorpusRepositoryInput:
    repository_id = _identifier(
        repository.repository_id,
        "repository_id",
        max_length=256,
    )
    repository_name = _identifier(
        repository.repository_name,
        "repository_name",
        max_length=256,
    )
    assert repository_id is not None
    assert repository_name is not None
    if repository_name in {".", ".."} or "/" in repository_name or "\\" in repository_name:
        raise ValueError("repository_name must be a single safe path component")
    if repository.expected_file_count < 0:
        raise ValueError("expected repository file count must be non-negative")
    if changed and repository.expected_file_count != 0:
        raise ValueError("changed repository must not declare candidate files")
    return CorpusRepositoryInput(
        repository_id=repository_id,
        repository_name=repository_name,
        expected_file_count=repository.expected_file_count,
        repository_revision=_identifier(
            repository.repository_revision,
            "repository_revision",
            max_length=256,
            optional=True,
        ),
        document_id=_identifier(
            repository.document_id,
            "document_id",
            max_length=36,
            optional=True,
        ),
        document_revision=_identifier(
            repository.document_revision,
            "document_revision",
            max_length=256,
            optional=True,
        ),
    )


def validate_snapshot_input(snapshot: CorpusSnapshotInput) -> ValidatedCorpusSnapshot:
    """Validate a complete trusted snapshot before any database write."""

    if snapshot.schema_version != CORPUS_SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported corpus snapshot schema version")
    if snapshot.ingestion_source is not CorpusIngestionSource.TRUSTED_SNAPSHOT_V1:
        raise ValueError("unsupported corpus ingestion source")

    scope_key = _identifier(snapshot.scope_key, "scope_key", max_length=256)
    generation_version = _identifier(
        snapshot.generation_version,
        "generation_version",
        max_length=128,
    )
    producer_id = _identifier(snapshot.producer_id, "producer_id", max_length=128)
    assert scope_key is not None
    assert generation_version is not None
    assert producer_id is not None
    changed = _validate_repository(snapshot.changed_repository, changed=True)
    siblings = tuple(
        _validate_repository(repository, changed=False)
        for repository in snapshot.sibling_repositories
    )
    sibling_ids = [repository.repository_id for repository in siblings]
    sibling_names = [repository.repository_name for repository in siblings]
    if len(sibling_ids) != len(set(sibling_ids)):
        raise ValueError("sibling repository identities must be unique")
    if len(sibling_names) != len(set(sibling_names)):
        raise ValueError("sibling repository names must be unique")
    if changed.repository_id in sibling_ids or changed.repository_name in sibling_names:
        raise ValueError("changed repository must not appear in sibling corpus")

    repositories = {repository.repository_id: repository for repository in siblings}
    normalized_files: list[CorpusFileInput] = []
    seen_paths: set[tuple[str, str]] = set()
    counts: Counter[str] = Counter()
    for supplied in snapshot.files:
        item = validate_corpus_file_input(supplied)
        repository = repositories.get(item.repository_id)
        if repository is None:
            raise ValueError("file repository is not a declared sibling")
        if item.repository_name != repository.repository_name:
            raise ValueError("file repository identity and name disagree")
        if item.repository_revision not in {None, repository.repository_revision}:
            raise ValueError("file repository revision disagrees with repository")
        if item.document_id not in {None, repository.document_id}:
            raise ValueError("file document identity disagrees with repository")
        if item.document_revision not in {None, repository.document_revision}:
            raise ValueError("file document revision disagrees with repository")
        key = (item.repository_id, item.relative_path)
        if key in seen_paths:
            raise ValueError("snapshot contains duplicate normalized repository paths")
        seen_paths.add(key)
        counts[item.repository_id] += 1
        normalized_files.append(
            replace(
                item,
                repository_revision=repository.repository_revision,
                document_id=item.document_id or repository.document_id,
                document_revision=repository.document_revision,
                source_snapshot_id=(
                    item.source_snapshot_id
                    or repository.repository_revision
                    or snapshot.snapshot_id
                ),
            )
        )

    for repository in siblings:
        if counts[repository.repository_id] != repository.expected_file_count:
            raise ValueError("declared sibling file count does not match snapshot")
    if sum(repository.expected_file_count for repository in siblings) != len(
        normalized_files
    ):
        raise ValueError("snapshot file count is incomplete")

    normalized = CorpusSnapshotInput(
        schema_version=CORPUS_SNAPSHOT_SCHEMA_VERSION,
        scope_key=scope_key,
        generation_version=generation_version,
        changed_repository=changed,
        sibling_repositories=siblings,
        files=tuple(normalized_files),
        ingestion_source=CorpusIngestionSource.TRUSTED_SNAPSHOT_V1,
        producer_id=producer_id,
        declared_manifest_hash=_sha256(
            snapshot.declared_manifest_hash,
            "declared_manifest_hash",
        )
        or "",
        snapshot_id=_identifier(
            snapshot.snapshot_id,
            "snapshot_id",
            max_length=256,
            optional=True,
        ),
        producer_version=_identifier(
            snapshot.producer_version,
            "producer_version",
            max_length=128,
            optional=True,
        ),
        source_revision=_identifier(
            snapshot.source_revision,
            "source_revision",
            max_length=256,
            optional=True,
        ),
        source_manifest_hash=_sha256(
            snapshot.source_manifest_hash,
            "source_manifest_hash",
            optional=True,
        ),
    )
    canonical_json = canonical_snapshot_manifest_json(normalized)
    canonical_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    if canonical_hash != normalized.declared_manifest_hash:
        raise ValueError("declared corpus manifest hash does not match snapshot")
    return ValidatedCorpusSnapshot(normalized, canonical_json, canonical_hash)


def corpus_generation_freshness(
    session: Session,
    generation_id: str,
) -> CorpusGenerationFreshness:
    ingestion = session.get(RetrievalCorpusIngestion, generation_id)
    if ingestion is not None:
        return CorpusGenerationFreshness(ingestion.status)
    generation = session.get(RetrievalCorpusGeneration, generation_id)
    if generation is None:
        raise ValueError("unknown corpus generation")
    return {
        CorpusGenerationStatus.STAGING.value: CorpusGenerationFreshness.INCOMPLETE,
        CorpusGenerationStatus.VALIDATED.value: CorpusGenerationFreshness.INCOMPLETE,
        CorpusGenerationStatus.ACTIVE.value: CorpusGenerationFreshness.INCOMPLETE,
        CorpusGenerationStatus.SUPERSEDED.value: CorpusGenerationFreshness.STALE,
        CorpusGenerationStatus.FAILED.value: CorpusGenerationFreshness.FAILED,
    }[generation.status]


class CorpusIngestionService:
    """Persist, validate, and atomically activate a trusted full snapshot."""

    def __init__(
        self,
        session_factory: Any,
        *,
        activate_generation: Callable[[Session, str], None] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._activate_generation = (
            activate_generation or CorpusLifecycle.activate_generation
        )

    def ingest(self, supplied: CorpusSnapshotInput) -> CorpusIngestionResult:
        validated = validate_snapshot_input(supplied)
        snapshot = validated.snapshot

        with self._session_factory.begin() as session:
            corpus = CorpusLifecycle.get_or_create_corpus(
                session,
                scope_key=snapshot.scope_key,
                changed_repository_id=snapshot.changed_repository.repository_id,
                source_document_id=snapshot.changed_repository.document_id,
            )
            generation = CorpusLifecycle.stage_generation(
                session,
                corpus=corpus,
                generation_version=snapshot.generation_version,
                files=snapshot.files,
                expected_repository_count=len(snapshot.sibling_repositories),
                expected_file_count=len(snapshot.files),
                source_revision=(
                    snapshot.source_revision
                    or snapshot.changed_repository.repository_revision
                ),
            )
            corpus_id = corpus.corpus_id
            generation_id = generation.generation_id
            session.add(
                RetrievalCorpusIngestion(
                    generation_id=generation_id,
                    snapshot_schema_version=snapshot.schema_version,
                    ingestion_source=snapshot.ingestion_source.value,
                    producer_id=snapshot.producer_id,
                    canonical_manifest_hash=validated.canonical_manifest_hash,
                    canonical_manifest_json=validated.canonical_manifest_json,
                    repository_count=len(snapshot.sibling_repositories),
                    file_count=len(snapshot.files),
                    snapshot_id=snapshot.snapshot_id,
                    producer_version=snapshot.producer_version,
                    source_manifest_hash=snapshot.source_manifest_hash,
                )
            )

        failure_message: str | None = None
        with self._session_factory.begin() as session:
            validation = CorpusLifecycle.validate_generation(session, generation_id)
            if not validation.complete or validation.manifest_hash is None:
                failure_message = (
                    "persisted corpus generation is incomplete "
                    f"({validation.error_code or 'unknown'})"
                )
            else:
                ingestion = session.get(RetrievalCorpusIngestion, generation_id)
                persisted_files = CorpusLifecycle.ordered_files(session, generation_id)
                expected_files = tuple(
                    sorted(
                        snapshot.files,
                        key=lambda item: (
                            item.repository_name,
                            item.relative_path,
                            item.repository_id,
                        ),
                    )
                )
                persisted_matches = len(persisted_files) == len(
                    expected_files
                ) and all(
                    (
                        row.repository_id,
                        row.repository_name,
                        row.relative_path,
                        row.file_state,
                        row.content_hash,
                        row.byte_size,
                        row.document_id,
                        row.source_snapshot_id,
                    )
                    == (
                        item.repository_id,
                        item.repository_name,
                        item.relative_path,
                        item.file_state.value,
                        item.content_hash,
                        item.byte_size,
                        item.document_id,
                        item.source_snapshot_id,
                    )
                    for row, item in zip(persisted_files, expected_files)
                )
                if (
                    ingestion is None
                    or not persisted_matches
                    or ingestion.canonical_manifest_json
                    != validated.canonical_manifest_json
                    or ingestion.canonical_manifest_hash
                    != validated.canonical_manifest_hash
                    or hashlib.sha256(
                        ingestion.canonical_manifest_json.encode("utf-8")
                    ).hexdigest()
                    != ingestion.canonical_manifest_hash
                ):
                    generation = session.get(
                        RetrievalCorpusGeneration,
                        generation_id,
                    )
                    if generation is not None:
                        generation.status = CorpusGenerationStatus.FAILED.value
                        generation.failure_code = "ingestion_manifest_mismatch"
                    if ingestion is not None:
                        ingestion.status = CorpusIngestionStatus.FAILED.value
                        ingestion.failure_code = "ingestion_manifest_mismatch"
                        ingestion.updated_at = datetime.now(timezone.utc)
                    failure_message = (
                        "persisted ingestion manifest does not match snapshot"
                    )

        if failure_message is not None:
            raise ValueError(failure_message)

        with self._session_factory.begin() as session:
            self._activate_generation(session, generation_id)

        return CorpusIngestionResult(
            corpus_id=corpus_id,
            generation_id=generation_id,
            generation_version=snapshot.generation_version,
            manifest_hash=validated.canonical_manifest_hash,
            status=CorpusGenerationFreshness.ACTIVE,
        )
