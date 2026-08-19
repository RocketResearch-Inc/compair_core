"""Manual orchestration for one protected document-level baseline run.

This module exposes no HTTP or task entry point.  The only execution argument
is an opaque run-job UUID; protected query material is opened by the existing
lease-owning executor and is never accepted by this boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from uuid import UUID

from sqlalchemy import inspect, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from ...baseline_control_plane_schema import (
    baseline_run_job,
    compatible_index_job,
)
from ...baseline_generation.ollama import (
    OllamaBaselineGenerationProvider,
    OllamaGenerationConfig,
    validate_baseline_generation_endpoint,
)
from ...baseline_generation.profile import (
    ACCELERATED_GENERATION_TIMEOUT_SECONDS,
    required_generation_lease_seconds,
)
from ...schema_migrations import (
    CORE_SCHEMA_MIGRATIONS,
    MIGRATION_TABLE_NAME,
    schema_migration_table,
)
from .control_document_scope import (
    ControlDocumentCorpusScopeError,
    control_document_corpus_identity,
)
from .control_plane_v2 import (
    V2IndexPublication,
    V2RunCapability,
    not_ready_run_capability,
    ready_run_capability,
)
from .corpus import (
    RetrievalBaselineIndexBuild,
    RetrievalBaselineIndexPublication,
    RetrievalCorpus,
)
from .embedding import (
    BaselineEmbeddingAdapterError,
    create_configured_persistent_baseline_retriever,
    require_configured_baseline_embedding_adapter,
)
from .generation import (
    GENERATION_OUTPUT_SCHEMA_SHA256,
    GENERATION_OUTPUT_SCHEMA_VERSION,
    BaselineControlGenerationReceipt,
    BaselineGenerationBusyError,
    BaselineGenerationError,
    BaselineGenerationProvider,
    BaselineGenerationService,
    ReviewerBaselineGenerationProvider,
)
from .persistent import published_index_fingerprint
from .run_executor import (
    BaselineDocumentRunExecutor,
    BaselineRunExecutionOutcome,
    BaselineRunExecutorError,
    InternalBaselineRunWorkerIdentity,
)
from .run_jobs import (
    BaselineRunJobError,
    BaselineRunJobService,
    keyring_from_settings,
)

RUN_OPERATOR_CONTRACT_VERSION = "baseline-run-manual-operator.v1"
RUN_SCHEMA_MIGRATION_ID = "0012_baseline_control_generation_v1"
_RUN_REQUIRED_TABLES = frozenset(
    {
        MIGRATION_TABLE_NAME,
        "baseline_control_run_job",
        "baseline_control_run_payload",
        "baseline_control_job",
        "baseline_compatible_index_job",
        "baseline_control_repository_registration",
        "baseline_control_repository_approval",
        "baseline_retrieval_run",
        "baseline_evidence_artifact",
        "baseline_selected_evidence",
        "baseline_notification_outbox",
        "feedback",
        "reference",
        "retrieval_corpus",
        "retrieval_corpus_generation",
        "retrieval_corpus_ingestion",
        "retrieval_index_state",
        "retrieval_baseline_index_build",
        "retrieval_baseline_index_publication",
    }
)


class BaselineRunRuntimeError(RuntimeError):
    """Sanitized runtime/readiness failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class BaselineManualRunOutcome:
    job_id: str
    state: str
    persisted_run_id: str | None
    evidence_count: int
    reference_count: int
    feedback_count: int
    notification_outbox_count: int
    replayed: bool
    reason_code: str | None = None


class BaselineManualRunOperator:
    """Compose the existing retrieval and generation state machines."""

    contract_version = RUN_OPERATOR_CONTRACT_VERSION

    def __init__(
        self,
        engine: Engine,
        *,
        executor: BaselineDocumentRunExecutor,
        generation: BaselineGenerationService,
        provider: BaselineGenerationProvider,
    ) -> None:
        self.engine = engine
        self.sessions = sessionmaker(engine, expire_on_commit=False)
        self.executor = executor
        self.generation = generation
        self.provider = provider

    @staticmethod
    def _job_id(value: str) -> str:
        try:
            parsed = UUID(value)
        except (AttributeError, ValueError):
            raise BaselineRunRuntimeError("job_not_found_or_forbidden") from None
        if str(parsed) != value.lower():
            raise BaselineRunRuntimeError("job_not_found_or_forbidden")
        return str(parsed)

    def _snapshot(self, job_id: str) -> Mapping[str, Any]:
        with self.sessions() as session:
            row = (
                session.execute(
                    select(baseline_run_job).where(baseline_run_job.c.job_id == job_id)
                )
                .mappings()
                .one_or_none()
            )
        if row is None:
            raise BaselineRunRuntimeError("job_not_found_or_forbidden")
        return dict(row)

    @staticmethod
    def _outcome(row: Mapping[str, Any], *, replayed: bool) -> BaselineManualRunOutcome:
        return BaselineManualRunOutcome(
            job_id=str(row["job_id"]),
            state=str(row["state"]),
            persisted_run_id=(
                str(row["persisted_run_id"])
                if row["persisted_run_id"] is not None
                else None
            ),
            evidence_count=int(row["evidence_count"]),
            reference_count=int(row["reference_count"]),
            feedback_count=int(row["feedback_count"]),
            notification_outbox_count=int(row["notification_outbox_count"]),
            replayed=replayed,
            reason_code=(
                str(row["reason_code"]) if row["reason_code"] is not None else None
            ),
        )

    def process(self, job_id: str) -> BaselineManualRunOutcome:
        """Process one opaque job through the next durable boundary."""

        job_id = self._job_id(job_id)
        before = self._snapshot(job_id)
        state = str(before["state"])
        if state == "feedback_persisted":
            receipt = self.generation.generate_control(job_id, self.provider)
            return self._from_generation(receipt)
        if state in {
            "insufficient",
            "terminal_failed",
            "blocked",
            "cancelled",
        }:
            return self._outcome(before, replayed=True)

        if before["persisted_run_id"] is not None:
            try:
                generated = self.generation.generate_control(job_id, self.provider)
            except (BaselineGenerationBusyError, BaselineGenerationError) as exc:
                raise BaselineRunRuntimeError(
                    getattr(exc, "code", "internal_failure")
                ) from None
            return self._from_generation(generated)

        if state != "references_persisted":
            try:
                retrieval = self.executor.execute(job_id)
            except BaselineRunExecutorError as exc:
                raise BaselineRunRuntimeError(exc.code) from None
            if retrieval.state != "references_persisted":
                return self._from_retrieval(retrieval)

        try:
            generated = self.generation.generate_control(job_id, self.provider)
        except (BaselineGenerationBusyError, BaselineGenerationError) as exc:
            raise BaselineRunRuntimeError(
                getattr(exc, "code", "internal_failure")
            ) from None
        return self._from_generation(generated)

    def _from_retrieval(
        self, receipt: BaselineRunExecutionOutcome
    ) -> BaselineManualRunOutcome:
        row = self._snapshot(receipt.job_id)
        return self._outcome(row, replayed=receipt.replayed)

    def _from_generation(
        self, receipt: BaselineControlGenerationReceipt
    ) -> BaselineManualRunOutcome:
        row = self._snapshot(receipt.job_id)
        return self._outcome(row, replayed=receipt.replayed)


def _configured_generation_provider(settings: Any) -> BaselineGenerationProvider:
    mode = str(getattr(settings, "baseline_generation_provider", "disabled"))
    try:
        if mode == "ollama":
            native = OllamaBaselineGenerationProvider(
                OllamaGenerationConfig.from_settings(settings)
            )
            native.attest()
            provider: BaselineGenerationProvider = native
        elif mode == "http":
            endpoint = getattr(settings, "baseline_generation_endpoint", None)
            model = getattr(settings, "baseline_generation_model", None)
            if (
                not isinstance(endpoint, str)
                or not endpoint
                or not isinstance(model, str)
            ):
                raise ValueError("baseline HTTP generation is not configured")
            endpoint = validate_baseline_generation_endpoint(
                endpoint,
                allow_loopback_http=bool(
                    getattr(
                        settings,
                        "baseline_generation_allow_loopback_http",
                        False,
                    )
                ),
                require_root_path=False,
            )
            reviewer = SimpleNamespace(
                provider="http",
                custom_endpoint=endpoint,
                model=model,
                model_version=getattr(
                    settings,
                    "baseline_generation_model_version",
                    None,
                ),
                baseline_timeout_seconds=float(
                    getattr(settings, "baseline_generation_timeout_seconds", 60.0)
                ),
            )
            provider = ReviewerBaselineGenerationProvider(reviewer)
        else:
            raise ValueError("baseline generation provider is disabled")
    except Exception:  # noqa: BLE001 - configuration boundary is sanitized
        raise BaselineRunRuntimeError("worker_unavailable") from None
    if GENERATION_OUTPUT_SCHEMA_VERSION != "baseline-generation-output.v2":
        raise BaselineRunRuntimeError("worker_unavailable")
    if len(GENERATION_OUTPUT_SCHEMA_SHA256) != 64:
        raise BaselineRunRuntimeError("worker_unavailable")
    return provider


class BaselineRunRuntime:
    """Configured manual runtime plus read-only truthfulness checks."""

    def __init__(
        self,
        engine: Engine,
        settings: Any,
        *,
        provider_factory: Callable[[], BaselineGenerationProvider] | None = None,
    ) -> None:
        self.engine = engine
        self.settings = settings
        self.sessions = sessionmaker(engine, expire_on_commit=False)
        try:
            keyring = keyring_from_settings(settings)
            self.jobs = BaselineRunJobService.from_settings(engine, settings)
        except BaselineRunJobError:
            raise BaselineRunRuntimeError("capability_unavailable") from None
        try:
            self.embedding_adapter = require_configured_baseline_embedding_adapter(
                settings
            )
        except BaselineEmbeddingAdapterError as exc:
            code = (
                "embedding_identity_mismatch"
                if any(
                    marker in exc.code
                    for marker in ("identity", "model", "revision", "dimension")
                )
                else "embedding_unavailable"
            )
            raise BaselineRunRuntimeError(code) from None
        self.provider = (
            provider_factory()
            if provider_factory is not None
            else _configured_generation_provider(settings)
        )
        provider_timeout_seconds = float(
            getattr(
                settings,
                "baseline_generation_timeout_seconds",
                ACCELERATED_GENERATION_TIMEOUT_SECONDS,
            )
        )
        executor = BaselineDocumentRunExecutor(
            engine,
            identity=InternalBaselineRunWorkerIdentity.create("manual-operator"),
            keyring=keyring,
            retriever_factory=lambda: create_configured_persistent_baseline_retriever(
                self.sessions,
                settings=settings,
            ),
        )
        self.operator = BaselineManualRunOperator(
            engine,
            executor=executor,
            generation=BaselineGenerationService(
                self.sessions,
                lease_seconds=required_generation_lease_seconds(
                    provider_timeout_seconds
                ),
                provider_timeout_seconds=provider_timeout_seconds,
                notifications_enabled=False,
            ),
            provider=self.provider,
        )

    def _validate_schema_and_database(self) -> None:
        try:
            names = set(inspect(self.engine).get_table_names())
            if not _RUN_REQUIRED_TABLES <= names:
                raise BaselineRunRuntimeError("capability_unavailable")
            required_migrations = tuple(
                item
                for item in CORE_SCHEMA_MIGRATIONS
                if item.migration_id <= RUN_SCHEMA_MIGRATION_ID
            )
            if (
                not required_migrations
                or required_migrations[-1].migration_id != RUN_SCHEMA_MIGRATION_ID
            ):
                raise BaselineRunRuntimeError("capability_unavailable")
            with self.engine.connect() as connection:
                rows = {
                    str(row["migration_id"]): row
                    for row in (
                        connection.execute(
                            select(schema_migration_table).where(
                                schema_migration_table.c.migration_id.in_(
                                    [item.migration_id for item in required_migrations]
                                )
                            )
                        )
                        .mappings()
                        .all()
                    )
                }
                for migration in required_migrations:
                    row = rows.get(migration.migration_id)
                    if (
                        row is None
                        or row["state"] != "applied"
                        or row["checksum"] != migration.checksum
                    ):
                        raise BaselineRunRuntimeError("capability_unavailable")
                    migration.validate(connection)
                connection.exec_driver_sql("SELECT 1").scalar_one()
        except BaselineRunRuntimeError:
            raise
        except (SQLAlchemyError, StopIteration):
            raise BaselineRunRuntimeError("capability_unavailable") from None
        except Exception:  # noqa: BLE001 - validators expose no backend details
            raise BaselineRunRuntimeError("capability_unavailable") from None

    def _has_authorized_publication(
        self,
        *,
        group_id: str,
        caller_user_id: str,
        embedding_fingerprint: str,
    ) -> bool:
        with self.sessions() as session:
            corpora = session.scalars(
                select(RetrievalCorpus).order_by(RetrievalCorpus.corpus_id)
            ).all()
            for corpus in corpora:
                if (
                    corpus.source_document_id is None
                    or corpus.active_generation_id is None
                    or corpus.changed_repository_id is None
                ):
                    continue
                try:
                    corpus_identity = control_document_corpus_identity(
                        group_id=group_id,
                        changed_repository_registration_id=(
                            corpus.changed_repository_id
                        ),
                        source_document_id=corpus.source_document_id,
                    )
                except ControlDocumentCorpusScopeError:
                    continue
                if not corpus_identity.matches_stored_corpus(
                    scope_key=corpus.scope_key,
                    changed_repository_id=corpus.changed_repository_id,
                    source_document_id=corpus.source_document_id,
                ):
                    continue
                publication = session.get(
                    RetrievalBaselineIndexPublication, corpus.corpus_id
                )
                if publication is None or publication.index_id is None:
                    continue
                build = session.get(RetrievalBaselineIndexBuild, publication.index_id)
                if (
                    build is None
                    or build.embedding_fingerprint != embedding_fingerprint
                ):
                    continue
                extension = (
                    session.execute(
                        select(compatible_index_job).where(
                            compatible_index_job.c.group_id == group_id,
                            compatible_index_job.c.result_index_id == build.index_id,
                        )
                    )
                    .mappings()
                    .one_or_none()
                )
                if extension is None:
                    continue
                try:
                    identity = V2IndexPublication(
                        index_publication_id=build.index_id,
                        corpus_generation_id=build.generation_id,
                        corpus_manifest_hash=str(extension["corpus_manifest_hash"]),
                        index_format_version=str(extension["index_format_version"]),
                        tokenizer_version=str(extension["tokenizer_version"]),
                        retrieval_config_fingerprint=(
                            str(extension["retrieval_config_fingerprint"])
                        ),
                        embedding_fingerprint=str(extension["embedding_fingerprint"]),
                        index_fingerprint=published_index_fingerprint(build),
                    )
                    self.jobs._authorize_publication(
                        session,
                        caller_user_id=caller_user_id,
                        group_id=group_id,
                        source_document_id=corpus.source_document_id,
                        changed_registration_id=corpus.changed_repository_id,
                        publication=identity,
                        lock=False,
                    )
                except (BaselineRunJobError, ValueError):
                    continue
                return True
        return False

    def capability(self, *, group_id: str, caller_user_id: str) -> V2RunCapability:
        try:
            self._validate_schema_and_database()
            identity = self.embedding_adapter.attest()
        except BaselineEmbeddingAdapterError as exc:
            code = (
                "embedding_identity_mismatch"
                if any(
                    marker in exc.code
                    for marker in ("identity", "model", "revision", "dimension")
                )
                else "embedding_unavailable"
            )
            return not_ready_run_capability(code)
        except BaselineRunRuntimeError as exc:
            return not_ready_run_capability(exc.code)
        if not self._has_authorized_publication(
            group_id=group_id,
            caller_user_id=caller_user_id,
            embedding_fingerprint=identity.fingerprint,
        ):
            return not_ready_run_capability("index_publication_stale")
        if not callable(getattr(self.jobs, "cleanup_protected_payloads", None)):
            return not_ready_run_capability("worker_unavailable")
        if not callable(getattr(self.operator, "process", None)):
            return not_ready_run_capability("worker_unavailable")
        return ready_run_capability()


def process_baseline_run_job(job_id: str) -> BaselineManualRunOutcome:
    """Trusted operator callable whose sole input is an opaque job UUID."""

    from ...db import engine
    from ...server.settings import get_settings

    settings = get_settings()
    if not settings.baseline_runs_enabled:
        raise BaselineRunRuntimeError("capability_unavailable")
    return BaselineRunRuntime(engine, settings).operator.process(job_id)


__all__ = [
    "RUN_OPERATOR_CONTRACT_VERSION",
    "RUN_SCHEMA_MIGRATION_ID",
    "BaselineManualRunOperator",
    "BaselineManualRunOutcome",
    "BaselineRunRuntime",
    "BaselineRunRuntimeError",
    "process_baseline_run_job",
]
