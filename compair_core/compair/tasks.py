from __future__ import annotations

import logging
from typing import Mapping, Optional

logger = logging.getLogger(__name__)

try:
    from compair_cloud.tasks import (  # type: ignore
        process_document_task,
        process_text_task,
        check_trial_expirations,
        expire_group_invitations,
        send_trial_warnings,
        send_feature_announcement_task,
        send_deactivate_request_email,
        send_help_request_email,
        send_waitlist_signup_email,
        send_daily_usage_report,
    )
except (ImportError, ModuleNotFoundError) as exc:
    logger.info(
        "compair_cloud.tasks unavailable; using core task implementations (%s: %s)",
        exc.__class__.__name__,
        exc,
    )
    from sqlalchemy.orm import joinedload

    def _lazy_components():
        from . import Session as SessionMaker
        from .embeddings import Embedder
        from .feedback import Reviewer
        from .logger import log_event
        from .main import process_document
        from .models import Document, User
        from .topic_tags import extract_topic_tags
        from .utils import sanitize_text_for_database

        return (
            SessionMaker,
            Embedder,
            Reviewer,
            log_event,
            process_document,
            Document,
            User,
            extract_topic_tags,
            sanitize_text_for_database,
        )

    logger = logging.getLogger(__name__)

    def process_document_task(
        user_id: str,
        doc_id: str,
        doc_text: str,
        generate_feedback: bool = True,
        chunk_mode: Optional[str] = None,
        reanalyze_existing: bool = False,
        snapshot_payload_key: Optional[str] = None,
        reference_doc_ids: Optional[list[str]] = None,
        focus_manifest: Optional[Mapping[str, object]] = None,
        retrieval_query: Optional[str] = None,
        retrieval_engine: str = "legacy",
        processing_run_key: Optional[str] = None,
        group_id: Optional[str] = None,
    ) -> Mapping[str, object]:
        from .retrieval import (
            BASELINE_RETRIEVAL_ENGINE,
            RetrievalQueryOrigin,
            baseline_document_processing_outcome,
            processing_run_trace_id,
            retrieval_query_provenance,
            validate_baseline_group_id,
            validate_processing_run_key,
            validate_retrieval_engine_name,
        )

        retrieval_engine = validate_retrieval_engine_name(retrieval_engine)
        validated_processing_run_key: str | None = None
        validated_group_id: str | None = None
        if retrieval_engine == BASELINE_RETRIEVAL_ENGINE:
            validated_processing_run_key = validate_processing_run_key(
                processing_run_key
            )
            try:
                validated_group_id = validate_baseline_group_id(group_id)
            except ValueError:
                return {
                    "chunk_task_ids": [],
                    "baseline_processing": baseline_document_processing_outcome(
                        [],
                        group_id=None,
                        parent_run_trace_id=processing_run_trace_id(
                            validated_processing_run_key,
                            None,
                        ),
                        error_code=(
                            "explicit_group_id_absent"
                            if group_id is None
                            else "explicit_group_id_invalid"
                        ),
                        query_provenance=retrieval_query_provenance(
                            retrieval_query,
                            (
                                RetrievalQueryOrigin.EXPLICIT
                                if retrieval_query is not None
                                else RetrievalQueryOrigin.ABSENT
                            ),
                        ),
                    ),
                }

        def baseline_failure(error_code: str) -> Mapping[str, object]:
            assert validated_processing_run_key is not None
            assert validated_group_id is not None
            return {
                "chunk_task_ids": [],
                "baseline_processing": baseline_document_processing_outcome(
                    [],
                    group_id=validated_group_id,
                    parent_run_trace_id=processing_run_trace_id(
                        validated_processing_run_key,
                        validated_group_id,
                    ),
                    error_code=error_code,
                    query_provenance=retrieval_query_provenance(
                        retrieval_query,
                        (
                            RetrievalQueryOrigin.EXPLICIT
                            if retrieval_query is not None
                            else RetrievalQueryOrigin.ABSENT
                        ),
                    ),
                ),
            }
        (
            SessionMaker,
            Embedder,
            Reviewer,
            log_event,
            process_document,
            Document,
            User,
            extract_topic_tags,
            sanitize_text_for_database,
        ) = _lazy_components()
        with SessionMaker() as session:
            user = session.query(User).filter(User.user_id == user_id).first()
            if not user:
                if retrieval_engine == BASELINE_RETRIEVAL_ENGINE:
                    return baseline_failure("caller_absent")
                logger.warning(
                    "User not found for document processing", extra={"user_id": user_id}
                )
                return {"chunk_task_ids": []}

            doc = (
                session.query(Document)
                .options(joinedload(Document.groups))
                .filter(Document.document_id == doc_id)
                .first()
            )
            if not doc:
                if retrieval_engine == BASELINE_RETRIEVAL_ENGINE:
                    return baseline_failure("source_document_absent")
                logger.warning(
                    "Document not found for processing", extra={"document_id": doc_id}
                )
                return {"chunk_task_ids": []}

            doc_text = sanitize_text_for_database(doc_text)
            doc.content = doc_text
            doc.topic_tags = extract_topic_tags(doc_text)
            session.add(doc)

            if retrieval_engine == BASELINE_RETRIEVAL_ENGINE:
                embedder = None
                reviewer = None
            else:
                embedder = Embedder()
                reviewer = Reviewer()

            baseline_scope_kwargs = (
                {"group_id": validated_group_id}
                if retrieval_engine == BASELINE_RETRIEVAL_ENGINE
                else {}
            )
            processing_result = process_document(
                user,
                session,
                embedder,
                reviewer,
                doc,
                generate_feedback=generate_feedback,
                chunk_mode=chunk_mode,
                reanalyze_existing=reanalyze_existing,
                reference_doc_ids=reference_doc_ids,
                focus_manifest=focus_manifest,
                retrieval_query=retrieval_query,
                retrieval_engine=retrieval_engine,
                processing_run_key=processing_run_key,
                **baseline_scope_kwargs,
            )

            processed_event = {
                "user_id": user_id,
                "document_id": doc_id,
                "feedback_requested": generate_feedback,
                "retrieval_engine": retrieval_engine,
            }
            if retrieval_engine == BASELINE_RETRIEVAL_ENGINE:
                processed_event["group_id"] = group_id
            log_event(
                "core_document_processed",
                **processed_event,
            )

            if retrieval_engine == "baseline_v1":
                return {
                    "chunk_task_ids": [],
                    "baseline_processing": processing_result.get(
                        "baseline_processing",
                        {
                            "schema_version": "baseline-document-processing.v2",
                            "engine": "baseline_v1",
                            "generation_bypassed": True,
                            "group_id": group_id,
                            "status": "error",
                            "error_code": "baseline_processing_outcome_absent",
                            "outcomes": [],
                        },
                    ),
                }
            return {"chunk_task_ids": []}

    def process_text_task(*args, **kwargs):  # pragma: no cover
        raise RuntimeError(
            "process_text_task is only available in the Compair Cloud edition."
        )

    def check_trial_expirations():  # pragma: no cover
        raise RuntimeError(
            "check_trial_expirations is only available in the Compair Cloud edition."
        )

    def expire_group_invitations():  # pragma: no cover
        raise RuntimeError(
            "expire_group_invitations is only available in the Compair Cloud edition."
        )

    def send_trial_warnings():  # pragma: no cover
        raise RuntimeError(
            "send_trial_warnings is only available in the Compair Cloud edition."
        )

    def send_feature_announcement_task():  # pragma: no cover
        raise RuntimeError(
            "send_feature_announcement_task is only available in the Compair Cloud edition."
        )

    def send_deactivate_request_email(*args, **kwargs):  # pragma: no cover
        raise RuntimeError(
            "send_deactivate_request_email is only available in the Compair Cloud edition."
        )

    def send_help_request_email(*args, **kwargs):  # pragma: no cover
        raise RuntimeError(
            "send_help_request_email is only available in the Compair Cloud edition."
        )

    def send_waitlist_signup_email(*args, **kwargs):  # pragma: no cover
        raise RuntimeError(
            "send_waitlist_signup_email is only available in the Compair Cloud edition."
        )

    def send_daily_usage_report():  # pragma: no cover
        raise RuntimeError(
            "send_daily_usage_report is only available in the Compair Cloud edition."
        )

    def process_file_with_ocr_task(*args, **kwargs):  # pragma: no cover
        raise RuntimeError(
            "OCR processing is only available in the Compair Cloud edition."
        )
