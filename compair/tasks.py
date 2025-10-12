from __future__ import annotations

import logging
from typing import Mapping

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
        send_daily_usage_report,
    )
except (ImportError, ModuleNotFoundError):
    from sqlalchemy.orm import joinedload

    def _lazy_components():
        from . import Session as SessionMaker
        from .embeddings import Embedder
        from .feedback import Reviewer
        from .logger import log_event
        from .main import process_document
        from .models import Document, User

        return SessionMaker, Embedder, Reviewer, log_event, process_document, Document, User

    logger = logging.getLogger(__name__)

    def process_document_task(
        user_id: str,
        doc_id: str,
        doc_text: str,
        generate_feedback: bool = True,
    ) -> Mapping[str, list[str]]:
        SessionMaker, Embedder, Reviewer, log_event, process_document, Document, User = _lazy_components()
        with SessionMaker() as session:
            user = session.query(User).filter(User.user_id == user_id).first()
            if not user:
                logger.warning("User not found for document processing", extra={"user_id": user_id})
                return {"chunk_task_ids": []}

            doc = (
                session.query(Document)
                .options(joinedload(Document.groups))
                .filter(Document.document_id == doc_id)
                .first()
            )
            if not doc:
                logger.warning("Document not found for processing", extra={"document_id": doc_id})
                return {"chunk_task_ids": []}

            doc.content = doc_text
            session.add(doc)

            embedder = Embedder()
            reviewer = Reviewer()

            process_document(user, session, embedder, reviewer, doc, generate_feedback=generate_feedback)

            log_event(
                "core_document_processed",
                user_id=user_id,
                document_id=doc_id,
                feedback_requested=generate_feedback,
            )

            return {"chunk_task_ids": []}

    def process_text_task(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("process_text_task is only available in the Compair Cloud edition.")

    def check_trial_expirations():  # pragma: no cover
        raise RuntimeError("check_trial_expirations is only available in the Compair Cloud edition.")

    def expire_group_invitations():  # pragma: no cover
        raise RuntimeError("expire_group_invitations is only available in the Compair Cloud edition.")

    def send_trial_warnings():  # pragma: no cover
        raise RuntimeError("send_trial_warnings is only available in the Compair Cloud edition.")

    def send_feature_announcement_task():  # pragma: no cover
        raise RuntimeError("send_feature_announcement_task is only available in the Compair Cloud edition.")

    def send_deactivate_request_email(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("send_deactivate_request_email is only available in the Compair Cloud edition.")

    def send_help_request_email(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("send_help_request_email is only available in the Compair Cloud edition.")

    def send_daily_usage_report():  # pragma: no cover
        raise RuntimeError("send_daily_usage_report is only available in the Compair Cloud edition.")

    def process_file_with_ocr_task(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("OCR processing is only available in the Compair Cloud edition.")
