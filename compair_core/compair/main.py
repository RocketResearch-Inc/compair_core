from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Mapping

import Levenshtein
from sqlalchemy import select
from sqlalchemy.orm.attributes import get_history
from sqlalchemy.orm import Session as SASession

from .embeddings import create_embedding, Embedder
from .feedback import get_feedback, Reviewer
from .models import (
    Chunk,
    Document,
    Feedback,
    Group,
    Note,
    Reference,
    User,
    VECTOR_BACKEND,
    cosine_similarity,
)
from .utils import chunk_text, log_activity


def process_document(
    user: User,
    session: SASession,
    embedder: Embedder,
    reviewer: Reviewer,
    doc: Document,
    generate_feedback: bool = True,
) -> Mapping[str, int]:
    new = False

    prev_content = get_history(doc, "content").deleted
    prev_chunks: list[str] = []
    if prev_content:
        prev_chunks = chunk_text(prev_content[-1])

    feedback_limit_env = os.getenv("COMPAIR_CORE_FEEDBACK_LIMIT")
    try:
        feedback_limit = int(feedback_limit_env) if feedback_limit_env else None
    except ValueError:
        feedback_limit = None
    time_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

    recent_feedback_count = session.query(Feedback).filter(
        Feedback.source_chunk_id.in_(
            session.query(Chunk.chunk_id).filter(Chunk.document_id == doc.document_id)
        ),
        Feedback.timestamp >= time_cutoff,
    ).count()

    content = doc.content
    chunks = chunk_text(content)
    new_chunks = list(set(chunks) - set(prev_chunks))

    prioritized_chunk_indices: list[int] = []
    if generate_feedback:
        prioritized_chunk_indices = detect_significant_edits(prev_chunks=prev_chunks, new_chunks=new_chunks)

    if feedback_limit is None:
        indices_to_generate_feedback = prioritized_chunk_indices
    else:
        num_chunks_can_generate_feedback = max((feedback_limit - recent_feedback_count), 0)
        indices_to_generate_feedback = prioritized_chunk_indices[:num_chunks_can_generate_feedback]

    for i, chunk in enumerate(new_chunks):
        should_generate_feedback = i in indices_to_generate_feedback
        process_text(
            session=session,
            embedder=embedder,
            reviewer=reviewer,
            doc=doc,
            text=chunk,
            generate_feedback=should_generate_feedback,
        )

    remove_chunks = set(prev_chunks) - set(chunks)
    for chunk in remove_chunks:
        remove_text(session=session, text=chunk, document_id=doc.document_id)

    if doc.groups:
        log_activity(
            session=session,
            user_id=doc.author_id,
            group_id=doc.groups[0].group_id,
            action="update",
            object_id=doc.document_id,
            object_name=doc.title,
            object_type="document",
        )

    session.commit()
    return {"new": new}


def detect_significant_edits(
    prev_chunks: list[str],
    new_chunks: list[str],
    threshold: float = 0.5,
) -> list[int]:
    prev_set = set(prev_chunks)
    new_set = set(new_chunks)

    unchanged_chunks = prev_set & new_set
    new_chunks_to_check = new_set - unchanged_chunks

    process_chunks: list[str] = []
    for new_chunk in new_chunks_to_check:
        best_match = max((Levenshtein.ratio(new_chunk, prev_chunk) for prev_chunk in prev_chunks), default=0.0)
        if best_match < threshold:
            process_chunks.append(new_chunk)

    prioritized_chunks = prioritize_chunks(process_chunks)
    return [new_chunks.index(new_chunk) for new_chunk in prioritized_chunks if new_chunk in new_chunks]


def prioritize_chunks(chunks: list[str], limit: int | None = None) -> list[str]:
    limit = limit or len(chunks)
    indexed_chunks = [(chunk, idx) for idx, chunk in enumerate(chunks)]
    indexed_chunks.sort(key=lambda x: (-len(x[0]), x[1]))
    return [chunk for chunk, _ in indexed_chunks[:limit]]


def process_text(
    session: SASession,
    embedder: Embedder,
    reviewer: Reviewer,
    doc: Document,
    text: str,
    generate_feedback: bool = True,
    note: Note | None = None,
) -> None:
    logger = logging.getLogger(__name__)
    chunk_hash = hash(text)

    chunk_type = "note" if note else "document"
    note_id = note.note_id if note else None

    existing_chunks = session.query(Chunk).filter(
        Chunk.hash == str(chunk_hash),
        Chunk.document_id == doc.document_id,
        Chunk.chunk_type == chunk_type,
        Chunk.note_id == note_id,
    )

    user = session.query(User).filter(User.user_id == doc.author_id).first()
    if existing_chunks.first():
        for chunk in existing_chunks:
            if chunk.embedding is None:
                embedding = create_embedding(embedder, text, user=user)
                existing_chunks.update({"embedding": embedding})
        session.commit()
    else:
        chunk = Chunk(
            hash=chunk_hash,
            document_id=doc.document_id,
            note_id=note_id,
            chunk_type=chunk_type,
            content=text,
        )
        embedding = create_embedding(embedder, text, user=user)
        chunk.embedding = embedding
        session.add(chunk)
        session.commit()
        existing_chunk = chunk
    existing_chunk = session.query(Chunk).filter(
        Chunk.document_id == doc.document_id,
        Chunk.hash == str(chunk_hash),
        Chunk.chunk_type == chunk_type,
        Chunk.note_id == note_id,
    ).first()

    references: list[Chunk] = []
    if generate_feedback and existing_chunk:
        doc_group_ids = [g.group_id for g in doc.groups]
        target_embedding = existing_chunk.embedding

        if target_embedding is not None:
            base_query = (
                session.query(Chunk)
                .join(Chunk.document)
                .join(Document.groups)
                .filter(
                    Document.is_published.is_(True),
                    Document.document_id != doc.document_id,
                    Chunk.chunk_type == "document",
                    Group.group_id.in_(doc_group_ids),
                )
            )

            if VECTOR_BACKEND == "pgvector":
                references = (
                    base_query.order_by(
                        Chunk.embedding.cosine_distance(existing_chunk.embedding)
                    )
                    .limit(3)
                    .all()
                )
            else:
                candidates = base_query.all()
                scored: list[tuple[float, Chunk]] = []
                for candidate in candidates:
                    score = cosine_similarity(candidate.embedding, target_embedding)
                    if score is not None:
                        scored.append((score, candidate))
                scored.sort(key=lambda item: item[0], reverse=True)
                references = [chunk for _, chunk in scored[:3]]

        sql_references: list[Reference] = []
        for ref_chunk in references:
            sql_references.append(
                Reference(
                    source_chunk_id=existing_chunk.chunk_id,
                    reference_type="document",
                    reference_document_id=ref_chunk.document_id,
                    reference_note_id=None,
                )
            )

        if sql_references:
            session.add_all(sql_references)
            session.commit()

        feedback = get_feedback(reviewer, doc, text, references, user)
        if feedback != "NONE":
            sql_feedback = Feedback(
                source_chunk_id=existing_chunk.chunk_id,
                feedback=feedback,
                model=reviewer.model,
            )
            session.add(sql_feedback)
            session.commit()


def remove_text(session: SASession, text: str, document_id: str) -> None:
    chunks = session.query(Chunk).filter(
        Chunk.document_id == document_id,
        Chunk.content == text,
    )
    chunks.delete(synchronize_session=False)
    session.commit()


def get_all_chunks_for_document(session: SASession, doc: Document) -> list[Chunk]:
    doc_chunks = session.query(Chunk).filter(Chunk.document_id == doc.document_id).all()
    note_chunks: list[Chunk] = []
    notes = session.query(Note).filter(Note.document_id == doc.document_id).all()
    for note in notes:
        note_text_chunks = chunk_text(note.content)
        for text in note_text_chunks:
            chunk_hash = hash(text)
            existing = session.query(Chunk).filter(
                Chunk.hash == str(chunk_hash),
                Chunk.document_id == doc.document_id,
                Chunk.content == text,
            ).first()
            if not existing:
                embedding = create_embedding(Embedder(), text, user=doc.author_id)
                note_chunk = Chunk(
                    hash=str(chunk_hash),
                    document_id=doc.document_id,
                    content=text,
                    embedding=embedding,
                )
                session.add(note_chunk)
                session.commit()
                note_chunks.append(note_chunk)
            else:
                note_chunks.append(existing)
    return doc_chunks + note_chunks
