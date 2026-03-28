from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Mapping, Optional

import Levenshtein
from sqlalchemy import select
from sqlalchemy.orm.attributes import get_history
from sqlalchemy.orm import Session as SASession

from .embeddings import create_embedding, create_embeddings, Embedder
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
from .topic_tags import extract_topic_tags
from .utils import (
    chunk_text_with_mode,
    count_tokens,
    log_activity,
    stable_chunk_hash,
)


_CODE_REPO_DOC_TYPE = "code-repo"
_HIGH_SIGNAL_PATH_HINTS = (
    "/api/",
    "/auth/",
    "/route",
    "/router",
    "/server/",
    "/settings",
    "/config",
    "/billing",
    "/payment",
    "/notification",
    "/sync",
    "/main.",
    "/app.",
)
_REFERENCE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_REFERENCE_SUBTOKEN_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+")
_REFERENCE_TOKEN_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "return",
    "returns",
    "class",
    "function",
    "const",
    "true",
    "false",
    "none",
    "null",
    "json",
    "text",
    "type",
    "value",
    "data",
    "item",
    "items",
    "file",
    "path",
    "chunk",
    "document",
    "note",
}


def is_code_review_document(doc: Document, chunk_mode: Optional[str]) -> bool:
    doc_type = (getattr(doc, "doc_type", "") or "").strip().lower()
    mode = (chunk_mode or "").strip().lower()
    return doc_type == _CODE_REPO_DOC_TYPE or mode in {"client", "preserve", "prechunked"}


def _extract_snapshot_file_path(chunk: str) -> str:
    match = re.search(r"^### File:\s+([^\n(]+)", chunk, re.MULTILINE)
    if not match:
        return ""
    return match.group(1).strip()


def _chunk_priority_key(chunk: str, idx: int, code_focus: bool) -> tuple[int, int, int, int, int]:
    if not code_focus:
        return (0, 0, 0, -len(chunk), idx)

    stripped = chunk.lstrip()
    path = _extract_snapshot_file_path(chunk).lower()
    has_file_header = bool(path)
    has_code_fence = "```" in chunk

    if stripped.startswith("# Compair baseline snapshot") and not has_file_header:
        category = 3
    elif stripped.startswith("## Snapshot limits") and not has_file_header:
        category = 2
    elif has_file_header:
        category = 0
    else:
        category = 1

    path_rank = 2
    if path:
        if any(hint in path for hint in _HIGH_SIGNAL_PATH_HINTS):
            path_rank = 0
        else:
            path_rank = 1

    fence_rank = 0 if has_code_fence else 1
    return (category, path_rank, fence_rank, -count_tokens(chunk), idx)


def _is_snapshot_metadata_chunk(chunk: str) -> bool:
    stripped = (chunk or "").lstrip()
    return stripped.startswith("# Compair baseline snapshot") or stripped.startswith("## Snapshot limits")


def _is_code_review_chunk(doc: Document, text: str) -> bool:
    doc_type = (getattr(doc, "doc_type", "") or "").strip().lower()
    if doc_type == _CODE_REPO_DOC_TYPE:
        return True
    return "### File:" in (text or "") and "```" in (text or "")


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _reference_selection_config(code_focus: bool) -> tuple[int, int, int]:
    if code_focus:
        return (
            _int_env("COMPAIR_CODE_REPO_REFERENCE_CANDIDATES", 10),
            _int_env("COMPAIR_CODE_REPO_REFERENCE_LIMIT", 4),
            _int_env("COMPAIR_REFERENCE_MAX_PER_SOURCE", 2),
        )
    return (
        _int_env("COMPAIR_REFERENCE_CANDIDATES", 6),
        _int_env("COMPAIR_REFERENCE_LIMIT", 3),
        _int_env("COMPAIR_REFERENCE_MAX_PER_SOURCE", 2),
    )


def _identifier_tokens(text: str, limit: int = 64) -> set[str]:
    tokens: set[str] = set()
    for raw in _REFERENCE_TOKEN_RE.findall(text or ""):
        parts = [raw]
        parts.extend(
            subtoken
            for subtoken in _REFERENCE_SUBTOKEN_RE.findall(raw.replace("_", " "))
            if subtoken
        )
        for part in parts:
            token = part.lower()
            if len(token) < 3 or token in _REFERENCE_TOKEN_STOPWORDS:
                continue
            tokens.add(token)
            if len(tokens) >= limit:
                return tokens
    return tokens


def _path_token_set(path: str) -> set[str]:
    return {
        token
        for token in re.split(r"[/._:-]+", (path or "").lower())
        if len(token) >= 2 and token not in _REFERENCE_TOKEN_STOPWORDS
    }


def _path_overlap_score(target_path: str, candidate_path: str) -> float:
    if not target_path or not candidate_path:
        return 0.0
    target_norm = target_path.strip().lower()
    candidate_norm = candidate_path.strip().lower()
    if not target_norm or not candidate_norm:
        return 0.0
    score = 0.0
    if target_norm == candidate_norm:
        score += 3.0
    elif os.path.basename(target_norm) == os.path.basename(candidate_norm):
        score += 1.5
    shared = _path_token_set(target_norm) & _path_token_set(candidate_norm)
    if shared:
        score += min(2.0, 0.5 * len(shared))
    return score


def _token_overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / float(max(1, min(len(left), len(right))))


def _is_doc_like_path(path: str) -> bool:
    normalized = (path or "").strip().lower().replace("\\", "/")
    if not normalized:
        return False
    base = os.path.basename(normalized)
    doc_basenames = {
        "readme",
        "readme.md",
        "readme.rst",
        "readme.txt",
        "changelog",
        "changelog.md",
        "changelog.rst",
        "changelog.txt",
        "security",
        "security.md",
        "security.rst",
        "security.txt",
        "contributing",
        "contributing.md",
        "contributing.rst",
        "contributing.txt",
        "architecture",
        "architecture.md",
        "architecture.rst",
        "architecture.txt",
    }
    return base in doc_basenames or "/docs/" in normalized or "/doc/" in normalized


def _chunk_relevance_score(
    chunk: str,
    idx: int,
    code_focus: bool,
    novelty_score: float,
) -> float:
    token_score = min(float(count_tokens(chunk)) / 240.0, 2.0)
    relevance = max(0.0, novelty_score) * 2.4
    if not code_focus:
        return relevance + token_score - (0.01 * float(min(idx, 50)))

    category, path_rank, _, _, _ = _chunk_priority_key(chunk, idx, code_focus)
    path = _extract_snapshot_file_path(chunk)
    relevance += token_score
    relevance += max(0.0, 1.2 - (0.4 * float(category)))
    relevance += max(0.0, 0.7 - (0.35 * float(path_rank)))
    if "```" in chunk:
        relevance += 0.4
    if "diff --git" in chunk or "@@" in chunk or "+++ b/" in chunk:
        relevance += 0.45
    if _is_doc_like_path(path):
        relevance -= 0.8
    return relevance - (0.015 * float(min(idx, 50)))


def _chunk_redundancy_score(left: str, right: str, code_focus: bool) -> float:
    lexical = _token_overlap_ratio(_identifier_tokens(left), _identifier_tokens(right))
    if not code_focus:
        return lexical
    path_score = min(
        _path_overlap_score(_extract_snapshot_file_path(left), _extract_snapshot_file_path(right)) / 3.0,
        1.0,
    )
    return lexical + (0.35 * path_score)


def _reference_source_key(chunk: Chunk) -> str:
    if getattr(chunk, "document_id", None):
        return f"document:{chunk.document_id}"
    if getattr(chunk, "note_id", None):
        return f"note:{chunk.note_id}"
    return f"chunk:{getattr(chunk, 'chunk_id', '')}"


def _reference_chunk_key(chunk: Chunk) -> str:
    chunk_id = getattr(chunk, "chunk_id", None)
    if chunk_id:
        return str(chunk_id)
    source_key = _reference_source_key(chunk)
    content = getattr(chunk, "content", "") or ""
    return f"{source_key}:{stable_chunk_hash(content)}"


def _merge_reference_candidates(
    primary: list[Chunk],
    secondary: list[Chunk],
    limit: int,
) -> list[Chunk]:
    merged: list[Chunk] = []
    seen: set[str] = set()
    for candidate in [*primary, *secondary]:
        key = _reference_chunk_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        merged.append(candidate)
        if len(merged) >= limit:
            break
    return merged


def _lexical_reference_candidates(
    target_text: str,
    candidates: list[Chunk],
    *,
    limit: int,
    code_focus: bool,
) -> list[Chunk]:
    if not candidates or limit <= 0:
        return []

    target_tokens = _identifier_tokens(target_text, limit=96)
    target_path = _extract_snapshot_file_path(target_text)
    scored: list[tuple[float, int, Chunk]] = []
    for idx, candidate in enumerate(candidates):
        if getattr(candidate, "chunk_type", "") != "document":
            continue
        content = getattr(candidate, "content", "") or ""
        if not content or _is_snapshot_metadata_chunk(content):
            continue

        candidate_tokens = _identifier_tokens(content, limit=96)
        lexical_score = _token_overlap_ratio(target_tokens, candidate_tokens)
        path_score = _path_overlap_score(target_path, _extract_snapshot_file_path(content))
        code_bonus = 0.35 if "```" in content else 0.0
        diff_bonus = 0.25 if "diff --git" in content or "@@" in content or "+++ b/" in content else 0.0
        doc_penalty = 0.75 if code_focus and _is_doc_like_path(_extract_snapshot_file_path(content)) else 0.0
        score = (lexical_score * 5.0) + (path_score * 1.5) + code_bonus + diff_bonus - doc_penalty
        if score <= 0.0:
            continue
        scored.append((score, idx, candidate))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [candidate for _, _, candidate in scored[:limit]]


def _rerank_reference_chunks(target_text: str, candidates: list[Chunk], code_focus: bool) -> list[Chunk]:
    candidate_limit, final_limit, max_per_source = _reference_selection_config(code_focus)
    if not candidates:
        return []

    trim_limit = max(candidate_limit, min(len(candidates), candidate_limit * 2))
    trimmed = candidates[:trim_limit]
    if not code_focus:
        return trimmed[:final_limit]

    target_path = _extract_snapshot_file_path(target_text)
    target_tokens = _identifier_tokens(target_text)
    used_indices: set[int] = set()
    source_counts: dict[str, int] = {}
    selected: list[Chunk] = []
    selected_tokens: list[set[str]] = []

    while len(selected) < final_limit:
        best_index = -1
        best_score = float("-inf")
        best_tokens: set[str] | None = None
        for idx, candidate in enumerate(trimmed):
            if idx in used_indices:
                continue
            if getattr(candidate, "chunk_type", "") != "document":
                continue
            content = getattr(candidate, "content", "") or ""
            if _is_snapshot_metadata_chunk(content):
                continue
            source_key = _reference_source_key(candidate)
            if source_counts.get(source_key, 0) >= max_per_source:
                continue

            candidate_tokens = _identifier_tokens(content)
            base_score = float(len(trimmed) - idx) / float(max(1, len(trimmed)))
            lexical_score = _token_overlap_ratio(target_tokens, candidate_tokens)
            path_score = _path_overlap_score(target_path, _extract_snapshot_file_path(content))
            code_bonus = 0.4 if "```" in content else 0.0
            diversity_penalty = 0.0
            if selected_tokens:
                diversity_penalty = max(_token_overlap_ratio(candidate_tokens, prev) for prev in selected_tokens)
            source_penalty = 0.75 * float(source_counts.get(source_key, 0))
            score = (base_score * 3.0) + (lexical_score * 4.0) + path_score + code_bonus - (diversity_penalty * 2.5) - source_penalty
            if score > best_score:
                best_score = score
                best_index = idx
                best_tokens = candidate_tokens

        if best_index < 0:
            break

        chosen = trimmed[best_index]
        used_indices.add(best_index)
        source_key = _reference_source_key(chosen)
        source_counts[source_key] = source_counts.get(source_key, 0) + 1
        selected.append(chosen)
        selected_tokens.append(best_tokens or set())

    if selected:
        return selected
    return trimmed[:final_limit]


def process_document(
    user: User,
    session: SASession,
    embedder: Embedder,
    reviewer: Reviewer,
    doc: Document,
    generate_feedback: bool = True,
    chunk_mode: Optional[str] = None,
    reanalyze_existing: bool = False,
) -> Mapping[str, int]:
    new = False

    prev_content = get_history(doc, "content").deleted
    prev_chunks: list[str] = []
    if prev_content:
        prev_chunks = chunk_text_with_mode(prev_content[-1], chunk_mode=chunk_mode)

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
    doc.topic_tags = extract_topic_tags(content)
    chunks = chunk_text_with_mode(content, chunk_mode=chunk_mode)
    prev_set = set(prev_chunks)
    new_chunks = [c for c in chunks if c not in prev_set]

    prioritized_chunk_indices: list[int] = []
    code_focus = False
    if generate_feedback:
        code_focus = is_code_review_document(doc, chunk_mode)
        prioritized_chunk_indices = detect_significant_edits(
            prev_chunks=prev_chunks,
            new_chunks=new_chunks,
            code_focus=code_focus,
        )
        if code_focus:
            prioritized_chunk_indices = [
                i for i in prioritized_chunk_indices if not _is_snapshot_metadata_chunk(new_chunks[i])
            ]

    feedback_min_tokens = int(os.getenv("COMPAIR_FEEDBACK_MIN_TOKENS", "120"))
    feedback_fallback_min = int(os.getenv("COMPAIR_FEEDBACK_MIN_TOKENS_FALLBACK", "20"))
    token_lens = [count_tokens(c) for c in new_chunks]
    eligible_indices = [i for i, t in enumerate(token_lens) if t >= feedback_min_tokens]
    if generate_feedback and feedback_min_tokens > 0:
        prioritized_chunk_indices = [i for i in prioritized_chunk_indices if i in eligible_indices]
        if not prioritized_chunk_indices and eligible_indices:
            fallback_idx = max(eligible_indices, key=lambda idx: token_lens[idx])
            prioritized_chunk_indices = [fallback_idx]
        elif not eligible_indices and new_chunks:
            fallback_idx = max(range(len(token_lens)), key=lambda idx: token_lens[idx])
            if token_lens[fallback_idx] >= feedback_fallback_min:
                prioritized_chunk_indices = [fallback_idx]

    if feedback_limit is None:
        indices_to_generate_feedback = prioritized_chunk_indices
    else:
        num_chunks_can_generate_feedback = max((feedback_limit - recent_feedback_count), 0)
        indices_to_generate_feedback = prioritized_chunk_indices[:num_chunks_can_generate_feedback]

    existing_indices_to_generate_feedback: list[int] = []
    if generate_feedback and reanalyze_existing and not new_chunks:
        existing_indices = _existing_feedback_candidate_indices(
            session=session,
            doc=doc,
            chunks=chunks,
            code_focus=code_focus,
            feedback_min_tokens=feedback_min_tokens,
            feedback_fallback_min=feedback_fallback_min,
        )
        if feedback_limit is None:
            existing_indices_to_generate_feedback = existing_indices
        else:
            remaining_slots = max((feedback_limit - recent_feedback_count - len(indices_to_generate_feedback)), 0)
            existing_indices_to_generate_feedback = existing_indices[:remaining_slots]

    new_chunk_embeddings = create_embeddings(embedder, new_chunks, user=user) if new_chunks else []
    for i, chunk in enumerate(new_chunks):
        should_generate_feedback = i in indices_to_generate_feedback
        process_text(
            session=session,
            embedder=embedder,
            reviewer=reviewer,
            doc=doc,
            text=chunk,
            generate_feedback=should_generate_feedback,
            precomputed_embedding=new_chunk_embeddings[i],
        )

    for idx in existing_indices_to_generate_feedback:
        process_text(
            session=session,
            embedder=embedder,
            reviewer=reviewer,
            doc=doc,
            text=chunks[idx],
            generate_feedback=True,
        )

    removed = [c for c in prev_chunks if c not in set(chunks)]
    for chunk in removed:
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
    code_focus: bool = False,
) -> list[int]:
    if not new_chunks:
        return []
    if not prev_chunks:
        novelty_scores = {i: 1.0 for i in range(len(new_chunks))}
        return prioritize_chunks(
            list(range(len(new_chunks))),
            new_chunks,
            code_focus=code_focus,
            novelty_scores=novelty_scores,
        )
    candidate_indices: list[int] = []
    novelty_scores: dict[int, float] = {}
    for idx, new_chunk in enumerate(new_chunks):
        if new_chunk in prev_chunks:
            continue
        best_match = max((Levenshtein.ratio(new_chunk, prev_chunk) for prev_chunk in prev_chunks), default=0.0)
        novelty_scores[idx] = max(0.0, 1.0 - best_match)
        if best_match < threshold:
            candidate_indices.append(idx)
    return prioritize_chunks(
        candidate_indices,
        new_chunks,
        code_focus=code_focus,
        novelty_scores=novelty_scores,
    )


def prioritize_chunks(
    indices: list[int],
    chunks: list[str],
    limit: int = 10,
    code_focus: bool = False,
    novelty_scores: Mapping[int, float] | None = None,
) -> list[int]:
    if not indices:
        return []
    if code_focus:
        indices = [i for i in indices if not _is_snapshot_metadata_chunk(chunks[i])]
        if not indices:
            return []
        try:
            limit = int(os.getenv("COMPAIR_CODE_REPO_FEEDBACK_CANDIDATES", str(limit)))
        except ValueError:
            pass
    else:
        indices.sort(key=lambda i: _chunk_priority_key(chunks[i], i, code_focus))
        return indices[:limit]

    if novelty_scores is None:
        novelty_scores = {}

    remaining = list(indices)
    selected: list[int] = []
    while remaining and len(selected) < limit:
        best_idx = -1
        best_score = float("-inf")
        for idx in remaining:
            relevance = _chunk_relevance_score(chunks[idx], idx, code_focus, novelty_scores.get(idx, 1.0))
            redundancy = 0.0
            if selected:
                redundancy = max(
                    _chunk_redundancy_score(chunks[idx], chunks[chosen_idx], code_focus)
                    for chosen_idx in selected
                )
            score = relevance - (2.6 * redundancy)
            if best_idx < 0 or score > best_score:
                best_idx = idx
                best_score = score
        if best_idx < 0:
            break
        if selected and best_score < 1.8:
            break
        selected.append(best_idx)
        remaining = [idx for idx in remaining if idx != best_idx]

    if selected:
        return selected
    indices.sort(key=lambda i: _chunk_priority_key(chunks[i], i, code_focus))
    return indices[:limit]


def _existing_feedback_candidate_indices(
    session: SASession,
    doc: Document,
    chunks: list[str],
    code_focus: bool,
    feedback_min_tokens: int,
    feedback_fallback_min: int,
) -> list[int]:
    if not chunks:
        return []

    chunk_set = set(chunks)
    content_to_chunk_id: dict[str, str] = {}
    existing_rows = session.query(Chunk.chunk_id, Chunk.content).filter(
        Chunk.document_id == doc.document_id,
        Chunk.chunk_type == "document",
    ).all()
    for chunk_id, content in existing_rows:
        if content in chunk_set and content not in content_to_chunk_id:
            content_to_chunk_id[content] = chunk_id
    if not content_to_chunk_id:
        return []

    feedback_chunk_ids = {
        source_chunk_id
        for (source_chunk_id,) in session.query(Feedback.source_chunk_id).filter(
            Feedback.source_chunk_id.in_(list(content_to_chunk_id.values()))
        ).distinct().all()
    }

    candidate_indices = [
        idx
        for idx, chunk in enumerate(chunks)
        if content_to_chunk_id.get(chunk) and content_to_chunk_id[chunk] not in feedback_chunk_ids
    ]
    if code_focus:
        candidate_indices = [idx for idx in candidate_indices if not _is_snapshot_metadata_chunk(chunks[idx])]
    if not candidate_indices:
        return []

    token_lens = {idx: count_tokens(chunks[idx]) for idx in candidate_indices}
    if feedback_min_tokens > 0:
        eligible = [idx for idx in candidate_indices if token_lens[idx] >= feedback_min_tokens]
        if eligible:
            candidate_indices = eligible
        else:
            fallback_idx = max(candidate_indices, key=lambda idx: token_lens[idx])
            if token_lens[fallback_idx] < feedback_fallback_min:
                return []
            candidate_indices = [fallback_idx]

    return prioritize_chunks(
        candidate_indices,
        chunks,
        code_focus=code_focus,
        novelty_scores={idx: 1.0 for idx in candidate_indices},
    )


def process_text(
    session: SASession,
    embedder: Embedder,
    reviewer: Reviewer,
    doc: Document,
    text: str,
    generate_feedback: bool = True,
    note: Note | None = None,
    precomputed_embedding: list[float] | None = None,
) -> None:
    logger = logging.getLogger(__name__)
    chunk_hash = stable_chunk_hash(text)

    chunk_type = "note" if note else "document"
    note_id = note.note_id if note else None

    existing_chunks = session.query(Chunk).filter(
        Chunk.document_id == doc.document_id,
        Chunk.chunk_type == chunk_type,
        Chunk.note_id == note_id,
        Chunk.content == text,
    )

    user = session.query(User).filter(User.user_id == doc.author_id).first()
    existing_rows = existing_chunks.all()
    existing_chunk = existing_rows[0] if existing_rows else None

    embedding = precomputed_embedding
    if existing_rows:
        for chunk in existing_rows:
            if chunk.hash != chunk_hash:
                chunk.hash = chunk_hash
            if embedding is None and chunk.embedding is not None:
                embedding = chunk.embedding
        if embedding is None:
            embedding = create_embedding(embedder, text, user=user)
        for chunk in existing_rows:
            if chunk.embedding is None:
                chunk.embedding = embedding
        session.commit()
    else:
        chunk = Chunk(
            hash=chunk_hash,
            document_id=doc.document_id,
            note_id=note_id,
            chunk_type=chunk_type,
            content=text,
        )
        if embedding is None:
            embedding = create_embedding(embedder, text, user=user)
        chunk.embedding = embedding
        session.add(chunk)
        session.commit()
        existing_chunk = chunk
    if existing_chunk is None:
        existing_chunk = session.query(Chunk).filter(
            Chunk.document_id == doc.document_id,
            Chunk.chunk_type == chunk_type,
            Chunk.note_id == note_id,
            Chunk.content == text,
        ).first()

    references: list[Chunk] = []
    if generate_feedback and existing_chunk:
        doc_group_ids = [g.group_id for g in doc.groups]
        target_embedding = existing_chunk.embedding
        code_focus = _is_code_review_chunk(doc, text)
        candidate_limit, _, _ = _reference_selection_config(code_focus)
        merge_limit = max(candidate_limit, candidate_limit * 2)

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

            all_candidates: list[Chunk] | None = None
            if VECTOR_BACKEND == "pgvector":
                candidates = (
                    base_query.order_by(
                        Chunk.embedding.cosine_distance(existing_chunk.embedding)
                    )
                    .limit(candidate_limit)
                    .all()
                )
                if code_focus:
                    all_candidates = base_query.all()
            else:
                all_candidates = base_query.all()
                scored: list[tuple[float, Chunk]] = []
                for candidate in all_candidates:
                    score = cosine_similarity(candidate.embedding, target_embedding)
                    if score is not None:
                        scored.append((score, candidate))
                scored.sort(key=lambda item: item[0], reverse=True)
                candidates = [chunk for _, chunk in scored[:candidate_limit]]
            if code_focus:
                lexical_candidates = _lexical_reference_candidates(
                    text,
                    all_candidates or [],
                    limit=candidate_limit,
                    code_focus=True,
                )
                candidates = _merge_reference_candidates(candidates, lexical_candidates, merge_limit)
            references = _rerank_reference_chunks(text, candidates, code_focus=code_focus)
            if not references:
                logger.info(
                    "[feedback] no references for doc=%s group_ids=%s code_focus=%s",
                    doc.document_id,
                    doc_group_ids,
                    code_focus,
                )

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
        if not references:
            return

        feedback = get_feedback(reviewer, doc, text, references, user)
        if feedback != "NONE":
            sql_feedback = Feedback(
                source_chunk_id=existing_chunk.chunk_id,
                feedback=feedback,
                model=reviewer.model,
            )
            session.add(sql_feedback)
            session.commit()
            try:
                from .notifications.service import (
                    NotificationCandidate,
                    PeerCandidate,
                    is_scoring_enabled,
                    score_and_route_candidate,
                )

                if is_scoring_enabled() and doc.groups:
                    peer_candidates: list[PeerCandidate] = []
                    for ref_chunk in references:
                        ref_doc = ref_chunk.document
                        ref_user = ref_doc.user if ref_doc else None
                        peer_candidates.append(
                            PeerCandidate(
                                doc_id=ref_doc.document_id if ref_doc else "",
                                doc_title=ref_doc.title if ref_doc else "",
                                chunk_id=ref_chunk.chunk_id,
                                chunk_text=ref_chunk.content,
                                doc_type=ref_doc.doc_type if ref_doc else "",
                                author_role=ref_user.role if ref_user and ref_user.role else "",
                                author_team="",
                                last_modified_utc=(
                                    ref_doc.datetime_modified.isoformat()
                                    if ref_doc and ref_doc.datetime_modified
                                    else None
                                ),
                                similarity=None,
                            )
                        )
                    if peer_candidates:
                        candidate = NotificationCandidate(
                            user_id=user.user_id if user else doc.user_id,
                            group_id=doc.groups[0].group_id,
                            target_doc_id=doc.document_id,
                            target_chunk_id=existing_chunk.chunk_id,
                            target_text=text,
                            target_doc_title=doc.title or "",
                            target_doc_type=doc.doc_type or "",
                            target_last_modified_utc=(
                                doc.datetime_modified.isoformat() if doc.datetime_modified else None
                            ),
                            user_role=user.role if user and user.role else "",
                            user_team="",
                            user_is_doc_author=True,
                            user_is_group_admin=False,
                            peer_candidates=tuple(peer_candidates),
                            generated_feedback={"summary": feedback},
                            run_id=f"feedback_{sql_feedback.feedback_id}",
                            now_utc=datetime.now(timezone.utc),
                        )
                        score_and_route_candidate(
                            session,
                            candidate,
                            commit=True,
                            delivery_channel="inbox_only",
                        )
            except Exception as exc:
                logger.warning("Notification scoring failed: %s", exc)


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
        note_text_chunks = chunk_text_with_mode(note.content)
        for text in note_text_chunks:
            chunk_hash = stable_chunk_hash(text)
            existing = session.query(Chunk).filter(
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
                if existing.hash != chunk_hash:
                    existing.hash = chunk_hash
                    session.commit()
                note_chunks.append(existing)
    return doc_chunks + note_chunks
