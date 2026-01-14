from __future__ import annotations

import hashlib
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from sqlalchemy.orm import Session

from .models import Activity


CLIENT_CHUNK_DELIMITER = "<<<COMPAIR_CHUNK>>>"


def chunk_text(text: str) -> list[str]:
    chunks = text.split("\n\n")
    chunks = [c.strip() for c in chunks]
    return [c for c in chunks if c]


def split_client_chunks(text: str, delimiter: str = CLIENT_CHUNK_DELIMITER) -> list[str]:
    if not text or delimiter not in text:
        return []
    parts = [p.strip() for p in text.split(delimiter)]
    return [p for p in parts if p]


def stable_chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()


def _get_token_counter() -> Callable[[str], int]:
    encoding_name = os.getenv("COMPAIR_CHUNK_TOKEN_ENCODING", "cl100k_base")
    try:
        import tiktoken  # type: ignore

        encoding = tiktoken.get_encoding(encoding_name)

        def _count(text: str) -> int:
            return len(encoding.encode(text or ""))

        return _count
    except Exception:
        chars_per_token = int(os.getenv("COMPAIR_CHUNK_CHARS_PER_TOKEN", "4"))

        def _count(text: str) -> int:
            raw = len(text or "")
            return max(1, raw // max(1, chars_per_token))

        return _count


_TOKEN_COUNTER = _get_token_counter()


def count_tokens(text: str) -> int:
    return _TOKEN_COUNTER(text)


def _chunking_defaults() -> tuple[int, int, int, int]:
    target = int(os.getenv("COMPAIR_CHUNK_TARGET_TOKENS", "512"))
    overlap = int(os.getenv("COMPAIR_CHUNK_OVERLAP_TOKENS", "80"))
    min_tokens = int(os.getenv("COMPAIR_CHUNK_MIN_TOKENS", "140"))
    max_tokens = int(os.getenv("COMPAIR_CHUNK_MAX_TOKENS", "768"))
    return target, overlap, min_tokens, max_tokens


def _is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if stripped.isupper() and len(stripped) <= 80:
        return True
    if stripped.endswith(":") and len(stripped) <= 80:
        word_count = len(stripped.split())
        return word_count <= 8
    return False


def _split_blocks(text: str) -> list[dict[str, str | None]]:
    blocks: list[dict[str, str | None]] = []
    current_heading: str | None = None
    paragraphs = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    for para in paragraphs:
        lines = [ln for ln in para.splitlines() if ln.strip()]
        if len(lines) == 1 and _is_heading_line(lines[0]):
            heading = lines[0].strip().lstrip("#").strip().rstrip(":").strip()
            if heading:
                current_heading = heading
            continue
        blocks.append({"text": para.strip(), "heading": current_heading})
    return blocks


def _split_by_tokens(text: str, target: int, overlap: int, token_counter: Callable[[str], int]) -> list[str]:
    if not text:
        return []
    try:
        import tiktoken  # type: ignore

        encoding_name = os.getenv("COMPAIR_CHUNK_TOKEN_ENCODING", "cl100k_base")
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        step = max(1, target - overlap)
        out: list[str] = []
        for start in range(0, len(tokens), step):
            slice_tokens = tokens[start:start + target]
            if not slice_tokens:
                continue
            out.append(encoding.decode(slice_tokens).strip())
        return [c for c in out if c]
    except Exception:
        chars_per_token = int(os.getenv("COMPAIR_CHUNK_CHARS_PER_TOKEN", "4"))
        char_target = max(1, target * chars_per_token)
        char_overlap = max(0, overlap * chars_per_token)
        step = max(1, char_target - char_overlap)
        out = []
        for start in range(0, len(text), step):
            chunk = text[start:start + char_target].strip()
            if chunk:
                out.append(chunk)
        return out


def _tail_by_tokens(text: str, overlap: int) -> str:
    if overlap <= 0 or not text:
        return ""
    try:
        import tiktoken  # type: ignore

        encoding_name = os.getenv("COMPAIR_CHUNK_TOKEN_ENCODING", "cl100k_base")
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        if not tokens:
            return ""
        tail = tokens[-overlap:]
        return encoding.decode(tail).strip()
    except Exception:
        chars_per_token = int(os.getenv("COMPAIR_CHUNK_CHARS_PER_TOKEN", "4"))
        char_overlap = max(0, overlap * chars_per_token)
        return text[-char_overlap:].strip() if char_overlap else ""


def chunk_text_smart(
    text: str,
    *,
    target_tokens: int | None = None,
    overlap_tokens: int | None = None,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
) -> list[str]:
    if not text or not text.strip():
        return []
    target, overlap, min_len, max_len = _chunking_defaults()
    if target_tokens is not None:
        target = target_tokens
    if overlap_tokens is not None:
        overlap = overlap_tokens
    if min_tokens is not None:
        min_len = min_tokens
    if max_tokens is not None:
        max_len = max_tokens
    max_len = max(max_len, target)
    token_counter = _TOKEN_COUNTER
    blocks = _split_blocks(text)
    if not blocks:
        return [text.strip()]

    chunks: list[str] = []
    cur_parts: list[str] = []
    cur_tokens = 0
    last_heading: str | None = None

    def flush() -> None:
        nonlocal cur_parts, cur_tokens, last_heading
        if not cur_parts:
            return
        chunk_text_val = "\n\n".join(cur_parts).strip()
        if chunk_text_val:
            chunks.append(chunk_text_val)
        cur_parts = []
        cur_tokens = 0
        last_heading = None

    for block in blocks:
        heading = block.get("heading")
        text_block = (block.get("text") or "").strip()
        if not text_block:
            continue
        prefix = ""
        if heading and heading != last_heading:
            prefix = f"{heading}\n"
            last_heading = heading
        block_text = f"{prefix}{text_block}".strip()
        block_tokens = token_counter(block_text)

        if block_tokens > max_len:
            if cur_parts:
                flush()
            for split in _split_by_tokens(block_text, target, overlap, token_counter):
                chunks.append(split)
            continue

        if cur_parts and (cur_tokens + block_tokens) > target and cur_tokens >= min_len:
            prev_chunk = "\n\n".join(cur_parts).strip()
            flush()
            overlap_text = _tail_by_tokens(prev_chunk, overlap)
            if overlap_text:
                cur_parts = [overlap_text]
                cur_tokens = token_counter(overlap_text)

        cur_parts.append(block_text)
        cur_tokens += block_tokens

    flush()

    if len(chunks) >= 2:
        last = chunks[-1]
        if token_counter(last) < min_len:
            merged = chunks[-2] + "\n\n" + last
            if token_counter(merged) <= max_len:
                chunks[-2] = merged
                chunks.pop()

    return [c for c in chunks if c]


def chunk_text_with_mode(text: str, chunk_mode: Optional[str] = None) -> list[str]:
    if not text:
        return []
    mode = (chunk_mode or "").strip().lower()
    if mode in {"client", "preserve", "prechunked", "auto"}:
        chunks = split_client_chunks(text)
        if chunks:
            return chunks
        if mode in {"client", "preserve", "prechunked"}:
            trimmed = text.strip()
            return [trimmed] if trimmed else []
    if mode in {"", "auto", "smart"}:
        chunks = split_client_chunks(text)
        if chunks:
            return chunks
        return chunk_text_smart(text)
    if mode in {"legacy", "paragraph"}:
        return chunk_text(text)
    return chunk_text_smart(text)


def generate_verification_token() -> tuple[str, datetime]:
    token = secrets.token_urlsafe(32)
    expiration = datetime.now(timezone.utc) + timedelta(hours=24)
    return token, expiration


def log_activity(
    session: Session,
    user_id: str,
    group_id: str,
    action: str,
    object_id: str,
    object_name: str,
    object_type: str,
) -> None:
    activity = Activity(
        user_id=user_id,
        group_id=group_id,
        action=action,
        object_id=object_id,
        object_name=object_name,
        object_type=object_type,
        timestamp=datetime.now(timezone.utc),
    )
    session.add(activity)
    session.commit()
