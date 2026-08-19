"""Exact local context-budget profile for the qualified native Ollama tuple."""

from __future__ import annotations

import bisect
import hashlib
import heapq
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any

from ..compair.retrieval.generation import BaselineGenerationProviderError
from .profile import (
    QUALIFIED_BUDGET_PROFILE_ASSET_SHA256,
    QUALIFIED_BUDGET_PROFILE_FINGERPRINT,
    QUALIFIED_CONTEXT_TOKENS,
    QUALIFIED_OLLAMA_RUNTIME_VERSION,
    QUALIFIED_OUTPUT_TOKENS,
    RECOMMENDED_GENERATION_MODEL,
    RECOMMENDED_GENERATION_MODEL_DIGEST,
)

BUDGET_PROFILE_ASSET = "qwen3-14b-ollama-0.32.14.profile.json"
BUDGET_PROFILE_ASSET_SHA256 = QUALIFIED_BUDGET_PROFILE_ASSET_SHA256
BUDGET_PROFILE_FINGERPRINT = QUALIFIED_BUDGET_PROFILE_FINGERPRINT


def _unsupported() -> BaselineGenerationProviderError:
    return BaselineGenerationProviderError(
        "unsupported_runtime",
        "baseline Ollama generation is unavailable",
        retryable=False,
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class QualifiedBudgetProfile:
    fingerprint: str
    system_prefix: str
    system_suffix: str
    user_prefix: str
    user_suffix: str
    attestation_system: str
    attestation_user: str
    tokenizer: Qwen2Tokenizer

    def render(self, system: str, user: str) -> str:
        return (
            self.system_prefix
            + system
            + self.system_suffix
            + self.user_prefix
            + user
            + self.user_suffix
        )

    @property
    def attestation_render(self) -> str:
        return self.render(self.attestation_system, self.attestation_user)

    def count(self, system: str, user: str) -> int:
        return len(self.tokenizer.encode(self.render(system, user)))


class Qwen2Tokenizer:
    """Port of the pinned llama.cpp qwen2 pre-tokenizer and BPE session."""

    def __init__(self, data: dict[str, Any]) -> None:
        tokens = data.get("tokens")
        merges = data.get("merges")
        specials = data.get("special_tokens")
        ranges = data.get("unicode_ranges_flags")
        lowercase = data.get("unicode_map_lowercase")
        whitespace = data.get("unicode_set_whitespace")
        if not all(
            isinstance(value, list)
            for value in (tokens, merges, specials, ranges, lowercase, whitespace)
        ):
            raise _unsupported()
        self._vocab = {token: token_id for token_id, token in enumerate(tokens)}
        if len(self._vocab) != len(tokens):
            raise _unsupported()
        self._merges: dict[tuple[str, str], int] = {}
        for rank, merge in enumerate(merges):
            if not isinstance(merge, str) or " " not in merge:
                raise _unsupported()
            left, right = merge.split(" ", 1)
            self._merges[(left, right)] = rank
        self._specials = sorted(
            ((str(value[0]), int(value[1])) for value in specials),
            key=lambda item: len(item[0].encode("utf-8")),
            reverse=True,
        )
        self._ranges = [(int(value[0]), int(value[1])) for value in ranges]
        self._range_starts = [value[0] for value in self._ranges]
        self._lowercase = {int(value[0]): int(value[1]) for value in lowercase}
        self._whitespace = {int(value) for value in whitespace}
        direct = list(range(ord("!"), ord("~") + 1))
        direct.extend(range(0xA1, 0xAC + 1))
        direct.extend(range(0xAE, 0xFF + 1))
        remaining = [value for value in range(256) if value not in direct]
        self._byte_characters = {value: chr(value) for value in direct}
        self._byte_characters.update(
            {value: chr(256 + index) for index, value in enumerate(remaining)}
        )

    def _flags(self, codepoint: int) -> int:
        if not 0 <= codepoint < 0x110000:
            return 0
        index = bisect.bisect_right(self._range_starts, codepoint) - 1
        flags = self._ranges[index][1]
        if codepoint in self._whitespace:
            flags |= 0x0100
        return flags

    def _split(self, text: str) -> list[str]:
        codepoints = [ord(character) for character in text]
        words: list[str] = []
        previous = 0
        position = 0
        end = len(codepoints)

        def codepoint(at: int) -> int:
            return codepoints[at] if 0 <= at < end else 0xFFFFFFFF

        def flags(at: int) -> int:
            return self._flags(codepoint(at)) if 0 <= at < end else 0

        def add(at: int) -> int:
            nonlocal previous
            if at > previous:
                words.append("".join(chr(value) for value in codepoints[previous:at]))
            length = at - previous
            previous = at
            return length

        while position < end:
            current = codepoint(position)
            current_flags = flags(position)
            if current == ord("'") and position + 1 < end:
                following = self._lowercase.get(
                    codepoint(position + 1), codepoint(position + 1)
                )
                if following in (ord("s"), ord("t"), ord("m"), ord("d")):
                    position += add(position + 2)
                    continue
                if position + 2 < end:
                    next_following = self._lowercase.get(
                        codepoint(position + 2), codepoint(position + 2)
                    )
                    if (following, next_following) in (
                        (ord("r"), ord("e")),
                        (ord("v"), ord("e")),
                        (ord("l"), ord("l")),
                    ):
                        position += add(position + 3)
                        continue
            if (
                current not in (ord("\r"), ord("\n"))
                and not (current_flags & 0x0002)
                and ((current_flags & 0x0004) or (flags(position + 1) & 0x0004))
            ):
                position += 1
                while flags(position) & 0x0004:
                    position += 1
                add(position)
                continue
            if current_flags & 0x0002:
                position += 1
                add(position)
                continue
            following_flags = (
                flags(position + 1) if current == ord(" ") else current_flags
            )
            if not (following_flags & (0x0100 | 0x0004 | 0x0002)) and current_flags:
                position += current == ord(" ")
                while (
                    not (following_flags & (0x0100 | 0x0004 | 0x0002))
                    and following_flags
                ):
                    position += 1
                    following_flags = flags(position)
                while codepoint(position) in (ord("\r"), ord("\n")):
                    position += 1
                add(position)
                continue
            whitespace_count = 0
            last_newline = 0
            while flags(position + whitespace_count) & 0x0100:
                if codepoint(position + whitespace_count) in (ord("\r"), ord("\n")):
                    last_newline = position + whitespace_count + 1
                whitespace_count += 1
            if last_newline:
                position = last_newline
                add(position)
                continue
            if (
                whitespace_count > 1
                and codepoint(position + whitespace_count) != 0xFFFFFFFF
            ):
                position += whitespace_count - 1
                add(position)
                continue
            if whitespace_count:
                position += whitespace_count
                add(position)
                continue
            position += 1
            add(position)
        return words

    def _bpe(self, word: str) -> list[int]:
        encoded = "".join(
            self._byte_characters[value] for value in word.encode("utf-8")
        )
        symbols = [
            {
                "text": character,
                "prev": index - 1,
                "next": index + 1 if index + 1 < len(encoded) else -1,
                "live": True,
            }
            for index, character in enumerate(encoded)
        ]
        queue: list[tuple[int, int, int, str]] = []

        def add(left: int, right: int) -> None:
            if (
                left < 0
                or right < 0
                or not symbols[left]["live"]
                or not symbols[right]["live"]
            ):
                return
            pair = (str(symbols[left]["text"]), str(symbols[right]["text"]))
            rank = self._merges.get(pair)
            if rank is not None:
                heapq.heappush(queue, (rank, left, right, pair[0] + pair[1]))

        for index in range(1, len(symbols)):
            add(index - 1, index)
        while queue:
            _rank, left, right, merged = heapq.heappop(queue)
            if not symbols[left]["live"] or not symbols[right]["live"]:
                continue
            if str(symbols[left]["text"]) + str(symbols[right]["text"]) != merged:
                continue
            symbols[left]["text"] = merged
            symbols[right]["live"] = False
            symbols[left]["next"] = symbols[right]["next"]
            right_next = int(symbols[right]["next"])
            if right_next >= 0:
                symbols[right_next]["prev"] = left
            add(int(symbols[left]["prev"]), left)
            add(left, int(symbols[left]["next"]))
        output: list[int] = []
        for symbol in symbols:
            if not symbol["live"]:
                continue
            piece = str(symbol["text"])
            token_id = self._vocab.get(piece)
            if token_id is None:
                try:
                    output.extend(self._vocab[character] for character in piece)
                except KeyError:
                    raise _unsupported() from None
            else:
                output.append(token_id)
        return output

    def _encode_raw(self, text: str) -> list[int]:
        output: list[int] = []
        for word in self._split(text):
            output.extend(self._bpe(word))
        return output

    def encode(self, text: str) -> list[int]:
        fragments: list[tuple[str, str | int]] = [("raw", text)]
        for special, token_id in self._specials:
            updated: list[tuple[str, str | int]] = []
            for kind, value in fragments:
                if kind != "raw":
                    updated.append((kind, value))
                    continue
                pieces = str(value).split(special)
                for index, piece in enumerate(pieces):
                    if piece:
                        updated.append(("raw", piece))
                    if index + 1 < len(pieces):
                        updated.append(("special", token_id))
            fragments = updated
        output: list[int] = []
        for kind, value in fragments:
            if kind == "special":
                output.append(int(value))
            else:
                output.extend(self._encode_raw(str(value)))
        return output


@lru_cache(maxsize=1)
def qualified_budget_profile() -> QualifiedBudgetProfile:
    raw = (
        files("compair_core.baseline_generation")
        .joinpath(BUDGET_PROFILE_ASSET)
        .read_bytes()
    )
    if hashlib.sha256(raw).hexdigest() != BUDGET_PROFILE_ASSET_SHA256:
        raise _unsupported()
    try:
        data = json.loads(raw.decode("utf-8", errors="strict"))
        fingerprint = data.pop("profile_fingerprint")
        calculated = hashlib.sha256(_canonical(data)).hexdigest()
        qualification = data["qualification"]
        framing = data["framing"]
        tokenizer_data = data["tokenizer"]
    except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        raise _unsupported() from None
    if fingerprint != BUDGET_PROFILE_FINGERPRINT or calculated != fingerprint:
        raise _unsupported()
    if qualification != {
        "context_tokens": QUALIFIED_CONTEXT_TOKENS,
        "model": RECOMMENDED_GENERATION_MODEL,
        "model_digest": RECOMMENDED_GENERATION_MODEL_DIGEST,
        "ollama_truncate": False,
        "ollama_version": QUALIFIED_OLLAMA_RUNTIME_VERSION,
        "output_tokens": QUALIFIED_OUTPUT_TOKENS,
        "schema_consumes_prompt_tokens": False,
    }:
        raise _unsupported()
    if not isinstance(framing, dict) or not isinstance(tokenizer_data, dict):
        raise _unsupported()
    try:
        return QualifiedBudgetProfile(
            fingerprint=fingerprint,
            system_prefix=str(framing["system_prefix"]),
            system_suffix=str(framing["system_suffix"]),
            user_prefix=str(framing["user_prefix"]),
            user_suffix=str(framing["user_suffix"]),
            attestation_system=str(framing["attestation_system"]),
            attestation_user=str(framing["attestation_user"]),
            tokenizer=Qwen2Tokenizer(tokenizer_data),
        )
    except (KeyError, TypeError, ValueError):
        raise _unsupported() from None


__all__ = [
    "BUDGET_PROFILE_ASSET",
    "BUDGET_PROFILE_ASSET_SHA256",
    "BUDGET_PROFILE_FINGERPRINT",
    "QualifiedBudgetProfile",
    "Qwen2Tokenizer",
    "qualified_budget_profile",
]
