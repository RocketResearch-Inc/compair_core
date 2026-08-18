"""Pure validation and generation for the protected baseline-run keyring."""

from __future__ import annotations

import base64
import json
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

RUN_KEYRING_VERSION = "baseline-run-keyring.v1"
RUN_KEY_BYTES = 32
_SAFE_KEY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]{0,127}$")


class RunKeyringValidationError(ValueError):
    """Sanitized validation failure for an external run keyring."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class RunKeyringGenerationError(RuntimeError):
    """Sanitized failure while generating a new run keyring."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True, repr=False)
class ParsedRunKeyring:
    """Validated keyring data with a deliberately redacted representation."""

    active_key_id: str
    keys: Mapping[str, bytes]

    def __repr__(self) -> str:
        return "ParsedRunKeyring(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class GeneratedRunKeyring:
    """New serialized keyring; callers must never log or print this object."""

    serialized: str
    active_key_id: str
    key_count: int

    def __repr__(self) -> str:
        return "GeneratedRunKeyring(<redacted>)"


def _invalid(code: str = "run_keyring_invalid") -> RunKeyringValidationError:
    return RunKeyringValidationError(code)


def parse_run_keyring(raw: str) -> ParsedRunKeyring:
    """Parse the exact production ``baseline-run-keyring.v1`` contract."""

    if not isinstance(raw, str) or not raw:
        raise _invalid("run_keyring_unavailable")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _invalid()
            result[key] = value
        return result

    def reject_nonfinite(_value: str) -> None:
        raise _invalid()

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except RunKeyringValidationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError):
        raise _invalid() from None
    if not isinstance(value, Mapping) or set(value) != {
        "version",
        "active_key_id",
        "keys",
    }:
        raise _invalid()
    active = value["active_key_id"]
    entries = value["keys"]
    if (
        value["version"] != RUN_KEYRING_VERSION
        or not isinstance(active, str)
        or _SAFE_KEY_ID.fullmatch(active) is None
        or not isinstance(entries, list)
        or not entries
    ):
        raise _invalid()
    keys: dict[str, bytes] = {}
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {"key_id", "key_base64"}:
            raise _invalid()
        key_id = entry["key_id"]
        encoded = entry["key_base64"]
        if (
            not isinstance(key_id, str)
            or _SAFE_KEY_ID.fullmatch(key_id) is None
            or key_id in keys
            or not isinstance(encoded, str)
        ):
            raise _invalid()
        try:
            key = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            raise _invalid() from None
        if len(key) != RUN_KEY_BYTES:
            raise _invalid()
        keys[key_id] = key
    if active not in keys:
        raise _invalid()
    return ParsedRunKeyring(active, MappingProxyType(keys))


def generate_run_keyring(
    *,
    key_factory: Callable[[int], bytes] = secrets.token_bytes,
    key_id_factory: Callable[[int], str] = secrets.token_urlsafe,
) -> GeneratedRunKeyring:
    """Generate and validate one new production keyring without publishing it."""

    try:
        key = key_factory(RUN_KEY_BYTES)
        random_key_id = key_id_factory(24)
    except Exception:  # noqa: BLE001 - failure details must never escape
        raise RunKeyringGenerationError("random_generation_failed") from None
    key_id = f"key-{random_key_id}" if isinstance(random_key_id, str) else ""
    if (
        not isinstance(key, bytes)
        or len(key) != RUN_KEY_BYTES
        or not isinstance(key_id, str)
        or _SAFE_KEY_ID.fullmatch(key_id) is None
    ):
        raise RunKeyringGenerationError("random_generation_failed")
    payload = {
        "version": RUN_KEYRING_VERSION,
        "active_key_id": key_id,
        "keys": [
            {
                "key_id": key_id,
                "key_base64": base64.b64encode(key).decode("ascii"),
            }
        ],
    }
    try:
        serialized = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        parsed = parse_run_keyring(serialized)
    except (RunKeyringValidationError, TypeError, ValueError):
        raise RunKeyringGenerationError("serialization_failed") from None
    if (
        parsed.active_key_id != key_id
        or len(parsed.keys) != 1
        or parsed.keys[key_id] != key
    ):
        raise RunKeyringGenerationError("serialization_failed")
    return GeneratedRunKeyring(serialized, key_id, 1)


__all__ = [
    "RUN_KEYRING_VERSION",
    "RUN_KEY_BYTES",
    "GeneratedRunKeyring",
    "ParsedRunKeyring",
    "RunKeyringGenerationError",
    "RunKeyringValidationError",
    "generate_run_keyring",
    "parse_run_keyring",
]
