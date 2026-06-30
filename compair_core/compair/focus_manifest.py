from __future__ import annotations

import fnmatch
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional


FocusManifest = Optional[Mapping[str, Any]]

MAX_FOCUS_MANIFEST_BYTES = 256_000
DEFAULT_MAX_BOOST = 3.0
DEFAULT_MIN_UNFOCUSED_FRACTION = 0.30


def normalize_focus_manifest(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if len(stripped.encode("utf-8")) > MAX_FOCUS_MANIFEST_BYTES:
            raise ValueError("focus_manifest is too large")
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"focus_manifest must be valid JSON: {exc}") from exc
    elif isinstance(value, Mapping):
        parsed = dict(value)
    else:
        raise ValueError("focus_manifest must be a JSON object")

    if not isinstance(parsed, Mapping):
        raise ValueError("focus_manifest must be a JSON object")
    encoded = json.dumps(parsed, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    if len(encoded) > MAX_FOCUS_MANIFEST_BYTES:
        raise ValueError("focus_manifest is too large")
    return dict(parsed)


def focus_manifest_hash(manifest: FocusManifest) -> str:
    if not manifest:
        return ""
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _focus_limits(manifest: FocusManifest) -> tuple[float, float]:
    if not isinstance(manifest, Mapping):
        return DEFAULT_MAX_BOOST, DEFAULT_MIN_UNFOCUSED_FRACTION
    limits = manifest.get("limits")
    if not isinstance(limits, Mapping):
        limits = {}
    max_boost = _float_value(
        limits.get("max_boost", manifest.get("max_boost", DEFAULT_MAX_BOOST)),
        DEFAULT_MAX_BOOST,
    )
    min_unfocused = _float_value(
        limits.get("min_unfocused_fraction", manifest.get("min_unfocused_fraction", DEFAULT_MIN_UNFOCUSED_FRACTION)),
        DEFAULT_MIN_UNFOCUSED_FRACTION,
    )
    return _clamp_float(max_boost, 0.0, 8.0), _clamp_float(min_unfocused, 0.0, 0.9)


def focus_manifest_min_unfocused_fraction(manifest: FocusManifest) -> float:
    _, min_unfocused = _focus_limits(manifest)
    return min_unfocused


def _entry_weight(entry: Mapping[str, Any], max_boost: float) -> float:
    explicit = entry.get("weight", entry.get("boost"))
    if explicit is not None:
        return _clamp_float(_float_value(explicit, 0.0), 0.0, max_boost)

    signal = max(
        _float_value(entry.get("combined_score"), 0.0),
        _float_value(entry.get("historical_score"), 0.0),
        _float_value(entry.get("layout_score"), 0.0),
    )
    if signal <= 0.0:
        return min(1.0, max_boost)
    return _clamp_float(0.75 + (signal / 10.0), 0.0, max_boost)


def _append_focus_entries(entries: list[dict[str, Any]], raw_entries: Any, max_boost: float) -> None:
    if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes, bytearray)):
        return
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            continue
        path_glob = str(
            raw_entry.get("glob")
            or raw_entry.get("path_glob")
            or raw_entry.get("path")
            or ""
        ).strip()
        if not path_glob:
            continue
        normalized = path_glob.replace("\\", "/").lstrip("/")
        entries.append(
            {
                "glob": normalized,
                "weight": _entry_weight(raw_entry, max_boost),
                "reason": str(raw_entry.get("reason") or raw_entry.get("label") or "").strip(),
            }
        )


def focus_manifest_entries(manifest: FocusManifest) -> list[dict[str, Any]]:
    if not isinstance(manifest, Mapping):
        return []
    max_boost, _ = _focus_limits(manifest)
    entries: list[dict[str, Any]] = []
    _append_focus_entries(entries, manifest.get("areas"), max_boost)
    _append_focus_entries(entries, manifest.get("focus_areas"), max_boost)
    _append_focus_entries(entries, manifest.get("priority_areas"), max_boost)

    repos = manifest.get("repos")
    if isinstance(repos, Mapping):
        for repo_value in repos.values():
            if isinstance(repo_value, Mapping):
                _append_focus_entries(entries, repo_value.get("areas"), max_boost)
                _append_focus_entries(entries, repo_value.get("focus_areas"), max_boost)
            else:
                _append_focus_entries(entries, repo_value, max_boost)

    orgs = manifest.get("orgs")
    if isinstance(orgs, Sequence) and not isinstance(orgs, (str, bytes, bytearray)):
        for org in orgs:
            if isinstance(org, Mapping):
                _append_focus_entries(entries, org.get("focus_areas"), max_boost)
                _append_focus_entries(entries, org.get("areas"), max_boost)

    return [entry for entry in entries if _float_value(entry.get("weight"), 0.0) > 0.0]


def focus_manifest_enabled(manifest: FocusManifest) -> bool:
    return bool(focus_manifest_entries(manifest))


def _path_matches_glob(path: str, path_glob: str) -> bool:
    normalized_path = (path or "").strip().replace("\\", "/").lstrip("/")
    normalized_glob = (path_glob or "").strip().replace("\\", "/").lstrip("/")
    if not normalized_path or not normalized_glob:
        return False
    return (
        fnmatch.fnmatch(normalized_path, normalized_glob)
        or fnmatch.fnmatch(normalized_path.lower(), normalized_glob.lower())
    )


def focus_score_for_path(path: str, manifest: FocusManifest) -> float:
    score = 0.0
    for entry in focus_manifest_entries(manifest):
        if _path_matches_glob(path, str(entry.get("glob") or "")):
            score = max(score, _float_value(entry.get("weight"), 0.0))
    return score


def focus_match_for_path(path: str, manifest: FocusManifest) -> bool:
    return focus_score_for_path(path, manifest) > 0.0


def focus_selected_counts(
    indices: Sequence[int],
    manifest: FocusManifest,
    path_for_index: Callable[[int], str],
) -> dict[str, int | str]:
    if not focus_manifest_enabled(manifest):
        return {"manifest_hash": "", "focused": 0, "unfocused": len(indices)}
    focused = 0
    for idx in indices:
        if focus_match_for_path(path_for_index(idx), manifest):
            focused += 1
    return {
        "manifest_hash": focus_manifest_hash(manifest),
        "focused": focused,
        "unfocused": max(0, len(indices) - focused),
    }


def reserve_unfocused_indices(
    selected: Sequence[int],
    candidates: Sequence[int],
    manifest: FocusManifest,
    path_for_index: Callable[[int], str],
) -> list[int]:
    selected_list = list(selected)
    if len(selected_list) <= 1 or not focus_manifest_enabled(manifest):
        return selected_list
    min_unfocused_fraction = focus_manifest_min_unfocused_fraction(manifest)
    if min_unfocused_fraction <= 0.0:
        return selected_list

    min_unfocused = int(len(selected_list) * min_unfocused_fraction)
    if len(selected_list) * min_unfocused_fraction > min_unfocused:
        min_unfocused += 1
    if min_unfocused <= 0:
        return selected_list

    selected_set = set(selected_list)
    unfocused_selected = [idx for idx in selected_list if not focus_match_for_path(path_for_index(idx), manifest)]
    needed = min_unfocused - len(unfocused_selected)
    if needed <= 0:
        return selected_list

    replacements = [
        idx
        for idx in candidates
        if idx not in selected_set and not focus_match_for_path(path_for_index(idx), manifest)
    ]
    if not replacements:
        return selected_list

    balanced = list(selected_list)
    for replacement in replacements[:needed]:
        for pos in range(len(balanced) - 1, -1, -1):
            if focus_match_for_path(path_for_index(balanced[pos]), manifest):
                balanced[pos] = replacement
                break
    return balanced
