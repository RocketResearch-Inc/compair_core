from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence


_RAW_TOKEN_RE = re.compile(r"--[A-Za-z0-9][A-Za-z0-9-]*|[A-Za-z0-9_./:-]{2,}")
_SUBTOKEN_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+")
_QUOTED_RE = re.compile(r"""(?P<quote>["'`])(?P<value>[^"'`\n]{1,120})(?P=quote)""")
_ASSIGNMENT_RE = re.compile(
    r"""
    ^\s*
    (?:
        ["'`](?P<quoted_key>[A-Za-z_][A-Za-z0-9_.-]{0,79})["'`]
        |
        (?P<plain_key>[A-Za-z_][A-Za-z0-9_.-]{0,79})
    )
    \s*(?:\?|!)?\s*[:=]\s*(?P<value>.+?)\s*$
    """,
    re.VERBOSE,
)
_PATH_RE = re.compile(
    r"""
    https?://[^\s"'`]+
    |
    (?:/[A-Za-z0-9._~%:+-]+)+(?:/[A-Za-z0-9._~%:+-]*)?
    |
    \b[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+\b
    """,
    re.VERBOSE,
)
_DIFF_LINE_RE = re.compile(r"^\s*[+-](?![+-])")
_META_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s+|generated:|document(?:\s+id)?:|summary:|content budget:|files included:)",
    re.IGNORECASE,
)
_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "into", "will", "have",
    "about", "there", "their", "while", "which", "when", "where", "would", "should",
    "could", "after", "before", "because", "through", "against", "between", "under",
    "over", "without", "these", "those", "they", "them", "then", "than", "only",
    "still", "being", "been", "also", "does", "used", "using", "docs", "document",
    "documents", "chunk", "file", "files", "line", "lines", "part", "parts",
    "section", "sections", "reference", "references", "related", "content", "changed",
    "text", "texts", "says", "say",
}


@dataclass(frozen=True)
class ReferenceText:
    label: str
    text: str


@dataclass(frozen=True)
class AssignmentArtifact:
    key: str
    value: str
    normalized_value: str
    value_tokens: frozenset[str]
    paths: tuple[str, ...]


@dataclass(frozen=True)
class CompoundArtifact:
    full: str
    prefix: tuple[str, ...]
    member: str


@dataclass(frozen=True)
class ArtifactProfile:
    text: str
    tokens: frozenset[str]
    key_names: frozenset[str]
    quoted_values: tuple[str, ...]
    quoted_norm: frozenset[str]
    paths: tuple[str, ...]
    path_tokens: frozenset[str]
    assignments: tuple[AssignmentArtifact, ...]
    compounds: tuple[CompoundArtifact, ...]
    compound_prefixes: frozenset[tuple[str, ...]]
    structured_score: int
    looks_prose: bool


@dataclass(frozen=True)
class RelationAssessment:
    kind: str | None
    confidence: int
    target_artifact: str = ""
    peer_artifact: str = ""


@dataclass(frozen=True)
class ReferenceMatch:
    reference_label: str
    target_excerpt: str
    peer_excerpt: str
    relation: RelationAssessment
    score: int


def normalize_text(value: str | None) -> str:
    return " ".join((value or "").split())


def reference_label_from_text(text: str) -> str:
    for line in (text or "").splitlines():
        candidate = normalize_text(line)
        if not candidate:
            continue
        return candidate[:120]
    return "a related reference"


def _clean_diff_prefix(line: str) -> str:
    stripped = line.rstrip()
    if re.match(r"^[+-](?![+-])", stripped):
        stripped = stripped[1:]
    return normalize_text(stripped)


def _expand_token(token: str) -> set[str]:
    expanded = {token.lower()}
    for piece in re.split(r"[_./:-]+", token):
        lowered = piece.strip().lower()
        if lowered:
            expanded.add(lowered)
        for subtoken in _SUBTOKEN_RE.findall(piece):
            normalized = subtoken.strip().lower()
            if normalized:
                expanded.add(normalized)
    return expanded


def excerpt_tokens(*values: str | None) -> set[str]:
    out: set[str] = set()
    for value in values:
        if not value:
            continue
        for raw in _RAW_TOKEN_RE.findall(value):
            for token in _expand_token(raw):
                if len(token) < 3 or token in _STOPWORDS:
                    continue
                out.add(token)
    return out


def _extract_paths(text: str) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for match in _PATH_RE.findall(text or ""):
        candidate = normalize_text(match)
        lowered = candidate.lower()
        if len(candidate) < 3 or lowered in seen:
            continue
        seen.add(lowered)
        out.append(candidate)
    return tuple(out[:24])


def _clean_value_text(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"\s*(?://|#).*$", "", cleaned)
    cleaned = cleaned.rstrip(",;")
    return normalize_text(cleaned[:240])


def _extract_assignments(text: str) -> tuple[AssignmentArtifact, ...]:
    assignments: list[AssignmentArtifact] = []
    for raw_line in (text or "").splitlines():
        line = _clean_diff_prefix(raw_line)
        if not line:
            continue
        match = _ASSIGNMENT_RE.match(line)
        if not match:
            continue
        key = (match.group("quoted_key") or match.group("plain_key") or "").strip()
        if not key:
            continue
        value = _clean_value_text(match.group("value") or "")
        if not value:
            continue
        assignments.append(
            AssignmentArtifact(
                key=key.lower(),
                value=value,
                normalized_value=value.lower(),
                value_tokens=frozenset(excerpt_tokens(value)),
                paths=_extract_paths(value),
            )
        )
    return tuple(assignments[:32])


def _extract_quoted_values(text: str) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for match in _QUOTED_RE.finditer(text or ""):
        candidate = normalize_text(match.group("value"))
        lowered = candidate.lower()
        if len(candidate) < 2 or lowered in seen:
            continue
        seen.add(lowered)
        out.append(candidate)
    return tuple(out[:24])


def _extract_compounds(text: str) -> tuple[CompoundArtifact, ...]:
    out: list[CompoundArtifact] = []
    seen: set[tuple[tuple[str, ...], str]] = set()
    for raw in _RAW_TOKEN_RE.findall(text or ""):
        if raw.startswith(("http://", "https://", "/", "--")):
            continue
        if not any(sep in raw for sep in ("_", ".", "-")):
            continue
        parts = [part.lower() for part in re.split(r"[_.-]+", raw) if len(part) >= 2]
        if len(parts) < 2:
            continue
        for depth in range(1, len(parts)):
            prefix = tuple(parts[:depth])
            member = parts[depth]
            key = (prefix, member)
            if key in seen:
                continue
            seen.add(key)
            out.append(CompoundArtifact(full=".".join((*prefix, member)), prefix=prefix, member=member))
    return tuple(out[:48])


def extract_artifacts(text: str) -> ArtifactProfile:
    normalized = normalize_text(text)
    tokens = frozenset(excerpt_tokens(normalized))
    assignments = _extract_assignments(text)
    quoted_values = _extract_quoted_values(text)
    quoted_norm = frozenset(value.lower() for value in quoted_values)
    paths = _extract_paths(text)
    path_tokens = frozenset(excerpt_tokens(*paths))
    compounds = _extract_compounds(text)
    key_names = frozenset(assignment.key for assignment in assignments)
    compound_prefixes = frozenset(compound.prefix for compound in compounds if compound.prefix)
    structured_score = (
        min(len(assignments), 2)
        + min(len(paths), 2)
        + min(len(compounds), 2)
        + min(len(quoted_values), 1)
    )
    alpha_tokens = sum(1 for token in tokens if token.isalpha())
    looks_prose = structured_score <= 1 and alpha_tokens >= 6
    return ArtifactProfile(
        text=normalized,
        tokens=tokens,
        key_names=key_names,
        quoted_values=quoted_values,
        quoted_norm=quoted_norm,
        paths=paths,
        path_tokens=path_tokens,
        assignments=assignments,
        compounds=compounds,
        compound_prefixes=compound_prefixes,
        structured_score=structured_score,
        looks_prose=looks_prose,
    )


def _segments(text: str) -> list[str]:
    lines = [_clean_diff_prefix(line) for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        flat = normalize_text(text)
        return [flat] if flat else []

    out: list[str] = []
    for idx, line in enumerate(lines):
        window = [line]
        out.append(line)
        for span in range(1, 3):
            if idx + span >= len(lines):
                break
            window.append(lines[idx + span])
            out.append(" ".join(window))

    deduped: list[str] = []
    seen: set[str] = set()
    for segment in out:
        if segment in seen:
            continue
        seen.add(segment)
        deduped.append(segment)
    return deduped[:240]


def _changed_line_signals(text: str) -> tuple[str, ...]:
    lines = (text or "").splitlines()
    if not lines:
        return ()

    out: list[str] = []
    seen: set[str] = set()
    indexed_changed_lines: list[tuple[int, str]] = []
    for idx, raw_line in enumerate(lines):
        if _DIFF_LINE_RE.match(raw_line):
            indexed_changed_lines.append((idx, raw_line))

    ordered_changed_lines = [
        (idx, raw_line)
        for idx, raw_line in indexed_changed_lines
        if raw_line.lstrip().startswith("+")
    ] + [
        (idx, raw_line)
        for idx, raw_line in indexed_changed_lines
        if raw_line.lstrip().startswith("-")
    ]

    for idx, raw_line in ordered_changed_lines:
        cleaned = _clean_diff_prefix(raw_line)
        if not cleaned:
            continue
        candidates = [cleaned]
        next_idx = idx + 1
        if next_idx < len(lines) and _DIFF_LINE_RE.match(lines[next_idx]):
            paired = normalize_text(f"{cleaned} {_clean_diff_prefix(lines[next_idx])}")
            if paired and paired != cleaned:
                candidates.append(paired)
        for candidate in candidates:
            lowered = candidate.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            out.append(candidate)
            if len(out) >= 24:
                return tuple(out)
    return tuple(out)


def _merge_signal_texts(*groups: Sequence[str]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            candidate = normalize_text(value)
            lowered = candidate.lower()
            if not candidate or lowered in seen:
                continue
            seen.add(lowered)
            merged.append(candidate)
    return tuple(merged)


def _segment_penalty(segment: str, profile: ArtifactProfile) -> int:
    penalty = 0
    if _META_LINE_RE.match(segment):
        penalty += 3
    if segment.lstrip().startswith("```"):
        penalty += 2
    if segment.lstrip().startswith("#") and profile.structured_score == 0:
        penalty += 2
    if len(profile.tokens) <= 1:
        penalty += 1
    return penalty


def _overlap_score(left: ArtifactProfile, right: ArtifactProfile) -> int:
    score = len(left.tokens & right.tokens)
    score += 2 * len(left.key_names & right.key_names)
    score += 2 * len(left.quoted_norm & right.quoted_norm)
    score += 2 * len(left.path_tokens & right.path_tokens)
    score += 2 * len(left.compound_prefixes & right.compound_prefixes)
    return score


def _signal_overlap_score(profile: ArtifactProfile, signal_texts: Sequence[str]) -> int:
    if not signal_texts:
        return 0
    return len(profile.tokens & excerpt_tokens(*signal_texts))


def best_grounded_excerpt(
    text: str,
    signal_texts: Sequence[str],
    preferred_excerpt: str = "",
    *,
    anchor_texts: Sequence[str] = (),
    limit: int = 280,
) -> str:
    source = normalize_text(text)
    if not source:
        return ""

    preferred = normalize_text(preferred_excerpt)
    if preferred and preferred in source:
        return preferred[:limit]

    signal_profile = extract_artifacts(" ".join(signal_texts))
    preferred_profile = extract_artifacts(preferred)
    anchor_profile = extract_artifacts(" ".join(anchor_texts))

    best_segment = ""
    best_score = -10
    for segment in _segments(text):
        segment_profile = extract_artifacts(segment)
        if not segment_profile.tokens and segment_profile.structured_score == 0:
            continue
        score = _overlap_score(segment_profile, signal_profile)
        score += 2 * _overlap_score(segment_profile, preferred_profile)
        score += 3 * _overlap_score(segment_profile, anchor_profile)
        if anchor_texts:
            score += 2 * _signal_overlap_score(segment_profile, anchor_texts)
            if any(normalize_text(anchor) == segment for anchor in anchor_texts if anchor):
                score += 3
        score -= _segment_penalty(segment, segment_profile)
        if score > best_score or (score == best_score and score > -10 and len(segment) < len(best_segment)):
            best_score = score
            best_segment = segment

    if best_score <= 0:
        return ""
    return best_segment[:limit]


def _paths_share_shape(left: str, right: str) -> bool:
    left_tokens = excerpt_tokens(left)
    right_tokens = excerpt_tokens(right)
    return bool(left_tokens & right_tokens)


def _is_prose_like(profile: ArtifactProfile, text: str) -> bool:
    if profile.looks_prose:
        return True
    if len(text.split()) < 6:
        return False
    return len(profile.assignments) == 0 and profile.structured_score <= 2


def _is_weak_peer_match(
    target: ArtifactProfile,
    peer: ArtifactProfile,
    relation: RelationAssessment,
    overlap: int,
    anchor_overlap: int,
) -> bool:
    if relation.kind is None:
        return True
    if relation.kind == "generic divergence" and (overlap < 3 or anchor_overlap < 2):
        return True
    if peer.structured_score == 0 and not peer.quoted_values and relation.kind != "docs-vs-impl mismatch":
        if overlap < 4 or anchor_overlap < 2:
            return True
    if target.structured_score >= 2 and peer.structured_score == 0 and relation.kind != "docs-vs-impl mismatch":
        return True
    return False


def assess_relation(target_excerpt: str, peer_excerpt: str) -> RelationAssessment:
    target = extract_artifacts(target_excerpt)
    peer = extract_artifacts(peer_excerpt)
    support_score = _overlap_score(target, peer)
    target_is_prose = _is_prose_like(target, target_excerpt)
    peer_is_prose = _is_prose_like(peer, peer_excerpt)

    best: RelationAssessment | None = None

    def consider(candidate: RelationAssessment | None) -> None:
        nonlocal best
        if candidate is None:
            return
        if best is None or candidate.confidence > best.confidence:
            best = candidate

    target_assignments = {assignment.key: assignment for assignment in target.assignments}
    peer_assignments = {assignment.key: assignment for assignment in peer.assignments}

    for key in sorted(target.key_names & peer.key_names):
        target_assignment = target_assignments.get(key)
        peer_assignment = peer_assignments.get(key)
        if target_assignment is None or peer_assignment is None:
            continue
        if target_assignment.normalized_value == peer_assignment.normalized_value:
            continue
        kind = "route/path mismatch" if target_assignment.paths or peer_assignment.paths else "value mismatch"
        confidence = 5 + len(target_assignment.value_tokens & peer_assignment.value_tokens)
        consider(
            RelationAssessment(
                kind=kind,
                confidence=confidence,
                target_artifact=target_assignment.value or target_assignment.key,
                peer_artifact=peer_assignment.value or peer_assignment.key,
            )
        )

    for target_compound in target.compounds:
        for peer_compound in peer.compounds:
            if target_compound.prefix != peer_compound.prefix or target_compound.member == peer_compound.member:
                continue
            confidence = 4 + min(len(target_compound.prefix), 3)
            consider(
                RelationAssessment(
                    kind="rename",
                    confidence=confidence,
                    target_artifact=target_compound.full,
                    peer_artifact=peer_compound.full,
                )
            )

    for target_assignment in target.assignments:
        for peer_assignment in peer.assignments:
            if target_assignment.key == peer_assignment.key:
                continue
            if not target_assignment.value_tokens or not peer_assignment.value_tokens:
                continue
            overlap = len(target_assignment.value_tokens & peer_assignment.value_tokens)
            if overlap < 2:
                continue
            consider(
                RelationAssessment(
                    kind="rename",
                    confidence=4 + overlap,
                    target_artifact=target_assignment.key,
                    peer_artifact=peer_assignment.key,
                )
            )

    if target.paths and peer.paths and not set(path.lower() for path in target.paths) & set(path.lower() for path in peer.paths):
        for target_path in target.paths:
            for peer_path in peer.paths:
                if not _paths_share_shape(target_path, peer_path):
                    continue
                consider(
                    RelationAssessment(
                        kind="route/path mismatch",
                        confidence=4 + len(excerpt_tokens(target_path) & excerpt_tokens(peer_path)),
                        target_artifact=target_path,
                        peer_artifact=peer_path,
                    )
                )

    if (target_is_prose and peer.structured_score >= 2) or (peer_is_prose and target.structured_score >= 2):
        if support_score >= 1 or target.paths or peer.paths or target.quoted_norm & peer.quoted_norm:
            consider(RelationAssessment(kind="docs-vs-impl mismatch", confidence=4))

    if support_score >= 2:
        target_only_keys = sorted(target.key_names - peer.key_names)
        if target_only_keys:
            consider(
                RelationAssessment(
                    kind="presence/absence",
                    confidence=3,
                    target_artifact=target_only_keys[0],
                )
            )
        elif target.paths:
            peer_paths_lower = {path.lower() for path in peer.paths}
            for path in target.paths:
                if path.lower() not in peer_paths_lower:
                    consider(
                        RelationAssessment(
                            kind="presence/absence",
                            confidence=3,
                            target_artifact=path,
                        )
                    )
                    break

    if best is not None:
        return best
    if support_score >= 2:
        return RelationAssessment(kind="generic divergence", confidence=1)
    return RelationAssessment(kind=None, confidence=0)


def summarize_comparison(
    target_excerpt: str,
    peer_excerpt: str,
    reference_label: str = "a related reference",
    relation: RelationAssessment | None = None,
) -> str | None:
    target_excerpt = normalize_text(target_excerpt)
    peer_excerpt = normalize_text(peer_excerpt)
    if not target_excerpt or not peer_excerpt:
        return None

    relation = relation or assess_relation(target_excerpt, peer_excerpt)
    if relation.kind is None and relation.confidence <= 0:
        return None

    if relation.kind == "rename" and relation.target_artifact and relation.peer_artifact:
        return (
            f'This may rename "{relation.peer_artifact}" to "{relation.target_artifact}" '
            f'in the changed content, while {reference_label} still uses "{relation.peer_artifact}".'
        )
    if relation.kind == "presence/absence" and relation.target_artifact:
        return (
            f'The changed content introduces "{relation.target_artifact}", but {reference_label} '
            "does not show the corresponding item."
        )
    if relation.kind == "docs-vs-impl mismatch":
        return (
            f'The changed text says "{target_excerpt}", while the reference implementation or '
            f'config in {reference_label} says "{peer_excerpt}".'
        )
    if relation.kind in {"value mismatch", "route/path mismatch", "generic divergence"}:
        return (
            f'The changed content says "{target_excerpt}", while {reference_label} says "{peer_excerpt}".'
        )
    return None


def best_reference_match(
    text: str,
    references: Sequence[ReferenceText],
    *,
    focus_text: str = "",
    change_context: str = "",
) -> ReferenceMatch | None:
    cleaned_references = [ref for ref in references if normalize_text(ref.text)]
    if not normalize_text(text) or not cleaned_references:
        return None

    focus = normalize_text(focus_text)
    change = normalize_text(change_context)
    target_source_text = focus or text
    reference_signal = " ".join(ref.text for ref in cleaned_references[:6])
    changed_signals = _merge_signal_texts(
        _changed_line_signals(change_context),
        _changed_line_signals(target_source_text),
    )
    target_signals = [reference_signal]
    if focus:
        target_signals.append(focus)
    if change:
        target_signals.append(change)
    target_signals.extend(changed_signals)
    preferred_target_excerpt = changed_signals[0] if changed_signals else focus
    target_excerpt = best_grounded_excerpt(
        target_source_text,
        target_signals,
        preferred_target_excerpt,
        anchor_texts=changed_signals,
        limit=240,
    )
    if not target_excerpt:
        target_excerpt = best_grounded_excerpt(
            target_source_text,
            [target_source_text, *changed_signals],
            preferred_target_excerpt,
            anchor_texts=changed_signals,
            limit=240,
        )
    if not target_excerpt:
        return None

    target_profile = extract_artifacts(target_excerpt)
    best_match: ReferenceMatch | None = None
    for reference in cleaned_references[:6]:
        peer_anchor_texts = [target_excerpt, *changed_signals]
        peer_excerpt = best_grounded_excerpt(
            reference.text,
            [text, target_source_text, target_excerpt, *changed_signals],
            anchor_texts=peer_anchor_texts,
            limit=240,
        )
        if not peer_excerpt:
            continue
        relation = assess_relation(target_excerpt, peer_excerpt)
        peer_profile = extract_artifacts(peer_excerpt)
        overlap = _overlap_score(target_profile, peer_profile)
        anchor_overlap = _signal_overlap_score(peer_profile, changed_signals or (target_excerpt,))
        if _is_weak_peer_match(target_profile, peer_profile, relation, overlap, anchor_overlap):
            continue
        score = overlap + 4 * relation.confidence + (2 * anchor_overlap)
        if relation.kind is None and overlap < 2:
            continue
        candidate = ReferenceMatch(
            reference_label=reference.label,
            target_excerpt=target_excerpt,
            peer_excerpt=peer_excerpt,
            relation=relation,
            score=score,
        )
        if best_match is None or candidate.score > best_match.score:
            best_match = candidate
    return best_match


def summarize_reference_feedback(
    text: str,
    references: Sequence[ReferenceText],
    *,
    focus_text: str = "",
    change_context: str = "",
) -> str | None:
    match = best_reference_match(text, references, focus_text=focus_text, change_context=change_context)
    if match is None:
        return None
    return summarize_comparison(match.target_excerpt, match.peer_excerpt, match.reference_label, match.relation)


def reference_payload_texts(references: Iterable[ReferenceText], limit: int = 4) -> list[str]:
    payload: list[str] = []
    for reference in references:
        snippet = (reference.text or "").strip()
        if not snippet:
            continue
        payload.append(f"{reference.label}\n{snippet}" if reference.label else snippet)
        if len(payload) >= limit:
            break
    return payload
