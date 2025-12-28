from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9']{2,}")

_STOPWORDS = {
    "a", "about", "above", "across", "after", "again", "against", "all", "almost", "along",
    "also", "although", "always", "among", "an", "and", "any", "are", "around", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can", "cannot", "could", "did", "do", "does", "doing", "down", "during", "each",
    "either", "else", "enough", "etc", "even", "ever", "every", "for", "from", "further",
    "get", "got", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its",
    "itself", "just", "keep", "let", "like", "likely", "may", "me", "might", "more", "most",
    "much", "must", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "one", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "within", "without",
    "would", "you", "your", "yours", "yourself", "yourselves",
}


def _tokenize(text: str) -> list[str]:
    words = _WORD_RE.findall(text or "")
    tokens: list[str] = []
    for w in words:
        t = w.lower()
        if t in _STOPWORDS:
            continue
        if len(t) < 3:
            continue
        tokens.append(t)
    return tokens


def _bigrams(tokens: Iterable[str]) -> list[str]:
    items = list(tokens)
    out: list[str] = []
    for i in range(len(items) - 1):
        a = items[i]
        b = items[i + 1]
        if a in _STOPWORDS or b in _STOPWORDS:
            continue
        out.append(f"{a} {b}")
    return out


def extract_topic_tags(text: str, max_tags: int = 8) -> list[str]:
    """
    Lightweight keyword extraction for topic tags.
    No external dependencies; safe for core/local deployments.
    """
    if not text or not text.strip():
        return []

    tokens = _tokenize(text)
    if not tokens:
        return []
    if len(tokens) > 5000:
        tokens = tokens[:5000]

    unigram_counts = Counter(tokens)
    bigram_counts = Counter(_bigrams(tokens))

    scores: dict[str, int] = {}
    for term, count in unigram_counts.items():
        scores[term] = scores.get(term, 0) + count
    for term, count in bigram_counts.items():
        scores[term] = scores.get(term, 0) + (count * 2)

    candidates = [
        (term, score)
        for term, score in scores.items()
        if 3 <= len(term) <= 48
    ]
    candidates.sort(key=lambda x: (-x[1], -len(x[0]), x[0]))

    tags: list[str] = []
    for term, _ in candidates:
        if len(tags) >= max_tags:
            break
        if term in tags:
            continue
        tags.append(term)
    return tags
