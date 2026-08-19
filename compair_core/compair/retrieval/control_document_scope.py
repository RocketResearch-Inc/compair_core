"""Canonical corpus identity for document-level baseline control-plane work."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

import rfc8785

CONTROL_DOCUMENT_CORPUS_SCOPE_VERSION = "baseline-control-document-corpus-scope.v1"
CONTROL_DOCUMENT_CORPUS_SCOPE_PREFIX = (
    f"{CONTROL_DOCUMENT_CORPUS_SCOPE_VERSION}:sha256:"
)
CONTROL_DOCUMENT_CORPUS_SCOPE_MAX_LENGTH = 256


class ControlDocumentCorpusScopeError(RuntimeError):
    """Sanitized internal scope failure with no source metadata."""

    def __init__(self, code: str = "control_document_corpus_scope_conflict") -> None:
        self.code = code
        super().__init__(code)


def _identity_component(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value
        or len(value.encode("utf-8")) > 256
        or any(ord(character) < 32 for character in value)
    ):
        raise ControlDocumentCorpusScopeError(f"{label}_invalid")
    return value


@dataclass(frozen=True, slots=True)
class ControlDocumentCorpusIdentity:
    """Exact authorization identity for one independently active corpus."""

    group_id: str
    changed_repository_registration_id: str
    source_document_id: str
    contract_version: str = CONTROL_DOCUMENT_CORPUS_SCOPE_VERSION

    @classmethod
    def create(
        cls,
        *,
        group_id: str,
        changed_repository_registration_id: str,
        source_document_id: str,
    ) -> ControlDocumentCorpusIdentity:
        return cls(
            group_id=_identity_component(group_id, "group_id"),
            changed_repository_registration_id=_identity_component(
                changed_repository_registration_id,
                "changed_repository_registration_id",
            ),
            source_document_id=_identity_component(
                source_document_id,
                "source_document_id",
            ),
        )

    def __post_init__(self) -> None:
        if self.contract_version != CONTROL_DOCUMENT_CORPUS_SCOPE_VERSION:
            raise ControlDocumentCorpusScopeError("scope_contract_version_invalid")
        for value, label in (
            (self.group_id, "group_id"),
            (
                self.changed_repository_registration_id,
                "changed_repository_registration_id",
            ),
            (self.source_document_id, "source_document_id"),
        ):
            if _identity_component(value, label) != value:
                raise ControlDocumentCorpusScopeError(f"{label}_invalid")

    @property
    def canonical_payload(self) -> dict[str, str]:
        return {
            "changed_repository_registration_id": (
                self.changed_repository_registration_id
            ),
            "group_id": self.group_id,
            "scope_contract_version": self.contract_version,
            "source_document_id": self.source_document_id,
        }

    @property
    def identity_sha256(self) -> str:
        try:
            canonical = rfc8785.dumps(self.canonical_payload)
        except (TypeError, ValueError, rfc8785.CanonicalizationError):
            raise ControlDocumentCorpusScopeError("scope_identity_invalid") from None
        return hashlib.sha256(canonical).hexdigest()

    @property
    def scope_key(self) -> str:
        value = f"{CONTROL_DOCUMENT_CORPUS_SCOPE_PREFIX}{self.identity_sha256}"
        if len(value) > CONTROL_DOCUMENT_CORPUS_SCOPE_MAX_LENGTH:
            raise ControlDocumentCorpusScopeError("scope_key_too_long")
        return value

    @property
    def legacy_group_scope_key(self) -> str:
        return f"group:{self.group_id}"

    @property
    def accepted_scope_keys(self) -> tuple[str, str]:
        """New key first, exact-match legacy key second."""

        return self.scope_key, self.legacy_group_scope_key

    def matches_stored_corpus(
        self,
        *,
        scope_key: object,
        changed_repository_id: object,
        source_document_id: object,
    ) -> bool:
        return (
            changed_repository_id == self.changed_repository_registration_id
            and source_document_id == self.source_document_id
            and scope_key in self.accepted_scope_keys
        )


def control_document_corpus_identity(
    *,
    group_id: str,
    changed_repository_registration_id: str,
    source_document_id: str,
) -> ControlDocumentCorpusIdentity:
    """Construct the one frozen document-level corpus identity."""

    return ControlDocumentCorpusIdentity.create(
        group_id=group_id,
        changed_repository_registration_id=changed_repository_registration_id,
        source_document_id=source_document_id,
    )


def choose_control_document_corpus_scope_key(
    identity: ControlDocumentCorpusIdentity,
    existing_corpora: Iterable[tuple[object, object, object]],
) -> str:
    """Choose an exact legacy row or the source-specific canonical key.

    Each tuple is ``(scope_key, changed_repository_id, source_document_id)``.
    An existing canonical key with different stored identities is corruption or
    a cryptographic collision and therefore fails closed.
    """

    canonical_match = False
    legacy_match = False
    for scope_key, changed_repository_id, source_document_id in existing_corpora:
        exact_identity = (
            changed_repository_id == identity.changed_repository_registration_id
            and source_document_id == identity.source_document_id
        )
        if scope_key == identity.legacy_group_scope_key:
            legacy_match = exact_identity
        elif scope_key == identity.scope_key:
            if not exact_identity:
                raise ControlDocumentCorpusScopeError()
            canonical_match = True
    if legacy_match:
        return identity.legacy_group_scope_key
    if canonical_match:
        return identity.scope_key
    return identity.scope_key
