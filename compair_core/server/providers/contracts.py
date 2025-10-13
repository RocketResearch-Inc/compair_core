"""Provider protocol definitions shared across editions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO, Iterable, Mapping, Protocol


@dataclass(slots=True)
class BillingSession:
    """Represents the result of creating a checkout session."""

    id: str
    url: str


class StorageProvider(Protocol):
    def put_file(self, key: str, fileobj: BinaryIO, content_type: str) -> str: ...

    def get_file(self, key: str) -> tuple[BinaryIO, str]: ...

    def build_url(self, key: str) -> str: ...


class BillingProvider(Protocol):
    def ensure_customer(self, *, user_email: str, user_id: str) -> str: ...

    def create_checkout_session(
        self,
        *,
        customer_id: str,
        price_id: str,
        qty: int,
        success_url: str,
        cancel_url: str,
        metadata: Mapping[str, str] | None = None,
    ) -> BillingSession: ...

    def retrieve_session(self, session_id: str) -> BillingSession: ...

    def get_checkout_url(self, session_id: str) -> str: ...

    def create_customer_portal(self, *, customer_id: str, return_url: str) -> str: ...

    def create_coupon(self, amount: int) -> str: ...

    def apply_coupon(self, *, customer_id: str, coupon_id: str) -> None: ...

    def construct_event(self, payload: bytes, signature: str | None) -> Mapping[str, object]: ...


class OCRProvider(Protocol):
    def submit(
        self, *, user_id: str, filename: str, data: bytes, document_id: str | None
    ) -> str: ...

    def status(self, task_id: str) -> Mapping[str, object]: ...


class Mailer(Protocol):
    def send(self, subject: str, sender: str, receivers: Iterable[str], html: str) -> None: ...


class Analytics(Protocol):
    def track(
        self, event_name: str, user_id: str, params: Mapping[str, object] | None = None
    ) -> None: ...
