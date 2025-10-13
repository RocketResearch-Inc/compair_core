"""Billing provider placeholder for the Core edition."""
from __future__ import annotations

from .contracts import BillingSession


class NoopBilling:
    def ensure_customer(self, **_: object) -> str:
        raise NotImplementedError("Billing is not available in the Core edition.")

    def create_checkout_session(self, **_: object) -> BillingSession:
        raise NotImplementedError("Billing is not available in the Core edition.")

    def retrieve_session(self, *_: object, **__: object) -> BillingSession:
        raise NotImplementedError("Billing is not available in the Core edition.")

    def get_checkout_url(self, *_: object, **__: object) -> str:
        raise NotImplementedError("Billing is not available in the Core edition.")

    def create_customer_portal(self, *_: object, **__: object) -> str:
        raise NotImplementedError("Billing is not available in the Core edition.")

    def create_coupon(self, *_: object, **__: object) -> str:
        raise NotImplementedError("Billing is not available in the Core edition.")

    def apply_coupon(self, *_: object, **__: object) -> None:
        raise NotImplementedError("Billing is not available in the Core edition.")

    def construct_event(self, *_: object, **__: object) -> dict[str, object]:
        raise NotImplementedError("Billing is not available in the Core edition.")
