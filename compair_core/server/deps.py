"""Dependency entry points for features that differ by edition."""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from .providers.console_mailer import ConsoleMailer
from .providers.contracts import Analytics, BillingProvider, Mailer, OCRProvider, StorageProvider
from .providers.http_ocr import HTTPOCR
from .providers.local_storage import LocalStorage
from .providers.noop_analytics import NoopAnalytics
from .providers.noop_billing import NoopBilling
from .providers.noop_ocr import NoopOCR
from .settings import Settings, get_settings


@lru_cache
def _local_storage_factory(base_dir: str, base_url: str) -> LocalStorage:
    return LocalStorage(base_dir=base_dir, base_url=base_url)


@lru_cache
def _noop_billing() -> NoopBilling:
    return NoopBilling()


@lru_cache
def _noop_ocr() -> NoopOCR:
    return NoopOCR()


@lru_cache
def _http_ocr(endpoint: str, timeout: float) -> HTTPOCR:
    return HTTPOCR(endpoint=endpoint, timeout=timeout)


@lru_cache
def _console_mailer() -> ConsoleMailer:
    return ConsoleMailer()


@lru_cache
def _noop_analytics() -> NoopAnalytics:
    return NoopAnalytics()


def get_settings_dependency() -> Settings:
    """Expose settings as a FastAPI dependency."""
    return get_settings()



def get_storage(
    settings: Settings = Depends(get_settings_dependency),
) -> StorageProvider:
    return _local_storage_factory(settings.local_upload_dir, settings.local_upload_base_url)


def get_billing() -> BillingProvider:
    return _noop_billing()


def get_ocr(
    settings: Settings = Depends(get_settings_dependency),
) -> OCRProvider:
    if settings.ocr_enabled and settings.ocr_endpoint:
        return _http_ocr(settings.ocr_endpoint, settings.ocr_request_timeout)
    return _noop_ocr()


def get_mailer() -> Mailer:
    return _console_mailer()


def get_analytics() -> Analytics:
    return _noop_analytics()
