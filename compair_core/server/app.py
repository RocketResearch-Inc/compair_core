"""FastAPI app factory supporting Core and Cloud editions."""
from __future__ import annotations

from fastapi import FastAPI

from .deps import (
    get_analytics,
    get_billing,
    get_mailer,
    get_ocr,
    get_settings_dependency,
    get_storage,
)
from .providers.local_storage import LocalStorage
from .routers.capabilities import router as capabilities_router
from .settings import Settings, get_settings


def _normalize_edition(value: str) -> str:
    return (value or "core").lower()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Instantiate the FastAPI application with edition-specific wiring."""

    resolved_settings = settings or get_settings()
    edition = _normalize_edition(resolved_settings.edition)

    app = FastAPI(title="Compair API", version=resolved_settings.version)

    from ..api import core_router, router as legacy_router

    if edition == "cloud":
        app.include_router(legacy_router)
    else:
        if resolved_settings.include_legacy_routes:
            app.include_router(legacy_router)
        else:
            app.include_router(core_router)
    app.include_router(capabilities_router)

    # Share the resolved settings with request handlers
    app.dependency_overrides[get_settings_dependency] = lambda: resolved_settings

    if edition == "cloud":
        try:
            from compair_cloud.analytics.ga4 import GA4Analytics
            from compair_cloud.billing.stripe_provider import StripeBilling
            from compair_cloud.mailer.transactional import TransactionalMailer
            from compair_cloud.ocr.claude_ocr import ClaudeOCR
            from compair_cloud.storage.r2_storage import R2Storage
        except ImportError as exc:  # pragma: no cover - only triggered in misconfigured builds
            raise RuntimeError(
                "Cloud edition requires the private 'compair_cloud' package to be installed."
            ) from exc

        storage_provider = R2Storage(
            bucket=resolved_settings.r2_bucket,
            cdn_base=resolved_settings.r2_cdn_base,
            access_key=resolved_settings.r2_access_key,
            secret_key=resolved_settings.r2_secret_key,
            endpoint_url=resolved_settings.r2_endpoint_url,
        )
        billing_provider = StripeBilling(
            stripe_key=resolved_settings.stripe_key,
            endpoint_secret=resolved_settings.stripe_endpoint_secret,
        )
        ocr_provider = ClaudeOCR()
        mailer_provider = TransactionalMailer()

        analytics_provider = None
        if resolved_settings.ga4_measurement_id and resolved_settings.ga4_api_secret:
            analytics_provider = GA4Analytics(
                measurement_id=resolved_settings.ga4_measurement_id,
                api_secret=resolved_settings.ga4_api_secret,
            )

        app.dependency_overrides[get_storage] = lambda sp=storage_provider: sp
        app.dependency_overrides[get_billing] = lambda bp=billing_provider: bp
        app.dependency_overrides[get_ocr] = lambda op=ocr_provider: op
        if analytics_provider is not None:
            app.dependency_overrides[get_analytics] = lambda ap=analytics_provider: ap
        app.dependency_overrides[get_mailer] = lambda mp=mailer_provider: mp
        object.__setattr__(resolved_settings, "ocr_enabled", True)

    else:
        storage_provider = LocalStorage(
            base_dir=resolved_settings.local_upload_dir,
            base_url=resolved_settings.local_upload_base_url,
        )
        app.dependency_overrides[get_storage] = lambda sp=storage_provider: sp

    return app


# Uvicorn compatibility: allow ``uvicorn server.app:app``
app = create_app()
