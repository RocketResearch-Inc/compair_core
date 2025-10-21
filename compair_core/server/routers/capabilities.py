"""Meta endpoints that describe edition capabilities for the CLI."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..settings import Settings, get_settings

router = APIRouter(tags=["meta"])


@router.get("/capabilities")
def capabilities(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    edition = settings.edition.lower()
    require_auth = settings.require_authentication
    return {
        "auth": {
            "device_flow": edition == "cloud",
            "password_login": require_auth,
            "required": require_auth,
            "single_user": not require_auth,
        },
        "inputs": {
            "text": True,
            "ocr": settings.ocr_enabled,
            "repos": True,
        },
        "models": {
            "premium": settings.premium_models,
            "open": True,
        },
        "integrations": {
            "slack": settings.integrations_enabled,
            "github": settings.integrations_enabled,
        },
        "limits": {
            "docs": None if edition == "core" else 100,
            "feedback_per_day": None if edition == "core" else 50,
        },
        "features": {
            "ocr_upload": settings.ocr_enabled,
            "activity_feed": edition == "cloud",
        },
        "server": "Compair Cloud" if edition == "cloud" else "Compair Core",
        "version": settings.version,
        "legacy_routes": settings.include_legacy_routes,
    }
