"""Meta endpoints that describe edition capabilities for the CLI."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text

from ...compair.notifications.service import is_scoring_enabled
from ...compair.retrieval.embedding import assess_baseline_embedding
from ...compair.retrieval.transport import assess_retrieval_query_transport
from ...db import engine
from ...runtime_config import build_runtime_configuration
from ..settings import Settings, get_settings

router = APIRouter(tags=["meta"])


def _retrieval_query_transport(settings: Settings) -> dict[str, object]:
    # Import lazily so capability discovery preserves Core's optional Cloud
    # task boundary and reports the task implementation actually installed.
    from ...compair.tasks import process_document_task

    return assess_retrieval_query_transport(
        process_document_task,
        allow_insecure_local_transport=(
            settings.retrieval_query_allow_insecure_local_transport
        ),
    ).as_dict()


def _baseline_embedding(settings: Settings) -> dict[str, object]:
    return assess_baseline_embedding(settings).as_dict()


def _runtime_configuration(settings: Settings) -> dict[str, object]:
    try:
        return build_runtime_configuration(
            settings,
            database_url=engine.url,
        ).safe_summary()
    except Exception:  # noqa: BLE001 - capability output is non-reflective
        return {
            "contract_version": "baseline-runtime-config.v1",
            "status": "unavailable",
        }


@router.get("/capabilities")
def capabilities(
    settings: Settings = Depends(get_settings),  # noqa: B008 - FastAPI dependency
) -> dict[str, object]:
    edition = settings.edition.lower()
    require_auth = settings.require_authentication
    google_oauth_configured = (
        settings.google_oauth_enabled
        and bool((settings.google_oauth_client_id or "").strip())
        and bool((settings.google_oauth_client_secret or "").strip())
        and bool((settings.google_oauth_redirect_uri or "").strip())
        and edition == "cloud"
    )
    return {
        "auth": {
            "device_flow": edition == "cloud",
            "password_login": require_auth,
            "password_reset": require_auth,
            "required": require_auth,
            "single_user": not require_auth,
            "google_oauth": google_oauth_configured,
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
            "activity_feed": True,
            "notification_events": True,
            "notification_scoring": is_scoring_enabled(),
            "notification_preferences": True,
            "notification_delivery": edition == "cloud",
        },
        "baseline_embedding": _baseline_embedding(settings),
        "baseline_runtime_configuration": _runtime_configuration(settings),
        "retrieval_query_transport": _retrieval_query_transport(settings),
        "server": "Compair Cloud" if edition == "cloud" else "Compair Core",
        "version": settings.version,
        "legacy_routes": settings.include_legacy_routes,
    }


@router.get("/health")
def health(
    settings: Settings = Depends(get_settings),  # noqa: B008 - FastAPI dependency
) -> dict[str, object]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="database_unavailable") from exc
    return {
        "status": "ok",
        "edition": settings.edition,
        "version": settings.version,
        "baseline_embedding": _baseline_embedding(settings),
        "baseline_runtime_configuration": _runtime_configuration(settings),
        "retrieval_query_transport": _retrieval_query_transport(settings),
    }
