"""Application settings and feature flag definitions."""
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration injected via COMPAIR_ environment variables."""

    # Edition metadata
    edition: str = "core"  # core | cloud
    version: str = "dev"

    # Feature gates
    ocr_enabled: bool = False
    billing_enabled: bool = False
    integrations_enabled: bool = False
    premium_models: bool = False
    require_authentication: bool = False
    single_user_username: str = "compair-local@example.com"
    single_user_name: str = "Compair Local User"
    include_legacy_routes: bool = False

    # Core/local storage defaults
    local_upload_dir: str = "~/.compair-core/data/uploads"
    local_upload_base_url: str = "/uploads"

    # Cloud storage (R2/S3-compatible)
    r2_bucket: str | None = None
    r2_cdn_base: str | None = None
    r2_access_key: str | None = None
    r2_secret_key: str | None = None
    r2_endpoint_url: str | None = None

    # Optional cloud secrets
    stripe_key: str | None = None
    stripe_endpoint_secret: str | None = None
    stripe_success_url: str = "https://compair.sh/home"
    stripe_cancel_url: str = "https://compair.sh/home"
    ga4_measurement_id: str | None = None
    ga4_api_secret: str | None = None

    # Local model endpoints
    local_model_url: str = "http://local-model:9000"
    local_embedding_route: str = "/embed"
    local_generation_route: str = "/generate"

    class Config:
        env_prefix = "COMPAIR_"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance for dependency injection."""
    return Settings()
