"""Application settings and feature flag definitions."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings

from ..baseline_generation.profile import (
    ACCELERATED_GENERATION_TIMEOUT_SECONDS,
    MAXIMUM_GENERATION_TIMEOUT_SECONDS,
    QUALIFIED_CONTEXT_TOKENS,
    QUALIFIED_OUTPUT_TOKENS,
    RECOMMENDED_GENERATION_MODEL,
    RECOMMENDED_GENERATION_MODEL_DIGEST,
)


class Settings(BaseSettings):
    """Configuration injected via COMPAIR_ environment variables."""

    # Edition metadata
    edition: str = "core"  # core | cloud
    version: str = "dev"

    # Feature gates
    ocr_enabled: bool = True
    billing_enabled: bool = False
    integrations_enabled: bool = False
    premium_models: bool = False
    require_authentication: bool = False
    require_email_verification: bool = False
    single_user_username: str = "compair-local@example.com"
    single_user_name: str = "Compair Local User"
    include_legacy_routes: bool = False
    cors_allow_origins: str | None = None

    # Explicit retrieval-query transport. This is intentionally off by
    # default and can only relax policy for direct/eager/test/loopback paths.
    retrieval_engine: str = "legacy"
    retrieval_query_allow_insecure_local_transport: bool = False

    # Staging control-plane writes require HTTPS. Plain HTTP is available only
    # when this explicit override is set for a direct loopback peer. A reverse
    # proxy is trusted only when its immediate peer IP matches this comma-
    # separated IP/CIDR allowlist and it attests one unambiguous HTTPS scheme.
    baseline_control_plane_allow_insecure_loopback: bool = False
    baseline_control_plane_trusted_proxy_allowlist: str = ""

    # Public baseline-control-plane.v2 run submission is separately gated.
    # Enabling it permits durable submission only after runtime readiness;
    # dispatch remains manual unless the separate database worker mode is set.
    # The API process itself never starts a worker or Celery task.
    baseline_runs_enabled: bool = False

    # The durable database worker is a separate executable. ``manual`` keeps
    # the existing operator-only capability and never assumes that a worker is
    # running. ``database`` requires a recent compatible heartbeat and bounded
    # queue capacity before protected run admission.
    baseline_worker_mode: Literal["manual", "database"] = "manual"
    baseline_worker_poll_interval_seconds: float = Field(
        default=2.0,
        ge=0.1,
        le=60.0,
    )
    baseline_worker_heartbeat_interval_seconds: float = Field(
        default=5.0,
        ge=0.5,
        le=60.0,
    )
    baseline_worker_heartbeat_ttl_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
    )
    baseline_worker_cleanup_interval_seconds: int = Field(
        default=30,
        ge=1,
        le=3600,
    )
    baseline_worker_max_pending_per_slot: int = Field(
        default=8,
        ge=1,
        le=64,
    )
    baseline_worker_max_attempts: int = Field(default=5, ge=1, le=100)
    baseline_worker_max_backoff_seconds: float = Field(
        default=30.0,
        ge=1.0,
        le=MAXIMUM_GENERATION_TIMEOUT_SECONDS,
    )

    # Baseline_v1 uses a separate fail-closed embedding provider. It never
    # inherits or falls back to the legacy embedding configuration.
    baseline_embedding_provider: Literal["disabled", "http"] = "disabled"
    baseline_embedding_endpoint: str | None = None
    baseline_embedding_model: str = "BAAI/bge-small-en-v1.5"
    baseline_embedding_revision: str | None = None
    baseline_embedding_dimension: int = Field(default=384, ge=1, le=8192)
    baseline_embedding_timeout_seconds: float = Field(
        default=10.0,
        ge=0.1,
        le=60.0,
    )
    baseline_embedding_batch_size: int = Field(default=32, ge=1, le=256)
    baseline_embedding_allow_insecure_loopback: bool = False
    baseline_model_cache: str | None = None

    # Baseline generation is independent of legacy generation routing. The
    # native Ollama mode attests the configured tag's immutable digest before
    # sending any source or evidence bytes and never falls back or pulls.
    baseline_generation_provider: Literal["disabled", "http", "ollama"] = "disabled"
    baseline_generation_endpoint: str | None = None
    baseline_generation_model: str | None = RECOMMENDED_GENERATION_MODEL
    baseline_generation_model_digest: str | None = RECOMMENDED_GENERATION_MODEL_DIGEST
    baseline_generation_model_version: str | None = None
    baseline_generation_timeout_seconds: float = Field(
        default=ACCELERATED_GENERATION_TIMEOUT_SECONDS,
        ge=0.1,
        le=300.0,
    )
    baseline_generation_allow_loopback_http: bool = False
    baseline_generation_max_request_bytes: int = Field(
        default=256_000,
        ge=4_096,
        le=8_000_000,
    )
    baseline_generation_max_response_bytes: int = Field(
        default=200_000,
        ge=4_096,
        le=1_000_000,
    )
    baseline_generation_context_tokens: int = Field(
        default=QUALIFIED_CONTEXT_TOKENS,
        ge=2_048,
        le=131_072,
    )
    baseline_generation_output_tokens: int = Field(
        default=QUALIFIED_OUTPUT_TOKENS,
        ge=64,
        le=4_096,
    )
    baseline_generation_seed: int = Field(
        default=0,
        ge=0,
        le=2_147_483_647,
    )

    # Baseline-run submissions use a separate external AES-256-GCM keyring.
    # SecretStr prevents accidental settings/repr disclosure; the value is a
    # strict baseline-run-keyring.v1 JSON object parsed only by the run service.
    baseline_run_encryption_keyring: SecretStr | None = None
    baseline_run_payload_ttl_seconds: int = Field(default=900, ge=60, le=3600)

    # Baseline notification delivery remains an explicit default-off setting.
    baseline_notifications_enabled: bool = False

    # Core/local storage defaults
    local_upload_dir: str = "~/.compair-core/data/uploads"
    local_upload_base_url: str = "/uploads"

    # Cloud storage (R2/S3-compatible)
    r2_bucket: str | None = None
    r2_cdn_base: str | None = None
    r2_access_key: str | None = None
    r2_secret_key: str | None = None
    r2_account_id: str | None = None
    r2_endpoint_url: str | None = None

    # Optional cloud secrets
    stripe_key: str | None = None
    stripe_endpoint_secret: str | None = None
    stripe_success_url: str = "https://compair.sh/home"
    stripe_cancel_url: str = "https://compair.sh/home"
    review_now_credit_price_id: str | None = None
    review_now_credit_pack_cents: int = 1000
    review_now_max_quote_cents: int = 200
    review_now_min_charge_cents: int = 1
    review_now_quote_ttl_sec: int = 1800
    ga4_measurement_id: str | None = None
    ga4_api_secret: str | None = None
    telemetry_enabled: bool = False
    telemetry_base_url: str = "https://app.compair.sh/api"
    telemetry_install_id: str | None = None
    google_oauth_enabled: bool = False
    google_oauth_client_id: str | None = None
    google_oauth_client_secret: str | None = None
    google_oauth_redirect_uri: str | None = None
    google_oauth_state_secret: str | None = None
    notification_unsubscribe_secret: str | None = None
    google_oauth_web_success_url: str | None = None
    google_oauth_web_error_url: str | None = None
    google_oauth_device_ttl_sec: int = 600

    # Local model endpoints
    local_model_url: str = "http://127.0.0.1:9000"
    local_embedding_route: str = "/embed"
    local_generation_route: str = "/generate"

    # OCR
    ocr_endpoint: str | None = "http://127.0.0.1:9001"
    ocr_request_timeout: float = 30.0

    class Config:
        env_prefix = "COMPAIR_"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance for dependency injection."""
    settings = Settings()
    # Auto-enable OCR when a local endpoint is configured (Core) unless explicitly disabled.
    if settings.ocr_endpoint and not settings.ocr_enabled:
        object.__setattr__(settings, "ocr_enabled", True)
    if not settings.ocr_endpoint and settings.edition.lower() != "cloud":
        object.__setattr__(settings, "ocr_enabled", False)
    return settings
