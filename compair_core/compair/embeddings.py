import hashlib
import os
from typing import Any, List, Optional

import requests

from .logger import log_event

try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    openai = None  # type: ignore

try:
    from compair_cloud.embeddings import Embedder as CloudEmbedder  # type: ignore
    from compair_cloud.embeddings import create_embedding as cloud_create_embedding  # type: ignore
except (ImportError, ModuleNotFoundError):
    CloudEmbedder = None
    cloud_create_embedding = None


class Embedder:
    def __init__(self) -> None:
        self.edition = os.getenv("COMPAIR_EDITION", "core").lower()
        self._cloud_impl = None
        if self.edition == "cloud" and CloudEmbedder is not None:
            self._cloud_impl = CloudEmbedder()

        if self._cloud_impl is None:
            self.provider = os.getenv("COMPAIR_EMBEDDING_PROVIDER", "local").lower()
            self.model = os.getenv("COMPAIR_LOCAL_EMBED_MODEL", "hash-embedding")
            default_dim = 1536 if self.edition == "cloud" else 384
            dim_env = (
                os.getenv("COMPAIR_EMBEDDING_DIM")
                or os.getenv("COMPAIR_EMBEDDING_DIMENSION")
                or os.getenv("COMPAIR_LOCAL_EMBED_DIM")
                or str(default_dim)
            )
            try:
                self.dimension = int(dim_env)
            except ValueError:  # pragma: no cover - invalid configuration
                self.dimension = default_dim
            base_url = os.getenv("COMPAIR_LOCAL_MODEL_URL", "http://127.0.0.1:9000")
            route = os.getenv("COMPAIR_LOCAL_EMBED_ROUTE", "/embed")
            self.endpoint = f"{base_url.rstrip('/')}{route}"
            self.openai_embed_model = os.getenv("COMPAIR_OPENAI_EMBED_MODEL", "text-embedding-3-small")
            self._openai_client: Optional[Any] = None
            if self.provider == "openai":
                if openai is None:
                    log_event("openai_embedding_unavailable", reason="openai_library_missing")
                    self.provider = "local"
                else:
                    api_key = os.getenv("COMPAIR_OPENAI_API_KEY")
                    if hasattr(openai, "api_key") and api_key:
                        openai.api_key = api_key  # type: ignore[assignment]
                    if hasattr(openai, "OpenAI"):
                        try:  # pragma: no cover - optional runtime dependency
                            self._openai_client = openai.OpenAI(api_key=api_key)  # type: ignore[attr-defined]
                        except Exception:  # pragma: no cover - if instantiation fails
                            self._openai_client = None

    @property
    def is_cloud(self) -> bool:
        return self._cloud_impl is not None


def _hash_embedding(text: str, dimension: int) -> List[float]:
    """Generate a deterministic embedding using repeated SHA-256 hashing."""
    if not text:
        text = " "
    digest = hashlib.sha256(text.encode("utf-8", "ignore")).digest()
    vector: List[float] = []
    while len(vector) < dimension:
        for byte in digest:
            vector.append((byte / 255.0) * 2 - 1)
            if len(vector) == dimension:
                break
        digest = hashlib.sha256(digest).digest()
    return vector


def create_embedding(embedder: Embedder, text: str, user=None) -> list[float]:
    if embedder.is_cloud and cloud_create_embedding is not None:
        return cloud_create_embedding(embedder._cloud_impl, text, user=user)

    provider = getattr(embedder, "provider", "local")
    if provider == "openai" and openai is not None:
        vector = _openai_embedding(embedder, text)
        if vector:
            return vector

    # Local/core path
    endpoint = getattr(embedder, "endpoint", None)
    if endpoint:
        try:
            response = requests.post(endpoint, json={"text": text}, timeout=15)
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding") or data.get("vector")
            if embedding:
                return embedding
        except Exception as exc:
            log_event("local_embedding_failed", error=str(exc))

    return _hash_embedding(text, embedder.dimension)


def _openai_embedding(embedder: Embedder, text: str) -> list[float] | None:
    if openai is None:
        return None
    client = getattr(embedder, "_openai_client", None)
    if client is None and hasattr(openai, "OpenAI"):
        api_key = os.getenv("COMPAIR_OPENAI_API_KEY")
        try:  # pragma: no cover - optional client differences
            client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()  # type: ignore[attr-defined]
        except TypeError:
            client = openai.OpenAI()
        embedder._openai_client = client  # type: ignore[attr-defined]

    try:
        if client is not None and hasattr(client, "embeddings"):
            response = client.embeddings.create(
                model=embedder.openai_embed_model,
                input=text,
            )
            data = getattr(response, "data", None)
            if data:
                vector = getattr(data[0], "embedding", None)
                if isinstance(vector, list):
                    return vector
        elif hasattr(openai, "Embedding"):
            response = openai.Embedding.create(  # type: ignore[attr-defined]
                model=embedder.openai_embed_model,
                input=text,
            )
            vector = response["data"][0]["embedding"]  # type: ignore[index]
            if isinstance(vector, list):
                return vector
    except Exception as exc:  # pragma: no cover - network/API failure
        log_event("openai_embedding_failed", error=str(exc))
    return None
