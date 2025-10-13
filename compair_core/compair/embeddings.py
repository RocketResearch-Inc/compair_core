import hashlib
import os
from typing import List

import requests

from .logger import log_event

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
            self.model = os.getenv("COMPAIR_LOCAL_EMBED_MODEL", "hash-embedding")
            self.dimension = int(os.getenv("COMPAIR_LOCAL_EMBED_DIM", "384"))
            base_url = os.getenv("COMPAIR_LOCAL_MODEL_URL", "http://local-model:9000")
            route = os.getenv("COMPAIR_LOCAL_EMBED_ROUTE", "/embed")
            self.endpoint = f"{base_url.rstrip('/')}{route}"

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

    # Local/core path
    try:
        response = requests.post(embedder.endpoint, json={"text": text}, timeout=15)
        response.raise_for_status()
        data = response.json()
        embedding = data.get("embedding") or data.get("vector")
        if embedding:
            return embedding
    except Exception as exc:
        log_event("local_embedding_failed", error=str(exc))

    return _hash_embedding(text, embedder.dimension)
