import hashlib
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, List, Optional

import requests
import tiktoken

from .logger import log_event

try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    openai = None  # type: ignore

try:
    from compair_cloud.embeddings import Embedder as CloudEmbedder  # type: ignore
    from compair_cloud.embeddings import create_embedding as cloud_create_embedding  # type: ignore
    from compair_cloud.embeddings import create_embeddings as cloud_create_embeddings  # type: ignore
except (ImportError, ModuleNotFoundError):
    CloudEmbedder = None
    cloud_create_embedding = None
    cloud_create_embeddings = None


def _openai_api_key() -> str | None:
    return os.getenv("COMPAIR_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")


def _openai_base_url() -> str | None:
    return os.getenv("COMPAIR_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage_total_tokens(response: Any) -> int | None:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    value = getattr(usage, "total_tokens", None)
    if value is None and isinstance(usage, dict):
        value = usage.get("total_tokens")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


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
                    api_key = _openai_api_key()
                    base_url = _openai_base_url()
                    if hasattr(openai, "api_key") and api_key:
                        openai.api_key = api_key  # type: ignore[assignment]
                    if hasattr(openai, "OpenAI"):
                        try:  # pragma: no cover - optional runtime dependency
                            kwargs: dict[str, Any] = {}
                            if api_key:
                                kwargs["api_key"] = api_key
                            if base_url:
                                kwargs["base_url"] = base_url
                            self._openai_client = openai.OpenAI(**kwargs)  # type: ignore[attr-defined]
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


def _embed_max_tokens() -> int:
    raw = os.getenv("COMPAIR_OPENAI_EMBED_MAX_TOKENS", "8000")
    try:
        value = int(raw)
    except ValueError:
        return 8000
    return value if value > 0 else 8000


def _embed_encoding(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _token_count(text: str, model_name: str) -> int:
    return len(_embed_encoding(model_name).encode(text or " "))


def _split_text_for_embedding(text: str, model_name: str, max_tokens: int) -> list[tuple[str, int]]:
    encoding = _embed_encoding(model_name)
    tokens = encoding.encode(text or " ")
    if len(tokens) <= max_tokens:
        return [(text or " ", len(tokens))]
    segments: list[tuple[str, int]] = []
    for start in range(0, len(tokens), max_tokens):
        token_chunk = tokens[start : start + max_tokens]
        segment = encoding.decode(token_chunk)
        if segment.strip():
            segments.append((segment, len(token_chunk)))
    return segments or [(" ", 1)]


def _combine_embeddings(vectors: list[list[float]], weights: list[int]) -> list[float]:
    if not vectors:
        return []
    if len(vectors) == 1:
        return vectors[0]
    dim = len(vectors[0])
    total_weight = float(sum(max(1, weight) for weight in weights))
    combined = [0.0] * dim
    for vector, weight in zip(vectors, weights):
        scale = float(max(1, weight))
        for idx, value in enumerate(vector):
            combined[idx] += float(value) * scale
    combined = [value / total_weight for value in combined]
    norm = math.sqrt(sum(value * value for value in combined))
    if norm > 0:
        combined = [value / norm for value in combined]
    return combined


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


def create_embeddings(embedder: Embedder, texts: list[str], user=None) -> list[list[float]]:
    if not texts:
        return []
    if embedder.is_cloud and cloud_create_embeddings is not None:
        return cloud_create_embeddings(embedder._cloud_impl, texts, user=user)

    provider = getattr(embedder, "provider", "local")
    if provider == "openai" and openai is not None:
        client = getattr(embedder, "_openai_client", None)
        if client is None and hasattr(openai, "OpenAI"):
            api_key = _openai_api_key()
            base_url = _openai_base_url()
            try:  # pragma: no cover - optional client differences
                kwargs: dict[str, Any] = {}
                if api_key:
                    kwargs["api_key"] = api_key
                if base_url:
                    kwargs["base_url"] = base_url
                client = openai.OpenAI(**kwargs) if kwargs else openai.OpenAI()  # type: ignore[attr-defined]
            except TypeError:
                client = openai.OpenAI()
            embedder._openai_client = client  # type: ignore[attr-defined]
        try:
            vectors: list[list[float] | None] = [None] * len(texts)
            regular: list[tuple[int, str]] = []
            max_tokens = _embed_max_tokens()
            for idx, text in enumerate(texts):
                if _token_count(text, embedder.openai_embed_model) > max_tokens:
                    vectors[idx] = create_embedding(embedder, text, user=user)
                else:
                    regular.append((idx, text))
            if client is not None and hasattr(client, "embeddings"):
                if regular:
                    started_at = time.time()
                    response = client.embeddings.create(
                        model=embedder.openai_embed_model,
                        input=[text for _, text in regular],
                    )
                    ordered = sorted(
                        list(getattr(response, "data", []) or []),
                        key=lambda row: getattr(row, "index", 0),
                    )
                    for (item_idx, _), row in zip(regular, ordered):
                        vectors[item_idx] = getattr(row, "embedding", None)
                    log_event(
                        "openai_embedding_created",
                        model=embedder.openai_embed_model,
                        token_count=_usage_total_tokens(response),
                        batch_size=len(regular),
                        duration_sec=round(time.time() - started_at, 3),
                        user_id=getattr(user, "user_id", None) if user is not None else None,
                        created_at=_utc_now(),
                    )
                if all(isinstance(vector, list) for vector in vectors):
                    return vectors  # type: ignore[return-value]
            elif hasattr(openai, "Embedding"):
                if regular:
                    started_at = time.time()
                    response = openai.Embedding.create(  # type: ignore[attr-defined]
                        model=embedder.openai_embed_model,
                        input=[text for _, text in regular],
                    )
                    data = sorted(response["data"], key=lambda row: row.get("index", 0))  # type: ignore[index]
                    for (item_idx, _), row in zip(regular, data):
                        vectors[item_idx] = row.get("embedding")
                    log_event(
                        "openai_embedding_created",
                        model=embedder.openai_embed_model,
                        token_count=_usage_total_tokens(response),
                        batch_size=len(regular),
                        duration_sec=round(time.time() - started_at, 3),
                        user_id=getattr(user, "user_id", None) if user is not None else None,
                        created_at=_utc_now(),
                    )
                if all(isinstance(vector, list) for vector in vectors):
                    return vectors  # type: ignore[return-value]
        except Exception as exc:  # pragma: no cover - network/API failure
            log_event("openai_embedding_batch_failed", error=str(exc))

    return [create_embedding(embedder, text, user=user) for text in texts]


def _openai_embedding(embedder: Embedder, text: str) -> list[float] | None:
    if openai is None:
        return None
    client = getattr(embedder, "_openai_client", None)
    if client is None and hasattr(openai, "OpenAI"):
        api_key = _openai_api_key()
        base_url = _openai_base_url()
        try:  # pragma: no cover - optional client differences
            kwargs: dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            client = openai.OpenAI(**kwargs) if kwargs else openai.OpenAI()  # type: ignore[attr-defined]
        except TypeError:
            client = openai.OpenAI()
        embedder._openai_client = client  # type: ignore[attr-defined]

    try:
        started_at = time.time()
        token_count = _token_count(text, embedder.openai_embed_model)
        if token_count > _embed_max_tokens():
            parts = _split_text_for_embedding(text, embedder.openai_embed_model, _embed_max_tokens())
            if client is not None and hasattr(client, "embeddings"):
                response = client.embeddings.create(
                    model=embedder.openai_embed_model,
                    input=[part for part, _ in parts],
                )
                data = sorted(list(getattr(response, "data", []) or []), key=lambda row: getattr(row, "index", 0))
                vectors = [getattr(row, "embedding", None) for row in data]
            elif hasattr(openai, "Embedding"):
                response = openai.Embedding.create(  # type: ignore[attr-defined]
                    model=embedder.openai_embed_model,
                    input=[part for part, _ in parts],
                )
                data = sorted(response["data"], key=lambda row: row.get("index", 0))  # type: ignore[index]
                vectors = [row.get("embedding") for row in data]
            else:
                vectors = None
            if vectors and all(isinstance(vector, list) for vector in vectors):
                log_event(
                    "openai_embedding_created",
                    model=embedder.openai_embed_model,
                    token_count=_usage_total_tokens(response) or token_count,
                    split_input=True,
                    segment_count=len(parts),
                    duration_sec=round(time.time() - started_at, 3),
                    created_at=_utc_now(),
                )
                return _combine_embeddings(vectors, [weight for _, weight in parts])  # type: ignore[arg-type]
        if client is not None and hasattr(client, "embeddings"):
            response = client.embeddings.create(
                model=embedder.openai_embed_model,
                input=text,
            )
            data = getattr(response, "data", None)
            if data:
                vector = getattr(data[0], "embedding", None)
                if isinstance(vector, list):
                    log_event(
                        "openai_embedding_created",
                        model=embedder.openai_embed_model,
                        token_count=_usage_total_tokens(response) or token_count,
                        split_input=False,
                        duration_sec=round(time.time() - started_at, 3),
                        created_at=_utc_now(),
                    )
                    return vector
        elif hasattr(openai, "Embedding"):
            response = openai.Embedding.create(  # type: ignore[attr-defined]
                model=embedder.openai_embed_model,
                input=text,
            )
            vector = response["data"][0]["embedding"]  # type: ignore[index]
            if isinstance(vector, list):
                log_event(
                    "openai_embedding_created",
                    model=embedder.openai_embed_model,
                    token_count=_usage_total_tokens(response) or token_count,
                    split_input=False,
                    duration_sec=round(time.time() - started_at, 3),
                    created_at=_utc_now(),
                )
                return vector
    except Exception as exc:  # pragma: no cover - network/API failure
        log_event("openai_embedding_failed", error=str(exc))
    return None
