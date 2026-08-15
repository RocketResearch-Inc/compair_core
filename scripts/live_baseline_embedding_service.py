"""Loopback-only FastEmbed service for live ``baseline_v1`` validation.

This is an operator helper, not a Core runtime provider.  It adapts the
existing self-hosted FastEmbed/BGE implementation to
``baseline-embedding-http.v1`` and requires a pre-downloaded immutable model
snapshot.  It never downloads weights itself.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastembed import TextEmbedding

CONTRACT_VERSION = "baseline-embedding-http.v1"
PROVIDER = "baseline_http_v1"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIMENSION = 384
MAX_BATCH_SIZE = 256
MAX_TEXT_CHARACTERS = 1_000_000
SNAPSHOT_FILE_SHA256 = {
    "config.json": "13582bcf2effc85b7bf3d3f5532e686bc1c9ce86bb009d10f0ec33cbe92299dd",
    "model_optimized.onnx": (
        "51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431"
    ),
    "special_tokens_map.json": (
        "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a"
    ),
    "tokenizer.json": (
        "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"
    ),
    "tokenizer_config.json": (
        "0b29c7bfc889e53b36d9dd3e686dd4300f6525110eaa98c76a5dafceb2029f53"
    ),
}

MODEL = os.getenv("COMPAIR_BASELINE_EMBEDDING_MODEL", DEFAULT_MODEL).strip()
REVISION = os.getenv("COMPAIR_BASELINE_EMBEDDING_REVISION", "").strip()
SNAPSHOT_DIR = Path(
    os.getenv("COMPAIR_BASELINE_EMBEDDING_SNAPSHOT_DIR", "")
).expanduser()
DIMENSION = int(
    os.getenv("COMPAIR_BASELINE_EMBEDDING_DIMENSION", str(DEFAULT_DIMENSION))
)
THREADS = int(os.getenv("COMPAIR_BASELINE_EMBEDDING_THREADS", "8"))
INFERENCE_BATCH_SIZE = int(os.getenv("COMPAIR_BASELINE_EMBEDDING_BATCH_SIZE", "32"))

app = FastAPI(
    title="Compair Baseline Embedding Live Validation Adapter",
    version="baseline-embedding-http.v1",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
_lock = threading.Lock()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_configuration() -> None:
    if MODEL != DEFAULT_MODEL:
        raise RuntimeError("baseline_embedding_model_not_permitted")
    if not REVISION:
        raise RuntimeError("baseline_embedding_revision_missing")
    if DIMENSION != DEFAULT_DIMENSION:
        raise RuntimeError("baseline_embedding_dimension_not_permitted")
    if not 1 <= THREADS <= 64:
        raise RuntimeError("baseline_embedding_threads_invalid")
    if not 1 <= INFERENCE_BATCH_SIZE <= MAX_BATCH_SIZE:
        raise RuntimeError("baseline_embedding_batch_size_invalid")
    if not SNAPSHOT_DIR.is_absolute():
        raise RuntimeError("baseline_embedding_snapshot_must_be_absolute")
    try:
        resolved_snapshot = SNAPSHOT_DIR.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("baseline_embedding_snapshot_unavailable") from exc
    if resolved_snapshot.name != REVISION:
        raise RuntimeError("baseline_embedding_snapshot_revision_mismatch")
    for required_file, expected_hash in SNAPSHOT_FILE_SHA256.items():
        path = resolved_snapshot / required_file
        if not path.is_file():
            raise RuntimeError("baseline_embedding_snapshot_incomplete")
        if _file_sha256(path) != expected_hash:
            raise RuntimeError("baseline_embedding_snapshot_hash_mismatch")


_validate_configuration()


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    return TextEmbedding(
        MODEL,
        specific_model_path=str(SNAPSHOT_DIR.resolve(strict=True)),
        threads=THREADS,
        local_files_only=True,
    )


def _identity() -> dict[str, object]:
    return {
        "contract_version": CONTRACT_VERSION,
        "provider": PROVIDER,
        "model": MODEL,
        "revision": REVISION,
        "dimension": DIMENSION,
    }


def _error(status_code: int, code: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": code})


def _validate_request(payload: Any) -> tuple[str, ...] | None:
    if not isinstance(payload, dict):
        return None
    identity = _identity()
    for field in (
        "contract_version",
        "provider",
        "model",
        "revision",
        "dimension",
    ):
        if payload.get(field) != identity[field]:
            return None
    texts = payload.get("texts")
    if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_BATCH_SIZE:
        return None
    if any(
        not isinstance(value, str) or len(value) > MAX_TEXT_CHARACTERS
        for value in texts
    ):
        return None
    return tuple(texts)


@app.get("/v1/health")
def health() -> dict[str, object]:
    # Readiness includes successful offline model initialization.
    _model()
    return {"status": "ok", **_identity()}


@app.post("/v1/embeddings")
async def embeddings(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001 - the response must never echo request data
        return _error(400, "embedding_request_malformed")
    texts = _validate_request(payload)
    if texts is None:
        return _error(400, "embedding_request_invalid")

    try:
        with _lock:
            vectors = tuple(
                _model().embed(
                    texts,
                    batch_size=min(INFERENCE_BATCH_SIZE, len(texts)),
                )
            )
    except Exception:  # noqa: BLE001 - avoid provider details or source text
        return _error(503, "embedding_model_unavailable")
    if len(vectors) != len(texts):
        return _error(500, "embedding_vector_count_mismatch")

    output: list[list[float]] = []
    for vector in vectors:
        values = np.asarray(vector, dtype=np.float32)
        if values.ndim != 1 or len(values) != DIMENSION:
            return _error(500, "embedding_dimension_mismatch")
        serialized = [float(value) for value in values]
        if not all(math.isfinite(value) for value in serialized):
            return _error(500, "embedding_contains_non_finite_value")
        output.append(serialized)
    return JSONResponse(content={**_identity(), "vectors": output})
