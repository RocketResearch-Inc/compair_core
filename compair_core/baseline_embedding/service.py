"""Loopback-only ``baseline-embedding-http.v1`` service.

The service loads only a fully verified local snapshot. It never invokes a
download path and never logs request bodies or submitted text.
"""

from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import logging
import math
import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Protocol

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .cache import (
    BaselineModelCacheError,
    VerifiedBaselineModel,
    verify_baseline_model,
)
from .manifest import BaselineModelManifest, load_baseline_model_manifest

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9010
DEFAULT_THREADS = 8
DEFAULT_INFERENCE_BATCH_SIZE = 32
DEFAULT_MAX_REQUEST_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_TEXT_ITEMS = 256
DEFAULT_MAX_TEXT_BYTES = 1_000_000
DEFAULT_MAX_TOTAL_TEXT_BYTES = 4_000_000


class EmbeddingModel(Protocol):
    def embed(
        self,
        documents: Sequence[str],
        *,
        batch_size: int,
    ) -> Any: ...


ModelFactory = Callable[[VerifiedBaselineModel, int], EmbeddingModel]


class BaselineEmbeddingServiceError(RuntimeError):
    """A sanitized service failure that never includes submitted content."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _load_fastembed_model(
    verified: VerifiedBaselineModel,
    threads: int,
) -> EmbeddingModel:
    manifest = verified.manifest
    for package_name, expected_version in manifest.runtime_packages:
        try:
            installed_version = version(package_name)
        except PackageNotFoundError:
            raise BaselineEmbeddingServiceError(
                "baseline_embedding_runtime_unavailable"
            ) from None
        if installed_version != expected_version:
            raise BaselineEmbeddingServiceError(
                "baseline_embedding_runtime_version_mismatch"
            )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from fastembed import TextEmbedding

        return TextEmbedding(
            manifest.logical_model,
            specific_model_path=str(verified.snapshot_dir),
            threads=threads,
            local_files_only=True,
        )
    except Exception:  # noqa: BLE001 - hide runtime paths/details
        raise BaselineEmbeddingServiceError(
            "baseline_embedding_model_unavailable"
        ) from None


@dataclass(slots=True)
class BaselineEmbeddingRuntime:
    cache_root: str | None = field(default=None, repr=False)
    threads: int = DEFAULT_THREADS
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE
    manifest: BaselineModelManifest = field(
        default_factory=load_baseline_model_manifest
    )
    model_factory: ModelFactory = field(default=_load_fastembed_model, repr=False)
    _model: EmbeddingModel | None = field(default=None, init=False, repr=False)
    _load_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )
    _inference_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not 1 <= self.threads <= 64:
            raise BaselineEmbeddingServiceError("baseline_embedding_threads_invalid")
        if not 1 <= self.inference_batch_size <= DEFAULT_MAX_TEXT_ITEMS:
            raise BaselineEmbeddingServiceError("baseline_embedding_batch_size_invalid")

    def identity(self) -> dict[str, object]:
        return {
            "contract_version": self.manifest.contract_version,
            "provider": self.manifest.provider,
            "model": self.manifest.logical_model,
            "revision": self.manifest.revision,
            "dimension": self.manifest.dimension,
        }

    def ensure_ready(self) -> EmbeddingModel:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                verified = verify_baseline_model(
                    self.cache_root,
                    manifest=self.manifest,
                )
            except BaselineModelCacheError as exc:
                raise BaselineEmbeddingServiceError(exc.code) from None
            self._model = self.model_factory(verified, self.threads)
            return self._model

    def embed(self, texts: Sequence[str]) -> tuple[list[float], ...]:
        model = self.ensure_ready()
        try:
            with self._inference_lock:
                vectors = tuple(
                    model.embed(
                        texts,
                        batch_size=min(self.inference_batch_size, len(texts)),
                    )
                )
        except Exception:  # noqa: BLE001 - never expose model/source details
            raise BaselineEmbeddingServiceError(
                "baseline_embedding_inference_failed"
            ) from None
        if len(vectors) != len(texts):
            raise BaselineEmbeddingServiceError(
                "baseline_embedding_vector_count_mismatch"
            )
        output: list[list[float]] = []
        for raw_vector in vectors:
            values = np.asarray(raw_vector, dtype="<f4")
            if values.shape != (self.manifest.dimension,):
                raise BaselineEmbeddingServiceError(
                    "baseline_embedding_dimension_mismatch"
                )
            serialized = [float(value) for value in values]
            if not all(math.isfinite(value) for value in serialized):
                raise BaselineEmbeddingServiceError(
                    "baseline_embedding_vector_nonfinite"
                )
            output.append(serialized)
        return tuple(output)


def _error(status_code: int, code: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": code})


def _health_error(code: str) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"status": "unavailable", "reason": code},
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate_key")
        output[key] = value
    return output


async def _bounded_body(request: Request, maximum_bytes: int) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > maximum_bytes:
                raise BaselineEmbeddingServiceError(
                    "baseline_embedding_request_too_large"
                )
        except ValueError:
            raise BaselineEmbeddingServiceError(
                "baseline_embedding_request_invalid"
            ) from None
    received = bytearray()
    async for block in request.stream():
        received.extend(block)
        if len(received) > maximum_bytes:
            raise BaselineEmbeddingServiceError("baseline_embedding_request_too_large")
    return bytes(received)


def _parse_request(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("non_finite")
            ),
        )
    except (UnicodeDecodeError, ValueError, TypeError):
        raise BaselineEmbeddingServiceError(
            "baseline_embedding_request_malformed"
        ) from None
    if not isinstance(value, dict):
        raise BaselineEmbeddingServiceError("baseline_embedding_request_invalid")
    return value


def create_app(
    runtime: BaselineEmbeddingRuntime | None = None,
    *,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    max_text_items: int = DEFAULT_MAX_TEXT_ITEMS,
    max_text_bytes: int = DEFAULT_MAX_TEXT_BYTES,
    max_total_text_bytes: int = DEFAULT_MAX_TOTAL_TEXT_BYTES,
) -> FastAPI:
    """Create the two-route service app without starting background work."""

    if (
        max_request_bytes < 1
        or not 1 <= max_text_items <= DEFAULT_MAX_TEXT_ITEMS
        or max_text_bytes < 1
        or max_total_text_bytes < 1
    ):
        raise BaselineEmbeddingServiceError("baseline_embedding_limits_invalid")
    provider = runtime or BaselineEmbeddingRuntime()
    app = FastAPI(
        title="Compair baseline embedding service",
        version=provider.manifest.contract_version,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.get("/v1/health")
    def health() -> JSONResponse:
        try:
            provider.ensure_ready()
        except BaselineEmbeddingServiceError as exc:
            return _health_error(exc.code)
        return JSONResponse(content={"status": "ok", **provider.identity()})

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> JSONResponse:
        try:
            provider.ensure_ready()
        except BaselineEmbeddingServiceError as exc:
            return _error(503, exc.code)
        content_type = request.headers.get("content-type", "")
        media_type, _, charset = content_type.partition(";")
        if media_type.strip().lower() != "application/json" or (
            charset and charset.strip().lower().replace(" ", "") != "charset=utf-8"
        ):
            return _error(415, "baseline_embedding_content_type_invalid")
        try:
            payload = _parse_request(await _bounded_body(request, max_request_bytes))
            identity = provider.identity()
            for key, expected in identity.items():
                if payload.get(key) != expected:
                    raise BaselineEmbeddingServiceError(
                        f"baseline_embedding_{key}_mismatch"
                    )
            if set(payload) != {*identity, "texts"}:
                raise BaselineEmbeddingServiceError(
                    "baseline_embedding_request_invalid"
                )
            texts = payload["texts"]
            if not isinstance(texts, list) or not 1 <= len(texts) <= max_text_items:
                raise BaselineEmbeddingServiceError(
                    "baseline_embedding_text_count_invalid"
                )
            total_bytes = 0
            for text in texts:
                if not isinstance(text, str):
                    raise BaselineEmbeddingServiceError(
                        "baseline_embedding_text_invalid"
                    )
                size = len(text.encode("utf-8"))
                if size > max_text_bytes:
                    raise BaselineEmbeddingServiceError(
                        "baseline_embedding_text_too_large"
                    )
                total_bytes += size
            if total_bytes > max_total_text_bytes:
                raise BaselineEmbeddingServiceError(
                    "baseline_embedding_total_text_too_large"
                )
            vectors = await asyncio.to_thread(provider.embed, tuple(texts))
        except BaselineEmbeddingServiceError as exc:
            status = 413 if "too_large" in exc.code else 400
            if exc.code in {
                "baseline_embedding_inference_failed",
                "baseline_embedding_vector_count_mismatch",
                "baseline_embedding_dimension_mismatch",
                "baseline_embedding_vector_nonfinite",
            }:
                status = 503
            return _error(status, exc.code)
        return JSONResponse(content={**provider.identity(), "vectors": vectors})

    return app


def _loopback_host(value: str) -> str:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "host must be a literal loopback address"
        ) from None
    if not address.is_loopback:
        raise argparse.ArgumentTypeError("host must be a literal loopback address")
    return value


def _bounded_integer(minimum: int, maximum: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError("value must be an integer") from None
        if not minimum <= number <= maximum:
            raise argparse.ArgumentTypeError(
                f"value must be between {minimum} and {maximum}"
            )
        return number

    return parse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compair-core-embedding-service",
        description="Serve the verified baseline-v1 model on loopback only.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, type=_loopback_host)
    parser.add_argument("--port", default=DEFAULT_PORT, type=_bounded_integer(1, 65535))
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--threads", default=DEFAULT_THREADS, type=_bounded_integer(1, 64)
    )
    parser.add_argument(
        "--batch-size",
        default=DEFAULT_INFERENCE_BATCH_SIZE,
        type=_bounded_integer(1, DEFAULT_MAX_TEXT_ITEMS),
    )
    parser.add_argument(
        "--max-request-bytes",
        default=DEFAULT_MAX_REQUEST_BYTES,
        type=_bounded_integer(1, 64 * 1024 * 1024),
    )
    parser.add_argument(
        "--max-text-items",
        default=DEFAULT_MAX_TEXT_ITEMS,
        type=_bounded_integer(1, DEFAULT_MAX_TEXT_ITEMS),
    )
    parser.add_argument(
        "--max-text-bytes",
        default=DEFAULT_MAX_TEXT_BYTES,
        type=_bounded_integer(1, 8 * 1024 * 1024),
    )
    parser.add_argument(
        "--max-total-text-bytes",
        default=DEFAULT_MAX_TOTAL_TEXT_BYTES,
        type=_bounded_integer(1, 64 * 1024 * 1024),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for logger_name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    runtime = BaselineEmbeddingRuntime(
        cache_root=args.cache_dir,
        threads=args.threads,
        inference_batch_size=args.batch_size,
    )
    app = create_app(
        runtime,
        max_request_bytes=args.max_request_bytes,
        max_text_items=args.max_text_items,
        max_text_bytes=args.max_text_bytes,
        max_total_text_bytes=args.max_total_text_bytes,
    )
    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        access_log=False,
        server_header=False,
        log_level="warning",
        timeout_graceful_shutdown=10,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - installed entry point
    raise SystemExit(main())


__all__ = [
    "BaselineEmbeddingRuntime",
    "BaselineEmbeddingServiceError",
    "create_app",
    "main",
]
