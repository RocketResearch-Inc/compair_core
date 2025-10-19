"""Minimal FastAPI application serving local embedding and generation endpoints."""
from __future__ import annotations

import hashlib
import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Compair Local Model", version="0.1.0")

_DEFAULT_DIM = 384
_DIM_ENV = (
    os.getenv("COMPAIR_EMBEDDING_DIM")
    or os.getenv("COMPAIR_EMBEDDING_DIMENSION")
    or os.getenv("COMPAIR_LOCAL_EMBED_DIM")
    or str(_DEFAULT_DIM)
)
try:
    EMBED_DIMENSION = int(_DIM_ENV)
except ValueError:  # pragma: no cover - invalid configuration
    EMBED_DIMENSION = _DEFAULT_DIM


def _hash_embedding(text: str, dimension: int = EMBED_DIMENSION) -> List[float]:
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


class EmbedRequest(BaseModel):
    text: str


class EmbedResponse(BaseModel):
    embedding: List[float]


class GenerateRequest(BaseModel):
    # Legacy format used by the CLI shim
    system: str | None = None
    prompt: str | None = None
    verbosity: str | None = None

    # Core API payload (document + references)
    document: str | None = None
    references: List[str] | None = None
    length_instruction: str | None = None


class GenerateResponse(BaseModel):
    feedback: str


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    return EmbedResponse(embedding=_hash_embedding(request.text))


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    # Determine the main text input (document or prompt)
    text_input = request.document or request.prompt or ""
    text_input = text_input.strip()

    if not text_input:
        return GenerateResponse(feedback="NONE")

    first_sentence = text_input.split("\n", 1)[0][:200]
    verbosity = request.length_instruction or request.verbosity or "brief response"
    ref_snippet = ""
    if request.references:
        top_ref = (request.references[0] or "").strip()
        if top_ref:
            ref_snippet = f" Reference: {top_ref[:160]}"

    feedback = f"[local-feedback] {verbosity}: {first_sentence}{ref_snippet}".strip()
    return GenerateResponse(feedback=feedback or "NONE")
