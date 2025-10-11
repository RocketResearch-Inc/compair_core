"""Minimal FastAPI application serving local embedding and generation endpoints."""
from __future__ import annotations

import hashlib
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Compair Local Model", version="0.1.0")

EMBED_DIMENSION = 384


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
    system: str | None = None
    prompt: str
    verbosity: str | None = None


class GenerateResponse(BaseModel):
    text: str


@app.post("/embed", response_model=EmbedResponse)
def embed(request: EmbedRequest) -> EmbedResponse:
    return EmbedResponse(embedding=_hash_embedding(request.text))


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    prompt = request.prompt.strip()
    if not prompt:
        return GenerateResponse(text="NONE")

    first_sentence = prompt.split("\n", 1)[0][:200]
    verbosity = request.verbosity or "default"
    return GenerateResponse(
        text=f"[local-{verbosity}] Key takeaway: {first_sentence}"
    )
