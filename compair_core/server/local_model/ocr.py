"""Minimal OCR endpoint leveraging pytesseract when available."""
from __future__ import annotations

import io
import os
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI(title="Compair Local OCR", version="0.1.0")

try:  # Optional dependency
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover - optional
    pytesseract = None  # type: ignore
    Image = None  # type: ignore

_OCR_FALLBACK = os.getenv("COMPAIR_LOCAL_OCR_FALLBACK", "text")  # text | none


def _extract_text(data: bytes) -> str:
    if pytesseract is None or Image is None:
        if _OCR_FALLBACK == "text":
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data.decode("latin-1", errors="ignore")
        return ""
    try:
        image = Image.open(io.BytesIO(data))
        return pytesseract.image_to_string(image)
    except Exception:
        return ""


@app.post("/ocr-file")
async def ocr_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    payload = await file.read()
    text = _extract_text(payload)
    if not text:
        raise HTTPException(status_code=501, detail="OCR not available or failed to extract text.")
    return {"extracted_text": text}

