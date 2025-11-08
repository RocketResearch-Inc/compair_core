"""Minimal OCR endpoint leveraging pytesseract when available."""
from __future__ import annotations

import io
import os
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI(title="Compair Local OCR", version="0.1.0")

try:  # Optional dependency
    import pytesseract  # type: ignore
    from pytesseract import TesseractNotFoundError  # type: ignore
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover - optional
    pytesseract = None  # type: ignore
    TesseractNotFoundError = OSError  # type: ignore
    Image = None  # type: ignore

try:  # Optional: text extraction for PDFs
    from pypdf import PdfReader  # type: ignore
except ImportError:  # pragma: no cover - optional
    PdfReader = None  # type: ignore

_OCR_FALLBACK = os.getenv("COMPAIR_LOCAL_OCR_FALLBACK", "text")  # text | none
_TESSERACT_AVAILABLE = False
if pytesseract is not None and Image is not None:
    try:  # pragma: no cover - runtime probe
        pytesseract.get_tesseract_version()
        _TESSERACT_AVAILABLE = True
    except (TesseractNotFoundError, OSError):
        _TESSERACT_AVAILABLE = False


def _is_pdf(data: bytes) -> bool:
    return data.startswith(b"%PDF")


def _extract_pdf_text(data: bytes) -> str:
    if PdfReader is None:
        return ""
    try:
        reader = PdfReader(io.BytesIO(data))
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text)
        return "\n".join(parts).strip()
    except Exception:
        return ""


def _fallback_text(data: bytes) -> str:
    if _OCR_FALLBACK == "text":
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("latin-1", errors="ignore")
    return ""


def _extract_text(data: bytes) -> tuple[str, bool]:
    provider_used = False
    if _is_pdf(data):
        if PdfReader is not None:
            provider_used = True
            text = _extract_pdf_text(data)
            if text:
                return text, provider_used
        if pytesseract is None or Image is None:
            text = _fallback_text(data)
            if text:
                provider_used = True
            return text, provider_used

    if pytesseract is not None and Image is not None and _TESSERACT_AVAILABLE:
        provider_used = True
        try:
            image = Image.open(io.BytesIO(data))
            text = pytesseract.image_to_string(image)
            if text:
                return text, provider_used
        except Exception:
            pass

    text = _fallback_text(data)
    if text:
        provider_used = True
    return text, provider_used


@app.post("/ocr-file")
async def ocr_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    payload = await file.read()
    text, provider_available = _extract_text(payload)
    if not provider_available:
        if pytesseract is not None and not _TESSERACT_AVAILABLE:
            detail = "Tesseract CLI is not installed or accessible. Install it (e.g., via your package manager) or run the container image, then retry."
        else:
            detail = "OCR extraction failed. Supported formats include PDFs (text-only) and common image types."
        raise HTTPException(
            status_code=501,
            detail=detail,
        )
    return {"extracted_text": text or "NONE"}
