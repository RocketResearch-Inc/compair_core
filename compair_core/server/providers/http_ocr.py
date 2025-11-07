from __future__ import annotations

import threading
import uuid
from typing import Dict

import requests


class HTTPOCR:
    """Simple OCR provider that forwards uploads to an HTTP endpoint."""

    def __init__(self, endpoint: str, *, timeout: float = 30.0) -> None:
        self.endpoint = endpoint
        self.timeout = timeout
        self._lock = threading.Lock()
        self._results: Dict[str, dict[str, object]] = {}

    def submit(
        self,
        *,
        user_id: str,
        filename: str,
        data: bytes,
        document_id: str | None,
    ) -> str:
        files = {"file": (filename, data)}
        response = requests.post(self.endpoint, files=files, timeout=self.timeout)
        response.raise_for_status()
        payload: dict[str, object]
        try:
            payload = response.json()
        except ValueError:
            payload = {"extracted_text": response.text}

        extracted = (
            payload.get("extracted_text")
            or payload.get("text")
            or payload.get("content")
        )
        if not extracted:
            raise RuntimeError("OCR endpoint returned an empty response.")

        task_id = str(uuid.uuid4())
        result = {
            "status": "completed",
            "extracted_text": extracted,
            "document_id": document_id,
            "user_id": user_id,
        }
        with self._lock:
            self._results[task_id] = result
        return task_id

    def status(self, task_id: str) -> dict[str, object]:
        with self._lock:
            result = self._results.get(task_id)
        if result is None:
            return {"status": "unknown", "task_id": task_id}
        return result
