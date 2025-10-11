"""OCR provider placeholder for the Core edition."""
from __future__ import annotations


class NoopOCR:
    def submit(self, **_: object) -> str:
        raise NotImplementedError("OCR is not available in the Core edition.")

    def status(self, task_id: str) -> dict[str, object]:
        return {"status": "unknown", "task_id": task_id}
