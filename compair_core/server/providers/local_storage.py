"""Local filesystem storage provider used in the Core edition."""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO


class LocalStorage:
    def __init__(self, base_dir: str = "/data/uploads", base_url: str = "/uploads") -> None:
        self.base_dir = Path(base_dir)
        self.base_url = base_url.rstrip("/") or "/uploads"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def put_file(self, key: str, fileobj: BinaryIO, content_type: str) -> str:
        destination = self.base_dir / key
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as dest:
            dest.write(fileobj.read())
        return self.build_url(key)

    def get_file(self, key: str) -> tuple[BinaryIO, str]:
        path = self.base_dir / key
        if not path.exists():
            raise FileNotFoundError(key)
        return path.open("rb"), "application/octet-stream"

    def build_url(self, key: str) -> str:
        return f"{self.base_url}/{key}".replace("//", "/")
