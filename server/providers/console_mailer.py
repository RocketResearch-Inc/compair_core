"""Console mailer used in Core builds to avoid delivering real email."""
from __future__ import annotations

from typing import Iterable


class ConsoleMailer:
    def send(self, subject: str, sender: str, receivers: Iterable[str], html: str) -> None:
        print(f"[MAIL] {subject} -> {list(receivers)}")
