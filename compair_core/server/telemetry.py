"""Opt-in anonymous usage telemetry for self-hosted Core."""
from __future__ import annotations

import hashlib
import json
import secrets
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import platform
import requests

from .settings import Settings

_HEARTBEAT_INTERVAL = timedelta(days=1)


def start_usage_telemetry(settings: Settings) -> None:
    if not settings.telemetry_enabled:
        return

    worker = threading.Thread(
        target=_maybe_send_daily_heartbeat,
        args=(settings,),
        name="compair-core-telemetry",
        daemon=True,
    )
    worker.start()


def _maybe_send_daily_heartbeat(settings: Settings) -> None:
    state = _load_state()
    install_id = (
        (settings.telemetry_install_id or "").strip()
        or (state.get("install_id") or "").strip()
        or secrets.token_hex(16)
    )
    state["install_id"] = install_id

    last_raw = str(state.get("last_heartbeat_at") or "").strip()
    if last_raw:
        try:
            last = datetime.fromisoformat(last_raw)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - last < _HEARTBEAT_INTERVAL:
                _write_state(state)
                return
        except Exception:
            pass

    now = datetime.now(timezone.utc)
    payload = {
        "client": "core",
        "events": [
            {
                "install_id": install_id,
                "client_event_id": _heartbeat_event_id(install_id, now),
                "event": "active_day",
                "source": "core",
                "kind": "usage",
                "app_version": settings.version,
                "os": platform.system().lower(),
                "arch": platform.machine().lower(),
                "ts": int(now.timestamp()),
                "payload": {
                    "edition": settings.edition,
                },
            }
        ],
    }
    url = settings.telemetry_base_url.rstrip("/") + "/client-metrics/anonymous"
    try:
        response = requests.post(
            url,
            json=payload,
            timeout=5,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "compair-core",
                "X-Compair-Client": "core",
            },
        )
        if response.status_code >= 300:
            return
    except Exception:
        return

    state["last_heartbeat_at"] = now.isoformat()
    _write_state(state)


def _state_path() -> Path:
    return Path.home() / ".compair-core" / "telemetry.json"


def _load_state() -> dict[str, Any]:
    path = _state_path()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(state: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _heartbeat_event_id(install_id: str, now: datetime) -> str:
    digest = hashlib.sha256(
        f"{install_id}|core|active_day|{now.astimezone(timezone.utc).date().isoformat()}".encode("utf-8")
    ).hexdigest()
    return digest[:32]
