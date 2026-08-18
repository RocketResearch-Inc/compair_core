from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from compair_core import api_cli
from compair_core.server import app as app_module
from compair_core.server.settings import Settings


def test_api_entrypoint_defaults_loopback_and_disables_access_logs(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def run(app, **kwargs):
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))
    assert api_cli.main([]) == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8000
    assert captured["access_log"] is False
    assert captured["proxy_headers"] is False
    assert captured["log_level"] == "warning"
    assert captured["timeout_graceful_shutdown"] == 10


def test_api_entrypoint_non_loopback_requires_explicit_opt_in(
    monkeypatch,
) -> None:
    invoked = False

    def run(*_args, **_kwargs):
        nonlocal invoked
        invoked = True

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))
    assert api_cli.main(["--host", "0.0.0.0"]) == 2
    assert invoked is False
    assert api_cli.main(["--host", "0.0.0.0", "--allow-non-loopback"]) == 0
    assert invoked is True


def test_app_factory_starts_optional_telemetry_only_at_asgi_startup(
    monkeypatch,
) -> None:
    calls: list[Settings] = []
    monkeypatch.setattr(
        app_module,
        "start_usage_telemetry",
        lambda settings: calls.append(settings),
    )
    settings = Settings(telemetry_enabled=True)
    app = app_module.create_app(settings)
    assert calls == []
    startup = app.router.on_startup[-1]
    startup()
    assert calls == [settings]


def test_doctor_import_does_not_import_legacy_application_or_run_startup(
    tmp_path,
) -> None:
    root = Path(__file__).parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import compair_core.doctor; "
                "print(int('compair_core.compair' in sys.modules))"
            ),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(root),
            "COMPAIR_DB_DIR": str(tmp_path / "database"),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout == "0\n"
    assert completed.stderr == ""
    assert not (tmp_path / "database").exists()
