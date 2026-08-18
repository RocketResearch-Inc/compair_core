"""Stable installed launcher for the Core API."""

from __future__ import annotations

import argparse
import ipaddress
import logging
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from .db import engine
from .runtime_config import RuntimeConfigurationError, validate_runtime_configuration
from .server.settings import Settings

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
_LOGGER = logging.getLogger("compair_core.api_cli")


def _port(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("port must be an integer") from None
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return parsed


def _loopback(value: str) -> bool:
    if value.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compair-core-api",
        description="Start the Compair Core API with privacy-safe defaults.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT, type=_port)
    parser.add_argument(
        "--allow-non-loopback",
        action="store_true",
        help=(
            "explicitly permit a non-loopback bind; deploy only behind the "
            "documented TLS reverse-proxy policy"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    for logger_name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    if not _loopback(args.host) and not args.allow_non_loopback:
        _LOGGER.error("core_api event=start_failed reason=non_loopback_opt_in_required")
        return 2
    try:
        selected = Settings()
        validate_runtime_configuration(selected, database_url=engine.url)
        # Import after argument/config validation. The existing application
        # factory owns normal schema startup and provider wiring.
        from .server.app import create_app

        app = create_app(selected)
        import uvicorn

        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=1,
            access_log=False,
            proxy_headers=False,
            server_header=False,
            log_level="warning",
            timeout_graceful_shutdown=10,
        )
        return 0
    except RuntimeConfigurationError as exc:
        _LOGGER.error("core_api event=start_failed reason=%s", exc.code)
        return 2
    except ValidationError:
        _LOGGER.error("core_api event=start_failed reason=configuration_invalid")
        return 2
    except KeyboardInterrupt:
        return 0
    except Exception:  # noqa: BLE001 - startup output must remain sanitized
        _LOGGER.error("core_api event=start_failed reason=api_startup_failed")
        return 3


if __name__ == "__main__":  # pragma: no cover - installed entry point
    raise SystemExit(main())


__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "main"]
