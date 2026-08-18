"""Installed, privacy-safe verification command for baseline generation."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from ..server.settings import Settings
from .ollama import OllamaGenerationReadiness, verify_ollama_generation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compair-core-generation",
        description="Verify the configured baseline generation provider.",
    )
    parser.add_argument(
        "command",
        choices=("verify",),
        help="check configured Ollama runtime/model identity",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="run one private-data-free strict structured-output probe",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify_ollama_generation(Settings(), probe=bool(args.probe))
    except Exception:  # noqa: BLE001 - command output must remain sanitized JSON
        result = OllamaGenerationReadiness(
            status="provider_unconfigured",
            ready=False,
            provider=None,
            model=None,
            expected_digest=None,
            runtime_version=None,
            identity_fingerprint=None,
            probe_performed=bool(args.probe),
            probe_outcome=None,
        )
    print(
        json.dumps(
            result.as_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if result.ready else 3


if __name__ == "__main__":  # pragma: no cover - installed entry point
    raise SystemExit(main())
