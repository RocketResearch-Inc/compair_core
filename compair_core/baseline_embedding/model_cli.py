"""Installed operator commands for the frozen ``baseline_v1`` model."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .cache import (
    BaselineModelCacheError,
    cleanup_incomplete_fetches,
    fetch_baseline_model,
    verify_baseline_model,
)
from .manifest import (
    PROFILE,
    BaselineModelManifestError,
    load_baseline_model_manifest,
)


def _emit(value: dict[str, object], *, stream: object = sys.stdout) -> None:
    print(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
        file=stream,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="compair-core-models",
        description="Acquire and verify pinned Compair Core model artifacts.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch", "verify"):
        command = commands.add_parser(name)
        command.add_argument("profile", choices=(PROFILE,))
        command.add_argument(
            "--cache-dir",
            help="Private model-cache root (defaults to COMPAIR_BASELINE_MODEL_CACHE).",
        )
    clean = commands.add_parser("clean")
    clean.add_argument("profile", choices=(PROFILE,))
    clean.add_argument(
        "--cache-dir",
        help="Private model-cache root (defaults to COMPAIR_BASELINE_MODEL_CACHE).",
    )
    clean.add_argument(
        "--incomplete",
        action="store_true",
        required=True,
        help="Remove interrupted staging directories only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = load_baseline_model_manifest()
        if args.command == "fetch":
            _emit(
                {
                    "event": "baseline_model_fetch_plan",
                    **manifest.safe_summary(),
                    "files": [
                        {
                            "path": artifact.path,
                            "size": artifact.size,
                            "sha256": artifact.sha256,
                        }
                        for artifact in manifest.files
                    ],
                },
                stream=sys.stderr,
            )
            _emit(
                fetch_baseline_model(
                    args.cache_dir,
                    manifest=manifest,
                ).safe_summary()
            )
            return 0
        if args.command == "verify":
            _emit(
                verify_baseline_model(
                    args.cache_dir,
                    manifest=manifest,
                ).safe_summary()
            )
            return 0
        if args.command == "clean":
            _emit(
                {
                    "status": "cleaned",
                    "profile": manifest.profile,
                    "incomplete_directories_removed": cleanup_incomplete_fetches(
                        args.cache_dir,
                        manifest=manifest,
                    ),
                    "manifest_fingerprint": manifest.manifest_fingerprint,
                }
            )
            return 0
    except (BaselineModelCacheError, BaselineModelManifestError) as exc:
        code = getattr(exc, "code", str(exc))
        _emit({"status": "error", "reason": code}, stream=sys.stderr)
        return 3
    return 2  # pragma: no cover - argparse exhausts supported commands


if __name__ == "__main__":  # pragma: no cover - installed entry point
    raise SystemExit(main())
