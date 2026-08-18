"""Secure first-time initialization of the baseline-run secrets fragment."""

from __future__ import annotations

import argparse
import errno
import json
import os
import secrets
import stat
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .run_keyring import (
    RUN_KEYRING_VERSION,
    GeneratedRunKeyring,
    RunKeyringGenerationError,
    generate_run_keyring,
)

CONFIG_INIT_RESULT_SCHEMA_VERSION = "baseline-config-init-result.v1"
CONFIG_INIT_FILENAME = "baseline.env"
CONFIG_INIT_ENVIRONMENT_VARIABLE = "COMPAIR_BASELINE_RUN_ENCRYPTION_KEYRING"

EXIT_SUCCESS = 0
EXIT_USAGE_OR_PATH = 2
EXIT_DESTINATION_EXISTS = 3
EXIT_INSECURE_PATH = 4
EXIT_GENERATION = 5
EXIT_PUBLICATION = 6
EXIT_INTERNAL = 7

_UMASK_LOCK = threading.Lock()
_UNSUPPORTED_DIRECTORY_FSYNC = frozenset(
    value
    for value in (
        errno.EINVAL,
        getattr(errno, "ENOTSUP", None),
        getattr(errno, "EOPNOTSUPP", None),
    )
    if value is not None
)


class ConfigInitError(RuntimeError):
    """Sanitized first-time initialization failure."""

    def __init__(self, code: str, exit_code: int) -> None:
        self.code = code
        self.exit_code = exit_code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ConfigInitResult:
    """Privacy-safe result for a successfully published secrets fragment."""

    active_key_id: str
    destination_classification: str
    timestamp: datetime

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": CONFIG_INIT_RESULT_SCHEMA_VERSION,
            "created": True,
            "keyring_schema_version": RUN_KEYRING_VERSION,
            "active_key_id": self.active_key_id,
            "key_count": 1,
            "file_mode": "0600",
            "destination": self.destination_classification,
            "timestamp": self.timestamp.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        }


def _failure_payload(code: str) -> dict[str, object]:
    return {
        "schema_version": CONFIG_INIT_RESULT_SCHEMA_VERSION,
        "created": False,
        "reason_code": code,
    }


def default_config_path(
    *,
    environ: Mapping[str, str] | None = None,
    home_directory: Path | None = None,
) -> Path:
    """Return the supported default without touching the filesystem."""

    values = os.environ if environ is None else environ
    configured = values.get("XDG_CONFIG_HOME", "")
    if configured:
        root = Path(configured)
        if not root.is_absolute():
            raise ConfigInitError("xdg_config_home_not_absolute", EXIT_USAGE_OR_PATH)
    else:
        root = (Path.home() if home_directory is None else home_directory) / ".config"
    return root / "compair-core" / CONFIG_INIT_FILENAME


def _validate_output_path(path: Path) -> None:
    rendered = os.fspath(path)
    if (
        os.name != "posix"
        or not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
    ):
        raise ConfigInitError("platform_security_unsupported", EXIT_INSECURE_PATH)
    if (
        not rendered
        or "\x00" in rendered
        or not path.is_absolute()
        or rendered.startswith("//")
        or path.name in {"", ".", ".."}
        or any(part in {".", ".."} for part in path.parts[1:])
    ):
        raise ConfigInitError("invalid_output_path", EXIT_USAGE_OR_PATH)


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _path_open_error(error: OSError) -> ConfigInitError:
    if error.errno in {errno.EACCES, errno.EPERM}:
        return ConfigInitError("insecure_parent_permissions", EXIT_INSECURE_PATH)
    if error.errno in {errno.ELOOP, errno.ENOTDIR}:
        return ConfigInitError("parent_symlink_rejected", EXIT_INSECURE_PATH)
    return ConfigInitError("invalid_output_path", EXIT_USAGE_OR_PATH)


def _open_private_parent(path: Path) -> int:
    flags = _directory_flags()
    try:
        current = os.open(os.sep, flags)
    except OSError:
        raise ConfigInitError("insecure_output_path", EXIT_INSECURE_PATH) from None
    try:
        for component in path.parent.parts[1:]:
            created = False
            try:
                following = os.open(component, flags, dir_fd=current)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o700, dir_fd=current)
                    created = True
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise _path_open_error(exc) from None
                try:
                    following = os.open(component, flags, dir_fd=current)
                except OSError as exc:
                    raise _path_open_error(exc) from None
            except OSError as exc:
                raise _path_open_error(exc) from None
            if created:
                try:
                    os.fchmod(following, 0o700)
                except OSError:
                    os.close(following)
                    raise ConfigInitError(
                        "private_parent_creation_failed",
                        EXIT_INSECURE_PATH,
                    ) from None
            os.close(current)
            current = following
        info = os.fstat(current)
        mode = stat.S_IMODE(info.st_mode)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or mode & 0o022
        ):
            raise ConfigInitError(
                "insecure_parent_permissions",
                EXIT_INSECURE_PATH,
            )
        return current
    except Exception:
        os.close(current)
        raise


def _classify_existing_destination(parent_fd: int, name: str) -> None:
    try:
        info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError:
        raise ConfigInitError("insecure_output_path", EXIT_INSECURE_PATH) from None
    if stat.S_ISLNK(info.st_mode):
        raise ConfigInitError("destination_symlink_rejected", EXIT_INSECURE_PATH)
    raise ConfigInitError("destination_already_exists", EXIT_DESTINATION_EXISTS)


def _fsync_directory(directory_fd: int) -> None:
    try:
        os.fsync(directory_fd)
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED_DIRECTORY_FSYNC:
            raise


def _write_all(file_fd: int, content: bytes) -> None:
    offset = 0
    while offset < len(content):
        written = os.write(file_fd, content[offset:])
        if written <= 0:
            raise OSError(errno.EIO, "write_failed")
        offset += written


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _unlink_owned(
    parent_fd: int,
    name: str | None,
    owned: os.stat_result | None,
) -> None:
    if name is None or owned is None:
        return
    try:
        observed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if _same_inode(observed, owned):
            os.unlink(name, dir_fd=parent_fd)
    except FileNotFoundError:
        return
    except OSError:
        return


def _render_fragment(generated: GeneratedRunKeyring) -> bytes:
    if "'" in generated.serialized or "\n" in generated.serialized:
        raise ConfigInitError("serialization_failed", EXIT_GENERATION)
    return (
        f"{CONFIG_INIT_ENVIRONMENT_VARIABLE}='{generated.serialized}'\n"
    ).encode()


def _publish_exclusive(
    parent_fd: int,
    destination_name: str,
    content: bytes,
    *,
    temporary_name_factory: Callable[[], str],
    fault_injector: Callable[[str], None] | None,
) -> None:
    temporary_name: str | None = None
    temporary_info: os.stat_result | None = None
    file_fd: int | None = None
    destination_linked = False
    try:
        file_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        for _attempt in range(16):
            token = temporary_name_factory()
            if (
                not isinstance(token, str)
                or not token
                or any(character not in "0123456789abcdef" for character in token)
            ):
                raise ConfigInitError("atomic_publication_failed", EXIT_PUBLICATION)
            candidate = f".{CONFIG_INIT_FILENAME}.tmp.{token}"
            try:
                file_fd = os.open(candidate, file_flags, 0o600, dir_fd=parent_fd)
                temporary_name = candidate
                break
            except FileExistsError:
                continue
        if file_fd is None or temporary_name is None:
            raise ConfigInitError("atomic_publication_failed", EXIT_PUBLICATION)
        os.fchmod(file_fd, 0o600)
        temporary_info = os.fstat(file_fd)
        if fault_injector is not None:
            fault_injector("before_temporary_write")
        _write_all(file_fd, content)
        if fault_injector is not None:
            fault_injector("after_temporary_write")
        os.fsync(file_fd)
        if stat.S_IMODE(os.fstat(file_fd).st_mode) != 0o600:
            raise OSError(errno.EPERM, "private_mode_failed")
        os.close(file_fd)
        file_fd = None
        try:
            os.link(
                temporary_name,
                destination_name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            _classify_existing_destination(parent_fd, destination_name)
            raise AssertionError("unreachable")
        destination_linked = True
        _fsync_directory(parent_fd)
        os.unlink(temporary_name, dir_fd=parent_fd)
        temporary_name = None
        _fsync_directory(parent_fd)
        final = os.stat(
            destination_name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISREG(final.st_mode) or stat.S_IMODE(final.st_mode) != 0o600:
            raise OSError(errno.EPERM, "published_mode_invalid")
    except ConfigInitError:
        if file_fd is not None:
            os.close(file_fd)
        if destination_linked:
            _unlink_owned(parent_fd, destination_name, temporary_info)
        _unlink_owned(parent_fd, temporary_name, temporary_info)
        try:
            _fsync_directory(parent_fd)
        except OSError:
            pass
        raise
    except Exception:  # noqa: BLE001 - publication errors are path/content safe
        if file_fd is not None:
            os.close(file_fd)
        if destination_linked:
            _unlink_owned(parent_fd, destination_name, temporary_info)
        _unlink_owned(parent_fd, temporary_name, temporary_info)
        try:
            _fsync_directory(parent_fd)
        except OSError:
            pass
        raise ConfigInitError("atomic_publication_failed", EXIT_PUBLICATION) from None


def initialize_baseline_config(
    output: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    home_directory: Path | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    generator: Callable[[], GeneratedRunKeyring] = generate_run_keyring,
    temporary_name_factory: Callable[[], str] = lambda: secrets.token_hex(16),
    fault_injector: Callable[[str], None] | None = None,
) -> ConfigInitResult:
    """Generate and atomically publish one private baseline secrets fragment."""

    destination_classification = "default" if output is None else "explicit"
    try:
        destination = (
            default_config_path(
                environ=environ,
                home_directory=home_directory,
            )
            if output is None
            else Path(output)
        )
    except (OSError, TypeError, ValueError):
        raise ConfigInitError("invalid_output_path", EXIT_USAGE_OR_PATH) from None
    _validate_output_path(destination)

    with _UMASK_LOCK:
        previous_umask = os.umask(0o077)
        parent_fd: int | None = None
        try:
            parent_fd = _open_private_parent(destination)
            _classify_existing_destination(parent_fd, destination.name)
            try:
                generated = generator()
            except RunKeyringGenerationError as exc:
                raise ConfigInitError(exc.code, EXIT_GENERATION) from None
            except Exception:  # noqa: BLE001 - random/provider details are private
                raise ConfigInitError("random_generation_failed", EXIT_GENERATION) from None
            fragment = _render_fragment(generated)
            _publish_exclusive(
                parent_fd,
                destination.name,
                fragment,
                temporary_name_factory=temporary_name_factory,
                fault_injector=fault_injector,
            )
            return ConfigInitResult(
                active_key_id=generated.active_key_id,
                destination_classification=destination_classification,
                timestamp=clock(),
            )
        finally:
            if parent_fd is not None:
                os.close(parent_fd)
            os.umask(previous_umask)


def add_config_init_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        help=(
            "absolute destination path; defaults to the private platform "
            "configuration location"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="emit one privacy-safe baseline-config-init-result.v1 value",
    )


def run_config_init_command(args: Any) -> int:
    try:
        result = initialize_baseline_config(args.output)
    except ConfigInitError as exc:
        if bool(args.json_output):
            print(
                json.dumps(
                    _failure_payload(exc.code),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        else:
            print(f"config init failed: {exc.code}", file=sys.stderr)
        return exc.exit_code
    except Exception:  # noqa: BLE001 - final command boundary is non-reflective
        if bool(args.json_output):
            print(
                json.dumps(
                    _failure_payload("internal_failure"),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        else:
            print("config init failed: internal_failure", file=sys.stderr)
        return EXIT_INTERNAL
    if bool(args.json_output):
        print(json.dumps(result.as_dict(), sort_keys=True, separators=(",", ":")))
    else:
        print(
            "created private baseline keyring secrets fragment "
            f"({result.destination_classification} destination)",
            file=sys.stderr,
        )
    return EXIT_SUCCESS


__all__ = [
    "CONFIG_INIT_ENVIRONMENT_VARIABLE",
    "CONFIG_INIT_FILENAME",
    "CONFIG_INIT_RESULT_SCHEMA_VERSION",
    "EXIT_DESTINATION_EXISTS",
    "EXIT_GENERATION",
    "EXIT_INSECURE_PATH",
    "EXIT_INTERNAL",
    "EXIT_PUBLICATION",
    "EXIT_SUCCESS",
    "EXIT_USAGE_OR_PATH",
    "ConfigInitError",
    "ConfigInitResult",
    "add_config_init_arguments",
    "default_config_path",
    "initialize_baseline_config",
    "run_config_init_command",
]
