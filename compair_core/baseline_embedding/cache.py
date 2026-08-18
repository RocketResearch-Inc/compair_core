"""Secure acquisition and offline verification for the frozen BGE snapshot."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from .manifest import BaselineModelManifest, load_baseline_model_manifest

CACHE_ENVIRONMENT_VARIABLE = "COMPAIR_BASELINE_MODEL_CACHE"
_STAGING_DIRECTORY = ".staging"
_LOCK_FILE = ".baseline-v1.fetch.lock"

DownloadArtifact = Callable[[str, str, str, Path], Path]


class BaselineModelCacheError(RuntimeError):
    """Sanitized cache/acquisition error safe for status and logs."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class VerifiedBaselineModel:
    manifest: BaselineModelManifest
    snapshot_dir: Path = field(repr=False)

    def safe_summary(self) -> dict[str, object]:
        return {"status": "verified", **self.manifest.safe_summary()}


@dataclass(frozen=True, slots=True)
class BaselineModelFetchResult:
    model: VerifiedBaselineModel
    downloaded: bool

    def safe_summary(self) -> dict[str, object]:
        return {
            **self.model.safe_summary(),
            "downloaded": self.downloaded,
        }


def default_cache_root() -> Path:
    """Return the operator-private model cache root without creating it."""

    configured = os.environ.get(CACHE_ENVIRONMENT_VARIABLE)
    if configured:
        candidate = Path(configured).expanduser()
    else:
        xdg_cache = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
        candidate = base / "compair-core" / "models"
    if not candidate.is_absolute():
        raise BaselineModelCacheError("baseline_model_cache_path_invalid")
    return candidate


def _cache_root(value: str | os.PathLike[str] | None) -> Path:
    candidate = Path(value).expanduser() if value is not None else default_cache_root()
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise BaselineModelCacheError("baseline_model_cache_path_invalid")
    return candidate


def _lstat(path: Path) -> os.stat_result:
    try:
        return path.lstat()
    except FileNotFoundError:
        raise BaselineModelCacheError("baseline_model_absent") from None
    except OSError:
        raise BaselineModelCacheError("baseline_model_cache_unavailable") from None


def _require_private_directory(path: Path, *, absent_code: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise BaselineModelCacheError(absent_code) from None
    except OSError:
        raise BaselineModelCacheError("baseline_model_cache_unavailable") from None
    if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
        raise BaselineModelCacheError("baseline_model_cache_unsafe")
    if metadata.st_mode & 0o022:
        raise BaselineModelCacheError("baseline_model_cache_unsafe")


def _make_private_directory(path: Path) -> None:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError:
        raise BaselineModelCacheError("baseline_model_cache_unavailable") from None
    _require_private_directory(path, absent_code="baseline_model_cache_unavailable")
    try:
        os.chmod(path, 0o700, follow_symlinks=False)
    except (NotImplementedError, OSError):
        raise BaselineModelCacheError("baseline_model_cache_unavailable") from None


def snapshot_path(
    cache_root: str | os.PathLike[str] | None = None,
    *,
    manifest: BaselineModelManifest | None = None,
) -> Path:
    frozen = manifest or load_baseline_model_manifest()
    return _cache_root(cache_root) / frozen.profile / frozen.revision


def _hash_regular_file(path: Path, *, expected_size: int) -> str:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except (OSError, ValueError):
        raise BaselineModelCacheError("baseline_model_artifact_unsafe") from None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise BaselineModelCacheError("baseline_model_artifact_mismatch")
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise BaselineModelCacheError("baseline_model_artifact_changed")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def verify_snapshot_directory(
    directory: Path,
    *,
    manifest: BaselineModelManifest | None = None,
) -> VerifiedBaselineModel:
    """Verify exactly the frozen regular-file set without following symlinks."""

    frozen = manifest or load_baseline_model_manifest()
    _require_private_directory(directory, absent_code="baseline_model_absent")
    try:
        entries = tuple(sorted(directory.iterdir(), key=lambda value: value.name))
    except OSError:
        raise BaselineModelCacheError("baseline_model_cache_unavailable") from None
    if tuple(entry.name for entry in entries) != frozen.expected_paths:
        raise BaselineModelCacheError("baseline_model_manifest_mismatch")
    for artifact, path in zip(frozen.files, entries):
        metadata = _lstat(path)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_mode & 0o022
        ):
            raise BaselineModelCacheError("baseline_model_artifact_unsafe")
        actual_hash = _hash_regular_file(path, expected_size=artifact.size)
        if actual_hash != artifact.sha256:
            raise BaselineModelCacheError("baseline_model_artifact_hash_mismatch")
    return VerifiedBaselineModel(manifest=frozen, snapshot_dir=directory)


def verify_baseline_model(
    cache_root: str | os.PathLike[str] | None = None,
    *,
    manifest: BaselineModelManifest | None = None,
) -> VerifiedBaselineModel:
    frozen = manifest or load_baseline_model_manifest()
    root = _cache_root(cache_root)
    _require_private_directory(root, absent_code="baseline_model_absent")
    _require_private_directory(
        root / frozen.profile,
        absent_code="baseline_model_absent",
    )
    return verify_snapshot_directory(
        snapshot_path(root, manifest=frozen),
        manifest=frozen,
    )


def _download_with_hugging_face(
    repository: str,
    revision: str,
    filename: str,
    local_dir: Path,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import disable_progress_bars
    except ImportError:
        raise BaselineModelCacheError(
            "baseline_model_fetch_dependency_unavailable"
        ) from None
    try:
        disable_progress_bars()
        downloaded = hf_hub_download(
            repo_id=repository,
            filename=filename,
            revision=revision,
            repo_type="model",
            local_dir=str(local_dir),
        )
    except Exception:  # noqa: BLE001 - never disclose URLs, tokens, or paths
        raise BaselineModelCacheError("baseline_model_fetch_failed") from None
    return Path(downloaded)


def _copy_verified_download(
    source: Path,
    destination: Path,
    *,
    expected_size: int,
    expected_hash: str,
) -> None:
    source_metadata = _lstat(source)
    if stat.S_ISLNK(source_metadata.st_mode) or not stat.S_ISREG(
        source_metadata.st_mode
    ):
        raise BaselineModelCacheError("baseline_model_download_unsafe")
    actual_hash = _hash_regular_file(source, expected_size=expected_size)
    if actual_hash != expected_hash:
        raise BaselineModelCacheError("baseline_model_download_hash_mismatch")

    source_flags = os.O_RDONLY
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
        destination_flags |= os.O_NOFOLLOW
    source_fd: int | None = None
    destination_fd: int | None = None
    try:
        source_fd = os.open(source, source_flags)
        destination_fd = os.open(destination, destination_flags, 0o600)
        while block := os.read(source_fd, 1024 * 1024):
            view = memoryview(block)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
        os.fsync(destination_fd)
    except OSError:
        raise BaselineModelCacheError("baseline_model_cache_unavailable") from None
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)
    if _hash_regular_file(destination, expected_size=expected_size) != expected_hash:
        raise BaselineModelCacheError("baseline_model_download_hash_mismatch")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


@contextmanager
def _exclusive_lock(path: Path, timeout_seconds: float):
    """Hold a bounded POSIX advisory lock without an optional dependency."""

    try:
        import fcntl
    except ImportError:
        raise BaselineModelCacheError("baseline_model_lock_unavailable") from None
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
    except OSError:
        raise BaselineModelCacheError("baseline_model_cache_unsafe") from None
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise BaselineModelCacheError("baseline_model_cache_unsafe")
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise BaselineModelCacheError(
                        "baseline_model_fetch_lock_timeout"
                    ) from None
                time.sleep(0.05)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def fetch_baseline_model(
    cache_root: str | os.PathLike[str] | None = None,
    *,
    manifest: BaselineModelManifest | None = None,
    downloader: DownloadArtifact | None = None,
    lock_timeout_seconds: float = 600.0,
) -> BaselineModelFetchResult:
    """Explicitly fetch, verify, and atomically publish the frozen snapshot."""

    frozen = manifest or load_baseline_model_manifest()
    root = _cache_root(cache_root)
    _make_private_directory(root)
    profile_root = root / frozen.profile
    staging_root = profile_root / _STAGING_DIRECTORY
    _make_private_directory(profile_root)
    _make_private_directory(staging_root)
    lock_path = profile_root / _LOCK_FILE
    with _exclusive_lock(lock_path, lock_timeout_seconds):
        target = snapshot_path(root, manifest=frozen)
        if target.exists() or target.is_symlink():
            return BaselineModelFetchResult(
                model=verify_snapshot_directory(target, manifest=frozen),
                downloaded=False,
            )

        try:
            stage = Path(tempfile.mkdtemp(prefix="fetch-", dir=staging_root))
            os.chmod(stage, 0o700)
            download_dir = stage / "download"
            publication_dir = stage / "snapshot"
            download_dir.mkdir(mode=0o700)
            publication_dir.mkdir(mode=0o700)
        except OSError:
            raise BaselineModelCacheError("baseline_model_cache_unavailable") from None

        fetch_one = downloader or _download_with_hugging_face
        for artifact in frozen.files:
            try:
                source = fetch_one(
                    frozen.artifact_repository,
                    frozen.revision,
                    artifact.path,
                    download_dir,
                )
            except BaselineModelCacheError:
                raise
            except Exception:  # noqa: BLE001 - sanitize downloader failures
                raise BaselineModelCacheError("baseline_model_fetch_failed") from None
            if source != download_dir / artifact.path:
                raise BaselineModelCacheError("baseline_model_download_unsafe")
            _copy_verified_download(
                source,
                publication_dir / artifact.path,
                expected_size=artifact.size,
                expected_hash=artifact.sha256,
            )

        verify_snapshot_directory(publication_dir, manifest=frozen)
        _fsync_directory(publication_dir)
        try:
            os.rename(publication_dir, target)
        except FileExistsError:
            return BaselineModelFetchResult(
                model=verify_snapshot_directory(target, manifest=frozen),
                downloaded=False,
            )
        except OSError:
            raise BaselineModelCacheError("baseline_model_publish_failed") from None
        _fsync_directory(profile_root)
        published = verify_snapshot_directory(target, manifest=frozen)
        try:
            shutil.rmtree(stage)
        except OSError:
            pass
        return BaselineModelFetchResult(model=published, downloaded=True)


def cleanup_incomplete_fetches(
    cache_root: str | os.PathLike[str] | None = None,
    *,
    manifest: BaselineModelManifest | None = None,
) -> int:
    """Remove only incomplete ``fetch-*`` staging directories under the lock."""

    frozen = manifest or load_baseline_model_manifest()
    root = _cache_root(cache_root)
    profile_root = root / frozen.profile
    staging_root = profile_root / _STAGING_DIRECTORY
    if not staging_root.exists():
        return 0
    _require_private_directory(
        staging_root, absent_code="baseline_model_cache_unavailable"
    )
    removed = 0
    with _exclusive_lock(profile_root / _LOCK_FILE, 600.0):
        for candidate in sorted(staging_root.iterdir(), key=lambda value: value.name):
            metadata = candidate.lstat()
            if (
                candidate.name.startswith("fetch-")
                and stat.S_ISDIR(metadata.st_mode)
                and not stat.S_ISLNK(metadata.st_mode)
            ):
                shutil.rmtree(candidate)
                removed += 1
    return removed


__all__ = [
    "CACHE_ENVIRONMENT_VARIABLE",
    "BaselineModelCacheError",
    "BaselineModelFetchResult",
    "DownloadArtifact",
    "VerifiedBaselineModel",
    "cleanup_incomplete_fetches",
    "default_cache_root",
    "fetch_baseline_model",
    "snapshot_path",
    "verify_baseline_model",
    "verify_snapshot_directory",
]
