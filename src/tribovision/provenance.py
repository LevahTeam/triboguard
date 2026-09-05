"""Reproducibility metadata captured alongside every generated result file.

Result files are evidence. A reviewer must be able to tell which code revision,
which dependency versions, which random seed, and which input manifests produced
a number. Paths are recorded relative to the repository root so that artifacts
stay portable and do not leak the author's home directory.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

TRACKED_PACKAGES = ("torch", "numpy", "pillow", "pycocotools", "scipy", "remotezip")


def repo_root() -> Path:
    """Return the repository root that contains the ``src`` package directory."""
    return Path(__file__).resolve().parents[2]


def relative_to_repo(path: Path) -> str:
    """Render *path* relative to the repository root when possible.

    Absolute paths in committed result files expose the author's username and
    make artifacts non-portable, so this is used for every recorded path.
    """
    resolved = Path(path).resolve()
    root = repo_root()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.name


def git_revision() -> dict[str, str | bool | None]:
    """Return the current commit and whether the working tree is dirty."""

    def _run(*args: str) -> str | None:
        try:
            output = subprocess.run(
                ["git", *args],
                cwd=repo_root(),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return output.stdout.strip() if output.returncode == 0 else None

    commit = _run("rev-parse", "HEAD")
    status = _run("status", "--porcelain")
    return {
        "commit": commit,
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status.strip()),
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def environment() -> dict[str, object]:
    """Capture everything needed to explain why a rerun may differ."""
    record: dict[str, object] = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
        "git": git_revision(),
    }
    try:
        import torch

        record["torch_threads"] = torch.get_num_threads()
        record["accelerators"] = {
            "cuda": bool(torch.cuda.is_available()),
            "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        }
    except ImportError:  # pragma: no cover - torch is a declared dependency
        pass
    return record
