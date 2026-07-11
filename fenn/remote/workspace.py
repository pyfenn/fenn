"""Workspace packing for remote job submission.

Builds a ``tar.gz`` of the user's project directory, excluding transient
output directories (``logger/``, ``export/``…), VCS metadata, virtualenvs,
and anything matched by a ``.fennignore`` file or extra exclude patterns.
"""

from __future__ import annotations

import fnmatch
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from fenn.exceptions import WorkspaceTooLargeError

DEFAULT_MAX_BYTES = 100 * 1024 * 1024  # 100 MB uncompressed

DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    ".venv",
    "venv",
    ".fenn",
    "__pycache__",
    "node_modules",
    "logger",
    "export",
    "exports",
}

FENNIGNORE = ".fennignore"


@dataclass
class WorkspacePack:
    """Handle to a packed workspace tarball (delete via :meth:`cleanup`)."""

    path: Path
    script_relpath: str
    file_count: int
    uncompressed_bytes: int
    _tmpdir: Optional[str] = field(default=None, repr=False)

    def cleanup(self) -> None:
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None
        elif self.path.exists():
            self.path.unlink(missing_ok=True)


def _load_fennignore(root: Path) -> list[str]:
    ignore_file = root / FENNIGNORE
    if not ignore_file.is_file():
        return []
    patterns: list[str] = []
    for raw in ignore_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            patterns.append(line.rstrip("/"))
    return patterns


def _is_excluded(relpath: Path, patterns: Sequence[str]) -> bool:
    posix = relpath.as_posix()
    parts = relpath.parts
    for pattern in patterns:
        if fnmatch.fnmatch(posix, pattern):
            return True
        if any(fnmatch.fnmatch(part, pattern) for part in parts):
            return True
        if posix.startswith(pattern.rstrip("/") + "/"):
            return True
    return False


def _iter_files(
    root: Path,
    patterns: Sequence[str],
    extra_includes: Sequence[Path],
) -> Iterable[Path]:
    include_roots = {p.resolve() for p in extra_includes}

    def walk(directory: Path):
        for entry in sorted(directory.iterdir()):
            rel = entry.relative_to(root)
            forced = any(
                entry.resolve() == inc or inc in entry.resolve().parents
                for inc in include_roots
            )
            if entry.is_dir():
                if not forced and (
                    entry.name in DEFAULT_EXCLUDED_DIRS or _is_excluded(rel, patterns)
                ):
                    continue
                yield from walk(entry)
            elif entry.is_file():
                if forced or not _is_excluded(rel, patterns):
                    yield entry

    yield from walk(root)


def pack_workspace(
    root: Path,
    script: Path,
    *,
    extra_includes: Sequence[Path] = (),
    extra_excludes: Sequence[str] = (),
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> WorkspacePack:
    """Pack ``root`` into a tar.gz for upload.

    Args:
        root: Project directory (becomes the tar root).
        script: Entrypoint file; must live inside ``root``.
        extra_includes: Paths (relative to ``root`` or absolute) to force in
            even when an exclude rule matches them.
        extra_excludes: Extra shell-glob patterns to skip.
        max_bytes: Uncompressed size cap.

    Raises:
        ValueError: if ``script`` is outside ``root``.
        WorkspaceTooLargeError: if the uncompressed payload exceeds
            ``max_bytes``.
    """
    root = Path(root).resolve()
    script = Path(script).resolve()
    try:
        script_rel = script.relative_to(root)
    except ValueError:
        raise ValueError(
            f"Entrypoint {script} is outside the project root {root}; "
            "run `fenn run` from your project directory."
        ) from None

    patterns = list(extra_excludes) + _load_fennignore(root)
    includes = [
        p if p.is_absolute() else root / p for p in (Path(p) for p in extra_includes)
    ]

    tmpdir = tempfile.mkdtemp(prefix="fenn-workspace-")
    tar_path = Path(tmpdir) / "workspace.tar.gz"

    file_count = 0
    total_bytes = 0
    try:
        with tarfile.open(tar_path, mode="w:gz") as tar:
            for path in _iter_files(root, patterns, includes):
                size = path.stat().st_size
                total_bytes += size
                if total_bytes > max_bytes:
                    raise WorkspaceTooLargeError(
                        f"Workspace exceeds {max_bytes / (1024 * 1024):.0f} MB "
                        f"uncompressed. Move datasets out of the project dir or "
                        f"add them to {FENNIGNORE}."
                    )
                tar.add(path, arcname=path.relative_to(root).as_posix())
                file_count += 1
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise

    return WorkspacePack(
        path=tar_path,
        script_relpath=script_rel.as_posix(),
        file_count=file_count,
        uncompressed_bytes=total_bytes,
        _tmpdir=tmpdir,
    )


def detect_venv_spec(root: Path) -> Optional[dict]:
    """Return a venv build spec when the project ships a requirements file."""
    requirements = Path(root) / "requirements.txt"
    if requirements.is_file():
        return {"enabled": True, "requirements": "requirements.txt"}
    return None
