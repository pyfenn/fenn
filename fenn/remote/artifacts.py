"""Extraction of job artifact tarballs downloaded from the remote service."""

from __future__ import annotations

import tarfile
from pathlib import Path


def extract_artifacts(tar_path: Path, dest_root: Path) -> list[Path]:
    """Safely extract ``tar_path`` under ``dest_root``.

    Members that would escape ``dest_root`` (absolute paths, ``..`` traversal,
    symlinks) are skipped. Returns the list of files written.
    """
    dest_root = Path(dest_root).resolve()
    written: list[Path] = []

    with tarfile.open(tar_path, mode="r:*") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            target = (dest_root / member.name).resolve()
            if dest_root != target and dest_root not in target.parents:
                continue  # path traversal attempt — skip
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            with source, open(target, "wb") as fh:
                fh.write(source.read())
            written.append(target)

    return written
