"""Run a project entrypoint locally, mirroring the remote execution layout.

The remote runner executes the entrypoint as a module inside its parent
directory (so ``from .modules import x`` style relative imports work). This
local fallback reproduces that behaviour.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def run_local(script: Path) -> None:
    """Execute ``script`` with its directory as the package root."""
    script = Path(script).resolve()
    package_dir = script.parent
    search_root = package_dir.parent

    old_cwd = os.getcwd()
    sys.path.insert(0, str(search_root))
    try:
        os.chdir(package_dir)
        runpy.run_module(
            f"{package_dir.name}.{script.stem}",
            run_name="__main__",
            alter_sys=True,
        )
    finally:
        os.chdir(old_cwd)
        try:
            sys.path.remove(str(search_root))
        except ValueError:
            pass
