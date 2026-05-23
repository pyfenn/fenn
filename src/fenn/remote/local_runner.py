"""Local execution branch for ``fenn run`` (used when no ``--host`` is set).

The default path mirrors ``python main.py``. If the entry script uses explicit
relative imports, we execute it as a module so imports such as
``from .modules import Model`` have a package context.
"""

from __future__ import annotations

import ast
import os
import runpy
import sys
import tokenize
from pathlib import Path


def run_local(script: Path) -> None:
    """Execute ``script`` as if it were invoked with ``python <script>``.

    Sets ``sys.argv[0]`` to the script path, sets ``__name__`` to ``__main__``,
    and chdirs to the script's parent so ``fenn.yaml`` is auto-discovered
    relative to the script (matching the behavior of ``python main.py`` from
    the project root).
    """
    script = script.resolve()
    if not script.is_file():
        raise FileNotFoundError(f"Script not found: {script}")

    relative_import_level = _max_relative_import_level(script)
    module_context = _module_context(script, relative_import_level)

    previous_argv = sys.argv[:]
    previous_cwd = os.getcwd()
    previous_path = sys.path[:]
    sys.argv = [str(script)]
    try:
        os.chdir(script.parent)
        _prepend_sys_path(script.parent)
        if relative_import_level and module_context is not None:
            module_name, module_root = module_context
            _prepend_sys_path(module_root)
            _prepend_sys_path(script.parent)
            runpy.run_module(module_name, run_name="__main__", alter_sys=True)
        else:
            runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path
        os.chdir(previous_cwd)


def _max_relative_import_level(script: Path) -> int:
    try:
        with tokenize.open(str(script)) as fh:
            tree = ast.parse(fh.read(), filename=str(script))
    except (OSError, SyntaxError, UnicodeError):
        return 0

    return max(
        (
            node.level
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level
        ),
        default=0,
    )


def _module_context(script: Path, min_package_depth: int) -> tuple[str, Path] | None:
    if min_package_depth <= 0:
        return None
    if not script.stem.isidentifier():
        return None

    parts = [script.stem]
    package_dir = script.parent
    package_depth = 0

    while True:
        if not package_dir.name.isidentifier():
            return None

        parts.insert(0, package_dir.name)
        package_depth += 1

        parent = package_dir.parent
        if parent == package_dir:
            return None

        parent_is_package = (parent / "__init__.py").is_file()
        if package_depth >= min_package_depth and not parent_is_package:
            return ".".join(parts), parent

        package_dir = parent


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    while path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)
