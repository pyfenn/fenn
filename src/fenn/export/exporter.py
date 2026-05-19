from pathlib import Path
from typing import Any, Dict, Optional


class Exporter:
    """Singleton responsible for managing a single export directory."""

    _instance: Optional["Exporter"] = None

    def __new__(cls) -> "Exporter":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._export_dir = Path("export/fenn")
        self._initialized = True

    def configure(self, args: Dict[str, Any]) -> Path:
        """Configure the export directory from arguments."""

        export_conf = args.get("export", {}) or {}

        export_dir = export_conf.get("dir", "export")
        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        export_project_dir = args.get("project", "fenn")
        export_project_dir = export_dir / export_project_dir
        export_project_dir.mkdir(parents=True, exist_ok=True)

        self._export_dir = export_project_dir

        return self._export_dir

    @property
    def export_dir(self) -> Path:
        """Return the export directory, ensuring it exists."""
        self._export_dir.mkdir(parents=True, exist_ok=True)
        return self._export_dir
