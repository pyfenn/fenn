from pathlib import Path
from typing import Any, Dict, Optional


class Exporter:
    """Singleton responsible for centralized export directory management."""

    _instance: Optional["Exporter"] = None

    def __new__(cls) -> "Exporter":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self._root_dir = Path("exports")
        self._project: Optional[str] = None
        self._initialized = True

    def configure(self, args: Dict[str, Any]) -> Path:
        """Configure the exporter from parsed YAML arguments."""

        export_conf = args.get("export", {}) or {}
        export_dir = export_conf.get("dir", "exports")

        self._root_dir = Path(export_dir).expanduser()
        self._project = str(args.get("project")) if args.get("project") else None

        self._root_dir.mkdir(parents=True, exist_ok=True)
        return self._root_dir

    @property
    def root_dir(self) -> Path:
        """Return the configured export root directory."""
        return self._root_dir

    def get_export_dir(
        self,
        *parts: str,
        include_project: bool = True,
        create: bool = True,
    ) -> Path:
        """Return a resolved export directory beneath the configured root."""

        export_dir = self._root_dir
        if include_project and self._project:
            export_dir = export_dir / self._project

        for part in parts:
            export_dir = export_dir / part

        if create:
            export_dir.mkdir(parents=True, exist_ok=True)

        return export_dir