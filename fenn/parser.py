from pathlib import Path
from typing import Any

import yaml
from colorama import init

from fenn.keystore import KeyStore
from fenn.logging import logger


class Parser:
    _instance = None

    @classmethod
    def reset(cls) -> None:
        """Reset singleton state. Use between tests or distinct Fenn instances.
        Note: all existing Parser references become stale after calling this.
        """
        cls._instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_file: str | Path = "fenn.yaml") -> None:
        if hasattr(self, "_initialized"):
            return  # singleton: __init__ is a no-op after first construction

        self._config_file: Path = Path(config_file)
        self._args: dict[str, Any] = {}

        self._keystore: KeyStore = KeyStore()

        init(autoreset=True)

        self._initialized = True

    def _config_missing(self) -> None:
        logger.exception(f"Configuration file {self._config_file} was not found.")

        raise FileNotFoundError(
            0,
            f"Configuration file {self._config_file} was not found.",
            self._config_file,
        )

    def load_configuration(self) -> Any:
        """Loads the YAML configuration into the _args dictionary."""

        if not self._config_file.exists():
            self._config_missing()

        # File exists → load YAML
        with open(self._config_file) as f:
            self._args = yaml.safe_load(f)
            self._args["project"] = self._config_file.stem

        return self._args

    def print(self) -> None:
        """Public method to trigger the flattened print with colored paths."""
        if not self._args:
            raise RuntimeError(
                "Parser.print() called before load_configuration(). "
                "Call load_configuration() first."
            )
        logger.write_config(self._args, self.config_file)

    @property
    def config_file(self) -> str:
        return self._config_file

    @config_file.setter
    def config_file(self, config_file: str) -> None:
        self._config_file: Path = Path(config_file)

    @property
    def args(self) -> dict[str, Any]:
        return self._args
