import os
from typing import Any, Dict

import yaml
from colorama import init

from fenn.secrets.keystore import KeyStore


class Parser:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:

        self._config_file: str = "fenn.yaml"
        self._args: Dict[str, Any] = {}

        self._keystore: KeyStore = KeyStore()

        init(autoreset=True)

    def load_configuration(self) -> Any:
        """Loads the YAML configuration into the _args dictionary."""
        from fenn.logging import Logger

        logger = Logger()

        if not os.path.isfile(self._config_file):
            logger.display_excpetion(
                f"Configuration file {self._config_file} was not found."
            )

            raise FileNotFoundError(
                0,
                f"Configuration file {self._config_file} was not found.",
                self._config_file,
            )

        # File exists → load YAML
        with open(self._config_file) as f:
            self._args = yaml.safe_load(f)
            self._args["project"] = self._config_file.split("/")[-1].split(".")[0]

        return self._args

    def print(self) -> None:
        """Public method to trigger the flattened print with colored paths."""
        from fenn.logging import Logger

        Logger().write_config(self._args)

    @property
    def config_file(self) -> str:
        return self._config_file

    @config_file.setter
    def config_file(self, config_file: str) -> None:
        self._config_file = config_file

    @property
    def args(self) -> Dict[str, Any]:
        return self._args
