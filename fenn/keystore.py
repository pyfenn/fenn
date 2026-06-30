import os

from dotenv import dotenv_values


class KeyStore:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_keys"):
            self._keys = dotenv_values(".env")

    def set_key(self, key: str, value: str) -> None:
        self._keys[key] = value

    def get_key(self, key: str) -> str:
        env_value = os.getenv(key)
        if env_value is not None:
            return env_value

        dotenv_value = self._keys.get(key)
        if dotenv_value is not None:
            return dotenv_value

        raise KeyError(f"Key {key!r} not found in .env or environment")
