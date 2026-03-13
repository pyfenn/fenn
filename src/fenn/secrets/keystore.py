import os

from dotenv import dotenv_values


class KeyStore:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._keys = dotenv_values(".env")

    def set_key(self, service: str, key: str):
        self._keys[service] = key

    def get_key(self, service: str) -> str:
        # Check if the key is already in the environment
        key = os.getenv(service)
        if key:
            return key
        if service in self._keys.keys():
            return self._keys[service]
        else:
            raise KeyError(f"Key {service} not found in .env or environment")
