"""Local credential storage for the Fenn remote service.

A single credential lives in ``~/.fenn/credentials`` as JSON::

    {"api_key": "fk_live_...", "host": null}

Resolution order for :func:`resolve_api_key`:

1. ``FENN_API_KEY`` environment variable
2. the stored credential
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fenn.exceptions import CredentialsError

ENV_API_KEY = "FENN_API_KEY"


def _home() -> Path:
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    return Path(home) if home else Path.home()


CREDENTIALS_DIR = _home() / ".fenn"
CREDENTIALS_PATH = CREDENTIALS_DIR / "credentials"


@dataclass(frozen=True)
class Credentials:
    api_key: str
    host: Optional[str] = None


def _read_store() -> dict:
    if not CREDENTIALS_PATH.is_file():
        return {}
    try:
        with open(CREDENTIALS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise CredentialsError(
            f"Could not read credentials file {CREDENTIALS_PATH}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise CredentialsError(
            f"Malformed credentials file {CREDENTIALS_PATH}; "
            "delete it and run `fenn auth login` again."
        )
    return data


def _write_store(data: dict) -> None:
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CREDENTIALS_PATH, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    try:
        os.chmod(CREDENTIALS_PATH, 0o600)
    except OSError:
        # Best effort — not supported on all platforms (e.g. Windows).
        pass


def write_credentials(api_key: str, *, host: Optional[str] = None) -> Path:
    """Persist ``api_key`` and return the credentials file path."""
    _write_store({"api_key": api_key, "host": host})
    return CREDENTIALS_PATH


def load_credentials() -> Optional[Credentials]:
    """Return the stored :class:`Credentials`, or ``None`` if unset."""
    store = _read_store()
    if not isinstance(store, dict) or not store.get("api_key"):
        return None
    return Credentials(
        api_key=str(store["api_key"]),
        host=store.get("host") or None,
    )


def delete_credentials() -> bool:
    """Remove the stored credential. Returns True if one existed."""
    if not CREDENTIALS_PATH.is_file():
        return False
    if load_credentials() is None:
        return False
    _write_store({})
    return True


def resolve_api_key() -> Credentials:
    """Resolve an API key from env > credentials file.

    Raises:
        CredentialsError: when no key can be found anywhere.
    """
    env_key = os.environ.get(ENV_API_KEY)
    if env_key:
        return Credentials(api_key=env_key)

    creds = load_credentials()
    if creds is None:
        raise CredentialsError(
            f"No API key found. Set {ENV_API_KEY} or run `fenn auth login`."
        )
    return creds


def mask_key(api_key: str) -> str:
    """Return a redacted form safe for terminal output."""
    if len(api_key) <= 12:
        return "*" * len(api_key)
    return f"{api_key[:8]}...{api_key[-4:]}"
