"""Local credential storage for the Fenn remote service.

Credentials live in ``~/.fenn/credentials`` as JSON, one entry per named
profile::

    {
      "profiles": {
        "default": {"api_key": "fk_live_...", "host": null}
      }
    }

Resolution order for :func:`resolve_api_key`:

1. explicit ``--api-key`` flag
2. ``FENN_API_KEY`` environment variable
3. profile from the credentials file (name from argument, else
   ``FENN_PROFILE`` env var, else ``default``)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fenn.exceptions import CredentialsError

DEFAULT_PROFILE = "default"

ENV_API_KEY = "FENN_API_KEY"
ENV_PROFILE = "FENN_PROFILE"


def _home() -> Path:
    home = os.environ.get("HOME") or os.environ.get("USERPROFILE")
    return Path(home) if home else Path.home()


CREDENTIALS_DIR = _home() / ".fenn"
CREDENTIALS_PATH = CREDENTIALS_DIR / "credentials"


@dataclass(frozen=True)
class Credentials:
    profile: str
    api_key: str
    host: Optional[str] = None


def _read_store() -> dict:
    if not CREDENTIALS_PATH.is_file():
        return {"profiles": {}}
    try:
        with open(CREDENTIALS_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise CredentialsError(
            f"Could not read credentials file {CREDENTIALS_PATH}: {exc}"
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("profiles"), dict):
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


def write_credentials(
    api_key: str,
    *,
    profile: str = DEFAULT_PROFILE,
    host: Optional[str] = None,
) -> Path:
    """Persist ``api_key`` under ``profile`` and return the file path."""
    store = _read_store()
    store["profiles"][profile] = {"api_key": api_key, "host": host}
    _write_store(store)
    return CREDENTIALS_PATH


def load_credentials(profile: str = DEFAULT_PROFILE) -> Optional[Credentials]:
    """Return the stored :class:`Credentials` for ``profile``, or ``None``."""
    store = _read_store()
    entry = store["profiles"].get(profile)
    if not isinstance(entry, dict) or not entry.get("api_key"):
        return None
    return Credentials(
        profile=profile,
        api_key=str(entry["api_key"]),
        host=entry.get("host") or None,
    )


def delete_profile(profile: str = DEFAULT_PROFILE) -> bool:
    """Remove ``profile`` from the store. Returns True if it existed."""
    store = _read_store()
    if profile not in store["profiles"]:
        return False
    del store["profiles"][profile]
    _write_store(store)
    return True


def resolve_api_key(
    *,
    explicit: Optional[str] = None,
    profile: Optional[str] = None,
) -> Credentials:
    """Resolve an API key from flag > env > credentials file.

    Raises:
        CredentialsError: when no key can be found anywhere.
    """
    if explicit:
        return Credentials(profile=profile or DEFAULT_PROFILE, api_key=explicit)

    env_key = os.environ.get(ENV_API_KEY)
    if env_key:
        return Credentials(profile=profile or DEFAULT_PROFILE, api_key=env_key)

    profile_name = profile or os.environ.get(ENV_PROFILE) or DEFAULT_PROFILE
    creds = load_credentials(profile_name)
    if creds is None:
        raise CredentialsError(
            f"No API key found. Pass --api-key, set {ENV_API_KEY}, or run "
            f"`fenn auth login` (profile: {profile_name})."
        )
    return creds


def mask_key(api_key: str) -> str:
    """Return a redacted form safe for terminal output."""
    if len(api_key) <= 12:
        return "*" * len(api_key)
    return f"{api_key[:8]}...{api_key[-4:]}"
