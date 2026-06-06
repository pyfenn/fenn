"""Profile-aware credentials store for the Fenn remote service.

The credentials file lives at ``~/.fenn/credentials`` and follows a TOML-like
flat profile layout, mirroring the AWS CLI's ``~/.aws/credentials``::

    [default]
    api_key = "fk_live_..."

    [work]
    api_key = "fk_live_..."

API key resolution order (highest priority first):

1. Explicit ``--api-key`` flag (passed in by the caller).
2. ``FENN_API_KEY`` environment variable.
3. ``~/.fenn/credentials`` ``[profile]`` section.
4. ``.env`` via :class:`fenn.secrets.keystore.KeyStore`.

Reads use stdlib ``tomllib`` (Python >=3.11). Writes use a small hand-rolled
serializer so we do not need to add ``tomli_w`` as a dependency.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import tomllib

from fenn.remote.exceptions import CredentialsError

DEFAULT_PROFILE = "default"
CREDENTIALS_DIR = Path.home() / ".fenn"
CREDENTIALS_FILE = CREDENTIALS_DIR / "credentials"
ENV_API_KEY = "FENN_API_KEY"
ENV_PROFILE = "FENN_PROFILE"


@dataclass
class Credentials:
    """A single resolved profile entry."""

    profile: str
    api_key: str
    host: Optional[str] = None


def _read_file() -> Dict[str, Dict[str, str]]:
    if not CREDENTIALS_FILE.exists():
        return {}
    try:
        with open(CREDENTIALS_FILE, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise CredentialsError(
            f"Failed to read credentials file {CREDENTIALS_FILE}: {exc}"
        ) from exc

    out: Dict[str, Dict[str, str]] = {}
    for profile, section in data.items():
        if not isinstance(section, dict):
            continue
        out[profile] = {str(k): str(v) for k, v in section.items()}
    return out


def load_credentials(profile: str = DEFAULT_PROFILE) -> Optional[Credentials]:
    """Return the ``[profile]`` section, or ``None`` if missing."""
    data = _read_file()
    section = data.get(profile)
    if section is None or "api_key" not in section:
        return None
    return Credentials(
        profile=profile,
        api_key=section["api_key"],
        host=section.get("host"),
    )


def write_credentials(
    api_key: str,
    *,
    profile: str = DEFAULT_PROFILE,
    host: Optional[str] = None,
) -> Path:
    """Persist ``api_key`` under ``[profile]``.

    ``host`` is retained for backward-compatible callers; the CLI uses the
    fixed remote endpoint from :mod:`fenn.remote.client`.

    Existing profiles are preserved. The file is created with mode ``0o600``
    on POSIX; on Windows the umask is left to the OS (the file lives under
    the user's home, which is already user-private).
    """
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    data = _read_file()
    data[profile] = {"api_key": api_key}
    if host:
        data[profile]["host"] = host

    serialized = _serialize(data)
    CREDENTIALS_FILE.write_text(serialized, encoding="utf-8")
    if sys.platform != "win32":
        try:
            os.chmod(CREDENTIALS_FILE, 0o600)
        except OSError:
            pass
    return CREDENTIALS_FILE


def delete_profile(profile: str = DEFAULT_PROFILE) -> bool:
    """Remove ``[profile]`` from the credentials file. Returns ``True`` if removed."""
    data = _read_file()
    if profile not in data:
        return False
    del data[profile]
    if data:
        CREDENTIALS_FILE.write_text(_serialize(data), encoding="utf-8")
    else:
        try:
            CREDENTIALS_FILE.unlink()
        except FileNotFoundError:
            pass
    return True


def _serialize(data: Dict[str, Dict[str, str]]) -> str:
    """Render the profile dict back into TOML.

    Only string values are supported; keys are constrained to a small ASCII
    set so we can emit them as bare keys. Values are emitted as basic strings
    with backslash escaping.
    """
    out: list[str] = []
    for profile in sorted(data.keys()):
        out.append(f"[{profile}]")
        for key in sorted(data[profile].keys()):
            value = data[profile][key]
            out.append(f"{key} = {_toml_string(value)}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def _toml_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def resolve_api_key(
    *,
    explicit: Optional[str] = None,
    profile: Optional[str] = None,
) -> Credentials:
    """Resolve an API key using the documented priority chain.

    Raises:
        CredentialsError: if no key can be resolved.
    """
    profile = profile or os.getenv(ENV_PROFILE) or DEFAULT_PROFILE

    if explicit:
        return Credentials(profile=profile, api_key=explicit, host=None)

    env_value = os.getenv(ENV_API_KEY)
    if env_value:
        return Credentials(profile=profile, api_key=env_value, host=None)

    creds = load_credentials(profile)
    if creds is not None:
        return creds

    try:
        from fenn.secrets.keystore import KeyStore

        dotenv_value = KeyStore().get_key(ENV_API_KEY)
    except KeyError:
        dotenv_value = None
    except Exception:
        dotenv_value = None

    if dotenv_value:
        return Credentials(profile=profile, api_key=dotenv_value, host=None)

    raise CredentialsError(
        "No Fenn API key found. Run `fenn auth login` to save one, "
        f"or set the {ENV_API_KEY} environment variable."
    )


def mask_key(api_key: str) -> str:
    """Return a display-safe rendering of ``api_key`` (first 8 + last 4 chars)."""
    if len(api_key) <= 12:
        return "*" * len(api_key)
    return f"{api_key[:8]}...{api_key[-4:]}"
