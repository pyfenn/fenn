"""Client library for the Fenn remote execution service.

Used by ``fenn auth`` and ``fenn run``. Public surface:

- :mod:`fenn.remote.credentials` — ``~/.fenn/credentials`` profile store
- :mod:`fenn.remote.client` — HTTP/SSE client for the ``/v1/*`` API
- :mod:`fenn.remote.workspace` — project packing for upload
- :mod:`fenn.remote.artifacts` — artifact tarball extraction
- :mod:`fenn.remote.exceptions` — typed error hierarchy
"""

from fenn.exceptions import (
    AuthError,
    CredentialsError,
    InsufficientCreditsError,
    JobFailedError,
    NetworkError,
    RemoteError,
    WorkspaceTooLargeError,
)
from fenn.remote.client import DEFAULT_REMOTE_HOST, RemoteClient
from fenn.remote.credentials import (
    DEFAULT_PROFILE,
    Credentials,
    delete_profile,
    load_credentials,
    mask_key,
    resolve_api_key,
    write_credentials,
)

__all__ = [
    "DEFAULT_REMOTE_HOST",
    "DEFAULT_PROFILE",
    "Credentials",
    "RemoteClient",
    "delete_profile",
    "load_credentials",
    "mask_key",
    "resolve_api_key",
    "write_credentials",
    "AuthError",
    "CredentialsError",
    "InsufficientCreditsError",
    "JobFailedError",
    "NetworkError",
    "RemoteError",
    "WorkspaceTooLargeError",
]
