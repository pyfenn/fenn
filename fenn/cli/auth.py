"""``fenn auth`` — manage local credentials for the Fenn remote service."""

from __future__ import annotations

import argparse
import getpass
import sys

from colorama import Fore, Style

from fenn.exceptions import AuthError, NetworkError, RemoteError
from fenn.logging import logger
from fenn.remote.client import DEFAULT_REMOTE_HOST, RemoteClient
from fenn.remote.credentials import (
    delete_profile,
    load_credentials,
    mask_key,
    write_credentials,
)


def execute(args: argparse.Namespace) -> None:
    """Entrypoint wired from :func:`fenn.cli.build_parser` for ``fenn auth``.

    Dispatches on ``args.auth_command`` (``login`` / ``status`` / ``logout``).
    """
    command = args.auth_command
    if command == "login":
        _login(args)
    elif command == "status":
        _status(args)
    elif command == "logout":
        _logout(args)
    else:  # pragma: no cover - argparse enforces valid subcommands
        raise SystemExit(f"Unknown auth command: {command}")


def _login(args: argparse.Namespace) -> None:
    profile = args.profile or "default"
    api_key = args.api_key
    if not api_key:
        if sys.stdin.isatty():
            api_key = getpass.getpass("API key: ").strip()
        else:
            api_key = sys.stdin.readline().strip()

    if not api_key:
        logger.info(f"{Fore.RED}No API key provided.{Style.RESET_ALL}")
        sys.exit(2)

    try:
        client = RemoteClient(DEFAULT_REMOTE_HOST, api_key)
        client.me()
    except AuthError as exc:
        logger.info(f"{Fore.RED}Key rejected by server: {exc}{Style.RESET_ALL}")
        sys.exit(2)
    except (NetworkError, RemoteError) as exc:
        logger.info(f"{Fore.RED}Could not verify key: {exc}{Style.RESET_ALL}")
        sys.exit(1)

    path = write_credentials(api_key, profile=profile)
    logger.info(
        f"{Fore.GREEN}Saved key {mask_key(api_key)} to profile "
        f"'{profile}' ({path}).{Style.RESET_ALL}"
    )


def _status(args: argparse.Namespace) -> None:
    profile = args.profile or "default"
    creds = load_credentials(profile)
    if creds is None:
        logger.info(
            f"{Fore.YELLOW}No credentials for profile '{profile}'. "
            f"Run `fenn auth login --profile {profile}`.{Style.RESET_ALL}"
        )
        sys.exit(1)

    logger.info(f"Profile: {profile}  Key: {mask_key(creds.api_key)}")

    host = creds.host or DEFAULT_REMOTE_HOST
    try:
        client = RemoteClient(host, creds.api_key)
        info = client.me()
    except AuthError as exc:
        logger.info(f"{Fore.RED}Key rejected by server: {exc}{Style.RESET_ALL}")
        sys.exit(2)
    except (NetworkError, RemoteError) as exc:
        logger.info(f"{Fore.RED}Could not reach {host}: {exc}{Style.RESET_ALL}")
        sys.exit(1)

    plan = info.get("plan", "?")
    credits = info.get("credits", "?")
    machine_classes = ", ".join(info.get("machine_classes", []) or [])
    logger.info(f"Plan: {plan}  Credits: {credits}")
    if machine_classes:
        logger.info(f"Machine classes: {machine_classes}")


def _logout(args: argparse.Namespace) -> None:
    profile = args.profile or "default"
    if delete_profile(profile):
        logger.info(f"{Fore.GREEN}Removed profile '{profile}'.{Style.RESET_ALL}")
    else:
        logger.info(f"{Fore.YELLOW}No profile named '{profile}'.{Style.RESET_ALL}")
