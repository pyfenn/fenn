"""``fenn auth`` — manage credentials for the Fenn remote service."""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import requests
from colorama import Fore, Style

from fenn.remote.client import DEFAULT_REMOTE_HOST
from fenn.remote.credentials import (
    DEFAULT_PROFILE,
    delete_profile,
    load_credentials,
    mask_key,
    write_credentials,
)
from fenn.remote.exceptions import RemoteError
from fenn.utils.logging import logger

if TYPE_CHECKING:
    from fenn.remote.client import RemoteClient
    from fenn.remote.credentials import Credentials


def execute(args: argparse.Namespace) -> None:
    sub: str | None = getattr(args, "auth_command", None)
    if sub is None:
        logger.info(
            f"{Fore.RED}Missing auth subcommand. Try: "
            f"{Fore.LIGHTYELLOW_EX}fenn auth login{Style.RESET_ALL}"
        )
        sys.exit(1)

    if sub == "login":
        _login(args)
    elif sub == "status":
        _status(args)
    elif sub == "logout":
        _logout(args)
    else:
        logger.info(
            f"{Fore.RED}Unknown auth subcommand: {sub}{Style.RESET_ALL}",
        )
        sys.exit(1)


def _login(args: argparse.Namespace) -> None:
    profile: str = args.profile or DEFAULT_PROFILE

    api_key: str | None = args.api_key
    if not api_key:
        existing: Credentials | None = load_credentials(profile)
        if existing is not None:
            logger.info(
                f"{Fore.GREEN}Already logged in (profile: {profile}, "
                f"key: {mask_key(existing.api_key)}). "
                f"Run {Fore.LIGHTYELLOW_EX}fenn auth logout{Fore.GREEN} first to switch keys.{Style.RESET_ALL}"
            )
            return

        if sys.stdin.isatty():
            api_key = getpass.getpass(
                f"Paste Fenn API key for profile [{profile}]: "
            ).strip()
        else:
            api_key = sys.stdin.readline().strip()

    if not api_key:
        logger.info(f"{Fore.RED}No API key provided.{Style.RESET_ALL}")
        sys.exit(1)

    path: Path = write_credentials(api_key, profile=profile)
    logger.info(
        f"{Fore.GREEN}Saved credentials to "
        f"{Fore.LIGHTYELLOW_EX}{path}{Fore.GREEN} (profile: {profile}).{Style.RESET_ALL}"
    )


def _status(args: argparse.Namespace) -> None:
    profile: str = args.profile or DEFAULT_PROFILE
    creds: Credentials | None = load_credentials(profile)
    if creds is None:
        logger.info(
            f"{Fore.YELLOW}No saved credentials for profile {profile!r}. "
            f"Run {Fore.LIGHTYELLOW_EX}fenn auth login{Fore.YELLOW} to add one.{Style.RESET_ALL}"
        )
        sys.exit(1)

    logger.info(
        f"{Fore.CYAN}profile : {Fore.LIGHTYELLOW_EX}{creds.profile}{Style.RESET_ALL}"
    )
    logger.info(
        f"{Fore.CYAN}api_key : "
        f"{Fore.LIGHTYELLOW_EX}{mask_key(creds.api_key)}{Style.RESET_ALL}"
    )
    logger.info(
        f"{Fore.CYAN}host    : "
        f"{Fore.LIGHTYELLOW_EX}{DEFAULT_REMOTE_HOST}{Style.RESET_ALL}"
    )

    try:
        from fenn.remote.client import RemoteClient

        with RemoteClient(DEFAULT_REMOTE_HOST, creds.api_key) as client:
            client: RemoteClient
            me: dict[str, object] = client.me()
        credits_remaining: object = me.get("credits")
        plan: object = me.get("plan")
        logger.info(
            f"{Fore.GREEN}credits : {Fore.LIGHTYELLOW_EX}{credits_remaining}"
            f"{Fore.GREEN}  plan: {plan}{Style.RESET_ALL}"
        )
    except requests.exceptions.SSLError:
        logger.info(
            f"{Fore.RED}SSL verification failed for {DEFAULT_REMOTE_HOST}. "
            f"Try: pip install --upgrade certifi{Style.RESET_ALL}",
        )
    except (RemoteError, requests.exceptions.ConnectionError) as exc:
        logger.info(
            f"{Fore.RED}Could not reach host: {exc}{Style.RESET_ALL}",
        )


def _logout(args: argparse.Namespace) -> None:
    profile: str = args.profile or DEFAULT_PROFILE
    if delete_profile(profile):
        logger.info(
            f"{Fore.GREEN}Removed credentials for profile "
            f"{Fore.LIGHTYELLOW_EX}{profile}{Style.RESET_ALL}"
        )
    else:
        logger.info(
            f"{Fore.YELLOW}No credentials found for profile {profile!r}.{Style.RESET_ALL}"
        )
