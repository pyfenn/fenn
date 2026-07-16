import argparse
import sys

import requests
from colorama import Fore, Style
from rich.console import Console
from rich.table import Table

from fenn.exceptions import NetworkError
from fenn.logging import logger

TEMPLATES_REPO = "pyfenn/templates"
REPO_NAME = "templates"
GITHUB_API_BASE = "https://api.github.com"
GITHUB_ARCHIVE_BASE = "https://github.com"


def execute(args: argparse.Namespace) -> None:
    """
    Execute the fenn list command, to show all accessible templates.
    """
    try:
        _list_templates()
    except NetworkError as e:
        logger.error(f"{Fore.RED}Network error: {e}{Style.RESET_ALL}")
        sys.exit(1)


def get_available_templates() -> list[str]:
    """
    Fetch and return the available public templates.

    Returns:
        Sorted template directory names.

    Raises:
        NetworkError: If GitHub cannot be reached or returns an invalid response.
    """
    api_url = f"{GITHUB_API_BASE}/repos/{TEMPLATES_REPO}/contents"

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        contents = response.json()
    except requests.exceptions.RequestException as exc:
        raise NetworkError(f"Failed to fetch template list: {exc}") from exc
    except ValueError as exc:
        raise NetworkError("GitHub returned an invalid template response") from exc

    if not isinstance(contents, list):
        raise NetworkError("GitHub returned an invalid template response")

    templates = [
        item["name"]
        for item in contents
        if isinstance(item, dict)
        and item.get("type") == "dir"
        and isinstance(item.get("name"), str)
        and not item["name"].endswith("dev-only")
    ]

    return sorted(templates)


def _list_templates() -> None:
    """Display all available templates."""
    templates = get_available_templates()

    if not templates:
        logger.info(
            f"{Fore.YELLOW}No templates found in the repository.{Style.RESET_ALL}"
        )
        return

    console = Console()
    table = Table(title="")
    table.add_column("Available templates", style="", width=50)

    for template in templates:
        table.add_row(f"- {template}")

    console.print(table)
    console.print(
        "[cyan]Use [yellow]fenn pull <template>[/yellow] to download a template.[/cyan]"
    )
