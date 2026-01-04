import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import requests
from colorama import Fore, Style

TEMPLATES_REPO = "pyfenn/templates"
REPO_NAME = "templates"
GITHUB_API_BASE = "https://api.github.com"
GITHUB_ARCHIVE_BASE = "https://github.com"

def execute(args: argparse.Namespace) -> None:
    """
    Execute the fenn pull command to download a template from GitHub.

    Args:
        args: Parsed command-line arguments containing:
            - template: Name of the template to download (optional if --list is used)
            - path: Target directory (default: current directory)
            - force: Whether to overwrite existing files
    """
    try:
        _list_templates()
    except NetworkError as e:
        print(f"{Fore.RED}Network error: {e}{Style.RESET_ALL}")
        sys.exit(1)

class NetworkError(Exception):
    """Raised when a network request fails."""
    pass

def _list_templates() -> None:
    """
    List all available template directories in the templates repository.

    Raises:
        NetworkError: If network request fails
    """
    api_url = f"{GITHUB_API_BASE}/repos/{TEMPLATES_REPO}/contents"

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise NetworkError(f"Failed to fetch template list: {e}")

    contents = response.json()

    templates = [
        item["name"] for item in contents
        if item.get("type") == "dir"
    ]

    if not templates:
        print(f"{Fore.YELLOW}No templates found in the repository.{Style.RESET_ALL}")
        return

    templates.sort()

    print(f"{Fore.GREEN}Available templates:{Style.RESET_ALL}")
    for template in templates:
        if not template.endswith("dev-only"):
            print(f" - {Fore.LIGHTYELLOW_EX}{template}{Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}Use {Fore.LIGHTYELLOW_EX}fenn pull <template>{Fore.CYAN} to download a template.{Style.RESET_ALL}")
