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
    template_name = args.template
    target_dir = Path(args.path).resolve() if args.path else Path.cwd()
    force = args.force

    if not template_name:
        print(f"{Fore.RED}Template name is required (example: {Fore.LIGHTYELLOW_EX}fenn pull base{Fore.RED}){Style.RESET_ALL}")
        print(f"{Fore.CYAN}Use {Fore.LIGHTYELLOW_EX}fenn list{Fore.CYAN} to see available templates.{Style.RESET_ALL}")
        sys.exit(1)

    has_visible_files = any(
        not item.name.startswith('.') for item in target_dir.iterdir()
    )

    if target_dir.exists() and has_visible_files and not force:
        print(
            f"{Fore.RED}Refusing to pull into non-empty directory "
            f"{Fore.LIGHTYELLOW_EX}{target_dir}{Fore.RED}. \n"
            f"Use {Fore.LIGHTYELLOW_EX}--force{Fore.RED} to override existing files.{Style.RESET_ALL}"
        )
        sys.exit(1)

    try:
        # Download and extract template
        _download_template(template_name, target_dir, force)
        print(
            f"{Fore.GREEN}Successfully pulled template "
            f"{Fore.LIGHTYELLOW_EX}{template_name}{Fore.GREEN} into "
            f"{Fore.LIGHTYELLOW_EX}{target_dir}{Fore.GREEN}.{Style.RESET_ALL}"
        )
    except TemplateNotFoundError as e:
        print(f"{Fore.RED}{e}{Style.RESET_ALL}")
        sys.exit(1)
    except NetworkError as e:
        print(f"{Fore.RED}Network error: {e}{Style.RESET_ALL}")
        sys.exit(1)
    except TemplateError as e:
        print(f"{Fore.RED}Template error: {e}{Style.RESET_ALL}")
        sys.exit(1)


def _download_template(template_name: str, target_dir: Path, force: bool) -> None:
    """
    Download a template from the GitHub repository and extract it.

    Args:
        template_name: Name of the template folder in the repository
        target_dir: Directory where template should be extracted
        force: Whether to overwrite existing files

    Raises:
        TemplateNotFoundError: If template doesn't exist
        NetworkError: If network request fails
        TemplateError: If template structure is invalid
    """
    # Check if template exists using GitHub API
    template_path = f"repos/{TEMPLATES_REPO}/contents/{template_name}"
    api_url = f"{GITHUB_API_BASE}/{template_path}"

    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise TemplateNotFoundError(
                f"Template {Fore.LIGHTYELLOW_EX}{template_name}{Fore.RED} not found. "
                f"Use {Fore.LIGHTYELLOW_EX}fenn pull --list{Fore.RED} to see available templates, "
                f"or visit {Fore.CYAN}https://github.com/{TEMPLATES_REPO}{Style.RESET_ALL}"
            )
        raise NetworkError(f"Failed to check template existence: {e}")
    except requests.exceptions.RequestException as e:
        raise NetworkError(f"Network request failed: {e}")

    # Download the template as a zip archive
    archive_url = f"{GITHUB_ARCHIVE_BASE}/{TEMPLATES_REPO}/archive/refs/heads/main.zip"

    try:
        response = requests.get(archive_url, timeout=30, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise NetworkError(f"Failed to download template archive: {e}")

    # Extract only the specific template directory from the archive
    with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
        try:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            tmp_file.flush()

            with zipfile.ZipFile(tmp_file.name, 'r') as zip_ref:
                # Find all files in the template directory
                template_prefix = f"{REPO_NAME}-main/{template_name}/"
                template_files = [
                    f for f in zip_ref.namelist()
                    if f.startswith(template_prefix)
                ]

                if not template_files:
                    raise TemplateError(
                        f"Template {Fore.LIGHTYELLOW_EX}{template_name}{Fore.RED} "
                        f"appears to be empty or has an unexpected structure."
                    )

                target_dir.mkdir(parents=True, exist_ok=True)

                # Extract files and dirs, removing the template prefix from paths
                for file_path in template_files:
                    relative_path = file_path[len(template_prefix):]
                    if not relative_path:
                        continue
                    if file_path.endswith('/'):
                        (target_dir / relative_path.rstrip('/')).mkdir(parents=True, exist_ok=True)
                        continue
                    dest_path = target_dir / relative_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    with zip_ref.open(file_path) as source:
                        dest_path.write_bytes(source.read())
        finally:
            # Clean up temporary file
            os.unlink(tmp_file.name)


class TemplateNotFoundError(Exception):
    """Raised when a template is not found in the repository."""
    pass


class NetworkError(Exception):
    """Raised when a network request fails."""
    pass


class TemplateError(Exception):
    """Raised when a template has an invalid structure."""
    pass