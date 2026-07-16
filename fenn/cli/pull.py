import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import requests
from colorama import Fore, Style

from fenn.exceptions import NetworkError, TemplateError, TemplateNotFoundError
from fenn.logging import logger

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
    )

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

TEMPLATES_REPO = "pyfenn/templates"
REPO_NAME = "templates"
GITHUB_API_BASE = "https://api.github.com"
GITHUB_ARCHIVE_BASE = "https://github.com"


def execute(args: argparse.Namespace) -> None:
    """
    Execute the fenn pull command to download a template from GitHub.

    Args:
        args: Parsed command-line arguments containing:
            - template: Name of the template to download
            - path: Target directory
            - force: Whether to overwrite existing files
    """
    template_name = args.template
    target_dir = Path(args.path).resolve() if args.path else Path.cwd()
    force = args.force

    if not template_name:
        logger.error(
            f"{Fore.RED}Template name is required "
            f"(example: {Fore.LIGHTYELLOW_EX}fenn pull base"
            f"{Fore.RED}){Style.RESET_ALL}"
        )
        logger.info(
            f"{Fore.CYAN}Use {Fore.LIGHTYELLOW_EX}fenn list"
            f"{Fore.CYAN} to see available templates."
            f"{Style.RESET_ALL}"
        )
        sys.exit(1)

    try:
        target_dir = pull_template(template_name, target_dir, force)

        logger.info(
            f"{Fore.GREEN}Successfully pulled template "
            f"{Fore.LIGHTYELLOW_EX}{template_name}{Fore.GREEN} into "
            f"{Fore.LIGHTYELLOW_EX}{target_dir}{Fore.GREEN}."
            f"{Style.RESET_ALL}"
        )

        requirements_file = target_dir / "requirements.txt"

        if requirements_file.exists():
            if HAS_RICH:
                Console().print(
                    "\n[bold green]"
                    "📦 Found template dependencies at requirements.txt"
                    "[/bold green]"
                )
            else:
                logger.info(
                    f"\n{Fore.GREEN}"
                    "📦 Found template dependencies at requirements.txt"
                    f"{Style.RESET_ALL}"
                )

            try:
                response = (
                    input(
                        "Would you like to automatically install requirements? [y/N]: "
                    )
                    .strip()
                    .lower()
                )
            except (KeyboardInterrupt, EOFError):
                response = "n"
                print()

            if response in ("y", "yes"):
                if HAS_RICH:
                    Console().print(
                        "[cyan]Installing requirements automatically... "
                        "(this may take a moment)[/cyan]\n"
                    )
                else:
                    logger.info(
                        f"{Fore.CYAN}"
                        "Installing requirements automatically... "
                        "(this may take a moment)"
                        f"{Style.RESET_ALL}\n"
                    )

                try:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "-r",
                            str(requirements_file),
                        ],
                        check=True,
                        capture_output=False,
                    )

                    if HAS_RICH:
                        Console().print(
                            "\n[bold green]"
                            "✅ All dependencies installed successfully!"
                            "[/bold green]"
                        )
                    else:
                        logger.info(
                            f"\n{Fore.GREEN}"
                            "✅ All dependencies installed successfully!"
                            f"{Style.RESET_ALL}"
                        )

                except subprocess.CalledProcessError as exc:
                    if HAS_RICH:
                        Console().print(
                            "\n[bold red]"
                            "⚠️ Automatic installation failed with exit "
                            f"code {exc.returncode}."
                            "[/bold red]"
                        )
                        Console().print(
                            "[yellow]"
                            "Please run 'pip install -r requirements.txt' "
                            "manually."
                            "[/yellow]"
                        )
                    else:
                        logger.error(
                            f"\n{Fore.RED}"
                            "⚠️ Automatic installation failed with exit "
                            f"code {exc.returncode}."
                            f"{Style.RESET_ALL}"
                        )
                        logger.info(
                            f"{Fore.YELLOW}"
                            "Please run 'pip install -r requirements.txt' "
                            "manually."
                            f"{Style.RESET_ALL}"
                        )
            else:
                if HAS_RICH:
                    Console().print(
                        "[yellow]Skipping dependency installation.[/yellow]\n"
                    )
                else:
                    logger.info(
                        f"{Fore.YELLOW}Skipping dependency installation."
                        f"{Style.RESET_ALL}\n"
                    )

    except FileExistsError as exc:
        logger.error(f"{Fore.RED}{exc}{Style.RESET_ALL}")
        sys.exit(1)
    except TemplateNotFoundError as exc:
        logger.error(f"{Fore.RED}{exc}{Style.RESET_ALL}")
        sys.exit(1)
    except NetworkError as exc:
        logger.error(f"{Fore.RED}Network error: {exc}{Style.RESET_ALL}")
        sys.exit(1)
    except TemplateError as exc:
        logger.error(f"{Fore.RED}Template error: {exc}{Style.RESET_ALL}")
        sys.exit(1)


def pull_template(
    template_name: str,
    target_dir: Path,
    force: bool = False,
) -> Path:
    """Download a template into a validated target directory.

    Args:
        template_name: Name of the template to download.
        target_dir: Directory where the template will be extracted.
        force: Whether existing visible files may be overwritten.

    Returns:
        The resolved target directory.

    Raises:
        TemplateError: If the template name is empty.
        FileExistsError: If the target directory is non-empty without force.
        TemplateNotFoundError: If the requested template does not exist.
        NetworkError: If the template repository cannot be reached.
    """
    template_name = template_name.strip()

    if not template_name:
        raise TemplateError("Template name is required")

    target_dir = target_dir.expanduser().resolve()

    has_visible_files = target_dir.exists() and any(
        not item.name.startswith(".") for item in target_dir.iterdir()
    )

    if has_visible_files and not force:
        raise FileExistsError(
            f"Refusing to pull into non-empty directory {target_dir}. "
            "Use --force to override existing files."
        )

    _download_template(template_name, target_dir, force)
    return target_dir


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
        if e.response.status_code == 404:  # ty: ignore[unresolved-attribute]
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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
        try:
            if HAS_RICH:
                console = Console()
                with Progress(
                    TextColumn("[bold blue]Downloading {task.fields[template]}"),
                    BarColumn(),
                    DownloadColumn(),
                    TimeElapsedColumn(),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        "Downloading", template=template_name, total=None
                    )
                    for chunk in response.iter_content(chunk_size=8192):
                        tmp_file.write(chunk)
                        progress.update(task, advance=len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
            tmp_file.flush()

            with zipfile.ZipFile(tmp_file.name, "r") as zip_ref:
                # Find all files in the template directory
                template_prefix = f"{REPO_NAME}-main/{template_name}/"
                template_files = [
                    f for f in zip_ref.namelist() if f.startswith(template_prefix)
                ]

                if not template_files:
                    raise TemplateError(
                        f"Template {Fore.LIGHTYELLOW_EX}{template_name}{Fore.RED} "
                        f"appears to be empty or has an unexpected structure."
                    )

                target_dir.mkdir(parents=True, exist_ok=True)

                # Extract files and dirs, removing the template prefix from paths
                for file_path in template_files:
                    relative_path = file_path[len(template_prefix) :]
                    if not relative_path:
                        continue
                    if file_path.endswith("/"):
                        (target_dir / relative_path.rstrip("/")).mkdir(
                            parents=True, exist_ok=True
                        )
                        continue
                    dest_path = target_dir / relative_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    with zip_ref.open(file_path) as source:
                        dest_path.write_bytes(source.read())
        finally:
            # Clean up temporary file
            tmp_file.close()
            os.unlink(tmp_file.name)
