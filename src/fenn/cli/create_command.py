import argparse
from pathlib import Path
from importlib.resources import files
from colorama import Fore, Style
import sys
import os
from fenn.cli.utils import copy_template

def execute(args: argparse.Namespace) -> None:

    root_dir = Path(args.path).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    # Get the path to the 'templates' folder relative to the current package
    templates_path = files(__package__).joinpath('templates')

    source: Path | None = None
    destination: Path | None = None

    if args.filetype == "yaml":
        source = templates_path/"fenn.yaml"
        destination = root_dir / "fenn.yaml"
    elif args.filetype == "env":
        source = templates_path/"dotenv"
        destination = root_dir / ".env"
    else:
        print(f"{Fore.RED}[FENN] Template not found for filetype {Fore.LIGHTYELLOW_EX}{args.filetype}{Fore.RED}.{Style.RESET_ALL}")
        sys.exit(1)

    copy_template(source, destination)
    print(f"{Fore.GREEN}[FENN] Initialized {Fore.LIGHTYELLOW_EX}{args.filetype}{Fore.GREEN} file in {Fore.LIGHTYELLOW_EX}{root_dir}{Fore.GREEN} directory.{Style.RESET_ALL}")
