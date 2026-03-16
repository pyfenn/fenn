from pathlib import Path


def copy_template(source: Path, destination: Path) -> None:
    with open(source, "r") as f:
        template = f.read()
    destination.write_text(template)
