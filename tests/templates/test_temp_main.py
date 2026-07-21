import os
import subprocess
import sys

import pytest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.abspath(os.path.join(TEST_DIR, "..", "templates"))

TEMPLATES_LIST = [
    "chatbot",
    "cnn",
    "empty",
    "lora-cls",
    "lstm-cls",
    "lstm-gen",
    "mlp-binary",
    "mlp-multiclass",
    "mlp-regression",
    "timm-cls",
    "vae",
    "yolo",
]


def get_template_directories():
    if not os.path.exists(TEMPLATE_DIR):
        return []

    return [
        d
        for d in os.listdir(TEMPLATE_DIR)
        if os.path.isdir(os.path.join(TEMPLATE_DIR, d))
        and d not in [".git", "__pycache__"]
    ]


def test_templates_dir_exists():
    assert os.path.exists(TEMPLATE_DIR), (
        f"Templates directory not found at {TEMPLATE_DIR}. Did you pull the submodule?"
    )


@pytest.mark.parametrize("template_name", get_template_directories())
def test_template_structure_and_execution(template_name):
    assert template_name in TEMPLATES_LIST, (
        f"Unexpected directory found in templates: {template_name}"
    )

    template_path = os.path.join(TEMPLATE_DIR, template_name)
    main_path = os.path.join(template_path, "main.py")
    requirement_path = os.path.join(template_path, "requirements.txt")

    assert os.path.exists(main_path), f"Missing main.py in {template_name}"
    assert os.path.exists(requirement_path), (
        f"Missing requirements.txt in {template_name}"
    )

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirement_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [sys.executable, main_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        raise AssertionError(f"Template '{template_name}' execution failed: {e}")
    finally:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", "-r", requirement_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
