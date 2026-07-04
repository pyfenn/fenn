import os
import subprocess
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.abspath(os.path.join(TEST_DIR, "..", "templates"))


def test_check_all_templates_dir():
    templates = [
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
    assert os.path.exists(TEMPLATE_DIR), (
        f"Templates directory not found at {TEMPLATE_DIR}"
    )

    template_dir = os.listdir(TEMPLATE_DIR)
    result = False
    for test in template_dir:
        if test in [".git", ".gitignore", "__init__.py", "LICENSE"]:
            continue
        elif test in templates:
            result = True
        else:
            result = False
    assert result is True


def test_check_for_main():
    template_dir = os.listdir(TEMPLATE_DIR)
    result = False
    for test in template_dir:
        main_path = os.path.join(TEMPLATE_DIR, test, "main.py")
        if test in [".git", ".gitignore", "__init__.py", "LICENSE"]:
            continue
        elif os.path.exists(main_path):
            result = True
        else:
            result = False
    assert result is True


def test_check_for_requirement():
    template_dir = os.listdir(TEMPLATE_DIR)
    result = False
    for test in template_dir:
        requirement_path = os.path.join(TEMPLATE_DIR, test, "requirements.txt")
        if test in [".git", ".gitignore", "__init__.py", "LICENSE"]:
            continue
        elif os.path.exists(requirement_path):
            result = True
        else:
            result = False
    assert result is True


def test_main():
    template_dir = os.listdir(TEMPLATE_DIR)
    result = False
    for test in template_dir:
        requirement_path = os.path.join(TEMPLATE_DIR, test, "requirements.txt")
        main_path = os.path.join(TEMPLATE_DIR, test, "main.py")

        if test in [".git", ".gitignore", "__init__.py", "LICENSE"]:
            continue
        else:
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

                result = True

            except subprocess.CalledProcessError as e:
                raise AssertionError(f"{test} failed: {e}")

            finally:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "uninstall",
                        "-y",
                        "-r",
                        requirement_path,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

    assert result is True
