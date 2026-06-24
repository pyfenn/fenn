import os
import subprocess
import sys
import pytest

def test_check_all_templates_dir():
    templates = [ "chatbot", "cnn", "empty", "lora-cls", "lstm-cls", "lstm-gen", "mlp-binary",
                "mlp-multiclass", "mlp-regression", "timm-cls", "vae", "yolo"]
    template_dir = os.listdir(r"\fenn\templates")
    result  = False
    for test in template_dir:
        if test in [".git",".gitignore","__init__.py","LICENSE"]:
            continue
        elif test in templates:
            result = True
        else:
            result = False
    assert result == True

def test_check_for_main():
    path = r"\fenn\templates"
    template_dir = os.listdir(path)
    result  = False
    for test in template_dir:
        main_path = os.path.join(path, test, "main.py")
        if test in [".git",".gitignore","__init__.py","LICENSE"]:
            continue
        elif os.path.exists(main_path):
            result = True
        else:
            result = False
    assert result == True

def test_check_for_requirement():
    path = r"\fenn\templates"
    template_dir = os.listdir(path)
    result  = False
    for test in template_dir:
        requirement_path = os.path.join(path, test, "requirements.txt")
        if test in [".git",".gitignore","__init__.py","LICENSE"]:
            continue
        elif os.path.exists(requirement_path):
            result = True
        else:
            result = False
    assert result == True

def test_main():
    path = r"\fenn\templates"
    template_dir = os.listdir(path)
    result  = False
    for test in template_dir:
        requirement_path = os.path.join(path, test, "requirements.txt")
        main_path = os.path.join(path, test, "main.py")
        if test in [".git",".gitignore","__init__.py","LICENSE"]:
            continue
        else:
            try:
                print(f"\nTesting {test}")

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

                print(f"{test} ran successfully")

            except subprocess.CalledProcessError as e:
                raise AssertionError(f"{test} failed: {e}")

            finally:
                subprocess.run(
                    [sys.executable, "-m", "pip", "uninstall", "-y", "-r", requirement_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

    assert result == True