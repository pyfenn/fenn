#!/usr/bin/env bash
set -e

# Project root = directory of this script
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$ROOT_DIR"

# Clean previous build artefacts
rm -rf dist
rm -rf src/fenn.egg-info

# python -m pip install build
python -m build
# python -m pip install twine
python -m twine upload dist/* --verbose