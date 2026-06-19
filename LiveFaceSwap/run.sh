#!/usr/bin/env bash
set -e

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$APP_DIR/.venv"
PYTHON_EXEC="$(command -v python3 || command -v python || true)"

if [ -z "$PYTHON_EXEC" ]; then
  echo "Python 3 was not found on PATH. Install Python 3 and try again."
  exit 1
fi

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "Creating local virtual environment..."
  "$PYTHON_EXEC" -m venv "$VENV_DIR"
  if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment. Ensure Python 3 is installed and on PATH."
    exit 1
  fi
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$APP_DIR/requirements.txt"
python "$APP_DIR/app.py"
