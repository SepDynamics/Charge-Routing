#!/bin/bash
set -e

# Basic Python environment setup
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi

# Optional plotting tool for quick data visualization
if ! command -v gnuplot >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y gnuplot
  else
    echo "gnuplot not found; install manually if you need plotting"
  fi
fi

# Ensure shellcheck is available for script linting
if ! command -v shellcheck >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y shellcheck
  else
    echo "shellcheck not found; please install it manually for linting"
  fi
fi

echo "Environment setup complete."
