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

echo "Environment setup complete."
