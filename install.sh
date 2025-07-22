#!/bin/bash
# Basic setup script for this project
set -e

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "No requirements.txt found."
fi

echo "Environment setup complete."
