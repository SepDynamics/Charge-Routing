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

# Graphviz for network visualization
if ! command -v dot >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y graphviz
  else
    echo "graphviz not found; install manually if you need diagrams"
  fi
fi

# ffmpeg for generating animations
if ! command -v ffmpeg >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y ffmpeg
  else
    echo "ffmpeg not found; install manually if you need animations"
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

# Node.js for running simple web servers
if ! command -v node >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update && sudo apt-get install -y nodejs npm
  else
    echo "nodejs not found; install manually if you need local servers"
  fi
fi

echo "Environment setup complete."
