#!/bin/bash
# codex_install.sh - Extended setup script for SEP Dynamics projects
#
# This script augments the existing Python environment setup with optional
# installation of two different "Codex" tools:
#  1. Codex Storage (a decentralized storage network) – installed using the official
#     install script. See docs: https://docs.codex.storage/learn/quick-start
#  2. OpenAI Codex CLI (a local coding agent) – installed via npm. See docs:
#     https://github.com/openai/codex for details.
#
# The script assumes you have Python, curl and (optionally) Node.js available.
# It checks for each tool before attempting installation, and prints guidance
# if dependencies are missing. Run this script from the root of your project.

set -e

# -----------------------------------------------------------------------------
# 1. Python virtual environment setup (unchanged from your original script)
# -----------------------------------------------------------------------------
if [ ! -d .venv ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv .venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install Python dependencies if a requirements file is present
if [ -f requirements.txt ]; then
  echo "Installing Python dependencies from requirements.txt..."
  pip install -r requirements.txt
else
  echo "No requirements.txt found. Skipping Python package installation."
fi

# -----------------------------------------------------------------------------
# 2. Install Codex Storage (optional)
#    Uses the official install script recommended in the Codex documentation.
#    On Linux/macOS the recommended command is:
#      curl -s https://get.codex.storage/install.sh | bash【423999350284784†L112-L135】.
#    The script downloads the latest binary and places it in $HOME/.codex.
#    Debian-based distributions also need the libgomp1 runtime【423999350284784†L112-L135】.
# -----------------------------------------------------------------------------

install_codex_storage() {
  if command -v codex >/dev/null 2>&1; then
    echo "Codex Storage already installed (codex command found). Skipping."
  else
    echo "Installing Codex Storage via the official install script..."
    # Download and execute the installer. It will prompt for confirmation and
    # install the binary into ~/.codex by default. If you run into permission
    # issues, you may need to re-run this command with sudo.
    curl -s https://get.codex.storage/install.sh | bash

    # If apt is available (Debian/Ubuntu) install libgomp1 for runtime support
    if [ "$(uname)" = "Linux" ] && command -v apt >/dev/null 2>&1; then
      echo "Installing libgomp1 dependency via apt..."
      # Use sudo if available; otherwise prompt the user
      if command -v sudo >/dev/null 2>&1; then
        sudo apt update && sudo apt install -y libgomp1
      else
        echo "Please install the libgomp1 package manually (requires root privileges)."
      fi
    fi

    # Verify installation
    if command -v codex >/dev/null 2>&1; then
      echo "Codex Storage installation complete."
    else
      echo "Codex Storage installation failed. Please check the installer output."
    fi
  fi
}

# -----------------------------------------------------------------------------
# 3. Install OpenAI Codex CLI (optional)
#    According to OpenAI's Quickstart, you can install the Codex CLI globally
#    with npm using: `npm install -g @openai/codex`【787858843003820†L392-L398】.
#    This requires Node.js and npm to be installed on your system. If Node is
#    unavailable, the script will print guidance instead of failing.
# -----------------------------------------------------------------------------

install_openai_codex() {
  if command -v codex >/dev/null 2>&1 && codex --version 2>/dev/null | grep -q "codex-cli"; then
    echo "OpenAI Codex CLI already installed (codex CLI detected). Skipping."
  else
    if command -v npm >/dev/null 2>&1; then
      echo "Installing OpenAI Codex CLI globally via npm..."
      npm install -g @openai/codex
      echo "OpenAI Codex CLI installation complete."
      echo "Note: set your OpenAI API key in the OPENAI_API_KEY environment variable to use the CLI【787858843003820†L392-L405】."
    else
      echo "npm (Node.js) is not installed; cannot install OpenAI Codex CLI."
      echo "Please install Node.js and npm, then rerun this script to install the Codex CLI."
    fi
  fi
}

# -----------------------------------------------------------------------------
# 4. Prompt user to select which Codex tool(s) to install
# -----------------------------------------------------------------------------

# Only prompt if running in an interactive terminal
if [ -t 1 ]; then
  echo "\nCodex installation options:"
  echo "  1) Install Codex Storage (decentralised storage node)"
  echo "  2) Install OpenAI Codex CLI (local coding agent)"
  echo "  3) Install both"
  echo "  4) Skip Codex installation"
  read -p "Select an option [1-4]: " choice
  case "$choice" in
    1) install_codex_storage ;;
    2) install_openai_codex ;;
    3) install_codex_storage; install_openai_codex ;;
    *) echo "Skipping Codex installation." ;;
  esac
else
  # Non-interactive mode: install nothing by default
  echo "Non-interactive session detected; skipping Codex installation."
fi

echo "Environment setup complete."
