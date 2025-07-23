#!/bin/bash
# Additional setup for simulation tools
set -e

# Ensure base environment
./install.sh

# Install FEniCS and gmsh if available
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update && sudo apt-get install -y fenics gmsh
else
  echo "apt-get not found; please install FEniCS and gmsh manually."
fi

echo "Research tools installation complete."
