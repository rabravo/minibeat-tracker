#!/usr/bin/env bash
# setup_server.sh — create the minibeat-hpc conda environment.
#
# Usage:
#   ./setup_server.sh
#
# The environment is created at:
#   cardio-tracker/envs/minibeat-hpc/
#
# After setup, activate with:
#   conda activate minibeat-hpc
#   # or: source cardio-tracker/envs/minibeat-hpc/bin/activate
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ENV_PREFIX="$SCRIPT_DIR/../envs/minibeat-hpc"

if [[ -d "$ENV_PREFIX" ]]; then
  echo "Environment already exists at $ENV_PREFIX"
  echo "To rebuild it: rm -rf $ENV_PREFIX && ./setup_server.sh"
  exit 0
fi

echo "Creating conda environment minibeat-hpc..."
echo "  Target prefix: $ENV_PREFIX"
echo ""

conda env create \
  --file environment.yml \
  --prefix "$ENV_PREFIX"

echo ""
echo "Environment created at $ENV_PREFIX"
echo ""
echo "To start the server:"
echo "  conda activate minibeat-hpc"
echo "  ./run_server.sh"
