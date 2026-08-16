#!/usr/bin/env bash
# run_server.sh — launch the MiniBeat HPC web controller on this node.
#
# The server binds to 127.0.0.1 by default, so you must open an SSH tunnel
# from your laptop to reach it:
#
#     ssh -N -L 8766:localhost:8766 user@cluster.example.edu
#     open http://localhost:8766/
#
# Override host, port, or job data directory with env vars:
#
#     PORT=9000 DATA_ROOT=/scratch/$USER/minibeat-jobs ./run_server.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Prefer the minibeat-hpc conda env; fall back to whatever Python is active
HPC_ENV_PYTHON="$SCRIPT_DIR/../envs/minibeat-hpc/bin/python3"
PYTHON=${PYTHON:-${HPC_ENV_PYTHON:-python3}}

export HOST="${HOST:-127.0.0.1}"
export PORT="${PORT:-8766}"
export DATA_ROOT="${DATA_ROOT:-${SCRIPT_DIR}/WebJobs}"

mkdir -p "$DATA_ROOT"

echo "Starting MiniBeat HPC server on ${HOST}:${PORT}"
echo "  Job data root: ${DATA_ROOT}"

exec "$PYTHON" mb_server.py \
  --host "$HOST" \
  --port "$PORT" \
  --data-root "$DATA_ROOT" \
  "$@"
