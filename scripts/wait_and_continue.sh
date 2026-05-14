#!/usr/bin/env bash
# Wait for doubles coarse (PID in $1) to finish, then run stages 2-4.
# Usage: bash scripts/wait_and_continue.sh <PID>
set -e
PID=${1:?usage: $0 <doubles_coarse_PID>}
echo "[wait_and_continue] Waiting for PID $PID..."
while kill -0 "$PID" 2>/dev/null; do sleep 10; done
echo "[wait_and_continue] PID $PID done. Starting stages 2-4 at $(date)"
bash "$(dirname "$0")/run_stages_2_4.sh" 2>&1 | tee logs/run_stages_2_4.log
