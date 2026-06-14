#!/usr/bin/env bash
# Continuously refresh the World Cup 2026 predictions: pull the latest match
# results and regenerate the Excel + Markdown (and the Desktop copy) on a loop.
#
# Usage:  ./update.sh [interval_seconds]   (default 1800 = 30 min)
set -euo pipefail
cd "$(dirname "$0")"
INTERVAL="${1:-1800}"
echo "Refreshing World Cup 2026 predictions every ${INTERVAL}s. Ctrl-C to stop."
while true; do
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') ==="
  python run.py || echo "run failed; will retry next cycle"
  echo
  sleep "$INTERVAL"
done
