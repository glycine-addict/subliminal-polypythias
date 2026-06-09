#!/usr/bin/env bash
# Pull results/ from the VAST pod back to local. Usage: scripts/pull_results.sh [instance_id]
set -euo pipefail

ID="${1:-$(cat /tmp/vast_instance_id)}"
URL=$(vastai ssh-url "$ID")
HOSTPORT=${URL#ssh://root@}
HOST=${HOSTPORT%:*}
PORT=${HOSTPORT#*:}
KEY="$HOME/.ssh/id_ed25519"

mkdir -p results
echo "[pull] $HOST:$PORT:/root/subliminal/results -> ./results"
scp -q -i "$KEY" -P "$PORT" -r "root@$HOST:/root/subliminal/results/." results/ 2>/dev/null || {
  echo "[pull] no results yet"; exit 0;
}
ls -la results/ | tail -10
