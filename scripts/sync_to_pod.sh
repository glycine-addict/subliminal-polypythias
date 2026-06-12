#!/usr/bin/env bash
# Push code to the VAST pod over scp. Reads host/port from `vastai ssh-url <id>`.
# Usage: scripts/sync_to_pod.sh [instance_id]
#   instance_id defaults to the contents of /tmp/vast_instance_id
set -euo pipefail

ID="${1:-$(cat /tmp/vast_instance_id)}"
URL=$(vastai ssh-url "$ID")           # ssh://root@HOST:PORT
HOSTPORT=${URL#ssh://root@}
HOST=${HOSTPORT%:*}
PORT=${HOSTPORT#*:}
KEY="$HOME/.ssh/id_ed25519"
REMOTE=/root/subliminal

echo "[sync] instance $ID -> $HOST:$PORT"
ssh -o StrictHostKeyChecking=accept-new -i "$KEY" -p "$PORT" "root@$HOST" "mkdir -p $REMOTE"

# Code only: no .git, no results, no PDFs (those stay local / come back separately).
for d in src experiments scripts; do
  [ -e "$d" ] && scp -q -i "$KEY" -P "$PORT" -r "$d" "root@$HOST:$REMOTE/"
done
scp -q -i "$KEY" -P "$PORT" requirements.txt "root@$HOST:$REMOTE/" 2>/dev/null || true

# Stamp the commit the code came from, so results written on the pod carry a real
# git hash instead of "unknown" (config.git_hash falls back to this file).
HASH=$(git rev-parse HEAD 2>/dev/null || echo unknown)
git diff --quiet 2>/dev/null || HASH="$HASH-dirty"
echo "$HASH" | ssh -i "$KEY" -p "$PORT" "root@$HOST" "cat > $REMOTE/GIT_HASH"

echo "[sync] done -> $REMOTE"
