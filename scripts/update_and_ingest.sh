#!/usr/bin/env bash
set -euo pipefail

NOTES_DIR="/home/rag/notes_repo"
APP_DIR="/home/rag/rag_app"
ENV_FILE="/etc/rag.env"

echo "[update_and_ingest] $(date -Is) starting"

cd "$NOTES_DIR"
git fetch origin

LOCAL="$(git rev-parse HEAD)"
REMOTE="$(git rev-parse @{u} 2>/dev/null || true)"

# Ensure upstream is set (first run)
if [[ -z "$REMOTE" ]]; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  git branch --set-upstream-to=origin/main "$BRANCH" >/dev/null 2>&1 || true
  git fetch origin
  REMOTE="$(git rev-parse @{u})"
fi

if [[ "$LOCAL" == "$REMOTE" ]]; then
  echo "[update_and_ingest] No changes in notes repo. Skipping ingest."
  exit 0
fi

echo "[update_and_ingest] Changes detected. Pulling notes..."
git pull --ff-only

echo "[update_and_ingest] Running ingest..."
cd "$APP_DIR"

set -a
source "$ENV_FILE"
set +a

source "$APP_DIR/venv/bin/activate"
python -m ingest.ingest

echo "[update_and_ingest] Restarting rag service..."
sudo /bin/systemctl restart rag

echo "[update_and_ingest] Done."
