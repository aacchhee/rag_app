#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/ache/notes_repo"
APP_DIR="/home/ache/rag_app"
ENV_FILE="/etc/rag.env"

echo "[update_and_ingest] $(date -Is) starting"

cd "$REPO_DIR"

# Fetch updates
git fetch origin

# Determine current branch and upstream (handles first-time)
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
UPSTREAM="$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null || true)"

if [[ -z "$UPSTREAM" ]]; then
  # If no upstream set, attempt main
  git branch --set-upstream-to=origin/main "$BRANCH" >/dev/null 2>&1 || true
  git fetch origin
fi

LOCAL="$(git rev-parse HEAD)"
REMOTE="$(git rev-parse @{u} 2>/dev/null || echo "$LOCAL")"

if [[ "$LOCAL" == "$REMOTE" ]]; then
  echo "[update_and_ingest] No changes. Skipping ingest."
  exit 0
fi

echo "[update_and_ingest] Changes detected. Pulling..."
git pull --ff-only

echo "[update_and_ingest] Running ingest..."
cd "$APP_DIR"

# Load env vars for embedding + vector store path
set -a
source "$ENV_FILE"
set +a

source "$APP_DIR/venv/bin/activate"
python -m ingest.ingest

echo "[update_and_ingest] Restarting rag service..."
sudo systemctl restart rag

echo "[update_and_ingest] Done."
