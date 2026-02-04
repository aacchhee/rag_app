#!/usr/bin/env bash
set -euo pipefail

NOTES_DIR="/home/rag/notes_repo"
APP_DIR="/home/rag/rag_app"
ENV_FILE="/etc/rag.env"

echo "[update_and_ingest] $(date -Is) starting"

# Update notes repo as user rag (so permissions stay consistent)
sudo -u rag bash -lc "
cd '$NOTES_DIR'
git fetch origin
LOCAL=\$(git rev-parse HEAD)
REMOTE=\$(git rev-parse @{u} 2>/dev/null || true)

if [[ -z \"\$REMOTE\" ]]; then
  BRANCH=\$(git rev-parse --abbrev-ref HEAD)
  git branch --set-upstream-to=origin/main \"\$BRANCH\" >/dev/null 2>&1 || true
  git fetch origin
  REMOTE=\$(git rev-parse @{u})
fi

if [[ \"\$LOCAL\" == \"\$REMOTE\" ]]; then
  echo '[update_and_ingest] No changes in notes repo. Skipping ingest.'
  exit 0
fi

echo '[update_and_ingest] Changes detected. Pulling notes...'
git pull --ff-only
"

# Run ingest as user rag (needs NTNU-only embeddings access, which this VM has)
sudo -u rag bash -lc "
set -a
source '$ENV_FILE'
set +a
cd '$APP_DIR'
source venv/bin/activate
python -m ingest.ingest
"

# Restart service (this script is meant to be run by a sudo-capable user)
sudo /bin/systemctl restart rag

echo "[update_and_ingest] Done."
