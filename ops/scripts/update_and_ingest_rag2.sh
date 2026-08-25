#!/usr/bin/env bash
set -euo pipefail

NOTES_DIRS=("/home/rag/notes_repo" "/home/rag/ingmat3a_ntnu")
APP_DIR="/home/rag/rag_app_qdrant"
ENV_FILE="/etc/rag2.env"

echo "[rag_update_and_ingest] $(date -Is) starting"

changed=false
for NOTES_DIR in "${NOTES_DIRS[@]}"; do
    if [[ ! -d "$NOTES_DIR/.git" ]]; then
        echo "[rag_update_and_ingest] $NOTES_DIR not a git repo, skipping"
        continue
    fi

    read -r LOCAL REMOTE < <(/usr/bin/sudo -u rag /bin/bash -c "
cd '$NOTES_DIR'
/usr/bin/git fetch origin >/dev/null 2>&1 || true
LOCAL=\$(/usr/bin/git rev-parse HEAD)
if ! /usr/bin/git rev-parse @{u} >/dev/null 2>&1; then
  BRANCH=\$(/usr/bin/git rev-parse --abbrev-ref HEAD)
  /usr/bin/git branch --set-upstream-to=origin/main \"\$BRANCH\" >/dev/null 2>&1 || true
  /usr/bin/git fetch origin >/dev/null 2>&1 || true
fi
REMOTE=\$(/usr/bin/git rev-parse @{u} 2>/dev/null || echo \$LOCAL)
echo \"\$LOCAL \$REMOTE\"
")

    if [[ "$LOCAL" != "$REMOTE" ]]; then
        changed=true
        echo "[rag_update_and_ingest] Changes detected in $NOTES_DIR"
    fi
done

if ! $changed; then
    echo "[rag_update_and_ingest] No changes in any notes repo. Skipping ingest."
    exit 0
fi

echo "[rag_update_and_ingest] Pulling notes..."
for NOTES_DIR in "${NOTES_DIRS[@]}"; do
    if [[ -d "$NOTES_DIR/.git" ]]; then
        /usr/bin/sudo -u rag /bin/bash -c "cd '$NOTES_DIR' && /usr/bin/git pull --ff-only"
    fi
done

echo "[rag_update_and_ingest] Running ingest..."
/usr/bin/sudo -u rag /bin/bash -c "
set -a
source '$ENV_FILE'
set +a
cd '$APP_DIR'
source venv/bin/activate
python -m ingest.ingest
"

echo "[rag_update_and_ingest] Restarting rag2..."
/usr/bin/sudo /bin/systemctl restart rag2

echo "[rag_update_and_ingest] Done."
