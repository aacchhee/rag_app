#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/rag/rag_app_qdrant"
BRANCH="qdrant"

echo "[rag_deploy] $(date -Is) starting"

# Pull latest as rag
sudo -u rag -H bash -lc "
set -euo pipefail
cd '$APP_DIR'
git fetch origin
git checkout '$BRANCH'
git pull --ff-only origin '$BRANCH'
"

# (Optional) install/update python deps if requirements changed
sudo -u rag -H bash -lc "
set -euo pipefail
cd '$APP_DIR'
source venv/bin/activate
pip install -U pip >/dev/null
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi
"

# Restart service
systemctl restart rag2

echo "[rag_deploy] $(date -Is) done"
