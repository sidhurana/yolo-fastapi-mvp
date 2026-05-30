#!/usr/bin/env bash
# Fresh Ubuntu 22.04/24.04 install (VM, EC2, etc.)
# Usage: curl -fsSL ... | bash   OR   ./scripts/install.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_DIR"

echo "==> System packages"
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq \
    python3 python3-venv python3-pip \
    libgl1 libglib2.0-0 \
    git curl
  # Docker (optional but recommended for Postgres)
  if ! command -v docker >/dev/null 2>&1; then
    sudo apt-get install -y -qq docker.io docker-compose-plugin || true
    sudo usermod -aG docker "$USER" 2>/dev/null || true
    echo "Note: log out/in or run 'newgrp docker' to use docker without sudo."
  fi
fi

echo "==> Python virtualenv"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo "==> Configuration"
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

mkdir -p weights uploads/events

echo "==> PostgreSQL (Docker)"
if command -v docker >/dev/null 2>&1; then
  docker compose up db -d
  echo "Waiting for Postgres..."
  sleep 5
else
  echo "Docker not found. Install Postgres manually and set DATABASE_URL in .env"
fi

echo "==> Database setup"
python scripts/setup_db.py

echo "==> Download YOLO plant model (~457 MB, one-time)"
python scripts/download_model.py --preset plant_disease

echo ""
echo "============================================"
echo "Install complete."
echo ""
echo "Start server:"
echo "  cd $REPO_DIR"
echo "  source .venv/bin/activate"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "URLs (replace with your server IP):"
echo "  Camera:  http://<SERVER_IP>:8000"
echo "  Admin:   http://<SERVER_IP>:8000/admin"
echo "  API docs http://<SERVER_IP>:8000/docs"
echo ""
echo "HTTPS live camera: ./scripts/run_https.sh"
echo "============================================"
