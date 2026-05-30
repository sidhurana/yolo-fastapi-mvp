#!/usr/bin/env bash
# HTTPS server so phones can use LIVE camera (browsers block getUserMedia on http://).
set -euo pipefail
cd "$(dirname "$0")/.."

CERT_DIR="${CERT_DIR:-.certs}"
KEY="${CERT_DIR}/key.pem"
CERT="${CERT_DIR}/cert.pem"
PORT="${PORT:-8000}"

# Best-effort LAN IP for cert + instructions
LAN_IP=""
if command -v ipconfig >/dev/null 2>&1; then
  LAN_IP="$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)"
fi
if [[ -z "$LAN_IP" ]] && command -v hostname >/dev/null 2>&1; then
  LAN_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
fi
[[ -z "$LAN_IP" ]] && LAN_IP="127.0.0.1"

mkdir -p "$CERT_DIR"
if [[ ! -f "$KEY" || ! -f "$CERT" ]]; then
  echo "Generating certificate (SAN: localhost, ${LAN_IP}) …"
  openssl req -x509 -newkey rsa:2048 \
    -keyout "$KEY" -out "$CERT" -days 365 -nodes \
    -subj "/CN=${LAN_IP}" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:${LAN_IP}" 2>/dev/null \
    || openssl req -x509 -newkey rsa:2048 \
         -keyout "$KEY" -out "$CERT" -days 365 -nodes \
         -subj "/CN=${LAN_IP}"
fi

echo ""
echo "=== Live camera on phone ==="
echo "1. Keep this server running"
echo "2. On phone (same Wi‑Fi), open:"
echo "   https://${LAN_IP}:${PORT}"
echo "3. Accept the certificate warning once (Advanced → Proceed)"
echo "4. Allow camera when prompted"
echo ""

exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" \
  --ssl-keyfile "$KEY" --ssl-certfile "$CERT"
