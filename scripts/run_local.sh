#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv. Create one with:"
  echo "  python -m venv .venv"
  exit 1
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"

python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt

if [[ -f "requirements-dev.txt" ]]; then
  python -m pip install -q -r requirements-dev.txt
fi

if [[ ! -f ".env" ]]; then
  echo "Missing .env. Create it with:"
  echo "  cp .env.example .env"
  exit 1
fi

PORT="${PORT:-8000}"

port_open() {
  local port="$1"
  python - <<PY
import socket
s = socket.socket()
s.settimeout(0.2)
try:
    s.connect(("127.0.0.1", int("$port")))
    print("open")
except Exception:
    print("closed")
finally:
    s.close()
PY
}

if [[ "$(port_open "$PORT")" == "open" ]]; then
  # Find a free port in a small range to avoid the common "8000 already in use" issue.
  for p in 8001 8002 8003 8004 8005 8010 8020; do
    if [[ "$(port_open "$p")" == "closed" ]]; then
      PORT="$p"
      break
    fi
  done
fi

echo "Starting ArmchairGPT at http://localhost:${PORT}"
exec uvicorn api:app --reload --host 127.0.0.1 --port "${PORT}"

