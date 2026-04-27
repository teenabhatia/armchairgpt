#!/usr/bin/env bash
# Run the full agent evaluation surface: JSONL harness (offline) + pytest agent_eval.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv. Create: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt"
  exit 1
fi

# shellcheck disable=SC1091
source ".venv/bin/activate"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "== eval/run_eval.py (offline, all cases) =="
python3 eval/run_eval.py --mode offline

echo ""
echo "== pytest -m agent_eval =="
python3 -m pytest -m agent_eval -q
