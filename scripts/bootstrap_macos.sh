#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This project requires an Apple Silicon Mac." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.11-3.13 is required." >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 11) <= sys.version_info[:2] < (3, 14)):
    raise SystemExit(f"Python 3.11-3.13 is required; found {sys.version.split()[0]}")
PY

"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip

if [[ "${INSTALL_MLX:-0}" == "1" ]]; then
  .venv/bin/pip install -e ".[mac,dev]"
else
  .venv/bin/pip install -e ".[dev]"
fi

.venv/bin/gkr doctor
if [[ ! -f artifacts/authority.sqlite ]]; then
  .venv/bin/gkr ingest knowledge/demo_records.jsonl
fi
PYTHON_BIN=.venv/bin/python ./scripts/verify_gate0.sh
.venv/bin/gkr eval
.venv/bin/pytest

echo
echo "Local runtime ready. Activate it with: source .venv/bin/activate"
echo "Install MLX support with: INSTALL_MLX=1 ./scripts/bootstrap_macos.sh"
