#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VIRTUAL_ENV_PROMPT="nts_chat2" source "$ROOT_DIR/venv/bin/activate"

exec "$ROOT_DIR/venv/bin/python" run.py "$@"
