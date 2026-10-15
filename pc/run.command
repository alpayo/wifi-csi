#!/bin/bash
# SGP SGPCard Command — one-click launcher (macOS/Linux)
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Setting up (first run)…"
  python3 -m venv .venv
  ./.venv/bin/pip install -q --upgrade pip
  ./.venv/bin/pip install -q -r requirements.txt
fi
echo "Make sure your computer is connected to the card's WiFi (SGP_SGPCARD_xxxx)."
./.venv/bin/python sgpcard.py "$@"
