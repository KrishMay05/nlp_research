#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
VENV="${QUOTE_POWER_VENV:-$PWD/.venv}"
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -r requirements.lock
"$VENV/bin/python" -m pip install --no-deps -e .
if [ ! -d data/upstream/.git ]; then
  git clone --no-checkout https://github.com/kandicew/social-reasoning.git data/upstream
  git -C data/upstream checkout --detach fed11b69765b4a398ce4a985adedfb65ec3098ab
fi
EXPECTED=fed11b69765b4a398ce4a985adedfb65ec3098ab
ACTUAL=$(git -C data/upstream rev-parse HEAD)
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo 'Upstream checkout differs from pinned release. Preserve local work and inspect before switching.' >&2
  exit 1
fi
"$VENV/bin/quote-power" prepare --upstream data/upstream
"$VENV/bin/python" -m pytest -q
"$VENV/bin/quote-power" evaluate
