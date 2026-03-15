#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python -m pip install --upgrade pip
python -m pip install -e "$ROOT"
python -m pip install \
  -e "$ROOT/src/sgs_discovery/vendor/giverny/givernylocal" \
  -e "$ROOT/src/sgs_discovery/vendor/giverny/giverny"

echo "Installed SGS discovery package and vendored giverny packages into the active environment."
