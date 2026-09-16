#!/usr/bin/env bash
# Run the whole suite. No dependencies beyond node and python3 — the project
# deliberately has no build step, and the tests keep it that way.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== model math (index.html) =="
node tests/model.test.js

echo "== JS/Python parity (index.html vs generate_report.py) =="
python3 tests/parity.test.py

echo "== report generator cfg builders (generate_report.py) =="
python3 tests/report.test.py

echo "== generated data blocks (tools/sync_data.py) =="
python3 tests/sync.test.py

echo "== published images (tools/make_assets.py) =="
python3 tests/assets.test.py

echo "All suites passed."
