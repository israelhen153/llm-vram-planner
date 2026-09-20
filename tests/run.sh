#!/usr/bin/env bash
# Run the whole suite. Needs node, python3 and reportlab — reportlab only because
# tests/report.test.py imports generate_report.py as a real module. The project
# deliberately has no build step, and the tests keep it that way.
set -euo pipefail
cd "$(dirname "$0")/.."

# First, deliberately. This one reports what the rest of the suite is verifying, and
# run.sh is `set -e`: last, it printed only when everything else already passed, which
# is exactly when nobody needs telling.
echo "== benchmark coverage (benchmarks/data.json) =="
python3 tests/coverage.test.py

echo "== model math (index.html) =="
node tests/model.test.js

echo "== JS/Python parity (index.html vs generate_report.py) =="
python3 tests/parity.test.py

echo "== report generator cfg builders (generate_report.py) =="
python3 tests/report.test.py

echo "== generated data blocks (tools/sync_data.py) =="
python3 tests/sync.test.py

echo "== price-refresh sanity checks (tools/price_check.py) =="
python3 tests/price_check.test.py

echo "== published images (tools/make_assets.py) =="
python3 tests/assets.test.py

echo "All suites passed."
