#!/usr/bin/env bash
# Run the whole suite. Needs node, python3, reportlab and pyyaml. reportlab only
# because tests/report.test.py imports generate_report.py as a real module; pyyaml
# only because tests/workflow.test.py judges what GitHub will do with a workflow
# file, and a guard that reads YAML differently from YAML is wrong by construction
# — a hand-rolled reader there passed a workflow that reinstated the bug it was
# written to prevent. The project deliberately has no build step, and the tests
# keep it that way.
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

echo "== CI workflows (.github/workflows) =="
python3 tests/workflow.test.py

echo "== the sabotage corpus can reach the engine (tests/sabotage/anchors.py) =="
python3 tests/corpus.test.py

echo "== published images (tools/make_assets.py) =="
python3 tests/assets.test.py

echo "All suites passed."
