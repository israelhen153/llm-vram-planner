#!/usr/bin/env bash
# Runs the suites that judge a sabotage (assets.test.py hashes index.html
# and goes red on any edit, so it is excluded). Prints one line per suite and
# the first FAIL lines of any red one.
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
label="${1:-}"
echo "=== $label ==="
declare -A CMD=(
  [model]="node tests/model.test.js"
  [parity]="python3 tests/parity.test.py"
  [report]="python3 tests/report.test.py"
  [sync]="python3 tests/sync.test.py"
  [price]="python3 tests/price_check.test.py"
  [workflow]="python3 tests/workflow.test.py"
)
for s in model parity report sync price workflow; do
  out=$(${CMD[$s]} 2>&1); rc=$?
  summary=$(echo "$out" | grep -E '^[0-9]+ passed, [0-9]+ failed' | tail -1)
  if [ $rc -eq 0 ]; then
    echo "  $s: GREEN ($summary)"
  else
    echo "  $s: RED ($summary, rc=$rc)"
    echo "$out" | grep -E 'FAIL|Error|error' | head -${2:-4} | sed 's/^/      /'
  fi
done
