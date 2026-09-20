#!/usr/bin/env bash
# M13: a perfKey typo on one catalog row, propagated through tools/sync_data.py into
# both generated blocks — the shape of a real contributor mistake.
# M14: the same, but the typo row is the default card (a100-40).
#
# Reports in the corpus' vocabulary: "red <name>" when a suite noticed, and
# "GREEN <name>   <-- SURVIVED" when none did. An earlier version called a helper
# that no longer existed, ran no suite at all, and still exited 0.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$(git -C "$HERE" rev-parse --show-toplevel)"

[ -x "$HERE/suites.sh" ] || { echo "missing $HERE/suites.sh — cannot judge anything" >&2; exit 2; }

# Same reason the Python drivers prove a green baseline: a suite already red for an
# unrelated reason makes every sabotage read as caught, silently.
if "$HERE/suites.sh" "baseline (no sabotage applied)" 1 2>&1 | grep -q 'RED'; then
  echo "refusing to judge: a suite is already red on the unmodified tree" >&2
  exit 2
fi

caught=0; survived=0
for slug in l40s-48 a100-40; do
  name="M13/M14 $slug.perfKey = 'nvdia', synced into both engines"
  python3 - "$slug" <<'PY' || { echo "  ERROR  $name (could not apply)" >&2; exit 2; }
import sys
slug = sys.argv[1]
p = "data/gpus.json"
s = open(p).read()
line = [l for l in s.splitlines() if l.strip().startswith(f'"{slug}":')][0]
assert line.count('"perfKey": "nvidia"') == 1, f"{slug}: expected exactly one perfKey"
open(p, "w").write(s.replace(line, line.replace('"perfKey": "nvidia"', '"perfKey": "nvdia"')))
PY
  python3 tools/sync_data.py > /dev/null 2>&1
  out="$("$HERE/suites.sh" "$name" 3 2>&1)"
  git checkout -- data/gpus.json index.html generate_report.py
  if grep -q 'RED' <<<"$out"; then
    caught=$((caught+1))
    echo "  red    $name"
    grep -E '^\s+(model|parity|report|sync|price): RED' <<<"$out" | sed 's/^/       /'
  else
    survived=$((survived+1))
    echo "  GREEN  $name   <-- SURVIVED"
  fi
done

[ -z "$(git status --porcelain)" ] || { echo "tree not clean after restore" >&2; exit 2; }
echo
echo "$caught caught, $survived survived"
