#!/usr/bin/env bash
# M13: a perfKey typo on one catalog row, propagated through the sync tool into
# both generated blocks — the shape of a real contributor mistake.
# M14: the same, but the typo row is the default card (a100-40).
set -u
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
for slug in l40s-48 a100-40; do
  echo "=== M13/M14: data/gpus.json ${slug}.perfKey = 'nvdia', then tools/sync_data.py ==="
  python3 - "$slug" <<'EOF'
import sys, re
slug = sys.argv[1]
p = "data/gpus.json"
s = open(p).read()
line = [l for l in s.splitlines() if l.strip().startswith(f'"{slug}":')][0]
assert line.count('"perfKey": "nvidia"') == 1
s = s.replace(line, line.replace('"perfKey": "nvidia"', '"perfKey": "nvdia"'))
open(p, "w").write(s)
EOF
  python3 tools/sync_data.py
  git status --porcelain
  bash tmp/suites.sh "after typo on $slug" 3
  git checkout -- data/gpus.json index.html generate_report.py
  git status --porcelain
done
