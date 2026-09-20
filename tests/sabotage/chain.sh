#!/usr/bin/env bash
# Run sabotage drivers back to back and report what survived.
#
# Usage:  tests/sabotage/chain.sh [driver ...]     (no args = every sab*.py)
#         LOGDIR=path tests/sabotage/chain.sh      (default: tmp/sabotage, gitignored)
#
# A driver that prints "N caught, 0 survived" found no gap. A survivor is a
# sabotage the suite did not notice, which is the finding — not an error here.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
cd "$ROOT"

LOGDIR="${LOGDIR:-tmp/sabotage}"
mkdir -p "$LOGDIR"

if [ "$#" -gt 0 ]; then
  drivers=("$@")
else
  # Derived, not enumerated: a driver added later is picked up without editing
  # this file, which is how the round-2 list went stale the first time.
  # Both extensions — the first version of this globbed *.py only and silently
  # skipped sab3.sh, which is the same failure it was written to prevent.
  mapfile -t drivers < <(cd "$HERE" && ls sab*.py sab*.sh 2>/dev/null | sed 's/\.\(py\|sh\)$//' | sort -V)
fi

# The drivers restore with `git checkout --`, which restores the index rather
# than unsaved edits. Refusing to start on a dirty tree is what keeps a sabotage
# run from eating work in progress.
if [ -n "$(git status --porcelain)" ]; then
  echo "refusing to run: working tree is not clean (commit first — the drivers restore from the index)" >&2
  git status --short >&2
  exit 1
fi

fail=0
for d in "${drivers[@]}"; do
  log="$LOGDIR/$d.log"
  printf '%-10s ' "$d"
  if [ -f "$HERE/$d.py" ]; then "$HERE/$d.py" > "$log" 2>&1; else bash "$HERE/$d.sh" > "$log" 2>&1; fi; rc=$?
  echo "driver exit $rc" >> "$log"
  # Two output shapes, because not every driver uses sab.py's summary: the shared
  # machinery prints "N caught, M survived", while a single-sabotage driver prints
  # only "<-- SURVIVED" per survivor and no summary at all. Keying on the summary
  # alone reports a clean single-sabotage run as a survivor, which is the kind of
  # false alarm that teaches people to ignore the runner.
  summary="$(grep -oE '[0-9]+ caught, [0-9]+ survived' "$log" | tail -1)"
  survivors="$(grep -c -- '<-- SURVIVED' "$log")"
  if [ "$rc" -ne 0 ]; then
    echo "driver errored (exit $rc) -> $log"; fail=1
  elif [ "$survivors" -gt 0 ] || { [ -n "$summary" ] && ! grep -q '0 survived' <<<"$summary"; }; then
    echo "SURVIVOR(S) -> $log"; fail=1
  else
    echo "${summary:-caught, none survived}  -> $log"
  fi
done

if [ -n "$(git status --porcelain)" ]; then
  echo "tree not clean after the run — a driver did not restore:" >&2
  git status --short >&2
  exit 1
fi
exit $fail
