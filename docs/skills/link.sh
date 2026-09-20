#!/usr/bin/env bash
# Link every skill in this directory into .claude/skills/, where Claude Code looks.
#
# Why the indirection: .claude/ is gitignored (globally, on purpose — editor and
# agent config does not belong in a repo). But these files are project knowledge,
# not config: they name perfKey, vendor gating and which axes have hidden leaks in
# this engine. Kept only under .claude/ they would be untracked, undiffable, and
# would drift silently as the code they describe moves. So the real file lives here
# and is versioned; .claude/skills/ holds a symlink, which git never sees.
#
# Run after cloning. Re-running is safe.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(git -C "$HERE" rev-parse --show-toplevel)"
DEST="$ROOT/.claude/skills"
mkdir -p "$DEST"

linked=0
# Derived from the directory: a skill added later needs no edit here.
for d in "$HERE"/*/; do
  name="$(basename "$d")"
  [ -f "$d/SKILL.md" ] || continue
  ln -sfn "../../docs/skills/$name" "$DEST/$name"
  echo "  linked $name -> docs/skills/$name"
  linked=$((linked+1))
done
echo "$linked skill(s) linked into .claude/skills/"
