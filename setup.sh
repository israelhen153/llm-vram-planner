#!/usr/bin/env bash
# Fork setup: point a clone of this repo at your own GitHub account.
#
# The tool hardcodes its own home — GitHub links in the page footer, the
# contribute and issue links, the live URL in the README — because it is a
# single self-contained file with no build step and no config to read. This
# rewrites those to yours in one pass.
#
# Usage: ./setup.sh <github-username> [goatcounter-site-code]

set -euo pipefail

UPSTREAM_USER="israelhen153"

if [ $# -eq 0 ]; then
    echo "Usage: ./setup.sh <github-username> [goatcounter-site-code]"
    echo "Example: ./setup.sh johndoe"
    echo
    echo "Rewrites ${UPSTREAM_USER} to <github-username> in the tool, the README"
    echo "and the contributing guide, and repoints or removes the page-view counter."
    exit 1
fi

USERNAME="$1"
FILES=("index.html" "README.md" "CONTRIBUTING.md")

if [ "$USERNAME" = "$UPSTREAM_USER" ]; then
    echo "That is already the upstream account — nothing to rewrite."
    exit 1
fi

# Rewrite the two forms the name appears in — the Pages URL and the repo URL —
# rather than every occurrence of the string. The analytics beacon points at
# <user>.goatcounter.com, which is a different service with a different account,
# and renaming it here would silently aim a fork at a site that does not exist.
# It is handled below, on its own terms.
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        before=$(grep -o -e "${UPSTREAM_USER}\.github\.io" -e "github\.com/${UPSTREAM_USER}" "$f" | wc -l)
        sed -i -e "s#${UPSTREAM_USER}\.github\.io#${USERNAME}.github.io#g" \
               -e "s#github\.com/${UPSTREAM_USER}#github.com/${USERNAME}#g" "$f"
        echo "Updated $f ($before link(s))"
    fi
done

# GoatCounter analytics — already enabled, pointed at the upstream author's
# site. A fork must not report its traffic there: repoint it, or remove it.
if [ -n "${2:-}" ]; then
    sed -i "s#${UPSTREAM_USER}.goatcounter.com#$2.goatcounter.com#g" index.html
    echo "Pointed the page-view counter at $2.goatcounter.com"
else
    # Remove rather than leave it: an unattended fork reporting to someone
    # else's analytics is the wrong default, and the README documents how to
    # put a counter back.
    python3 - <<'PYEOF'
import re
# Both regions, never one: the footer notice tells the reader the page counts
# views, so leaving it behind after deleting the beacon would make the fork
# claim something it no longer does — the exact failure this notice exists to
# fix, running the other way.
src = open("index.html").read()
new = src
for tag in ("ANALYTICS-BEACON", "ANALYTICS-NOTICE"):
    new = re.sub(r"[ \t]*<!-- %s:BEGIN.*?%s:END -->\n?" % (tag, tag), "", new, flags=re.S)
open("index.html", "w").write(new)
print("Removed the page-view counter and the notice that described it."
      if new != src else "No page-view counter found — nothing to remove.")
PYEOF
    echo "      Pass a GoatCounter site code as the 2nd argument to keep one."
    echo "      See the Analytics section of README.md."
fi

echo ""
echo "Now run the suite — it checks the generated data blocks as well as the math:"
echo "  ./tests/run.sh"
echo ""
echo "Done. Next steps:"
echo "  1. git init && git add -A && git commit -m 'Initial commit'"
echo "  2. git remote add origin git@github.com:${USERNAME}/llm-vram-planner.git"
echo "  3. git push -u origin main"
echo "  4. Go to repo Settings → Pages → Source: main branch → Save"
echo "  5. Live at: https://${USERNAME}.github.io/llm-vram-planner/"
