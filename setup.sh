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
SITE_CODE="${2:-}"
FILES=("index.html" "README.md" "CONTRIBUTING.md")

if [ "$USERNAME" = "$UPSTREAM_USER" ]; then
    echo "That is already the upstream account — nothing to rewrite."
    exit 1
fi

# Both values are interpolated into sed expressions. A & in the replacement
# means "the whole match" and a # ends the expression, so an unchecked argument
# corrupts every URL while the script reports success. GitHub usernames and
# GoatCounter site codes are both [A-Za-z0-9-] anyway.
for arg in "$USERNAME" ${SITE_CODE:+"$SITE_CODE"}; do
    if ! printf '%s' "$arg" | grep -qE '^[A-Za-z0-9][A-Za-z0-9-]*$'; then
        echo "Refusing '$arg': expected letters, digits and hyphens only."
        exit 1
    fi
done

# Rewrite the two forms the name appears in — the Pages URL and the repo URL —
# rather than every occurrence of the string. The analytics beacon points at
# <user>.goatcounter.com, which is a different service with a different account,
# and renaming it here would silently aim a fork at a site that does not exist.
# It is handled below, on its own terms.
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        # `|| true`: grep exits 1 when there is nothing left to rewrite, and
        # under `set -o pipefail` that killed the whole script on any second
        # run — including the re-run this script itself suggests.
        before=$(grep -o -e "${UPSTREAM_USER}\.github\.io" -e "github\.com/${UPSTREAM_USER}" "$f" | wc -l || true)
        sed -i -e "s#${UPSTREAM_USER}\.github\.io#${USERNAME}.github.io#g" \
               -e "s#github\.com/${UPSTREAM_USER}#github.com/${USERNAME}#g" "$f"
        echo "Updated $f ($before link(s))"
    fi
done

# GoatCounter analytics — already enabled, pointed at the upstream author's
# site. A fork must not report its traffic there: repoint it, or remove it.
if [ -n "$SITE_CODE" ]; then
    # Say what happened, not what was asked for: on a tree whose counter has
    # already been removed there is nothing to repoint, and claiming otherwise
    # is the same kind of false statement this whole section exists to end.
    if grep -q "data-goatcounter" index.html; then
        sed -i "s#${UPSTREAM_USER}.goatcounter.com#${SITE_CODE}.goatcounter.com#g" index.html
        echo "Pointed the page-view counter at ${SITE_CODE}.goatcounter.com"
    else
        echo "No page-view counter in index.html — nothing to point at ${SITE_CODE}.goatcounter.com."
        echo "      A previous run removed it; restore it from git history if you want one."
    fi
else
    # Remove rather than leave it: an unattended fork reporting to someone
    # else's analytics is the wrong default, and the README documents how to
    # put a counter back.
    # Every region at once, never one of them: the footer line and the README
    # section both tell the reader the page counts views, so leaving either
    # behind would make the fork claim something it no longer does — the same
    # defect as an undisclosed counter, running the other way.
    # sed rather than python3: this used to shell out to python3, which is not
    # present everywhere, and when it was missing the script died with the
    # links already rewritten and the counter still aimed upstream.
    removed=0
    for pair in "index.html ANALYTICS-BEACON" "index.html ANALYTICS-NOTICE" \
                "README.md ANALYTICS-SECTION" "README.md ANALYTICS-DOC"; do
        set -- $pair
        [ -f "$1" ] || continue
        if grep -q "$2:BEGIN" "$1"; then
            # ANALYTICS-DOC sits inside a sentence; the others own their lines.
            if [ "$2" = "ANALYTICS-DOC" ]; then
                sed -i "s#<!-- $2:BEGIN -->.*<!-- $2:END -->##" "$1"
            else
                sed -i "/$2:BEGIN/,/$2:END/d" "$1"
            fi
            removed=$((removed + 1))
        fi
    done
    if [ "$removed" -gt 0 ]; then
        echo "Removed the page-view counter and every claim about it ($removed region(s))."
    else
        echo "No page-view counter found — nothing to remove."
    fi
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
