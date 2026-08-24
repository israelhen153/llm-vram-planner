#!/usr/bin/env bash
# Quick setup: replaces YOURUSERNAME placeholder across the project
# Usage: ./setup.sh your-github-username

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: ./setup.sh <github-username>"
    echo "Example: ./setup.sh johndoe"
    exit 1
fi

USERNAME="$1"
FILES=("index.html" "README.md" "CONTRIBUTING.md")

for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        sed -i "s/YOURUSERNAME/$USERNAME/g" "$f"
        echo "Updated $f"
    fi
done

# GoatCounter analytics — already enabled, pointed at the author's site
# A fork must not report its traffic there. Repoint it or remove it.
if [ -n "${2:-}" ]; then
    sed -i "s#israelhen153.goatcounter.com#$2.goatcounter.com#g" index.html
    echo "Pointed GoatCounter at $2.goatcounter.com"
else
    echo ""
    echo "NOTE: index.html counts page views via israelhen153.goatcounter.com."
    echo "      Pass your own site code as the 2nd argument to repoint it, or delete"
    echo "      the <script data-goatcounter=...> block at the end of index.html."
    echo "      See the Analytics section of README.md."
fi

echo ""
echo "Done. Next steps:"
echo "  1. git init && git add -A && git commit -m 'Initial commit'"
echo "  2. git remote add origin git@github.com:${USERNAME}/llm-vram-planner.git"
echo "  3. git push -u origin main"
echo "  4. Go to repo Settings → Pages → Source: main branch → Save"
echo "  5. Live at: https://${USERNAME}.github.io/llm-vram-planner/"
