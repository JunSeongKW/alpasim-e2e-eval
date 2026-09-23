#!/usr/bin/env bash
# Close a working session so the next agent can pick it up: stamp HANDOFF.md,
# stage everything .gitignore allows, commit on the personal branch.
#
#   tools/handoff-commit.sh "<agent>" "[eval] one-line subject"
#
# The commit body is HANDOFF.md section 3 ("마지막 커밋 이후 바뀐 것"): write
# there what you did and why BEFORE running this, then it lands in git log too.
# Refuses on any branch that is not junseong/*, so upstream branches stay clean.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
AGENT="${1:?usage: handoff-commit.sh <agent-name> <commit subject>}"
SUBJECT="${2:?usage: handoff-commit.sh <agent-name> <commit subject>}"

branch="$(git rev-parse --abbrev-ref HEAD)"
[[ "$branch" == junseong/* ]] || { echo "refusing: on branch '$branch' (need junseong/*)" >&2; exit 2; }
[[ -f HANDOFF.md ]] || { echo "refusing: no HANDOFF.md" >&2; exit 2; }

stamp="$(date '+%Y-%m-%d %H:%M') KST ($AGENT)"
sed -i -E "s|^마지막 갱신: .*|마지막 갱신: ${stamp}|" HANDOFF.md

# Section 3 of HANDOFF.md becomes the commit body.
body="$(awk '/^## 3\./{f=1; next} /^## /{f=0} f' HANDOFF.md | sed '/^[[:space:]]*$/d')"
git add -A
if git diff --cached --quiet; then
    echo "nothing to commit (HANDOFF.md stamped only? it is staged too, so this means no change at all)"
    exit 0
fi
{
    echo "$SUBJECT"
    echo
    echo "$body"
    echo
    echo "Agent: $AGENT"
} | git commit -q -F -
echo "committed on $branch:"; git log --oneline -1
echo "changed files: $(git show --stat --oneline HEAD | tail -1)"
