#!/usr/bin/env bash
set -euo pipefail

# Default threshold: 90 MiB (safe under GitHub's 100MB hard limit)
THRESHOLD_MIB="${1:-90}"
THRESHOLD_BYTES=$((THRESHOLD_MIB * 1024 * 1024))

# Set DRY_RUN=1 to preview without changes
DRY_RUN="${DRY_RUN:-0}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "❌ Not inside a git repository."
  exit 1
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

GITIGNORE="$REPO_ROOT/.gitignore"
touch "$GITIGNORE"

echo "🔎 Scanning for files >= ${THRESHOLD_MIB} MiB in working tree..."
mapfile -t BIG_FILES < <(
  # Exclude .git directory; follow regular files only
  find . -type f -not -path "./.git/*" -printf "%s\t%p\n" \
  | awk -v thr="$THRESHOLD_BYTES" '$1 >= thr {print $2}' \
  | sed 's|^\./||'
)

if [ "${#BIG_FILES[@]}" -eq 0 ]; then
  echo "✅ No files found above threshold."
  exit 0
fi

echo "⚠️ Found ${#BIG_FILES[@]} large file(s):"
for f in "${BIG_FILES[@]}"; do
  size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo "?")
  echo "  - $f  (${size} bytes)"
done

# Function to add a line to .gitignore if absent (exact match)
add_to_gitignore() {
  local pattern="$1"
  # Escape for grep fixed match
  if ! grep -Fxq "$pattern" "$GITIGNORE"; then
    if [ "$DRY_RUN" = "1" ]; then
      echo "DRY_RUN: would append to .gitignore: $pattern"
    else
      echo "$pattern" >> "$GITIGNORE"
      echo "➕ Added to .gitignore: $pattern"
    fi
  else
    echo "ℹ️ Already in .gitignore: $pattern"
  fi
}

# Determine which are tracked by git
echo "🔎 Checking which of these files are tracked by git..."
TRACKED_TO_REMOVE=()
for f in "${BIG_FILES[@]}"; do
  if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    TRACKED_TO_REMOVE+=("$f")
  fi
done

# Update .gitignore for all found big files (even untracked)
echo "📝 Updating .gitignore..."
for f in "${BIG_FILES[@]}"; do
  # Ignore the exact path (most reliable)
  add_to_gitignore "/$f"
done

# Remove tracked ones from index so they won't be pushed anymore
if [ "${#TRACKED_TO_REMOVE[@]}" -gt 0 ]; then
  echo "🧹 Removing ${#TRACKED_TO_REMOVE[@]} tracked large file(s) from git index (keep on disk)..."
  if [ "$DRY_RUN" = "1" ]; then
    for f in "${TRACKED_TO_REMOVE[@]}"; do
      echo "DRY_RUN: would run: git rm --cached -- \"$f\""
    done
  else
    # Remove in one go
    git rm --cached -- "${TRACKED_TO_REMOVE[@]}"
  fi
else
  echo "✅ None of the large files are tracked by git (no need to git rm --cached)."
fi

# Show status
echo ""
echo "📌 Git status (summary):"
git status -sb || true

echo ""
echo "✅ Done."
echo "Next steps:"
echo "  1) Review .gitignore changes"
echo "  2) Commit:"
echo "     git add .gitignore"
if [ "${#TRACKED_TO_REMOVE[@]}" -gt 0 ]; then
  echo "     git commit -m \"Ignore and untrack large files\""
else
  echo "     git commit -m \"Update .gitignore for large files\""
fi
echo "  3) Push:"
echo "     git push"
echo ""
echo "⚠️ Note:"
echo "If GitHub is rejecting because a >100MB file is already in git history,"
echo "you must rewrite history (git filter-repo / BFG)."
