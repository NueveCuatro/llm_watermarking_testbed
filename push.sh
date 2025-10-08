#!/usr/bin/env bash
# push.sh — add selected files/folders, commit, and push in one go
# Usage:
#   ./push.sh -a path1 path2 … -m "Your commit message"
# Example:
#   ./push.sh -a folder1/ folder2/ file1 -m "Fix broken links"

set -e  # Exit immediately on any error

### ---- Parse flags ---- ###
files=()
message=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--add)
      shift
      # Collect everything until the next flag or end of args
      while [[ $# -gt 0 && "$1" != -* ]]; do
        files+=("$1")
        shift
      done
      ;;
    -m|--message)
      shift
      message="$1"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 -a <paths…> -m \"commit message\""
      exit 0
      ;;
    *)
      echo "Unknown option or argument: $1"
      exit 1
      ;;
  esac
done

# Basic validation
if [[ ${#files[@]} -eq 0 ]]; then
  echo "Error: no paths supplied with -a / --add"
  exit 1
fi
if [[ -z $message ]]; then
  echo "Error: commit message required with -m / --message"
  exit 1
fi

### ---- Git workflow ---- ###
echo "→ Pulling latest changes…"
git pull

echo "→ Repository status:"
git status

echo "→ Adding paths: ${files[*]}"
git add "${files[@]}"

echo "→ Committing with message: \"$message\""
git commit -m "$message"

echo "→ Pushing to remote…"
git push

echo "✓ Done!"

git status