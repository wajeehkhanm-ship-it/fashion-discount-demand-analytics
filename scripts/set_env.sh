#!/usr/bin/env bash
# Store a secret in .env without it appearing on screen or in shell history.
# Usage: scripts/set_env.sh VAR_NAME      (e.g. scripts/set_env.sh KAGGLE_API_TOKEN)
set -euo pipefail
cd "$(dirname "$0")/.."

name="${1:?usage: scripts/set_env.sh VAR_NAME}"
[ -f .env ] || cp .env.example .env

read -r -s -p "Paste the value for $name and press Enter (typing is hidden): " value
echo
if [ -z "$value" ]; then
  echo "Nothing entered; .env unchanged."
  exit 1
fi

tmp="$(mktemp)"
grep -v "^${name}=" .env > "$tmp" || true
printf '%s=%s\n' "$name" "$value" >> "$tmp"
mv "$tmp" .env
chmod 600 .env
echo "Saved $name to .env (value not shown; .env is git-ignored)."
