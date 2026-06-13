#!/usr/bin/env bash
# Download POPEv2/dataset assets from RUCAIBox/POPE:
#   - images.zip (~21MB)
#   - annotations.json (not zipped)
# That GitHub dir has one zip + one json only (no second zip).
# Usage (from repo root or LLaVA/):
#   bash LLaVA/bash_scripts/download_popev2_dataset.sh
#   bash LLaVA/bash_scripts/download_popev2_dataset.sh /path/to/outdir
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="${1:-${ROOT}/dataset/POPEv2/dataset}"
mkdir -p "$DEST"
cd "$DEST"

BASE="https://raw.githubusercontent.com/RUCAIBox/POPE/main/POPEv2/dataset"
# If direct GitHub fails: export GITHUB_PROXY_PREFIX=https://ghfast.top/
PREFIX="${GITHUB_PROXY_PREFIX:-}"

fetch() {
  local url="$1" out="$2"
  if curl -fL --connect-timeout 15 --max-time 600 -o "$out.tmp" "${PREFIX}${url}" && mv "$out.tmp" "$out"; then
    return 0
  fi
  rm -f "$out.tmp"
  return 1
}

echo "Saving to: $DEST"

if ! fetch "$BASE/images.zip" "images.zip"; then
  echo "Download failed (PREFIX=${PREFIX:-empty}). Try:" >&2
  echo "  export GITHUB_PROXY_PREFIX='https://ghfast.top/'" >&2
  echo "  bash $0 \"$DEST\"" >&2
  exit 1
fi

fetch "$BASE/annotations.json" "annotations.json"

ls -la images.zip annotations.json
echo "Done."
