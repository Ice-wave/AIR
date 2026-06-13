#!/usr/bin/env bash
# Download HF dataset Shengcao1006/MMHal-Bench to ../dataset/mmhal (matches decoding.sh default).
# Mirror for CN networks: export HF_ENDPOINT=https://hf-mirror.com
set -euo pipefail
cd "$(dirname "$0")/../.."

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
DEST="${1:-dataset/mmhal}"
mkdir -p "$DEST"

python3 << PY
import os
from huggingface_hub import snapshot_download

dest = os.path.abspath("${DEST}")
os.makedirs(dest, exist_ok=True)
snapshot_download(
    repo_id="Shengcao1006/MMHal-Bench",
    repo_type="dataset",
    local_dir=dest,
)
print("Saved to:", dest)
PY

echo "Template path: $(realpath "$DEST")/response_template.json"
