#!/usr/bin/env bash
# Quick validation: baseline vs AIR CHAIR on 50 caption samples (greedy)
set -uo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=True

mkdir -p ./results/coco/_val_base ./results/coco/_val_air

gen() {
  local out="$1"; shift
  echo ">>> [$(date +%T)] generating: $out  args: $*"
  CUDA_VISIBLE_DEVICES=0 python3 -m eval_scripts.eval_caption_air \
    --model-path liuhaotian/llava-v1.5-7b \
    --image-folder ../dataset/coco/val2014 \
    --caption_file_path ../dataset/coco/annotations/captions_val2014.json \
    --answers-file "$out" --dataset coco --temperature 0 --conv-mode vicuna_v1 \
    --num_samples 50 --batch-size 1 --max_new_tokens 64 --seed 42 "$@" \
    > "${out%.jsonl}.genlog" 2>&1
  echo "    gen exit=$? lines=$(wc -l < "$out" 2>/dev/null)"
}

chair() {
  local out="$1"
  echo ">>> [$(date +%T)] CHAIR: $out"
  python3 eval_scripts/eval_utils/eval_chair.py \
    --annotation-dir ../dataset/coco/annotations \
    --answers-file "$out" \
    --caption_file captions_val2014.json \
    > "${out%.jsonl}.chairlog" 2>&1
  echo "    chair exit=$?"
}

gen ./results/coco/_val_base/captions.jsonl
gen ./results/coco/_val_air/captions.jsonl --air --air-beta 0.1 --air-layer-low 5 --air-layer-high 18
chair ./results/coco/_val_base/captions.jsonl
chair ./results/coco/_val_air/captions.jsonl

echo "================ SUMMARY ================"
echo "--- BASELINE ---"; tail -1 ./results/coco/_val_base/captions.chairlog
echo "--- AIR ---";      tail -1 ./results/coco/_val_air/captions.chairlog
echo "ALL DONE [$(date +%T)]"
