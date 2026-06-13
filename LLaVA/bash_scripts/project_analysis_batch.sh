#!/usr/bin/env bash
# Patch→vocab projection analysis on a COCO subset (eval_scripts/project_analysis.py).
# Use captions_eval_results.json from the same model, or CHAIR words won't align with captions.
#
# Typical flow (defaults match v1.5-7b + decoding.sh result_path):
#   1) Run decoding.sh → captions.jsonl + eval_chair → captions_eval_results.json
#   2) Run this script: project_analysis.py → batch_summary.json, then correlate_chair_patch_hits.py
#      (or export MODEL_PATH / RESULT_TAG, etc.)
#
set -euo pipefail
cd "$(dirname "$0")/.."

# Same as decoding.sh: HF mirror when huggingface.co is unreachable; use HF_HUB_OFFLINE=True if cached
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-False}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# LLaVA-v1.5-7B (HF: liuhaotian/llava-v1.5-7b)
MODEL_PATH="${MODEL_PATH:-liuhaotian/llava-v1.5-7b}"
MODEL_BASE="${MODEL_BASE:-}"

# Match decoding.sh result_path: baseline ${model_name}_n500; AIR ${model_name}_air*_n500 (override via RESULT_TAG)
RESULT_TAG="${RESULT_TAG:-llava-v1.5-7b_n500}"
CHAIR_RESULTS="${CHAIR_RESULTS:-./results/coco/${RESULT_TAG}/captions_eval_results.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-../dataset/coco/val2014}"

# Define before OUTPUT_DIR; default dir includes top_k so different TOP_K values don't overwrite
TOP_K="${TOP_K:-40}"
SCORE_MODE="${SCORE_MODE:-lm_head}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/project_analysis/batch_${RESULT_TAG}_topk${TOP_K}}"

BATCH_SUMMARY="${OUTPUT_DIR}/batch_summary.json"
# CHAIR × patch correlation (summary also printed to stdout); set false to skip step 2
RUN_CHAIR_CORRELATE="${RUN_CHAIR_CORRELATE:-true}"
CORRELATE_JSON="${CORRELATE_JSON:-${OUTPUT_DIR}/chair_patch_correlation.json}"

# COCO annotation dir (optional); if unset, batch_summary annotation_summary is null
ANNOTATION_DIR="${ANNOTATION_DIR:-../dataset/coco/annotations}"

MAX_IMAGES="${MAX_IMAGES:-}"
EXTRA_ARGS=()
if [[ -n "${MAX_IMAGES}" ]]; then
  EXTRA_ARGS+=(--max-images "${MAX_IMAGES}")
fi
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-existing --save-per-image)
fi

if [[ ! -f "${CHAIR_RESULTS}" ]]; then
  echo "CHAIR results not found: ${CHAIR_RESULTS}" >&2
  echo "Run decoding.sh (eval_caption_air + eval_chair) for this model, or set CHAIR_RESULTS to an existing captions_eval_results.json." >&2
  exit 1
fi

ANN_ARGS=()
if [[ -n "${ANNOTATION_DIR}" && -d "${ANNOTATION_DIR}" ]]; then
  ANN_ARGS+=(--annotation-dir "${ANNOTATION_DIR}")
fi

CMD=(
  python3 eval_scripts/project_analysis.py
  --model-path "${MODEL_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --image-folder "${IMAGE_FOLDER}"
  --chair-results-file "${CHAIR_RESULTS}"
  --top-k "${TOP_K}"
  --score-mode "${SCORE_MODE}"
  "${ANN_ARGS[@]}"
  "${EXTRA_ARGS[@]}"
)
if [[ -n "${MODEL_BASE}" ]]; then
  CMD+=(--model-base "${MODEL_BASE}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"

if [[ "${RUN_CHAIR_CORRELATE}" == "true" ]]; then
  if [[ ! -f "${BATCH_SUMMARY}" ]]; then
    echo "batch_summary not found, skipping correlate: ${BATCH_SUMMARY}" >&2
    exit 1
  fi
  CORR_CMD=(
    python3 eval_scripts/correlate_chair_patch_hits.py
    --chair-results "${CHAIR_RESULTS}"
    --batch-summary "${BATCH_SUMMARY}"
    --output-json "${CORRELATE_JSON}"
  )
  echo "Running: ${CORR_CMD[*]}"
  "${CORR_CMD[@]}"
else
  echo "Skip correlate_chair_patch_hits.py (RUN_CHAIR_CORRELATE=${RUN_CHAIR_CORRELATE})"
fi
