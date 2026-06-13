#!/usr/bin/env bash
# Always cd to LLaVA root first (relative paths like eval_chair.py break otherwise).
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=False
export HF_ENDPOINT=https://hf-mirror.com

model_name=llava-v1.5-7b
model_path=liuhaotian/llava-v1.5-7b
# LLaVA-v1.6-Vicuna-7B: swap the three lines below; use conv llava_v1 (matches run_llava.py v1 name heuristic)
# model_name=llava-v1.6-vicuna-7b
# model_path=liuhaotian/llava-v1.6-vicuna-7b
# conv_mode=llava_v1
# model_name=llava-v1.5-13b
# model_path=liuhaotian/llava-v1.5-13b
# model_name=llava-v1.6-34b
# model_path=liuhaotian/llava-v1.6-34b
dataset=coco
data_path=../dataset
num_samples=500
batch_size=4

conv_mode="${conv_mode:-vicuna_v1}"
# max_new_tokens (optional):
#   - export max_new_tokens=256           → same value for all four stages
#   - or per-stage: max_new_tokens_mmhal, max_new_tokens_caption, max_new_tokens_pope, max_new_tokens_popev2
#   - per-stage overrides max_new_tokens; caption defaults to 64 if unset (pope/popev2=32, mmhal=512)
#   - result dir suffix _<N> matches caption --max_new_tokens, e.g. llava-v1.5-7b_air_n500_64
max_new_tokens_global="${max_new_tokens:-}"
max_new_tokens_caption="${max_new_tokens_caption:-${max_new_tokens_global:-64}}"
max_new_tokens_pope="${max_new_tokens_pope:-${max_new_tokens_global:-32}}"
max_new_tokens_popev2="${max_new_tokens_popev2:-${max_new_tokens_global:-32}}"
max_new_tokens_mmhal="${max_new_tokens_mmhal:-${max_new_tokens_global:-512}}"

# AIR (paper §5.2): sole decoding intervention in this repo. export use_air=false for baseline only.
#   Modules in order: modality rebalancing / cross-head vision lens / conditional AD-HH + variance projection.
#   air_qk_rescale=true enables line-1 pre-softmax spectral-energy rescale (default off).
use_air="${use_air:-true}"
air_beta="${air_beta:-0.1}"
air_eps="${air_eps:-1e-8}"
air_layer_low="${air_layer_low:-5}"
air_layer_high="${air_layer_high:-18}"
air_qk_rescale="${air_qk_rescale:-false}"
air_qk_scale="${air_qk_scale:-1.0}"
# Internal module hyperparameters (modality rebalancing / cross-head lens / conditional AD-HH)
air_gamma_img="${air_gamma_img:-1.08}"
air_delta_sys="${air_delta_sys:-0.97}"
air_mod_layer_low="${air_mod_layer_low:-9}"
air_mod_layer_high="${air_mod_layer_high:-15}"
air_lens_layer_low="${air_lens_layer_low:-5}"
air_lens_layer_high="${air_lens_layer_high:-18}"
air_alpha_lens="${air_alpha_lens:-0.28}"
air_adhh_threshold="${air_adhh_threshold:-0.4}"
# Conditional AD-HH: true passes --air-no-conditional-adhh to evals (modality rebalancing + lens only)
air_no_conditional_adhh="${air_no_conditional_adhh:-false}"
# Dynamic vision boost (decode-time only; const = constant behavior)
#   air_gamma_schedule: const|exp|log
#   air_gamma_img_max:  schedule ceiling; empty string → constant g in Python
#   air_gamma_tau:      time constant (generated tokens), controls saturation
#   air_gamma_kappa:    slope for log schedule
air_gamma_schedule="${air_gamma_schedule:-const}"
air_gamma_img_max="${air_gamma_img_max:-}"
air_gamma_tau="${air_gamma_tau:-32}"
air_gamma_kappa="${air_gamma_kappa:-0.05}"

# Intervention args via arrays to avoid line-continuation + empty-arg parse errors
air_args_caption=()
air_args_pope=()
air_args_popev2=()
air_args_mmhal=()
if [ "$use_air" = "true" ]; then
    air_args=(
        --air
        --air-beta "$air_beta"
        --air-eps "$air_eps"
        --air-layer-low "$air_layer_low"
        --air-layer-high "$air_layer_high"
        --air-qk-scale "$air_qk_scale"
        --air-gamma-img "$air_gamma_img"
        --air-delta-sys "$air_delta_sys"
        --air-mod-layer-low "$air_mod_layer_low"
        --air-mod-layer-high "$air_mod_layer_high"
        --air-lens-layer-low "$air_lens_layer_low"
        --air-lens-layer-high "$air_lens_layer_high"
        --air-alpha-lens "$air_alpha_lens"
        --air-adhh-threshold "$air_adhh_threshold"
        --air-gamma-schedule "$air_gamma_schedule"
        --air-gamma-tau "$air_gamma_tau"
        --air-gamma-kappa "$air_gamma_kappa"
    )
    if [ -n "$air_gamma_img_max" ]; then
        air_args+=(--air-gamma-img-max "$air_gamma_img_max")
    fi
    if [ "$air_qk_rescale" = "true" ]; then
        air_args+=(--air-qk-rescale)
    fi
    if [ "$air_gamma_schedule" = "const" ]; then
        air_suffix=""
    else
        air_suffix="_dyn${air_gamma_schedule}"
    fi
    if [ "$air_no_conditional_adhh" = "true" ]; then
        air_args+=(--air-no-conditional-adhh)
        result_path=./results/$dataset/${model_name}_air${air_suffix}_nocadh_n${num_samples}_${max_new_tokens_caption}
    else
        result_path=./results/$dataset/${model_name}_air${air_suffix}_n${num_samples}_${max_new_tokens_caption}
    fi
    air_args_caption=("${air_args[@]}")
    air_args_pope=("${air_args[@]}")
    air_args_popev2=("${air_args[@]}")
    air_args_mmhal=("${air_args[@]}")
else
    # baseline (no intervention)
    result_path=./results/$dataset/${model_name}_n${num_samples}_${max_new_tokens_caption}
fi

pope_question_file=$data_path/pope/llava_pope_test.jsonl
pope_annotation_dir=$data_path/pope/coco
pope_result_path=$result_path/pope

# MMHal-Bench: put HF response_template.json at this path; runs only if the file exists.
mmhal_template=$data_path/mmhal/response_template.json
mmhal_image_root="${mmhal_image_root:-}"
mmhal_result_path=$result_path/mmhal
mmhal_image_cache=$mmhal_result_path/image_cache
# GPT judging (WenWen): needs WENWEN_API_KEY or OPENAI_API_KEY; false = generate only, no scoring.
run_mmhal_gpt_eval="${run_mmhal_gpt_eval:-true}"
mmhal_gpt_model="${MMHAL_GPT_MODEL:-}"

# POPEv2 (full annotations.json; images via bash_scripts/download_popev2_dataset.sh)
popev2_dataset_dir=$data_path/POPEv2/dataset
popev2_ann=$popev2_dataset_dir/annotations.json
popev2_result_path=$result_path/popev2

CUDA_VISIBLE_DEVICES='0' python3 -m eval_scripts.eval_caption_air \
--model-path $model_path \
--image-folder $data_path/coco/val2014 \
--caption_file_path $data_path/coco/annotations/captions_val2014.json \
--answers-file $result_path/captions.jsonl \
--dataset $dataset \
--temperature 0 \
--conv-mode ${conv_mode} \
--num_samples $num_samples \
--batch-size $batch_size \
--max_new_tokens "$max_new_tokens_caption" \
"${air_args_caption[@]}"

python3 eval_scripts/eval_utils/eval_chair.py \
    --annotation-dir $data_path/coco/annotations \
    --answers-file $result_path/captions.jsonl \
    --caption_file captions_val2014.json

if [ -f "$pope_question_file" ] && [ -d "$pope_annotation_dir" ]; then
    mkdir -p "$pope_result_path"

    CUDA_VISIBLE_DEVICES='0' python3 -m eval_scripts.eval_pope_air \
    --model-path $model_path \
    --image-folder $data_path/coco/val2014 \
    --question-file $pope_question_file \
    --answers-file $pope_result_path/answers.jsonl \
    --conv-mode ${conv_mode} \
    --batch-size $batch_size \
    --max_new_tokens "$max_new_tokens_pope" \
    "${air_args_pope[@]}"

    python3 -m llava.eval.eval_pope \
    --annotation-dir $pope_annotation_dir \
    --question-file $pope_question_file \
    --result-file $pope_result_path/answers.jsonl | tee $pope_result_path/eval.txt
else
    echo "Skip POPE evaluation: expected question file at $pope_question_file and annotation dir at $pope_annotation_dir"
fi

if [ -f "$popev2_ann" ]; then
    mkdir -p "$popev2_result_path"
    CUDA_VISIBLE_DEVICES='0' python3 -m eval_scripts.eval_popev2_dataset \
        --run-model \
        --dataset-dir "$popev2_dataset_dir" \
        --answers-file "$popev2_result_path/answers.jsonl" \
        --metrics-json "$popev2_result_path/metrics.json" \
        --model-path "$model_path" \
        --conv-mode "${conv_mode}" \
        --batch-size "$batch_size" \
        --max-new-tokens "$max_new_tokens_popev2" \
        "${air_args_popev2[@]}" \
        | tee "$popev2_result_path/eval_console.txt"
else
    echo "Skip POPEv2: not found $popev2_ann (run bash bash_scripts/download_popev2_dataset.sh)"
fi

if [ -f "$mmhal_template" ]; then
    mkdir -p "$mmhal_result_path"
    mmhal_img_args=()
    if [ -n "$mmhal_image_root" ]; then
        mmhal_img_args+=(--image-root "$mmhal_image_root")
    fi
    CUDA_VISIBLE_DEVICES='0' python3 -m eval_scripts.eval_mmhal_bench \
        --model-path "$model_path" \
        --input-json "$mmhal_template" \
        --output-json "$mmhal_result_path/responses.json" \
        --image-cache-dir "$mmhal_image_cache" \
        "${mmhal_img_args[@]}" \
        --conv-mode "${conv_mode}" \
        --batch-size 1 \
        --max-new-tokens "$max_new_tokens_mmhal" \
        "${air_args_mmhal[@]}"
    echo "MMHal-Bench responses written: $mmhal_result_path/responses.json"

    mmhal_gpt_model_args=()
    if [ -n "$mmhal_gpt_model" ]; then
        mmhal_gpt_model_args+=(--gpt-model "$mmhal_gpt_model")
    fi
    if [ "$run_mmhal_gpt_eval" = "true" ] && { [ -n "${WENWEN_API_KEY:-}" ] || [ -n "${OPENAI_API_KEY:-}" ]; }; then
        mkdir -p "$mmhal_result_path"
        CUDA_VISIBLE_DEVICES='' python3 -m eval_scripts.eval_mmhal_bench \
            --gpt-eval-only \
            --response-json "$mmhal_result_path/responses.json" \
            --gpt-eval-output "$mmhal_result_path/responses_gpt_eval.json" \
            "${mmhal_gpt_model_args[@]}" \
            | tee "$mmhal_result_path/gpt_eval_console.txt"
        echo "MMHal GPT judge output and summary: $mmhal_result_path/responses_gpt_eval.json and *_summary.json in same dir"
    elif [ "$run_mmhal_gpt_eval" != "true" ]; then
        echo "Skip MMHal GPT judge: run_mmhal_gpt_eval=$run_mmhal_gpt_eval"
    else
        echo "Skip MMHal GPT judge: set WENWEN_API_KEY or OPENAI_API_KEY (optional MMHAL_GPT_MODEL overrides default judge model)"
    fi
else
    echo "Skip MMHal-Bench: template JSON not found; put response_template.json at $mmhal_template"
fi

echo ""
python3 -m eval_scripts.print_decoding_eval_summary --result-path "$result_path"

