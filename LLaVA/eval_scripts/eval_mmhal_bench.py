"""
Run LLaVA on MMHal-Bench data; write JSON with model_answer for official eval_gpt4.py scoring.

Data: response_template.json from HF Shengcao1006/MMHal-Bench (array of image_src, question).

Images:
  - Prefer local HF images/ next to the template (basename matches URL tail) to avoid Flickr 410.
  - If missing and image_src is http(s), download to --image-cache-dir.
  - Local relative paths: join with --image-root and copy to cache.

Usage (from LLaVA/):
  python3 -m eval_scripts.eval_mmhal_bench \\
    --model-path liuhaotian/llava-v1.5-7b \\
    --input-json ../dataset/mmhal/response_template.json \\
    --output-json ./results/coco/my_run/mmhal/responses.json \\
    --conv-mode vicuna_v1

AIR (sole decoding intervention; paper §5.2):
  --air [--air-beta 0.1 --air-layer-low 5 --air-layer-high 18 --air-gamma-img 1.08 --air-alpha-lens 0.28 ...]

GPT judging (WenWen only: https://api.wenwen-ai.com/v1, default gpt-4o):
  export WENWEN_API_KEY='...'   # or export OPENAI_API_KEY='...'
  python3 -m eval_scripts.eval_mmhal_bench --gpt-eval-only \\
    --response-json ./results/mmhal/llava-v15-7b/responses.json \\
    --gpt-eval-output ./results/mmhal/llava-v15-7b/gpt_eval.json
  # optional: --gpt-model for other WenWen model ids

Official eval_gpt4.py (legacy openai.ChatCompletion API) also works.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from io import BytesIO
from urllib.parse import urlparse

import requests
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from eval_scripts.eval_pope_air import (
    add_air_arguments,
    air_kwargs_from_args,
    apply_air_intervention,
)

# GPT judging: fixed WenWen gateway; default model gpt-4o.
WENWEN_API_BASE = "https://api.wenwen-ai.com/v1"
DEFAULT_MMHAL_JUDGE_MODEL = "gpt-4o"

# Same template as MMHal-Bench eval_gpt4.py (GPT judge prompt)
MMHAL_JUDGE_TEMPLATE = """Please act as an impartial and objective judge and evaluate the quality of the response provided by a Large Multimodal Model (LMM) to the user question. Your evaluation should be mainly based on whether the response is informative, and whether the response contains any hallucination. Hallucination, in this context, refers to a situation where the LMM generates a response that includes information not present or implied in the image or previous conversation. A hallucination could be a false claim about an object, action, emotion, or any other detail that is not grounded in the image.

For clarity, consider these examples:

### Example 1:
- Image Contents: Lynx, Cat
- Question: How many cats can you see in this image?
- Standard Human-Generated Answer: There are two cats visible in the image.
- LMM Response to Evaluate: There is one cat in the image.
- Analysis: The LMM's response is false because it claims there is only one cat in the image, when in fact there are two.
- Hallucination: Yes.

### Example 2:
- Image Contents: Tree, Plant, Panda, Animal
- Question: What can you see from this photo?
- Standard Human-Generated Answer: In this photo we can see branches. We can see a panda on a branch. The background is blurry.
- LMM Response to Evaluate: In the photo, I can see a large black and white panda sitting on a tree branch. The image captures the panda's unique appearance and its natural habitat, showcasing the beauty and rarity of these animals.
- Analysis: The LMM's response is not false because it does not make any false claims about the image contents. For example, it does not claim that the panda is eating bamboo.
- Hallucination: No.

### Example 3:
- Image Contents: Fashion accessory, Mammal, Dog, Umbrella
- Question: Describe this image in details.
- Standard Human-Generated Answer: In this picture I can see a dog laying under a yellow umbrella on a beach, and there is a rope on the right side. In the background I can see few persons talking or lying. At the top of the image, I can see the blue and clear sky.
- LMM Response to Evaluate: The image features a brown dog laying on the sand at the beach, underneath a yellow umbrella. The dog is positioned near the water, and there are people in the background. The beach is sandy, and the sun is shining, creating a pleasant atmosphere. The dog is enjoying the shade provided by the umbrella, which is a common way to protect pets from the sun's harmful rays. The presence of people in the background suggests that the beach is a popular spot for leisure activities, such as sunbathing, swimming, and relaxing.
- Analysis: The LMM's response is false because it claims that the dog is positioned near the water, but there is no water in the image.
- Hallucination: Yes.

### Example 4:
- Image Contents: Clothing, Luggage and bags, Man, Building, Woman, Person
- Question: Where is this photo taken?
- Standard Human-Generated Answer: From the slogan "Welcome to Houston, Texas", we can infer that this photo is taken at an airport in Houston, Texas.
- LMM Response to Evaluate: This photo is taken at an airport, specifically at the entrance of a terminal. However, the specific location of this airport cannot be determined from the image.
- Analysis: Although the LMM's response does not explicitly mention the location "Houston, Texas" and does not perfectly answer the question, it makes no false claims about the image contents. Therefore, this response is not considered hallucination. It is OK if the LMM fails to directly answer the question, expresses uncertainty, or cannot include all the objects or details in the human-generated answer.
- Hallucination: No.

### Example 5:
- Image Contents: Tree, Tower
- Question: What environmental concerns are involved in this photo?
- Standard Human-Generated Answer: This photo shows a smokestack emitting smoke. It may cause severe air pollution which threatens human health.
- LMM Response to Evaluate: In the photo, there are several environmental concerns related to the smokestack emitting smoke. The smoke from the smokestack is a byproduct of industrial processes, which can contribute to air pollution, climate change, and negative impacts on human health. The smoke contains harmful pollutants, such as particulate matter, sulfur dioxide, nitrogen oxides, and carbon monoxide, which can lead to respiratory problems, heart disease, and other health issues. Additionally, the smoke contributes to the greenhouse effect, which can lead to global warming and climate change. The smokestack's emissions also affect the environment, as they can harm wildlife, vegetation, and ecosystems. Therefore, it is essential to address these environmental concerns by implementing measures to reduce emissions and promote sustainable practices in industrial processes.
- Analysis: Although the LMM's response is significantly longer than the standard human-generated answer, it does not contain any false claims about the image contents. Instead, it provides additional general information about the environmental concerns, which can be inferred from the smoke emission. Such detailed analysis or reasoning should be considered as a positive aspect, as long as it contains no false claims.
- Hallucination: No.

With these examples in mind, please help me evaluate whether the response by the LMM is informative, and whether hallucination exists in it, based on the comparison between the LMM's response and the factual information provided in the image contents, question, and the standard human-generated answer below.

Please note that the standard human-generated answer may only contain factual information but may not give a detailed analysis. Also, the standard human-generated answer may not be completely comprehensive in describing all the objects and their attributes, so please be a bit more cautious during evalutation. LMM's detailed analysis or reasoning should be encouraged.

To evaluate the LMM responses, first, begin your evaluation by providing a short explanation. Second, after providing your explanation, you must rate the response by choosing from the following options:
- Rating: 6, very informative with good analysis or reasoning, no hallucination
- Rating: 5, very informative, no hallucination
- Rating: 4, somewhat informative, no hallucination
- Rating: 3, not informative, no hallucination
- Rating: 2, very informative, with hallucination
- Rating: 1, somewhat informative, with hallucination
- Rating: 0, not informative, with hallucination

### Image Contents
{}

### Question
{}

### Standard Human-Generated Answer
{}

### LMM Response to Evaluate
{}
"""


def _load_wenwen_api_key() -> str:
    for name in ("WENWEN_API_KEY", "OPENAI_API_KEY"):
        v = os.environ.get(name)
        if v and v.strip():
            return v.strip()
    return ""


def _parse_mmhal_scores(judge_texts: list[str]) -> list[int]:
    scores = []
    for i, text in enumerate(judge_texts):
        t = text.lower()
        found = [s for s in range(7) if f"rating: {s}" in t]
        if len(found) == 1:
            scores.append(found[0])
        else:
            print(f"Warning: multiple or zero scores for item {i}")
            print(text[:500])
            scores.append(0)
    return scores


def _build_mmhal_summary(scores: list[int]) -> dict:
    """Summary metrics matching terminal output; per-type averages use index i%8 (official eval_gpt4)."""
    n = len(scores)
    if n == 0:
        return {"num_items": 0}
    hallucination_flags = [0 if s >= 3 else 1 for s in scores]
    scores_each: list[list[int]] = [[] for _ in range(8)]
    for i in range(n):
        scores_each[i % 8].append(scores[i])
    by_type = []
    for bucket in scores_each:
        by_type.append(round(sum(bucket) / len(bucket), 2) if bucket else 0.0)
    return {
        "num_items": n,
        "average_score": round(sum(scores) / n, 2),
        "hallucination_rate": round(sum(hallucination_flags) / n, 2),
        "average_score_by_type_index_0_to_7": by_type,
        "per_item_scores": list(scores),
        "per_item_hallucination_flag": hallucination_flags,
    }


def _print_mmhal_summary(summary: dict) -> None:
    if summary.get("num_items", 0) == 0:
        print("No scores to summarize.")
        return
    print(f"Average score: {summary['average_score']:.2f}")
    print(f"Hallucination rate: {summary['hallucination_rate']:.2f}")
    parts = [str(x) for x in summary["average_score_by_type_index_0_to_7"]]
    print("Average score for each question type:", ",".join(parts), flush=True)


def run_gpt_eval(args) -> None:
    """Score MMHal responses.json via WenWen OpenAI-compatible API (same as test.py)."""
    try:
        from openai import OpenAI, AuthenticationError
    except ImportError as e:
        raise SystemExit("Install: pip install openai") from e

    api_key = _load_wenwen_api_key()
    if not api_key:
        checked = ("WENWEN_API_KEY", "OPENAI_API_KEY")
        hint = ", ".join(f"{n}={'set' if os.environ.get(n) else 'unset'}" for n in checked)
        raise SystemExit(
            "WenWen API key not found. Set:\n"
            "  export WENWEN_API_KEY='...'\n"
            "  or export OPENAI_API_KEY='...'\n"
            f"Gateway: {WENWEN_API_BASE}\n"
            f"Status: {hint}"
        )

    client = OpenAI(api_key=api_key, base_url=WENWEN_API_BASE.rstrip("/"))

    resp_path = os.path.expanduser(args.response_json)
    with open(resp_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if len(records) != 96:
        print(f"Warning: expected 96 MMHal samples, got {len(records)}; continuing with all.")

    stored: list[dict] = []
    judge_texts: list[str] = []

    for i, record in enumerate(tqdm(records, desc="GPT judge")):
        image_content = ", ".join(record.get("image_content") or [])
        user_prompt = MMHAL_JUDGE_TEMPLATE.format(
            image_content,
            record.get("question", ""),
            record.get("gt_answer", ""),
            record.get("model_answer", ""),
        )

        content: str | None = None
        completion = None
        while content is None:
            try:
                completion = client.chat.completions.create(
                    model=args.gpt_model,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.0,
                )
                content = completion.choices[0].message.content or ""
            except AuthenticationError as e:
                raise SystemExit(
                    "401: WenWen rejected the API key (invalid or expired). Check WENWEN_API_KEY / OPENAI_API_KEY."
                ) from e
            except Exception as e:
                print(e)
                print("retrying in 10s...")
                time.sleep(10)
                continue

        print(i, content[:200].replace("\n", " "), "...", flush=True)
        judge_texts.append(content)
        stored.append(
            {
                "index": i,
                "question_type": record.get("question_type"),
                "content": content,
                "model": getattr(completion, "model", None) or args.gpt_model,
            }
        )
        time.sleep(1)

    out_path = os.path.expanduser(args.gpt_eval_output)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stored, f, indent=2, ensure_ascii=False)
    print(f"Saved raw judge output: {out_path}")

    scores = _parse_mmhal_scores(judge_texts)
    summary = _build_mmhal_summary(scores)
    summary["response_json"] = resp_path
    summary["gpt_eval_items_json"] = out_path
    summary["gpt_model"] = args.gpt_model
    summary["wenwen_api_base"] = WENWEN_API_BASE

    root, ext = os.path.splitext(out_path)
    summary_path = f"{root}_summary{ext or '.json'}"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved summary metrics: {summary_path}")

    _print_mmhal_summary(summary)


def _basename_from_image_src(src: str) -> str:
    if src.startswith("http://") or src.startswith("https://"):
        return os.path.basename(urlparse(src).path)
    return os.path.basename(src) if src else ""


def materialize_image_row(
    row: dict,
    idx: int,
    image_cache_dir: str,
    image_root: str | None,
    image_archive_dir: str | None,
) -> str:
    """
    Resolve image_src to a filename under image_cache_dir and ensure the file exists.
    Prefer local archive by URL basename (Flickr links often return 410).
    """
    os.makedirs(image_cache_dir, exist_ok=True)
    src = row.get("image_src") or ""
    if not src:
        raise ValueError(f"Row {idx}: missing image_src")

    basename = _basename_from_image_src(src)
    if not basename:
        raise ValueError(f"Row {idx}: cannot parse filename from image_src: {src!r}")

    def _copy_into_cache(src_file: str, dest_name: str) -> str:
        dest_path = os.path.join(image_cache_dir, dest_name)
        if not os.path.isfile(dest_path):
            shutil.copy2(src_file, dest_path)
        return dest_name

    archive_path = None
    if image_archive_dir:
        cand = os.path.join(image_archive_dir, basename)
        if os.path.isfile(cand):
            archive_path = cand

    if src.startswith("http://") or src.startswith("https://"):
        dest_name = basename
        dest_path = os.path.join(image_cache_dir, dest_name)
        if os.path.isfile(dest_path):
            return dest_name
        if archive_path is not None:
            return _copy_into_cache(archive_path, dest_name)
        try:
            r = requests.get(src, timeout=120, headers={"User-Agent": "MMHal-Eval/1.0"})
            r.raise_for_status()
            Image.open(BytesIO(r.content)).convert("RGB").save(dest_path)
            return dest_name
        except (requests.HTTPError, requests.Timeout, requests.ConnectionError):
            if image_archive_dir:
                cand = os.path.join(image_archive_dir, basename)
                if os.path.isfile(cand):
                    return _copy_into_cache(cand, dest_name)
            raise RuntimeError(
                f"Row {idx}: cannot fetch image and no local backup: {src}\n"
                f"Place the HF dataset images/ at {image_archive_dir or '(images/ next to response_template.json)'} "
                f"and ensure file exists: {basename}"
            ) from None

    if image_root:
        full = os.path.join(os.path.expanduser(image_root), src)
    else:
        full = os.path.expanduser(src)
    if not os.path.isfile(full):
        raise FileNotFoundError(f"Image not found: {full} (row {idx})")

    iid = row.get("image_id", idx)
    ext = os.path.splitext(full)[1] or ".jpg"
    local_name = f"local_{iid}_{idx}{ext}"
    local_path = os.path.join(image_cache_dir, local_name)
    if not os.path.isfile(local_path):
        shutil.copy2(full, local_path)
    return local_name


class MMHalDataset(Dataset):
    def __init__(self, items, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.items = items
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.items[index]
        image_file = line["image"]
        qs = line["text"]
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, image_file)).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        return input_ids, image_tensor, image.size, line

    def __len__(self):
        return len(self.items)


def collate_fn(batch, pad_token_id):
    input_ids, image_tensors, image_sizes, lines = zip(*batch)
    input_ids = [x.squeeze(0) for x in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = input_ids.ne(pad_token_id)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, attention_mask, image_tensors, image_sizes, lines


def create_mmhal_loader(items, image_folder, tokenizer, image_processor, model_config, conv_mode, batch_size, num_workers):
    dataset = MMHalDataset(items, image_folder, tokenizer, image_processor, model_config, conv_mode)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
    )


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name)

    in_path = os.path.expanduser(args.input_json)
    with open(in_path, "r", encoding="utf-8") as f:
        template = json.load(f)
    if not isinstance(template, list):
        raise ValueError("input-json must be a JSON array (like response_template.json)")

    image_cache_dir = os.path.expanduser(args.image_cache_dir)
    image_root = os.path.expanduser(args.image_root) if args.image_root else None

    default_archive = os.path.join(os.path.dirname(in_path), "images")
    if args.image_archive_dir:
        image_archive_dir = os.path.expanduser(args.image_archive_dir)
    elif os.path.isdir(default_archive):
        image_archive_dir = default_archive
    else:
        image_archive_dir = None

    loader_items = []
    for idx, row in enumerate(template):
        local_name = materialize_image_row(row, idx, image_cache_dir, image_root, image_archive_dir)
        text = (row.get("question") or "").strip()
        if not text:
            raise ValueError(f"Row {idx}: missing question")
        qid = f"mmhal_{idx}"
        loader_items.append(
            {
                "question_id": qid,
                "image": local_name,
                "text": text,
                "_row_index": idx,
            }
        )

    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in args.conv_mode:
        args.conv_mode = args.conv_mode + "_mmtag"
        print(f"Auto-switched conv_mode to {args.conv_mode}")

    if getattr(args, "air", False):
        apply_air_intervention(model_path, model.config, **air_kwargs_from_args(args))

    data_loader = create_mmhal_loader(
        loader_items,
        image_cache_dir,
        tokenizer,
        image_processor,
        model.config,
        args.conv_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    answers_by_index: dict[int, str] = {}
    for input_ids, attention_mask, image_tensor, image_sizes, lines in tqdm(data_loader, total=len(data_loader)):
        input_ids = input_ids.to(device="cuda", non_blocking=True)
        attention_mask = attention_mask.to(device="cuda", non_blocking=True)
        image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for line, output in zip(lines, outputs):
            answers_by_index[int(line["_row_index"])] = output.strip()

    for idx, ans in answers_by_index.items():
        template[idx]["model_answer"] = ans

    out_path = os.path.expanduser(args.output_json)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(answers_by_index)} model_answer entries; wrote {out_path}")
    print("GPT judge: python3 -m eval_scripts.eval_mmhal_bench --gpt-eval-only --response-json <file> ...")


def main():
    parser = argparse.ArgumentParser(description="MMHal-Bench: LLaVA generation or GPT judging (WenWen only).")
    parser.add_argument(
        "--gpt-eval-only",
        action="store_true",
        help="WenWen GPT judge only (no LLaVA); needs --response-json and WENWEN_API_KEY or OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--response-json",
        type=str,
        default=None,
        help="MMHal JSON with model_answer (e.g. responses.json).",
    )
    parser.add_argument(
        "--gpt-eval-output",
        type=str,
        default=None,
        help="Save per-item GPT judge text; default <response-json-dir>/*_gpt_eval.json",
    )
    parser.add_argument(
        "--gpt-model",
        type=str,
        default=DEFAULT_MMHAL_JUDGE_MODEL,
        help=f"WenWen model id; default {DEFAULT_MMHAL_JUDGE_MODEL}",
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--input-json", type=str, default=None)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument(
        "--image-cache-dir",
        type=str,
        default="./results/mmhal_image_cache",
        help="Cache downloaded/copied images for DataLoader",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=None,
        help="Root when image_src is a local relative path",
    )
    parser.add_argument(
        "--image-archive-dir",
        type=str,
        default=None,
        dest="image_archive_dir",
        help="Bundled MMHal images/ (sibling of response_template.json). Default: images/ next to input-json",
    )
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512, dest="max_new_tokens")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    # AIR: sole decoding intervention (core regularization + internal module hyperparameters)
    add_air_arguments(parser)
    args = parser.parse_args()

    if args.gpt_eval_only:
        if not args.response_json:
            parser.error("--gpt-eval-only requires --response-json")
        if not args.gpt_eval_output:
            base, ext = os.path.splitext(os.path.expanduser(args.response_json))
            args.gpt_eval_output = f"{base}_gpt_eval{ext or '.json'}"
        run_gpt_eval(args)
        return

    missing = []
    if not args.model_path:
        missing.append("--model-path")
    if not args.input_json:
        missing.append("--input-json")
    if not args.output_json:
        missing.append("--output-json")
    if missing:
        parser.error("Generation mode requires: " + ", ".join(missing) + " (or use --gpt-eval-only)")

    eval_model(args)


if __name__ == "__main__":
    main()
