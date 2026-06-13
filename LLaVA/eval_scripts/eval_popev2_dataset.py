

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import shortuuid
import torch
from tqdm import tqdm

from eval_scripts.eval_pope_air import (
    add_air_arguments,
    air_kwargs_from_args,
    apply_air_intervention,
    create_data_loader,
    get_chunk,
)
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def script_default_dataset_dir() -> Path:
    """Default dataset dir: repo_root/dataset/POPEv2/dataset (this file is under LLaVA/eval_scripts/)."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    return repo_root / "dataset" / "POPEv2" / "dataset"


def resolve_coco_image_with_fallback(dataset_dir: Path, image_name: str) -> Path:
    """
    Resolve COCO image paths in order:
    - <dataset-dir>/coco/train20xx/<name>.jpg
    - <repo>/dataset/coco/train20xx/<name>.jpg
    Also accepts COCO_train20xx_<name>.jpg; falls back to dataset/images/<name>.jpg.
    """
    name = image_name.strip().replace("\\", "/")
    parts = Path(name).parts
    train_split = None
    for p in parts:
        if p in ("train2014", "train2017"):
            train_split = p
            break
    if train_split is None:
        raise SystemExit(f"Unrecognized COCO split: {image_name!r}")

    base = Path(name).name
    prefixed = f"COCO_{train_split}_{base}"
    coco_roots = [
        (dataset_dir / "coco" / train_split).resolve(),
        (dataset_dir.parent.parent / "coco" / train_split).resolve(),
    ]
    for root in coco_roots:
        for fn in (base, prefixed):
            p = (root / fn).resolve()
            if p.is_file():
                return p
    # Fallback: bundled POPEv2 images/
    return (dataset_dir / "images" / base).resolve()


def resolve_image_path(dataset_dir: Path, image_name: str) -> Path:
    """Map annotation image_name to a local path under the POPEv2 package."""
    name = image_name.strip()
    if name.startswith("/images/"):
        return dataset_dir / "images" / Path(name).name
    if (
        "/coco/train2017/" in name
        or name.startswith("/coco/train2017/")
        or "/coco/train2014/" in name
        or name.startswith("/coco/train2014/")
    ):
        return resolve_coco_image_with_fallback(dataset_dir, name)
    raise SystemExit(f"Only /images/... or /coco/train2014|train2017/... supported; got image_name: {name!r}")


def row_abs_image_path(row: dict[str, Any], dataset_dir: Path) -> Path:
    """Absolute image path for one POPEv2 annotation row."""
    name = (row.get("image_name") or "").strip().replace("\\", "/")
    if name.startswith("/images/"):
        return (dataset_dir / "images" / Path(name).name).resolve()
    if (
        "/coco/train2017/" in name
        or name.startswith("/coco/train2017/")
        or "/coco/train2014/" in name
        or name.startswith("/coco/train2014/")
    ):
        return resolve_coco_image_with_fallback(dataset_dir, name)
    raise SystemExit(f"Only /images/... or /coco/train2014|train2017/... supported; got image_name: {name!r}")


def build_questions_from_rows(rows: list[dict[str, Any]], dataset_dir: Path) -> list[dict[str, Any]]:
    """Build eval_pope_air-style question dicts; image is absolute path, image_folder can be empty."""
    questions: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        img_path = row_abs_image_path(row, dataset_dir)
        if not img_path.is_file():
            raise SystemExit(f"Row {i}: image not found: {img_path}")
        q = row.get("query") or row.get("question")
        if not q:
            raise SystemExit(f"Row {i}: missing query")
        questions.append(
            {
                "question_id": i,
                "text": str(q),
                "image": str(img_path),
            }
        )
    return questions


def text_to_yes_no(text: str) -> str:
    """Match eval_pope.py: first sentence, strip commas; no/not → no, else yes."""
    if not text:
        return "no"
    t = text
    if "." in t:
        t = t.split(".", 1)[0]
    t = t.replace(",", "")
    words = t.split()
    if "No" in words or "not" in words or "no" in words:
        return "no"
    return "yes"


def label_to_binary(label: str) -> int:
    s = (label or "").strip().lower()
    return 1 if s in ("yes", "y", "1", "true") else 0


def eval_binary(pred: list[int], gold: list[int]) -> dict[str, float]:
    pos, neg = 1, 0
    tp = fp = tn = fn = 0
    for p, g in zip(pred, gold):
        if p == pos and g == pos:
            tp += 1
        elif p == pos and g == neg:
            fp += 1
        elif p == neg and g == neg:
            tn += 1
        else:
            fn += 1
    n = tp + fp + tn + fn
    pos_preds = sum(pred)
    yes_ratio = pos_preds / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / n if n else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    pbo = (pos_preds / n * 100.0) - 50.0 if n else float("nan")
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Yes_ratio": yes_ratio,
        "TNR": tnr,
        "PBO": pbo,
    }


def load_annotations(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"Expected JSON array, got: {type(data)}")
    return data


def filter_images_only_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only /images/... rows (bundled POPEv2 images; no COCO train2017 needed)."""
    out: list[dict[str, Any]] = []
    for r in rows:
        name = (r.get("image_name") or "").strip().replace("\\", "/")
        if name.startswith("/images/"):
            out.append(r)
    return out


def load_answers_jsonl_by_question_id(path: Path, n_expected: int) -> list[str]:
    """Load text aligned by question_id 0..n-1 (compatible with eval_pope_air output)."""
    by_id: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            qid = obj.get("question_id")
            if qid is None:
                raise SystemExit(f"Line {i+1}: missing question_id")
            if "text" in obj:
                by_id[int(qid)] = str(obj["text"])
            elif "answer" in obj:
                by_id[int(qid)] = str(obj["answer"])
            else:
                raise SystemExit(f"Line {i+1}: need text or answer: {obj.keys()}")
    missing = [k for k in range(n_expected) if k not in by_id]
    if missing:
        raise SystemExit(f"Answers missing question_id (sample): {missing[:10]}")
    return [by_id[i] for i in range(n_expected)]


def load_answers_jsonl_sequential(path: Path) -> list[str]:
    """Load text in file line order (legacy; matches annotation array order)."""
    texts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" in obj:
                texts.append(str(obj["text"]))
            elif "answer" in obj:
                texts.append(str(obj["answer"]))
            else:
                raise SystemExit(f"Line {i+1}: need 'text' or 'answer' field: {obj.keys()}")
    return texts


def run_llava_inference(args: argparse.Namespace, questions: list[dict[str, Any]]) -> None:
    """Same inference as eval_pope_air.eval_model; questions use absolute image paths."""
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file) or ".", exist_ok=True)
    ans_file = open(answers_file, "w", encoding="utf-8")

    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in args.conv_mode:
        args.conv_mode = args.conv_mode + "_mmtag"
        print(f"Auto-switched conv_mode to {args.conv_mode}")

    if getattr(args, "air", False):
        apply_air_intervention(model_path, model.config, **air_kwargs_from_args(args))

    image_folder = os.path.expanduser(args.image_folder) if args.image_folder else ""
    data_loader = create_data_loader(
        questions,
        image_folder,
        tokenizer,
        image_processor,
        model.config,
        args.conv_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

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
            ans_id = shortuuid.uuid()
            ans_file.write(
                json.dumps(
                    {
                        "question_id": line["question_id"],
                        "prompt": line["text"],
                        "text": output.strip(),
                        "answer_id": ans_id,
                        "model_id": model_name,
                        "metadata": {},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            ans_file.flush()

    ans_file.close()
    print(f"Wrote: {answers_file}")


def print_metrics(
    rows: list[dict[str, Any]],
    texts: list[str],
    metrics_json: Path | None = None,
) -> dict[str, Any]:
    n = len(rows)
    gold = [label_to_binary(str(r.get("label", ""))) for r in rows]
    pred = [label_to_binary(text_to_yes_no(t)) for t in texts]
    m = eval_binary(pred, gold)
    m_out: dict[str, Any] = {**m, "num_samples": n}
    if metrics_json is not None:
        metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(m_out, f, indent=2, ensure_ascii=False)
        print(f"Metrics written: {metrics_json}")
    print(f"\n=== POPE-style metrics (all {n} rows) ===")
    print("TP\tFP\tTN\tFN")
    print(f"{m['TP']}\t{m['FP']}\t{m['TN']}\t{m['FN']}")
    print(f"Accuracy: {m['Accuracy']:.6f}")
    print(f"Precision: {m['Precision']:.6f}")
    print(f"Recall: {m['Recall']:.6f}")
    print(f"F1 score: {m['F1']:.6f}")
    print(f"Yes ratio: {m['Yes_ratio']:.6f}")
    print(f"TNR (TN/(TN+FP)): {m['TNR']:.6f}")
    print(f"PBO (yes-ratio×100−50, percentage points): {m['PBO']:.4f}")
    print(f"{m['F1']:.3f}, {m['Accuracy']:.3f}, {m['Precision']:.3f}, {m['Recall']:.3f}, {m['Yes_ratio']:.3f}")
    return m_out


def main() -> None:
    parser = argparse.ArgumentParser(description="POPEv2: QA checks / LLaVA inference / metrics")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Dir with annotations.json and images/; default repo_root/dataset/POPEv2/dataset",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="annotations.json; default <dataset-dir>/annotations.json",
    )
    parser.add_argument(
        "--answers",
        "--eval-answers",
        type=Path,
        dest="eval_answers",
        default=None,
        help="Eval only: model output JSONL (same as --answers-file after --run-model)",
    )
    parser.add_argument(
        "--align-by-question-id",
        action="store_true",
        help="Align answers by question_id (recommended; matches eval_pope_air); else file line order",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Eval only /images/... rows (500 rows, usually all No)",
    )
    # ---- Inference (aligned with eval_pope_air) ----
    parser.add_argument("--run-model", action="store_true", help="Run LLaVA inference on full POPEv2")
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--image-folder",
        type=str,
        default="",
        help="Usually empty; POPEv2 uses absolute image path per question",
    )
    parser.add_argument(
        "--answers-file",
        type=str,
        default=None,
        help="Required with --run-model: output JSONL path",
    )
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32, dest="max_new_tokens")
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        type=int,
        default=1,
        dest="batch_size",
        help="Inference batch size (default 1)",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    # AIR: sole decoding intervention (core regularization + internal module hyperparameters)
    add_air_arguments(parser)
    parser.add_argument(
        "--skip-metrics-after-model",
        action="store_true",
        help="Do not print metrics after --run-model",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Write TP/FP/Accuracy/F1 etc. to JSON (same as print_metrics)",
    )
    args = parser.parse_args()

    dataset_dir = (args.dataset_dir or script_default_dataset_dir()).resolve()
    ann_path = args.annotations or (dataset_dir / "annotations.json")
    if not ann_path.is_file():
        raise SystemExit(f"Annotation file not found: {ann_path}")

    rows = load_annotations(ann_path)
    n_full = len(rows)
    if args.images_only:
        rows = filter_images_only_rows(rows)
        print(f"--images-only: kept {len(rows)} of {n_full} rows (/images/ only)")
    else:
        print(f"Using full annotations: {n_full} rows")
    n = len(rows)
    print(f"Annotation rows: {n}")
    print(f"Dataset dir: {dataset_dir}")

    labels = Counter((r.get("label") or "").strip() for r in rows)
    print(f"Label distribution: {dict(labels)}")

    image_ids = {r["image_id"] for r in rows if "image_id" in r}
    print(f"Unique image_id count: {len(image_ids)}")

    missing_images: list[str] = []
    ok_images = 0
    for r in rows:
        name = r.get("image_name") or ""
        local = resolve_image_path(dataset_dir, name)
        if local.is_file():
            ok_images += 1
        else:
            missing_images.append(str(local))

    uniq_missing_img = sorted(set(missing_images))
    print(f"/images/ rows: {ok_images} files exist; {len(uniq_missing_img)} unique paths missing")
    if uniq_missing_img:
        print("Missing examples (up to 5):")
        for p in uniq_missing_img[:5]:
            print(" ", p)

    if args.run_model:
        if not args.answers_file:
            raise SystemExit("--run-model requires --answers-file for output JSONL")
        questions = build_questions_from_rows(rows, dataset_dir)
        questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
        print(f"Inference rows this chunk: {len(questions)}")
        run_llava_inference(args, questions)
        if not args.skip_metrics_after_model and args.num_chunks != 1:
            print("Skipping auto metrics when num_chunks!=1; merge JSONL and use --eval-answers.")

    metrics_json_path = args.metrics_json.resolve() if args.metrics_json else None

    if args.eval_answers:
        path = Path(args.eval_answers).expanduser()
        texts = (
            load_answers_jsonl_by_question_id(path, n)
            if args.align_by_question_id
            else load_answers_jsonl_sequential(path)
        )
        if len(texts) != n:
            raise SystemExit(f"Answer count {len(texts)} != annotation count {n}")
        print_metrics(rows, texts, metrics_json_path)
    elif args.run_model and not args.skip_metrics_after_model and args.num_chunks == 1:
        texts = load_answers_jsonl_by_question_id(Path(args.answers_file), n)
        if len(texts) != n:
            raise SystemExit(f"Answer count {len(texts)} != annotation count {n}")
        print_metrics(rows, texts, metrics_json_path)


if __name__ == "__main__":
    main()
