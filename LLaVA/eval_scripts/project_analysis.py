import argparse
import json
import os
import re
from collections import Counter

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from llava.mm_utils import get_model_name_from_path, process_images
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect projected visual patch embeddings in LLaVA."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument(
        "--image-file",
        type=str,
        default=None,
        help="Single-image mode: path to one image to analyze.",
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        default=None,
        help="Batch mode: folder containing images referenced by --chair-results-file.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--score-mode",
        type=str,
        default="lm_head",
        choices=["lm_head", "embed_cosine"],
        help="How to map projected patch embeddings to vocabulary tokens.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of candidate tokens to keep for each patch.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device, e.g. cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="Optionally map patch tokens back to the image and save overlay figures.",
    )
    parser.add_argument(
        "--overlay-scale",
        type=int,
        default=4,
        help="Scale factor for the large overlay image.",
    )
    parser.add_argument(
        "--annotation-dir",
        type=str,
        default=None,
        help="Optional COCO annotation directory for matching patch tokens against GT objects.",
    )
    parser.add_argument(
        "--chair-results-file",
        type=str,
        default=None,
        help="Optional captions_eval_results.json for checking whether final hallucinated words already appear in patch tokens. Also drives batch mode when --image-file is omitted.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Batch mode: optionally limit the number of images to process.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Batch mode: skip images whose patch_token_analysis.json already exists.",
    )
    parser.add_argument(
        "--save-per-image",
        action="store_true",
        help="Batch mode: save per-image json/txt outputs. By default batch mode only writes batch_summary.json.",
    )
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    return parser.parse_args()


def load_image(image_file):
    return Image.open(image_file).convert("RGB")


def is_batch_mode(args):
    return args.image_file is None


def validate_args(args):
    if is_batch_mode(args):
        if not args.chair_results_file:
            raise ValueError("Batch mode requires --chair-results-file.")
        if not args.image_folder:
            raise ValueError("Batch mode requires --image-folder.")
    elif args.image_file is None:
        raise ValueError("Please provide --image-file for single-image mode.")


def resolve_device_map(device):
    if device == "cuda":
        return "auto"
    return device


def prepare_image_tensor(image, image_processor, model_config, device):
    image_tensor = process_images([image], image_processor, model_config)
    if isinstance(image_tensor, list):
        raise ValueError("This script currently expects a single stacked image tensor.")
    return image_tensor.to(device=device, dtype=torch.float16)


@torch.no_grad()
def get_projected_patch_features(model, image_tensor):
    vision_tower = model.get_vision_tower()
    raw_patch_features = vision_tower(image_tensor)
    projected_patch_features = model.get_model().mm_projector(raw_patch_features)
    return raw_patch_features, projected_patch_features


@torch.no_grad()
def project_to_vocab(model, projected_patch_features, score_mode):
    if score_mode == "lm_head":
        scores = model.lm_head(projected_patch_features)
    elif score_mode == "embed_cosine":
        embed_weight = model.get_model().embed_tokens.weight
        normalized_patches = F.normalize(projected_patch_features, dim=-1)
        normalized_embeds = F.normalize(embed_weight, dim=-1)
        scores = torch.matmul(normalized_patches, normalized_embeds.transpose(0, 1))
    else:
        raise ValueError(f"Unsupported score mode: {score_mode}")
    return scores


def decode_token(tokenizer, token_id):
    token_text = tokenizer.convert_ids_to_tokens(token_id)
    if token_text is None:
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
    return token_text


def get_patch_geometry(image_size, patches_per_side, image_aspect_ratio):
    width, height = image_size

    if patches_per_side is None:
        return None

    if image_aspect_ratio == "pad":
        square_size = max(width, height)
        offset_x = (square_size - width) / 2.0
        offset_y = (square_size - height) / 2.0
        canvas_width = square_size
        canvas_height = square_size
    else:
        offset_x = 0.0
        offset_y = 0.0
        canvas_width = float(width)
        canvas_height = float(height)

    patch_width = canvas_width / patches_per_side
    patch_height = canvas_height / patches_per_side
    image_box = (offset_x, offset_y, offset_x + width, offset_y + height)

    geometry = []
    for row in range(patches_per_side):
        for col in range(patches_per_side):
            canvas_box = (
                col * patch_width,
                row * patch_height,
                (col + 1) * patch_width,
                (row + 1) * patch_height,
            )
            visible_box = (
                max(canvas_box[0], image_box[0]),
                max(canvas_box[1], image_box[1]),
                min(canvas_box[2], image_box[2]),
                min(canvas_box[3], image_box[3]),
            )
            if visible_box[0] >= visible_box[2] or visible_box[1] >= visible_box[3]:
                visible_box = None
            else:
                visible_box = (
                    visible_box[0] - offset_x,
                    visible_box[1] - offset_y,
                    visible_box[2] - offset_x,
                    visible_box[3] - offset_y,
                )
            geometry.append(
                {
                    "canvas_box": [float(value) for value in canvas_box],
                    "image_box": [float(value) for value in visible_box] if visible_box else None,
                }
            )

    return geometry


def build_patch_records(tokenizer, scores, top_k, patches_per_side, patch_geometry=None):
    top_k = min(top_k, scores.shape[-1])
    probs = torch.softmax(scores, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=top_k, dim=-1)

    patch_records = []
    flat_top_ids = top_ids.tolist()
    flat_top_probs = top_probs.tolist()

    for patch_idx, (patch_ids, patch_probs) in enumerate(zip(flat_top_ids, flat_top_probs)):
        row = patch_idx // patches_per_side if patches_per_side else None
        col = patch_idx % patches_per_side if patches_per_side else None
        candidates = []
        for token_id, prob in zip(patch_ids, patch_probs):
            candidates.append(
                {
                    "token_id": token_id,
                    "token": decode_token(tokenizer, token_id),
                    "prob": float(prob),
                }
            )
        patch_records.append(
            {
                "patch_index": patch_idx,
                "row": row,
                "col": col,
                "canvas_box": None if patch_geometry is None else patch_geometry[patch_idx]["canvas_box"],
                "image_box": None if patch_geometry is None else patch_geometry[patch_idx]["image_box"],
                "top_token_id": candidates[0]["token_id"],
                "top_token": candidates[0]["token"],
                "top_prob": candidates[0]["prob"],
                "top_k": candidates,
            }
        )

    return patch_records


def build_token_grid(patch_records, patches_per_side):
    if not patches_per_side:
        return [record["top_token"] for record in patch_records]

    grid = []
    for row_idx in range(patches_per_side):
        start = row_idx * patches_per_side
        end = start + patches_per_side
        grid.append([record["top_token"] for record in patch_records[start:end]])
    return grid


def make_display_token(token):
    token = token.replace("▁", " ").strip()
    token = token.replace("\t", " ").replace("\n", " ")
    if not token:
        token = "<sp>"
    return token


def singularize_token(token):
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("ses") or token.endswith("xes") or token.endswith("zes") or token.endswith("ches") or token.endswith("shes"):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize_match_token(token):
    token = make_display_token(token).lower()
    token = token.replace("-", " ")
    token = re.sub(r"[^a-z\s]", " ", token)
    token = re.sub(r"\s+", " ", token).strip()
    return token


def extract_image_id(image_file):
    matches = re.findall(r"\d+", os.path.basename(image_file))
    if not matches:
        return None
    return int(matches[-1])


def expand_match_terms(words):
    expanded = set()
    for item in words or []:
        entries = item if isinstance(item, (list, tuple)) else [item]
        for entry in entries:
            normalized = normalize_match_token(str(entry))
            if not normalized:
                continue
            expanded.add(normalized)
            for part in normalized.split():
                if not part:
                    continue
                expanded.add(part)
                expanded.add(singularize_token(part))
            singularized_phrase = " ".join(singularize_token(part) for part in normalized.split())
            if singularized_phrase:
                expanded.add(singularized_phrase)
    expanded.discard("")
    return expanded


def get_dataset_split(image_file):
    basename = os.path.basename(image_file).lower()
    if "val2014" in basename:
        return "val"
    if "train2014" in basename:
        return "train"
    raise ValueError(f"Could not infer COCO split from image filename: {image_file}")


def build_double_word_dict():
    coco_double_words = [
        "motor bike",
        "motor cycle",
        "air plane",
        "traffic light",
        "street light",
        "traffic signal",
        "stop light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "suit case",
        "sports ball",
        "baseball bat",
        "baseball glove",
        "tennis racket",
        "wine glass",
        "hot dog",
        "cell phone",
        "mobile phone",
        "teddy bear",
        "hair drier",
        "potted plant",
        "bow tie",
        "laptop computer",
        "stove top oven",
        "home plate",
        "train track",
    ]
    animal_words = [
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "animal",
        "cub",
    ]
    vehicle_words = ["jet", "train"]

    double_word_dict = {double_word: double_word for double_word in coco_double_words}
    for animal_word in animal_words:
        double_word_dict[f"baby {animal_word}"] = animal_word
        double_word_dict[f"adult {animal_word}"] = animal_word
    for vehicle_word in vehicle_words:
        double_word_dict[f"passenger {vehicle_word}"] = vehicle_word
    double_word_dict["bow tie"] = "tie"
    double_word_dict["toilet seat"] = "toilet"
    double_word_dict["wine glas"] = "wine glass"
    return double_word_dict


def load_coco_vocab():
    synonyms_path = os.path.join(
        os.path.dirname(__file__),
        "eval_utils",
        "data",
        "synonyms.txt",
    )
    with open(synonyms_path, "r", encoding="utf-8") as f:
        synonyms = [line.strip().split(", ") for line in f if line.strip()]

    mscoco_objects = set()
    inverse_synonym_dict = {}
    for synonym_group in synonyms:
        canonical = synonym_group[0]
        for synonym in synonym_group:
            normalized = normalize_match_token(synonym)
            if not normalized:
                continue
            mscoco_objects.add(normalized)
            inverse_synonym_dict[normalized] = canonical

    return {
        "mscoco_objects": mscoco_objects,
        "inverse_synonym_dict": inverse_synonym_dict,
        "double_word_dict": build_double_word_dict(),
    }


def canonicalize_coco_term(term, coco_vocab):
    normalized = normalize_match_token(term)
    if not normalized:
        return None

    variants = [normalized]
    singularized = " ".join(singularize_token(part) for part in normalized.split())
    if singularized and singularized != normalized:
        variants.append(singularized)

    for variant in variants:
        if variant in coco_vocab["inverse_synonym_dict"]:
            return coco_vocab["inverse_synonym_dict"][variant]
    return None


def caption_to_coco_nodes(caption, coco_vocab):
    words = re.findall(r"[a-z]+", caption.lower())
    i = 0
    merged_words = []
    while i < len(words):
        double_word = " ".join(words[i : i + 2])
        singular_double = " ".join(singularize_token(part) for part in double_word.split())
        if singular_double in coco_vocab["double_word_dict"]:
            merged_words.append(coco_vocab["double_word_dict"][singular_double])
            i += 2
        else:
            merged_words.append(words[i])
            i += 1

    if "toilet" in merged_words and "seat" in merged_words:
        merged_words = [word for word in merged_words if word != "seat"]

    node_words = []
    for word in merged_words:
        canonical = canonicalize_coco_term(word, coco_vocab)
        if canonical is not None:
            node_words.append(canonical)
    return node_words


def load_annotation_entry(annotation_dir, image_file):
    store = load_annotation_store(annotation_dir, [image_file])
    return get_annotation_entry_from_store(store, image_file)


def load_annotation_store(annotation_dir, image_files):
    annotation_dir = os.path.expanduser(annotation_dir)
    coco_vocab = load_coco_vocab()
    image_ids_by_split = {}
    for image_file in image_files:
        split = get_dataset_split(image_file)
        image_id = extract_image_id(image_file)
        if image_id is None:
            raise ValueError(f"Could not extract image id from {image_file}")
        image_ids_by_split.setdefault(split, set()).add(image_id)

    splits = {}
    for split, target_ids in image_ids_by_split.items():
        instances_path = os.path.join(annotation_dir, f"instances_{split}2014.json")
        captions_path = os.path.join(annotation_dir, f"captions_{split}2014.json")
        with open(instances_path, "r", encoding="utf-8") as f:
            instances_data = json.load(f)
        with open(captions_path, "r", encoding="utf-8") as f:
            captions_data = json.load(f)

        category_id_to_name = {category["id"]: category["name"] for category in instances_data["categories"]}
        instance_words_by_image = {image_id: set() for image_id in target_ids}
        gt_words_by_image = {image_id: set() for image_id in target_ids}
        caption_words_by_image = {image_id: set() for image_id in target_ids}

        for annotation in instances_data["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in target_ids:
                continue
            category_name = category_id_to_name[annotation["category_id"]]
            canonical = canonicalize_coco_term(category_name, coco_vocab)
            if canonical is not None:
                instance_words_by_image[image_id].add(canonical)
                gt_words_by_image[image_id].add(canonical)

        for annotation in captions_data["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in target_ids:
                continue
            caption_words = set(caption_to_coco_nodes(annotation["caption"], coco_vocab))
            caption_words_by_image[image_id].update(caption_words)
            gt_words_by_image[image_id].update(caption_words)

        splits[split] = {
            "instance_words_by_image": instance_words_by_image,
            "caption_words_by_image": caption_words_by_image,
            "gt_words_by_image": gt_words_by_image,
        }

    return {
        "annotation_dir": annotation_dir,
        "coco_vocab": coco_vocab,
        "splits": splits,
    }


def get_annotation_entry_from_store(annotation_store, image_file):
    split = get_dataset_split(image_file)
    image_id = extract_image_id(image_file)
    if image_id is None:
        raise ValueError(f"Could not extract image id from {image_file}")
    split_store = annotation_store["splits"].get(split)
    if split_store is None:
        raise ValueError(f"Split {split} was not loaded from {annotation_store['annotation_dir']}")

    return {
        "image_id": image_id,
        "image": os.path.basename(image_file),
        "split": split,
        "gt_words": sorted(split_store["gt_words_by_image"].get(image_id, set())),
        "instance_words": sorted(split_store["instance_words_by_image"].get(image_id, set())),
        "caption_words": sorted(split_store["caption_words_by_image"].get(image_id, set())),
        "coco_vocab": annotation_store["coco_vocab"],
    }


def annotate_patch_records_with_annotations(patch_records, annotation_entry):
    gt_terms = set(annotation_entry["gt_words"])
    coco_vocab = annotation_entry["coco_vocab"]

    category_counts = Counter()
    matched_words = {
        "hallucinated": Counter(),
        "gt": Counter(),
    }
    hallucinated_patch_indices = []
    gt_patch_indices = []

    for record in patch_records:
        normalized_candidates = []
        canonical_candidates = set()
        for candidate in record["top_k"]:
            normalized = normalize_match_token(candidate["token"])
            if normalized:
                normalized_candidates.append(normalized)
                singularized = " ".join(singularize_token(part) for part in normalized.split())
                if singularized and singularized != normalized:
                    normalized_candidates.append(singularized)
            canonical = canonicalize_coco_term(candidate["token"], coco_vocab)
            if canonical is not None:
                canonical_candidates.add(canonical)

        normalized_candidates = sorted(set(normalized_candidates))
        canonical_candidates = sorted(canonical_candidates)

        gt_hits = sorted(set(canonical_candidates) & gt_terms)
        hallucinated_hits = sorted(set(canonical_candidates) - gt_terms)

        if gt_hits:
            match_label = "gt"
            gt_patch_indices.append(record["patch_index"])
            for word in gt_hits:
                matched_words["gt"][word] += 1
        elif hallucinated_hits:
            match_label = "hallucinated"
            hallucinated_patch_indices.append(record["patch_index"])
            for word in hallucinated_hits:
                matched_words["hallucinated"][word] += 1
        else:
            match_label = "other"

        record["top_token_normalized"] = normalize_match_token(record["top_token"])
        record["top_k_normalized"] = normalized_candidates
        record["annotation_canonical_hits"] = canonical_candidates
        record["annotation_match_label"] = match_label
        record["annotation_hallucinated_hits"] = hallucinated_hits
        record["annotation_gt_hits"] = gt_hits
        category_counts[match_label] += 1

    summary = {
        "image_id": annotation_entry["image_id"],
        "image": annotation_entry["image"],
        "split": annotation_entry["split"],
        "gt_words": annotation_entry["gt_words"],
        "instance_words": annotation_entry["instance_words"],
        "caption_words": annotation_entry["caption_words"],
        "matched_patch_counts": dict(category_counts),
        "matched_words": {
            key: value.most_common()
            for key, value in matched_words.items()
        },
        "hallucinated_patch_indices": hallucinated_patch_indices,
        "gt_patch_indices": gt_patch_indices,
    }
    return summary


def load_chair_entry(chair_results_file, image_file):
    chair_index = load_chair_results_index(chair_results_file)
    return get_chair_entry_from_index(chair_index, image_file)


def load_chair_results_index(chair_results_file):
    chair_results_file = os.path.expanduser(chair_results_file)
    with open(chair_results_file, "r", encoding="utf-8") as f:
        chair_results = json.load(f)

    sentences = chair_results.get("sentences", [])
    by_image_name = {}
    by_image_id = {}
    for sentence in sentences:
        image_name = sentence.get("image")
        image_id = sentence.get("image_id")
        if image_name is not None:
            by_image_name[image_name] = sentence
        if image_id is not None:
            by_image_id[int(image_id)] = sentence

    return {
        "path": chair_results_file,
        "sentences": sentences,
        "by_image_name": by_image_name,
        "by_image_id": by_image_id,
    }


def get_chair_entry_from_index(chair_index, image_file):
    image_basename = os.path.basename(image_file)
    image_id = extract_image_id(image_file)
    if image_basename in chair_index["by_image_name"]:
        return chair_index["by_image_name"][image_basename]
    if image_id is not None and image_id in chair_index["by_image_id"]:
        return chair_index["by_image_id"][image_id]
    return None


def annotate_patch_records_with_chair_results(patch_records, chair_entry):
    hallucinated_terms = expand_match_terms(chair_entry.get("mscoco_hallucinated_words", []))
    non_hallucinated_terms = expand_match_terms(chair_entry.get("mscoco_non_hallucinated_words", []))
    gt_terms = expand_match_terms(chair_entry.get("mscoco_gt_words", []))
    generated_terms = expand_match_terms(chair_entry.get("mscoco_generated_words", []))

    category_counts = Counter()
    matched_words = {
        "hallucinated": Counter(),
        "non_hallucinated": Counter(),
        "gt": Counter(),
    }
    hallucinated_patch_indices = []
    non_hallucinated_patch_indices = []
    gt_patch_indices = []

    for record in patch_records:
        normalized_candidates = set(record.get("top_k_normalized", []))
        if not normalized_candidates:
            for candidate in record["top_k"]:
                normalized = normalize_match_token(candidate["token"])
                if not normalized:
                    continue
                normalized_candidates.add(normalized)
                singularized = " ".join(singularize_token(part) for part in normalized.split())
                if singularized:
                    normalized_candidates.add(singularized)

        hallucinated_hits = sorted(normalized_candidates & hallucinated_terms)
        non_hallucinated_hits = sorted(normalized_candidates & non_hallucinated_terms)
        gt_hits = sorted(normalized_candidates & gt_terms)

        if hallucinated_hits:
            match_label = "hallucinated"
            hallucinated_patch_indices.append(record["patch_index"])
            for word in hallucinated_hits:
                matched_words["hallucinated"][word] += 1
        elif non_hallucinated_hits:
            match_label = "non_hallucinated"
            non_hallucinated_patch_indices.append(record["patch_index"])
            for word in non_hallucinated_hits:
                matched_words["non_hallucinated"][word] += 1
        elif gt_hits:
            match_label = "gt"
            gt_patch_indices.append(record["patch_index"])
            for word in gt_hits:
                matched_words["gt"][word] += 1
        else:
            match_label = "other"

        record["chair_match_label"] = match_label
        record["chair_hallucinated_hits"] = hallucinated_hits
        record["chair_non_hallucinated_hits"] = non_hallucinated_hits
        record["chair_gt_hits"] = gt_hits
        category_counts[match_label] += 1

    summary = {
        "image_id": chair_entry.get("image_id"),
        "image": chair_entry.get("image"),
        "caption": chair_entry.get("caption"),
        "hallucinated_words": sorted(hallucinated_terms),
        "non_hallucinated_words": sorted(non_hallucinated_terms),
        "gt_words": sorted(gt_terms),
        "generated_words": sorted(generated_terms),
        "matched_patch_counts": dict(category_counts),
        "matched_words": {
            key: value.most_common()
            for key, value in matched_words.items()
        },
        "hallucinated_patch_indices": hallucinated_patch_indices,
        "non_hallucinated_patch_indices": non_hallucinated_patch_indices,
        "gt_patch_indices": gt_patch_indices,
    }
    return summary


def clamp01(value):
    return max(0.0, min(1.0, value))


def get_prob_normalizer(patch_records):
    probs = [record["top_prob"] for record in patch_records if record["image_box"] is not None]
    if not probs:
        return 0.0, 1.0
    min_prob = min(probs)
    max_prob = max(probs)
    if max_prob - min_prob < 1e-8:
        max_prob = min_prob + 1.0
    return min_prob, max_prob


def get_patch_style(prob, min_prob, max_prob):
    norm = clamp01((prob - min_prob) / (max_prob - min_prob))
    outline = (
        int(90 + 120 * norm),
        int(140 + 80 * norm),
        int(255 - 30 * norm),
        int(150 + 70 * norm),
    )
    fill = (
        int(40 + 40 * norm),
        int(110 + 80 * norm),
        int(255 - 20 * norm),
        int(25 + 80 * norm),
    )
    label_bg = (
        int(8 + 20 * norm),
        int(20 + 25 * norm),
        int(45 + 35 * norm),
        185,
    )
    return outline, fill, label_bg, norm


def get_scaled_image(image, scale):
    if scale == 1:
        return image.copy()
    resampling = getattr(Image, "Resampling", Image).NEAREST
    return image.resize((image.width * scale, image.height * scale), resampling)


def get_overlay_font(font_size):
    for font_name in ["DejaVuSansMono.ttf", "DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(font_name, font_size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_overlay_image(image, patch_records, output_path, scale=1):
    overlay = get_scaled_image(image, scale)
    draw = ImageDraw.Draw(overlay, "RGBA")
    font_size = max(12, 10 * scale)
    font = get_overlay_font(font_size)
    line_width = max(1, scale)
    min_prob, max_prob = get_prob_normalizer(patch_records)
    rounded_radius = max(3, 2 * scale)

    for record in sorted(patch_records, key=lambda record: record["top_prob"]):
        image_box = record["image_box"]
        if image_box is None:
            continue

        left, top, right, bottom = image_box
        box = (
            int(left * scale),
            int(top * scale),
            int(right * scale),
            int(bottom * scale),
        )
        outline, fill, label_bg, norm = get_patch_style(record["top_prob"], min_prob, max_prob)
        draw.rounded_rectangle(box, radius=rounded_radius, fill=fill, outline=outline, width=line_width)

        label = make_display_token(record["top_token"])[:12]
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        padding = max(2, scale)
        text_x = box[0] + padding
        text_y = box[1] + padding
        text_bg = (
            text_x,
            text_y,
            min(text_x + text_width + 2 * padding, overlay.width),
            min(text_y + text_height + 2 * padding, overlay.height),
        )
        draw.rounded_rectangle(text_bg, radius=max(2, scale), fill=label_bg)
        draw.text(
            (text_x + padding // 2, text_y),
            label,
            fill=(255, 255, 255, 255),
            font=font,
            stroke_width=max(1, scale // 2),
            stroke_fill=(10, 18, 30, 220),
        )

        prob_bar_height = max(2, scale)
        prob_bar_width = max(4, int((box[2] - box[0]) * norm))
        if prob_bar_width > 0:
            prob_bar_left = box[0] + line_width
            prob_bar_top = max(box[3] - prob_bar_height - line_width, box[1])
            prob_bar_right = min(box[0] + prob_bar_width, box[2] - line_width)
            prob_bar_bottom = box[3] - line_width
            prob_bar = (
                prob_bar_left,
                prob_bar_top,
                prob_bar_right,
                prob_bar_bottom,
            )
            if prob_bar_right > prob_bar_left and prob_bar_bottom > prob_bar_top:
                prob_bar_radius = min(
                    max(1, scale),
                    max(1, (prob_bar_right - prob_bar_left) // 2),
                    max(1, (prob_bar_bottom - prob_bar_top) // 2),
                )
                draw.rounded_rectangle(prob_bar, radius=prob_bar_radius, fill=(255, 255, 255, 160))

    overlay.save(output_path)
    return output_path


def build_annotation_match_overlay(image, patch_records, output_path, scale=1):
    overlay = get_scaled_image(image, scale)
    draw = ImageDraw.Draw(overlay, "RGBA")
    font_size = max(12, 10 * scale)
    font = get_overlay_font(font_size)
    line_width = max(1, scale)

    style_map = {
        "hallucinated": {
            "outline": (235, 87, 87, 235),
            "fill": (235, 87, 87, 80),
            "label_bg": (88, 24, 24, 205),
        },
        "gt": {
            "outline": (64, 135, 255, 235),
            "fill": (64, 135, 255, 60),
            "label_bg": (17, 41, 87, 205),
        },
    }

    records_to_draw = [record for record in patch_records if record.get("annotation_match_label") in style_map]
    for record in sorted(records_to_draw, key=lambda record: {"gt": 0, "hallucinated": 1}[record["annotation_match_label"]]):
        image_box = record["image_box"]
        if image_box is None:
            continue
        style = style_map[record["annotation_match_label"]]
        left, top, right, bottom = image_box
        box = (
            int(left * scale),
            int(top * scale),
            int(right * scale),
            int(bottom * scale),
        )
        draw.rounded_rectangle(box, radius=max(3, 2 * scale), fill=style["fill"], outline=style["outline"], width=line_width)

        if record["annotation_match_label"] == "hallucinated":
            hits = record.get("annotation_hallucinated_hits", [])
        else:
            hits = record.get("annotation_gt_hits", [])

        label = hits[0] if hits else make_display_token(record["top_token"])
        label = label[:14]
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        padding = max(2, scale)
        text_bg = (
            box[0] + padding,
            box[1] + padding,
            min(box[0] + padding + text_width + 2 * padding, overlay.width),
            min(box[1] + padding + text_height + 2 * padding, overlay.height),
        )
        draw.rounded_rectangle(text_bg, radius=max(2, scale), fill=style["label_bg"])
        draw.text(
            (text_bg[0] + padding // 2, text_bg[1]),
            label,
            fill=(255, 255, 255, 255),
            font=font,
            stroke_width=max(1, scale // 2),
            stroke_fill=(18, 18, 18, 220),
        )

    overlay.save(output_path)
    return output_path


def build_chair_match_overlay(image, patch_records, output_path, scale=1):
    overlay = get_scaled_image(image, scale)
    draw = ImageDraw.Draw(overlay, "RGBA")
    font_size = max(12, 10 * scale)
    font = get_overlay_font(font_size)
    line_width = max(1, scale)

    style_map = {
        "hallucinated": {
            "outline": (235, 87, 87, 235),
            "fill": (235, 87, 87, 80),
            "label_bg": (88, 24, 24, 205),
        },
        "non_hallucinated": {
            "outline": (56, 176, 119, 235),
            "fill": (56, 176, 119, 70),
            "label_bg": (18, 61, 41, 205),
        },
        "gt": {
            "outline": (64, 135, 255, 235),
            "fill": (64, 135, 255, 60),
            "label_bg": (17, 41, 87, 205),
        },
    }

    records_to_draw = [record for record in patch_records if record.get("chair_match_label") in style_map]
    order = {"gt": 0, "non_hallucinated": 1, "hallucinated": 2}
    for record in sorted(records_to_draw, key=lambda record: order[record["chair_match_label"]]):
        image_box = record["image_box"]
        if image_box is None:
            continue
        style = style_map[record["chair_match_label"]]
        left, top, right, bottom = image_box
        box = (
            int(left * scale),
            int(top * scale),
            int(right * scale),
            int(bottom * scale),
        )
        draw.rounded_rectangle(box, radius=max(3, 2 * scale), fill=style["fill"], outline=style["outline"], width=line_width)

        if record["chair_match_label"] == "hallucinated":
            hits = record.get("chair_hallucinated_hits", [])
        elif record["chair_match_label"] == "non_hallucinated":
            hits = record.get("chair_non_hallucinated_hits", [])
        else:
            hits = record.get("chair_gt_hits", [])

        label = (hits[0] if hits else make_display_token(record["top_token"]))[:14]
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        padding = max(2, scale)
        text_bg = (
            box[0] + padding,
            box[1] + padding,
            min(box[0] + padding + text_width + 2 * padding, overlay.width),
            min(box[1] + padding + text_height + 2 * padding, overlay.height),
        )
        draw.rounded_rectangle(text_bg, radius=max(2, scale), fill=style["label_bg"])
        draw.text(
            (text_bg[0] + padding // 2, text_bg[1]),
            label,
            fill=(255, 255, 255, 255),
            font=font,
            stroke_width=max(1, scale // 2),
            stroke_fill=(18, 18, 18, 220),
        )

    overlay.save(output_path)
    return output_path


def write_outputs(output_dir, image_file, score_mode, raw_patch_features, projected_patch_features, patch_records, token_grid, annotation_summary=None, chair_summary=None):
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "image_file": image_file,
        "score_mode": score_mode,
        "num_patches": len(patch_records),
        "raw_patch_feature_shape": list(raw_patch_features.shape),
        "projected_patch_feature_shape": list(projected_patch_features.shape),
        "top_token_counter": Counter(record["top_token"] for record in patch_records).most_common(50),
        "patches": patch_records,
        "token_grid": token_grid,
    }
    if annotation_summary is not None:
        metadata["annotation_summary"] = annotation_summary
    if chair_summary is not None:
        metadata["chair_summary"] = chair_summary

    json_path = os.path.join(output_dir, "patch_token_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    txt_path = os.path.join(output_dir, "patch_token_grid.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        if token_grid and isinstance(token_grid[0], list):
            for row in token_grid:
                f.write("\t".join(row) + "\n")
        else:
            for token in token_grid:
                f.write(f"{token}\n")

    return json_path, txt_path


def print_summary(patch_records):
    token_counter = Counter(record["top_token"] for record in patch_records)
    print("Top repeated patch tokens:")
    for token, count in token_counter.most_common(20):
        print(f"  {token!r}: {count}")

    print("\nFirst 10 patches:")
    for record in patch_records[:10]:
        print(
            f"  patch={record['patch_index']:>3} "
            f"pos=({record['row']},{record['col']}) "
            f"token={record['top_token']!r} "
            f"prob={record['top_prob']:.6f}"
        )


def print_annotation_summary(annotation_summary):
    print("\nAnnotation patch matches:")
    for label in ["hallucinated", "gt", "other"]:
        print(f"  {label}: {annotation_summary['matched_patch_counts'].get(label, 0)}")
    hallucinated_words = annotation_summary["matched_words"]["hallucinated"]
    if hallucinated_words:
        print("  hallucinated hits:", ", ".join(f"{word}({count})" for word, count in hallucinated_words[:10]))


def print_chair_summary(chair_summary):
    print("\nCHAIR patch matches:")
    for label in ["hallucinated", "non_hallucinated", "gt", "other"]:
        print(f"  {label}: {chair_summary['matched_patch_counts'].get(label, 0)}")
    hallucinated_words = chair_summary["matched_words"]["hallucinated"]
    if hallucinated_words:
        print("  hallucinated hits:", ", ".join(f"{word}({count})" for word, count in hallucinated_words[:10]))


def load_model_components(args):
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    device_map = resolve_device_map(args.device)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=args.model_base,
        model_name=model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device_map=device_map,
        device=args.device,
    )
    model.eval()
    return tokenizer, model, image_processor


def maybe_empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def analyze_single_image(
    image_file,
    output_dir,
    tokenizer,
    model,
    image_processor,
    args,
    annotation_entry=None,
    chair_entry=None,
    verbose=True,
    save_outputs=True,
):
    image_file = os.path.expanduser(image_file)
    image = load_image(image_file)
    image_tensor = prepare_image_tensor(
        image,
        image_processor,
        model.config,
        model.get_vision_tower().device,
    )

    raw_patch_features, projected_patch_features = get_projected_patch_features(model, image_tensor)
    scores = project_to_vocab(model, projected_patch_features[0], args.score_mode)

    vision_tower = model.get_vision_tower()
    patches_per_side = getattr(vision_tower, "num_patches_per_side", None)
    if patches_per_side is not None and patches_per_side * patches_per_side != scores.shape[0]:
        patches_per_side = None
    patch_geometry = get_patch_geometry(
        image_size=image.size,
        patches_per_side=patches_per_side,
        image_aspect_ratio=getattr(model.config, "image_aspect_ratio", None),
    )

    patch_records = build_patch_records(
        tokenizer=tokenizer,
        scores=scores.float().cpu(),
        top_k=args.top_k,
        patches_per_side=patches_per_side,
        patch_geometry=patch_geometry,
    )
    token_grid = build_token_grid(patch_records, patches_per_side)
    annotation_summary = None
    if annotation_entry is not None:
        annotation_summary = annotate_patch_records_with_annotations(patch_records, annotation_entry)
    chair_summary = None
    if chair_entry is not None:
        chair_summary = annotate_patch_records_with_chair_results(patch_records, chair_entry)

    json_path = None
    txt_path = None
    if save_outputs:
        json_path, txt_path = write_outputs(
            output_dir=os.path.expanduser(output_dir),
            image_file=image_file,
            score_mode=args.score_mode,
            raw_patch_features=raw_patch_features.cpu(),
            projected_patch_features=projected_patch_features.cpu(),
            patch_records=patch_records,
            token_grid=token_grid,
            annotation_summary=annotation_summary,
            chair_summary=chair_summary,
        )
    if verbose:
        print_summary(patch_records)
        if annotation_summary is not None:
            print_annotation_summary(annotation_summary)
        if chair_summary is not None:
            print_chair_summary(chair_summary)
        if save_outputs:
            print(f"\nSaved JSON to: {json_path}")
            print(f"Saved token grid to: {txt_path}")
    if save_outputs and args.save_overlay:
        overlay_path = build_overlay_image(
            image=image,
            patch_records=patch_records,
            output_path=os.path.join(os.path.expanduser(output_dir), "patch_token_overlay.png"),
            scale=1,
        )
        large_overlay_path = build_overlay_image(
            image=image,
            patch_records=patch_records,
            output_path=os.path.join(os.path.expanduser(output_dir), "patch_token_overlay_large.png"),
            scale=max(1, args.overlay_scale),
        )
        if verbose:
            print(f"Saved overlay to: {overlay_path}")
            print(f"Saved large overlay to: {large_overlay_path}")
        if annotation_summary is not None:
            annotation_overlay_path = build_annotation_match_overlay(
                image=image,
                patch_records=patch_records,
                output_path=os.path.join(os.path.expanduser(output_dir), "patch_annotation_match_overlay.png"),
                scale=1,
            )
            annotation_large_overlay_path = build_annotation_match_overlay(
                image=image,
                patch_records=patch_records,
                output_path=os.path.join(os.path.expanduser(output_dir), "patch_annotation_match_overlay_large.png"),
                scale=max(1, args.overlay_scale),
            )
            if verbose:
                print(f"Saved annotation overlay to: {annotation_overlay_path}")
                print(f"Saved large annotation overlay to: {annotation_large_overlay_path}")
        if chair_summary is not None:
            chair_overlay_path = build_chair_match_overlay(
                image=image,
                patch_records=patch_records,
                output_path=os.path.join(os.path.expanduser(output_dir), "patch_chair_match_overlay.png"),
                scale=1,
            )
            chair_large_overlay_path = build_chair_match_overlay(
                image=image,
                patch_records=patch_records,
                output_path=os.path.join(os.path.expanduser(output_dir), "patch_chair_match_overlay_large.png"),
                scale=max(1, args.overlay_scale),
            )
            if verbose:
                print(f"Saved CHAIR overlay to: {chair_overlay_path}")
                print(f"Saved large CHAIR overlay to: {chair_large_overlay_path}")

    maybe_empty_cuda_cache()
    return {
        "image_file": image_file,
        "json_path": json_path,
        "txt_path": txt_path,
        "annotation_summary": annotation_summary,
        "chair_summary": chair_summary,
    }


def run_single_mode(args, tokenizer, model, image_processor):
    annotation_entry = None
    if args.annotation_dir:
        annotation_entry = load_annotation_entry(args.annotation_dir, args.image_file)
    chair_entry = None
    if args.chair_results_file:
        chair_entry = load_chair_entry(args.chair_results_file, args.image_file)
        if chair_entry is None:
            raise ValueError(f"Could not find {os.path.basename(args.image_file)} in {args.chair_results_file}")
    analyze_single_image(
        image_file=args.image_file,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        args=args,
        annotation_entry=annotation_entry,
        chair_entry=chair_entry,
        verbose=True,
        save_outputs=True,
    )


def run_batch_mode(args, tokenizer, model, image_processor):
    chair_index = load_chair_results_index(args.chair_results_file)
    sentences = chair_index["sentences"]
    if args.max_images is not None:
        sentences = sentences[:args.max_images]

    image_folder = os.path.expanduser(args.image_folder)
    output_root = os.path.expanduser(args.output_dir)
    os.makedirs(output_root, exist_ok=True)

    image_files = [os.path.join(image_folder, sentence["image"]) for sentence in sentences]
    annotation_store = None
    if args.annotation_dir:
        annotation_store = load_annotation_store(args.annotation_dir, image_files)

    batch_results = []
    skipped_images = []
    for sentence in tqdm(sentences, desc="Analyzing images"):
        image_name = sentence["image"]
        image_file = os.path.join(image_folder, image_name)
        if not os.path.exists(image_file):
            skipped_images.append({"image": image_name, "reason": "missing_image"})
            continue

        image_stem = os.path.splitext(image_name)[0]
        image_output_dir = os.path.join(output_root, image_stem)
        result_json_path = os.path.join(image_output_dir, "patch_token_analysis.json")
        if args.skip_existing and args.save_per_image and os.path.exists(result_json_path):
            skipped_images.append({"image": image_name, "reason": "existing_output"})
            continue

        annotation_entry = None
        if annotation_store is not None:
            annotation_entry = get_annotation_entry_from_store(annotation_store, image_file)

        result = analyze_single_image(
            image_file=image_file,
            output_dir=image_output_dir,
            tokenizer=tokenizer,
            model=model,
            image_processor=image_processor,
            args=args,
            annotation_entry=annotation_entry,
            chair_entry=sentence,
            verbose=False,
            save_outputs=args.save_per_image,
        )
        batch_results.append(
            {
                "image": image_name,
                "image_file": result["image_file"],
                "output_dir": image_output_dir if args.save_per_image else None,
                "annotation_summary": result["annotation_summary"],
                "chair_summary": result["chair_summary"],
            }
        )

    summary = {
        "num_processed": len(batch_results),
        "num_skipped": len(skipped_images),
        "skipped_images": skipped_images,
        "images": batch_results,
    }
    summary_path = os.path.join(output_root, "batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(batch_results)} images.")
    if skipped_images:
        print(f"Skipped {len(skipped_images)} images.")
    print(f"Saved batch summary to: {summary_path}")


def main():
    args = parse_args()
    validate_args(args)
    disable_torch_init()
    tokenizer, model, image_processor = load_model_components(args)
    if is_batch_mode(args):
        run_batch_mode(args, tokenizer, model, image_processor)
    else:
        run_single_mode(args, tokenizer, model, image_processor)


if __name__ == "__main__":
    main()
