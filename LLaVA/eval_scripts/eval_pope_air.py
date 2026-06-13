import argparse
import json
import math
import os

import shortuuid
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


def split_list(lst, n):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.questions[index]
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
        return len(self.questions)


def collate_fn(batch, pad_token_id):
    input_ids, image_tensors, image_sizes, lines = zip(*batch)
    input_ids = [x.squeeze(0) for x in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = input_ids.ne(pad_token_id)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, attention_mask, image_tensors, image_sizes, lines


def create_data_loader(
    questions, image_folder, tokenizer, image_processor, model_config, conv_mode, batch_size=1, num_workers=4
):
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config, conv_mode)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id),
    )
    return data_loader


def _configure_llava_hal_heads_and_vision_layout(model_path: str, model_config) -> None:
    """Set hal_attention_heads and img_start_pos/img_length (required by conditional AD-HH)."""
    if model_path == "liuhaotian/llava-v1.5-7b":
        model_config.hal_attention_heads = [
            [16, 29],
            [26, 9],
            [13, 31],
            [15, 10],
            [20, 12],
            [30, 9],
            [19, 18],
            [17, 0],
            [18, 9],
            [26, 28],
            [19, 27],
            [18, 26],
            [15, 25],
            [14, 16],
            [31, 26],
            [15, 24],
            [31, 3],
            [22, 20],
            [27, 29],
            [17, 28],
        ]
        model_config.img_start_pos = 35
        model_config.img_length = 576
    elif model_path == "liuhaotian/llava-v1.5-13b":
        model_config.hal_attention_heads = [
            [0, 8],
            [29, 27],
            [23, 18],
            [20, 11],
            [36, 26],
            [19, 37],
            [22, 16],
            [22, 34],
            [21, 31],
            [20, 34],
            [37, 11],
            [17, 25],
            [35, 10],
            [17, 5],
            [15, 26],
            [0, 22],
            [19, 5],
            [19, 0],
            [14, 1],
            [23, 20],
            [21, 6],
            [30, 24],
            [26, 27],
            [21, 32],
            [15, 28],
            [15, 31],
            [19, 30],
            [20, 8],
            [19, 14],
            [14, 9],
            [39, 26],
            [25, 1],
            [18, 32],
            [17, 27],
            [39, 32],
        ]
        model_config.img_start_pos = 35
        model_config.img_length = 576
    elif model_path == "liuhaotian/llava-v1.6-34b":
        model_config.hal_attention_heads = [
            [45, 34],
            [43, 4],
            [43, 48],
            [44, 29],
            [35, 47],
            [40, 27],
            [54, 34],
            [37, 48],
            [43, 2],
            [41, 34],
        ]
        model_config.img_start_pos = 33
        model_config.img_length = 1948
    else:
        raise ValueError(f"Unsupported model for hal-head / vision layout: {model_path}")


def add_air_arguments(parser) -> None:
    """Register all AIR CLI flags (core regularization + internal module hyperparameters)."""
    parser.add_argument(
        "--air",
        action="store_true",
        default=False,
        dest="air",
        help="Enable AIR (modality rebalancing / cross-head lens / conditional AD-HH + variance projection)",
    )
    parser.add_argument(
        "--air-beta",
        type=float,
        default=0.1,
        dest="air_beta",
        help="Shrinkage β∈[0,1]; larger → closer to mean matrix (default 0.1)",
    )
    parser.add_argument(
        "--air-eps",
        type=float,
        default=1e-8,
        dest="air_eps",
        help="Numerical stability ε for Frobenius / spectral rescale (default 1e-8)",
    )
    parser.add_argument(
        "--air-layer-low",
        type=int,
        default=5,
        dest="air_layer_low",
        help="AIR regularization layer range lower bound (default 5)",
    )
    parser.add_argument(
        "--air-layer-high",
        type=int,
        default=18,
        dest="air_layer_high",
        help="AIR regularization layer range upper bound (default 18)",
    )
    parser.add_argument(
        "--air-qk-rescale",
        action="store_true",
        default=False,
        dest="air_qk_rescale",
        help="Enable line-1 pre-softmax QK spectral-energy rescale (default off)",
    )
    parser.add_argument(
        "--air-qk-scale",
        type=float,
        default=1.0,
        dest="air_qk_scale",
        help="Line-1 target energy relative to cross-head mean (default 1.0)",
    )
    # Internal module hyperparameters: modality rebalancing / cross-head lens / conditional AD-HH
    parser.add_argument("--air-gamma-img", type=float, default=1.08, dest="air_gamma_img",
                        help="Modality rebalance: image attention gain g (default 1.08)")
    parser.add_argument("--air-delta-sys", type=float, default=0.97, dest="air_delta_sys",
                        help="Modality rebalance: system prompt suppression d (default 0.97)")
    parser.add_argument("--air-mod-layer-low", type=int, default=9, dest="air_mod_layer_low",
                        help="Modality rebalance layer range lower bound (default 9)")
    parser.add_argument("--air-mod-layer-high", type=int, default=15, dest="air_mod_layer_high",
                        help="Modality rebalance layer range upper bound (default 15)")
    parser.add_argument("--air-lens-layer-low", type=int, default=5, dest="air_lens_layer_low",
                        help="Cross-head lens layer range lower bound (default 5)")
    parser.add_argument("--air-lens-layer-high", type=int, default=18, dest="air_lens_layer_high",
                        help="Cross-head lens layer range upper bound (default 18)")
    parser.add_argument("--air-alpha-lens", type=float, default=0.28, dest="air_alpha_lens",
                        help="Cross-head lens mix coefficient α (default 0.28)")
    parser.add_argument("--air-adhh-threshold", type=float, default=0.5, dest="air_adhh_threshold",
                        help="Conditional AD-HH trigger threshold (default 0.5)")
    parser.add_argument("--air-no-conditional-adhh", action="store_true", default=False,
                        dest="air_no_conditional_adhh",
                        help="Disable post-softmax conditional AD-HH; keep rebalance + lens only")
    parser.add_argument("--air-gamma-schedule", type=str, default="const",
                        choices=("const", "exp", "log"), dest="air_gamma_schedule",
                        help="Dynamic vision boost schedule (decode-time): const|exp|log (default const)")
    parser.add_argument("--air-gamma-img-max", type=float, default=None, dest="air_gamma_img_max",
                        help="Dynamic g ceiling; None → constant --air-gamma-img")
    parser.add_argument("--air-gamma-tau", type=float, default=32.0, dest="air_gamma_tau",
                        help="Dynamic vision boost time constant τ in tokens (default 32)")
    parser.add_argument("--air-gamma-kappa", type=float, default=0.05, dest="air_gamma_kappa",
                        help="Log schedule slope κ (default 0.05)")


def air_kwargs_from_args(args) -> dict:
    """Extract all AIR hyperparameters from CLI args (core regularization + internal modules)."""
    return dict(
        air_beta=getattr(args, "air_beta", 0.1),
        air_eps=getattr(args, "air_eps", 1e-8),
        air_layer_low=getattr(args, "air_layer_low", 5),
        air_layer_high=getattr(args, "air_layer_high", 18),
        air_qk_rescale=getattr(args, "air_qk_rescale", False),
        air_qk_scale=getattr(args, "air_qk_scale", 1.0),
        gamma_img=getattr(args, "air_gamma_img", 1.08),
        delta_sys=getattr(args, "air_delta_sys", 0.97),
        mod_layer_low=getattr(args, "air_mod_layer_low", 9),
        mod_layer_high=getattr(args, "air_mod_layer_high", 15),
        lens_layer_low=getattr(args, "air_lens_layer_low", 5),
        lens_layer_high=getattr(args, "air_lens_layer_high", 18),
        alpha_lens=getattr(args, "air_alpha_lens", 0.28),
        adhh_threshold=getattr(args, "air_adhh_threshold", 0.5),
        conditional_adhh=not getattr(args, "air_no_conditional_adhh", False),
        gamma_schedule=getattr(args, "air_gamma_schedule", "const"),
        gamma_img_max=getattr(args, "air_gamma_img_max", None),
        gamma_tau=getattr(args, "air_gamma_tau", 32.0),
        gamma_kappa=getattr(args, "air_gamma_kappa", 0.05),
    )


def apply_air_intervention(
    model_path: str,
    model_config,
    *,
    air_beta: float = 0.1,
    air_eps: float = 1e-8,
    air_layer_low: int = 5,
    air_layer_high: int = 18,
    air_qk_rescale: bool = False,
    air_qk_scale: float = 1.0,
    gamma_img: float = 1.08,
    delta_sys: float = 0.97,
    mod_layer_low: int = 9,
    mod_layer_high: int = 15,
    lens_layer_low: int = 5,
    lens_layer_high: int = 18,
    alpha_lens: float = 0.28,
    adhh_threshold: float = 0.5,
    conditional_adhh: bool = True,
    gamma_schedule: str = "const",
    gamma_img_max=None,
    gamma_tau: float = 32.0,
    gamma_kappa: float = 0.05,
) -> None:
    """Enable AIR, the sole decoding intervention (see ``llava.model.air_method`` and ``air_method.md``).

    Modules applied in order inside LlamaAttention forward:
    1. (Optional) line-1 QK spectral-energy rescale (``air_qk_rescale``, pre-softmax);
    2. Modality rebalancing: boost image keys, suppress system prompt logits (optional dynamic vision boost);
    3. Cross-head vision lens: cross-head mix over visual keys for the current step;
    4. Conditional AD-HH: zero post-image text attention on hallucination heads when over threshold;
    5. Variance-constrained projection: zero-trace + Frobenius rescale + mean shrinkage (paper §5.2).

    Dynamic vision boost (bounded schedule for g in module 2, decode-time only):
    - ``gamma_schedule``: "const"|"exp"|"log"; const = constant g.
    - ``gamma_img_max``: ceiling; None → constant g.
    - ``gamma_tau``: time constant (generated tokens), saturation speed.
    - ``gamma_kappa``: slope for log schedule.
    """
    _configure_llava_hal_heads_and_vision_layout(model_path, model_config)
    model_config.air_intervention = True
    # Core regularization + optional line 1
    model_config.air_beta = float(air_beta)
    model_config.air_eps = float(air_eps)
    model_config.air_layer_low = int(air_layer_low)
    model_config.air_layer_high = int(air_layer_high)
    model_config.air_qk_rescale = bool(air_qk_rescale)
    model_config.air_qk_scale = float(air_qk_scale)
    # Module 2: modality rebalancing
    model_config.air_gamma_img = gamma_img
    model_config.air_delta_sys = delta_sys
    model_config.air_mod_layer_low = mod_layer_low
    model_config.air_mod_layer_high = mod_layer_high
    model_config.air_gamma_schedule = gamma_schedule
    model_config.air_gamma_img_max = float(gamma_img_max) if gamma_img_max is not None else float(gamma_img)
    model_config.air_gamma_tau = float(gamma_tau)
    model_config.air_gamma_kappa = float(gamma_kappa)
    # Module 3: cross-head vision lens
    model_config.air_lens_layer_low = lens_layer_low
    model_config.air_lens_layer_high = lens_layer_high
    model_config.air_alpha_lens = alpha_lens
    # Module 4: conditional AD-HH
    model_config.air_adhh_threshold = adhh_threshold
    model_config.air_conditional_adhh = conditional_adhh


def eval_model(args):
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if "plain" in model_name and "finetune" not in model_name.lower() and "mmtag" not in args.conv_mode:
        args.conv_mode = args.conv_mode + "_mmtag"
        print(f"It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.")

    if getattr(args, "air", False):
        apply_air_intervention(model_path, model.config, **air_kwargs_from_args(args))

    data_loader = create_data_loader(
        questions,
        args.image_folder,
        tokenizer,
        image_processor,
        model.config,
        args.conv_mode,
        batch_size=args.batch_size,
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
                    }
                )
                + "\n"
            )
            ans_file.flush()

    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answers.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    # AIR: sole decoding intervention (core regularization + internal module hyperparameters)
    add_air_arguments(parser)
    args = parser.parse_args()

    eval_model(args)
