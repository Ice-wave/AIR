# Mitigating Object Hallucination in LVLMs via Attention Imbalance Rectification (CVPR 2026 Findings Track)

<img src='intro-refine.png' width=600>

Official implementation of **AIR** (Attention Imbalance Rectification), a training-free decoding intervention designed to mitigate object hallucinations in large vision-language models (LVLMs). AIR operates along two complementary dimensions: (1) **Modality-Balanced Attention Reallocation**, which redistributes attention to reduce excessive imbalance between visual and textual modalities; and (2) **Variance-Constrained Projection Regularization**, which constrains the variance of the attention matrix to promote more stable and uniform attention distributions. **AIR*** further improves both inter-head balance and inter-modal balance.

```
AIR/
├── LLaVA/                  # Main codebase (LLaVA v1.5 / v1.6)
│   ├── requirements.txt    # pip install -r (see Setup)
│   ├── pyproject.toml      # dependency versions
│   ├── bash_scripts/       # Attribution, analysis, decoding, fine-tuning
│   ├── eval_scripts/       # Evaluation (CHAIR, POPE, POPEv2, MMHal)
│   ├── llava/model/        # AIR implementation in modeling_llama.py
│   └── run_air_val.sh      # Quick baseline vs AIR sanity check
├── dataset/                # Eval datasets (gitignored; see Setup below)
├── figs/                   # Figures from analysis scripts
└── newer_models/           # Head-pruning eval on newer architectures
```

### Max New Tokens: 256


| Methods       | LLaVA-1.5 (C_S↓) | LLaVA-1.5 (C_I↓) | MiniGPT4 (C_S↓) | MiniGPT4 (C_I↓) | InstructBLIP (C_S↓) | InstructBLIP (C_I↓) | Shikra (C_S↓) | Shikra (C_I↓) |
| ------------- | ---------------- | ---------------- | --------------- | --------------- | ------------------- | ------------------- | ------------- | ------------- |
| Greedy        | 51.8             | 13.7             | 43.0            | 13.4            | 54.7                | 20.2                | 55.4          | 15.2          |
| FarSight      | 43.9             | 12.5             | 39.2            | 12.4            | 48.3                | 18.7                | 51.6          | 12.2          |
| VCD           | 59.4             | 16.0             | 40.0            | 14.6            | 50.1                | 19.4                | 54.9          | 15.1          |
| DoLA          | 54.0             | 14.2             | 42.0            | 13.5            | 52.8                | 19.1                | 55.7          | 14.8          |
| HALC          | 51.4             | 13.1             | 37.4            | 12.1            | 48.0                | 15.3                | 47.2          | 14.5          |
| OPERA         | 44.1             | 12.8             | 37.3            | 13.5            | 48.2                | 14.3                | 38.2          | 14.2          |
| AD-HH         | 35.2             | 8.8              | 32.8            | 11.5            | 36.0                | 10.3                | 36.9          | 13.7          |
| **AIR*** | **21.6**         | **7.9**          | **16.0**        | **8.2**         | **22.6**            | **7.9**             | **22.7**      | **8.5**       |


### Max New Tokens: 64


| Methods       | LLaVA-1.5 (C_S↓) | LLaVA-1.5 (C_I↓) | MiniGPT4 (C_S↓) | MiniGPT4 (C_I↓) | InstructBLIP (C_S↓) | InstructBLIP (C_I↓) | Shikra (C_S↓) | Shikra (C_I↓) |
| ------------- | ---------------- | ---------------- | --------------- | --------------- | ------------------- | ------------------- | ------------- | ------------- |
| Greedy        | 20.8             | 6.2              | 28.8            | 11.4            | 29.8                | 14.2                | 22.3          | 8.2           |
| FarSight      | 18.6             | 5.9              | 22.5            | 8.9             | 20.2                | 9.6                 | 15.4          | 7.6           |
| VCD           | 22.8             | 7.1              | 28.0            | 11.3            | 34.1                | 15.8                | 23.5          | 8.3           |
| DoLA          | 22.2             | 6.7              | 28.6            | 11.5            | 28.1                | 14.2                | 20.2          | 9.6           |
| HALC          | 21.2             | 6.6              | 24.2            | 9.7             | 24.2                | 14.1                | 14.4          | 7.8           |
| OPERA         | 17.8             | 6.5              | 26.6            | 10.4            | 18.4                | 8.7                 | 14.6          | 7.4           |
| AD-HH         | 15.6             | 5.7              | 22.0            | 8.5             | 19.6                | 8.5                 | 13.8          | 7.0           |
| **AIR*** | **12.0**         | **4.1**          | **13.9**        | **6.0**         | **13.2**            | **6.5**             | **10.5**      | **5.6**       |


## Setup

Requires **Python ≥3.8**, a **CUDA GPU**, and commands run from `LLaVA/` (eval scripts are not installed as a package).

```bash
cd LLaVA
python3 -m venv .venv && source .venv/bin/activate   # recommended
pip install -r requirements.txt
python3 scripts/download_nltk_data.py
```

Dependencies are declared in `[LLaVA/pyproject.toml](LLaVA/pyproject.toml)` (`pip install -r requirements.txt` installs editable `llava` with the `[eval]` extra: CHAIR, HF downloads, MMHal GPT judge, etc.). Install a [PyTorch build](https://pytorch.org/get-started/locally/) matching your CUDA if needed.

For `**newer_models/**`, use a separate env: `cd newer_models && pip install -r requirements.txt` (`transformers==4.45.2`).

> **Transformers note:** Uncomment `self._validate_model_kwargs(model_kwargs.copy())` in `transformers/generation/utils.py` in your installed Transformers package. Some `model_kwargs` used here are not yet supported in upstream Transformers.

### Datasets

Prepare data under `dataset/` (paths are relative to the repo root; scripts under `LLaVA/` use `../dataset`):


| Dataset         | Used for                     | Location                  | How to obtain                                                                                                                                               |
| --------------- | ---------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **COCO**        | CHAIR captions, POPE images  | `dataset/coco/`           | [COCO](https://cocodataset.org/#download): `train2014/`, `val2014/`, and `annotations/`                                                                     |
| **POPE**        | POPE accuracy                | `dataset/pope/`           | LLaVA POPE eval files: `llava_pope_test.jsonl` and `pope/coco/` (see [LLaVA Evaluation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)) |
| **POPEv2**      | POPEv2 metrics               | `dataset/POPEv2/dataset/` | `bash LLaVA/bash_scripts/download_popev2_dataset.sh`                                                                                                        |
| **MMHal-Bench** | MMHal generation / GPT judge | `dataset/mmhal/`          | `bash LLaVA/bash_scripts/download_mmhal_bench.sh` (optional: `export HF_ENDPOINT=https://hf-mirror.com`)                                                    |


`decoding.sh` skips POPE / POPEv2 / MMHal stages if the corresponding files are missing.

## Quick Validation

Run a small CHAIR comparison (50 COCO captions, baseline vs AIR):

```bash
cd LLaVA
sh run_air_val.sh
```

## Decoding with AIR

Full evaluation on COCO captions (CHAIR), POPE, POPEv2, and MMHal-Bench:

```bash
cd LLaVA
sh ./bash_scripts/decoding.sh
```

AIR is enabled by default (`use_air=true`). Run baseline only:

```bash
export use_air=false
sh ./bash_scripts/decoding.sh
```

Key hyperparameters (see `bash_scripts/decoding.sh`):


| Flag                   | Default | Description                               |
| ---------------------- | ------- | ----------------------------------------- |
| `--air`                | on      | Enable AIR pipeline                       |
| `--air-beta`           | 0.1     | Variance projection shrinkage             |
| `--air-layer-low/high` | 5 / 18  | Layers for variance regularization        |
| `--air-gamma-img`      | 1.08    | Image attention gain (modality rebalance) |
| `--air-alpha-lens`     | 0.28    | Cross-head visual lens mixing             |
| `--air-adhh-threshold` | 0.4     | Conditional trigger threshold             |


## Attribution & Analysis

Identify and analyze hallucination-sensitive attention heads (LLaVA v1.5-7B):

```bash
cd LLaVA
sh ./bash_scripts/attribute.sh
```

Analysis scripts (see `bash_scripts/analysis/`):

- `attention_bias.sh` — attention bias of hallucination vs non-hallucination heads
- `attention_inheritance.sh` — inheritance from base language model
- `js_div_in_training.sh` — attention drift during instruction tuning
- `attention_reweight_txt.sh` / `attention_reweight_img.sh` — reweighting ablations

## Acknowledgements

This work builds upon [LLaVA](https://github.com/haotian-liu/LLaVA) and [Hallucination-Attribution](https://github.com/TianyunYoung/Hallucination-Attribution). We sincerely thank the authors for their outstanding contributions and for making their valuable work publicly available.

## Citation

If you find this work useful for your research, please cite [our paper](https://arxiv.org/pdf/2603.24058):

```
@InProceedings{Sun_2026_CVPR,
    author    = {Sun, Han and Li, Qin and Wang, Peixin and Zhang, Min},
    title     = {Mitigating Object Hallucinations in LVLMs via Attention Imbalance Rectification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {8930-8940}
}
```

