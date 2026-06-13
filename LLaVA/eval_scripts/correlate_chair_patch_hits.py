"""
Align captions_eval_results.json (CHAIR sentence metrics) with project_analysis batch_summary.json
(patch-level hallucinated-token hits) by image and measure correlation.
summary includes num_images_patch_halluc_aligned / frac_matched_images_patch_halluc_aligned:
count and fraction of matched images with at least one patch flagged as hallucination-aligned.

Usage (from LLaVA/):
  python3 eval_scripts/correlate_chair_patch_hits.py \\
    --chair-results results/coco/llava-v1.5-7b/captions_eval_results.json \\
    --batch-summary results/project_analysis/batch_llava_v15_7b/batch_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, List, Tuple


def pearsonr(x: List[float], y: List[float]) -> Tuple[float, int]:
    n = len(x)
    if n < 2 or n != len(y):
        return float("nan"), n
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denx = math.sqrt(sum((xi - mx) ** 2 for xi in x))
    deny = math.sqrt(sum((yi - my) ** 2 for yi in y))
    if denx < 1e-12 or deny < 1e-12:
        return float("nan"), n
    return num / (denx * deny), n


def rankdata(values: List[float]) -> List[float]:
    """Average ranks, 1-based; ties get mean rank."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        v = values[indexed[i]]
        while j < n and values[indexed[j]] == v:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k]] = avg_rank
        i = j
    return ranks


def spearmanr(x: List[float], y: List[float]) -> Tuple[float, int]:
    n = len(x)
    if n < 2 or n != len(y):
        return float("nan"), n
    return pearsonr(rankdata(x), rankdata(y))[0], n


def load_chair_by_image(path: str) -> Dict[str, dict]:
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, dict] = {}
    for row in data.get("sentences", []):
        name = row.get("image")
        if not name:
            continue
        m = row.get("metrics") or {}
        out[name] = {
            "CHAIRs": float(m.get("CHAIRs", 0)),
            "CHAIRi": float(m.get("CHAIRi", 0.0)),
            "n_halluc_groups": len(row.get("mscoco_hallucinated_words") or []),
        }
    return out


def patch_stats_from_summary(chair_summary: dict) -> dict:
    counts = chair_summary.get("matched_patch_counts") or {}
    total = sum(counts.values())
    n_halluc_patches = int(counts.get("hallucinated", 0))
    frac = (n_halluc_patches / total) if total else 0.0
    mw = chair_summary.get("matched_words") or {}
    halluc_mw = mw.get("hallucinated") or []
    has_matched_halluc_word = len(halluc_mw) > 0
    n_halluc_words = len(chair_summary.get("hallucinated_words") or [])
    return {
        "n_halluc_words": n_halluc_words,
        "n_halluc_patches": n_halluc_patches,
        "total_patches": total,
        "frac_halluc_patches": frac,
        "has_matched_halluc_word": 1.0 if has_matched_halluc_word else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Correlate CHAIR metrics with patch-level hallucinated-token hits."
    )
    parser.add_argument(
        "--chair-results",
        type=str,
        required=True,
        help="captions_eval_results.json from eval_chair.py",
    )
    parser.add_argument(
        "--batch-summary",
        type=str,
        required=True,
        help="batch_summary.json from project_analysis.py batch mode",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write per-image merged table + summary stats as JSON.",
    )
    args = parser.parse_args()

    chair_by_image = load_chair_by_image(args.chair_results)
    with open(os.path.expanduser(args.batch_summary), "r", encoding="utf-8") as f:
        batch = json.load(f)

    merged = []
    xs_chairi = []
    ys_frac = []
    ys_has_word = []
    xs_nwords = []
    chair_s_vals = []
    patch_hit_binary = []

    for item in batch.get("images", []):
        name = item.get("image")
        if not name or name not in chair_by_image:
            continue
        ch = chair_by_image[name]
        cs = item.get("chair_summary")
        if not cs:
            continue
        ps = patch_stats_from_summary(cs)
        row = {**ch, **ps, "image": name}
        merged.append(row)

        xs_chairi.append(ch["CHAIRi"])
        ys_frac.append(ps["frac_halluc_patches"])
        ys_has_word.append(ps["has_matched_halluc_word"])
        xs_nwords.append(float(ps["n_halluc_words"]))
        chair_s_vals.append(ch["CHAIRs"])
        patch_hit_binary.append(1.0 if ps["n_halluc_patches"] > 0 else 0.0)

    n = len(merged)
    if n == 0:
        raise SystemExit("No overlapping images between chair results and batch summary.")

    r_pear_chairi_frac, _ = pearsonr(xs_chairi, ys_frac)
    r_spear_chairi_frac, _ = spearmanr(xs_chairi, ys_frac)
    r_pear_chairi_hasw, _ = pearsonr(xs_chairi, ys_has_word)
    r_pear_nwords_frac, _ = pearsonr(xs_nwords, ys_frac)
    r_pear_nwords_hasw, _ = pearsonr(xs_nwords, ys_has_word)

    # CHAIRs (sentence hallucination) vs patch hit rate
    s1 = sum(chair_s_vals)
    n1 = int(s1)
    n0 = n - n1
    hit_when_s1 = sum(
        patch_hit_binary[i] for i in range(n) if chair_s_vals[i] >= 0.5
    ) / max(n1, 1)
    hit_when_s0 = sum(
        patch_hit_binary[i] for i in range(n) if chair_s_vals[i] < 0.5
    ) / max(n0, 1)
    hasword_when_s1 = sum(
        ys_has_word[i] for i in range(n) if chair_s_vals[i] >= 0.5
    ) / max(n1, 1)
    hasword_when_s0 = sum(
        ys_has_word[i] for i in range(n) if chair_s_vals[i] < 0.5
    ) / max(n0, 1)

    # Quartile stratification for CHAIRi (min, Q1, Q2, Q3, max)
    sorted_c = sorted(xs_chairi)

    def qtr(p: float) -> float:
        if n == 1:
            return sorted_c[0]
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return sorted_c[lo]
        w = idx - lo
        return sorted_c[lo] * (1 - w) + sorted_c[hi] * w

    q_edges = [qtr(0.0), qtr(0.25), qtr(0.5), qtr(0.75), qtr(1.0)]

    def chiari_bucket(ci: float) -> int:
        if ci <= q_edges[1] + 1e-15:
            return 0
        if ci <= q_edges[2] + 1e-15:
            return 1
        if ci <= q_edges[3] + 1e-15:
            return 2
        return 3

    bucket_mean_frac = [0.0] * 4
    bucket_cnt = [0] * 4
    for i in range(n):
        b = chiari_bucket(xs_chairi[i])
        bucket_mean_frac[b] += ys_frac[i]
        bucket_cnt[b] += 1
    for b in range(4):
        if bucket_cnt[b]:
            bucket_mean_frac[b] /= bucket_cnt[b]

    # Among all matched images: count with patch-side n_halluc_patches>0 (unconditional on CHAIRs)
    patch_hit_count = int(sum(patch_hit_binary))
    frac_patch_hit_all = (patch_hit_count / n) if n else 0.0

    summary = {
        "num_matched_images": n,
        "num_images_patch_halluc_aligned": patch_hit_count,
        "frac_matched_images_patch_halluc_aligned": frac_patch_hit_all,
        "pearson_CHAIRi_vs_frac_halluc_patches": r_pear_chairi_frac,
        "spearman_CHAIRi_vs_frac_halluc_patches": r_spear_chairi_frac,
        "pearson_CHAIRi_vs_has_matched_halluc_word": r_pear_chairi_hasw,
        "pearson_n_halluc_words_vs_frac_halluc_patches": r_pear_nwords_frac,
        "pearson_n_halluc_words_vs_has_matched_halluc_word": r_pear_nwords_hasw,
        "CHAIRs1_count": n1,
        "CHAIRs0_count": n0,
        "mean_patch_halluc_label_rate_when_CHAIRs_1": hit_when_s1,
        "mean_patch_halluc_label_rate_when_CHAIRs_0": hit_when_s0,
        "mean_has_matched_halluc_word_when_CHAIRs_1": hasword_when_s1,
        "mean_has_matched_halluc_word_when_CHAIRs_0": hasword_when_s0,
        "CHAIRi_quartile_bin_counts": bucket_cnt,
        "mean_frac_halluc_patches_by_CHAIRi_quartile": bucket_mean_frac,
        "CHAIRi_quartile_edges": q_edges,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output_json:
        out_path = os.path.expanduser(args.output_json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_image": merged}, f, indent=2, ensure_ascii=False)
        print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
