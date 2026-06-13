#!/usr/bin/env python3
"""Aggregate CHAIR / POPE / MMHal / POPEv2 metrics from one decoding.sh run; print and optionally write JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _find_mmhal_summary(mmhal_dir: Path) -> Path | None:
    preferred = mmhal_dir / "responses_gpt_eval_summary.json"
    if preferred.is_file():
        return preferred
    summaries = sorted(mmhal_dir.glob("*_summary.json"))
    return summaries[0] if summaries else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-path",
        type=Path,
        required=True,
        help="Same as decoding.sh result_path, e.g. ./results/coco/llava-v1.5-7b_n500",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional output path (default <result-path>/decoding_eval_summary.json)",
    )
    args = parser.parse_args()

    rp = args.result_path.expanduser().resolve()
    out_path = args.summary_json or (rp / "decoding_eval_summary.json")

    summary: dict[str, object] = {"result_path": str(rp)}

    chair_path = rp / "captions_eval_results.json"
    chair = _load_json(chair_path)
    if chair and "overall_metrics" in chair:
        summary["chair"] = chair["overall_metrics"]
    else:
        summary["chair"] = {"status": "missing_or_invalid", "expected": str(chair_path)}

    pope_txt = rp / "pope" / "eval.txt"
    if pope_txt.is_file():
        summary["pope_eval_txt"] = pope_txt.read_text(encoding="utf-8").strip()
    else:
        summary["pope_eval_txt"] = None

    mmhal_summary_path = _find_mmhal_summary(rp / "mmhal")
    mmhal = _load_json(mmhal_summary_path) if mmhal_summary_path else None
    if mmhal:
        summary["mmhal"] = {
            k: mmhal[k]
            for k in (
                "average_score",
                "hallucination_rate",
                "num_items",
                "average_score_by_type_index_0_to_7",
            )
            if k in mmhal
        }
        summary["mmhal_summary_path"] = str(mmhal_summary_path)
    else:
        summary["mmhal"] = {"status": "missing", "hint": "Run MMHal GPT judge to produce *_summary.json"}

    popev2_metrics = rp / "popev2" / "metrics.json"
    pv2 = _load_json(popev2_metrics)
    summary["popev2"] = pv2 if pv2 else {"status": "missing", "expected": str(popev2_metrics)}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    sep = "=" * 72
    print(sep)
    print("Decoding eval summary (see JSON for full detail)")
    print(f"Wrote: {out_path}")
    print(sep)

    om = summary.get("chair")
    print("\n【CHAIR】overall_metrics:")
    print(json.dumps(om, indent=2, ensure_ascii=False) if isinstance(om, dict) else om)

    print("\n[POPE] eval.txt excerpt (full text in JSON pope_eval_txt):")
    pte = summary.get("pope_eval_txt")
    if pte:
        print(pte[:2000] + ("\n...(truncated)" if len(str(pte)) > 2000 else ""))
    else:
        print("(none)")

    print("\n【MMHal】:")
    print(json.dumps(summary.get("mmhal"), indent=2, ensure_ascii=False))

    print("\n【POPEv2】metrics.json:")
    print(json.dumps(summary.get("popev2"), indent=2, ensure_ascii=False))
    print(sep)


if __name__ == "__main__":
    main()
