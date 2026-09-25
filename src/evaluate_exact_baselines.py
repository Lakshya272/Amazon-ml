#!/usr/bin/env python3
"""
src/evaluate_exact_baselines.py — Amazon ML Challenge 2026 Business Entity Resolution
Exact Matching Baselines & Legal-Suffix Canonicalization Benchmark

Benchmarks and evaluates exact matching strategies across the training dataset:
  1. exact_raw_name: Raw business name + country
  2. exact_base_name: Base normalized (NFKC + lowercase) name + country
  3. exact_clean_name: Clean alphanumeric name + country
  4. exact_legal_name: Legal-suffix canonicalized name (Pvt->Private, Ltd->Limited, etc.) + country
  5. exact_raw_address: Raw business address + country
  6. exact_clean_address: Clean alphanumeric address + country
  7. exact_clean_pair: Clean name AND Clean address + country
  8. exact_legal_pair: Legal name AND Canonical address + country
  9. exact_clean_name_or_address: Union of clean name OR clean address

Optimized single-pass streaming architecture:
  - Phase 1: Reads S2 and S3 once to construct all candidate inverted indices.
  - Phase 2: Reads S1 once, evaluating all 9 exact modes in parallel on the fly.
Runs in ~3-4 minutes on EC2 with ~4 GiB RAM peak.
"""

import argparse
import csv
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional

from metrics import compute_entity_metrics
from normalize import (
    normalize_raw,
    normalize_base,
    normalize_clean,
    normalize_legal_name,
    normalize_address,
)

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass


def load_ground_truth(gt_path: str, target_s1_ids: Optional[Set[str]] = None) -> Dict[str, Set[str]]:
    """Load ground truth mapping source1_entity_id -> set of matched entity IDs."""
    print(f"[{time.strftime('%H:%M:%S')}] Loading ground truth from {os.path.basename(gt_path)}...")
    t0 = time.time()
    truth: Dict[str, Set[str]] = {}
    with open(gt_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)
        for row in reader:
            if not row or len(row) < 1:
                continue
            s1_id = row[0].strip()
            if target_s1_ids is not None and s1_id not in target_s1_ids:
                continue
            matched_str = row[1].strip() if len(row) > 1 else ""
            if matched_str:
                truth[s1_id] = {x.strip() for x in matched_str.split(',') if x.strip()}
            else:
                truth[s1_id] = set()
    elapsed = round(time.time() - t0, 2)
    print(f"[{time.strftime('%H:%M:%S')}] Loaded {len(truth):,} ground truth entries in {elapsed}s.")
    return truth


def run_benchmark(
    data_dir: str,
    output_dir: str,
    modes: List[str],
    max_s1: Optional[int] = None,
    max_candidates: Optional[int] = None,
    candidate_cap: int = 50
) -> List[Dict[str, Any]]:
    """
    Run exact matching benchmark with single-pass candidate indexing and single-pass S1 evaluation.
    """
    s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train", "train_source3.tsv")
    gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")

    # If max_s1 is requested, collect target S1 IDs first from s1_path
    target_s1_ids: Optional[Set[str]] = None
    if max_s1:
        target_s1_ids = set()
        with open(s1_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None)
            for row in reader:
                if len(row) >= 1:
                    target_s1_ids.add(row[0].strip())
                    if len(target_s1_ids) >= max_s1:
                        break

    truth = load_ground_truth(gt_path, target_s1_ids=target_s1_ids)

    # Inverted index dictionary: mode -> dict(key -> list of IDs)
    indices: Dict[str, Dict[Any, List[str]]] = {m: defaultdict(list) for m in modes}
    # Ensure clean_name and clean_address are indexed if union mode is present
    needs_clean_name = "exact_clean_name" in modes or "exact_clean_name_or_address" in modes
    needs_clean_addr = "exact_clean_address" in modes or "exact_clean_name_or_address" in modes
    if needs_clean_name and "exact_clean_name" not in indices:
        indices["exact_clean_name"] = defaultdict(list)
    if needs_clean_addr and "exact_clean_address" not in indices:
        indices["exact_clean_address"] = defaultdict(list)

    print(f"[{time.strftime('%H:%M:%S')}] Indexing candidates from Source 2 and Source 3...")
    t_idx_start = time.time()
    total_candidates = 0

    for source_path in [s2_path, s3_path]:
        s_name = os.path.basename(source_path)
        with open(source_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader, None)
            s_count = 0
            for row in reader:
                if len(row) < 4:
                    continue
                eid, b_name, b_addr, country = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()

                raw_n = normalize_raw(b_name) if ("exact_raw_name" in indices) else None
                base_n = normalize_base(b_name) if ("exact_base_name" in indices) else None
                clean_n = normalize_clean(b_name) if (needs_clean_name or "exact_clean_pair" in indices) else None
                legal_n = normalize_legal_name(b_name) if ("exact_legal_name" in indices or "exact_legal_pair" in indices) else None

                raw_a = normalize_raw(b_addr) if ("exact_raw_address" in indices) else None
                clean_a = normalize_clean(b_addr) if (needs_clean_addr or "exact_clean_pair" in indices) else None
                legal_a = normalize_address(b_addr, canonicalize_abbreviations=True) if ("exact_legal_pair" in indices) else None

                if "exact_raw_name" in indices and raw_n:
                    indices["exact_raw_name"][(country, raw_n)].append(eid)
                if "exact_base_name" in indices and base_n:
                    indices["exact_base_name"][(country, base_n)].append(eid)
                if "exact_clean_name" in indices and clean_n:
                    indices["exact_clean_name"][(country, clean_n)].append(eid)
                if "exact_legal_name" in indices and legal_n:
                    indices["exact_legal_name"][(country, legal_n)].append(eid)

                if "exact_raw_address" in indices and raw_a:
                    indices["exact_raw_address"][(country, raw_a)].append(eid)
                if "exact_clean_address" in indices and clean_a:
                    indices["exact_clean_address"][(country, clean_a)].append(eid)

                if "exact_clean_pair" in indices and clean_n and clean_a:
                    indices["exact_clean_pair"][(country, clean_n, clean_a)].append(eid)
                if "exact_legal_pair" in indices and legal_n and legal_a:
                    indices["exact_legal_pair"][(country, legal_n, legal_a)].append(eid)

                s_count += 1
                total_candidates += 1
                if max_candidates and total_candidates >= max_candidates:
                    break
        print(f"  Indexed {s_count:,} records from {s_name}...")
        if max_candidates and total_candidates >= max_candidates:
            break

    print(f"[{time.strftime('%H:%M:%S')}] Finished indexing {total_candidates:,} candidate records in {round(time.time() - t_idx_start, 2)}s.")

    # Tracking accumulators per mode
    stats: Dict[str, Dict[str, Any]] = {}
    for m in modes:
        stats[m] = {
            "total_s1": 0,
            "total_f05": 0.0,
            "total_tp": 0,
            "total_fp": 0,
            "total_fn": 0,
            "total_true_links": 0,
            "total_pred_links": 0,
            "true_singletons": 0,
            "pred_singletons": 0,
            "correct_singletons": 0,
            "false_merge_singletons": 0,
            "max_pred_count": 0,
            "s1_with_predictions": 0,
            "sum_prec_matched": 0.0,
            "count_matched_entities": 0,
            "sum_rec_with_truth": 0.0,
            "count_entities_with_truth": 0,
            "f05_bins": {
                "f05_exact_1.0": 0,
                "f05_0.8_to_1.0": 0,
                "f05_0.5_to_0.8": 0,
                "f05_0.0_to_0.5": 0,
                "f05_exact_0.0": 0,
            }
        }

    print(f"[{time.strftime('%H:%M:%S')}] Evaluating S1 across all {len(modes)} modes in single pass...")
    t_eval_start = time.time()
    s1_rows_evaluated = 0

    with open(s1_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader, None)

        for row in reader:
            if len(row) < 4:
                continue
            s1_id, b_name, b_addr, country = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()

            if s1_id not in truth:
                continue

            true_set = truth[s1_id]

            # Compute S1 representations
            raw_n = normalize_raw(b_name)
            base_n = normalize_base(b_name)
            clean_n = normalize_clean(b_name)
            legal_n = normalize_legal_name(b_name)

            raw_a = normalize_raw(b_addr)
            clean_a = normalize_clean(b_addr)
            legal_a = normalize_address(b_addr, canonicalize_abbreviations=True)

            # Evaluate each mode
            for mode in modes:
                candidates: Set[str] = set()

                if mode == "exact_raw_name":
                    if raw_n:
                        c_list = indices[mode].get((country, raw_n))
                        if c_list:
                            candidates = set(c_list[:candidate_cap] if candidate_cap and len(c_list) > candidate_cap else c_list)
                elif mode == "exact_base_name":
                    if base_n:
                        c_list = indices[mode].get((country, base_n))
                        if c_list:
                            candidates = set(c_list[:candidate_cap] if candidate_cap and len(c_list) > candidate_cap else c_list)
                elif mode == "exact_clean_name":
                    if clean_n:
                        c_list = indices[mode].get((country, clean_n))
                        if c_list:
                            candidates = set(c_list[:candidate_cap] if candidate_cap and len(c_list) > candidate_cap else c_list)
                elif mode == "exact_legal_name":
                    if legal_n:
                        c_list = indices[mode].get((country, legal_n))
                        if c_list:
                            candidates = set(c_list[:candidate_cap] if candidate_cap and len(c_list) > candidate_cap else c_list)
                elif mode == "exact_raw_address":
                    if raw_a:
                        c_list = indices[mode].get((country, raw_a))
                        if c_list:
                            candidates = set(c_list[:candidate_cap] if candidate_cap and len(c_list) > candidate_cap else c_list)
                elif mode == "exact_clean_address":
                    if clean_a:
                        c_list = indices[mode].get((country, clean_a))
                        if c_list:
                            candidates = set(c_list[:candidate_cap] if candidate_cap and len(c_list) > candidate_cap else c_list)
                elif mode == "exact_clean_pair":
                    if clean_n and clean_a:
                        c_list = indices[mode].get((country, clean_n, clean_a))
                        if c_list:
                            candidates = set(c_list[:candidate_cap] if candidate_cap and len(c_list) > candidate_cap else c_list)
                elif mode == "exact_legal_pair":
                    if legal_n and legal_a:
                        c_list = indices[mode].get((country, legal_n, legal_a))
                        if c_list:
                            candidates = set(c_list[:candidate_cap] if candidate_cap and len(c_list) > candidate_cap else c_list)
                elif mode == "exact_clean_name_or_address":
                    if clean_n:
                        c_n = indices["exact_clean_name"].get((country, clean_n))
                        if c_n:
                            candidates.update(c_n[:candidate_cap] if candidate_cap and len(c_n) > candidate_cap else c_n)
                    if clean_a:
                        c_a = indices["exact_clean_address"].get((country, clean_a))
                        if c_a:
                            candidates.update(c_a[:candidate_cap] if candidate_cap and len(c_a) > candidate_cap else c_a)

                n_pred = len(candidates)
                st = stats[mode]
                if n_pred > st["max_pred_count"]:
                    st["max_pred_count"] = n_pred
                if n_pred > 0:
                    st["s1_with_predictions"] += 1

                f05, prec, rec, tp, fp, fn = compute_entity_metrics(true_set, candidates)

                st["total_s1"] += 1
                st["total_f05"] += f05
                st["total_tp"] += tp
                st["total_fp"] += fp
                st["total_fn"] += fn
                st["total_true_links"] += len(true_set)
                st["total_pred_links"] += n_pred

                if len(true_set) == 0:
                    st["true_singletons"] += 1
                    if n_pred == 0:
                        st["correct_singletons"] += 1
                    else:
                        st["false_merge_singletons"] += 1
                else:
                    st["count_entities_with_truth"] += 1
                    st["sum_rec_with_truth"] += rec

                if n_pred == 0:
                    st["pred_singletons"] += 1
                else:
                    st["count_matched_entities"] += 1
                    st["sum_prec_matched"] += prec

                if f05 >= 0.999999:
                    st["f05_bins"]["f05_exact_1.0"] += 1
                elif f05 >= 0.8:
                    st["f05_bins"]["f05_0.8_to_1.0"] += 1
                elif f05 >= 0.5:
                    st["f05_bins"]["f05_0.5_to_0.8"] += 1
                elif f05 > 0.000001:
                    st["f05_bins"]["f05_0.0_to_0.5"] += 1
                else:
                    st["f05_bins"]["f05_exact_0.0"] += 1

            s1_rows_evaluated += 1
            if max_s1 and s1_rows_evaluated >= max_s1:
                break
            if s1_rows_evaluated % 500000 == 0:
                print(f"  Processed {s1_rows_evaluated:,} S1 entities...")

    elapsed_eval = round(time.time() - t_eval_start, 2)
    print(f"[{time.strftime('%H:%M:%S')}] Completed evaluation of {s1_rows_evaluated:,} S1 entities in {elapsed_eval}s.")

    # Clean up index memory
    del indices
    gc.collect()

    all_results = []
    for mode in modes:
        st = stats[mode]
        n_s1 = st["total_s1"]
        macro_f05 = st["total_f05"] / n_s1 if n_s1 else 0.0
        pooled_prec = st["total_tp"] / st["total_pred_links"] if st["total_pred_links"] else (1.0 if st["total_true_links"] == 0 else 0.0)
        pooled_rec = st["total_tp"] / st["total_true_links"] if st["total_true_links"] else (1.0 if st["total_pred_links"] == 0 else 0.0)
        pooled_denom = 0.25 * pooled_prec + pooled_rec
        pooled_f05 = (1.25 * pooled_prec * pooled_rec) / pooled_denom if pooled_denom > 0 else 0.0
        macro_prec = st["sum_prec_matched"] / st["count_matched_entities"] if st["count_matched_entities"] else 0.0
        macro_rec = st["sum_rec_with_truth"] / st["count_entities_with_truth"] if st["count_entities_with_truth"] else 0.0
        singleton_acc = st["correct_singletons"] / st["true_singletons"] if st["true_singletons"] else 1.0

        res = {
            "mode": mode,
            "elapsed_seconds": elapsed_eval,
            "total_s1_evaluated": n_s1,
            "macro_f05": round(macro_f05, 6),
            "pooled_f05": round(pooled_f05, 6),
            "pooled_precision": round(pooled_prec, 6),
            "pooled_recall": round(pooled_rec, 6),
            "macro_precision_on_predicted": round(macro_prec, 6),
            "macro_recall_on_true_matches": round(macro_rec, 6),
            "total_true_links": st["total_true_links"],
            "total_pred_links": st["total_pred_links"],
            "total_recovered_links": st["total_tp"],
            "total_false_positives": st["total_fp"],
            "total_false_negatives": st["total_fn"],
            "s1_with_predictions": st["s1_with_predictions"],
            "s1_with_predictions_pct": round(100.0 * st["s1_with_predictions"] / n_s1 if n_s1 else 0, 2),
            "avg_pred_links_per_s1": round(st["total_pred_links"] / n_s1 if n_s1 else 0, 4),
            "max_pred_links_single_s1": st["max_pred_count"],
            "true_singletons": st["true_singletons"],
            "predicted_singletons": st["pred_singletons"],
            "correct_singletons": st["correct_singletons"],
            "false_merge_singletons": st["false_merge_singletons"],
            "singleton_accuracy": round(singleton_acc, 6),
            "f05_distribution": {
                k: {
                    "count": v,
                    "percentage": round(100.0 * v / n_s1 if n_s1 else 0, 2)
                } for k, v in st["f05_bins"].items()
            }
        }
        all_results.append(res)
        print(f"[{mode}] Macro F0.5={macro_f05:.4f} | Link Recall={pooled_rec*100:.2f}% | Link Prec={pooled_prec*100:.2f}% | S1 with Preds={res['s1_with_predictions_pct']:.1f}%")

    return all_results


def generate_benchmark_markdown(results: List[Dict[str, Any]], output_path: str):
    """Generate Markdown comparison report of exact matching baselines."""
    lines = []
    lines.append("# Exact Matching Baselines & Legal Normalization Benchmark")
    lines.append("")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ")
    lines.append("")
    lines.append("## 1. Summary Benchmark Comparison")
    lines.append("")
    lines.append("| Baseline Strategy | Macro F0.5 | Link Recall (%) | Link Precision (%) | Predicted Links | Recovered Links | S1 with Preds (%) | Singleton Acc (%) |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for r in results:
        lines.append(
            f"| `{r['mode']}` | **{r['macro_f05']:.4f}** | "
            f"{r['pooled_recall']*100:.2f}% | {r['pooled_precision']*100:.2f}% | "
            f"{r['total_pred_links']:,} | {r['total_recovered_links']:,} | "
            f"{r['s1_with_predictions_pct']:.1f}% | {r['singleton_accuracy']*100:.2f}% |"
        )
    lines.append("")

    lines.append("## 2. In-Depth Strategy Observations & Error Dynamics")
    lines.append("")
    lines.append("### Legal-Suffix Normalization Effect")
    clean_r = next((x for x in results if x["mode"] == "exact_clean_name"), None)
    legal_r = next((x for x in results if x["mode"] == "exact_legal_name"), None)
    if clean_r and legal_r:
        rec_diff = legal_r["total_recovered_links"] - clean_r["total_recovered_links"]
        pred_diff = legal_r["total_pred_links"] - clean_r["total_pred_links"]
        f05_diff = legal_r["macro_f05"] - clean_r["macro_f05"]
        lines.append(f"- **Additional True Links Recovered:** +{rec_diff:,} true links")
        lines.append(f"- **Additional Candidate Predictions:** +{pred_diff:,} predictions")
        lines.append(f"- **Delta in Macro F0.5:** {f05_diff:+.6f}")
        lines.append(f"- **Precision Impact:** {clean_r['pooled_precision']*100:.2f}% -> {legal_r['pooled_precision']*100:.2f}%")
        lines.append(f"- **Link Recall Impact:** {clean_r['pooled_recall']*100:.2f}% -> {legal_r['pooled_recall']*100:.2f}%")
    lines.append("")

    lines.append("### Score Distributions (Per-Entity F0.5)")
    lines.append("")
    lines.append("| Baseline Strategy | F0.5 = 1.0 (%) | F0.5 in [0.8, 1.0) (%) | F0.5 in [0.5, 0.8) (%) | F0.5 in (0, 0.5) (%) | F0.5 = 0.0 (%) |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        dist = r["f05_distribution"]
        lines.append(
            f"| `{r['mode']}` | {dist['f05_exact_1.0']['percentage']:.2f}% | "
            f"{dist['f05_0.8_to_1.0']['percentage']:.2f}% | "
            f"{dist['f05_0.5_to_0.8']['percentage']:.2f}% | "
            f"{dist['f05_0.0_to_0.5']['percentage']:.2f}% | "
            f"{dist['f05_exact_0.0']['percentage']:.2f}% |"
        )
    lines.append("")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"[{time.strftime('%H:%M:%S')}] Saved benchmark report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate exact matching baselines")
    parser.add_argument("--data-dir", type=str, default="", help="Path to student_resource/dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/baselines", help="Output directory")
    parser.add_argument("--max-s1", type=int, default=0, help="Max S1 records to evaluate (0 for full)")
    parser.add_argument("--max-candidates", type=int, default=0, help="Max candidate records to index (0 for full)")
    parser.add_argument("--candidate-cap", type=int, default=50, help="Max candidates per bucket (default 50)")
    parser.add_argument("--modes", type=str, default="all", help="Comma-separated modes or 'all'")
    args = parser.parse_args()

    overall_start = time.time()

    data_dir = args.data_dir
    if not data_dir:
        candidates = [
            "student_resource/dataset",
            "dataset",
            os.path.expanduser("~/amazon-ml-challenge/student_resource/dataset"),
            os.path.expanduser("~/amazon-ml-challenge/dataset")
        ]
        for c in candidates:
            if os.path.isdir(c) and os.path.isdir(os.path.join(c, "train")):
                data_dir = c
                break

    if not data_dir:
        print("ERROR: Dataset directory not found.")
        sys.exit(1)

    all_modes = [
        "exact_raw_name",
        "exact_base_name",
        "exact_clean_name",
        "exact_legal_name",
        "exact_raw_address",
        "exact_clean_address",
        "exact_clean_pair",
        "exact_legal_pair",
        "exact_clean_name_or_address",
    ]

    if args.modes == "all":
        modes_to_test = all_modes
    else:
        modes_to_test = [m.strip() for m in args.modes.split(",") if m.strip()]

    os.makedirs(args.output_dir, exist_ok=True)
    max_s1 = args.max_s1 if args.max_s1 > 0 else None
    max_candidates = args.max_candidates if args.max_candidates > 0 else None

    print(f"=== Amazon ML Challenge 2026: Exact Matching Baselines ===")
    print(f"Data Dir: {data_dir}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Modes to test ({len(modes_to_test)}): {modes_to_test}")
    print(f"Max S1: {max_s1 or 'FULL'} | Max Candidates: {max_candidates or 'FULL'}")

    results = run_benchmark(
        data_dir=data_dir,
        output_dir=args.output_dir,
        modes=modes_to_test,
        max_s1=max_s1,
        max_candidates=max_candidates,
        candidate_cap=args.candidate_cap
    )

    json_path = os.path.join(args.output_dir, "exact_baselines.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] Saved JSON benchmark to {json_path}")

    md_path = os.path.join(args.output_dir, "exact_baselines.md")
    generate_benchmark_markdown(results, md_path)

    total_time = round(time.time() - overall_start, 2)
    print(f"=== Completed Exact Matching Baselines in {total_time}s ===")


if __name__ == "__main__":
    main()
