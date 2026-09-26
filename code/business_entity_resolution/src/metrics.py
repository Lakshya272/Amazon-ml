#!/usr/bin/env python3
"""
src/metrics.py — Amazon ML Challenge 2026 Business Entity Resolution
Official Evaluation Metric Implementation: Macro F0.5

Calculates per-Source-1-entity precision, recall, and F0.5:
    F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
Averages F0.5 across all Source 1 entities in the evaluation set (Macro F0.5).

Singleton Handling:
- True empty set (singleton) + predicted empty set = 1.0
- True empty set (singleton) + any predicted match = 0.0 (false merge penalty)
- True non-empty set + predicted empty set = 0.0 (missed match penalty)

Contains unit tests and command-line execution capability.
"""

from typing import Dict, Set, Tuple, Any, Iterable, Optional
import math
import sys

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass


def compute_entity_metrics(
    true_matches: Set[str],
    pred_matches: Set[str]
) -> Tuple[float, float, float, int, int, int]:
    """
    Compute (f05, precision, recall, tp, fp, fn) for a single Source 1 entity.

    Rules:
    - If true_matches is empty:
      - If pred_matches is empty: F0.5 = 1.0, precision = 1.0, recall = 1.0, TP = 0, FP = 0, FN = 0
      - If pred_matches is non-empty: F0.5 = 0.0, precision = 0.0, recall = 0.0, TP = 0, FP = len(pred_matches), FN = 0
    - If true_matches is non-empty:
      - If pred_matches is empty: F0.5 = 0.0, precision = 0.0, recall = 0.0, TP = 0, FP = 0, FN = len(true_matches)
      - If pred_matches is non-empty:
        TP = len(true_matches & pred_matches)
        FP = len(pred_matches - true_matches)
        FN = len(true_matches - pred_matches)
        if TP == 0:
            F0.5 = 0.0, precision = 0.0, recall = 0.0
        else:
            precision = TP / len(pred_matches)
            recall = TP / len(true_matches)
            # F_beta with beta = 0.5:
            # (1 + beta^2) * P * R / (beta^2 * P + R) = 1.25 * P * R / (0.25 * P + R)
            denom = 0.25 * precision + recall
            f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0
    """
    n_true = len(true_matches)
    n_pred = len(pred_matches)

    # Singleton case
    if n_true == 0:
        if n_pred == 0:
            return 1.0, 1.0, 1.0, 0, 0, 0
        else:
            return 0.0, 0.0, 0.0, 0, n_pred, 0

    # Non-empty true matches but predicted empty
    if n_pred == 0:
        return 0.0, 0.0, 0.0, 0, 0, n_true

    # Both non-empty
    tp = len(true_matches & pred_matches)
    fp = n_pred - tp
    fn = n_true - tp

    if tp == 0:
        return 0.0, 0.0, 0.0, 0, fp, fn

    precision = tp / n_pred
    recall = tp / n_true
    denom = 0.25 * precision + recall
    f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0

    return f05, precision, recall, tp, fp, fn


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    eval_entity_ids: Optional[Iterable[str]] = None
) -> Dict[str, Any]:
    """
    Evaluate predicted matches against ground truth for all evaluated S1 entities.

    Args:
        ground_truth: mapping of source1_entity_id -> set of true matching S2/S3 entity_ids
        predictions: mapping of source1_entity_id -> set of predicted matching S2/S3 entity_ids
        eval_entity_ids: optional subset of S1 entity IDs to evaluate. If None, evaluates all IDs in ground_truth.

    Returns:
        Dictionary of comprehensive evaluation metrics.
    """
    target_ids = list(eval_entity_ids) if eval_entity_ids is not None else list(ground_truth.keys())
    total_entities = len(target_ids)

    if total_entities == 0:
        return {"error": "No entities to evaluate"}

    total_f05 = 0.0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_true_links = 0
    total_pred_links = 0

    true_singletons = 0
    pred_singletons = 0
    correct_singletons = 0
    false_merge_singletons = 0

    # Histograms of per-entity F0.5
    f05_bins = {
        "f05_exact_1.0": 0,
        "f05_0.8_to_1.0": 0,
        "f05_0.5_to_0.8": 0,
        "f05_0.0_to_0.5": 0,
        "f05_exact_0.0": 0,
    }

    # Sum of precisions and recalls for macro-averaging on non-singleton entities
    sum_prec_matched = 0.0
    count_matched_entities = 0
    sum_rec_with_truth = 0.0
    count_entities_with_truth = 0

    for s1_id in target_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())

        f05, prec, rec, tp, fp, fn = compute_entity_metrics(true_set, pred_set)

        total_f05 += f05
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_true_links += len(true_set)
        total_pred_links += len(pred_set)

        # Singletons
        if len(true_set) == 0:
            true_singletons += 1
            if len(pred_set) == 0:
                correct_singletons += 1
            else:
                false_merge_singletons += 1
        else:
            count_entities_with_truth += 1
            sum_rec_with_truth += rec

        if len(pred_set) == 0:
            pred_singletons += 1
        else:
            count_matched_entities += 1
            sum_prec_matched += prec

        # Distribution binning
        if f05 >= 0.999999:
            f05_bins["f05_exact_1.0"] += 1
        elif f05 >= 0.8:
            f05_bins["f05_0.8_to_1.0"] += 1
        elif f05 >= 0.5:
            f05_bins["f05_0.5_to_0.8"] += 1
        elif f05 > 0.000001:
            f05_bins["f05_0.0_to_0.5"] += 1
        else:
            f05_bins["f05_exact_0.0"] += 1

    macro_f05 = total_f05 / total_entities
    pooled_prec = total_tp / total_pred_links if total_pred_links > 0 else (1.0 if total_true_links == 0 else 0.0)
    pooled_rec = total_tp / total_true_links if total_true_links > 0 else (1.0 if total_pred_links == 0 else 0.0)
    pooled_denom = 0.25 * pooled_prec + pooled_rec
    pooled_f05 = (1.25 * pooled_prec * pooled_rec) / pooled_denom if pooled_denom > 0 else 0.0

    macro_prec = sum_prec_matched / count_matched_entities if count_matched_entities > 0 else 0.0
    macro_rec = sum_rec_with_truth / count_entities_with_truth if count_entities_with_truth > 0 else 0.0
    singleton_acc = correct_singletons / true_singletons if true_singletons > 0 else 1.0

    return {
        "macro_f05": round(macro_f05, 6),
        "pooled_f05": round(pooled_f05, 6),
        "pooled_precision": round(pooled_prec, 6),
        "pooled_recall": round(pooled_rec, 6),
        "macro_precision_on_predicted": round(macro_prec, 6),
        "macro_recall_on_true_matches": round(macro_rec, 6),
        "total_s1_entities": total_entities,
        "total_true_links": total_true_links,
        "total_pred_links": total_pred_links,
        "total_recovered_links": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
        "avg_pred_links_per_s1": round(total_pred_links / total_entities, 4),
        "avg_true_links_per_s1": round(total_true_links / total_entities, 4),
        "true_singletons": true_singletons,
        "predicted_singletons": pred_singletons,
        "correct_singletons": correct_singletons,
        "false_merge_singletons": false_merge_singletons,
        "singleton_accuracy": round(singleton_acc, 6),
        "f05_distribution": {
            k: {
                "count": v,
                "percentage": round(100.0 * v / total_entities, 2)
            } for k, v in f05_bins.items()
        }
    }


def run_unit_tests():
    """Verify correctness on official challenge examples and corner cases."""
    print("Running metrics.py unit tests...")

    # Test 1: Official example from README.md
    # Ground truth: [S2-00047, S3-00812] (2 matches)
    # Prediction:   [S2-00047, S2-00193, S3-00812] (3 matches, 2 TP, 1 FP)
    # Precision = 2/3, Recall = 1.0 => F0.5 = 5/7 ≈ 0.714286
    f05, p, r, tp, fp, fn = compute_entity_metrics(
        {"S2-00047", "S3-00812"},
        {"S2-00047", "S2-00193", "S3-00812"}
    )
    assert abs(p - 2/3) < 1e-6, f"Precision mismatch: {p}"
    assert abs(r - 1.0) < 1e-6, f"Recall mismatch: {r}"
    assert abs(f05 - 5/7) < 1e-6, f"F0.5 mismatch: {f05}"
    assert tp == 2 and fp == 1 and fn == 0
    print("  [PASS] Test 1 passed (Official README example)")

    # Test 2: True singleton correctly identified
    f05, p, r, tp, fp, fn = compute_entity_metrics(set(), set())
    assert f05 == 1.0 and p == 1.0 and r == 1.0 and tp == 0 and fp == 0 and fn == 0
    print("  [PASS] Test 2 passed (True singleton correctly identified)")

    # Test 3: True singleton false merge (predicts candidate)
    f05, p, r, tp, fp, fn = compute_entity_metrics(set(), {"S2-99999"})
    assert f05 == 0.0 and p == 0.0 and r == 0.0 and fp == 1
    print("  [PASS] Test 3 passed (True singleton false merge penalized to 0.0)")

    # Test 4: True non-singleton missed completely (predicts empty)
    f05, p, r, tp, fp, fn = compute_entity_metrics({"S2-00001", "S3-00002"}, set())
    assert f05 == 0.0 and p == 0.0 and r == 0.0 and fn == 2
    print("  [PASS] Test 4 passed (Missed matches penalized to 0.0)")

    # Test 5: Perfect match
    f05, p, r, tp, fp, fn = compute_entity_metrics({"S2-1", "S3-2"}, {"S2-1", "S3-2"})
    assert f05 == 1.0 and p == 1.0 and r == 1.0 and tp == 2 and fp == 0 and fn == 0
    print("  [PASS] Test 5 passed (Perfect match scores 1.0)")

    # Test 6: Zero overlap on non-empty sets
    f05, p, r, tp, fp, fn = compute_entity_metrics({"S2-1"}, {"S3-2"})
    assert f05 == 0.0 and tp == 0 and fp == 1 and fn == 1
    print("  [PASS] Test 6 passed (Disjoint sets score 0.0)")

    # Test 7: Batch evaluate_predictions
    gt = {
        "S1-1": {"S2-1", "S3-1"}, # 2 matches, predict 2 matches => F0.5 = 1.0
        "S1-2": set(),            # singleton, predict empty => F0.5 = 1.0
        "S1-3": set(),            # singleton, predict 1 match => F0.5 = 0.0
        "S1-4": {"S2-4"},         # 1 match, predict empty => F0.5 = 0.0
    }
    pred = {
        "S1-1": {"S2-1", "S3-1"},
        "S1-2": set(),
        "S1-3": {"S2-99"},
        "S1-4": set(),
    }
    res = evaluate_predictions(gt, pred)
    # Expected macro F0.5: (1.0 + 1.0 + 0.0 + 0.0) / 4 = 0.50
    assert abs(res["macro_f05"] - 0.50) < 1e-6, f"Batch Macro F0.5 mismatch: {res['macro_f05']}"
    assert res["true_singletons"] == 2
    assert res["correct_singletons"] == 1
    assert res["false_merge_singletons"] == 1
    assert res["total_s1_entities"] == 4
    print("  [PASS] Test 7 passed (Batch evaluate_predictions)")

    print("All metrics unit tests passed successfully!")


if __name__ == "__main__":
    run_unit_tests()
