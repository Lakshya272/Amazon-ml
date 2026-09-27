#!/usr/bin/env python3
"""Measure whether cached Granite top-k candidates improve the A-L matcher.

This is a validation-only experiment. It evaluates both A-L and A-L plus Granite
on the exact 50,000 validation IDs used by EXP-014, with one trained model and
one frozen decision config. It never reads test data.
"""

import argparse
import csv
import json
import os
import time
from collections import Counter, OrderedDict

import lightgbm as lgb
import numpy as np

from features import EntityRecord, FEATURE_NAMES, extract_pair_features_fast
from metrics import evaluate_predictions
from normalize import normalize_clean, normalize_legal_name
from run_test_inference_a_l import CandidateIndex, scan_vocabulary


def entity(row):
    eid, name, address, country = (value.strip() for value in row[:4])
    return EntityRecord(
        eid, name, normalize_clean(name), normalize_legal_name(name),
        address, normalize_clean(address), country,
    )


def decide(scored, config):
    if not scored:
        return set()
    best = max(score for _, score in scored)
    if best < config["tau_null"]:
        return set()
    floor = max(config["tau"], best - config["delta_multi"])
    return {eid for eid, score in scored if score >= floor}


def run(args):
    started = time.time()
    with open(args.dense_candidates, encoding="utf-8") as handle:
        dense = json.load(handle)
    query_ids = set(dense)
    config = json.load(open(args.config, encoding="utf-8"))
    model = lgb.Booster(model_file=args.model)
    if model.feature_name() != FEATURE_NAMES:
        raise ValueError("Model feature order does not match the feature pipeline")

    truth = {}
    gt_path = os.path.join(args.data_dir, "train", "train_ground_truth.tsv")
    with open(gt_path, encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)
        for row in reader:
            if row and row[0] in query_ids:
                truth[row[0]] = set(row[1].split(",")) if len(row) > 1 and row[1] else set()

    s1 = {}
    s1_path = os.path.join(args.data_dir, "train", "train_source1.tsv")
    with open(s1_path, encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) >= 4 and row[0] in query_ids:
                s1[row[0]] = entity(row)
    if set(s1) != query_ids or set(truth) != query_ids:
        raise ValueError(f"Validation input mismatch: dense={len(query_ids)}, s1={len(s1)}, truth={len(truth)}")

    candidate_paths = [
        os.path.join(args.data_dir, "train", "train_source2.tsv"),
        os.path.join(args.data_dir, "train", "train_source3.tsv"),
    ]
    print("Scanning training candidate vocabulary...", flush=True)
    _, valid_tokens, valid_c3 = scan_vocabulary(candidate_paths, config["max_bucket_size"])
    print("Building A-L index...", flush=True)
    index = CandidateIndex(valid_tokens, valid_c3, config["cap_per_channel"])
    for path in candidate_paths:
        with open(path, encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            next(reader, None)
            for row in reader:
                if len(row) >= 4:
                    index.add(row)
        print(f"Indexed {len(index.raw):,} candidates", flush=True)
    del valid_tokens, valid_c3

    candidate_counts = Counter(value[2] for value in index.raw.values())
    universe = sum(
        sum(record.country == country for record in s1.values()) * n
        for country, n in candidate_counts.items()
    )
    baseline_preds, union_preds = {}, {}
    baseline_pairs = union_pairs = baseline_recovered = union_recovered = total_links = 0
    cache = OrderedDict()

    def get_candidate(cid):
        if cid in cache:
            cache.move_to_end(cid)
            return cache[cid]
        name, address, country = index.raw[cid]
        rec = EntityRecord(cid, name, normalize_clean(name), normalize_legal_name(name),
                            address, normalize_clean(address), country)
        cache[cid] = rec
        if len(cache) > args.cache_size:
            cache.popitem(last=False)
        return rec

    print("Scoring A-L baseline and A-L+Granite candidate union...", flush=True)
    for i, (sid, source) in enumerate(s1.items(), 1):
        hits = index.retrieve(source)
        base_ids = list(hits)
        union_hits = hits.copy()
        for cid in dense[sid]:
            if cid not in union_hits:
                union_hits[cid] = 0
        union_ids = list(union_hits)
        union_features = [
            extract_pair_features_fast(source, get_candidate(cid), union_hits[cid])
            for cid in union_ids
        ]
        scores = model.predict(np.asarray(union_features, dtype=np.float32), num_threads=args.threads)
        union_scored = list(zip(union_ids, scores))
        base_n = len(base_ids)
        base_scored = union_scored[:base_n]
        baseline_preds[sid] = decide(base_scored, config)
        union_preds[sid] = decide(union_scored, config)
        base_set, union_set, true = set(base_ids), set(union_ids), truth[sid]
        baseline_pairs += len(base_set)
        union_pairs += len(union_set)
        total_links += len(true)
        baseline_recovered += len(base_set & true)
        union_recovered += len(union_set & true)
        if i % 5000 == 0:
            print(f"Processed {i:,}/{len(s1):,} S1; {union_pairs:,} union candidates", flush=True)

    baseline = evaluate_predictions(truth, baseline_preds, s1.keys())
    union = evaluate_predictions(truth, union_preds, s1.keys())
    result = {
        "experiment": "EXP-016 A-L plus Granite top-20 candidate union",
        "validation_entities": len(s1),
        "model_validation_config": {k: config[k] for k in ("tau", "tau_null", "delta_multi")},
        "baseline_A_L": baseline,
        "union_A_L_plus_Granite": union,
        "candidate_metrics": {
            "baseline_pairs": baseline_pairs,
            "union_pairs": union_pairs,
            "baseline_candidate_recall": baseline_recovered / total_links,
            "union_candidate_recall": union_recovered / total_links,
            "baseline_reduction_ratio": 1 - baseline_pairs / universe,
            "union_reduction_ratio": 1 - union_pairs / universe,
            "same_country_cartesian_pairs": universe,
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, "fusion_metrics.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="student_resource/dataset")
    parser.add_argument("--dense-candidates", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-size", type=int, default=100000)
    parser.add_argument("--threads", type=int, default=8)
    run(parser.parse_args())
