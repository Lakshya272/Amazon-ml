#!/usr/bin/env python3
"""
src/validation_split.py — Amazon ML Challenge 2026 Business Entity Resolution
Deterministic, Leakage-Free Validation Split Generator

Creates a deterministic 80/20 train/validation split at the Source 1 entity level.
- Stratified by country (US, India) and match cardinality (singleton vs 1, 2, 3+ matches).
- Strictly zero S1 leakage between train and validation.
- Output: outputs/split/validation_s1_ids.json and outputs/split/train_s1_ids.json
"""

import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any

def generate_validation_split(
    data_dir: str,
    output_dir: str,
    val_ratio: float = 0.20,
    seed: int = 42
) -> Dict[str, Any]:
    print(f"[{time.strftime('%H:%M:%S')}] Generating deterministic {int(val_ratio*100)}% validation split...")
    os.makedirs(output_dir, exist_ok=True)

    gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    s1_path = os.path.join(data_dir, "train", "train_source1.tsv")

    # Read S1 countries
    s1_country = {}
    print(f"[{time.strftime('%H:%M:%S')}] Reading S1 countries...")
    with open(s1_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            if len(row) >= 4:
                s1_country[row[0].strip()] = row[3].strip()

    # Read ground truth link counts
    print(f"[{time.strftime('%H:%M:%S')}] Reading ground truth link counts...")
    s1_match_counts = {}
    with open(gt_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            if not row or len(row) < 1: continue
            s1_id = row[0].strip()
            matched = row[1].strip() if len(row) > 1 else ""
            n_matches = len([x for x in matched.split(',') if x.strip()]) if matched else 0
            s1_match_counts[s1_id] = n_matches

    # Group into strata: (country, match_bucket)
    # Buckets: 0 (singleton), 1, 2, 3-4, 5+
    strata = defaultdict(list)
    for s1_id, country in s1_country.items():
        n_m = s1_match_counts.get(s1_id, 0)
        if n_m == 0:
            bucket = "singleton"
        elif n_m == 1:
            bucket = "1_match"
        elif n_m == 2:
            bucket = "2_matches"
        elif n_m <= 4:
            bucket = "3_4_matches"
        else:
            bucket = "5_plus_matches"
        strata[(country, bucket)].append(s1_id)

    rng = random.Random(seed)
    train_ids = []
    val_ids = []

    print(f"[{time.strftime('%H:%M:%S')}] Stratified partitioning...")
    for (country, bucket), ids in sorted(strata.items()):
        rng.shuffle(ids)
        n_val = int(len(ids) * val_ratio)
        val_subset = ids[:n_val]
        train_subset = ids[n_val:]
        val_ids.extend(val_subset)
        train_ids.extend(train_subset)
        print(f"  Stratum ({country}, {bucket}): {len(ids):,} total -> {len(val_subset):,} val ({round(len(val_subset)/len(ids)*100, 1)}%)")

    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)

    # Verification: Mutual exclusivity
    assert len(train_ids_set & val_ids_set) == 0, "CRITICAL: Data leakage detected between train and val!"
    assert len(train_ids) + len(val_ids) == len(s1_country), "CRITICAL: Mismatch in total entity count!"

    val_path = os.path.join(output_dir, "val_s1_ids.json")
    train_path = os.path.join(output_dir, "train_s1_ids.json")

    with open(val_path, 'w', encoding='utf-8') as f:
        json.dump(val_ids, f)
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_ids, f)

    meta = {
        "val_ratio": val_ratio,
        "seed": seed,
        "total_s1": len(s1_country),
        "train_s1_count": len(train_ids),
        "val_s1_count": len(val_ids),
        "val_percentage": round(100.0 * len(val_ids) / len(s1_country), 2),
        "val_singletons": sum(1 for eid in val_ids if s1_match_counts.get(eid, 0) == 0),
        "val_total_links": sum(s1_match_counts.get(eid, 0) for eid in val_ids),
    }

    meta_path = os.path.join(output_dir, "split_metadata.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f"[{time.strftime('%H:%M:%S')}] Saved split:")
    print(f"  Train: {len(train_ids):,} entities (80.0%)")
    print(f"  Val:   {len(val_ids):,} entities ({meta['val_percentage']}%) | {meta['val_total_links']:,} true links | {meta['val_singletons']:,} singletons")
    return meta

if __name__ == "__main__":
    data_dir = "student_resource/dataset" if os.path.exists("student_resource/dataset") else "dataset"
    out_dir = "outputs/split"
    generate_validation_split(data_dir, out_dir)
