#!/usr/bin/env python3
"""
src/train_and_eval_matcher.py — Amazon ML Challenge 2026 Business Entity Resolution
Supervised GBDT Matcher & Nested Union Validation Pipeline

1. Loads deterministic train/val split (zero S1 leakage).
2. Builds blocking channels with fixed Channel B (sub-token / rare words) and Channel D (selective n-grams).
3. Evaluates entity-level candidate coverage for A, A+B, A+B+C.
4. Generates training pairs from train split candidates:
   - Positives: True S1->S2/S3 matches in candidate set.
   - Hard negatives: Non-matching candidate pairs retrieved by blocking (highest risk of false merges).
5. Trains LightGBM pairwise binary classification matcher.
6. Evaluates the SAME trained model across nested candidate unions on the validation split:
   - Union A (Exact Legal Name)
   - Union A+B (Exact Legal + Rare Token)
   - Union A+B+C (Exact Legal + Rare Token + Address Derived)
   - Union A+B+C+D (Extended with Relaxed N-grams)
7. Computes and reports official Macro F0.5, Precision, Recall, Candidate Volume, and Singleton Accuracy.
"""

import argparse
import csv
import gc
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional

import numpy as np
import lightgbm as lgb

from metrics import compute_entity_metrics, evaluate_predictions
from normalize import (
    normalize_clean,
    normalize_legal_name,
)
from features import extract_pair_features, FEATURE_NAMES

# Ensure UTF-8 output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

_DIGIT_SEQ_RE = re.compile(r'\b\d+\b')
_POSTAL_RE = re.compile(r'\b\d{5,6}\b')
_STOP_WORDS = {
    'and', 'the', 'of', 'in', 'at', 'on', 'for', 'with', 'to', 'a', 'an',
    'private', 'limited', 'corporation', 'incorporated', 'company', 'llc', 'sarl', 'sas', 'gmbh',
    'pvt', 'ltd', 'corp', 'inc', 'co', 'services', 'enterprises', 'trading', 'solutions', 'associates'
}


def get_name_tokens(clean_name: str) -> List[str]:
    toks = clean_name.split()
    return [t for t in toks if len(t) >= 3 and t not in _STOP_WORDS]


def get_char_ngrams(clean_name: str, n: int = 3) -> List[str]:
    s = f" {clean_name} "
    if len(s) < n:
        return [s]
    return [s[i:i+n] for i in range(len(s) - n + 1)]


def extract_address_keys(country: str, addr: str) -> List[Tuple[str, str]]:
    keys = []
    if not addr:
        return keys
    clean_a = normalize_clean(addr)
    postals = _POSTAL_RE.findall(clean_a)
    digits = _DIGIT_SEQ_RE.findall(clean_a)
    if postals:
        keys.append((f"{country}_postal", postals[0]))
    words = [w for w in clean_a.split() if w not in _STOP_WORDS and not w.isdigit() and len(w) >= 3]
    if digits and words:
        keys.append((f"{country}_num_word", f"{digits[0]}_{words[0]}"))
    elif words and len(words) >= 2:
        keys.append((f"{country}_addr_words", f"{words[0]}_{words[1]}"))
    return keys


def run_pipeline(
    data_dir: str,
    output_dir: str,
    split_dir: str,
    train_sample_s1: int = 40000,
    val_sample_s1: int = 25000,
    max_bucket_size: int = 80,
    cap_per_channel: int = 30,
    threshold: float = 0.50,
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train", "train_source3.tsv")

    # 1. Load Split
    val_split_file = os.path.join(split_dir, "val_s1_ids.json")
    train_split_file = os.path.join(split_dir, "train_s1_ids.json")
    with open(val_split_file, 'r', encoding='utf-8') as f:
        val_s1_ids = set(json.load(f))
    with open(train_split_file, 'r', encoding='utf-8') as f:
        train_s1_ids = set(json.load(f))

    print(f"[{time.strftime('%H:%M:%S')}] Loaded split: {len(train_s1_ids):,} train S1, {len(val_s1_ids):,} val S1.")

    # Load Ground Truth
    truth: Dict[str, Set[str]] = {}
    with open(gt_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            if not row: continue
            s1_id = row[0].strip()
            matched = row[1].strip() if len(row) > 1 else ""
            truth[s1_id] = {x.strip() for x in matched.split(',') if x.strip()} if matched else set()

    # 2. Vocabulary & DF for selective Channel B & D
    print(f"[{time.strftime('%H:%M:%S')}] Pre-scanning candidate vocabulary & document frequencies...")
    token_doc_freq = Counter()
    c3_doc_freq = Counter()
    total_candidate_docs = 0

    for path in [s2_path, s3_path]:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)
            for row in reader:
                if len(row) < 4: continue
                total_candidate_docs += 1
                name_clean = normalize_legal_name(row[1])
                for t in set(get_name_tokens(name_clean)):
                    token_doc_freq[t] += 1
                for ng3 in set(get_char_ngrams(name_clean, 3)):
                    c3_doc_freq[ng3] += 1

    valid_tokens = {t: math.log((total_candidate_docs + 1) / (df + 1)) for t, df in token_doc_freq.items() if 2 <= df <= max_bucket_size}
    valid_c3 = {ng: math.log((total_candidate_docs + 1) / (df + 1)) for ng, df in c3_doc_freq.items() if 3 <= df <= max_bucket_size}
    del token_doc_freq
    del c3_doc_freq
    gc.collect()

    # 3. Build Candidate Inverted Indices (S2, S3)
    print(f"[{time.strftime('%H:%M:%S')}] Indexing S2 and S3...")
    idx_a = defaultdict(list)
    idx_b = defaultdict(list)
    idx_c = defaultdict(list)
    idx_d = defaultdict(list)

    # Cache candidate entity features in RAM for ultra-fast pair extraction
    cand_records: Dict[str, Tuple[str, Set[str], str, Set[str], Set[str], Set[str]]] = {}

    for path in [s2_path, s3_path]:
        s_name = os.path.basename(path)
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)
            for row in reader:
                if len(row) < 4: continue
                eid, b_name, b_addr, country = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()

                legal_n = normalize_legal_name(b_name)
                addr_c = normalize_clean(b_addr)
                name_toks = set(get_name_tokens(legal_n))
                addr_toks = set(get_name_tokens(addr_c))
                postals = set(_POSTAL_RE.findall(addr_c))
                digits = set(_DIGIT_SEQ_RE.findall(addr_c))

                # Store pre-tokenized features
                cand_records[eid] = (legal_n, name_toks, addr_c, addr_toks, postals, digits)

                # Channel A
                if legal_n:
                    idx_a[(country, legal_n)].append(eid)

                # Channel B
                b_toks = [t for t in name_toks if t in valid_tokens]
                b_toks.sort(key=lambda t: valid_tokens[t], reverse=True)
                for t in b_toks[:3]:
                    idx_b[(country, t)].append(eid)

                # Channel C
                addr_keys = extract_address_keys(country, b_addr)
                for k_type, k_val in addr_keys:
                    idx_c[(k_type, k_val)].append(eid)

                # Channel D (Relaxed selective char 3-grams)
                c3s = [ng for ng in get_char_ngrams(legal_n, 3) if ng in valid_c3]
                c3s.sort(key=lambda ng: valid_c3[ng], reverse=True)
                for ng in c3s[:2]:
                    idx_d[(country, ng)].append(eid)

    print(f"[{time.strftime('%H:%M:%S')}] Finished candidate indexing. Cached {len(cand_records):,} candidate records.")

    # 4. Sample Training S1 subset to build training pairs
    print(f"[{time.strftime('%H:%M:%S')}] Generating training dataset with hard negatives...")
    rng = random.Random(42)
    train_s1_sample = set(rng.sample(list(train_s1_ids), min(train_sample_s1, len(train_s1_ids))))
    val_s1_sample = set(rng.sample(list(val_s1_ids), min(val_sample_s1, len(val_s1_ids))))

    # Read S1 records for train sample & val sample
    s1_records: Dict[str, Tuple[str, str, Tuple[str, Set[str], str, Set[str], Set[str], Set[str]]]] = {}
    with open(s1_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            if len(row) < 4: continue
            eid = row[0].strip()
            if eid in train_s1_sample or eid in val_s1_sample:
                country = row[3].strip()
                legal_n = normalize_legal_name(row[1].strip())
                addr_c = normalize_clean(row[2].strip())
                name_toks = set(get_name_tokens(legal_n))
                addr_toks = set(get_name_tokens(addr_c))
                postals = set(_POSTAL_RE.findall(addr_c))
                digits = set(_DIGIT_SEQ_RE.findall(addr_c))
                s1_records[eid] = (country, legal_n, (legal_n, name_toks, addr_c, addr_toks, postals, digits))

    def retrieve_candidates(s1_id: str) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
        country, legal_n, s1_feat = s1_records[s1_id]
        name_clean, name_toks, addr_clean, addr_toks, postals, digits = s1_feat

        # A
        cand_a = set()
        if legal_n:
            m = idx_a.get((country, legal_n))
            if m: cand_a = set(m[:cap_per_channel])

        # B
        cand_b = set()
        b_toks = [t for t in name_toks if t in valid_tokens]
        b_toks.sort(key=lambda t: valid_tokens[t], reverse=True)
        for t in b_toks[:3]:
            m = idx_b.get((country, t))
            if m: cand_b.update(m[:cap_per_channel // 2])

        # C
        cand_c = set()
        addr_keys = extract_address_keys(country, addr_clean)
        for k_type, k_val in addr_keys:
            m = idx_c.get((k_type, k_val))
            if m: cand_c.update(m[:cap_per_channel // 2])

        # D
        cand_d = set()
        c3s = [ng for ng in get_char_ngrams(legal_n, 3) if ng in valid_c3]
        c3s.sort(key=lambda ng: valid_c3[ng], reverse=True)
        for ng in c3s[:2]:
            m = idx_d.get((country, ng))
            if m: cand_d.update(m[:cap_per_channel // 2])

        return cand_a, cand_b, cand_c, cand_d

    # Build Training Feature Matrix X, y
    X_train_list = []
    y_train_list = []

    print(f"[{time.strftime('%H:%M:%S')}] Extracting training pairs from A+B+C candidate set...")
    for s1_id in train_s1_sample:
        cand_a, cand_b, cand_c, _ = retrieve_candidates(s1_id)
        candidates = cand_a | cand_b | cand_c
        true_set = truth.get(s1_id, set())
        s1_feat = s1_records[s1_id][2]

        pos_cands = candidates & true_set
        neg_cands = candidates - true_set

        # Add all retrieved true positives
        for cid in pos_cands:
            if cid in cand_records:
                c_feat = cand_records[cid]
                feats = extract_pair_features(*s1_feat, *c_feat)
                X_train_list.append(feats)
                y_train_list.append(1)

        # Add hard negatives (sample up to 4 per positive, or 2 if singleton)
        n_neg = max(2, len(pos_cands) * 4)
        sample_negs = rng.sample(list(neg_cands), min(n_neg, len(neg_cands)))
        for cid in sample_negs:
            if cid in cand_records:
                c_feat = cand_records[cid]
                feats = extract_pair_features(*s1_feat, *c_feat)
                X_train_list.append(feats)
                y_train_list.append(0)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    print(f"  Training dataset: {len(y_train):,} pairs ({int(sum(y_train)):,} pos, {len(y_train)-int(sum(y_train)):,} hard negs).")

    # 5. Train LightGBM Matcher
    print(f"[{time.strftime('%H:%M:%S')}] Training LightGBM binary classifier...")
    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'min_data_in_leaf': 50,
        'n_jobs': -1,
        'verbose': -1,
        'random_state': 42
    }
    model = lgb.train(params, lgb_train, num_boost_round=150)
    print(f"[{time.strftime('%H:%M:%S')}] Model training complete.")

    # 6. Evaluate on Validation Split across Nested Unions
    print(f"[{time.strftime('%H:%M:%S')}] Evaluating trained matcher across nested candidate unions on validation split...")

    # We benchmark 4 nested candidate unions:
    # 1. Union A
    # 2. Union A+B
    # 3. Union A+B+C
    # 4. Union A+B+C+D
    union_names = ["Union_A", "Union_A_B", "Union_A_B_C", "Union_A_B_C_D"]
    
    # Store predictions: union_name -> {s1_id -> set of predicted matched IDs}
    union_preds: Dict[str, Dict[str, Set[str]]] = {u: {} for u in union_names}
    union_cand_volumes: Dict[str, List[int]] = {u: [] for u in union_names}
    union_recovered_links: Dict[str, int] = {u: 0 for u in union_names}
    total_val_links = sum(len(truth.get(s1_id, set())) for s1_id in val_s1_sample)

    for s1_id in val_s1_sample:
        cand_a, cand_b, cand_c, cand_d = retrieve_candidates(s1_id)
        true_set = truth.get(s1_id, set())
        s1_feat = s1_records[s1_id][2]

        cand_unions = {
            "Union_A": cand_a,
            "Union_A_B": cand_a | cand_b,
            "Union_A_B_C": cand_a | cand_b | cand_c,
            "Union_A_B_C_D": cand_a | cand_b | cand_c | cand_d,
        }

        # Predict for each candidate union using the SAME trained model
        for u_name, cands in cand_unions.items():
            union_cand_volumes[u_name].append(len(cands))
            union_recovered_links[u_name] += len(cands & true_set)

            pred_set = set()
            if cands:
                cand_list = [c for c in cands if c in cand_records]
                if cand_list:
                    pair_feats = [extract_pair_features(*s1_feat, *cand_records[c]) for c in cand_list]
                    scores = model.predict(np.array(pair_feats, dtype=np.float32))
                    for c, s in zip(cand_list, scores):
                        if s >= threshold:
                            pred_set.add(c)

            union_preds[u_name][s1_id] = pred_set

    # 7. Compute Official Macro F0.5 for each Union
    results_summary = []
    print("\n" + "="*80)
    print("VALIDATION BENCHMARK RESULTS ACROSS NESTED UNIONS")
    print("="*80)

    for u_name in union_names:
        preds = union_preds[u_name]
        eval_metrics = evaluate_predictions(truth, preds, eval_entity_ids=val_s1_sample)
        
        cands_per_s1 = union_cand_volumes[u_name]
        cand_recall = union_recovered_links[u_name] / total_val_links if total_val_links > 0 else 0.0

        res_row = {
            "configuration": u_name,
            "macro_f05": eval_metrics["macro_f05"],
            "precision": eval_metrics["macro_precision_on_predicted"],
            "recall": eval_metrics["macro_recall_on_true_matches"],
            "pooled_f05": eval_metrics["pooled_f05"],
            "singleton_accuracy": eval_metrics["singleton_accuracy"],
            "candidate_recall": round(cand_recall * 100, 2),
            "avg_candidates_per_s1": round(np.mean(cands_per_s1), 2),
            "median_cands": int(np.median(cands_per_s1)),
            "p95_cands": int(np.percentile(cands_per_s1, 95)),
            "total_candidates": int(sum(cands_per_s1)),
            "total_predicted_links": eval_metrics["total_pred_links"],
        }
        results_summary.append(res_row)

        print(f"\nConfiguration: {u_name}")
        print(f"  Macro F0.5:         {res_row['macro_f05']:.4f}")
        print(f"  Candidate Recall:   {res_row['candidate_recall']:.2f}%")
        print(f"  Avg Candidates/S1:  {res_row['avg_candidates_per_s1']}")
        print(f"  Precision:          {res_row['precision']:.4f}")
        print(f"  Recall:             {res_row['recall']:.4f}")
        print(f"  Singleton Acc:      {res_row['singleton_accuracy']:.4f}")

    # Save results to JSON and Markdown report
    out_json = os.path.join(output_dir, "matcher_nested_union_benchmark.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n[{time.strftime('%H:%M:%S')}] Benchmark complete in {round(time.time() - start_time, 2)}s.")
    return results_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="student_resource/dataset")
    parser.add_argument("--output-dir", default="outputs/matcher")
    parser.add_argument("--split-dir", default="outputs/split")
    parser.add_argument("--train-sample-s1", type=int, default=40000)
    parser.add_argument("--val-sample-s1", type=int, default=25000)
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split_dir=args.split_dir,
        train_sample_s1=args.train_sample_s1,
        val_sample_s1=args.val_sample_s1,
        threshold=args.threshold,
    )
