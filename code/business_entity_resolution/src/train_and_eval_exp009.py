#!/usr/bin/env python3
"""
src/train_and_eval_exp009.py — Amazon ML Challenge 2026 Business Entity Resolution
EXP-009: End-to-End Supervised Pairwise GBDT Matcher over Full Blocking Channels (A...L)
with Entity-Level Null-Margin Calibration & Singleton Protection.

Key Innovations:
1. Full 12-channel blocking candidate set (A...L) with 81.62% link recall and 0.9175 Oracle Macro F0.5.
2. Rich 32-feature pair extraction via rapidfuzz C++ acceleration, phonetic transliteration, and channel hit signals.
3. Stratified hard-negative sampling with dedicated singleton hard-negative pairs.
4. Entity-level null margin calibration:
   - s_max < tau_null -> predict empty set (1.0 for true singletons)
   - s_ij >= tau_match and s_ij >= s_max - delta_multi -> predict match
5. Rigorous grid sweep to directly maximize official Macro F0.5 on validation split.
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
from normalize import normalize_clean, normalize_legal_name
from transliterate import transliterate_indic_to_latin
from features import (
    EntityRecord,
    extract_pair_features_fast,
    FEATURE_NAMES,
)

# Ensure UTF-8 output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

_DIGIT_SEQ_RE = re.compile(r'\b\d+\b')
_POSTAL_RE = re.compile(r'\b\d{5,6}\b')
_LANDMARK_RE = re.compile(r'\b(?:near|opp|opposite|behind|beside|adj|adjacent|next\s+to|opp\.)\s+([a-zA-Z0-9]+)', re.IGNORECASE)
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


def extract_component_drop_address_keys(country: str, addr: str) -> List[Tuple[str, str]]:
    keys = []
    if not addr:
        return keys
    clean_a = normalize_clean(addr)
    words = [w for w in clean_a.split() if w not in _STOP_WORDS and not w.isdigit() and len(w) >= 4]
    if len(words) >= 3:
        keys.append((f"{country}_addr_w1", words[1]))
        keys.append((f"{country}_addr_w2", words[2]))
    elif len(words) >= 2:
        keys.append((f"{country}_addr_w1", words[1]))
    return keys


def extract_drop_one_name_keys(country: str, legal_name: str) -> List[Tuple[str, str]]:
    toks = get_name_tokens(legal_name)
    keys = []
    if len(toks) >= 3:
        keys.append((f"{country}_drop0", f"{toks[1]}_{toks[2]}"))
        keys.append((f"{country}_drop1", f"{toks[0]}_{toks[2]}"))
        keys.append((f"{country}_drop2", f"{toks[0]}_{toks[1]}"))
    elif len(toks) == 2:
        keys.append((f"{country}_pair", f"{toks[0]}_{toks[1]}"))
    return keys


def extract_sorted_neighborhood_key(country: str, legal_name: str) -> Optional[Tuple[str, str]]:
    comp = "".join(legal_name.split())
    if len(comp) >= 5:
        return (f"{country}_snk", comp[:8])
    return None


def extract_numeric_address_keys(country: str, addr: str) -> List[Tuple[str, str]]:
    keys = []
    if not addr:
        return keys
    clean_a = normalize_clean(addr)
    postals = _POSTAL_RE.findall(clean_a)
    digits = [d for d in _DIGIT_SEQ_RE.findall(clean_a) if len(d) <= 5 and d != '0']
    
    if postals and digits:
        p = postals[0]
        for d in digits[:2]:
            if d != p:
                keys.append((f"{country}_num_post5", f"{d}_{p[:5]}"))
                if len(p) >= 3:
                    keys.append((f"{country}_num_post3", f"{d}_{p[:3]}"))
    elif len(digits) >= 2:
        keys.append((f"{country}_num_pair", f"{digits[0]}_{digits[1]}"))
    return keys


def extract_landmark_address_keys(country: str, addr: str) -> List[Tuple[str, str]]:
    keys = []
    if not addr:
        return keys
    matches = _LANDMARK_RE.findall(addr)
    for lm in matches:
        lm_clean = lm.lower().strip()
        if len(lm_clean) >= 4 and lm_clean not in _STOP_WORDS:
            keys.append((f"{country}_landmark", lm_clean))
            
    clean_a = normalize_clean(addr)
    words = [w for w in clean_a.split() if w not in _STOP_WORDS and not w.isdigit() and len(w) >= 4]
    if len(words) >= 4:
        keys.append((f"{country}_addr_edge", f"{words[0]}_{words[-1]}"))
        keys.append((f"{country}_addr_edge2", f"{words[1]}_{words[-1]}"))
    return keys


def extract_transliterated_keys(country: str, name: str) -> List[Tuple[str, str]]:
    keys = []
    if not name:
        return keys
    rom = transliterate_indic_to_latin(name)
    if rom:
        legal_rom = normalize_legal_name(rom)
        if legal_rom:
            keys.append((f"{country}_rom_exact", legal_rom))
            comp = "".join(legal_rom.split())
            if len(comp) >= 5:
                keys.append((f"{country}_rom_snk", comp[:8]))
            toks = [t for t in legal_rom.split() if len(t) >= 4 and t not in _STOP_WORDS]
            if len(toks) >= 2:
                keys.append((f"{country}_rom_pair", f"{toks[0]}_{toks[1]}"))
            elif len(toks) == 1:
                keys.append((f"{country}_rom_word", toks[0]))
    return keys


def extract_typo_tolerant_keys(country: str, name: str, addr: str) -> List[Tuple[str, str]]:
    keys = []
    if not name:
        return keys
    clean_n = normalize_clean(name)
    words = [w for w in clean_n.split() if w not in _STOP_WORDS and len(w) >= 5]
    if not words:
        return keys
    
    for w in words[:2]:
        sqz = re.sub(r'(.)\1+', r'\1', w)
        if sqz != w and len(sqz) >= 4:
            keys.append((f"{country}_sqz", sqz))
            
    postals = _POSTAL_RE.findall(addr) if addr else []
    p2 = postals[0][:2] if postals and len(postals[0]) >= 2 else ""
    
    for w in words[:2]:
        if 6 <= len(w) <= 12:
            if p2:
                keys.append((f"{country}_del_{p2}", w))
            elif len(w) >= 8:
                keys.append((f"{country}_del_rare", w))
            for i in range(len(w)):
                del_w = w[:i] + w[i+1:]
                if p2:
                    keys.append((f"{country}_del_{p2}", del_w))
                elif len(w) >= 8:
                    keys.append((f"{country}_del_rare", del_w))
    return keys


def extract_acronym_keys(country: str, name: str, addr: str) -> List[Tuple[str, str]]:
    keys = []
    if not name:
        return keys
    clean_n = normalize_clean(name)
    words = [w for w in clean_n.split() if w not in _STOP_WORDS]
    if not words:
        return keys
    
    postals = _POSTAL_RE.findall(addr) if addr else []
    p3 = postals[0][:3] if postals and len(postals[0]) >= 3 else ""
    
    if len(words) >= 2:
        acr = "".join(w[0] for w in words if w[0].isalnum())
        if 2 <= len(acr) <= 6:
            if p3:
                keys.append((f"{country}_acr_p3", f"{acr}_{p3}"))
            else:
                clean_a = normalize_clean(addr) if addr else ""
                a_words = [aw for aw in clean_a.split() if aw not in _STOP_WORDS and len(aw) >= 4]
                if a_words:
                    keys.append((f"{country}_acr_city", f"{acr}_{a_words[-1]}"))
    elif len(words) == 1 and 2 <= len(words[0]) <= 5 and words[0].isalpha():
        acr = words[0]
        if p3:
            keys.append((f"{country}_acr_p3", f"{acr}_{p3}"))
        else:
            clean_a = normalize_clean(addr) if addr else ""
            a_words = [aw for aw in clean_a.split() if aw not in _STOP_WORDS and len(aw) >= 4]
            if a_words:
                keys.append((f"{country}_acr_city", f"{acr}_{a_words[-1]}"))
    return keys


def run_training_and_evaluation(
    data_dir: str,
    output_dir: str,
    split_dir: str,
    train_sample_s1: int = 40000,
    val_sample_s1: int = 10000,
    max_bucket_size: int = 80,
    cap_per_channel: int = 30,
):
    print("=" * 80)
    print("EXP-009: SUPERVISED GBDT MATCHER (A...L) + NULL MARGIN CALIBRATION")
    print(f"Train S1 sample: {train_sample_s1:,} | Val S1 sample: {val_sample_s1:,}")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    # 1. Load Split
    val_split_file = os.path.join(split_dir, "val_s1_ids.json")
    train_split_file = os.path.join(split_dir, "train_s1_ids.json")
    with open(val_split_file, 'r', encoding='utf-8') as f:
        val_s1_ids = set(json.load(f))
    with open(train_split_file, 'r', encoding='utf-8') as f:
        train_s1_ids = set(json.load(f))

    print(f"[{time.strftime('%H:%M:%S')}] Loaded split: {len(train_s1_ids):,} train S1, {len(val_s1_ids):,} val S1.")

    # 2. Sample Train & Val S1 subsets
    rng = random.Random(42)
    eval_train_s1 = sorted(rng.sample(list(train_s1_ids), min(train_sample_s1, len(train_s1_ids))))
    eval_val_s1 = sorted(rng.sample(list(val_s1_ids), min(val_sample_s1, len(val_s1_ids))))
    all_needed_s1 = set(eval_train_s1) | set(eval_val_s1)

    # 3. Load Ground Truth
    gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    truth: Dict[str, Set[str]] = {}
    with open(gt_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            if not row: continue
            s1_id = row[0].strip()
            if s1_id in all_needed_s1:
                matched = row[1].strip() if len(row) > 1 else ""
                truth[s1_id] = {x.strip() for x in matched.split(',') if x.strip()} if matched else set()

    # 4. Load S1 Records
    s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
    s1_records: Dict[str, EntityRecord] = {}
    with open(s1_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            if len(row) < 4: continue
            eid = row[0].strip()
            if eid in all_needed_s1:
                b_name, b_addr, country = row[1].strip(), row[2].strip(), row[3].strip()
                clean_n = normalize_clean(b_name)
                legal_n = normalize_legal_name(b_name)
                clean_a = normalize_clean(b_addr)
                s1_records[eid] = EntityRecord(eid, b_name, clean_n, legal_n, b_addr, clean_a, country)

    print(f"[{time.strftime('%H:%M:%S')}] Cached {len(s1_records):,} S1 EntityRecords.")

    # 5. Build S2/S3 Candidate Indices
    s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train", "train_source3.tsv")

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

    print(f"[{time.strftime('%H:%M:%S')}] Building inverted indices for Channels A through L...")
    # Initialize index dicts
    idx_a = defaultdict(list)
    idx_b = defaultdict(list)
    idx_c = defaultdict(list)
    idx_d = defaultdict(list)
    idx_e = defaultdict(list)
    idx_f = defaultdict(list)
    idx_g = defaultdict(list)
    idx_h = defaultdict(list)
    idx_i = defaultdict(list)
    idx_j = defaultdict(list)
    idx_k = defaultdict(list)
    idx_l = defaultdict(list)

    # Store raw candidate fields for lazy EntityRecord instantiation
    raw_cand_store: Dict[str, Tuple[str, str, str]] = {}

    for path in [s2_path, s3_path]:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)
            for row in reader:
                if len(row) < 4: continue
                eid, b_name, b_addr, country = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
                raw_cand_store[eid] = (b_name, b_addr, country)

                legal_n = normalize_legal_name(b_name)
                # A
                if legal_n: idx_a[(country, legal_n)].append(eid)
                # B
                clean_n = normalize_clean(b_name)
                b_toks = [t for t in get_name_tokens(clean_n) if t in valid_tokens]
                b_toks.sort(key=lambda t: valid_tokens[t], reverse=True)
                for t in b_toks[:3]: idx_b[(country, t)].append(eid)
                # C
                for k_t, k_v in extract_address_keys(country, b_addr): idx_c[(k_t, k_v)].append(eid)
                # D
                c3s = [ng for ng in get_char_ngrams(legal_n, 3) if ng in valid_c3]
                c3s.sort(key=lambda ng: valid_c3[ng], reverse=True)
                for ng in c3s[:2]: idx_d[(country, ng)].append(eid)
                # E
                for k_t, k_v in extract_drop_one_name_keys(country, legal_n): idx_e[(k_t, k_v)].append(eid)
                # F
                for k_t, k_v in extract_component_drop_address_keys(country, b_addr): idx_f[(k_t, k_v)].append(eid)
                # G
                snk = extract_sorted_neighborhood_key(country, legal_n)
                if snk: idx_g[snk].append(eid)
                # H
                for k_t, k_v in extract_numeric_address_keys(country, b_addr): idx_h[(k_t, k_v)].append(eid)
                # I
                for k_t, k_v in extract_landmark_address_keys(country, b_addr): idx_i[(k_t, k_v)].append(eid)
                # J
                for k_t, k_v in extract_transliterated_keys(country, b_name): idx_j[(k_t, k_v)].append(eid)
                # K
                for k_t, k_v in extract_typo_tolerant_keys(country, b_name, b_addr): idx_k[(k_t, k_v)].append(eid)
                # L
                for k_t, k_v in extract_acronym_keys(country, b_name, b_addr): idx_l[(k_t, k_v)].append(eid)

    print(f"[{time.strftime('%H:%M:%S')}] Indexing complete. Total indexed candidates: {len(raw_cand_store):,}.")

    # Function to retrieve candidates and count channel hits
    def retrieve_candidates_with_hits(e1: EntityRecord, raw_name: str, raw_addr: str) -> Dict[str, int]:
        hits: Dict[str, int] = Counter()
        c = e1.country

        # A
        if e1.legal_name:
            for cid in idx_a.get((c, e1.legal_name), [])[:cap_per_channel]: hits[cid] += 1
        # B
        b_toks = [t for t in e1.name_toks if t in valid_tokens]
        b_toks.sort(key=lambda t: valid_tokens[t], reverse=True)
        for t in b_toks[:3]:
            for cid in idx_b.get((c, t), [])[:cap_per_channel // 2]: hits[cid] += 1
        # C
        for k_t, k_v in extract_address_keys(c, raw_addr):
            for cid in idx_c.get((k_t, k_v), [])[:cap_per_channel // 2]: hits[cid] += 1
        # D
        c3s = [ng for ng in get_char_ngrams(e1.legal_name, 3) if ng in valid_c3]
        c3s.sort(key=lambda ng: valid_c3[ng], reverse=True)
        for ng in c3s[:2]:
            for cid in idx_d.get((c, ng), [])[:cap_per_channel // 2]: hits[cid] += 1
        # E
        for k_t, k_v in extract_drop_one_name_keys(c, e1.legal_name):
            for cid in idx_e.get((k_t, k_v), [])[:cap_per_channel // 2]: hits[cid] += 1
        # F
        for k_t, k_v in extract_component_drop_address_keys(c, raw_addr):
            for cid in idx_f.get((k_t, k_v), [])[:cap_per_channel // 2]: hits[cid] += 1
        # G
        snk = extract_sorted_neighborhood_key(c, e1.legal_name)
        if snk:
            for cid in idx_g.get(snk, [])[:cap_per_channel // 2]: hits[cid] += 1
        # H
        for k_t, k_v in extract_numeric_address_keys(c, raw_addr):
            for cid in idx_h.get((k_t, k_v), [])[:cap_per_channel // 2]: hits[cid] += 1
        # I
        for k_t, k_v in extract_landmark_address_keys(c, raw_addr):
            for cid in idx_i.get((k_t, k_v), [])[:cap_per_channel // 2]: hits[cid] += 1
        # J
        for k_t, k_v in extract_transliterated_keys(c, raw_name):
            for cid in idx_j.get((k_t, k_v), [])[:cap_per_channel // 2]: hits[cid] += 1
        # K
        for k_t, k_v in extract_typo_tolerant_keys(c, raw_name, raw_addr):
            for cid in idx_k.get((k_t, k_v), [])[:cap_per_channel // 2]: hits[cid] += 1
        # L
        for k_t, k_v in extract_acronym_keys(c, raw_name, raw_addr):
            for cid in idx_l.get((k_t, k_v), [])[:cap_per_channel // 2]: hits[cid] += 1

        return hits

    # 6. Retrieve Candidate Hits for Train and Val
    print(f"[{time.strftime('%H:%M:%S')}] Retrieving candidates across channels A...L...")
    train_cands_hits: Dict[str, Dict[str, int]] = {}
    val_cands_hits: Dict[str, Dict[str, int]] = {}
    needed_cand_ids = set()

    for s1_id in eval_train_s1:
        e1 = s1_records[s1_id]
        h = retrieve_candidates_with_hits(e1, e1.clean_name, e1.clean_addr)
        train_cands_hits[s1_id] = h
        needed_cand_ids.update(h.keys())
        needed_cand_ids.update(truth.get(s1_id, set()))

    for s1_id in eval_val_s1:
        e1 = s1_records[s1_id]
        h = retrieve_candidates_with_hits(e1, e1.clean_name, e1.clean_addr)
        val_cands_hits[s1_id] = h
        needed_cand_ids.update(h.keys())

    print(f"[{time.strftime('%H:%M:%S')}] Instantiating {len(needed_cand_ids):,} candidate EntityRecords...")
    cand_records_cache: Dict[str, EntityRecord] = {}
    for cid in needed_cand_ids:
        if cid in raw_cand_store:
            b_n, b_a, c = raw_cand_store[cid]
            cand_records_cache[cid] = EntityRecord(
                cid, b_n, normalize_clean(b_n), normalize_legal_name(b_n),
                b_a, normalize_clean(b_a), c
            )

    # 7. Build Training Dataset with Stratified Hard Negatives
    print(f"[{time.strftime('%H:%M:%S')}] Extracting features for training pairs...")
    X_train_list = []
    y_train_list = []

    for s1_id in eval_train_s1:
        e1 = s1_records[s1_id]
        h_dict = train_cands_hits[s1_id]
        true_set = truth.get(s1_id, set())
        is_singleton = (len(true_set) == 0)

        cands = set(h_dict.keys())
        pos_cands = cands & true_set
        neg_cands = cands - true_set

        # True Positives
        for cid in pos_cands:
            if cid in cand_records_cache:
                feats = extract_pair_features_fast(e1, cand_records_cache[cid], h_dict.get(cid, 1))
                X_train_list.append(feats)
                y_train_list.append(1)

        # Hard Negatives
        if is_singleton:
            # For singletons, sample 8 hard negatives so model learns to suppress false positives!
            n_neg = min(8, len(neg_cands))
        else:
            # For non-singletons, sample up to 5 negatives per positive
            n_neg = min(max(3, len(pos_cands) * 5), len(neg_cands))

        if n_neg > 0:
            # Sort negatives by channel hits (hardest first), then sample
            sorted_negs = sorted(list(neg_cands), key=lambda x: h_dict.get(x, 1), reverse=True)
            chosen_negs = sorted_negs[:n_neg]
            for cid in chosen_negs:
                if cid in cand_records_cache:
                    feats = extract_pair_features_fast(e1, cand_records_cache[cid], h_dict.get(cid, 1))
                    X_train_list.append(feats)
                    y_train_list.append(0)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.int32)
    print(f"  Training matrix: {len(y_train):,} pairs ({int(sum(y_train)):,} positives, {len(y_train) - int(sum(y_train)):,} hard negatives).")

    # 8. Train LightGBM Pairwise Matcher
    print(f"[{time.strftime('%H:%M:%S')}] Training LightGBM matcher...")
    lgb_train = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.08,
        'num_leaves': 63,
        'max_depth': 8,
        'feature_fraction': 0.85,
        'min_data_in_leaf': 40,
        'n_jobs': -1,
        'verbose': -1,
        'random_state': 42
    }
    model = lgb.train(params, lgb_train, num_boost_round=250)
    print(f"[{time.strftime('%H:%M:%S')}] LightGBM model trained.")

    # 9. Extract Validation Pair Features and Run Model Inference
    print(f"[{time.strftime('%H:%M:%S')}] Scoring validation candidates...")
    val_scored_candidates: Dict[str, List[Tuple[str, float]]] = {}
    total_val_pairs = 0

    for s1_id in eval_val_s1:
        e1 = s1_records[s1_id]
        h_dict = val_cands_hits[s1_id]
        cands = [c for c in h_dict.keys() if c in cand_records_cache]
        if not cands:
            val_scored_candidates[s1_id] = []
            continue

        pair_feats = [extract_pair_features_fast(e1, cand_records_cache[c], h_dict[c]) for c in cands]
        scores = model.predict(np.array(pair_feats, dtype=np.float32))
        scored = list(zip(cands, scores))
        # Sort candidates descending by score
        scored.sort(key=lambda x: x[1], reverse=True)
        val_scored_candidates[s1_id] = scored
        total_val_pairs += len(cands)

    print(f"  Scored {total_val_pairs:,} validation candidate pairs across {len(eval_val_s1):,} S1 entities.")

    # 10. Entity-Level Null-Margin Grid Search
    print(f"\n[{time.strftime('%H:%M:%S')}] Tuning Entity-Level Decision Calibration...")
    print("-" * 105)
    print(f"{'Threshold (tau)':<16} | {'Null Margin (tau_null)':<22} | {'Multi Delta':<12} | {'Macro F0.5':<12} | {'Precision':<10} | {'Recall':<10} | {'Singleton Acc':<14}")
    print("-" * 105)

    best_f05 = -1.0
    best_config = None
    best_preds: Dict[str, Set[str]] = {}

    tau_candidates = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    margin_deltas = [0.0, 0.05, 0.10, 0.15]
    multi_deltas = [0.10, 0.15, 0.20, 1.0]

    for tau in tau_candidates:
        for m_delta in margin_deltas:
            tau_null = tau + m_delta
            for m_multi in multi_deltas:
                current_preds: Dict[str, Set[str]] = {}
                for s1_id in eval_val_s1:
                    scored = val_scored_candidates[s1_id]
                    if not scored:
                        current_preds[s1_id] = set()
                        continue

                    top_cand, top_score = scored[0]
                    # Null margin gate: if top candidate cannot clear tau_null, predict empty set!
                    if top_score < tau_null:
                        current_preds[s1_id] = set()
                        continue

                    # Multi-match inclusion: score >= tau and score >= top_score - m_multi
                    matched = set()
                    for cid, s in scored:
                        if s >= tau and s >= (top_score - m_multi):
                            matched.add(cid)
                        else:
                            break # Since scored is sorted descending, we can early stop!
                    current_preds[s1_id] = matched

                # Evaluate official competition metric
                m = evaluate_predictions(truth, current_preds, eval_val_s1)
                f05 = m['macro_f05']
                p = m['pooled_precision']
                r = m['pooled_recall']
                sing_acc = m.get('singleton_accuracy', 0.0)

                if f05 > best_f05:
                    best_f05 = f05
                    best_config = (tau, tau_null, m_multi, f05, p, r, sing_acc)
                    best_preds = current_preds
                    print(f"** {tau:<13.2f} | {tau_null:<22.2f} | {m_multi:<12.2f} | {f05:<12.4f} | {p:<10.4f} | {r:<10.4f} | {sing_acc:<14.4f} ** (NEW BEST)")

    print("-" * 105)
    print(f"\n[{time.strftime('%H:%M:%S')}] OPTIMAL CALIBRATION FOUND:")
    print(f"  tau = {best_config[0]:.2f}")
    print(f"  tau_null = {best_config[1]:.2f} (Null Margin gate = +{best_config[1]-best_config[0]:.2f})")
    print(f"  delta_multi = {best_config[2]:.2f}")
    print(f"  Official Macro F0.5 = {best_config[3]:.4f}")
    print(f"  Precision = {best_config[4]:.4f}")
    print(f"  Recall = {best_config[5]:.4f}")
    print(f"  Singleton Accuracy = {best_config[6]:.4f}")

    # 11. Feature Importances
    importances = model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    print("\nTOP 15 FEATURE IMPORTANCES (Gain):")
    for fname, gain in feat_imp[:15]:
        print(f"  {fname:<28}: {gain:,.1f}")

    # 12. Save Report & Artifacts
    report_path = os.path.join(output_dir, "EXP009_MATCHER_CALIBRATION_REPORT.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# EXP-009: Pairwise LightGBM Matcher over A...L Candidates with Entity Null Margin Calibration\n\n")
        f.write(f"- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Train Entities: {train_sample_s1:,} | Validation Entities: {val_sample_s1:,}\n")
        f.write(f"- Optimal tau: {best_config[0]:.2f}\n")
        f.write(f"- Optimal tau_null: {best_config[1]:.2f}\n")
        f.write(f"- Optimal delta_multi: {best_config[2]:.2f}\n")
        f.write(f"- Validation Macro F0.5: **{best_config[3]:.4f}**\n")
        f.write(f"- Validation Precision: **{best_config[4]:.4f}**\n")
        f.write(f"- Validation Recall: **{best_config[5]:.4f}**\n")
        f.write(f"- Singleton Accuracy: **{best_config[6]:.4f}**\n\n")
        f.write("## Feature Importances (Gain)\n")
        for fname, gain in feat_imp:
            f.write(f"- `{fname}`: {gain:,.1f}\n")

    print(f"\n[{time.strftime('%H:%M:%S')}] Saved calibration report to {report_path}.")
    print(f"Total EXP-009 elapsed time: {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="student_resource/dataset")
    parser.add_argument("--output-dir", default="outputs/exp009_matcher")
    parser.add_argument("--split-dir", default="outputs/split")
    parser.add_argument("--train-sample-s1", type=int, default=30000)
    parser.add_argument("--val-sample-s1", type=int, default=10000)
    args = parser.parse_args()

    run_training_and_evaluation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        split_dir=args.split_dir,
        train_sample_s1=args.train_sample_s1,
        val_sample_s1=args.val_sample_s1,
    )
