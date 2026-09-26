#!/usr/bin/env python3
"""
src/evaluate_blocking_system.py — Amazon ML Challenge 2026 Business Entity Resolution
Comprehensive Blocking & Candidate Generation Benchmark with Entity-Level Coverage

Fulfills:
1. Proper validation harness:
   - Link recall
   - Entity full-coverage rate (100% true links covered)
   - Entity zero-coverage rate (0% true links covered)
   - Coverage distribution: 0%, 1-49%, 50-99%, 100%
   - Candidate volume: total, avg, median, P95, P99, max
   - Breakdown by S2 vs S3, country (US vs India), and match-cardinality bucket (0, 1, 2, 3-4, 5+)
2. Diagnosis and fixes for Channel B and D:
   - Channel B (Informative Name Tokens): Multi-token inverted index with IDF filtering,
     no harsh DF cutoffs, preserving Unicode.
   - Channel D (Character N-grams): Discarding restrictive paired signatures in favor of
     selective char 3-gram and 4-gram hashing.
3. Better bucket channels:
   - A: Exact legal-normalized name
   - B: Informative name-token buckets (IDF-weighted)
   - C: Address-derived buckets (postal, street number + word)
   - D: Character n-gram retrieval (selective char 3-grams)
   - E: Drop-one-token / corrupted-key retrieval (for names >= 3 tokens, index prefixes and skip-grams)
   - F: Address component-drop retrieval (number-only, postal-only, street-word only)
   - G: Sorted-neighborhood / prefix key retrieval (blocking keys on first 8 chars of normalized name)
4. Nested Union Benchmark:
   - A
   - A+B
   - A+B+C
   - A+B+C+D
   - A+B+C+D+E
   - A+B+C+D+E+F
   - A+B+C+D+E+F+G
5. Embedding Interface:
   - Modular CandidateChannel interface so future RunPod GPU embeddings plug in as another channel.
"""

import argparse
import csv
import gc
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional

import numpy as np

from normalize import (
    normalize_clean,
    normalize_legal_name,
)
from metrics import compute_entity_metrics
from transliterate import transliterate_indic_to_latin

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
    """Channel F: Address component-drop keys."""
    keys = []
    if not addr:
        return keys
    clean_a = normalize_clean(addr)
    words = [w for w in clean_a.split() if w not in _STOP_WORDS and not w.isdigit() and len(w) >= 4]
    # Use 2nd or 3rd address word as fallback locality/street identifier
    if len(words) >= 3:
        keys.append((f"{country}_addr_w1", words[1]))
        keys.append((f"{country}_addr_w2", words[2]))
    elif len(words) >= 2:
        keys.append((f"{country}_addr_w1", words[1]))
    return keys


def extract_drop_one_name_keys(country: str, legal_name: str) -> List[Tuple[str, str]]:
    """Channel E: Drop-one-token name keys (for names with >= 3 tokens)."""
    toks = get_name_tokens(legal_name)
    keys = []
    if len(toks) >= 3:
        # Drop token 0, token 1, or token 2
        keys.append((f"{country}_drop0", f"{toks[1]}_{toks[2]}"))
        keys.append((f"{country}_drop1", f"{toks[0]}_{toks[2]}"))
        keys.append((f"{country}_drop2", f"{toks[0]}_{toks[1]}"))
    elif len(toks) == 2:
        keys.append((f"{country}_pair", f"{toks[0]}_{toks[1]}"))
    return keys


def extract_sorted_neighborhood_key(country: str, legal_name: str) -> Optional[Tuple[str, str]]:
    """Channel G: Sorted neighborhood / prefix blocking key (first 8 alphanumeric chars)."""
    comp = "".join(legal_name.split())
    if len(comp) >= 5:
        return (f"{country}_snk", comp[:8])
    return None


def extract_numeric_address_keys(country: str, addr: str) -> List[Tuple[str, str]]:
    """Channel H: Numeric address keys (Postal prefix + Building/House number)."""
    keys = []
    if not addr:
        return keys
    clean_a = normalize_clean(addr)
    postals = _POSTAL_RE.findall(clean_a)
    digits = [d for d in _DIGIT_SEQ_RE.findall(clean_a) if len(d) <= 5 and d != '0']
    
    if postals and digits:
        p = postals[0]
        # Pair house/unit number with 3-digit and 5-digit postal prefix
        for d in digits[:2]:
            if d != p:
                keys.append((f"{country}_num_post5", f"{d}_{p[:5]}"))
                if len(p) >= 3:
                    keys.append((f"{country}_num_post3", f"{d}_{p[:3]}"))
    elif len(digits) >= 2:
        # Pair first two numbers (e.g. plot/door number + sector/street number)
        keys.append((f"{country}_num_pair", f"{digits[0]}_{digits[1]}"))
    return keys


def extract_landmark_address_keys(country: str, addr: str) -> List[Tuple[str, str]]:
    """Channel I: Landmark / Relaxed address locality keys."""
    keys = []
    if not addr:
        return keys
    # 1. Regex landmark extraction
    matches = _LANDMARK_RE.findall(addr)
    for lm in matches:
        lm_clean = lm.lower().strip()
        if len(lm_clean) >= 4 and lm_clean not in _STOP_WORDS:
            keys.append((f"{country}_landmark", lm_clean))
            
    # 2. Relaxed address words (first and last significant words)
    clean_a = normalize_clean(addr)
    words = [w for w in clean_a.split() if w not in _STOP_WORDS and not w.isdigit() and len(w) >= 4]
    if len(words) >= 4:
        # First word + last word (e.g., Street name + City/Locality)
        keys.append((f"{country}_addr_edge", f"{words[0]}_{words[-1]}"))
        keys.append((f"{country}_addr_edge2", f"{words[1]}_{words[-1]}"))
    return keys


def extract_transliterated_keys(country: str, name: str) -> List[Tuple[str, str]]:
    """Channel J: Deterministic transliteration & phonetic romanization keys."""
    keys = []
    if not name:
        return keys
    
    # Transliterate Indic / non-Latin characters to Latin phonetics
    rom = transliterate_indic_to_latin(name)
    if rom:
        legal_rom = normalize_legal_name(rom)
        if legal_rom:
            # 1. Exact romanized legal name
            keys.append((f"{country}_rom_exact", legal_rom))
            # 2. Romanized prefix
            comp = "".join(legal_rom.split())
            if len(comp) >= 5:
                keys.append((f"{country}_rom_snk", comp[:8]))
            # 3. Romanized significant tokens
            toks = [t for t in legal_rom.split() if len(t) >= 4 and t not in _STOP_WORDS]
            if len(toks) >= 2:
                keys.append((f"{country}_rom_pair", f"{toks[0]}_{toks[1]}"))
            elif len(toks) == 1:
                keys.append((f"{country}_rom_word", toks[0]))
    return keys



def run_benchmark(
    data_dir: str,
    output_dir: str,
    max_s1: Optional[int] = None,
    max_candidates: Optional[int] = None,
    max_bucket_size: int = 100,
    cap_per_channel: int = 30,
):
    overall_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train", "train_source3.tsv")

    print(f"[{time.strftime('%H:%M:%S')}] Step 1: Loading full ground truth from {gt_path}...")
    truth: Dict[str, Set[str]] = {}
    total_true_links = 0
    with open(gt_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        count = 0
        for row in reader:
            if not row: continue
            s1_id = row[0].strip()
            matched = row[1].strip() if len(row) > 1 else ""
            if matched:
                matches = {x.strip() for x in matched.split(',') if x.strip()}
                truth[s1_id] = matches
                total_true_links += len(matches)
            else:
                truth[s1_id] = set()
            count += 1
            if max_s1 and count >= max_s1:
                break

    target_s1_ids = set(truth.keys())
    print(f"[{time.strftime('%H:%M:%S')}] Loaded {len(truth):,} S1 ground-truth entries with {total_true_links:,} links.")

    # -----------------------------------------------------------------
    # Step 2: Vocabulary Pre-scan for Selective Inverted Indexes
    # -----------------------------------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] Step 2: Scanning candidate vocabulary & document frequencies...")
    token_doc_freq = Counter()
    c3_doc_freq = Counter()
    total_cand_docs = 0

    for path in [s2_path, s3_path]:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)
            for row in reader:
                if len(row) < 4: continue
                total_cand_docs += 1
                name_clean = normalize_legal_name(row[1])
                for t in set(get_name_tokens(name_clean)):
                    token_doc_freq[t] += 1
                for ng3 in set(get_char_ngrams(name_clean, 3)):
                    c3_doc_freq[ng3] += 1
                if max_candidates and total_cand_docs >= max_candidates:
                    break
        if max_candidates and total_cand_docs >= max_candidates:
            break

    # Retention criteria for Channel B and Channel D
    valid_tokens = {t: math.log((total_cand_docs + 1) / (df + 1)) for t, df in token_doc_freq.items() if 2 <= df <= max_bucket_size}
    valid_c3 = {ng: math.log((total_cand_docs + 1) / (df + 1)) for ng, df in c3_doc_freq.items() if 3 <= df <= max_bucket_size}
    print(f"  Vocabulary built: {len(valid_tokens):,} selective words, {len(valid_c3):,} selective char 3-grams.")
    del token_doc_freq
    del c3_doc_freq
    gc.collect()

    # -----------------------------------------------------------------
    # Step 3: Build Candidate Inverted Indices (A, B, C, D, E, F, G)
    # -----------------------------------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] Step 3: Constructing candidate inverted indices...")
    t_idx_start = time.time()

    idx_a = defaultdict(list)  # Channel A: (country, legal_name)
    idx_b = defaultdict(list)  # Channel B: (country, token)
    idx_c = defaultdict(list)  # Channel C: (country_key, addr_val)
    idx_d = defaultdict(list)  # Channel D: (country, char_3gram)
    idx_e = defaultdict(list)  # Channel E: (country_drop_key, tokens)
    idx_f = defaultdict(list)  # Channel F: (country_addr_drop_key, word)
    idx_g = defaultdict(list)  # Channel G: (country_snk, prefix8)
    idx_h = defaultdict(list)  # Channel H: (country_num_key, val)
    idx_i = defaultdict(list)  # Channel I: (country_lm_key, val)
    idx_j = defaultdict(list)  # Channel J: (country_rom_key, val)

    total_indexed = 0
    candidate_source_map = {}  # EID -> 'S2' or 'S3'

    for path in [s2_path, s3_path]:
        s_tag = "S2" if "source2" in path else "S3"
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)
            for row in reader:
                if len(row) < 4: continue
                eid, b_name, b_addr, country = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
                total_indexed += 1
                candidate_source_map[eid] = s_tag

                legal_n = normalize_legal_name(b_name)

                # Channel A: Exact legal name
                if legal_n:
                    idx_a[(country, legal_n)].append(eid)

                # Channel B: Informative name tokens
                toks = [t for t in get_name_tokens(legal_n) if t in valid_tokens]
                toks.sort(key=lambda t: valid_tokens[t], reverse=True)
                for t in toks[:3]:
                    idx_b[(country, t)].append(eid)

                # Channel C: Address-derived keys
                addr_keys = extract_address_keys(country, b_addr)
                for k_type, k_val in addr_keys:
                    idx_c[(k_type, k_val)].append(eid)

                # Channel D: Character 3-grams
                c3s = [ng for ng in get_char_ngrams(legal_n, 3) if ng in valid_c3]
                c3s.sort(key=lambda ng: valid_c3[ng], reverse=True)
                for ng in c3s[:2]:
                    idx_d[(country, ng)].append(eid)

                # Channel E: Drop-one-token name keys
                drop_keys = extract_drop_one_name_keys(country, legal_n)
                for k_type, k_val in drop_keys:
                    idx_e[(k_type, k_val)].append(eid)

                # Channel F: Address component-drop keys
                f_keys = extract_component_drop_address_keys(country, b_addr)
                for k_type, k_val in f_keys:
                    idx_f[(k_type, k_val)].append(eid)

                # Channel G: Sorted-neighborhood / prefix key
                snk = extract_sorted_neighborhood_key(country, legal_n)
                if snk:
                    idx_g[snk].append(eid)

                # Channel H: Numeric address keys
                h_keys = extract_numeric_address_keys(country, b_addr)
                for k_type, k_val in h_keys:
                    idx_h[(k_type, k_val)].append(eid)

                # Channel I: Landmark / Relaxed address locality keys
                i_keys = extract_landmark_address_keys(country, b_addr)
                for k_type, k_val in i_keys:
                    idx_i[(k_type, k_val)].append(eid)

                # Channel J: Deterministic transliteration & phonetic romanization keys
                j_keys = extract_transliterated_keys(country, b_name)
                for k_type, k_val in j_keys:
                    idx_j[(k_type, k_val)].append(eid)

                if max_candidates and total_indexed >= max_candidates:
                    break
        print(f"  Indexed {s_tag} (total candidate records: {total_indexed:,})...")
        if max_candidates and total_indexed >= max_candidates:
            break

    print(f"[{time.strftime('%H:%M:%S')}] Finished building indices in {round(time.time() - t_idx_start, 2)}s.")

    # -----------------------------------------------------------------
    # Step 4: Stream S1 & Benchmark Nested Unions & Coverage Distributions
    # -----------------------------------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] Step 4: Streaming S1 & evaluating nested candidate unions...")

    union_names = [
        "A",
        "A+B",
        "A+B+C",
        "A+B+C+D",
        "A+B+C+D+E",
        "A+B+C+D+E+F",
        "A+B+C+D+E+F+G",
        "Channel_H_Standalone",
        "Channel_I_Standalone",
        "A-G+H",
        "A-G+H+I",
        "Channel_J_Standalone",
        "A-G+H+I+J",
    ]

    # Metrics container per union
    metrics = {
        u: {
            "recovered_links": 0,
            "total_candidate_pairs": 0,
            "candidate_counts": [],  # for quantiles
            # Entity-level coverage buckets
            "cov_0": 0,
            "cov_1_49": 0,
            "cov_50_99": 0,
            "cov_100": 0,
            "full_coverage_entities": 0,
            "zero_coverage_entities": 0,
            # Oracle Matcher Macro F0.5 tracking
            "sum_oracle_f05": 0.0,
            # Breakdown by Source
            "rec_s2": 0,
            "rec_s3": 0,
            "true_s2": 0,
            "true_s3": 0,
            # Breakdown by Country
            "rec_us": 0,
            "rec_in": 0,
            "true_us": 0,
            "true_in": 0,
            # Breakdown by Cardinality Bucket
            "card_stats": {
                "singleton": {"recovered": 0, "true": 0, "entities": 0},
                "1_match": {"recovered": 0, "true": 0, "entities": 0},
                "2_matches": {"recovered": 0, "true": 0, "entities": 0},
                "3_4_matches": {"recovered": 0, "true": 0, "entities": 0},
                "5_plus_matches": {"recovered": 0, "true": 0, "entities": 0},
            }
        } for u in union_names
    }

    s1_count = 0
    t_stream_start = time.time()

    with open(s1_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            if len(row) < 4: continue
            s1_id = row[0].strip()
            if s1_id not in target_s1_ids:
                continue

            s1_count += 1
            country = row[3].strip()
            legal_n = normalize_legal_name(row[1].strip())
            b_addr = row[2].strip()

            true_set = truth[s1_id]
            n_true = len(true_set)

            # Determine true match breakdown for this S1
            true_s2_ids = {x for x in true_set if candidate_source_map.get(x) == "S2"}
            true_s3_ids = {x for x in true_set if candidate_source_map.get(x) == "S3"}
            is_us = (country == "US")

            # Determine cardinality bucket
            if n_true == 0:
                card_bucket = "singleton"
            elif n_true == 1:
                card_bucket = "1_match"
            elif n_true == 2:
                card_bucket = "2_matches"
            elif n_true <= 4:
                card_bucket = "3_4_matches"
            else:
                card_bucket = "5_plus_matches"

            # 1. Channel A (Exact legal name)
            cand_a = set()
            if legal_n:
                m_a = idx_a.get((country, legal_n))
                if m_a: cand_a = set(m_a[:cap_per_channel])

            # 2. Channel B (Informative name tokens)
            cand_b = set()
            toks = [t for t in get_name_tokens(legal_n) if t in valid_tokens]
            toks.sort(key=lambda t: valid_tokens[t], reverse=True)
            for t in toks[:3]:
                m_b = idx_b.get((country, t))
                if m_b: cand_b.update(m_b[:cap_per_channel // 2])

            # 3. Channel C (Address-derived)
            cand_c = set()
            addr_keys = extract_address_keys(country, b_addr)
            for k_type, k_val in addr_keys:
                m_c = idx_c.get((k_type, k_val))
                if m_c: cand_c.update(m_c[:cap_per_channel // 2])

            # 4. Channel D (Character 3-grams)
            cand_d = set()
            c3s = [ng for ng in get_char_ngrams(legal_n, 3) if ng in valid_c3]
            c3s.sort(key=lambda ng: valid_c3[ng], reverse=True)
            for ng in c3s[:2]:
                m_d = idx_d.get((country, ng))
                if m_d: cand_d.update(m_d[:cap_per_channel // 2])

            # 5. Channel E (Drop-one-token name keys)
            cand_e = set()
            drop_keys = extract_drop_one_name_keys(country, legal_n)
            for k_type, k_val in drop_keys:
                m_e = idx_e.get((k_type, k_val))
                if m_e: cand_e.update(m_e[:cap_per_channel // 2])

            # 6. Channel F (Address component-drop keys)
            cand_f = set()
            f_keys = extract_component_drop_address_keys(country, b_addr)
            for k_type, k_val in f_keys:
                m_f = idx_f.get((k_type, k_val))
                if m_f: cand_f.update(m_f[:cap_per_channel // 2])

            # 7. Channel G (Sorted-neighborhood / prefix key)
            cand_g = set()
            snk = extract_sorted_neighborhood_key(country, legal_n)
            if snk:
                m_g = idx_g.get(snk)
                if m_g: cand_g.update(m_g[:cap_per_channel // 2])

            # 8. Channel H (Numeric address keys)
            cand_h = set()
            h_keys = extract_numeric_address_keys(country, b_addr)
            for k_type, k_val in h_keys:
                m_h = idx_h.get((k_type, k_val))
                if m_h: cand_h.update(m_h[:cap_per_channel // 2])

            # 9. Channel I (Landmark / Relaxed address locality keys)
            cand_i = set()
            i_keys = extract_landmark_address_keys(country, b_addr)
            for k_type, k_val in i_keys:
                m_i = idx_i.get((k_type, k_val))
                if m_i: cand_i.update(m_i[:cap_per_channel // 2])

            # 10. Channel J (Deterministic transliteration & phonetic romanization keys)
            cand_j = set()
            j_keys = extract_transliterated_keys(country, row[1].strip())
            for k_type, k_val in j_keys:
                m_j = idx_j.get((k_type, k_val))
                if m_j: cand_j.update(m_j[:cap_per_channel // 2])

            # Form Nested Unions
            u_a = cand_a
            u_ab = u_a | cand_b
            u_abc = u_ab | cand_c
            u_abcd = u_abc | cand_d
            u_abcde = u_abcd | cand_e
            u_abcdef = u_abcde | cand_f
            u_abcdefg = u_abcdef | cand_g
            u_ag_h = u_abcdefg | cand_h
            u_ag_h_i = u_ag_h | cand_i
            u_ag_h_i_j = u_ag_h_i | cand_j

            unions_dict = {
                "A": u_a,
                "A+B": u_ab,
                "A+B+C": u_abc,
                "A+B+C+D": u_abcd,
                "A+B+C+D+E": u_abcde,
                "A+B+C+D+E+F": u_abcdef,
                "A+B+C+D+E+F+G": u_abcdefg,
                "Channel_H_Standalone": cand_h,
                "Channel_I_Standalone": cand_i,
                "A-G+H": u_ag_h,
                "A-G+H+I": u_ag_h_i,
                "Channel_J_Standalone": cand_j,
                "A-G+H+I+J": u_ag_h_i_j,
            }

            # Evaluate each nested union
            for u_name, cand_set in unions_dict.items():
                m_dict = metrics[u_name]
                n_cands = len(cand_set)
                m_dict["total_candidate_pairs"] += n_cands
                if len(m_dict["candidate_counts"]) < 100000:
                    m_dict["candidate_counts"].append(n_cands)

                # True link recovery
                rec = cand_set & true_set
                n_rec = len(rec)
                m_dict["recovered_links"] += n_rec

                # Oracle Matcher: outputs (cand_set & true_set)
                # For singletons (n_true == 0): oracle outputs empty set -> f05 = 1.0
                # For non-singletons: oracle outputs rec. Since rec <= true_set, precision = 1.0, recall = n_rec / n_true
                oracle_f05, _, _, _, _, _ = compute_entity_metrics(true_set, rec)
                m_dict["sum_oracle_f05"] += oracle_f05

                # Coverage distribution
                if n_true == 0:
                    # Singletons: if no candidates or candidates don't matter, full credit
                    m_dict["cov_100"] += 1
                    m_dict["full_coverage_entities"] += 1
                else:
                    cov_pct = n_rec / n_true
                    if cov_pct == 1.0:
                        m_dict["cov_100"] += 1
                        m_dict["full_coverage_entities"] += 1
                    elif cov_pct >= 0.50:
                        m_dict["cov_50_99"] += 1
                    elif cov_pct > 0.0:
                        m_dict["cov_1_49"] += 1
                    else:
                        m_dict["cov_0"] += 1
                        m_dict["zero_coverage_entities"] += 1

                # Source & Country breakdown
                rec_s2 = len(rec & true_s2_ids)
                rec_s3 = len(rec & true_s3_ids)
                m_dict["rec_s2"] += rec_s2
                m_dict["rec_s3"] += rec_s3
                m_dict["true_s2"] += len(true_s2_ids)
                m_dict["true_s3"] += len(true_s3_ids)

                if is_us:
                    m_dict["rec_us"] += n_rec
                    m_dict["true_us"] += n_true
                else:
                    m_dict["rec_in"] += n_rec
                    m_dict["true_in"] += n_true

                # Cardinality breakdown
                c_stat = m_dict["card_stats"][card_bucket]
                c_stat["recovered"] += n_rec
                c_stat["true"] += n_true
                c_stat["entities"] += 1

            if s1_count % 500000 == 0:
                print(f"  Processed {s1_count:,} / {len(target_s1_ids):,} S1 entities ({round(s1_count/len(target_s1_ids)*100, 1)}%)...")

    print(f"[{time.strftime('%H:%M:%S')}] Finished S1 evaluation in {round(time.time() - t_stream_start, 2)}s.")

    # -----------------------------------------------------------------
    # Step 5: Assemble Benchmark Table & Report
    # -----------------------------------------------------------------
    benchmark_table = []
    print("\n" + "="*110)
    print("NESTED CANDIDATE UNION BENCHMARK REPORT WITH ENTITY COVERAGE")
    print("="*110)

    for u_name in union_names:
        m = metrics[u_name]
        cands_arr = np.array(m["candidate_counts"])
        link_recall = (m["recovered_links"] / total_true_links * 100) if total_true_links > 0 else 0.0
        full_cov_rate = (m["full_coverage_entities"] / s1_count * 100) if s1_count > 0 else 0.0
        zero_cov_rate = (m["zero_coverage_entities"] / s1_count * 100) if s1_count > 0 else 0.0

        cov_0_pct = (m["cov_0"] / s1_count * 100) if s1_count > 0 else 0.0
        cov_1_49_pct = (m["cov_1_49"] / s1_count * 100) if s1_count > 0 else 0.0
        cov_50_99_pct = (m["cov_50_99"] / s1_count * 100) if s1_count > 0 else 0.0
        cov_100_pct = (m["cov_100"] / s1_count * 100) if s1_count > 0 else 0.0

        rec_s2_pct = (m["rec_s2"] / m["true_s2"] * 100) if m["true_s2"] > 0 else 0.0
        rec_s3_pct = (m["rec_s3"] / m["true_s3"] * 100) if m["true_s3"] > 0 else 0.0
        rec_us_pct = (m["rec_us"] / m["true_us"] * 100) if m["true_us"] > 0 else 0.0
        rec_in_pct = (m["rec_in"] / m["true_in"] * 100) if m["true_in"] > 0 else 0.0

        avg_cands = round(m["total_candidate_pairs"] / s1_count, 2)
        med_cands = int(np.median(cands_arr))
        p95_cands = int(np.percentile(cands_arr, 95))
        p99_cands = int(np.percentile(cands_arr, 99))
        max_cands = int(np.max(cands_arr))

        oracle_macro_f05 = round(m["sum_oracle_f05"] / s1_count, 4) if s1_count > 0 else 0.0

        row = {
            "union": u_name,
            "link_recall": round(link_recall, 2),
            "oracle_macro_f05": oracle_macro_f05,
            "full_coverage_rate": round(full_cov_rate, 2),
            "zero_coverage_rate": round(zero_cov_rate, 2),
            "cov_0_pct": round(cov_0_pct, 2),
            "cov_1_49_pct": round(cov_1_49_pct, 2),
            "cov_50_99_pct": round(cov_50_99_pct, 2),
            "cov_100_pct": round(cov_100_pct, 2),
            "total_candidates": m["total_candidate_pairs"],
            "avg_cands_s1": avg_cands,
            "median_cands": med_cands,
            "p95_cands": p95_cands,
            "p99_cands": p99_cands,
            "max_cands": max_cands,
            "rec_s2": round(rec_s2_pct, 2),
            "rec_s3": round(rec_s3_pct, 2),
            "rec_us": round(rec_us_pct, 2),
            "rec_in": round(rec_in_pct, 2),
            "cardinality_recall": {
                k: round(v["recovered"] / v["true"] * 100, 2) if v["true"] > 0 else 100.0
                for k, v in m["card_stats"].items()
            }
        }
        benchmark_table.append(row)

        print(f"\nUnion: {u_name}")
        print(f"  Link Recall:        {row['link_recall']:.2f}%  |  Oracle Macro F0.5: {row['oracle_macro_f05']:.4f}")
        print(f"  Entity Coverage:    Full Coverage: {row['full_coverage_rate']:.2f}%  |  Zero Coverage: {row['zero_coverage_rate']:.2f}%")
        print(f"  Coverage Bins:      0%: {row['cov_0_pct']}%  |  1-49%: {row['cov_1_49_pct']}%  |  50-99%: {row['cov_50_99_pct']}%  |  100%: {row['cov_100_pct']}%")
        print(f"  Candidate Volume:   Total: {row['total_candidates']:,}  |  Avg/S1: {row['avg_cands_s1']}  |  Med: {row['median_cands']}  |  P95: {row['p95_cands']}  |  P99: {row['p99_cands']}  |  Max: {row['max_cands']}")
        print(f"  Sources & Country:  S2 Rec: {row['rec_s2']}%  |  S3 Rec: {row['rec_s3']}%  |  US Rec: {row['rec_us']}%  |  IN Rec: {row['rec_in']}%")
        print(f"  Cardinality Rec:    1-match: {row['cardinality_recall']['1_match']}%  |  2-matches: {row['cardinality_recall']['2_matches']}%  |  3-4: {row['cardinality_recall']['3_4_matches']}%  |  5+: {row['cardinality_recall']['5_plus_matches']}%")

    out_json = os.path.join(output_dir, "nested_union_benchmark.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(benchmark_table, f, indent=2)

    # Generate Markdown Table Report
    md_lines = [
        "# Nested Candidate Union Benchmark Report — Entity Coverage & Volume Frontier",
        f"\n**Evaluated on Full Ground Truth:** {s1_count:,} Source 1 Entities, {total_true_links:,} True Links",
        f"**Execution Runtime:** {round(time.time() - overall_start, 2)}s (~{round((time.time() - overall_start)/60, 1)} min)\n",
        "## 1. Candidate Union Frontier Table\n",
        "| Union | Link Recall (%) | Oracle Macro F0.5 | Full Cov (%) | Zero Cov (%) | Cov 0% | 1-49% | 50-99% | 100% | Total Cands | Avg/S1 | Med | P95 | P99 | Max | Rec S2 | Rec S3 | Rec US | Rec IN |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in benchmark_table:
        md_lines.append(
            f"| `{r['union']}` | **{r['link_recall']}%** | **{r['oracle_macro_f05']:.4f}** | {r['full_coverage_rate']}% | {r['zero_coverage_rate']}% | "
            f"{r['cov_0_pct']}% | {r['cov_1_49_pct']}% | {r['cov_50_99_pct']}% | {r['cov_100_pct']}% | "
            f"{r['total_candidates']:,} | {r['avg_cands_s1']} | {r['median_cands']} | {r['p95_cands']} | {r['p99_cands']} | {r['max_cands']} | "
            f"{r['rec_s2']}% | {r['rec_s3']}% | {r['rec_us']}% | {r['rec_in']}% |"
        )

    md_lines.append("\n## 2. Match Cardinality Link Recall Breakdown\n")
    md_lines.append("| Union | 1 Match (%) | 2 Matches (%) | 3–4 Matches (%) | 5+ Matches (%) |")
    md_lines.append("|---|---|---|---|---|")
    for r in benchmark_table:
        c = r["cardinality_recall"]
        md_lines.append(f"| `{r['union']}` | {c['1_match']}% | {c['2_matches']}% | {c['3_4_matches']}% | {c['5_plus_matches']}% |")

    out_md = os.path.join(output_dir, "BLOCKING_NESTED_UNION_REPORT.md")
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n[{time.strftime('%H:%M:%S')}] Benchmark report successfully saved to {out_md}.")
    return benchmark_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="student_resource/dataset")
    parser.add_argument("--output-dir", default="outputs/blocking")
    parser.add_argument("--max-s1", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--max-bucket-size", type=int, default=80)
    parser.add_argument("--cap-per-channel", type=int, default=30)
    args = parser.parse_args()

    run_benchmark(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_s1=args.max_s1 if args.max_s1 > 0 else None,
        max_candidates=args.max_candidates if args.max_candidates > 0 else None,
        max_bucket_size=args.max_bucket_size,
        cap_per_channel=args.cap_per_channel,
    )
