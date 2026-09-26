#!/usr/bin/env python3
"""
src/features.py — Pairwise Feature Extraction for Business Entity Resolution
Computes interpretable, fast similarity signals across name and address pairs.
Designed for high throughput: ~50,000-100,000 pairs/sec in pure Python/NumPy.
"""

import math
from typing import List, Tuple, Dict, Set

def jaccard_similarity(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0

def dice_similarity(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    denom = len(tokens_a) + len(tokens_b)
    return (2.0 * len(tokens_a & tokens_b)) / denom if denom > 0 else 0.0

def overlap_coefficient(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    denom = min(len(tokens_a), len(tokens_b))
    return len(tokens_a & tokens_b) / denom if denom > 0 else 0.0

def char_ngram_jaccard(str_a: str, str_b: str, n: int = 3) -> float:
    if not str_a or not str_b:
        return 0.0
    if str_a == str_b:
        return 1.0
    ng_a = {str_a[i:i+n] for i in range(len(str_a) - n + 1)} if len(str_a) >= n else {str_a}
    ng_b = {str_b[i:i+n] for i in range(len(str_b) - n + 1)} if len(str_b) >= n else {str_b}
    return jaccard_similarity(ng_a, ng_b)

def prefix_match_length(s1: str, s2: str) -> int:
    min_len = min(len(s1), len(s2))
    for i in range(min_len):
        if s1[i] != s2[i]:
            return i
    return min_len

FEATURE_NAMES = [
    # Name features
    "name_exact_match",
    "name_clean_exact",
    "name_token_jaccard",
    "name_token_dice",
    "name_token_overlap",
    "name_c3_jaccard",
    "name_c4_jaccard",
    "name_len_diff_ratio",
    "name_prefix_ratio",
    # Address features
    "addr_exact_match",
    "addr_token_jaccard",
    "addr_token_dice",
    "addr_token_overlap",
    "addr_c3_jaccard",
    "addr_len_diff_ratio",
    # Numeric/Postal address features
    "postal_exact_match",
    "digits_overlap_ratio",
    # Joint / Interaction features
    "name_x_addr_jaccard",
    "name_c3_x_addr_c3",
]

def extract_pair_features(
    s1_name_clean: str,
    s1_name_toks: Set[str],
    s1_addr_clean: str,
    s1_addr_toks: Set[str],
    s1_postals: Set[str],
    s1_digits: Set[str],
    cand_name_clean: str,
    cand_name_toks: Set[str],
    cand_addr_clean: str,
    cand_addr_toks: Set[str],
    cand_postals: Set[str],
    cand_digits: Set[str],
) -> List[float]:
    """Extract a dense numerical feature vector for an (S1, Candidate) pair."""
    # 1. Name exact match
    name_exact = 1.0 if s1_name_clean and s1_name_clean == cand_name_clean else 0.0
    
    # 2. Name token metrics
    name_jaccard = jaccard_similarity(s1_name_toks, cand_name_toks)
    name_dice = dice_similarity(s1_name_toks, cand_name_toks)
    name_overlap = overlap_coefficient(s1_name_toks, cand_name_toks)
    
    # 3. Name character n-grams
    name_c3 = char_ngram_jaccard(s1_name_clean, cand_name_clean, 3)
    name_c4 = char_ngram_jaccard(s1_name_clean, cand_name_clean, 4)
    
    len_sum = len(s1_name_clean) + len(cand_name_clean)
    name_len_diff = abs(len(s1_name_clean) - len(cand_name_clean)) / len_sum if len_sum > 0 else 0.0
    prefix_len = prefix_match_length(s1_name_clean, cand_name_clean)
    max_len = max(len(s1_name_clean), len(cand_name_clean))
    prefix_ratio = prefix_len / max_len if max_len > 0 else 0.0

    # 4. Address metrics
    addr_exact = 1.0 if s1_addr_clean and s1_addr_clean == cand_addr_clean else 0.0
    addr_jaccard = jaccard_similarity(s1_addr_toks, cand_addr_toks)
    addr_dice = dice_similarity(s1_addr_toks, cand_addr_toks)
    addr_overlap = overlap_coefficient(s1_addr_toks, cand_addr_toks)
    addr_c3 = char_ngram_jaccard(s1_addr_clean, cand_addr_clean, 3)
    
    addr_len_sum = len(s1_addr_clean) + len(cand_addr_clean)
    addr_len_diff = abs(len(s1_addr_clean) - len(cand_addr_clean)) / addr_len_sum if addr_len_sum > 0 else 0.0

    # 5. Postal and digits overlap
    postal_match = 1.0 if (s1_postals and cand_postals and bool(s1_postals & cand_postals)) else 0.0
    digits_overlap = overlap_coefficient(s1_digits, cand_digits)

    # 6. Interaction
    name_x_addr = name_jaccard * addr_jaccard
    name_c3_x_addr_c3 = name_c3 * addr_c3

    return [
        name_exact,
        name_exact,
        name_jaccard,
        name_dice,
        name_overlap,
        name_c3,
        name_c4,
        name_len_diff,
        prefix_ratio,
        addr_exact,
        addr_jaccard,
        addr_dice,
        addr_overlap,
        addr_c3,
        addr_len_diff,
        postal_match,
        digits_overlap,
        name_x_addr,
        name_c3_x_addr_c3,
    ]
