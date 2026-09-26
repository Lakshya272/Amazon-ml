#!/usr/bin/env python3
"""
src/features.py — Pairwise Feature Extraction for Business Entity Resolution
Computes interpretable, fast similarity signals across name and address pairs.
Designed for high throughput: rapidfuzz C++ acceleration + vectorized representations.
"""

import math
import re
from typing import List, Tuple, Dict, Set, Optional

try:
    from rapidfuzz import fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

from transliterate import transliterate_indic_to_latin


_DIGIT_SEQ_RE = re.compile(r'\b\d+\b')
_POSTAL_RE = re.compile(r'\b\d{5,6}\b')
_LANDMARK_RE = re.compile(r'\b(?:near|opp|opposite|behind|beside|adj|adjacent|next\s+to|opp\.)\s+([a-zA-Z0-9]+)', re.IGNORECASE)
_STOP_WORDS = {
    'and', 'the', 'of', 'in', 'at', 'on', 'for', 'with', 'to', 'a', 'an',
    'private', 'limited', 'corporation', 'incorporated', 'company', 'llc', 'sarl', 'sas', 'gmbh',
    'pvt', 'ltd', 'corp', 'inc', 'co', 'services', 'enterprises', 'trading', 'solutions', 'associates'
}


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


def squeeze_repeats(s: str) -> str:
    return re.sub(r'(.)\1+', r'\1', s)


def get_acronym(words: List[str]) -> str:
    return "".join(w[0] for w in words if w and w[0].isalnum()).lower()


FEATURE_NAMES = [
    # Name features
    "name_clean_exact",
    "name_legal_exact",
    "name_fuzz_ratio",
    "name_token_sort_ratio",
    "name_token_jaccard",
    "name_token_dice",
    "name_token_overlap",
    "name_c3_jaccard",
    "name_c4_jaccard",
    "name_len_diff_ratio",
    "name_prefix_ratio",
    "name_translit_c3",
    "name_translit_exact",
    "name_acronym_match",
    "name_squeezed_c3",
    # Address features
    "addr_clean_exact",
    "addr_fuzz_ratio",
    "addr_token_sort_ratio",
    "addr_token_jaccard",
    "addr_token_dice",
    "addr_token_overlap",
    "addr_c3_jaccard",
    "addr_len_diff_ratio",
    # Postal / Building number / Locality features
    "postal_exact_match",
    "postal_prefix3_match",
    "digits_overlap_ratio",
    "numeric_post_match",
    "landmark_match",
    # Joint & Interaction features
    "name_x_addr_jaccard",
    "name_c3_x_addr_c3",
    # Source & Channel signals
    "cand_is_s2",
    "channel_hits",
]


class EntityRecord:
    """Pre-computed entity representation for ultra-fast pair comparison."""
    __slots__ = (
        'eid', 'country', 'clean_name', 'legal_name', 'name_toks', 'name_words',
        'translit_name', 'sqz_name', 'acronym',
        'clean_addr', 'addr_toks', 'postals', 'postal_p3', 'digits', 'landmarks', 'is_s2'
    )

    def __init__(self, eid: str, raw_name: str, clean_name: str, legal_name: str,
                 raw_addr: str, clean_addr: str, country: str):
        self.eid = eid
        self.country = country
        self.clean_name = clean_name
        self.legal_name = legal_name
        
        words = [w for w in clean_name.split() if w not in _STOP_WORDS]
        self.name_words = words
        self.name_toks = set(words)
        
        # Transliteration & phonetics
        self.translit_name = transliterate_indic_to_latin(clean_name)
        self.sqz_name = squeeze_repeats(clean_name)
        self.acronym = get_acronym(words) if len(words) >= 2 else (words[0] if (len(words) == 1 and len(words[0]) <= 5) else "")

        # Address fields
        self.clean_addr = clean_addr
        addr_words = [w for w in clean_addr.split() if w not in _STOP_WORDS and not w.isdigit()]
        self.addr_toks = set(addr_words)
        
        postals = set(_POSTAL_RE.findall(clean_addr))
        self.postals = postals
        self.postal_p3 = {p[:3] for p in postals if len(p) >= 3}
        self.digits = set(_DIGIT_SEQ_RE.findall(clean_addr))
        
        lms = _LANDMARK_RE.findall(raw_addr)
        self.landmarks = {lm.lower().strip() for lm in lms if len(lm.strip()) >= 3}
        
        self.is_s2 = 1.0 if eid.startswith("s2_") or "_source2" in eid else 0.0


def extract_pair_features_fast(
    e1: EntityRecord,
    e2: EntityRecord,
    channel_hits: int = 1
) -> List[float]:
    """Extract full feature vector for (S1, Candidate) pair."""
    # 1. Name exact matches
    name_clean_exact = 1.0 if (e1.clean_name and e1.clean_name == e2.clean_name) else 0.0
    name_legal_exact = 1.0 if (e1.legal_name and e1.legal_name == e2.legal_name) else 0.0

    # 2. String fuzz ratios
    if _HAS_RAPIDFUZZ:
        name_fuzz_ratio = fuzz.ratio(e1.clean_name, e2.clean_name) / 100.0
        name_token_sort_ratio = fuzz.token_sort_ratio(e1.clean_name, e2.clean_name) / 100.0
    else:
        name_fuzz_ratio = char_ngram_jaccard(e1.clean_name, e2.clean_name, 3)
        name_token_sort_ratio = jaccard_similarity(e1.name_toks, e2.name_toks)

    # 3. Token metrics
    name_token_jaccard = jaccard_similarity(e1.name_toks, e2.name_toks)
    name_token_dice = dice_similarity(e1.name_toks, e2.name_toks)
    name_token_overlap = overlap_coefficient(e1.name_toks, e2.name_toks)

    # 4. Character n-grams
    name_c3 = char_ngram_jaccard(e1.clean_name, e2.clean_name, 3)
    name_c4 = char_ngram_jaccard(e1.clean_name, e2.clean_name, 4)

    len_sum = len(e1.clean_name) + len(e2.clean_name)
    name_len_diff = abs(len(e1.clean_name) - len(e2.clean_name)) / len_sum if len_sum > 0 else 0.0
    prefix_len = prefix_match_length(e1.clean_name, e2.clean_name)
    max_len = max(len(e1.clean_name), len(e2.clean_name))
    prefix_ratio = prefix_len / max_len if max_len > 0 else 0.0

    # 5. Multilingual / Phonetic / Acronym / Typo features
    name_translit_c3 = char_ngram_jaccard(e1.translit_name, e2.translit_name, 3)
    name_translit_exact = 1.0 if (e1.translit_name and e1.translit_name == e2.translit_name) else 0.0
    
    # Acronym match: either e1 acronym equals e2, or e2 acronym equals e1
    acr_match = 0.0
    if e1.acronym and e2.acronym:
        if e1.acronym == e2.acronym or e1.acronym == e2.clean_name or e2.acronym == e1.clean_name:
            acr_match = 1.0
    name_squeezed_c3 = char_ngram_jaccard(e1.sqz_name, e2.sqz_name, 3)

    # 6. Address metrics
    addr_clean_exact = 1.0 if (e1.clean_addr and e1.clean_addr == e2.clean_addr) else 0.0
    if _HAS_RAPIDFUZZ:
        addr_fuzz_ratio = fuzz.ratio(e1.clean_addr, e2.clean_addr) / 100.0
        addr_token_sort_ratio = fuzz.token_sort_ratio(e1.clean_addr, e2.clean_addr) / 100.0
    else:
        addr_fuzz_ratio = char_ngram_jaccard(e1.clean_addr, e2.clean_addr, 3)
        addr_token_sort_ratio = jaccard_similarity(e1.addr_toks, e2.addr_toks)

    addr_jaccard = jaccard_similarity(e1.addr_toks, e2.addr_toks)
    addr_dice = dice_similarity(e1.addr_toks, e2.addr_toks)
    addr_overlap = overlap_coefficient(e1.addr_toks, e2.addr_toks)
    addr_c3 = char_ngram_jaccard(e1.clean_addr, e2.clean_addr, 3)

    addr_len_sum = len(e1.clean_addr) + len(e2.clean_addr)
    addr_len_diff = abs(len(e1.clean_addr) - len(e2.clean_addr)) / addr_len_sum if addr_len_sum > 0 else 0.0

    # 7. Postal and Numeric alignment
    postal_exact = 1.0 if (e1.postals and e2.postals and bool(e1.postals & e2.postals)) else 0.0
    postal_p3_match = 1.0 if (e1.postal_p3 and e2.postal_p3 and bool(e1.postal_p3 & e2.postal_p3)) else 0.0
    digits_overlap = overlap_coefficient(e1.digits, e2.digits)

    # Numeric post match: shared building digits AND postal prefix
    num_post_match = 1.0 if (postal_p3_match and bool(e1.digits & e2.digits)) else 0.0
    landmark_match = 1.0 if (e1.landmarks and e2.landmarks and bool(e1.landmarks & e2.landmarks)) else 0.0

    # 8. Cross-field interaction
    name_x_addr = name_token_jaccard * addr_jaccard
    name_c3_x_addr_c3 = name_c3 * addr_c3

    return [
        name_clean_exact,
        name_legal_exact,
        name_fuzz_ratio,
        name_token_sort_ratio,
        name_token_jaccard,
        name_token_dice,
        name_token_overlap,
        name_c3,
        name_c4,
        name_len_diff,
        prefix_ratio,
        name_translit_c3,
        name_translit_exact,
        acr_match,
        name_squeezed_c3,
        addr_clean_exact,
        addr_fuzz_ratio,
        addr_token_sort_ratio,
        addr_jaccard,
        addr_dice,
        addr_overlap,
        addr_c3,
        addr_len_diff,
        postal_exact,
        postal_p3_match,
        digits_overlap,
        num_post_match,
        landmark_match,
        name_x_addr,
        name_c3_x_addr_c3,
        e2.is_s2,
        float(channel_hits),
    ]
