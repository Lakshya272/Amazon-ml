#!/usr/bin/env python3
"""
src/normalize.py — Amazon ML Challenge 2026 Business Entity Resolution
Multi-Level Text Normalization Library

Provides clear, modular, and measurable normalization layers:
  - raw: Minimal whitespace trimming
  - base (Level 1): Unicode NFKC, lowercase, whitespace collapse
  - clean (Level 2): Base + punctuation/symbol replacement with spaces
  - legal_norm (Level 3): Clean + domain-specific canonicalization of common legal suffix
    variants (e.g. Pvt -> Private, Ltd -> Limited, Corp -> Corporation, Inc -> Incorporated, Co -> Company)
    and address abbreviations (e.g. Rd -> Road, St -> Street, Ave -> Avenue, Blvd -> Boulevard)
"""

import re
import sys
import unicodedata
from typing import Optional

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

_WHITESPACE_RE = re.compile(r'\s+')
_NON_ALPHANUM_RE = re.compile(r'[^\w\s]', re.UNICODE)

# Indic numerals translation to canonical ASCII 0-9
_INDIC_DIGITS = (
    "०१२३४५६७८९"  # Devanagari
    "০১২৩৪৫৬৭৮৯"  # Bengali
    "੦੧੨੩੪੫੬੭੮੯"  # Gurmukhi
    "૦૧૨૩૪૫૬૭૮૯"  # Gujarati
    "୦୧୨୩૪୫୬୭୮୯"  # Oriya
    "௦௧௨௩௪௫௬௭௮௯"  # Tamil
    "౦౧౨౩౪౫౬౭౮౯"  # Telugu
    "೦೧೨೩೪೫೬೭೮೯"  # Kannada
    "൦൧൨൩൪൫൬൭൮൯"  # Malayalam
)
_ASCII_DIGITS = "0123456789" * 9
_DIGIT_TRANSLATE_TABLE = str.maketrans(_INDIC_DIGITS, _ASCII_DIGITS)

# Compile regexes with word boundaries for legal suffixes
_LEGAL_SUFFIX_REPLACEMENTS = [
    # Multi-token phrases first
    (re.compile(r'\b(pvt\s+ltd|pvt\.?\s*ltd\.?)\b', re.IGNORECASE), 'private limited'),
    (re.compile(r'\b(co\s+ltd|co\.?\s*ltd\.?)\b', re.IGNORECASE), 'company limited'),
    (re.compile(r'\b(pvt|pvt\.)\b', re.IGNORECASE), 'private'),
    (re.compile(r'\b(ltd|ltd\.)\b', re.IGNORECASE), 'limited'),
    (re.compile(r'\b(corp|corp\.)\b', re.IGNORECASE), 'corporation'),
    (re.compile(r'\b(inc|inc\.)\b', re.IGNORECASE), 'incorporated'),
    (re.compile(r'\b(co|co\.)\b', re.IGNORECASE), 'company'),
    (re.compile(r'\b(llc|l\.l\.c\.)\b', re.IGNORECASE), 'llc'),
    (re.compile(r'\b(sarl|s\.a\.r\.l\.)\b', re.IGNORECASE), 'sarl'),
    (re.compile(r'\b(sas|s\.a\.s\.)\b', re.IGNORECASE), 'sas'),
    (re.compile(r'\b(gmbh|g\.m\.b\.h\.)\b', re.IGNORECASE), 'gmbh'),
]

# Compile regexes with word boundaries for address abbreviations
_ADDRESS_REPLACEMENTS = [
    (re.compile(r'\b(rd|rd\.)\b', re.IGNORECASE), 'road'),
    (re.compile(r'\b(st|st\.)\b', re.IGNORECASE), 'street'),
    (re.compile(r'\b(ave|ave\.)\b', re.IGNORECASE), 'avenue'),
    (re.compile(r'\b(blvd|blvd\.)\b', re.IGNORECASE), 'boulevard'),
    (re.compile(r'\b(dr|dr\.)\b', re.IGNORECASE), 'drive'),
    (re.compile(r'\b(hwy|hwy\.)\b', re.IGNORECASE), 'highway'),
    (re.compile(r'\b(ste|ste\.)\b', re.IGNORECASE), 'suite'),
    (re.compile(r'\b(apt|apt\.)\b', re.IGNORECASE), 'apartment'),
    (re.compile(r'\b(fl|fl\.)\b', re.IGNORECASE), 'floor'),
    (re.compile(r'\b(bldg|bldg\.)\b', re.IGNORECASE), 'building'),
]


def normalize_raw(text: Optional[str]) -> str:
    """Level 0: Raw whitespace strip."""
    if not text:
        return ""
    return text.strip()


def normalize_base(text: Optional[str]) -> str:
    """
    Level 1: Minimal canonical base normalization.
    - Unicode NFKC canonical decomposition + composition
    - Lowercase
    - Collapse consecutive whitespace characters to a single space, strip ends
    - Preserves all native scripts and accents
    """
    if not text:
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    return _WHITESPACE_RE.sub(' ', text).strip()


def normalize_clean(text: Optional[str]) -> str:
    """
    Level 2: Alphanumeric clean normalization.
    - Applies Level 1 base normalization
    - Translates Indic numerals to ASCII 0-9
    - Replaces symbols and punctuation with spaces
    - Re-collapses whitespace
    """
    if not text:
        return ""
    text = normalize_base(text)
    # Translate Indic numerals to ASCII 0-9
    text = text.translate(_DIGIT_TRANSLATE_TABLE)
    # Replace '&' with 'and' before stripping symbols
    text = text.replace('&', ' and ')
    text = _NON_ALPHANUM_RE.sub(' ', text)
    return _WHITESPACE_RE.sub(' ', text).strip()


def normalize_legal_name(text: Optional[str]) -> str:
    """
    Level 3: Domain-specific legal suffix canonicalization for business names.
    - Applies Level 2 clean normalization
    - Replaces common legal suffix variants with canonical expanded forms:
      pvt -> private, ltd -> limited, corp -> corporation, inc -> incorporated, co -> company
    - Preserves the token in its standardized position (does NOT drop tokens)
    """
    if not text:
        return ""
    text = normalize_clean(text)
    for pattern, replacement in _LEGAL_SUFFIX_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return _WHITESPACE_RE.sub(' ', text).strip()


def normalize_address(text: Optional[str], canonicalize_abbreviations: bool = False) -> str:
    """
    Address normalization.
    - If canonicalize_abbreviations is False: applies Level 2 clean normalization
    - If canonicalize_abbreviations is True: expands common road/street/building abbreviations
    """
    if not text:
        return ""
    text = normalize_clean(text)
    if canonicalize_abbreviations:
        for pattern, replacement in _ADDRESS_REPLACEMENTS:
            text = pattern.sub(replacement, text)
        text = _WHITESPACE_RE.sub(' ', text).strip()
    return text


def run_unit_tests():
    """Verify normalization transformations and preservation of signals."""
    print("Running normalize.py unit tests...")

    # Test 1: Base normalization preserves non-Latin and accents
    assert normalize_base("  Héllo   Wörld  ") == "héllo wörld"
    assert normalize_base("मॉडर्न   फाइनेंस") == "मॉडर्न फाइनेंस"
    print("  [PASS] Test 1 passed (Unicode and accent preservation)")

    # Test 2: Clean normalization handles punctuation and '&'
    assert normalize_clean("B&B Retail, Inc.") == "b and b retail inc"
    assert normalize_clean("<< Team Ecole >>") == "team ecole"
    print("  [PASS] Test 2 passed (Punctuation and symbol cleaning)")

    # Test 3: Legal suffix canonicalization
    assert normalize_legal_name("XYZ Pvt Ltd") == "xyz private limited"
    assert normalize_legal_name("XYZ Private Limited") == "xyz private limited"
    assert normalize_legal_name("Acme Robotics Inc.") == "acme robotics incorporated"
    assert normalize_legal_name("Acme Robotics Incorporated") == "acme robotics incorporated"
    assert normalize_legal_name("Apex Corp") == "apex corporation"
    assert normalize_legal_name("Apex Corporation") == "apex corporation"
    assert normalize_legal_name("Delta Co.") == "delta company"
    print("  [PASS] Test 3 passed (Legal suffix canonicalization)")

    # Test 4: False positive protection (doesn't mangle substrings)
    assert normalize_legal_name("Income Tax Solutions") == "income tax solutions"
    assert normalize_legal_name("Corporate Headquarters") == "corporate headquarters"
    assert normalize_legal_name("Limited Edition") == "limited edition"
    print("  [PASS] Test 4 passed (Word boundary protection for substrings)")

    # Test 5: Address canonicalization
    assert normalize_address("123 Main St., Apt 4B", canonicalize_abbreviations=True) == "123 main street apartment 4b"
    assert normalize_address("456 Park Rd.", canonicalize_abbreviations=True) == "456 park road"
    print("  [PASS] Test 5 passed (Address abbreviation expansion)")

    print("All normalization unit tests passed successfully!")


if __name__ == "__main__":
    run_unit_tests()
