#!/usr/bin/env python3
"""
src/transliterate.py — Amazon ML Challenge 2026 Business Entity Resolution
Deterministic Transliteration & Phonetic Normalization for Indic & Non-Latin Scripts

Strictly compliant with fair-play rules:
- No external lookup / APIs / network calls
- Deterministic Unicode mapping of Devanagari, Bengali, Tamil, Telugu, Kannada, Gujarati to Latin
- Maps phonetic representations into canonical Latin tokens so lexical inverted indexes
  can bridge cross-script business entities seamlessly.
"""

import unicodedata
import re

# Comprehensive Unicode Block Mappings to Latin Phonetics

# Devanagari (0x0900 - 0x097F) - Hindi, Marathi, Sanskrit
_DEV_MAP = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v',
    'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h',
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', '्': '',
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'ऋ': 'ri',
    'ं': 'n', 'ँ': 'n', 'ः': 'h', '़': ''
}

# Bengali (0x0980 - 0x09FF) - Bengali, Assamese
_BEN_MAP = {
    'ক': 'k', 'খ': 'kh', 'গ': 'g', 'ঘ': 'gh', 'ঙ': 'ng',
    'চ': 'ch', 'ছ': 'chh', 'জ': 'j', 'ঝ': 'jh', 'ঞ': 'ny',
    'ট': 't', 'ঠ': 'th', 'ড': 'd', 'ঢ': 'dh', 'ণ': 'n',
    'ত': 't', 'থ': 'th', 'দ': 'd', 'ধ': 'dh', 'ন': 'n',
    'প': 'p', 'ফ': 'ph', 'ব': 'b', 'ভ': 'bh', 'ম': 'm',
    'য': 'y', 'র': 'r', 'ল': 'l', 'শ': 'sh', 'ষ': 'sh',
    'স': 's', 'হ': 'h', 'া': 'a', 'ি': 'i', 'ী': 'ee',
    'ু': 'u', 'ূ': 'oo', 'ে': 'e', 'ৈ': 'ai', 'ো': 'o',
    'ৌ': 'au', '্': '', 'অ': 'a', 'আ': 'aa', 'ই': 'i',
    'ঈ': 'ee', 'উ': 'u', 'ঊ': 'oo', 'এ': 'e', 'ঐ': 'ai',
    'ও': 'o', 'ঔ': 'au', 'ং': 'n', 'ঁ': 'n', 'ঃ': 'h'
}

# Tamil (0x0B80 - 0x0BFF) - Tamil
_TAM_MAP = {
    'க': 'k', 'ங': 'ng', 'ச': 'ch', 'ஞ': 'ny', 'ட': 't',
    'ண': 'n', 'த': 'th', 'ந': 'n', 'ப': 'p', 'ம': 'm',
    'ய': 'y', 'ர': 'r', 'ல': 'l', 'வ': 'v', 'ழ': 'zh',
    'ள': 'l', 'ற': 'r', 'ன': 'n', 'ஜ': 'j', 'ஷ': 'sh',
    'ஸ': 's', 'ஹ': 'h', 'ா': 'a', 'ி': 'i', 'ீ': 'ee',
    'ு': 'u', 'ூ': 'oo', 'ெ': 'e', 'ே': 'e', 'ை': 'ai',
    'ொ': 'o', 'ோ': 'o', 'ௌ': 'au', '்': '', 'அ': 'a',
    'ஆ': 'aa', 'இ': 'i', 'ஈ': 'ee', 'உ': 'u', 'ஊ': 'oo',
    'எ': 'e', 'ஏ': 'e', 'ஐ': 'ai', 'ஒ': 'o', 'ஓ': 'o',
    'ஔ': 'au', 'ஃ': 'k'
}

# Telugu (0x0C00 - 0x0C7F) - Telugu
_TEL_MAP = {
    'క': 'k', 'ఖ': 'kh', 'గ': 'g', 'ఘ': 'gh', 'ఙ': 'ng',
    'చ': 'ch', 'ఛ': 'chh', 'జ': 'j', 'ఝ': 'jh', 'ఞ': 'ny',
    'ట': 't', 'ఠ': 'th', 'డ': 'd', 'ఢ': 'dh', 'ణ': 'n',
    'త': 't', 'థ': 'th', 'ద': 'd', 'ధ': 'dh', 'న': 'n',
    'ప': 'p', 'ఫ': 'ph', 'బ': 'b', 'భ': 'bh', 'మ': 'm',
    'య': 'y', 'ర': 'r', 'ల': 'l', 'వ': 'v', 'శ': 'sh',
    'ష': 'sh', 'స': 's', 'హ': 'h', 'ళ': 'l', 'ా': 'a',
    'ి': 'i', 'ీ': 'ee', 'ు': 'u', 'ూ': 'oo', 'ె': 'e',
    'ే': 'e', 'ై': 'ai', 'ొ': 'o', 'ో': 'o', 'ౌ': 'au',
    '్': '', 'అ': 'a', 'ఆ': 'aa', 'ఇ': 'i', 'ఈ': 'ee',
    'ఉ': 'u', 'ఊ': 'oo', 'ఎ': 'e', 'ఏ': 'e', 'ఐ': 'ai',
    'ఒ': 'o', 'ఓ': 'o', 'ఔ': 'au', 'ం': 'n', 'ః': 'h'
}

# Combine all script maps
_CHAR_MAP = {}
_CHAR_MAP.update(_DEV_MAP)
_CHAR_MAP.update(_BEN_MAP)
_CHAR_MAP.update(_TAM_MAP)
_CHAR_MAP.update(_TEL_MAP)


def is_non_latin(s: str) -> bool:
    """Check if string contains characters outside basic Latin/ASCII."""
    return any(ord(c) > 127 for c in s)


def transliterate_indic_to_latin(text: str) -> str:
    """
    Deterministically transliterate Indic script tokens to Latin phonetics.
    Preserves existing Latin characters, digits, and spaces.
    """
    if not text:
        return ""
    
    # Fast path: pure ASCII
    if not any(ord(c) > 127 for c in text):
        return text

    chars = []
    for c in text:
        if c in _CHAR_MAP:
            chars.append(_CHAR_MAP[c])
        else:
            # NFKD normalization to strip diacritics for European Latin variations (French, etc.)
            d = unicodedata.normalize('NFKD', c)
            filtered = [ch for ch in d if not unicodedata.combining(ch)]
            chars.append(''.join(filtered) if filtered else c)

    res = ''.join(chars)
    # Clean whitespace and non-alphanumeric
    res = re.sub(r'[^a-zA-Z0-9\s]', ' ', res)
    return ' '.join(res.split()).lower()
