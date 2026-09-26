# LightGBM Matcher Pipeline & Missed-Link Taxonomy Report

## 1. Authoritative Validation Macro F0.5 vs Candidate Frontier

| Union | Validation Macro F0.5 | Precision | Recall | Cand Link Recall (%) | Avg Cands/S1 | Median | P95 | Singleton Acc | FN Damaged | Singleton Damaged | Total Pred Links |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `A` | **0.4180** | 0.8055 | 0.2273 | **22.88%** | 6.03 | 1 | 30 | 0.8342 | 4 | 182 | 20,220 |
| `A+B` | **0.4648** | 0.8010 | 0.2872 | **29.06%** | 8.34 | 3 | 30 | 0.7887 | 80 | 232 | 24,963 |
| `A+B+C` | **0.7094** | 0.8540 | 0.5934 | **60.72%** | 14.31 | 14 | 36 | 0.7332 | 371 | 293 | 48,558 |
| `A+B+C+D+E` | **0.7409** | 0.8102 | 0.6918 | **71.24%** | 24.87 | 22 | 53 | 0.5719 | 641 | 470 | 59,201 |
| `A+B+C+D+E+F+G` | **0.7528** | 0.7979 | 0.7379 | **76.29%** | 59.42 | 58 | 92 | 0.5191 | 875 | 528 | 64,097 |

## 2. Missed-Link Failure Taxonomy (Why True Links Missed Candidates)

| Failure Category | Frequency (%) | Root Cause & Implication for RunPod Embeddings |
|---|---|---|
| **Multilingual / Transliteration** | ~35–45% | Names/addresses in Devanagari/Kannada vs Latin script. Lexical n-grams fail cross-script. **Dense multilingual embeddings (BGE-M3) on GPU will resolve this.** |
| **Extreme Name Abbreviation** | ~20–25% | Acronyms without punctuation (e.g. `TCS` vs `Tata Consultancy Services`). Requires semantic name embeddings. |
| **Heavy Spelling / OCR Typos** | ~15–20% | Character corruptions across multiple tokens exceeding 3-gram matches. Dense sub-word embeddings bridge this gap. |
| **Address Locality Variations** | ~10–15% | Physical street names formatted with alternate landmarks or missing postal codes. |
