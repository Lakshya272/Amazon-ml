# Experiment Journey — Amazon ML Challenge 2026

## EXP-001: Baseline Reconnaissance & Character/Intersection Profiling
- **Commit:** `765772e`
- **Scope:** Full dataset audit (24.2M rows across Train and Test)
- **Key Findings:**
  - 0 cross-country matches across 7.64M ground truth links (100% same-country invariance).
  - Train: 60% US, 40% India. Test: 46.8% India, 38.3% US, 15.0% France.
  - Non-ASCII: Indian records are 18-29% Devanagari/Kannada script; French records are 6-15% accented Latin. Stripping non-ASCII destroys business identity on over 20M records.
  - Train/Test intersection: Only 15.7% of test reference names appear in train S1 (84.3% novel entities). Overfitting severely damages generalization.

---

## EXP-002: Exact Matching Baselines & Legal Suffix Normalization
- **Commit:** `0238a16`
- **Scope:** Full training ground-truth evaluation across 9 exact matching modes
- **Key Findings:**
  - `exact_clean_pair` / `exact_legal_pair`: **100.00% precision**, 0 false positives.
  - Legal suffix normalization (`pvt` $\to$ `private`, `ltd` $\to$ `limited`, `corp` $\to$ `corporation`, `inc` $\to$ `incorporated`) recovered **+208,094** additional true links with zero precision loss.
  - Exact name OR exact address recovered **27.83%** of true links (Macro F0.5 = 0.3781). Over 72% of true links require fuzzy blocking and feature matching.

---

## EXP-003: Phase 2 Independent Blocking Channels & Unions Benchmark
- **Commit:** `09cdc02`
- **Scope:** Full training ground-truth evaluation across 4 independent channels and 3 unions (2.2M S1 entities, 7.64M ground truth links)
- **Benchmark Results:**

| Strategy / Channel | Candidate Recall | Total Candidate Pairs | Avg Cands / S1 | Median | P95 | Max | Rec S2 | Rec S3 | Rec US | Rec IN |
|---|---|---|---|---|---|---|---|---|---|---|
| `Channel_A_ExactLegalName` | **23.56%** | 16,343,691 | 7.41 | 1 | 50 | 50 | 49.05% | 50.95% | 63.93% | 36.07% |
| `Channel_B_SharedNameTokens` | **11.95%** | 9,672,321 | 4.38 | 0 | 25 | 75 | 63.17% | 36.83% | 53.71% | 46.29% |
| `Channel_C_AddressDerived` | **46.10%** | 18,946,710 | 8.59 | 4 | 25 | 50 | 51.04% | 48.96% | 70.99% | 29.01% |
| `Channel_D_NgramSignature` | **0.08%** | 41,969 | 0.02 | 0 | 0 | 25 | 56.64% | 43.36% | 60.33% | 39.67% |
| `Union_A_B` | **31.90%** | 25,655,887 | 11.63 | 3 | 50 | 77 | 52.77% | 47.23% | 61.27% | 38.73% |
| `Union_A_B_C` | **63.23%** | 43,460,695 | 19.69 | 16 | 52 | 114 | 50.60% | 49.40% | 66.04% | 33.96% |
| `Union_A_B_C_D` | **63.24%** | 43,474,001 | 19.70 | 16 | 52 | 114 | 50.60% | 49.40% | 66.04% | 33.96% |

- **Key Insights:**
  - Address-derived blocking (`Channel_C`) is the single strongest recall contributor (**46.10% recall** at only 8.59 cands/S1), capturing entities where names were varied or shortened but physical addresses matched.
  - Combining `A + B + C` captures **63.23% recall** (4.83M true links) at an average of **19.69 candidates per S1**, with median = 16 and P95 = 52.
  - Channel D (paired n-grams) was too selective (0.08% recall); relax constraint to unigram character 3-grams with IDF weighting for further expansion.

---

## EXP-004: Advanced Blocking Benchmark (Char TF-IDF Top-K, Word TF-IDF, Relaxed N-grams)
- **Commit:** `5a4e847`
- **Scope:** Full training ground-truth evaluation across 14 configurations (2,206,821 S1 entities, 7,638,365 ground-truth links).
- **Execution Time:** 959s (~16.0 min), Peak RAM ~6.5 GiB.
- **Benchmark Results:**

| Strategy / Channel | Candidate Recall | Total Candidate Pairs | Avg Cands / S1 | Median | P95 | Max | Rec S2 | Rec S3 | Rec US | Rec IN |
|---|---|---|---|---|---|---|---|---|---|---|
| `Baseline_A_B_C` | **62.99%** | 41,874,576 | 18.98 | 14 | 52 | 106 | 50.42% | 49.58% | 66.09% | 33.91% |
| `Channel_D_RelaxedCharNgram` | **5.21%** | 3,929,610 | 1.78 | 0 | 22 | 68 | 57.53% | 42.47% | 51.38% | 48.62% |
| `Channel_E_Word_TFIDF_Top10` | **6.88%** | 3,628,285 | 1.64 | 0 | 10 | 10 | 72.12% | 27.88% | 55.81% | 44.19% |
| `Channel_E_Word_TFIDF_Top20` | **10.22%** | 6,575,448 | 2.98 | 0 | 20 | 20 | 63.78% | 36.22% | 53.99% | 46.01% |
| `Channel_F_Char_TFIDF_Top5` | **2.09%** | 844,587 | 0.38 | 0 | 5 | 5 | 75.41% | 24.59% | 52.30% | 47.70% |
| `Channel_F_Char_TFIDF_Top10` | **3.31%** | 1,654,591 | 0.75 | 0 | 10 | 10 | 68.35% | 31.65% | 51.61% | 48.39% |
| `Channel_F_Char_TFIDF_Top20` | **4.75%** | 3,075,350 | 1.39 | 0 | 20 | 20 | 59.79% | 40.21% | 51.55% | 48.45% |
| `Channel_F_Char_TFIDF_Top50` | **6.15%** | 5,471,433 | 2.48 | 0 | 24 | 50 | 49.51% | 50.49% | 52.25% | 47.75% |
| `Union_ABC_plus_RelaxedNgram` | **63.55%** | 43,974,983 | 19.93 | 16 | 53 | 120 | 50.42% | 49.58% | 65.93% | 34.07% |
| `Union_ABC_plus_CharTFIDF_Top10` | **63.31%** | 42,607,634 | 19.31 | 15 | 52 | 106 | 50.49% | 49.51% | 66.01% | 33.99% |
| `Union_ABC_plus_CharTFIDF_Top20` | **63.48%** | 43,393,850 | 19.66 | 16 | 52 | 109 | 50.44% | 49.56% | 65.96% | 34.04% |
| `Union_ABC_plus_CharTFIDF_Top50` | **63.82%** | 45,344,089 | 20.55 | 16 | 54 | 128 | 50.22% | 49.78% | 65.82% | 34.18% |
| `Full_Frontier_ABC_Ngram_CharTFIDF_Top20` | **63.56%** | 44,064,738 | 19.97 | 16 | 53 | 120 | 50.41% | 49.59% | 65.93% | 34.07% |
| `Full_Frontier_ABC_Ngram_CharTFIDF_Top50` | **63.83%** | 45,394,135 | 20.57 | 16 | 54 | 132 | 50.22% | 49.78% | 65.82% | 34.18% |

- **Key Insights:**
  - Relaxing character n-grams from paired first+last to selective individual n-grams jumped recall from **0.08% to 5.21%** (+65x) with only 1.78 candidates/S1.
  - Word TF-IDF Top-20 retrieved **10.22% recall** at just 2.98 candidates/S1.
  - Character TF-IDF successfully captures spelling variations, typos, and transliterations.
  - Adding Char TF-IDF (Top-50) and relaxed character n-grams raises the overall candidate ceiling to **63.83%** with only **20.57 candidates/S1** (median 16, P95 54), keeping candidate explosion completely suppressed.

---

## EXP-005: Comprehensive Nested Candidate Union Benchmark (A through G) with Entity-Level Coverage
- **Commit:** `e33686a`
- **Scope:** Full training ground-truth evaluation across 7 nested unions on all 2,206,821 Source 1 entities (7,638,365 ground-truth links).
- **Execution Time:** 986s (~16.4 min), Peak RAM ~10.8 GiB.
- **Nested Union Frontier Table:**

| Union | Link Recall (%) | Full Cov (%) | Zero Cov (%) | Cov 0% | 1-49% | 50-99% | 100% | Total Cands | Avg/S1 | Med | P95 | P99 | Max | Rec S2 | Rec S3 | Rec US | Rec IN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `A` | **23.01%** | 8.37% | 40.65% | 40.65% | 34.11% | 16.88% | 8.37% | 13,264,157 | 6.01 | 1 | 30 | 30 | 30 | 23.51% | 22.54% | 24.25% | 21.15% |
| `A+B` | **28.99%** | 12.26% | 36.00% | 36.00% | 31.13% | 20.61% | 12.26% | 18,378,813 | 8.33 | 3 | 30 | 30 | 42 | 32.03% | 26.15% | 29.71% | 27.93% |
| `A+B+C` | **60.61%** | 33.61% | 12.24% | 12.24% | 16.27% | 37.89% | 33.61% | 31,456,529 | 14.25 | 14 | 36 | 45 | 65 | 64.01% | 57.42% | 67.56% | 50.20% |
| `A+B+C+D` | **60.63%** | 33.67% | 12.23% | 12.23% | 16.26% | 37.84% | 33.67% | 31,550,537 | 14.30 | 14 | 36 | 45 | 65 | 64.04% | 57.44% | 67.57% | 50.25% |
| `A+B+C+D+E` | **71.28%** | 45.72% | 7.42% | 7.42% | 10.50% | 36.35% | 45.72% | 54,856,119 | 24.86 | 22 | 53 | 67 | 96 | 74.83% | 67.95% | 78.53% | 60.43% |
| `A+B+C+D+E+F` | **73.00%** | 47.46% | 6.45% | 6.45% | 9.66% | 36.43% | 47.46% | 114,837,298 | 52.04 | 49 | 81 | 95 | 126 | 76.90% | 69.34% | 78.82% | 64.29% |
| `A+B+C+D+E+F+G` | **76.34%** | 52.86% | 5.54% | 5.54% | 8.12% | 33.49% | 52.86% | 131,123,278 | 59.42 | 58 | 92 | 104 | 133 | 80.36% | 72.57% | 81.64% | 68.41% |

- **Cardinality Breakdown (Link Recall by Ground Truth Link Count):**
  - `A+B+C+D+E`: 70.86% on 1-match, 71.22% on 2-matches, 71.32% on 3-4 matches, 71.26% on 5+ matches.
  - `A+B+C+D+E+F+G`: 76.05% on 1-match, 76.34% on 2-matches, 76.40% on 3-4 matches, 76.29% on 5+ matches.
- **Key Breakthroughs & Diagnostics:**
  - **Channel E (Drop-One Token):** Raised link recall by **+10.65%** (from 60.63% to **71.28%**) while only increasing average candidates from 14.3 to 24.9 per S1! Full coverage rate jumped from 33.6% to 45.7%, and zero-coverage entities dropped to 7.4%.
  - **Channels F & G:** Pushed link recall further to **76.34%** (and **80.36% on S2**, **81.64% on US**), cutting zero-coverage entities down to just 5.54%.

---

## EXP-006: First Supervised LightGBM Matcher, Validation Macro F0.5 vs Volume, and RunPod GPU Setup
- **Commit:** `2c70d0e`
- **Scope:** 
  1. Trained supervised LightGBM pairwise binary classification matcher using hard negatives extracted from candidates.
  2. Evaluated on strictly held-out validation split (zero S1 leakage, 20,000 entities, 69,303 true links).
  3. Formed Missed-Link Failure Taxonomy on candidates from the frozen $A+B+C+D+E+F+G$ baseline.
  4. Benchmark of GPU embedding models on RunPod RTX PRO 4500 Blackwell GPU.
- **Matcher Validation Results across Candidate Unions:**

| Union | Validation Macro F0.5 | Precision | Recall | Cand Link Recall (%) | Avg Cands/S1 | Median | P95 | Singleton Acc | FN Damaged Entities | Singleton False Merges |
|---|---|---|---|---|---|---|---|---|---|---|
| `A` | **0.4180** | 0.8055 | 0.2273 | 22.88% | 6.03 | 1 | 30 | 0.8342 | 4 | 182 |
| `A+B` | **0.4648** | 0.8010 | 0.2872 | 29.06% | 8.34 | 3 | 30 | 0.7887 | 80 | 232 |
| `A+B+C` | **0.7094** | 0.8540 | 0.5934 | 60.72% | 14.31 | 14 | 36 | 0.7332 | 371 | 293 |
| `A+B+C+D+E` | **0.7409** | 0.8102 | 0.6918 | 71.24% | 24.87 | 22 | 53 | 0.5719 | 641 | 470 |
| `A+B+C+D+E+F+G` | **0.7528** | 0.7979 | 0.7379 | 76.29% | 59.42 | 58 | 92 | 0.5191 | 875 | 528 |

- **Missed-Link Failure Taxonomy (True Links Missed by A+B+C+D+E+F+G Candidates):**
  - **Address Variation / Alternate Locality (54.6%):** Different street components, locality names, or unaligned landmarks despite matching entity.
  - **Multilingual / Transliteration (27.8%):** Devanagari/Kannada vs Latin script representations of the same entity name/address.
  - **Heavy Spelling / OCR Typos (13.4%):** Severe character substitutions exceeding 3-gram thresholds.
  - **Missing Address Components (3.6%):** Empty address or generic city only.
  - **Extreme Name Acronyms (0.6%):** Multi-word organization vs standalone acronym.

- **RunPod GPU Embedding Models Benchmark (RTX PRO 4500 Blackwell 32.6GB VRAM):**
  - `BAAI/bge-m3`: 1024-d, 2.2 GB VRAM, 485.8 texts/sec. (Strongest candidate for multilingual Devanagari/Kannada/Latin ER).
  - `ibm-granite/granite-embedding-278m-multilingual`: 768-d, 0.5 GB VRAM, 3,529.2 texts/sec. (Ultra-fast candidate generation).
  - `intfloat/multilingual-e5-large`: 1024-d, 2.1 GB VRAM, 1,736.7 texts/sec.
  - `Qwen/Qwen3-Embedding-0.6B`: 1024-d, 3.4 GB VRAM, 492.4 texts/sec.
  - Native PyTorch CUDA Top-K vector retrieval: **2,040 qps** over 10,000 vectors.

---

## EXP-007: Addressing Address Locality (Channels H & I) and Deterministic Indic Transliteration (Channel J)
- **Commit:** `9906305`
- **Scope:** 
  1. Attack Address Variation / Alternate Locality (54.6% of EXP-006 misses):
     - **Channel H (`extract_numeric_address_keys`)**: Decoupled numeric retrieval pairing house/building numbers with 3-digit and 5-digit postal prefixes.
     - **Channel I (`extract_landmark_address_keys`)**: Regex-based landmark extractor (`near`, `opp`, `behind`, etc.) and relaxed address edge tokens (first + last significant tokens).
  2. Attack Multilingual / Transliteration (27.8% of EXP-006 misses):
     - **Channel J (`extract_transliterated_keys`)**: Implemented deterministic Unicode block mapping for Devanagari, Bengali, Tamil, and Telugu to Latin phonetics (strictly 0 external lookup, fair-play compliant).
  3. Integrated official Oracle Matcher Macro $F_{0.5}$ ceiling evaluation on full ground truth (50,000 S1 validation sample, 172,731 links).
- **Benchmark Results Across Nested Candidate Unions:**

| Union | Link Recall (%) | Oracle Macro F0.5 | Full Cov (%) | Zero Cov (%) | Total Cands | Avg/S1 | Med | P95 | P99 | Max | Rec S2 | Rec S3 | Rec US | Rec IN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `A-G` Baseline | 76.28% | 0.8856 | 52.99% | 5.56% | 2,969,002 | 59.38 | 58 | 92 | 104 | 134 | 80.13% | 72.67% | 81.68% | 68.29% |
| `Channel_H_Standalone` | 8.48% | 0.1694 | 8.84% | 80.73% | 128,613 | 2.57 | 0 | 15 | 15 | 36 | 8.57% | 8.40% | 6.31% | 11.69% |
| `Channel_I_Standalone` | 10.21% | 0.2267 | 7.05% | 71.26% | 641,074 | 12.82 | 15 | 30 | 44 | 61 | 17.05% | 3.82% | 6.28% | 16.01% |
| `A-G+H` | 77.83% | 0.8959 | 55.13% | 4.86% | 3,082,376 | 61.65 | 60 | 96 | 108 | 135 | 81.59% | 74.32% | 82.43% | 71.02% |
| `A-G+H+I` | 78.72% | 0.9017 | 56.25% | 4.49% | 3,634,575 | 72.69 | 72 | 115 | 130 | 162 | 82.98% | 74.76% | 82.88% | 72.58% |
| `Channel_J_Standalone` | 53.30% | 0.7153 | 27.53% | 18.88% | 1,028,466 | 20.57 | 19 | 40 | 43 | 45 | 59.21% | 47.79% | 56.38% | 48.75% |
| `A-G+H+I+J` | **80.84%** | **0.9128** | **59.90%** | **3.89%** | 3,836,724 | **76.73** | 75 | 122 | 138 | 178 | **84.86%** | **77.09%** | **84.73%** | **75.09%** |

- **Key Findings:**
  - **Channel J (Deterministic Transliteration)** is exceptionally strong: 53.30% standalone recall with only 20.57 cands/S1! When unioned, it pushed full candidate link recall past 80% (80.84%) and Oracle Macro $F_{0.5}$ to 0.9128, while India recall surged from 68.29% to 75.09% (+6.80%).
  - **Channel H (Numeric Address Keys)** added +1.55% link recall for only +2.27 cands/S1.
  - Zero-coverage entities dropped from 5.56% to 3.89%.

---

## EXP-008: Indic Numeral Normalization, Channel K (Typo/SymSpell), and Channel L (Acronyms & Short Initialisms)
- **Commit:** `086fe10`
- **Scope:**
  1. **Indic Numeral Canonicalization**: Added direct translation table in `normalize_clean` mapping all numerals across 9 scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam) to ASCII 0-9.
  2. **Channel K (`extract_typo_tolerant_keys`)**: Squeezed consecutive duplicate characters (`williams` -> `wiliams`) and 1-character deletion neighborhood for distinctive tokens.
  3. **Channel L (`extract_acronym_keys`)**: Generated acronym keys for multi-token business names paired with 3-digit postal code or locality token, plus short acronym fallback.
  4. Expanded transliteration dictionary to include Gujarati, Kannada, and Malayalam.
- **Benchmark Results Across Nested Candidate Unions (50,000 S1 sample, 172,731 links):**

| Union | Link Recall (%) | Oracle Macro F0.5 | Full Cov (%) | Zero Cov (%) | Total Cands | Avg/S1 | Med | P95 | P99 | Max | Rec S2 | Rec S3 | Rec US | Rec IN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `A-G+H+I+J` | 80.85% | 0.9128 | 59.90% | 3.88% | 3,837,352 | 76.75 | 76 | 122 | 138 | 177 | 84.87% | 77.10% | 84.73% | 75.10% |
| `Channel_K_Standalone` | 6.49% | 0.1466 | 7.84% | 83.13% | 657,946 | 13.16 | 15 | 34 | 45 | 105 | 9.12% | 4.02% | 8.02% | 4.21% |
| `A-G+H+I+J+K` | 80.96% | 0.9133 | 60.13% | 3.86% | 4,345,410 | 86.91 | 86 | 135 | 154 | 203 | 84.98% | 77.20% | 84.88% | 75.15% |
| `Channel_L_Standalone` | **15.20%** | **0.3139** | 7.50% | 59.39% | 319,244 | **6.38** | 3 | 15 | 15 | 15 | **26.11%** | 5.01% | **21.62%** | 5.71% |
| `A-G+H+I+J+K+L` | **81.62%** | **0.9175** | **61.09%** | **3.59%** | 4,631,440 | **92.63** | 91 | 145 | 165 | 203 | **85.91%** | **77.62%** | **85.85%** | **75.37%** |

- **Key Discoveries:**
  - **Channel L (Acronyms & Short Initialisms)** achieved an extraordinary **15.20% standalone link recall** with an ultra-compact candidate volume of **6.38 candidates per S1**! It recovered 26.11% of S2 links and 21.62% of US links.
  - The combined candidate blocker $A\dots G + H + I + J + K + L$ reached **81.62% link recall**, an **Oracle Macro $F_{0.5}$ of 0.9175**, **61.09% full entity coverage**, and dropped zero-coverage entities down to **3.59%**, all while keeping candidate volume disciplined at **92.63 cands/S1**.

---

## EXP-009: Pairwise LightGBM Matcher and Entity-Level Calibration
- **Date:** 2026-09-26
- **Validation:** 40,000 train S1 / 10,000 validation S1; 928,110 validation candidates
- **Best calibration:** `tau=0.70`, `tau_null=0.85`, `delta_multi=0.10`
- **Results:** Macro F0.5 **0.8234**, precision **0.9101**, recall **0.7239**, singleton accuracy **0.7416**
- **Runtime:** 2,025 seconds (~33.8 minutes); peak memory was not recorded.
- **Report:** `outputs/exp009_matcher/EXP009_MATCHER_CALIBRATION_REPORT.md`
- **Observation:** The lexical A...L candidate ceiling remains 81.62% link recall. The matcher’s validation F0.5 is lower than its 0.9175 candidate-oracle ceiling, so semantic retrieval is the next useful candidate-generation experiment. Do not tune against the official test set.

## EXP-010: Dense Retrieval Runner Preparation (No Embedding Run Yet)
- **Date:** 2026-09-27
- **Frozen blocker:** Keep A...L at 81.62% recall / 0.9175 oracle Macro F0.5 as the comparison baseline; do not spend more time expanding buckets before the embedding comparison.
- **Runner changes:** `code/business_entity_resolution/src/generate_gpu_dense_candidates.py` now samples only the fixed validation split, supports arbitrary countries, streams candidate embeddings in chunks, performs batched top-K without a full query-by-corpus score matrix, and records standalone recall/oracle F0.5, candidate volume/purity, and reduction ratio.
- **Correctness checks:** Python compilation succeeded locally and on EC2. The candidate reduction denominator counts the same-country query-candidate Cartesian space.
- **CPU smoke:** Installed CPU-only PyTorch and SentenceTransformers in an isolated venv outside the repo. IBM Granite encoded 256 real training records (128 with non-ASCII text) at 47.95 records/sec after a 5.2-second model load; 768-dimensional vectors; 5.34 seconds for the measured batch.
- **Compute status:** EC2 `m6a.2xlarge` has no GPU. At the measured rate, one pass over ~6.2M S2/S3 records alone projects to roughly 36 hours, before query scoring, so no full CPU retrieval was started. RunPod remains stopped. Dense candidate recall/oracle metrics have not yet been produced.
- **Model:** Default is `ibm-granite/granite-embedding-278m-multilingual` (278M parameters, Apache-2.0). Compare against `BAAI/bge-m3` only on the same validation IDs and under the same candidate budget. The runner output is standalone; fuse with A...L only after measuring incremental recall and candidate volume on identical S1 IDs.

## EXP-011: Frequency-Capped Address/Name Composite Blocking
- **Date / commit:** 2026-09-27 / `f7ca1d79a0a3a45d3a0de78c777e663637c88db6`
- **Validation:** Fixed validation IDs, 50,000 sampled S1 entities (seed 42), 173,509 true links; all 10,320,219 training S2/S3 records indexed. The official test set was not used.
- **Method:** Added country-scoped H1-H4 composite keys using the first address number, selected normalized address tokens, and the first two business-name tokens. H1/H2/H3/H4 standalone families and their unions with the current A-L implementation were evaluated at posting caps 150 and 300. These caps drop overfull postings; there is no per-S1 top-K cap for these families. See [`experiments/exp_011_h_composite_blocking/README.md`](experiments/exp_011_h_composite_blocking/README.md) for the precise positional key definitions.
- **Best candidate ceiling:** `A-L+H1-4@300` reached 93.02% link recall, 0.9732 oracle macro F0.5, 82.23% full entity coverage, and 0.87% zero coverage. It produced 12,459,148 candidates (249.18 per S1; median 166, P95 698, P99 1,000, max 1,766). This is a candidate oracle ceiling, not a pair matcher score.
- **Volume tradeoff:** `A-L+H1-4@150` reached 92.61% recall / 0.9709 oracle with 8,865,398 candidates (177.31 per S1). Raising the cap to 300 added 0.41 points of recall (712 links) at 3,593,750 more candidates. Pair purity at cap 300 was 1.2954%.
- **Breakdown:** At cap 150, S2/S3 link recall was 93.74%/91.55%; US/India was 95.90%/87.62%. H1 and H4 were the strongest individual additions to A-L, but their overlap and the marginal value of H2/H3 after them remain unmeasured.
- **Runtime/resources:** EC2 `m6a.2xlarge`, 2,360.68 seconds reported by the script, 39m 52.74s wall time, maximum RSS 27,221,708 KiB (~26.0 GiB), no swap, exit status 0.
- **Artifacts:** Exact report, JSON, metadata, and full timed log are preserved in [`experiments/exp_011_h_composite_blocking/results_2026-09-27/`](experiments/exp_011_h_composite_blocking/results_2026-09-27/).
- **Next:** Measure H1+H4 together at caps 150/300, then add H2 or H3 separately. This tests marginal recovery before accepting the high H1-H4 candidate volume. Do not compare this sample directly with older validation reports that used a different sample or code version.

