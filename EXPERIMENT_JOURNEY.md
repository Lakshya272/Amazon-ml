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


