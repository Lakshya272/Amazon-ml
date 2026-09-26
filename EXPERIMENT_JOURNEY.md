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
