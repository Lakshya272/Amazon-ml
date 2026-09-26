# Dataset Reconnaissance Report — Amazon ML Challenge 2026

**Generated:** 2026-09-25 10:20:56 UTC  
**Execution Runtime:** 491.71s  
**Max Rows Evaluated:** FULL DATASET  

---

## 1. Executive Summary & Core Reconnaissance Facts

| Split | Dataset File | Rows | Valid Schema | Unique IDs | Duplicate ID Rate | Top Country (%) | Missing Name (%) | Missing Addr (%) |
|---|---|---|---|---|---|---|---|---|
| Train | `train_source1.tsv` | 2,206,821 | PASS | 2,206,821 | 0 (100.00%) | US (59.9792%) | 0.000% | 0.000% |
| Train | `train_source2.tsv` | 5,034,616 | PASS | 5,034,616 | 0 (100.00%) | US (59.9215%) | 0.000% | 3.356% |
| Train | `train_source3.tsv` | 5,285,603 | PASS | 5,285,603 | 0 (100.00%) | US (59.9753%) | 0.000% | 3.328% |
| Test | `test_source1.tsv` | 1,732,544 | PASS | 1,732,544 | 0 (100.00%) | India (46.7513%) | 0.000% | 0.000% |
| Test | `test_source2.tsv` | 4,887,273 | PASS | 4,887,273 | 0 (100.00%) | India (47.3181%) | 0.000% | 2.648% |
| Test | `test_source3.tsv` | 5,082,316 | PASS | 5,082,316 | 0 (100.00%) | India (47.3209%) | 0.000% | 2.678% |

---

## 2. Country Distributions: Train vs. Test

### Country Counts & Percentages

| File | US Rows (%) | India Rows (%) | France Rows (%) | Total Rows |
|---|---|---|---|---|
| `train_source1.tsv` | 1,323,633 (59.9792%) | 883,188 (40.0208%) | 0 (0.0%) | 2,206,821 |
| `train_source2.tsv` | 3,016,817 (59.9215%) | 2,017,799 (40.0785%) | 0 (0.0%) | 5,034,616 |
| `train_source3.tsv` | 3,170,056 (59.9753%) | 2,115,547 (40.0247%) | 0 (0.0%) | 5,285,603 |
| `test_source1.tsv` | 663,106 (38.2735%) | 809,986 (46.7513%) | 259,452 (14.9752%) | 1,732,544 |
| `test_source2.tsv` | 1,871,330 (38.2899%) | 2,312,565 (47.3181%) | 703,378 (14.392%) | 4,887,273 |
| `test_source3.tsv` | 1,945,701 (38.2837%) | 2,405,000 (47.3209%) | 731,615 (14.3953%) | 5,082,316 |

### Candidate Pool Sizing per Country

In Entity Resolution, Candidate Pool = (Source 2 + Source 3) candidate records per Source 1 reference record.

| Split | Country | Source 1 Ref Records | Source 2 Records | Source 3 Records | Total Candidate Pool (S2+S3) | Candidate/Ref Ratio |
|---|---|---|---|---|---|---|
| Train | **US** | 1,323,633 | 3,016,817 | 3,170,056 | 6,186,873 | 4.67x |
| Train | **India** | 883,188 | 2,017,799 | 2,115,547 | 4,133,346 | 4.68x |
| Test | **US** | 663,106 | 1,871,330 | 1,945,701 | 3,817,031 | 5.76x |
| Test | **India** | 809,986 | 2,312,565 | 2,405,000 | 4,717,565 | 5.82x |
| Test | **France** | 259,452 | 703,378 | 731,615 | 1,434,993 | 5.53x |

---

## 3. Ground Truth Structure & Linkage Analysis

- **Total S1 Ground Truth Rows:** 2,206,821
- **Total True Links:** 7,638,365
- **Average Links per S1 Entity:** 3.46125
- **Links to Source 2:** 3,693,619 (48.3561%)
- **Links to Source 3:** 3,944,746 (51.6439%)
- **Internal Duplicate Matched IDs within a Row:** 0

### Ground Truth Country Integrity Check
- **Same-Country Matches:** 7,638,365
- **Cross-Country Matches:** 0 (Rate: 0.000000%)
- **Unknown Country Matches:** 0

> [!IMPORTANT]
> **Zero Cross-Country Matches Confirmed**: In 100% of ground-truth matches, Source 1 entities only ever match Source 2/Source 3 entities in the EXACT same country. Country partitioning is an absolute, lossless blocking key (100% recall ceiling, ~40-60% candidate search space reduction per country).

### Match Count Distribution (0 to 11+ Matches)

| Match Count (k) | S1 Entities | Percentage | Cumulative S1 Entities | Cumulative % | Total Links |
|---|---|---|---|---|---|
| 0 | 123,247 | 5.585% | 123,247 | 5.58% | 0 |
| 1 | 119,157 | 5.399% | 242,404 | 10.98% | 119,157 |
| 2 | 375,212 | 17.002% | 617,616 | 27.99% | 750,424 |
| 3 | 530,841 | 24.055% | 1,148,457 | 52.04% | 1,592,523 |
| 4 | 484,115 | 21.937% | 1,632,572 | 73.98% | 1,936,460 |
| 5 | 321,957 | 14.589% | 1,954,529 | 88.57% | 1,609,785 |
| 6 | 164,868 | 7.471% | 2,119,397 | 96.04% | 989,208 |
| 7 | 63,968 | 2.899% | 2,183,365 | 98.94% | 447,776 |
| 8 | 18,680 | 0.847% | 2,202,045 | 99.78% | 149,440 |
| 9 | 4,205 | 0.191% | 2,206,250 | 99.97% | 37,845 |
| 10 | 534 | 0.024% | 2,206,784 | 100.00% | 5,340 |
| 11 | 37 | 0.002% | 2,206,821 | 100.00% | 407 |

### S1 Match Composition (S2 vs S3 Overlap)

| Composition Category | S1 Entities | Percentage | Description |
|---|---|---|---|
| `singleton_zero_matches` | 123,247 | 5.5848% | True singletons (0 matches in both S2 and S3) |
| `s2_only` | 143,029 | 6.4812% | Matches exist ONLY in Source 2 (0 in S3) |
| `s3_only` | 164,498 | 7.4541% | Matches exist ONLY in Source 3 (0 in S2) |
| `both_s2_and_s3` | 1,776,047 | 80.4799% | Matches exist in BOTH Source 2 and Source 3 |

---

## 4. Text Length Distributions & Noise Characteristics

### Business Name Lengths

| File | Char Min | Char p10 | Char Median | Char p90 | Char p99 | Char Max | Char Mean | Words Median | Words Max |
|---|---|---|---|---|---|---|---|---|---|
| `train_source1.tsv` | 3 | 14 | 24 | 34 | 42 | 105 | 24.03 | 4 | 16 |
| `train_source2.tsv` | 2 | 14 | 25 | 37 | 48 | 104 | 25.1 | 4 | 15 |
| `train_source3.tsv` | 2 | 14 | 25 | 37 | 50 | 123 | 25.2 | 4 | 18 |
| `test_source1.tsv` | 3 | 14 | 24 | 34 | 42 | 92 | 23.84 | 4 | 14 |
| `test_source2.tsv` | 2 | 14 | 25 | 38 | 49 | 102 | 25.7 | 4 | 15 |
| `test_source3.tsv` | 2 | 14 | 25 | 38 | 50 | 103 | 25.66 | 4 | 16 |

### Business Address Lengths

| File | Char Min | Char p10 | Char Median | Char p90 | Char p99 | Char Max | Char Mean | Words Median | Words Max |
|---|---|---|---|---|---|---|---|---|---|
| `train_source1.tsv` | 11 | 29 | 41 | 90 | 124 | 256 | 52.07 | 7 | 43 |
| `train_source2.tsv` | 8 | 27 | 37 | 84 | 118 | 249 | 47.83 | 6 | 46 |
| `train_source3.tsv` | 2 | 30 | 42 | 78 | 116 | 240 | 48.32 | 6 | 43 |
| `test_source1.tsv` | 11 | 31 | 50 | 93 | 126 | 268 | 57.21 | 8 | 43 |
| `test_source2.tsv` | 5 | 28 | 43 | 87 | 120 | 269 | 51.78 | 7 | 43 |
| `test_source3.tsv` | 5 | 29 | 44 | 82 | 118 | 267 | 50.08 | 7 | 43 |

### Address Structural Signals

| File | Has Digits (%) | Has Comma (%) | US Zip Match Rate (%) | IN PIN Match Rate (%) | FR Postal Match Rate (%) |
|---|---|---|---|---|---|
| `train_source1.tsv` | 96.5% | 100.0% | 10.8% | 0.0% | 0.0% |
| `train_source2.tsv` | 93.8% | 100.0% | 10.1% | 0.0% | 0.0% |
| `train_source3.tsv` | 93.9% | 100.0% | 10.2% | 0.0% | 0.0% |
| `test_source1.tsv` | 95.9% | 100.0% | 10.8% | 0.0% | 0.4% |
| `test_source2.tsv` | 95.2% | 100.0% | 10.3% | 0.0% | 0.5% |
| `test_source3.tsv` | 95.0% | 100.0% | 10.4% | 0.0% | 0.5% |

---

## 5. Duplicate Analysis within Individual Sources

Duplicate rates within a single source reveal whether business names or addresses are highly repeated.

| File | Unique Level-1 Names | Name Duplicate Rate (%) | Max Name Freq | Unique Level-1 Addrs | Addr Duplicate Rate (%) | Max Addr Freq | Exact Record Dups (%) |
|---|---|---|---|---|---|---|---|
| `train_source1.tsv` | 1,538,804 | 38.34% | 253 | 2,130,606 | 5.27% | 14 | 0.00% |
| `train_source2.tsv` | 4,192,965 | 23.23% | 464 | 4,300,098 | 20.83% | 19 | 1.89% |
| `train_source3.tsv` | 4,473,473 | 21.55% | 462 | 4,631,187 | 16.88% | 29 | 1.25% |
| `test_source1.tsv` | 1,238,244 | 36.06% | 205 | 1,674,541 | 5.29% | 101 | 0.00% |
| `test_source2.tsv` | 4,129,352 | 21.72% | 302 | 4,182,032 | 22.11% | 90 | 1.65% |
| `test_source3.tsv` | 4,357,464 | 20.19% | 387 | 4,450,182 | 18.30% | 89 | 1.12% |

---

## 6. Normalization Specification

To ensure consistent entity resolution and reproducible blocking without altering raw TSVs, two deterministic normalization levels are defined:

1. **Level 1 — Minimal Canonical Normalization (`normalize_level1`):**
   - Unicode NFKC canonical decomposition and composition (normalizes unicode variants, ligatures, half/full width).
   - Lowercase.
   - Collapse all consecutive whitespace characters (`\r`, `\n`, `\t`, spaces) into a single ASCII space and strip boundaries.
   - Preserves all accented characters (e.g. French `é`, `à`, `ç`) and native scripts (e.g. Hindi Devanagari `मॉडर्न`).

2. **Level 2 — Alphanumeric Standard Normalization (`normalize_level2`):**
   - Applies Level 1 normalization.
   - Replaces non-alphanumeric punctuation and symbol characters (`[^\w\s]`) with spaces.
   - Re-collapses consecutive whitespace and strips.
