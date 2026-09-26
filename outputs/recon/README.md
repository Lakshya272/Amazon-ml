# Business Entity Resolution — Dataset Reconnaissance & Architectural Report

**Amazon ML Challenge 2026**  
**Phase 1 Deliverable: Dataset Reconnaissance, Normalization Specification, Metric Analysis & Validation Design**

---

## 1. Executive Summary

This report presents the complete empirical reconnaissance conducted on the Amazon ML Challenge 2026 dataset across all **24,228,873 records** spanning Source 1, Source 2, Source 3, and Ground Truth for both the training and test sets.

The profiling was executed on the AWS EC2 compute environment (`m6a.2xlarge`, 8 vCPUs, 30 GiB RAM, Amazon Linux 2023) using a streaming profiler (`src/recon.py`) that completed in **491.7 seconds (~8.2 minutes)** with memory utilization peaked at **~3.6 GiB** (well within available capacity).

### Core Takeaways
1. **Zero Cross-Country Matches (Absolute Invariance):** Among all **7,638,365** ground-truth links, exactly **0** cross national borders. Source 1 records in the US only match US records; Indian records only match Indian records. Partitioning the candidate generation stage strictly by country is 100% lossless (recall ceiling = 1.0) and immediately reduces the Cartesian comparison space by **~40% to 60%**.
2. **Country Distribution Shift in Test:** While training is strictly 60.0% US and 40.0% India, the test set exhibits a major distribution shift: **46.8% India, 38.3% US, and 15.0% France** (259,452 S1 test entities). Any pipeline hard-coded to US/India or fitted with country-specific priors will fail on 15% of the test set.
3. **Completeness & Missing Address Vulnerability:** Business name is 100.0% populated across all 24.2M records (0 missing names). In contrast, **3.35% of Source 2 and 3.33% of Source 3 records have empty addresses** (~330,000 candidate records in train; ~260,000 in test). Any blocking rule requiring address tokens or postal codes will immediately sacrifice ~3.3% recall unless paired with a name-based fallback.
4. **Heavy Multi-Match Ground Truth:** 94.42% of Source 1 entities have at least 1 match; **80.48% match entities across BOTH Source 2 and Source 3**. Only 5.58% are true singletons (0 matches). The average number of matches per S1 entity is **3.461**.
5. **Metric Asymmetry ($F_{0.5}$):** Because $F_{0.5}$ weights precision $2\times$ over recall ($4\times$ in squared denominator terms), predicting false merges heavily degrades the entity score. A model that predicts 2 true matches with 0 false positives scores **0.909**, whereas a model that achieves 100% recall (3 matches) with just 1 false positive scores **0.790**. High-precision filtering is decisive.

---

## 2. Dataset Dimensions & Schema Audit

All 7 challenge files were audited. Every file strictly complies with the expected 4-column schema (`entity_id`, `business_name`, `business_address`, `country`), using tab separation (`\t`) and UTF-8 encoding. Entity IDs are strictly unique within each file.

| Dataset File | Role | Total Rows | Valid Schema | Unique IDs | Duplicate IDs | Missing Name (%) | Missing Addr (%) | Both Missing (%) |
|---|---|---|---|---|---|---|---|---|
| `train_source1.tsv` | Train Reference | 2,206,821 | PASS | 2,206,821 | 0 (0.0%) | 0.000% | 0.000% | 0.000% |
| `train_source2.tsv` | Train Target A | 5,034,616 | PASS | 5,034,616 | 0 (0.0%) | 0.000% | 3.356% | 0.000% |
| `train_source3.tsv` | Train Target B | 5,285,603 | PASS | 5,285,603 | 0 (0.0%) | 0.000% | 3.328% | 0.000% |
| `train_ground_truth.tsv` | Train Labels | 2,206,821 | PASS | 2,206,821 | 0 (0.0%) | — | — | — |
| `test_source1.tsv` | Test Reference | 1,732,544 | PASS | 1,732,544 | 0 (0.0%) | 0.000% | 0.000% | 0.000% |
| `test_source2.tsv` | Test Target A | 4,887,273 | PASS | 4,887,273 | 0 (0.0%) | 0.000% | 2.648% | 0.000% |
| `test_source3.tsv` | Test Target B | 5,082,316 | PASS | 5,082,316 | 0 (0.0%) | 0.000% | 2.678% | 0.000% |

---

## 3. Country Distributions & Candidate Pool Sizing

### Country Representation
- **Training Set:** Exactly **59.98% US** and **40.02% India** across all three sources. France does not appear.
- **Test Set:** **46.75% India, 38.27% US, and 14.98% France**.

| File | US Rows (%) | India Rows (%) | France Rows (%) | Total Rows |
|---|---|---|---|---|
| `train_source1.tsv` | 1,323,633 (59.98%) | 883,188 (40.02%) | 0 (0.00%) | 2,206,821 |
| `train_source2.tsv` | 3,016,817 (59.92%) | 2,017,799 (40.08%) | 0 (0.00%) | 5,034,616 |
| `train_source3.tsv` | 3,170,056 (59.98%) | 2,115,547 (40.02%) | 0 (0.00%) | 5,285,603 |
| `test_source1.tsv` | 663,106 (38.27%) | 809,986 (46.75%) | 259,452 (14.98%) | 1,732,544 |
| `test_source2.tsv` | 1,871,330 (38.29%) | 2,312,565 (47.32%) | 703,378 (14.39%) | 4,887,273 |
| `test_source3.tsv` | 1,945,701 (38.28%) | 2,405,000 (47.32%) | 731,615 (14.40%) | 5,082,316 |

### Candidate Pool Ratio per Source 1 Entity
Because Source 1 matches against the union of Source 2 and Source 3:
- In **Train US**: 6,186,873 candidate records / 1,323,633 S1 records = **4.67 candidate records per S1 entity**.
- In **Train India**: 4,133,346 candidate records / 883,188 S1 records = **4.68 candidate records per S1 entity**.
- In **Test US**: 3,817,031 candidate records / 663,106 S1 records = **5.76 candidate records per S1 entity**.
- In **Test India**: 4,717,565 candidate records / 809,986 S1 records = **5.82 candidate records per S1 entity**.
- In **Test France**: 1,434,993 candidate records / 259,452 S1 records = **5.53 candidate records per S1 entity**.

The test set has slightly higher candidate density (~5.7x vs ~4.7x), indicating that candidate generation must scale efficiently without memory explosion.

---

## 4. Ground Truth Structure & Match Cardinality

### Verification of Core Match Properties
- **Total S1 Entities in Ground Truth:** 2,206,821 (100% coverage of S1)
- **Total Valid Match Links:** 7,638,365
- **Average Links per S1 Entity:** 3.46125
- **Links to Source 2:** 3,693,619 (48.36%)
- **Links to Source 3:** 3,944,746 (51.64%)
- **Internal Duplicates in Match Lists:** 0 (all ground truth match lists are cleanly deduplicated)
- **Cross-Country Match Links:** **0 out of 7,638,365** (0.000000%)

### Empirical Match Count Distribution ($k \in [0, 11]$)
The cardinality distribution is unimodal, peaking at $k = 3$ matches (24.06%), with 73.98% of entities possessing 2 to 4 matches.

| Match Count ($k$) | S1 Entities | Percentage | Cumulative Entities | Cumulative % | Total Ground Truth Links |
|---|---|---|---|---|---|
| **0 (Singletons)** | 123,247 | 5.585% | 123,247 | 5.58% | 0 |
| **1** | 119,157 | 5.399% | 242,404 | 10.98% | 119,157 |
| **2** | 375,212 | 17.002% | 617,616 | 27.99% | 750,424 |
| **3** | 530,841 | 24.055% | 1,148,457 | 52.04% | 1,592,523 |
| **4** | 484,115 | 21.937% | 1,632,572 | 73.98% | 1,936,460 |
| **5** | 321,957 | 14.589% | 1,954,529 | 88.57% | 1,609,785 |
| **6** | 164,868 | 7.471% | 2,119,397 | 96.04% | 989,208 |
| **7** | 63,968 | 2.899% | 2,183,365 | 98.94% | 447,776 |
| **8** | 18,680 | 0.847% | 2,202,045 | 99.78% | 149,440 |
| **9** | 4,205 | 0.191% | 2,206,250 | 99.97% | 37,845 |
| **10** | 534 | 0.024% | 2,206,784 | 100.00% | 5,340 |
| **11** | 37 | 0.002% | 2,206,821 | 100.00% | 407 |

### Multi-Source Overlap Breakdown
- **Singletons (0 matches in S2 & S3):** 123,247 entities (5.58%)
- **Source 2 Only:** 143,029 entities (6.48%)
- **Source 3 Only:** 164,498 entities (7.45%)
- **Both Source 2 and Source 3:** **1,776,047 entities (80.48%)**

Over 80% of entities match across both candidate sources simultaneously. The system cannot treat Source 2 and Source 3 in isolation; candidate generation must query both sources symmetrically.

---

## 5. Text Length, Duplicate Profiles & Structural Signals

### Text Distributions
- **Business Names:** Character length median is 24-25 characters; word count median is 4 words across all sources. Maximum name length is ~100-120 characters.
- **Business Addresses:** Character length median is 41-50 characters; word count median is 6-8 words. Maximum address length is ~270 characters.
- **Address Delimiters:** 100.0% of non-empty addresses contain at least one comma (`,`), reflecting structured multi-part addresses (street, city, state/region).
- **Numerical Content:** 93.8% to 96.5% of addresses contain digits (house numbers, building numbers, or postal codes).

### Duplicate Analysis Within Sources
- **Exact (Name + Address) Duplicates:**
  - Source 1: **0.00%** (strictly deduplicated).
  - Source 2: **1.89%** in train, **1.65%** in test.
  - Source 3: **1.25%** in train, **1.12%** in test.
- **Name-Only Duplicates:**
  - Source 1 has a **38.34%** normalized name duplicate rate (e.g., commercial retail chains, franchise branches, common Indian business prefixes like "Shri Ganesh Trading", "State Bank", etc.).
  - S2 and S3 have **20-23%** normalized name duplicate rates.
  - **Implication:** Name-only blocking without address context will produce massive Cartesian buckets for common names.

---

## 6. Normalization Specification

To ensure reproducible feature extraction and deterministic blocking without modifying raw TSV files, we specify two standard normalization layers implemented in `src/recon.py`:

### Layer 1: Minimal Canonical Normalization (`normalize_level1`)
1. **Unicode Canonical Normalization:** Apply `unicodedata.normalize('NFKC', text)` to standardize full-width/half-width characters, ligature forms (e.g. `ﬁ` $\to$ `fi`), and decomposed accents.
2. **Case Folding:** Convert to lower case using Python's standard unicode `.lower()`.
3. **Whitespace Normalization:** Collapse all consecutive whitespace characters (`\t`, `\n`, `\r`, spaces) to a single ASCII space ` ` and strip leading/trailing whitespace.
4. **Preservation:** Preserves native non-Latin scripts (Devanagari, Cyrillic) and Latin accented characters (`é`, `è`, `à`, `ç`).

### Layer 2: Alphanumeric Standard Normalization (`normalize_level2`)
1. Apply Layer 1.
2. Replace all punctuation and symbolic characters (`[^\w\s]` under Unicode mode) with a single space.
3. Collapse resulting whitespace and strip.

---

## 7. Mathematical Analysis: Macro $F_{0.5}$ & Cardinality Dynamics

### Formula & Asymmetry
The competition evaluation metric is the macro-averaged $F_{0.5}$ across all Source 1 entities:
$$F_{0.5} = \frac{(1 + \beta^2) \cdot \text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$

For each Source 1 entity:
- If ground truth has $|T| = 0$ (singleton):
  - $|P| = 0 \implies F_{0.5} = 1.0$ (full credit).
  - $|P| > 0 \implies F_{0.5} = 0.0$ (complete penalty for false merge).
- If $|T| > 0$:
  - $P = \frac{|P \cap T|}{|P|}$, $R = \frac{|P \cap T|}{|T|}$.

### The Asymmetric Cost of False Positives
Let $|T| = 3$ (the modal class, 24% of all entities):

| Prediction Outcome | $TP$ | $FP$ | $FN$ | Precision ($P$) | Recall ($R$) | Entity $F_{0.5}$ | Penalty vs Perfect |
|---|---|---|---|---|---|---|---|
| Perfect Prediction | 3 | 0 | 0 | 1.000 | 1.000 | **1.0000** | 0.0000 |
| Conservative: Miss 1 true, 0 FP | 2 | 0 | 1 | 1.000 | 0.667 | **0.9091** | -0.0909 |
| Conservative: Miss 2 true, 0 FP | 1 | 0 | 2 | 1.000 | 0.333 | **0.6897** | -0.3103 |
| Aggressive: Full recall + 1 FP | 3 | 1 | 0 | 0.750 | 1.000 | **0.7895** | **-0.2105** |
| Aggressive: Full recall + 2 FP | 3 | 2 | 0 | 0.600 | 1.000 | **0.6522** | **-0.3478** |
| Balanced: 2 true, 1 FP | 2 | 1 | 1 | 0.667 | 0.667 | **0.6667** | -0.3333 |

### Strategic Implications
1. **Precision Dominates Recall:** Missing one match drops $F_{0.5}$ to **0.9091**, but predicting one false positive drops $F_{0.5}$ to **0.7895**! In other words, **one false positive is more than $2.3\times$ as damaging as one missed match**.
2. **The Singleton Threshold Effect:** Singletons represent 5.58% (123,247 entities). A matcher that greedily matches every candidate entity loses up to **0.0558** macro points immediately. A high matching threshold or singleton classifier is essential.
3. **Cardinality Stopping Rule:** Only 0.2% of entities have $>8$ matches. A candidate ranking pipeline that truncates predictions at a calibrated per-entity threshold will prevent low-confidence long-tail false positives from devastating macro precision.

---

## 8. Proposed Validation Protocol

Because test labels are private, we establish an internal, leak-free validation protocol:

### Design Principles
1. **Reference Entity Split (80 / 20):**
   - Split `train_source1.tsv` into **Train-S1 (80%, ~1,765,456 entities)** and **Val-S1 (20%, ~441,365 entities)**.
2. **Stratification Variables:**
   - **Country:** Exactly 60% US, 40% India in both splits.
   - **Match Cardinality:** Exact preservation of the ground truth match count distribution ($k \in \{0, 1, 2, 3, 4, 5, 6, 7+\}$), ensuring the singleton rate is identically 5.585% in validation.
3. **Full Candidate Pool Exposure (No Target Splitting):**
   - Val-S1 queries against the **full** Source 2 and Source 3 training pools. This mirrors test inference where Test-S1 searches against all of Test-S2 and Test-S3.
4. **Leak-Free Threshold Tuning:**
   - All feature encoders, IDF vocabularies, classifier weights, and decision thresholds must be fitted strictly on Train-S1 pairs.
   - Val-S1 is held out exclusively for final scoring.

---

## 9. Recommended Candidate Generation (Blocking) Experiments

Blocking must prioritize recall ceiling while reducing the ~10-million record candidate space per country to a compact set ($\le 30-50$ candidates per S1 entity).

### Experiment Plan
We propose three blocking hypotheses to benchmark sequentially:

| Exp ID | Strategy Name | Description | Target Recall | Expected Reduction | Est. Compute (EC2) |
|---|---|---|---|---|---|
| **EXP-001** | Country Partition Baseline | Strict country blocking filter (`country_S1 == country_S23`) | 100.0% (Proven) | ~40-60% Cartesian space | < 1 min |
| **EXP-002** | Multi-Key Standard Inverted Index | Union of: (1) First 2 significant name tokens, (2) Clean address postal/city token + first name token, (3) Name Soundex/Metaphone | $\ge 96.0\%$ | $\ge 99.8\%$ | ~15-20 min CPU |
| **EXP-003** | Dual-Channel TF-IDF Cosine Retrieval | Sparse character 3-gram & word n-gram cosine retrieval via sparse matrix multiplication, taking top-$K$ ($K=30$) per country channel | $\ge 97.5\%$ | $\ge 99.9\%$ | ~30-45 min CPU |

### Computational Cost Estimate for First Experiments
- **EXP-001 & EXP-002 on EC2 `m6a.2xlarge`:**
  - Runtime: ~20-30 minutes of CPU time.
  - Memory: ~6-10 GiB RAM (well within 30 GiB).
  - AWS Cost: At on-demand rate ($0.3456/hr), running 1 hour of experiments costs **<$0.40**.
  - No GPU instances or external API calls required.

---

## 10. Audit & Submission Integrity

All code conforms to AGENTS.md:
- No external business lookup or API calls.
- Pure Python stdlib execution for reconnaissance (`src/recon.py`).
- Reproducible from project root.
