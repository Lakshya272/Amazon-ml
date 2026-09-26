# Amazon ML Challenge 2026 — Business Entity Resolution (Record Linkage)

Repository: [https://github.com/Lakshya272/Amazon-ml](https://github.com/Lakshya272/Amazon-ml)

An end-to-end two-stage entity resolution pipeline mapping Source 1 business records to zero, one, or multiple matching entities across Source 2 and Source 3.

---

## 1. Repository Layout

```text
├── code/
│   └── business_entity_resolution/
│       └── src/                             # Complete candidate blocking & evaluation pipeline
│           ├── evaluate_blocking_system.py  # Full Channels A–G nested candidate union benchmark
│           ├── evaluate_matcher_pipeline.py # LightGBM matcher, damaged entity analysis, taxonomy
│           ├── validation_split.py          # Deterministic stratified 80/20 split (0 S1 leakage)
│           ├── metrics.py                   # Official Macro F0.5 & singleton handling implementation
│           ├── normalize.py                 # Multi-level NFKC & legal suffix canonicalization
│           ├── features.py                  # High-throughput pairwise similarity features
│           ├── embedding_channel_interface.py# Modular interface for GPU dense embedding channels
│           ├── gpu_embedding_smoke_test.py  # RunPod GPU embedding benchmarks & vector retrieval
│           ├── recon.py                     # Initial dataset reconnaissance on all 24.2M records
│           └── analyze_chars_and_overlap.py # Non-ASCII & script distribution analysis
├── outputs/                                 # Raw metric outputs & reproducible audit reports
│   ├── blocking/                            # Raw JSON & MD outputs for all blocking benchmarks
│   │   ├── BLOCKING_NESTED_UNION_REPORT.md  # Official A through G nested union frontier table
│   │   ├── nested_union_benchmark.json      # Raw JSON metrics for candidate unions
│   │   └── BLOCKING_REPORT.md               # Historical channel evaluation report
│   ├── matcher/                             # Matcher validation and failure taxonomy reports
│   │   ├── MATCHER_AND_TAXONOMY_REPORT.md   # Final validation Macro F0.5 vs volume
│   │   └── matcher_pipeline_results.json    # Raw JSON matcher metrics & taxonomy
│   ├── baselines/                           # Exact matching baselines & legal suffix benchmarks
│   ├── analysis/                            # Character encoding, non-ASCII & train/test overlap
│   └── recon/                               # Full 24.2M row schema reconnaissance reports
├── BLOCKING_REPORT.md                       # High-level summary of candidate blocking benchmarks
├── EXPERIMENT_JOURNEY.md                    # Full chronological experiment log (EXP-001 to EXP-006)
├── requirements.txt                         # Pinned Python package dependencies
└── AGENTS.md                                # Project rules, architecture specifications, constraints
```

---

## 2. Environment Setup

### Prerequisites
- Python 3.9+ (Python 3.11 / 3.12 recommended)
- 16+ GB RAM recommended for full-dataset blocking (memory efficient streaming used throughout)

```bash
# Clone the repository
git clone git@github.com:Lakshya272/Amazon-ml.git
cd Amazon-ml

# Install pinned dependencies
pip install -r requirements.txt
```

---

## 3. Dataset Placement

Ensure the official challenge dataset is placed under `student_resource/dataset/`:
```text
student_resource/dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

*Note: Raw TSV files are hosted as GitHub Release assets and via Git LFS pointers to preserve repository performance.*

---

## 4. Exact Execution Instructions

### A. Run Full Channels A–G Candidate Blocking Benchmark
Streams all 2.2M Source 1 entities across 10.3M candidate records and evaluates nested unions ($A \to A+B \to A+B+C \to A+B+C+D+E \to A+B+C+D+E+F+G$):
```bash
python code/business_entity_resolution/src/evaluate_blocking_system.py \
  --data-dir student_resource/dataset \
  --output-dir outputs/blocking
```
- Output: `outputs/blocking/BLOCKING_NESTED_UNION_REPORT.md` and `outputs/blocking/nested_union_benchmark.json`.

### B. Run Deterministic Validation Split Generator
Generates a stratified 80/20 train/validation split (1.76M train S1, 441k val S1) with strictly zero S1 leakage:
```bash
python code/business_entity_resolution/src/validation_split.py \
  --data-dir student_resource/dataset \
  --output-dir outputs/split
```

### C. Run Supervised LightGBM Matcher, Validation F0.5 & Failure Taxonomy
Trains a pairwise GBDT matcher on hard negatives and evaluates across candidate unions on the validation split:
```bash
python code/business_entity_resolution/src/evaluate_matcher_pipeline.py \
  --data-dir student_resource/dataset \
  --output-dir outputs/matcher \
  --split-dir outputs/split \
  --train-sample-s1 40000 \
  --val-sample-s1 25000 \
  --threshold 0.50
```
- Output: `outputs/matcher/MATCHER_AND_TAXONOMY_REPORT.md` and `outputs/matcher/matcher_pipeline_results.json`.

### D. Run GPU Embedding Benchmarks (RunPod / CUDA)
Benchmarks cached embedding models (`BAAI/bge-m3`, `ibm-granite`, `multilingual-e5`) and vector retrieval:
```bash
python code/business_entity_resolution/src/gpu_embedding_smoke_test.py
```

---

## 5. Authoritative Benchmark Results Summary

### Candidate Blocking Frontier (Full Training Ground Truth: 2.2M S1, 7.64M True Links)
| Union | Link Recall (%) | Full Cov (%) | Zero Cov (%) | Total Candidates | Avg Cands/S1 | Median | P95 |
|---|---|---|---|---|---|---|---|
| `A` | 23.01% | 8.37% | 40.65% | 13,264,157 | 6.01 | 1 | 30 |
| `A+B` | 28.99% | 12.26% | 36.00% | 18,378,813 | 8.33 | 3 | 30 |
| `A+B+C` | 60.61% | 33.61% | 12.24% | 31,456,529 | 14.25 | 14 | 36 |
| `A+B+C+D+E` | 71.28% | 45.72% | 7.42% | 54,856,119 | 24.86 | 22 | 53 |
| `A+B+C+D+E+F+G` | **76.34%** | **52.86%** | **5.54%** | 131,123,278 | 59.42 | 58 | 92 |

### Matcher Validation Performance (Held-out Validation Split)
| Candidate Union | Validation Macro $F_{0.5}$ | Precision | Recall | Cand Link Recall (%) | Avg Cands/S1 |
|---|---|---|---|---|---|
| `A` | 0.4180 | 0.8055 | 0.2273 | 22.88% | 6.03 |
| `A+B` | 0.4648 | 0.8010 | 0.2872 | 29.06% | 8.34 |
| `A+B+C` | 0.7094 | **0.8540** | 0.5934 | 60.72% | 14.31 |
| `A+B+C+D+E` | 0.7409 | 0.8102 | 0.6918 | 71.24% | 24.87 |
| `A+B+C+D+E+F+G` | **0.7528** | 0.7979 | **0.7379** | **76.29%** | 59.42 |
