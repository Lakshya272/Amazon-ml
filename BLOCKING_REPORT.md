# Blocking & Candidate Generation Benchmark Pointer

> **NOTE:** This file has been superseded by the canonical benchmark reports produced from the full ground-truth evaluations:
> 
> - **Canonical Nested Union Benchmark Report (EXP-005):** [`outputs/blocking/BLOCKING_NESTED_UNION_REPORT.md`](file:///outputs/blocking/BLOCKING_NESTED_UNION_REPORT.md)
> - **Raw Machine-Readable JSON Output:** [`outputs/blocking/blocking_nested_union_benchmark.json`](file:///outputs/blocking/blocking_nested_union_benchmark.json)
> - **Full Experiment History & Findings:** [`EXPERIMENT_JOURNEY.md`](file:///EXPERIMENT_JOURNEY.md)
> - **Thin Supervised Matcher Pipeline Report (EXP-006):** [`outputs/matcher/MATCHER_VALIDATION_REPORT.md`](file:///outputs/matcher/MATCHER_VALIDATION_REPORT.md)

## Summary of Canonical Classical Blocking Baseline (EXP-005)

Evaluated over **2,206,821 Source 1 entities** and **7,638,365 true links**:

| Union | Link Recall (%) | Full Cov (%) | Zero Cov (%) | Total Cands | Avg/S1 | P95 | Rec S2 | Rec S3 | Rec US | Rec IN |
|---|---|---|---|---|---|---|---|---|---|---|
| `A` | **23.01%** | 8.37% | 40.65% | 13,264,157 | 6.01 | 30 | 23.51% | 22.54% | 24.25% | 21.15% |
| `A+B` | **28.99%** | 12.26% | 36.00% | 18,378,813 | 8.33 | 30 | 32.03% | 26.15% | 29.71% | 27.93% |
| `A+B+C` | **60.61%** | 33.61% | 12.24% | 31,456,529 | 14.25 | 36 | 64.01% | 57.42% | 67.56% | 50.20% |
| `A+B+C+D` | **60.63%** | 33.67% | 12.23% | 31,550,537 | 14.30 | 36 | 64.04% | 57.44% | 67.57% | 50.25% |
| `A+B+C+D+E` | **71.28%** | 45.72% | 7.42% | 54,856,119 | 24.86 | 53 | 74.83% | 67.95% | 78.53% | 60.43% |
| `A+B+C+D+E+F` | **73.00%** | 47.46% | 6.45% | 114,837,298 | 52.04 | 81 | 76.90% | 69.34% | 78.82% | 64.29% |
| `A+B+C+D+E+F+G` | **76.34%** | 52.86% | 5.54% | 131,123,278 | 59.42 | 92 | 80.36% | 72.57% | 81.64% | 68.41% |

Where:
- **A**: Exact legal-normalized business name
- **B**: Informative rare name tokens (IDF-filtered)
- **C**: Address-derived keys (postal code, building/house number + first street token)
- **D**: Selective character 3-grams
- **E**: Drop-one token name keys (for names $\ge 3$ tokens)
- **F**: Address component-drop keys (2nd/3rd address tokens)
- **G**: Sorted neighborhood / prefix key (first 8 chars of normalized name)