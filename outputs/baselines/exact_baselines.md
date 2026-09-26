# Exact Matching Baselines & Legal Normalization Benchmark

**Generated:** 2026-09-25 13:55:42 UTC  

## 1. Summary Benchmark Comparison

| Baseline Strategy | Macro F0.5 | Link Recall (%) | Link Precision (%) | Predicted Links | Recovered Links | S1 with Preds (%) | Singleton Acc (%) |
|---|---|---|---|---|---|---|---|
| `exact_raw_name` | **0.1054** | 4.48% | 5.49% | 6,237,812 | 342,321 | 35.8% | 72.51% |
| `exact_base_name` | **0.2557** | 15.30% | 10.67% | 10,954,553 | 1,168,793 | 61.6% | 65.16% |
| `exact_clean_name` | **0.3176** | 20.84% | 11.81% | 13,482,057 | 1,591,759 | 70.7% | 62.34% |
| `exact_legal_name` | **0.3411** | 23.56% | 11.01% | 16,343,691 | 1,799,853 | 74.2% | 60.97% |
| `exact_raw_address` | **0.1014** | 2.23% | 82.97% | 204,979 | 170,074 | 8.9% | 98.59% |
| `exact_clean_address` | **0.2038** | 8.27% | 82.02% | 769,814 | 631,369 | 24.1% | 96.96% |
| `exact_clean_pair` | **0.0828** | 1.30% | 100.00% | 99,283 | 99,283 | 4.2% | 100.00% |
| `exact_legal_pair` | **0.1006** | 2.21% | 100.00% | 168,541 | 168,541 | 6.8% | 100.00% |
| `exact_clean_name_or_address` | **0.3781** | 27.83% | 15.02% | 14,154,229 | 2,125,486 | 76.9% | 60.40% |

## 2. In-Depth Strategy Observations & Error Dynamics

### Legal-Suffix Normalization Effect
- **Additional True Links Recovered:** +208,094 true links
- **Additional Candidate Predictions:** +2,861,634 predictions
- **Delta in Macro F0.5:** +0.023553
- **Precision Impact:** 11.81% -> 11.01%
- **Link Recall Impact:** 20.84% -> 23.56%

### Score Distributions (Per-Entity F0.5)

| Baseline Strategy | F0.5 = 1.0 (%) | F0.5 in [0.8, 1.0) (%) | F0.5 in [0.5, 0.8) (%) | F0.5 in (0, 0.5) (%) | F0.5 = 0.0 (%) |
|---|---|---|---|---|---|
| `exact_raw_name` | 4.26% | 1.39% | 6.61% | 4.81% | 82.92% |
| `exact_base_name` | 4.63% | 6.49% | 19.96% | 12.36% | 56.55% |
| `exact_clean_name` | 5.07% | 9.90% | 23.17% | 15.47% | 46.39% |
| `exact_legal_name` | 5.31% | 11.52% | 23.95% | 17.43% | 41.80% |
| `exact_raw_address` | 5.56% | 0.49% | 6.26% | 0.89% | 86.79% |
| `exact_clean_address` | 5.96% | 4.73% | 15.25% | 1.82% | 72.24% |
| `exact_clean_pair` | 5.66% | 0.54% | 3.30% | 0.29% | 90.21% |
| `exact_legal_pair` | 5.72% | 1.02% | 5.27% | 0.41% | 87.57% |
| `exact_clean_name_or_address` | 5.83% | 14.01% | 24.45% | 19.04% | 36.67% |
