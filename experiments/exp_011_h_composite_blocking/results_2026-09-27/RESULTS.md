# EXP-011 results — 2026-09-27

## Setup

- Code commit: `f7ca1d79a0a3a45d3a0de78c777e663637c88db6`
- Command: `python3 code/business_entity_resolution/src/evaluate_blocking_system.py --data-dir student_resource/dataset --output-dir outputs/exp011_h_composite_blocking --validation-split outputs/split/val_s1_ids.json --validation-sample-s1 50000 --seed 42 --max-bucket-size 80 --cap-per-channel 30 --max-composite-bucket-size 300`
- Validation: 50,000 S1 entities from the fixed validation IDs, seed 42; 173,509 true links. All 10,320,219 training S2/S3 records were indexed. The official test set was not used.
- Host: existing EC2 `m6a.2xlarge`; no new instance was created.
- Runtime: 2,360.68 seconds (39m 20.7s in the script metadata; 39m 52.74s wall clock under `/usr/bin/time -v`). Maximum RSS: 27,221,708 KiB (~26.0 GiB). Exit status: 0; swap use: 0.
- H1-H4 implementation: country-scoped keys from normalized fields. The first address number is a heuristic, and H3/H4 use the last significant address tokens as locality proxies. Posting lists above the cap are discarded whole, not arbitrarily truncated.

## Candidate ceiling

| Union | Link recall | Oracle macro F0.5 | Full coverage | Zero coverage | Candidate pairs | Avg/S1 | P95 | P99 | Max | Unique true links beyond A-L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `A-L` | 81.71% | 0.9174 | 60.85% | 3.65% | 4,630,983 | 92.62 | 145 | 164 | 208 | 0 |
| `H1_Standalone@150` | 62.07% | 0.7678 | 38.57% | 15.93% | 1,233,026 | 24.66 | 118 | 175 | 368 | — |
| `A-L+H1@150` | 88.01% | 0.9506 | 72.01% | 1.82% | 5,655,954 | 113.12 | 226 | 293 | 452 | 10,938 |
| `H2_Standalone@150` | 54.04% | 0.7132 | 28.30% | 19.93% | 892,618 | 17.85 | 92 | 153 | 294 | — |
| `A-L+H2@150` | 86.15% | 0.9410 | 68.45% | 2.36% | 5,430,914 | 108.62 | 211 | 292 | 441 | 7,695 |
| `H3_Standalone@150` | 44.81% | 0.5993 | 25.79% | 32.62% | 815,288 | 16.31 | 93 | 147 | 267 | — |
| `A-L+H3@150` | 85.82% | 0.9388 | 68.27% | 2.50% | 5,338,460 | 106.77 | 202 | 267 | 407 | 7,122 |
| `H4_Standalone@150` | 58.41% | 0.7451 | 33.60% | 17.36% | 2,092,540 | 41.85 | 160 | 255 | 541 | — |
| `A-L+H4@150` | 88.07% | 0.9483 | 72.79% | 2.16% | 6,580,092 | 131.60 | 275 | 372 | 674 | 11,027 |
| `A-L+H1-4@150` | 92.61% | 0.9709 | 81.56% | 1.01% | 8,865,398 | 177.31 | 392 | 531 | 935 | 18,913 |
| `A-L+H1-4@300` | **93.02%** | **0.9732** | **82.23%** | **0.87%** | 12,459,148 | 249.18 | 698 | 1,000 | 1,766 | 19,625 |

The A-L row is the same run's reference union, so it is the right comparison for this experiment. It is a 50,000-entity sample and should not be directly compared to earlier reports with different sampled IDs or code versions. The full-training A-G result remains a separate measurement.

At cap 150, adding H1-H4 raised recall by 10.90 percentage points over A-L while increasing candidate volume by 4,234,415 pairs (+91%). Moving from cap 150 to 300 added 3,593,750 pairs (+40.5%) for 0.41 percentage points and 712 more true links. At cap 300, pair purity was 1.2954%, and 12,297,748 of 12,459,148 candidates were false pairs. The added recall is valuable, but candidate volume is now a material matcher cost.

The combined union recovered 93.74% of S2 links and 91.55% of S3 links; recall was 95.90% for US and 87.62% for India. Recall by entity match cardinality was 92.33% (one match), 92.30% (two), 92.64% (three or four), and 92.66% (five or more) at cap 150. The corresponding cap-300 rates were 92.75%, 92.77%, 93.04%, and 93.07%.

## Decision and next experiment

Keep the result as a measured recall frontier, not yet as a production candidate setting. H1 and H4 individually give the strongest A-L additions; H2 and H3 are weaker alone, but their incremental value after H1/H4 has not been measured. The next blocking experiment should compare `A-L+H1+H4`, then add H2 and H3 separately, at caps 150 and 300. This will show whether the two weaker families contribute unique matches worth their extra volume. Do not sum the standalone gains to estimate that overlap.

The raw artifacts are in this directory: [`BLOCKING_NESTED_UNION_REPORT.md`](BLOCKING_NESTED_UNION_REPORT.md), [`nested_union_benchmark.json`](nested_union_benchmark.json), [`blocking_experiment_metadata.json`](blocking_experiment_metadata.json), and [`run.log`](run.log).
