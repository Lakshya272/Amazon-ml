# EXP-012 results — 2026-09-27

## Setup

- Code commit: `059dcb6` (`feat: measure H1 and H4 blocking ablations`)
- Validation: same fixed sample as EXP-011: 50,000 S1 entities, seed 42, 173,509 true links. All 10,320,219 training S2/S3 records were indexed. Official test data was not used.
- Runtime: 2,340.87 seconds in script metadata; 39m 32.55s wall time under `/usr/bin/time -v`. Maximum RSS was 27,228,352 KiB (~26.0 GiB). No swap; exit status 0.
- This run adds H1+H4 unions and H2/H3 ablations to the existing A-L+H1-4 variants. A-L retains its per-channel cap of 30; H1-H4 use whole-posting-list caps of 150 or 300.

## Results

| Union | Link recall | Oracle macro F0.5 | Full coverage | Zero coverage | Candidate pairs | Avg/S1 | Median | P95 | P99 | Max | False pairs | Pair purity | Reduction | Added true links vs A-L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `A-L+H1+H4@150` | 91.82% | 0.9671 | 80.01% | 1.18% | 7,599,716 | 151.99 | 131 | 320 | 427 | 724 | 7,440,402 | 2.0963% | 99.9972% | 17,539 |
| `A-L+H1+H2+H4@150` | 92.33% | 0.9696 | 80.98% | 1.06% | 8,384,290 | 167.69 | 141 | 366 | 493 | 902 | 8,224,091 | 1.9107% | 99.9969% | 18,424 |
| `A-L+H1+H3+H4@150` | 92.14% | 0.9686 | 80.63% | 1.11% | 8,083,093 | 161.66 | 136 | 348 | 466 | 821 | 7,923,223 | 1.9778% | 99.9970% | 18,095 |
| `A-L+H1-4@150` | 92.61% | 0.9709 | 81.56% | 1.01% | 8,865,398 | 177.31 | 148 | 392 | 531 | 935 | 8,704,710 | 1.8125% | 99.9967% | 18,913 |
| `A-L+H1+H4@300` | 92.20% | 0.9692 | 80.59% | 1.06% | 10,058,957 | 201.18 | 142 | 529 | 750 | 1,491 | 9,898,986 | 1.5903% | 99.9963% | 18,196 |
| `A-L+H1+H2+H4@300` | 92.67% | 0.9716 | 81.49% | 0.94% | 11,369,231 | 227.38 | 156 | 624 | 900 | 1,678 | 11,208,432 | 1.4143% | 99.9958% | 19,024 |
| `A-L+H1+H3+H4@300` | 92.58% | 0.9711 | 81.36% | 0.97% | 11,153,998 | 223.08 | 152 | 607 | 853 | 1,713 | 10,993,361 | 1.4402% | 99.9958% | 18,862 |
| `A-L+H1-4@300` | **93.02%** | **0.9732** | **82.23%** | **0.87%** | 12,459,148 | 249.18 | 166 | 698 | 1,000 | 1,766 | 12,297,748 | 1.2954% | 99.9954% | 19,625 |

`A-L+H1+H4@150` is the lowest-volume measured composite union, but still averages 152 candidates/S1. Adding H2 after H1+H4 at cap 150 recovered 885 more true links (+0.51 percentage points) for 784,574 additional candidate pairs. Adding H3 instead recovered 556 (+0.32 points) for 483,377 more pairs. Adding both (the H1-H4 union) recovered 1,374 more than H1+H4 for 1,265,682 extra candidate pairs.

At cap 300, H2 after H1+H4 added 828 links (+0.47 points) for 1,310,274 additional candidates; H3 added 666 (+0.38 points) for 1,095,041 candidates. Adding both H2 and H3 after H1+H4 added 1,429 links for 2,400,191 candidates. Their marginal value falls as the posting cap rises.

The cap-150 full union had 93.74% S2 recall and 91.55% S3 recall; country recall was 95.90% US and 87.62% India. Link recall by match cardinality was 92.33% (one), 92.30% (two), 92.64% (three or four), and 92.66% (five or more). Mean candidate burden for singleton S1s was 176.47 at cap 150 and 248.52 at cap 300. H1+H4 alone used 151.06 candidates per singleton entity at cap 150. These are blocker/oracle diagnostics, not pair-matcher metrics or singleton accuracy.

## Decision

Keep `A-L+H1-4@150` as the highest measured cap-150 oracle frontier. H1+H4 alone cuts 25.32 candidate pairs/S1 for an oracle decrease of 0.0038, so it is a useful lower-volume ablation for the matcher. The @300 variants spend many more candidates for modest recall gains and should not be promoted without a measured matcher improvement.

The next blocking task is residual missed-link analysis on the same validation sample. The existing `find_true_misses.py` is not suitable: it takes the first 10,000 S1 rows and implements an older, different candidate generator. Extend the current evaluator to preserve a bounded missed-link sample for the best H1-H4 union, characterize name/address/script/numeric overlap from supplied fields, and use those examples to select the next retrieval experiment. Do not claim error categories from the old report without rechecking them.

Raw artifacts: [`BLOCKING_NESTED_UNION_REPORT.md`](BLOCKING_NESTED_UNION_REPORT.md), [`nested_union_benchmark.json`](nested_union_benchmark.json), [`blocking_experiment_metadata.json`](blocking_experiment_metadata.json), [`run.log`](run.log).
