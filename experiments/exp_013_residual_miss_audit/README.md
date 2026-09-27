# EXP-013: Residual Miss Audit for H1-H4

## Objective

The best current blocker is `A-L+H1-4@300` at 93.02% link recall. Identify observable field-level patterns among its validation misses before adding another retrieval channel. EXP-006's miss taxonomy and `find_true_misses.py` are not treated as current evidence: that script scans the first 10,000 S1 rows and rebuilds an older, different blocker.

## Method

Extend the canonical evaluator with an opt-in `--diagnose-misses` pass. On the fixed validation sample, compare every true link with `A-L+H1-4@150` and `@300`; aggregate overlap indicators across all missed true links and save a deterministic reservoir sample of at most 5,000 missed pairs per cap. The sample includes entity IDs, source/country/cardinality labels, script labels, and token/numeric overlap counts. It does not copy raw business names, addresses, tokens, or address numbers into the output.

The indicators are descriptive and may overlap. Script disagreement or zero token overlap is not by itself proof of a semantic or transliteration failure. Inspect sampled records from the supplied training files before naming a failure category.

## Run

Use the existing EC2 `m6a.2xlarge`; no additional instance is needed. Expect roughly 40 minutes and about 26 GiB peak RSS based on EXP-011/012. The extra diagnostic maps only validation ground-truth targets and retains a bounded sample.

```bash
PYTHONUNBUFFERED=1 /usr/bin/time -v python3 code/business_entity_resolution/src/evaluate_blocking_system.py \
  --data-dir student_resource/dataset \
  --output-dir outputs/exp013_residual_miss_audit \
  --validation-split outputs/split/val_s1_ids.json \
  --validation-sample-s1 50000 \
  --seed 42 \
  --max-bucket-size 80 \
  --cap-per-channel 30 \
  --max-composite-bucket-size 300 \
  --diagnose-misses \
  --miss-sample-size 5000
```

Expected diagnostics: `miss_link_diagnostics.json`, `missed_true_links_cap150_sample.tsv`, and `missed_true_links_cap300_sample.tsv`, alongside the standard nested-union report and JSON. Preserve the complete timed run log after completion.
