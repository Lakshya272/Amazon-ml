# EXP-011: Frequency-Capped Address/Name Composite Blocking

## Objective

Raise candidate recall beyond the frozen A...L baseline by testing four country-scoped H subchannels: H1 house number + significant address token; H2 informative name token + house number; H3 house number + terminal locality token; H4 informative name token + terminal locality token.

## Validation and candidate universe

- Use a deterministic sample of 50,000 S1 IDs from `outputs/split/val_s1_ids.json`, generated with seed 42. This is the existing stratified, held-out S1 split.
- Index every record in the training Source 2 and Source 3 files. Do not use the official test set.
- Keep original Unicode after NFKC/case/punctuation normalization; do not ASCII transliterate these keys.
- The A...L reference union keeps its established `cap_per_channel=30` behavior. H1-H4 have no per-query top-K truncation; their target posting lists are frequency-capped at 150 and 300.

## Planned command

```bash
python3 code/business_entity_resolution/src/evaluate_blocking_system.py \
  --data-dir student_resource/dataset \
  --output-dir outputs/exp011_h_composite_blocking \
  --validation-split outputs/split/val_s1_ids.json \
  --validation-sample-s1 50000 \
  --seed 42 \
  --max-bucket-size 80 \
  --cap-per-channel 30 \
  --max-composite-bucket-size 300
```

Run under `/usr/bin/time -v` on EC2 and retain the complete console log. The script writes the nested candidate metrics, candidate-pair counts/purity/reduction, singleton candidate burden, source/country/cardinality breakdown, marginal true links beyond A...L, and H-key retention/pruning counts.

## Acceptance rule

Promote an H subchannel only if its `A-L+Hn@150` result improves fixed-validation recall/oracle Macro F0.5 at a candidate volume the matcher can process. Treat `@300` as a recall/volume sensitivity comparison. H1-H4 combined channels must report unique true links gained beyond the baseline; do not infer a gain by summing standalone results.

The public-repository audit and validation caveats are in [`PUBLIC_BLOCKING_RESEARCH.md`](../../PUBLIC_BLOCKING_RESEARCH.md).
