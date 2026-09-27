# EXP-011: Frequency-Capped Address/Name Composite Blocking

## Objective

Raise candidate recall beyond the A...L implementation baseline by testing four country-scoped cross-field composite subchannels.

The implementation uses the first numeric sequence in the normalized address as `primary_number`; it does not establish that this is a house number. Address tokens exclude address stop words, digit-only tokens, and tokens shorter than three characters. H1 combines the primary number with up to the first three remaining address tokens; H2 combines it with the first two name tokens; H3 combines it with the last two remaining address tokens; H4 combines the first two name tokens with the last two remaining address tokens. These terminal tokens are positional heuristics, not parsed locality fields. Keys preserve Unicode and include country in the index key.

## Validation and candidate universe

- Use a deterministic sample of 50,000 S1 IDs from `outputs/split/val_s1_ids.json`, generated with seed 42. This is the existing stratified, held-out S1 split.
- Index every record in the training Source 2 and Source 3 files. Do not use the official test set.
- Keep original Unicode after NFKC/case/punctuation normalization; do not ASCII transliterate these keys.
- The A...L reference union keeps its established `cap_per_channel=30` behavior. H1-H4 have no per-query top-K truncation. Posting lists larger than the selected frequency cap are excluded entirely; the experiment tests caps of 150 and 300.
- Channel letters have been reused over the project history. For this run, `A-L` means the current twelve channels in `evaluate_blocking_system.py`; `H1-H4` means the new composite families above. In particular, H1-H4 are distinct from the earlier numeric-address `Channel_H`.

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

## Completed run

The fixed-sample benchmark completed successfully on 2026-09-27. Its exact report, machine-readable results, run metadata, and `/usr/bin/time -v` log are preserved in [`results_2026-09-27/`](results_2026-09-27/). The findings and the next ablation are summarized in [`results_2026-09-27/RESULTS.md`](results_2026-09-27/RESULTS.md).
