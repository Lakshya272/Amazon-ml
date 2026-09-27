# EXP-012: H1/H4 Composite Retrieval Ablation

## Hypothesis

EXP-011 showed that H1 (first address number + early address token) and H4 (name token + terminal address token) were the strongest individual additions to the current A-L union. Their overlap is unknown. This experiment measures the combined H1+H4 gain, then the marginal contribution of H2 and H3 after H1+H4, using the same fixed 50,000-S1 validation sample and full training S2/S3 pool.

## Variants

- `A-L+H1+H4`
- `A-L+H1+H2+H4`
- `A-L+H1+H3+H4`
- Reference: `A-L+H1-4`

Each is evaluated with H posting caps 150 and 300. The existing per-channel cap for A-L remains 30. H posting lists above the cap are excluded whole, without a per-S1 top-K limit.

## Reproducible command

Run on the existing EC2 `m6a.2xlarge` instance; no new compute is required or authorized by this experiment.

```bash
PYTHONUNBUFFERED=1 /usr/bin/time -v python3 code/business_entity_resolution/src/evaluate_blocking_system.py \
  --data-dir student_resource/dataset \
  --output-dir outputs/exp012_h14_ablation \
  --validation-split outputs/split/val_s1_ids.json \
  --validation-sample-s1 50000 \
  --seed 42 \
  --max-bucket-size 80 \
  --cap-per-channel 30 \
  --max-composite-bucket-size 300
```

The run log, generated report, JSON metrics, runtime, and peak memory will be recorded under `results_2026-09-27/` after the EC2 run completes.
