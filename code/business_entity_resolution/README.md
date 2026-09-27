# Reproduce the A–L entity resolution submission

Use Python 3.9 and the supplied challenge TSVs only. All paths below are relative
to the repository or submission package root. The full run requires approximately
32 GiB RAM and enough free disk for the candidate TSV. No external business data
or lookup service is used.

```bash
python3 -m pip install -r code/business_entity_resolution/requirements.txt
```

Place the official files under `student_resource/dataset/` and generate the fixed
training/validation split if it is not already present:

```bash
python3 code/business_entity_resolution/src/validation_split.py \
  --data-dir student_resource/dataset --output-dir outputs/split
```

Train and calibrate the 32-feature LightGBM pair matcher on Source 1 training
entities and the fixed validation split. Sampling is deterministic with seed 42.
The command writes the model, calibration JSON, and validation report:

```bash
python3 code/business_entity_resolution/src/train_and_eval_exp009.py \
  --data-dir student_resource/dataset --split-dir outputs/split \
  --output-dir outputs/submission_model \
  --train-sample-s1 40000 --val-sample-s1 10000
```

Run test inference. The script scans and indexes only the supplied test S2/S3
files, streams every test S1 row, applies the exact A–L candidate rules used by
training, scores candidate pairs in batches, and writes both required TSVs:

```bash
python3 code/business_entity_resolution/src/run_test_inference_a_l.py \
  --data-dir student_resource/dataset --output-dir output \
  --model outputs/submission_model/matcher_a_l.txt \
  --config outputs/submission_model/matcher_a_l_config.json
```

The calibrated decision rule first rejects an S1 entity when its top score is
below `tau_null`. Otherwise it predicts candidates whose scores meet both `tau`
and the `delta_multi` margin from the top score. The script emits an empty list
when no candidate qualifies. `candidate_pairs.tsv` contains exactly the pairs
scored by the model. `inference_metadata.json` records counts and runtime.

Validate the two completed files before upload:

```bash
python3 student_resource/utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir student_resource/dataset/test
```

The leaderboard accepts `output/matching_results.tsv`. The final archive also
requires `output/candidate_pairs.tsv`, this code folder, and the completed
`Documentation_template.md`. The threshold values come from training validation
only; the official test set is not used for tuning.
