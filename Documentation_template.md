# Amazon ML Challenge 2026: Business Entity Resolution

**Team Name:** test  
**Team Members:** Tarun Bansal, Lakshya Jain, Anuj Mohanty  
**Submission Date:** 2026-09-27

## 1. Executive Summary

We resolve Source 1 businesses against Sources 2 and 3 with country-scoped, multi-key blocking followed by a LightGBM pair classifier. Name, address, postal, numeric, and cross-field similarities support multi-match predictions and explicit empty predictions. All matching evidence comes from the supplied challenge files.

## 2. Methodology

### 2.1 Data analysis

The training set has 2,206,821 Source 1 entities, 10,320,219 Source 2/3 entities, and 7,638,365 true links. Ground truth includes 123,247 Source 1 entities with no match (5.585%). Every labeled link stays within its country. Names vary through legal suffixes, abbreviations, transliteration, spelling, and order; addresses vary through missing components, abbreviations, landmarks, and number formats. Training covers US and India. The test set also contains France, which is processed with the same open-country pipeline; France matching quality cannot be directly measured from the available labels.

### 2.2 Validation

We fix a deterministic 80/20 Source 1 train/validation partition with seed 42. The pair matcher experiment samples 40,000 training Source 1 entities and 10,000 validation Source 1 entities from that fixed split. All threshold choices use training/validation labels only. The primary selection metric is per-entity macro F0.5, including full credit for a correct empty prediction. We also track pooled precision and recall, singleton accuracy, candidate recall, and candidate reduction.

## 3. Candidate Generation

The A–L blocker indexes country-scoped normalized name, rare name tokens, address components, name character trigrams, drop-one-name variants, address variants, sorted name neighborhoods, numeric address keys, landmark keys, transliterated name keys, typo-tolerant keys, and acronyms. Frequent keys are pruned, and each channel has a retrieval cap. Candidate lists are deduplicated before pair scoring. The exact final candidate set is written to `output/candidate_pairs.tsv`.

On the fixed 50,000-entity labeled benchmark, A–L retrieved 4,631,440 pairs (92.63 per Source 1), recovered 81.62% of true links, and reduced the all-pairs search space by approximately 99.998%. Its 0.9175 oracle macro F0.5 is a blocking ceiling assuming perfect decisions over retrieved candidates; it is not the classifier result. The final test candidate count is recorded by `output/inference_metadata.json` after inference.

## 4. Matching Model

The 32 pair features cover exact and fuzzy normalized names, token Jaccard/Dice/overlap, character n-grams, transliteration, acronyms, address similarity, postal and numeric agreement, landmarks, name-address interactions, candidate source, and blocker channel hits. We train a LightGBM classifier on labeled pairs from the fixed training sample. The final decision rule predicts no match if the highest pair score falls below `tau_null`; otherwise it retains candidates meeting both `tau` and the allowed `delta_multi` margin below the top score. The three parameters are selected by validation macro F0.5. LightGBM is MIT licensed and has no neural parameter-count concern.

## 5. Results and Error Analysis

The prior A–L matcher run scored **0.8234 macro F0.5**, **0.9101 pooled precision**, **0.7239 pooled recall**, and **0.7416 singleton accuracy** on its 10,000-entity validation sample, with `tau=0.70`, `tau_null=0.85`, and `delta_multi=0.10`. A final wider calibration run is pending; replace these values with the selected saved model's report before packaging if it improves validation. The A–L blocker misses links when all indexed variants fail to retrieve a true pair. The matcher can also reject true candidates or merge lookalike businesses, especially where names are generic or address components are missing. The empty prediction gate limits false merges for entities without matches.

## 6. Reproducibility and Conclusion

`code/business_entity_resolution/README.md` gives the exact commands for deterministic splitting, training, full test inference, and the official validator. `requirements.txt` pins the Python package versions. Inference scans only the supplied test TSVs, scores each A–L candidate, and writes one row for every test Source 1 entity, including France and empty predictions. It installs completed output files atomically. The final files are validated before submission.

## Appendix A. Code artifacts

- `src/validation_split.py`: fixed train/validation partition.
- `src/train_and_eval_exp009.py`: A–L candidate generation, pair training, and threshold calibration.
- `src/run_test_inference_a_l.py`: end-to-end test inference and both TSV outputs.
- `src/normalize.py`, `src/transliterate.py`, `src/features.py`: normalization and pair features.
- `README.md`, `requirements.txt`: reproduction instructions and environment.

## Appendix B. Additional observations

A separate expanded H1–H4 blocker reached a 0.9732 candidate oracle ceiling on the labeled benchmark. It is not the final submitted blocker and does not imply a 0.9732 matcher score. A Granite embedding retrieval experiment is separate from this A–L pipeline and is excluded unless same-split matcher F0.5 improvement is demonstrated and full outputs can be regenerated and validated.
