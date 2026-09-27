# Public blocking research audit

Date checked: 2026-09-27

## Findings worth testing

The strongest public result located is [Akash-bardia/amazon-ml-challenge-2026](https://github.com/Akash-bardia/amazon-ml-challenge-2026). Its README reports 96.48% candidate recall, 14.5 candidates per S1, and validation Macro F0.5 of 0.976105 on 5,000 held-out S1 entities. Its keys include compact/core names, informative name tokens, house-number plus street/name/location composites, and frequency pruning. The raw [blocking implementation](https://github.com/Akash-bardia/amazon-ml-challenge-2026/blob/main/code/business_entity_resolution/src/blocking.py) is a useful concrete reference for these keys.

The public [PrateekTechie solution](https://github.com/PrateekTechie/business-entity-resolution-amazon-ml) corroborates country partitions, informative name/address tokens, composite number/name/location keys, frequency-bounded blocks, and explicit reporting of which measurements are actually full-run versus sample-only. A separate [AyanAhmedKhan solution](https://github.com/AyanAhmedKhan/amazon-ml-challenge) uses multiple sparse retrieval views, with transliteration retained as a learned alternate view and embeddings added as features; this supports using embeddings as a residual channel rather than replacing blocking.

## Important validation caveat

The 0.9761 result is self-reported and not independently reproduced here. More importantly, the public `train.py` validation path defaults to 300,000 background distractors, loads every known train/validation true target in addition to those distractors, and retrieves only the top 20 candidates per S1. These details are visible in the [training source](https://github.com/Akash-bardia/amazon-ml-challenge-2026/blob/main/train.py#L1108-L1212). The candidate pool for that validation run is therefore smaller and easier than our complete 10.3M-record S2/S3 pool, and its recall number is not yet an apples-to-apples target. The model-threshold validation also applies target-side exclusivity; that is a separately testable assumption, not a replacement for allowing multiple matches from one S1.

Its name normalization maps all text through `text_unidecode` before key generation ([source](https://github.com/Akash-bardia/amazon-ml-challenge-2026/blob/main/code/business_entity_resolution/src/normalization.py#L66-L94)). We should preserve original Unicode and add transliterated keys alongside it, consistent with our measured script distribution.

## Next experiment

Use the existing fixed, stratified validation split. On the full training S2/S3 pool, add separately measured country-scoped channels for: (1) house number + significant address token, (2) house number + informative name token, (3) number + terminal locality token(s), and (4) name token + locality token. Cap posting-list frequency instead of cutting each S1 to an arbitrary top-K. Report each channel's marginal true links beyond A...L, union link recall, macro oracle F0.5, per-entity full/zero coverage, candidate volume and quantiles, source/country/cardinality breakdown, frequency-pruned key counts, and runtime/memory. Run frequency caps 150 and 300 as a small controlled comparison. Do not promote a channel unless its held-out gains survive the full target pool.

## Sources

- [Akash README / claimed benchmark](https://github.com/Akash-bardia/amazon-ml-challenge-2026)
- [Akash blocking keys](https://github.com/Akash-bardia/amazon-ml-challenge-2026/blob/main/code/business_entity_resolution/src/blocking.py)
- [Akash validation target-pool construction and top-K retrieval](https://github.com/Akash-bardia/amazon-ml-challenge-2026/blob/main/train.py#L1108-L1212)
- [PrateekTechie README](https://github.com/PrateekTechie/business-entity-resolution-amazon-ml)
- [AyanAhmedKhan README](https://github.com/AyanAhmedKhan/amazon-ml-challenge)
