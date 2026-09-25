\# AGENTS.md — Amazon ML Challenge 2026



\## 1. Project



This repository is for the Amazon ML Challenge 2026 Business Entity Resolution (ER) task.



The objective is to identify, for every Source 1 entity, all matching entities from Source 2 and Source 3.



Each Source 1 entity may have:

\- zero matches

\- one match

\- multiple matches



The official challenge files are under:



`student\_resource/dataset/`



Do not unnecessarily modify or rename the supplied challenge files.



\---



\## 2. Critical Competition Rules



These rules are mandatory.



\### No external entity lookup



DO NOT use external data to identify or enrich businesses.



Forbidden:

\- commercial entity-resolution APIs

\- government business registries

\- geocoding APIs

\- internet business lookup

\- web searches for individual businesses

\- external business databases

\- external data augmentation



The challenge explicitly prohibits external lookup. Do not attempt to circumvent this restriction.



All evidence used for matching must come from the supplied challenge data.



\### Model constraint



Any final model must:

\- use an MIT or Apache 2.0 compatible license

\- have no more than 8 billion parameters



Do not introduce a model that violates these constraints.



\---



\## 3. Official Dataset



The supplied data is TSV, not CSV.



Training:



`student\_resource/dataset/train/train\_source1.tsv`



`student\_resource/dataset/train/train\_source2.tsv`



`student\_resource/dataset/train/train\_source3.tsv`



`student\_resource/dataset/train/train\_ground\_truth.tsv`



Test:



`student\_resource/dataset/test/test\_source1.tsv`



`student\_resource/dataset/test/test\_source2.tsv`



`student\_resource/dataset/test/test\_source3.tsv`



The training data contains ground truth.



The test data does not contain ground truth.



Do not use the test set to tune thresholds, blocking rules, models, or other decisions.



\---



\## 4. Evaluation Metric



The primary metric is macro F0.5.



For each Source 1 entity, calculate precision and recall and then:



F0.5 = (1.25 \* Precision \* Recall) /

&#x20;      (0.25 \* Precision + Recall)



Then average the per-entity F0.5 values across Source 1 entities.



Precision is weighted more heavily than recall.



False merges are therefore particularly costly.



A correct empty match list for a true singleton/no-match entity receives full credit for that entity.



Do not optimize ordinary accuracy as the primary objective.



Every serious experiment must report:



\- macro F0.5

\- precision

\- recall

\- singleton/no-match performance

\- candidate recall

\- candidate reduction ratio

\- number of predicted matches

\- false-positive behavior



\---



\## 5. System Architecture



Treat entity resolution as a two-stage system:



&#x20;   Source records

&#x20;        |

&#x20;        v

&#x20;   Blocking / Candidate Generation

&#x20;        |

&#x20;        v

&#x20;   Candidate Pairs

&#x20;        |

&#x20;        v

&#x20;   Pairwise Matching

&#x20;        |

&#x20;        v

&#x20;   Final Matching Decisions

&#x20;        |

&#x20;        v

&#x20;   matching\_results.tsv



Blocking and matching are separate problems.



\### Blocking



Blocking should prioritize high recall.



A true pair excluded during blocking can never be recovered by the matching model.



Measure blocking recall against training ground truth.



\### Matching



The matching stage should prioritize precision.



It must be capable of rejecting candidates and returning no match.



Do not assume that every Source 1 entity has a match.



\---



\## 6. Candidate Pairs



`candidate\_pairs.tsv` represents the final candidate set after blocking/filtering.



Every final predicted match MUST exist in the candidate set.



Candidate generation will be evaluated/audited for:

\- blocking recall

\- recall ceiling

\- reduction ratio

\- pipeline integrity



Do not generate an unnecessarily enormous Cartesian product unless an experiment explicitly demonstrates that it is useful.



\---



\## 7. Matching Features



Start with strong, interpretable features before introducing complex models.



Potential features include:



\### Name



\- normalized exact match

\- token overlap

\- Jaccard similarity

\- character n-gram similarity

\- Levenshtein/edit similarity

\- TF-IDF cosine similarity

\- abbreviation compatibility

\- legal suffix handling

\- token reordering robustness



\### Address



Consider:



\- normalized exact similarity

\- token overlap

\- character n-gram similarity

\- Levenshtein/edit similarity

\- TF-IDF cosine similarity

\- house/building number

\- street information

\- city

\- state/province

\- postal code

\- landmark

\- component overlap

\- component reordering



\### Other



\- country equality

\- country-specific normalization

\- missing-field patterns

\- name/address interaction features



Do not assume one similarity metric is sufficient.



\---



\## 8. Known Noise Patterns



The data may contain:



\### Business-name variation



\- abbreviations

\- legal suffixes

\- DBA/trade names

\- punctuation changes

\- word-order changes

\- typos



\### Address variation



\- abbreviations

\- transliteration

\- missing components

\- landmarks

\- municipal numbering

\- component reordering



Build normalization and matching logic to handle these systematically.



Do not hard-code assumptions based only on a handful of examples.



\---



\## 9. Country Handling



Training contains US and India.



The test set additionally contains France.



Do NOT hard-code the pipeline to only US/India.



Country-specific normalization and matching logic is allowed and should be evaluated empirically.



Do not assume that one normalization strategy works equally well for all countries.



\---



\## 10. Validation



Because the official test set has no labels:



Create an internal validation split from the training data.



The validation procedure must be reproducible.



Do not repeatedly change the validation split simply to obtain a better score.



Prefer a fixed validation split or a clearly documented cross-validation strategy.



All threshold/model/blocking decisions should be based on training/validation data only.



\---



\## 11. Experiment Discipline



Every meaningful experiment must record:



\- experiment ID

\- date/time

\- code/configuration version

\- blocking strategy

\- matching model

\- feature set

\- thresholds

\- training data used

\- validation data used

\- candidate count

\- candidate recall

\- reduction ratio

\- precision

\- recall

\- macro F0.5

\- runtime

\- memory usage

\- relevant observations



Do not overwrite good experiment results.



Use a structured experiment directory.



Example:



&#x20;   experiments/

&#x20;     exp\_001\_baseline/

&#x20;     exp\_002\_blocking\_v2/

&#x20;     exp\_003\_char\_features/

&#x20;     exp\_004\_pair\_classifier/



\---



\## 12. Data Engineering



The dataset is large.



Avoid blindly loading every TSV into pandas multiple times.



Before implementing expensive processing:



1\. inspect schemas

2\. inspect row counts

3\. inspect dtypes

4\. inspect missingness

5\. estimate memory usage

6\. identify appropriate indexes

7\. consider chunked processing

8\. benchmark bottlenecks



Prefer efficient representations for repeated matching operations.



Do not duplicate the multi-GB dataset unnecessarily.



\---



\## 13. AWS Execution Policy



\### Local PC



The local PC is primarily the control plane.



Agents may:

\- edit code

\- inspect configuration

\- issue AWS/SSH commands

\- inspect experiment results

\- orchestrate workloads



Avoid running the full multi-GB processing pipeline locally.



\### AWS EC2



Heavy workloads should run on EC2.



Use EC2 for:

\- dataset processing

\- blocking

\- feature generation

\- model training

\- evaluation

\- large experiments



\### S3



Current bucket:



`s3://amazon-ml-challenge-2026-689330778799`



Region:



`us-east-1`



Use S3 for persistent:

\- dataset archives

\- models/checkpoints

\- experiment artifacts

\- outputs

\- logs



Do not create unnecessary duplicate copies of large files.



\---



\## 14. AWS Cost Safety



AWS resources cost money.



Before creating expensive resources:



\- explicitly identify the instance type

\- identify expected runtime

\- check whether the resource is necessary

\- avoid unnecessary GPU usage

\- avoid leaving instances running when idle

\- save important artifacts to S3 before terminating compute



Do NOT autonomously:

\- create large GPU instances

\- create multiple expensive instances

\- increase quotas

\- modify billing settings

\- create unrelated AWS services



The user controls expensive infrastructure decisions.



CPU/GPU resources should be selected based on measured workload requirements rather than assumed requirements.



\---



\## 15. Current AWS Environment



Current region:



`us-east-1`



Current S3 bucket:



`s3://amazon-ml-challenge-2026-689330778799`



Current planned compute:



`m6a.2xlarge`



Current EC2 standard On-Demand quota:



8 vCPUs



The `m6a.2xlarge` uses 8 vCPUs.



Do not launch additional standard On-Demand instances without checking the available quota.



GPU quotas currently need to be treated as unavailable unless explicitly verified otherwise.



\---



\## 16. Initial Workflow



Do NOT immediately build a sophisticated model.



Follow this order:



\### Phase 1 — Dataset reconnaissance



Determine:



\- schemas

\- row counts

\- entity counts

\- missingness

\- country distribution

\- name/address statistics

\- ground-truth structure

\- match cardinality

\- singleton frequency

\- duplicate behavior

\- memory requirements



Produce a concise report.



\### Phase 2 — Validation framework



Implement:



\- deterministic train/validation split

\- ground-truth parsing

\- per-entity precision

\- per-entity recall

\- per-entity F0.5

\- macro F0.5

\- candidate recall

\- candidate reduction ratio



Verify the evaluator independently on simple synthetic cases.



\### Phase 3 — Blocking baseline



Implement several inexpensive blocking strategies.



Measure candidate recall before optimizing matching.



\### Phase 4 — Matching baseline



Build an interpretable pairwise baseline using engineered string/address features.



\### Phase 5 — Iteration



Improve:



\- normalization

\- blocking

\- pair features

\- classifier

\- thresholds

\- singleton handling



Every change must be validated quantitatively.



\### Phase 6 — Advanced methods



Only after establishing strong baselines consider:



\- learned embeddings

\- neural pair models

\- semantic similarity

\- larger models



Any advanced model must remain within the challenge's model-size and licensing constraints.



\### Phase 7 — Test inference



Only after the pipeline is locked:



\- generate test candidate pairs

\- run matching

\- generate `matching\_results.tsv`

\- validate the submission

\- preserve the exact required format



\---



\## 17. Submission Integrity



Final submission must contain the required files and structure.



At minimum:



`output/matching\_results.tsv`



`output/candidate\_pairs.tsv`



and the required code/documentation package.



Run the official validator before considering a submission complete:



&#x20;   python3 utils/validate\_submission.py \\

&#x20;     --matching output/matching\_results.tsv \\

&#x20;     --candidate output/candidate\_pairs.tsv \\

&#x20;     --test-dir dataset/test



Never submit an unvalidated file.



For `matching\_results.tsv`:



\- exactly one row per Source 1 test entity

\- `matched\_entity\_ids` contains comma-separated S2/S3 IDs

\- empty list means no match

\- no duplicate matched IDs

\- every ID must exist in Source 2 or Source 3

\- every final matched ID must occur in `candidate\_pairs.tsv`



\---



\## 18. Agent Behavior



Before making substantial architectural changes:



1\. inspect the existing code

2\. inspect the relevant data

3\. inspect previous experiment results

4\. explain the hypothesis

5\. implement the smallest useful experiment

6\. measure it

7\. preserve the result

8\. compare against the established baseline



Do not rewrite working components without evidence.



Do not claim an improvement without a measured validation result.



Do not optimize runtime at the expense of F0.5 without measuring the tradeoff.



Do not optimize recall alone if it causes precision/F0.5 to deteriorate.



When uncertain, run a small experiment rather than guessing.



\---



\## 19. External Tools and Internet



The challenge prohibits external business/entity lookup.



Do not use the internet to search for individual businesses or addresses.



General technical documentation, package documentation, and programming references are acceptable unless the challenge rules indicate otherwise.



Never use external business information as a matching signal.



\---



\## 20. Definition of Done



A solution is not considered complete merely because it produces matches.



A complete pipeline must have:



\- reproducible validation

\- measured blocking recall

\- measured candidate reduction

\- measured macro F0.5

\- strong singleton handling

\- documented matching logic

\- reproducible inference

\- valid candidate\_pairs.tsv

\- valid matching\_results.tsv

\- successful official submission validation

\- reproducible code and experiment records

\- no prohibited external data

