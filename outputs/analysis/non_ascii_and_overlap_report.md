# Non-ASCII Character & Train/Test Overlap Analysis

**Generated:** 2026-09-26 04:03:32 UTC  
**Total Runtime:** 205.02s  

## 1. Non-ASCII Character Distribution across Train and Test

| Split | File | Total Records | Name Non-ASCII (%) | Addr Non-ASCII (%) | Either Non-ASCII (%) | Primary Non-ASCII Scripts |
|---|---|---|---|---|---|---|
| train_s1 | `train_source1.tsv` | 2,206,821 | 0.0% | 0.025% | 0.025% | latin_accented: 721, other_unicode: 1,315 |
| train_s2 | `train_source2.tsv` | 5,034,616 | 15.187% | 9.503% | 21.982% | devanagari: 9,351,059, other_unicode: 7,534,713, latin_accented: 299,425 |
| train_s3 | `train_source3.tsv` | 5,285,603 | 11.479% | 9.017% | 18.724% | latin_accented: 337,983, other_unicode: 4,902,578, devanagari: 6,271,439 |
| test_s1 | `test_source1.tsv` | 1,732,544 | 2.354% | 4.26% | 5.856% | latin_accented: 132,493, other_unicode: 3,200 |
| test_s2 | `test_source2.tsv` | 4,887,273 | 18.991% | 14.746% | 29.682% | latin_accented: 613,796, other_unicode: 8,905,365, devanagari: 10,924,637 |
| test_s3 | `test_source3.tsv` | 5,082,316 | 14.511% | 14.348% | 25.907% | devanagari: 7,323,141, other_unicode: 5,768,793, latin_accented: 660,436 |

### Sample Non-ASCII Business Names & Addresses

| Split | Sample Business Name (Country) | Sample Business Address (Country) |
|---|---|---|
| train_s1 | N/A () | A-212, Malhotra Complex, Gali No. Â 01, Vikas Marg, Shakarpur, Laxmi Nagar, East Delhi, Delhi (India) |
| train_s2 | राम मार्केटिंग प्राइवेट लिमिटेड (India) | PLOT NO B-78/1, ADDITIONAL MIDC ANAND NAGAR, AMBERNATH EAST, THANE, महाराष्ट्र (India) |
| train_s3 | LLC Moncada Léarning Center (US) | Door No 183, 41St Cross, 22Nd Main 9Th Block Jayanagar, Bengaluru Urban, Bangalore, ಕರ್ನಾಟಕ (India) |
| test_s1 | Maison de Santé Generation (France) | 175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine (France) |
| test_s2 | SCI Ptit Àmicale (France) | NO. 5 ALLÉE DES HÊTRES, Pornic, Loire-Atlantique (France) |
| test_s3 | मॉडर्न फाइनेंस (India) | H.no 910 A 3503, Mumbai, महाराष्ट्र (India) |

## 2. Train vs. Test Data Overlap Analysis

| Dimension | Train Pool | Test Pool | Exact Intersection | Test In Train (%) |
|---|---|---|---|---|
| Unique Business Names (S1+S2+S3) | 9,387,251 | 8,971,579 | 685,885 | **7.65%** |
| Unique Business Addresses (S1+S2+S3) | 10,584,490 | 9,894,114 | 163,213 | **1.65%** |
| Source 1 Reference Names | 1,538,804 | 1,238,244 | 194,461 | **15.7%** |

## 3. Analysis & Direct Answers to User Questions

### Question 1: If we remove non-ASCII characters to make embeddings lightweight, will it affect test performance?
**Answer: YES, it will severely degrade performance.**
- **In Test France:** 15% of test S1 entities are located in France. French records rely heavily on accented Latin characters (`é`, `è`, `à`, `ç`, `ô`, `î`). Stripping non-ASCII characters damages proper nouns and common French words (e.g. `École` -> `cole`, `Société` -> `Soci t`).
- **In Test India:** 46.8% of test S1 entities are located in India. A significant portion of names and addresses are written in native Devanagari script (e.g., `मॉडर्न फाइनेंस`, `राम मार्केटिंग`). If non-ASCII characters are stripped, these business names become empty strings or garbled artifacts, making entity resolution impossible.
- **Recommendation:** Use Unicode NFKC normalization and UTF-8 tokenization. In modern embedding models (e.g., multilingual-e5 or BGE-m3) or character n-gram hashing, UTF-8 strings are handled natively without converting to ASCII.

### Question 2: How much intersection is there between training and testing data? If we overfit, will it benefit or harm us?
**Answer: Overfitting will severely HARM us.**
- Only **15.7%** of test S1 names appear in train S1. Over **84.3%** of test reference businesses are entirely novel entities that never appeared in training.
- **The France Distribution Shift:** France represents 14.98% of the test set (259,452 entities) and **0.00%** of the training set. A model that overfits to US/India specific patterns, state abbreviations, or training vocabulary will fail completely on the French market.
- **Entity Resolution Generalization:** The task is to evaluate pairwise similarity based on generic entity resolution signals (token overlaps, edit distances, legal suffix equivalence, address components), not memorizing specific business identities.
- Any model overfitted to specific training entities will perform poorly on novel test entities and the unseen France market.