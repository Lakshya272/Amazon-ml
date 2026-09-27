#!/usr/bin/env python3
"""Stream the calibrated A-L matcher over every official test Source 1 record.

The blocking keys and channel caps intentionally match train_and_eval_exp009.py.
Only the supplied challenge TSVs are read. Output files are installed atomically
after the complete run, so the official validator never sees partial results.
"""

import argparse
import csv
import gc
import json
import math
import os
import time
from collections import Counter, OrderedDict, defaultdict

import lightgbm as lgb
import numpy as np

from features import EntityRecord, FEATURE_NAMES, extract_pair_features_fast
from normalize import normalize_clean, normalize_legal_name
from train_and_eval_exp009 import (
    extract_acronym_keys,
    extract_address_keys,
    extract_component_drop_address_keys,
    extract_drop_one_name_keys,
    extract_landmark_address_keys,
    extract_numeric_address_keys,
    extract_sorted_neighborhood_key,
    extract_transliterated_keys,
    extract_typo_tolerant_keys,
    get_char_ngrams,
    get_name_tokens,
)


def candidate_paths(data_dir):
    return [
        os.path.join(data_dir, "test", "test_source2.tsv"),
        os.path.join(data_dir, "test", "test_source3.tsv"),
    ]


def scan_vocabulary(paths, max_bucket_size):
    token_df = Counter()
    c3_df = Counter()
    total = 0
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            next(reader, None)
            for row in reader:
                if len(row) < 4:
                    raise ValueError(f"Malformed candidate row in {path}: {row[:1]}")
                total += 1
                legal_name = normalize_legal_name(row[1])
                token_df.update(set(get_name_tokens(legal_name)))
                c3_df.update(set(get_char_ngrams(legal_name, 3)))
    valid_tokens = {
        token: math.log((total + 1) / (count + 1))
        for token, count in token_df.items() if 2 <= count <= max_bucket_size
    }
    valid_c3 = {
        gram: math.log((total + 1) / (count + 1))
        for gram, count in c3_df.items() if 3 <= count <= max_bucket_size
    }
    return total, valid_tokens, valid_c3


class CandidateIndex:
    def __init__(self, valid_tokens, valid_c3, cap_per_channel):
        self.valid_tokens = valid_tokens
        self.valid_c3 = valid_c3
        self.cap = cap_per_channel
        self.indices = [defaultdict(list) for _ in range(12)]
        self.raw = {}

    def add(self, row):
        eid, name, addr, country = (value.strip() for value in row[:4])
        if eid in self.raw:
            raise ValueError(f"Duplicate candidate entity ID: {eid}")
        self.raw[eid] = (name, addr, country)
        ix = self.indices
        legal_name = normalize_legal_name(name)
        if legal_name:
            ix[0][(country, legal_name)].append(eid)
        clean_name = normalize_clean(name)
        name_tokens = [token for token in get_name_tokens(clean_name) if token in self.valid_tokens]
        name_tokens.sort(key=lambda token: self.valid_tokens[token], reverse=True)
        for token in name_tokens[:3]:
            ix[1][(country, token)].append(eid)
        for key in extract_address_keys(country, addr):
            ix[2][key].append(eid)
        grams = [gram for gram in get_char_ngrams(legal_name, 3) if gram in self.valid_c3]
        grams.sort(key=lambda gram: self.valid_c3[gram], reverse=True)
        for gram in grams[:2]:
            ix[3][(country, gram)].append(eid)
        for key in extract_drop_one_name_keys(country, legal_name):
            ix[4][key].append(eid)
        for key in extract_component_drop_address_keys(country, addr):
            ix[5][key].append(eid)
        neighborhood = extract_sorted_neighborhood_key(country, legal_name)
        if neighborhood:
            ix[6][neighborhood].append(eid)
        for key in extract_numeric_address_keys(country, addr):
            ix[7][key].append(eid)
        for key in extract_landmark_address_keys(country, addr):
            ix[8][key].append(eid)
        for key in extract_transliterated_keys(country, name):
            ix[9][key].append(eid)
        for key in extract_typo_tolerant_keys(country, name, addr):
            ix[10][key].append(eid)
        for key in extract_acronym_keys(country, name, addr):
            ix[11][key].append(eid)

    def retrieve(self, record):
        # EXP-009 passes normalized query fields into these helpers.
        country = record.country
        name = record.clean_name
        addr = record.clean_addr
        ix = self.indices
        cap = self.cap
        hits = Counter()

        def take(channel, key, limit):
            for candidate_id in ix[channel].get(key, ())[:limit]:
                hits[candidate_id] += 1

        if record.legal_name:
            take(0, (country, record.legal_name), cap)
        name_tokens = [token for token in record.name_toks if token in self.valid_tokens]
        name_tokens.sort(key=lambda token: self.valid_tokens[token], reverse=True)
        for token in name_tokens[:3]:
            take(1, (country, token), cap // 2)
        for key in extract_address_keys(country, addr):
            take(2, key, cap // 2)
        grams = [gram for gram in get_char_ngrams(record.legal_name, 3) if gram in self.valid_c3]
        grams.sort(key=lambda gram: self.valid_c3[gram], reverse=True)
        for gram in grams[:2]:
            take(3, (country, gram), cap // 2)
        for key in extract_drop_one_name_keys(country, record.legal_name):
            take(4, key, cap // 2)
        for key in extract_component_drop_address_keys(country, addr):
            take(5, key, cap // 2)
        neighborhood = extract_sorted_neighborhood_key(country, record.legal_name)
        if neighborhood:
            take(6, neighborhood, cap // 2)
        for key in extract_numeric_address_keys(country, addr):
            take(7, key, cap // 2)
        for key in extract_landmark_address_keys(country, addr):
            take(8, key, cap // 2)
        for key in extract_transliterated_keys(country, name):
            take(9, key, cap // 2)
        for key in extract_typo_tolerant_keys(country, name, addr):
            take(10, key, cap // 2)
        for key in extract_acronym_keys(country, name, addr):
            take(11, key, cap // 2)
        return hits


def build_index(paths, valid_tokens, valid_c3, cap):
    index = CandidateIndex(valid_tokens, valid_c3, cap)
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            next(reader, None)
            for row in reader:
                if len(row) < 4:
                    raise ValueError(f"Malformed candidate row in {path}: {row[:1]}")
                index.add(row)
        print(f"Indexed {len(index.raw):,} candidates after {os.path.basename(path)}", flush=True)
    return index


def run(args):
    started = time.time()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config["blocking"] != "A-L" or config["feature_names"] != FEATURE_NAMES:
        raise ValueError("Model configuration is incompatible with this A-L feature pipeline")
    model = lgb.Booster(model_file=args.model)
    if model.feature_name() != FEATURE_NAMES:
        raise ValueError("Saved model feature order differs from features.py")

    paths = candidate_paths(args.data_dir)
    print("Scanning test candidate vocabulary...", flush=True)
    expected_count, valid_tokens, valid_c3 = scan_vocabulary(paths, config["max_bucket_size"])
    print(f"Scanned {expected_count:,} candidates; indexing A-L...", flush=True)
    index = build_index(paths, valid_tokens, valid_c3, config["cap_per_channel"])
    if len(index.raw) != expected_count:
        raise ValueError("Candidate count changed between vocabulary and indexing passes")
    del valid_tokens, valid_c3
    gc.collect()

    os.makedirs(args.output_dir, exist_ok=True)
    candidate_final = os.path.join(args.output_dir, "candidate_pairs.tsv")
    matching_final = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_partial = candidate_final + ".inprogress"
    matching_partial = matching_final + ".inprogress"
    s1_path = os.path.join(args.data_dir, "test", "test_source1.tsv")
    candidate_cache = OrderedDict()
    counts = {"source1_rows": 0, "candidate_pairs": 0, "predicted_matches": 0, "predicted_empty": 0}

    def get_candidate(candidate_id):
        cached = candidate_cache.get(candidate_id)
        if cached is not None:
            candidate_cache.move_to_end(candidate_id)
            return cached
        raw_name, raw_addr, country = index.raw[candidate_id]
        cached = EntityRecord(
            candidate_id, raw_name, normalize_clean(raw_name), normalize_legal_name(raw_name),
            raw_addr, normalize_clean(raw_addr), country,
        )
        candidate_cache[candidate_id] = cached
        if len(candidate_cache) > args.cache_size:
            candidate_cache.popitem(last=False)
        return cached

    with open(candidate_partial, "w", encoding="utf-8", newline="") as candidate_handle, \
         open(matching_partial, "w", encoding="utf-8", newline="") as matching_handle:
        candidate_writer = csv.writer(candidate_handle, delimiter="\t", lineterminator="\n")
        matching_writer = csv.writer(matching_handle, delimiter="\t", lineterminator="\n")
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        matching_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        batch = []
        feature_rows = []

        def score_batch():
            if not batch:
                return
            scores = model.predict(np.asarray(feature_rows, dtype=np.float32), num_threads=args.threads) \
                if feature_rows else np.empty(0, dtype=np.float32)
            offset = 0
            for source_id, candidate_ids in batch:
                row_scores = scores[offset:offset + len(candidate_ids)]
                offset += len(candidate_ids)
                selected = []
                if len(row_scores):
                    best = float(np.max(row_scores))
                    if best >= config["tau_null"]:
                        floor = max(config["tau"], best - config["delta_multi"])
                        selected = [candidate_id for candidate_id, score in zip(candidate_ids, row_scores) if score >= floor]
                matching_writer.writerow([source_id, ",".join(selected)])
                counts["predicted_matches"] += len(selected)
                counts["predicted_empty"] += not selected
            if offset != len(scores):
                raise RuntimeError("Pair score count differs from queued candidate count")
            batch.clear()
            feature_rows.clear()

        with open(s1_path, "r", encoding="utf-8", errors="replace", newline="") as source_handle:
            reader = csv.reader(source_handle, delimiter="\t")
            next(reader, None)
            for row in reader:
                if len(row) < 4:
                    raise ValueError(f"Malformed Source 1 row: {row[:1]}")
                source_id, name, addr, country = (value.strip() for value in row[:4])
                source = EntityRecord(
                    source_id, name, normalize_clean(name), normalize_legal_name(name),
                    addr, normalize_clean(addr), country,
                )
                hits = index.retrieve(source)
                candidate_ids = list(hits)
                candidate_writer.writerow([source_id, ",".join(candidate_ids)])
                for candidate_id in candidate_ids:
                    feature_rows.append(extract_pair_features_fast(source, get_candidate(candidate_id), hits[candidate_id]))
                batch.append((source_id, candidate_ids))
                counts["source1_rows"] += 1
                counts["candidate_pairs"] += len(candidate_ids)
                if len(batch) >= args.batch_s1:
                    score_batch()
                if counts["source1_rows"] % 10000 == 0:
                    print(f"Processed {counts['source1_rows']:,} S1; {counts['candidate_pairs']:,} pairs; "
                          f"{time.time() - started:.0f}s elapsed", flush=True)
        score_batch()

    os.replace(candidate_partial, candidate_final)
    os.replace(matching_partial, matching_final)
    counts["elapsed_seconds"] = round(time.time() - started, 2)
    counts["candidate_records_indexed"] = expected_count
    counts["model_path"] = os.path.abspath(args.model)
    counts["config_path"] = os.path.abspath(args.config)
    with open(os.path.join(args.output_dir, "inference_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(counts, handle, indent=2, sort_keys=True)
    print(f"Completed: {counts}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="student_resource/dataset")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model", default="outputs/submission_model/matcher_a_l.txt")
    parser.add_argument("--config", default="outputs/submission_model/matcher_a_l_config.json")
    parser.add_argument("--batch-s1", type=int, default=256)
    parser.add_argument("--cache-size", type=int, default=100000)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if min(args.batch_s1, args.cache_size, args.threads) < 1:
        parser.error("batch-s1, cache-size and threads must be positive")
    run(args)
