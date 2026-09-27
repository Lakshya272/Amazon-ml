#!/usr/bin/env python3
"""
src/generate_gpu_dense_candidates.py — Amazon ML Challenge 2026 Business Entity Resolution
GPU Dense Vector Embedding Retrieval & Candidate Generation Pipeline

Uses:
1. SOTA pre-trained multilingual embedding model (e.g., ibm-granite/granite-embedding-278m-multilingual or BAAI/bge-m3)
2. Chunked PyTorch matrix multiplication + top-k on GPU or CPU
3. Text representation: '{clean_legal_name} | {clean_address} | {country}'
4. Top-K nearest candidate retrieval per S1 entity
5. Samples entities from the fixed validation split
6. Saves dense candidates and oracle metrics for same-split fusion
"""

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Set, Tuple, Any, Optional

import numpy as np
import torch

from normalize import normalize_clean, normalize_legal_name
from metrics import compute_entity_metrics

# Ensure UTF-8 output
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass


def build_entity_text(name: str, addr: str, country: str) -> str:
    """Build standardized semantic input text for dense embedding."""
    clean_n = normalize_legal_name(name)
    clean_a = normalize_clean(addr)
    return f"{clean_n} | {clean_a} | {country}".strip()


def run_dense_retrieval(
    data_dir: str,
    output_dir: str,
    model_name: str = "ibm-granite/granite-embedding-278m-multilingual",
    max_s1: int = 10000,
    top_k: int = 20,
    batch_size: int = 256,
    split_dir: str = "outputs/split",
    seed: int = 42,
    candidate_chunk_size: int = 50000,
    query_batch_size: int = 128,
    hf_home: Optional[str] = None,
):
    if top_k < 1 or batch_size < 1 or candidate_chunk_size < 1 or query_batch_size < 1:
        raise ValueError("top_k, batch_size, candidate_chunk_size, and query_batch_size must be positive")

    print("=" * 60)
    print("RUNNING GPU DENSE EMBEDDING RETRIEVAL")
    print(f"Model: {model_name} | Top-K: {top_k} | Max S1: {max_s1:,}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    split_path = os.path.join(split_dir, "val_s1_ids.json")
    if not os.path.isfile(split_path):
        raise FileNotFoundError(f"Fixed validation IDs not found: {split_path}")
    with open(split_path, "r", encoding="utf-8") as f:
        validation_ids = json.load(f)
    sample_size = len(validation_ids) if max_s1 <= 0 else min(max_s1, len(validation_ids))
    selected_s1_ids = set(random.Random(seed).sample(validation_ids, sample_size))
    print(f"[{time.strftime('%H:%M:%S')}] Selected {len(selected_s1_ids):,} IDs from fixed validation split (seed={seed}).")

    # 1. Load Ground Truth
    gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    print(f"[{time.strftime('%H:%M:%S')}] Step 1: Loading ground truth...")
    truth: Dict[str, Set[str]] = {}
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if not row: continue
            s1_id = row[0].strip()
            if s1_id not in selected_s1_ids:
                continue
            matched = row[1].strip() if len(row) > 1 else ""
            if matched:
                matches = {x.strip() for x in matched.split(",") if x.strip()}
                truth[s1_id] = matches
            else:
                truth[s1_id] = set()
    missing_truth = selected_s1_ids.difference(truth)
    if missing_truth:
        raise ValueError(f"Validation split IDs missing from ground truth: {len(missing_truth):,}")
    total_links = sum(map(len, truth.values()))

    target_s1_ids = set(truth.keys())
    print(f"  Loaded {len(truth):,} S1 entities ({total_links:,} true links).")

    # 2. Load S1 Query Records
    s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
    print(f"[{time.strftime('%H:%M:%S')}] Step 2: Loading S1 records...")
    s1_ids = []
    s1_texts = []
    s1_countries = []
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 4: continue
            eid = row[0].strip()
            if eid in target_s1_ids:
                s1_ids.append(eid)
                s1_texts.append(build_entity_text(row[1].strip(), row[2].strip(), row[3].strip()))
                s1_countries.append(row[3].strip())

    print(f"  Prepared {len(s1_ids):,} S1 query texts.")

    # 3. Load Candidate Records (S2 & S3)
    # To optimize candidate search, index candidate records partitioned by country
    cand_by_country: Dict[str, Tuple[List[str], List[str]]] = {}
    s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train", "train_source3.tsv")

    print(f"[{time.strftime('%H:%M:%S')}] Step 3: Loading candidate records (S2 & S3)...")
    for path in [s2_path, s3_path]:
        s_tag = "S2" if "source2" in path else "S3"
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if len(row) < 4: continue
                eid = row[0].strip()
                country = row[3].strip()
                if country not in cand_by_country:
                    cand_by_country[country] = ([], [])
                cand_by_country[country][0].append(eid)
                cand_by_country[country][1].append(build_entity_text(row[1].strip(), row[2].strip(), country))

    for c, (c_ids, _) in cand_by_country.items():
        print(f"  Country '{c}': {len(c_ids):,} candidate records.")

    # 4. Load Embedding Model
    print(f"\n[{time.strftime('%H:%M:%S')}] Step 4: Loading SentenceTransformer model...")
    from sentence_transformers import SentenceTransformer
    if hf_home:
        os.environ["HF_HOME"] = hf_home
    elif os.path.isdir("/workspace/hf-cache"):
        os.environ.setdefault("HF_HOME", "/workspace/hf-cache")
    model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    print("  Model loaded successfully!")

    # 5. Encode S1 Query Vectors
    print(f"\n[{time.strftime('%H:%M:%S')}] Step 5: Encoding S1 query vectors...")
    t_s1_enc = time.time()
    s1_embeddings = model.encode(
        s1_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"  S1 vectors encoded in {time.time() - t_s1_enc:.2f}s | Shape: {s1_embeddings.shape}")

    # 6. Stream source vectors and query batches. Never allocate the full
    #    query-by-corpus similarity matrix or a duplicate full-corpus tensor.
    print(f"\n[{time.strftime('%H:%M:%S')}] Step 6: Performing chunked dense Top-K retrieval...")
    dense_candidates: Dict[str, Set[str]] = {eid: set() for eid in s1_ids}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for country, (c_ids, c_texts) in cand_by_country.items():
        # Get S1 indices for this country
        curr_s1_indices = [idx for idx, c in enumerate(s1_countries) if c == country]
        if not curr_s1_indices or not c_ids:
            continue

        print(f"\n  Processing country '{country}': {len(curr_s1_indices):,} S1 queries against {len(c_ids):,} candidates...")
        t_c_enc = time.time()
        
        k = min(top_k, len(c_ids))
        q_vecs = s1_embeddings[curr_s1_indices].to(device)
        best_scores = torch.full((len(curr_s1_indices), k), -torch.inf, device=device)
        best_indices = torch.full((len(curr_s1_indices), k), -1, dtype=torch.long, device=device)

        for cand_start in range(0, len(c_texts), candidate_chunk_size):
            cand_end = min(cand_start + candidate_chunk_size, len(c_texts))
            chunk_texts = c_texts[cand_start:cand_end]
            emb = model.encode(
                chunk_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device=device
            )
            emb = emb.to(device)
            local_k = min(k, cand_end - cand_start)
            for q_start in range(0, len(curr_s1_indices), query_batch_size):
                q_end = min(q_start + query_batch_size, len(curr_s1_indices))
                scores = torch.matmul(q_vecs[q_start:q_end], emb.T)
                local_scores, local_indices = torch.topk(scores, k=local_k, dim=1)
                local_indices += cand_start
                merged_scores = torch.cat((best_scores[q_start:q_end], local_scores), dim=1)
                merged_indices = torch.cat((best_indices[q_start:q_end], local_indices), dim=1)
                best_scores[q_start:q_end], keep = torch.topk(merged_scores, k=k, dim=1)
                best_indices[q_start:q_end] = torch.gather(merged_indices, 1, keep)
            del emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        top_indices = best_indices.cpu().numpy()
        for local_q, original_s1_idx in enumerate(curr_s1_indices):
            s1_id = s1_ids[original_s1_idx]
            for cand_idx in top_indices[local_q]:
                if cand_idx >= 0:
                    dense_candidates[s1_id].add(c_ids[int(cand_idx)])

        print(f"    Streamed {len(c_ids):,} candidates in {time.time() - t_c_enc:.2f}s.")
        del q_vecs, best_scores, best_indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # 7. Evaluate Dense Standalone Candidate Recall & Oracle F0.5
    print(f"\n[{time.strftime('%H:%M:%S')}] Step 7: Evaluating Dense Retrieval Candidate Coverage...")
    total_recovered = 0
    total_cands = 0
    cand_counts = []
    full_cov = 0
    zero_cov = 0
    oracle_f05_scores = []

    for s1_id, true_set in truth.items():
        retrieved = dense_candidates.get(s1_id, set())
        n_c = len(retrieved)
        total_cands += n_c
        cand_counts.append(n_c)
        rec = retrieved & true_set
        n_rec = len(rec)
        total_recovered += n_rec

        n_true = len(true_set)
        if n_true == 0:
            full_cov += 1
            oracle_f05_scores.append(1.0)
        else:
            if n_rec == n_true:
                full_cov += 1
            elif n_rec == 0:
                zero_cov += 1
            p = n_rec / max(n_rec, 1)  # Oracle precision is 1.0 on retrieved true
            r = n_rec / n_true
            if p + r > 0:
                f05 = (1.25 * p * r) / (0.25 * p + r)
            else:
                f05 = 0.0
            oracle_f05_scores.append(f05)

    link_recall = total_recovered / total_links * 100
    macro_f05 = np.mean(oracle_f05_scores)
    full_cov_pct = full_cov / len(truth) * 100
    zero_cov_pct = zero_cov / len(truth) * 100
    avg_cands = np.mean(cand_counts)
    med_cands = np.median(cand_counts)
    p95_cands = np.percentile(cand_counts, 95)
    max_cands = np.max(cand_counts)
    queries_by_country: Dict[str, int] = {}
    for country in s1_countries:
        queries_by_country[country] = queries_by_country.get(country, 0) + 1
    same_country_pair_space = sum(
        query_count * len(cand_by_country.get(country, ([], []))[0])
        for country, query_count in queries_by_country.items()
    )
    reduction_ratio = 1.0 - (total_cands / max(same_country_pair_space, 1))
    false_candidate_count = total_cands - total_recovered
    candidate_pair_precision = total_recovered / max(total_cands, 1)

    print("\n" + "=" * 60)
    print("DENSE EMBEDDING RETRIEVAL STANDALONE BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Link Recall:         {link_recall:.2f}% ({total_recovered:,}/{total_links:,})")
    print(f"Oracle Macro F0.5:   {macro_f05:.4f}")
    print(f"Full Entity Cov:     {full_cov_pct:.2f}%")
    print(f"Zero Entity Cov:     {zero_cov_pct:.2f}%")
    print(f"Candidate Volume:    Total={total_cands:,} | Avg={avg_cands:.2f} | Med={med_cands:.0f} | P95={p95_cands:.0f} | Max={max_cands}")
    print(f"Candidate Purity:    {candidate_pair_precision:.4%} ({total_recovered:,} true / {false_candidate_count:,} false candidates)")
    print(f"Candidate Reduction: {reduction_ratio:.6%} over same-country Cartesian space")
    print(f"Runtime:             {time.time() - t_start:.2f}s")
    print("=" * 60)

    # Save candidates dictionary for fusion with lexical blockers
    out_json = os.path.join(output_dir, "dense_candidates.json")
    print(f"\nSaving dense candidate pairs to {out_json}...")
    with open(out_json, "w", encoding="utf-8") as f:
        # Convert sets to lists for JSON serialization
        json.dump({k: list(v) for k, v in dense_candidates.items()}, f)

    metrics_json = {
        "model": model_name,
        "validation_split": os.path.abspath(split_path),
        "validation_sample_size": len(s1_ids),
        "seed": seed,
        "top_k": top_k,
        "total_true_links": total_links,
        "true_links_recovered": total_recovered,
        "candidate_link_recall": total_recovered / max(total_links, 1),
        "oracle_macro_f05": float(macro_f05),
        "full_entity_coverage": full_cov_pct / 100.0,
        "zero_coverage_nonempty": zero_cov_pct / 100.0,
        "total_candidates": int(total_cands),
        "avg_candidates_per_s1": float(avg_cands),
        "median_candidates_per_s1": float(med_cands),
        "p95_candidates_per_s1": float(p95_cands),
        "max_candidates_per_s1": int(max_cands),
        "true_candidate_pairs": int(total_recovered),
        "false_candidate_pairs": int(false_candidate_count),
        "candidate_pair_precision": float(candidate_pair_precision),
        "same_country_reduction_ratio": float(reduction_ratio),
        "runtime_seconds": time.time() - t_start,
    }
    metrics_path = os.path.join(output_dir, "dense_retrieval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    print("Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Dense Candidate Retrieval")
    parser.add_argument("--data-dir", type=str, default="student_resource/dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/dense_candidates")
    parser.add_argument("--model-name", type=str, default="ibm-granite/granite-embedding-278m-multilingual")
    parser.add_argument("--max-s1", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--split-dir", type=str, default="outputs/split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-chunk-size", type=int, default=50000)
    parser.add_argument("--query-batch-size", type=int, default=128)
    parser.add_argument("--hf-home", type=str, default=None)
    args = parser.parse_args()

    run_dense_retrieval(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_s1=args.max_s1,
        top_k=args.top_k,
        batch_size=args.batch_size,
        split_dir=args.split_dir,
        seed=args.seed,
        candidate_chunk_size=args.candidate_chunk_size,
        query_batch_size=args.query_batch_size,
        hf_home=args.hf_home,
    )
