#!/usr/bin/env python3
"""
src/generate_gpu_dense_candidates.py — Amazon ML Challenge 2026 Business Entity Resolution
GPU Dense Vector Embedding Retrieval & Candidate Generation Pipeline

Uses:
1. SOTA pre-trained multilingual embedding model (e.g., ibm-granite/granite-embedding-278m-multilingual or BAAI/bge-m3)
2. Native PyTorch cuBLAS matrix multiplication + torch.topk on NVIDIA RTX PRO 4500 GPU
3. Text representation: '{clean_legal_name} | {clean_address} | {country}'
4. Top-K nearest candidate retrieval per S1 entity
5. Unions dense vector candidates with lexical blocking candidates (A...L)
6. Computes official Candidate Link Recall and Oracle Macro F0.5 ceiling
"""

import argparse
import csv
import gc
import json
import math
import os
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
):
    print("=" * 60)
    print("RUNNING GPU DENSE EMBEDDING RETRIEVAL")
    print(f"Model: {model_name} | Top-K: {top_k} | Max S1: {max_s1:,}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    # 1. Load Ground Truth
    gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")
    print(f"[{time.strftime('%H:%M:%S')}] Step 1: Loading ground truth...")
    truth: Dict[str, Set[str]] = {}
    total_links = 0
    with open(gt_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        count = 0
        for row in reader:
            if not row: continue
            s1_id = row[0].strip()
            matched = row[1].strip() if len(row) > 1 else ""
            if matched:
                matches = {x.strip() for x in matched.split(",") if x.strip()}
                truth[s1_id] = matches
                total_links += len(matches)
            else:
                truth[s1_id] = set()
            count += 1
            if max_s1 and count >= max_s1:
                break

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
    cand_by_country = {"US": ([], []), "India": ([], [])}
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
    print(f"\n[{time.strftime('%H:%M:%S')}] Step 4: Loading SentenceTransformer model on GPU...")
    from sentence_transformers import SentenceTransformer
    os.environ["HF_HOME"] = "/workspace/hf-cache"
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

    # 6. Retrieve Nearest Candidates per Country using GPU cuBLAS
    print(f"\n[{time.strftime('%H:%M:%S')}] Step 6: Performing GPU cuBLAS Top-K retrieval...")
    dense_candidates: Dict[str, Set[str]] = {eid: set() for eid in s1_ids}
    s1_id_to_idx = {eid: idx for idx, eid in enumerate(s1_ids)}

    for country, (c_ids, c_texts) in cand_by_country.items():
        # Get S1 indices for this country
        curr_s1_indices = [idx for idx, c in enumerate(s1_countries) if c == country]
        if not curr_s1_indices or not c_ids:
            continue

        print(f"\n  Processing country '{country}': {len(curr_s1_indices):,} S1 queries against {len(c_ids):,} candidates...")
        t_c_enc = time.time()
        
        # Sub-batch candidate encoding to conserve VRAM
        cand_embeds_list = []
        chunk_size = 50000
        for i in range(0, len(c_texts), chunk_size):
            chunk_texts = c_texts[i:i + chunk_size]
            emb = model.encode(
                chunk_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                normalize_embeddings=True,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            cand_embeds_list.append(emb)

        cand_embeddings = torch.cat(cand_embeds_list, dim=0)
        print(f"    Candidate vectors encoded in {time.time() - t_c_enc:.2f}s | Shape: {cand_embeddings.shape}")

        # Query batching
        q_vecs = s1_embeddings[curr_s1_indices]
        q_batch_size = 5000
        for q_start in range(0, len(curr_s1_indices), q_batch_size):
            q_chunk = q_vecs[q_start:q_start + q_batch_size]
            # PyTorch cuBLAS GEMM: (Q, D) x (D, C) -> (Q, C)
            sim_matrix = torch.matmul(q_chunk, cand_embeddings.T)
            topk_res = torch.topk(sim_matrix, k=min(top_k, len(c_ids)), dim=1)
            topk_indices = topk_res.indices.cpu().numpy()

            for i, local_idx in enumerate(range(q_start, min(q_start + q_batch_size, len(curr_s1_indices)))):
                orig_s1_idx = curr_s1_indices[local_idx]
                s1_id = s1_ids[orig_s1_idx]
                for cand_idx in topk_indices[i]:
                    dense_candidates[s1_id].add(c_ids[cand_idx])

        # Free GPU memory for next country
        del cand_embeddings
        del cand_embeds_list
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
            if n_c == 0:
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

    print("\n" + "=" * 60)
    print("DENSE EMBEDDING RETRIEVAL STANDALONE BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Link Recall:         {link_recall:.2f}% ({total_recovered:,}/{total_links:,})")
    print(f"Oracle Macro F0.5:   {macro_f05:.4f}")
    print(f"Full Entity Cov:     {full_cov_pct:.2f}%")
    print(f"Zero Entity Cov:     {zero_cov_pct:.2f}%")
    print(f"Candidate Volume:    Total={total_cands:,} | Avg={avg_cands:.2f} | Med={med_cands:.0f} | P95={p95_cands:.0f} | Max={max_cands}")
    print(f"Runtime:             {time.time() - t_start:.2f}s")
    print("=" * 60)

    # Save candidates dictionary for fusion with lexical blockers
    out_json = os.path.join(output_dir, "dense_candidates.json")
    print(f"\nSaving dense candidate pairs to {out_json}...")
    with open(out_json, "w", encoding="utf-8") as f:
        # Convert sets to lists for JSON serialization
        json.dump({k: list(v) for k, v in dense_candidates.items()}, f)

    print("Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Dense Candidate Retrieval")
    parser.add_argument("--data-dir", type=str, default="student_resource/dataset")
    parser.add_argument("--output-dir", type=str, default="outputs/dense_candidates")
    parser.add_argument("--model-name", type=str, default="ibm-granite/granite-embedding-278m-multilingual")
    parser.add_argument("--max-s1", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    run_dense_retrieval(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        max_s1=args.max_s1,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )
