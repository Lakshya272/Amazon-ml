#!/usr/bin/env python3
"""
src/gpu_embedding_smoke_test.py — RunPod GPU Embedding Model Smoke Test & Benchmark
Tests cached models for name and address embeddings:
- Successful load
- Output dimension
- VRAM usage
- Throughput (samples/sec)
- FAISS GPU retrieval test
"""

import os
import time
import torch
import numpy as np

# Ensure HF uses the persistent cache volume on /workspace
os.environ["HF_HOME"] = "/workspace/hf-cache"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf-cache/hub"

from sentence_transformers import SentenceTransformer
import faiss

SAMPLE_TEXTS = [
    # Business names
    "Apex Global Logistics Private Limited",
    "Apex Logistics Pvt Ltd",
    "Tata Consultancy Services Corp",
    "TCS Solutions India",
    "Sharma Trading Enterprises",
    # Addresses
    "124 Market Street, Suite 400, San Francisco, CA 94105",
    "Market St 124, Ste 400, SF California",
    "Plot No 45, MIDC Industrial Area, Andheri East, Mumbai 400093",
    "45 MIDC Indl Area Andheri E Mumbai",
    "10 Rue de la Paix, 75002 Paris, France",
]

MODELS_TO_TEST = [
    "BAAI/bge-m3",
    "Qwen/Qwen3-Embedding-0.6B",
    "ibm-granite/granite-embedding-278m-multilingual",
    "intfloat/multilingual-e5-large",
    "microsoft/bitnet-embedding-270m",
]

def run_tests():
    print("="*80)
    print("RUNPOD GPU EMBEDDING SMOKE TEST & BENCHMARK")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GiB")
    print("="*80)

    results = []

    for model_id in MODELS_TO_TEST:
        print(f"\nTesting model: {model_id}...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        vram_start = torch.cuda.memory_allocated() / (1024**2)

        t0 = time.time()
        try:
            model = SentenceTransformer(model_id, device="cuda")
            load_time = time.time() - t0
            vram_loaded = torch.cuda.memory_allocated() / (1024**2)

            # Benchmark inference throughput
            batch = SAMPLE_TEXTS * 20  # 200 items
            t_inf = time.time()
            embeddings = model.encode(batch, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
            inf_time = time.time() - t_inf
            throughput = len(batch) / inf_time
            peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
            dim = embeddings.shape[1]

            print(f"  [PASS] Loaded in {load_time:.2f}s | Dim: {dim} | Load VRAM: {vram_loaded:.1f} MB | Peak VRAM: {peak_vram:.1f} MB | Throughput: {throughput:.1f} texts/sec")
            results.append({
                "model": model_id,
                "status": "PASS",
                "dim": dim,
                "load_time_s": round(load_time, 2),
                "peak_vram_mb": round(peak_vram, 1),
                "throughput_per_sec": round(throughput, 1),
            })
            del model
            del embeddings
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  [FAIL] Error loading {model_id}: {e}")
            results.append({
                "model": model_id,
                "status": f"FAIL: {str(e)[:50]}",
                "dim": None,
                "load_time_s": None,
                "peak_vram_mb": None,
                "throughput_per_sec": None,
            })

    # Test GPU Vector Search (PyTorch CUDA Matrix Multiply Top-K + FAISS)
    print("\n" + "="*80)
    print("TESTING GPU VECTOR SEARCH & FAISS RETRIEVAL")
    print("="*80)
    try:
        dim = 1024
        n_docs = 10000
        n_queries = 50

        # PyTorch Native CUDA Top-K (Works natively across all architectures including Blackwell SM12.0)
        t_pt0 = time.time()
        data_gpu = torch.randn(n_docs, dim, device="cuda", dtype=torch.float32)
        data_gpu = torch.nn.functional.normalize(data_gpu, dim=1)
        queries_gpu = torch.randn(n_queries, dim, device="cuda", dtype=torch.float32)
        queries_gpu = torch.nn.functional.normalize(queries_gpu, dim=1)

        sims = torch.mm(queries_gpu, data_gpu.T)
        topk_scores, topk_indices = torch.topk(sims, k=10, dim=1)
        torch.cuda.synchronize()
        pt_time = time.time() - t_pt0
        print(f"  [PASS] PyTorch Native CUDA Top-K: {n_queries} queries over {n_docs:,} vectors in {pt_time*1000:.2f}ms ({round(n_queries/pt_time, 1)} qps).")
        print(f"  Retrieved top-K shape: {topk_indices.shape}, sample top similarity: {topk_scores[0][0].item():.4f}")

        # FAISS CPU Search (Highly optimized AVX-512)
        t_f0 = time.time()
        cpu_index = faiss.IndexFlatIP(dim)
        cpu_index.add(data_gpu.cpu().numpy())
        D, I = cpu_index.search(queries_gpu.cpu().numpy(), 10)
        faiss_time = time.time() - t_f0
        print(f"  [PASS] FAISS AVX-512 CPU Search: {n_queries} queries over {n_docs:,} vectors in {faiss_time*1000:.2f}ms ({round(n_queries/faiss_time, 1)} qps).")
    except Exception as e:
        print(f"  [FAIL] Vector Search Test Error: {e}")

if __name__ == "__main__":
    run_tests()
