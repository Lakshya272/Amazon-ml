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

    # Test FAISS GPU Retrieval
    print("\n" + "="*80)
    print("TESTING FAISS GPU RETRIEVAL")
    print("="*80)
    try:
        dim = 1024
        n_docs = 10000
        n_queries = 50
        np.random.seed(42)

        data = np.random.randn(n_docs, dim).astype(np.float32)
        faiss.normalize_L2(data)
        queries = np.random.randn(n_queries, dim).astype(np.float32)
        faiss.normalize_L2(queries)

        res = faiss.StandardGpuResources()
        cpu_index = faiss.IndexFlatIP(dim)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        
        gpu_index.add(data)
        t_faiss = time.time()
        D, I = gpu_index.search(queries, k=10)
        faiss_time = time.time() - t_faiss
        print(f"  [PASS] FAISS GPU FlatIP Search: {n_queries} queries over {n_docs:,} vectors ({dim}-d) in {faiss_time*1000:.2f}ms ({round(n_queries/faiss_time, 1)} qps).")
        print(f"  Retrieved indices shape: {I.shape}, sample top distance: {D[0][0]:.4f}")
    except Exception as e:
        print(f"  [FAIL] FAISS GPU Test Error: {e}")

if __name__ == "__main__":
    run_tests()
