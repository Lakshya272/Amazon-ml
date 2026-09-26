#!/usr/bin/env python3
"""
src/check_gpu_env.py — Amazon ML Challenge 2026 Business Entity Resolution
GPU Environment & Embedding Model Verification Script

Performs:
1. GPU hardware & CUDA verification (device name, VRAM, CUDA version)
2. Dependency verification (torch, sentence-transformers, faiss, lightgbm, etc.)
3. Model smoke test for all cached models in /workspace/hf-cache/hub
4. Separate benchmark for name embeddings and address embeddings:
   - Output dimension
   - VRAM usage
   - Throughput (records/sec)
5. Tiny FAISS GPU retrieval example
"""

import os
import sys
import time
import gc

def main():
    print("=" * 60)
    print("RUNPOD GPU & ML RETRIEVAL ENVIRONMENT VERIFICATION")
    print("=" * 60)

    # 1. PyTorch & CUDA
    try:
        import torch
        print(f"[1] PyTorch Version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"    CUDA Available: {cuda_avail}")
        if cuda_avail:
            dev_name = torch.cuda.get_device_name(0)
            dev_props = torch.cuda.get_device_properties(0)
            vram_gb = dev_props.total_memory / (1024 ** 3)
            print(f"    GPU Device: {dev_name}")
            print(f"    VRAM Total: {vram_gb:.2f} GB")
            print(f"    CUDA Device Count: {torch.cuda.device_count()}")
            print(f"    Current CUDA Version: {torch.version.cuda}")
    except Exception as e:
        print(f"[!] PyTorch/CUDA Error: {e}")

    # 2. Libraries
    print("\n[2] Checking ML Libraries...")
    libs = [
        ("sentence_transformers", "sentence_transformers"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("safetensors", "safetensors"),
        ("faiss", "faiss"),
        ("sklearn", "scikit-learn"),
        ("lightgbm", "lightgbm"),
        ("xgboost", "xgboost"),
        ("rapidfuzz", "rapidfuzz"),
        ("polars", "polars"),
        ("pyarrow", "pyarrow"),
    ]
    for mod_name, label in libs:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"    [+] {label}: {ver}")
        except ImportError:
            print(f"    [-] {label}: NOT INSTALLED")
        except Exception as e:
            print(f"    [!] {label} Error: {e}")

    # 3. FAISS GPU Check
    print("\n[3] Testing FAISS GPU...")
    try:
        import faiss
        import numpy as np
        d = 64
        nb = 1000
        nq = 10
        xb = np.random.random((nb, d)).astype('float32')
        xq = np.random.random((nq, d)).astype('float32')
        
        # Test CPU Index
        index_cpu = faiss.IndexFlatIP(d)
        index_cpu.add(xb)
        D_cpu, I_cpu = index_cpu.search(xq, 5)
        print(f"    [+] FAISS CPU IndexFlatIP verified (added {nb} vectors, searched {nq} queries).")
        
        print("    [i] FAISS CPU verified. Note: Blackwell SM_100 GPU retrieval uses native PyTorch cuBLAS GEMM.")

        # Test Native PyTorch GPU GEMM Top-K retrieval
        if torch.cuda.is_available():
            t_xb = torch.from_numpy(xb).cuda()
            t_xq = torch.from_numpy(xq).cuda()
            t_sim = torch.matmul(t_xq, t_xb.T)
            t_topk = torch.topk(t_sim, k=5, dim=1)
            print(f"    [+] PyTorch GPU cuBLAS Top-5 retrieval verified on Blackwell GPU! Shape: {t_topk.indices.shape}")
    except Exception as e:
        print(f"    [!] Retrieval Test Error: {e}")

    # 4. HF Cache & Embedding Model Smoke Tests
    cache_hub = "/workspace/hf-cache/hub"
    if not os.path.exists(cache_hub):
        cache_hub = os.path.expanduser("~/.cache/huggingface/hub")
    
    print(f"\n[4] Inspecting Hugging Face Hub Cache: {cache_hub}")
    if os.path.exists(cache_hub):
        model_dirs = [d for d in os.listdir(cache_hub) if d.startswith("models--")]
        print(f"    Found {len(model_dirs)} cached models in hub:")
        for md in model_dirs:
            print(f"      - {md}")

    # Candidate models for ER benchmark
    candidate_models = [
        "BAAI/bge-m3",
        "ibm-granite/granite-embedding-278m-multilingual",
        "ibm-granite/granite-embedding-97m-multilingual-r2",
        "intfloat/multilingual-e5-large",
        "Qwen/Qwen3-Embedding-0.6B",
    ]

    sample_names = [
        "Orelee's Barbershop",
        "Prime Money",
        "B+ Retail Inc",
        "Tata Consultancy Services Limited",
        "स्टार प्रोड्यूसर प्राइवेट लिमिटेड",
        "জয় কনসাল্টিং",
        "Raj Investments",
    ]

    sample_addrs = [
        "1795 Westchester Drive, High Point, NC",
        "17560 Ellis Road, Tahlequah, OK",
        "1712 Montebello Avenue, Phoenix, AZ",
        "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
        "Plot 42, Sector 18, Electronic City, Bengaluru, Karnataka",
        "12 Rue de la Paix, Paris, France",
    ]

    print("\n[5] Embedding Model Smoke Tests & Benchmarks...")
    try:
        from sentence_transformers import SentenceTransformer
        
        for m_name in candidate_models:
            print(f"\n--- Testing: {m_name} ---")
            t0 = time.time()
            try:
                # Set HF_HOME or HF_HUB_CACHE
                os.environ["HF_HOME"] = "/workspace/hf-cache"
                model = SentenceTransformer(m_name, device="cuda" if torch.cuda.is_available() else "cpu")
                load_time = time.time() - t0
                
                # Check VRAM
                vram_used_mb = 0
                if torch.cuda.is_available():
                    vram_used_mb = torch.cuda.memory_allocated(0) / (1024 ** 2)

                # Benchmark Name Embeddings
                t_n0 = time.time()
                name_emb = model.encode(sample_names, convert_to_numpy=True, normalize_embeddings=True)
                t_n = time.time() - t_n0
                name_fps = len(sample_names) / max(t_n, 1e-4)

                # Benchmark Address Embeddings
                t_a0 = time.time()
                addr_emb = model.encode(sample_addrs, convert_to_numpy=True, normalize_embeddings=True)
                t_a = time.time() - t_a0
                addr_fps = len(sample_addrs) / max(t_a, 1e-4)

                dim = name_emb.shape[1]
                print(f"    [+] Load Success: {load_time:.2f}s | Output Dim: {dim}")
                print(f"    [+] VRAM Allocated: {vram_used_mb:.1f} MB")
                print(f"    [+] Name Throughput: {name_fps:.1f} records/sec (batch=7)")
                print(f"    [+] Address Throughput: {addr_fps:.1f} records/sec (batch=6)")
                
                # Clean up GPU memory for next model
                del model
                del name_emb
                del addr_emb
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"    [!] Failed to load {m_name}: {e}")

    except Exception as e:
        print(f"[!] SentenceTransformer error: {e}")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
