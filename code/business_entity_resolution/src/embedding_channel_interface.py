#!/usr/bin/env python3
"""
src/embedding_channel_interface.py — Amazon ML Challenge 2026 Business Entity Resolution
Modular Interface for Future RunPod A100 GPU Embeddings

Provides a pluggable CandidateChannel interface.
When dense vector embeddings are generated on RunPod (FAISS ANN search),
they produce candidate pairs as an additional channel that seamlessly integrates
into the blocking and candidate union pipeline:
    e.g., Union(A + B + C + D + E + F + G + Dense_Embeddings)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Tuple, Optional

class CandidateRetrievalChannel(ABC):
    """Abstract base class for all candidate generation / blocking channels."""

    @abstractmethod
    def name(self) -> str:
        """Channel identifier (e.g. 'Dense_Name_Embedding', 'Dense_Address_Embedding')."""
        pass

    @abstractmethod
    def retrieve(self, s1_id: str, country: str, business_name: str, business_address: str, top_k: int = 25) -> Set[str]:
        """
        Retrieve candidate entity IDs for a given Source 1 query.
        Returns a set of matching S2/S3 entity IDs.
        """
        pass


class DenseEmbeddingRunPodChannel(CandidateRetrievalChannel):
    """
    Candidate channel interface for precomputed or streaming dense embeddings
    generated on RunPod A100 (e.g. BAAI/bge-m3, Qwen2.5, or FAISS indices).
    """

    def __init__(self, index_path: Optional[str] = None, candidate_map_path: Optional[str] = None):
        self.index_path = index_path
        self.candidate_map_path = candidate_map_path
        self._is_loaded = False
        self._faiss_index = None

    def name(self) -> str:
        return "Channel_GPU_Dense_Embedding"

    def load_index(self):
        """Load precomputed FAISS ANN index generated on RunPod."""
        if self.index_path:
            # When index file is transferred from RunPod:
            # import faiss
            # self._faiss_index = faiss.read_index(self.index_path)
            self._is_loaded = True

    def retrieve(self, s1_id: str, country: str, business_name: str, business_address: str, top_k: int = 25) -> Set[str]:
        if not self._is_loaded:
            return set()
        # Query FAISS index and return top_k entity IDs
        return set()
