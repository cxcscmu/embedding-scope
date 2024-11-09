"""
Implement the dot-product dense retriever.
"""

from typing import List, Tuple
from faiss import IndexFlatIP, GpuMultipleClonerOptions, index_cpu_to_gpus_list
import numpy as np
from numpy.typing import NDArray
from source.interface.retriever import DenseRetriever


class DotProductRetriever(DenseRetriever):
    """
    Implementation for the dot-product dense retriever.

    This class leverages FAISS to perform efficient similarity search on dense
    vectors. It supports GPU acceleration and sharding across multiple devices.
    """

    def __init__(self, size: int, devices: List[int]):
        index = IndexFlatIP(size)
        options = GpuMultipleClonerOptions()
        options.shard = True
        self.ids: List[str] = []
        self.index: IndexFlatIP = index_cpu_to_gpus_list(index, options, devices)

    def add(self, ids: List[str], vectors: NDArray[np.float32]):
        self.ids.extend(ids)
        self.index.add(vectors)

    def search(
        self, vectors: NDArray[np.float32], topK: int
    ) -> Tuple[List[List[str]], List[List[float]]]:
        scores, indices = self.index.search(vectors, topK)
        return [[self.ids[i] for i in row] for row in indices], scores.tolist()
