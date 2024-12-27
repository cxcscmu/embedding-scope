"""
The dense retriever.

This file implements the dense retriever. The similarity between two dense
vectors is calculated using the dot product. The retriever supports batch
indexing and querying for efficiency.

@author Hao Kang <haok@andrew.cmu.edu>
@date December 27, 2024
"""

from typing import List, Dict

import numpy as np
from numpy import ndarray as NDArray
from faiss import IndexFlatIP, GpuMultipleClonerOptions, index_cpu_to_gpus_list


class Retriever:
    """
    The dense retriever.

    This retriever is designed for dense vectors, where each vector is a
    1D array of feature values. The similarity between two vectors is
    calculated using the dot product.

    The retriever uses Faiss as the backend for indexing and querying the
    vectors. Faiss is a library for efficient similarity search and
    clustering of dense vectors. It is optimized for high-dimensional
    vectors and supports both CPU and GPU acceleration.

    Unlike the sparse retriever, the dense retriever does not require an
    external server for indexing and querying. The vectors are stored in
    memory by Faiss for efficient retrieval.
    """

    def __init__(self, size: int, devices: List[int]) -> None:
        """
        Initialize the dense retriever.

        Parameters
        ----------
        size : int
            The dimensionality of the vectors.
        devices : List[int]
            The list of GPU devices to use for acceleration. If the list is
            empty, the retriever will use the CPU for computation.
        """
        self.size = size
        self.devices = devices
        self.built = False
        self.lookup: List[str] = []

    def __enter__(self):
        """
        Initialize the Faiss index for the dense vectors.
        """
        # index on GPUs doesn't support iteratively adding vectors so we need to
        # build the index on CPU first, and on the first query, move it to GPU.
        self.index = IndexFlatIP(self.size)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Free the resources used by the Faiss index.
        """
        del self.index

    def batch_index(self, payload: Dict[str, NDArray[np.float32]]):
        """
        Index a batch of dense vectors.

        Parameters
        ----------
        payload : Dict[str, NDArray[np.float32]]
            A dictionary of dense vectors, where the key is the document ID
            and the value is a 1D array of feature values.
        """
        self.lookup.extend(payload.keys())
        features = np.stack(list(payload.values()))
        self.index.add(features)

    def batch_query(
        self, payload: List[NDArray[np.float32]], top_k: int
    ) -> NDArray[np.int64]:
        """
        Query a batch of dense vectors.

        Parameters
        ----------
        payload : NDArray[np.float32]
            A 2D array of dense vectors, where each row is a 1D array of
            feature values.
        top_k : int
            The number of nearest neighbors to return for each query.

        Returns
        -------
        Tuple[List[List[str]], List[List[float]]]
            A tuple of two lists, where the first list contains the document
            IDs of the nearest neighbors and the second list contains the
            corresponding similarity scores.
        """
        if not self.built and self.devices:
            options = GpuMultipleClonerOptions()
            options.shard = True
            self.index = index_cpu_to_gpus_list(self.index, options, self.devices)
            self.built = True
        scores, indices = self.index.search(payload, top_k)
        indices = [[self.lookup[i] for i in row] for row in indices]
        return indices, scores.tolist()
