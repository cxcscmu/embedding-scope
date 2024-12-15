"""
The retriever module.

This module implements information retrieval system on the sparse features and
the dense embeddings. The sparse retriever relies on the Elasticsearch engine
while the dense retriever relies on the faiss library.

@author: Hao Kang <haok@andrew.cmu.edu>
@date: December 15, 2024
"""

from pathlib import Path
from source import workspace

workspace = Path(workspace, "retriever")
workspace.mkdir(mode=0o770, exist_ok=True)

from .sparse import Retriever as SparseRetriever

__all__ = ["SparseRetriever"]
