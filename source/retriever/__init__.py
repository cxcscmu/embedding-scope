"""
The retriever module.

This module implements the information retrieval system on the sparse features
and the dense embeddings. The sparse retriever relies on the Elasticsearch
engine while the dense retriever relies on the faiss library.
"""

from pathlib import Path
from source import workspace

workspace = Path(workspace, "retriever")
workspace.mkdir(mode=0o770, exist_ok=True)
