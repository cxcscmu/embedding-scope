"""
Unit tests for the FaissRetriever class.
"""

import numpy as np
from source.retriever.faiss import FaissRetriever


def test_faissRetriever_addSearch():
    """
    Test the add and search methods of the FaissRetriever class.
    """
    size, devices = 64, [0, 1]
    retriever = FaissRetriever(size, devices)

    # Add three vectors to the retriever.
    ids = ["0", "1", "2"]
    vectors = np.random.rand(len(ids), size)
    retriever.add(ids, vectors)

    # Search for the top-2 vectors given five queries.
    queries, topK = np.random.rand(5, size), 2
    results, scores = retriever.search(queries, topK)

    # Check the results.
    assert isinstance(results, list)
    assert all(isinstance(row, list) for row in results)
    assert all(len(row) == topK for row in results)
    assert all(isinstance(item, str) for row in results for item in row)

    # Check the scores.
    assert isinstance(scores, list)
    assert all(isinstance(row, list) for row in scores)
    assert all(len(row) == topK for row in scores)
    assert all(isinstance(item, float) for row in scores for item in row)
