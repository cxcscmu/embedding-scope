"""
Test the DotProductRetriever class.
"""

import numpy as np
from source.retriever.dense import DotProductRetriever


def test_addSearch():
    """
    Test the add and search methods of the FaissRetriever class.
    """
    # Create a new retriever.
    retriever = DotProductRetriever(64, [0])
    vectors = np.random.rand(8, 64)
    retriever.add(vectors)

    # Perform a search.
    queries, topK = np.random.rand(5, 64), 2
    results, scores = retriever.search(queries, topK)

    # Check the results.
    assert isinstance(results, list)
    assert all(isinstance(row, list) for row in results)
    assert all(len(row) == topK for row in results)
    assert all(isinstance(item, int) for row in results for item in row)

    # Check the scores.
    assert isinstance(scores, list)
    assert all(isinstance(row, list) for row in scores)
    assert all(len(row) == topK for row in scores)
    assert all(isinstance(item, float) for row in scores for item in row)
