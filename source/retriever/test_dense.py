"""
Test the dense retriever.

@author: Hao Kang <haok@andrew.cmu.edu>
@date: December 27, 2024
"""

import numpy as np

from source.retriever.dense import Retriever


def test_basic():
    """
    Test the index and query functions.
    """
    with Retriever(size=2, devices=[0]) as retriever:
        retriever.batch_index(
            {
                "1": np.array([1.0, 2.0]),
                "2": np.array([2.0, 3.0]),
            }
        )
        indices, scores = retriever.batch_query(
            np.stack([np.array([1.0, 2.0]), np.array([2.0, 3.0])]),
            top_k=2,
        )
        assert indices == [
            ["2", "1"],
            ["2", "1"],
        ]
        assert scores == [
            [2.0 * 1.0 + 3.0 * 2.0, 1.0 * 1.0 + 2.0 * 2.0],
            [2.0 * 2.0 + 3.0 * 3.0, 1.0 * 2.0 + 2.0 * 3.0],
        ]
