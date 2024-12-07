"""
Test the sparse retriever.
"""

from source.retriever.sparse import Retriever


def test_similarity():
    """
    Test the sparse retriever, which uses dot product for similarity.
    """
    with Retriever() as retriever:
        retriever.batch_index(
            {
                "1": {"a": 1.0, "b": 2.0},
                "2": {"a": 2.0, "b": 3.0},
            }
        )
        result = retriever.batch_query(
            [{"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 3.0}],
            top_k=2,
        )
        assert result == [["2", "1"], ["2", "1"]]
