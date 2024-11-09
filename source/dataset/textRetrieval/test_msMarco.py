"""
Test the implementation for the MS MARCO dataset.
"""

import torch
from source.embedding.miniCPM import MiniCPM
from source.dataset.textRetrieval.msMarco import MsMarcoDataset


def test_newPassageLoader():
    """
    Test the newPassageLoader method.
    """
    loader = MsMarcoDataset.newPassageLoader(256, True, 2)
    pids, passages = next(iter(loader))

    # Check the passage IDs.
    assert isinstance(pids, tuple)
    assert all(isinstance(pid, str) for pid in pids)

    # Check the passages.
    assert isinstance(passages, tuple)
    assert all(isinstance(passage, str) for passage in passages)

    # Check their lengths.
    assert len(pids) == len(passages)


def test_newPassageEmbeddingLoader():
    """
    Test the newPassageEmbeddingLoader method.
    """
    loader = MsMarcoDataset.newPassageEmbeddingLoader(MiniCPM, 256, True, 2)
    passageEmbeddings = next(iter(loader))

    # Check the passage embeddings.
    assert isinstance(passageEmbeddings, torch.Tensor)
    assert passageEmbeddings.shape == (256, MiniCPM.size)
    assert passageEmbeddings.dtype == torch.float32


def test_newQueryLoader():
    """
    Test the newQueryLoader method.
    """
    loader = MsMarcoDataset.newQueryLoader("train", 256, True, 2)
    qids, queries = next(iter(loader))

    # Check the query IDs.
    assert isinstance(qids, tuple)
    assert all(isinstance(qid, str) for qid in qids)

    # Check the queries.
    assert isinstance(queries, tuple)
    assert all(isinstance(query, str) for query in queries)

    # Check their lengths.
    assert len(qids) == len(queries)


def test_newQueryEmbeddingLoader():
    """
    Test the newQueryEmbeddingLoader method.
    """
    loader = MsMarcoDataset.newQueryEmbeddingLoader(MiniCPM, "train", 256, True, 2)
    queryEmbeddings = next(iter(loader))

    # Check the query embeddings.
    assert isinstance(queryEmbeddings, torch.Tensor)
    assert queryEmbeddings.shape == (256, MiniCPM.size)
    assert queryEmbeddings.dtype == torch.float32


def test_getQueryRelevance():
    """
    Test the getRelevance method.
    """
    relevance = MsMarcoDataset.getQueryRelevance("train")

    # Check the relevance.
    assert isinstance(relevance, dict)
    queryID, payload = next(iter(relevance.items()))
    assert isinstance(queryID, str)
    assert isinstance(payload, dict)
    passageID, score = next(iter(payload.items()))
    assert isinstance(passageID, str)
    assert isinstance(score, int)
