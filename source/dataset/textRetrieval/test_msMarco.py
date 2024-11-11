"""
Test the MsMarcoDataset class.
"""

import torch
from source.embedding.miniCPM import MiniCPM
from source.dataset.textRetrieval import MsMarcoDataset


def test_newPassageLoader():
    """
    Test the newPassageLoader method.
    """
    # Create a new passage loader
    fn = MsMarcoDataset.newPassageLoader
    loader = fn(batchSize=8, shuffle=False, numWorkers=4)

    # Check the first batch
    batch = next(iter(loader))
    assert isinstance(batch, list)
    assert len(batch) == 2

    # Check the unpacked
    pids, passages = batch
    assert isinstance(pids, tuple)
    assert len(pids) == 8
    assert all(isinstance(x, str) for x in pids)
    assert isinstance(passages, tuple)
    assert len(passages) == 8
    assert all(isinstance(x, str) for x in passages)

    # Check a few items
    assert pids[0] == "0"
    assert passages[0].startswith("The presence of communication")
    assert pids[3] == "3"
    assert passages[3].startswith("The Manhattan Project was the name")
    assert pids[6] == "6"
    assert passages[6].startswith("Nor will it attempt to substitute")

    # Check the statistics
    assert len(loader.dataset) == 8841823


def test_newQueryLoader_train():
    """
    Test the newQueryLoader method.
    """
    # Create a new query loader
    fn = MsMarcoDataset.newQueryLoader
    loader = fn("train", batchSize=8, shuffle=False, numWorkers=4)

    # Check the first batch
    batch = next(iter(loader))
    assert isinstance(batch, list)
    assert len(batch) == 2

    # Check the unpacked
    qids, queries = batch
    assert isinstance(qids, tuple)
    assert len(qids) == 8
    assert all(isinstance(x, str) for x in qids)
    assert isinstance(queries, tuple)
    assert len(queries) == 8
    assert all(isinstance(x, str) for x in queries)

    # Check a few items
    assert qids[0] == "121352"
    assert queries[0] == "define extreme"
    assert qids[3] == "510633"
    assert queries[3] == "tattoo fixers how much does it cost"
    assert qids[6] == "674172"
    assert queries[6] == "what is a bank transit number"

    # Check the statistics
    assert len(loader.dataset) == 808731


def test_newQueryLoader_dev():
    """
    Test the newQueryLoader method.
    """
    # Create a new query loader
    fn = MsMarcoDataset.newQueryLoader
    loader = fn("dev", batchSize=8, shuffle=False, numWorkers=4)

    # Check the first batch
    batch = next(iter(loader))
    assert isinstance(batch, list)
    assert len(batch) == 2

    # Check the unpacked
    qids, queries = batch
    assert isinstance(qids, tuple)
    assert len(qids) == 8
    assert all(isinstance(x, str) for x in qids)
    assert isinstance(queries, tuple)
    assert len(queries) == 8
    assert all(isinstance(x, str) for x in queries)

    # Check a few items
    assert qids[0] == "1048578"
    assert queries[0] == "cost of endless pools/swim spa"
    assert qids[3] == "1048581"
    assert queries[3] == "what is pbis?"
    assert qids[6] == "1048584"
    assert queries[6] == "what is pay range for warehouse specialist in minneapolis"

    # Check the statistics
    assert len(loader.dataset) == 101093


def test_newQueryLoader_eval():
    """
    Test the newQueryLoader method.
    """
    # Create a new query loader
    fn = MsMarcoDataset.newQueryLoader
    loader = fn("eval", batchSize=8, shuffle=False, numWorkers=4)

    # Check the first batch
    batch = next(iter(loader))
    assert isinstance(batch, list)
    assert len(batch) == 2

    # Check the unpacked
    qids, queries = batch
    assert isinstance(qids, tuple)
    assert len(qids) == 8
    assert all(isinstance(x, str) for x in qids)
    assert isinstance(queries, tuple)
    assert len(queries) == 8
    assert all(isinstance(x, str) for x in queries)

    # Check a few items
    assert qids[0] == "786436"
    assert queries[0] == "what is prescribed to treat thyroid storm"
    assert qids[3] == "524308"
    assert queries[3] == "treasury routing number"
    assert qids[6] == "786472"
    assert queries[6] == "what is president trump's twitter name"

    # Check the statistics
    assert len(loader.dataset) == 101092


def test_newQueryEmbeddingLoader_train():
    """
    Test the newQueryEmbeddingLoader method.
    """
    # Create a new query embedding loader
    fn = MsMarcoDataset.newQueryEmbeddingLoader
    loader = fn(MiniCPM, "train", batchSize=4, shuffle=False, numWorkers=4)

    # Check the first batch
    batch = next(iter(loader))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (4, MiniCPM.size)

    # Check the numerical values
    embedding = MiniCPM(devices=[0])
    queries = ["Query: define extreme"]
    vectors = torch.from_numpy(embedding.forward(queries))
    assert torch.allclose(batch[0], vectors[0], atol=1e-3)


def test_newQueryEmbeddingLoader_dev():
    """
    Test the newQueryEmbeddingLoader method.
    """
    # Create a new query embedding loader
    fn = MsMarcoDataset.newQueryEmbeddingLoader
    loader = fn(MiniCPM, "dev", batchSize=4, shuffle=False, numWorkers=4)

    # Check the first batch
    batch = next(iter(loader))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (4, MiniCPM.size)

    # Check the numerical values
    embedding = MiniCPM(devices=[0])
    queries = ["Query: cost of endless pools/swim spa"]
    vectors = torch.from_numpy(embedding.forward(queries))
    assert torch.allclose(batch[0], vectors[0], atol=1e-3)


def test_newQueryEmbeddingLoader_eval():
    """
    Test the newQueryEmbeddingLoader method.
    """
    # Create a new query embedding loader
    fn = MsMarcoDataset.newQueryEmbeddingLoader
    loader = fn(MiniCPM, "eval", batchSize=4, shuffle=False, numWorkers=4)

    # Check the first batch
    batch = next(iter(loader))
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (4, MiniCPM.size)

    # Check the numerical values
    embedding = MiniCPM(devices=[0])
    queries = ["Query: what is prescribed to treat thyroid storm"]
    vectors = torch.from_numpy(embedding.forward(queries))
    assert torch.allclose(batch[0], vectors[0], atol=1e-3)
