"""
Test the MsMarcoDataset class.
"""

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
    pids, passages = batch
    assert pids[0] == "0"
    assert passages[0].startswith("The presence of communication")
    assert pids[3] == "3"
    assert passages[3].startswith("The Manhattan Project was the name")
    assert pids[6] == "6"
    assert passages[6].startswith("Nor will it attempt to substitute")

    # Check the statistics
    assert len(loader.dataset) == 8841823
