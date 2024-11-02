"""
Unit tests for the MsMarco class.
"""

import pytest
from source.interface import TextRetrievalDataset
from source.dataset.msMarco import MsMarco


def test_msMarco_name():
    """
    Test the name of the MsMarco class.
    """
    assert MsMarco.name == "MsMarco"


@pytest.fixture(name="dataset")
def dataset():
    """
    Create an instance of the MsMarco class.
    """
    return MsMarco()


def test_msMarco_getPassages(dataset: TextRetrievalDataset):
    """
    Test the getPassages method of the MsMarco class.
    """
    passages = dataset.getPassages()
    item = next(passages)
    raise NotImplementedError(item)
    # assert len(passages) == 1
