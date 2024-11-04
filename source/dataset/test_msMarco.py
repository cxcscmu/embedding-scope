"""
Unit tests for the MsMarco class.
"""

import pytest
import numpy as np
from source.dataset.msMarco import MsMarco
from source.embedding.miniCPM import MiniCPM


def test_msMarco_name():
    """
    Test the name of the MsMarco class.
    """
    assert MsMarco.name == "MsMarco"


@pytest.fixture(name="setup")
def setup_fixture():
    """
    Create an instance of the MsMarco class.
    """
    return MsMarco()


def test_msMarco_getPassages(setup: MsMarco):
    """
    Test the getPassages method of the MsMarco class.
    """
    passageID, passageText = next(setup.getPassages())
    assert isinstance(passageID, str)
    assert passageID == "0"
    assert isinstance(passageText, str)
    assert passageText.startswith("The presence of communication")
    assert passageText.endswith("innocent lives obliterated.")


def test_msMarco_getPassageEmbeddings(setup: MsMarco):
    """
    Test the getPassageEmbeddings method of the MsMarco class.
    """
    passageID, passageEmbedding = next(setup.getPassageEmbeddings(MiniCPM))
    assert isinstance(passageID, str)
    assert passageID == "0"
    assert isinstance(passageEmbedding, np.ndarray)
    assert passageEmbedding.shape == (MiniCPM.size,)


def test_msMarco_getQueries(setup: MsMarco):
    """
    Test the getQueries method of the MsMarco class.
    """
    queryID, queryText = next(setup.getQueries("train"))
    assert isinstance(queryID, str)
    assert queryID == "121352"
    assert isinstance(queryText, str)
    assert queryText == "define extreme"
    queryID, queryText = next(setup.getQueries("dev"))
    assert isinstance(queryID, str)
    assert queryID == "1048578"
    assert isinstance(queryText, str)
    assert queryText == "cost of endless pools/swim spa"


def test_msMarco_getQueryEmbeddings(setup: MsMarco):
    """
    Test the getQueryEmbeddings method of the MsMarco class.
    """
    queryID, queryEmbedding = next(setup.getQueryEmbeddings("train", MiniCPM))
    assert isinstance(queryID, str)
    assert queryID == "121352"
    assert isinstance(queryEmbedding, np.ndarray)
    assert queryEmbedding.shape == (MiniCPM.size,)
    queryID, queryEmbedding = next(setup.getQueryEmbeddings("dev", MiniCPM))
    assert isinstance(queryID, str)
    assert queryID == "1048578"
    assert isinstance(queryEmbedding, np.ndarray)
    assert queryEmbedding.shape == (MiniCPM.size,)
