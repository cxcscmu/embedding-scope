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


@pytest.fixture(name="setup")
def setup_fixture():
    """
    Create an instance of the MsMarco class.
    """
    return MsMarco()


def test_msMarco_getPassages(setup: TextRetrievalDataset):
    """
    Test the getPassages method of the MsMarco class.
    """
    passageID, passageText = next(setup.getPassages())
    assert isinstance(passageID, str)
    assert passageID == "0"
    assert isinstance(passageText, str)
    assert passageText.startswith("The presence of communication")
    assert passageText.endswith("innocent lives obliterated.")


def test_msMarco_getQueries(setup: TextRetrievalDataset):
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
