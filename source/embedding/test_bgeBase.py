"""
Unit tests for the BgeBase class.
"""

import pytest
import numpy as np
from source.embedding.bgeBase import BgeBase


def test_name():
    """
    Test the name of the BgeBase class.
    """
    assert BgeBase.name == "bgeBase"


def test_size():
    """
    Test the size of the BgeBase class.
    """
    assert BgeBase.size == 768


@pytest.fixture(name="setup")
def setup_fixture():
    """
    Create an instance of the BgeBase class.
    """
    return BgeBase(devices=[0])


def test_forward(setup: BgeBase):
    """
    Test the forward method of the BgeBase class.
    """
    texts = ["This is not a test", "This isn't a test"]
    vectors = setup.forward(texts)
    assert vectors.shape == (2, 768) and vectors.dtype == np.float32
    assert np.all(np.isfinite(vectors))
