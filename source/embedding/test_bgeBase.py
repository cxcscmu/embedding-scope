"""
Unit tests for the BgeBase class.
"""

import pytest
import numpy as np
from source.interface import TextEmbedding
from source.embedding.bgeBase import BgeBase


def test_bgeBase_name():
    """
    Test the name of the BgeBase class.
    """
    assert BgeBase.name == "BgeBase"


def test_bgeBase_size():
    """
    Test the size of the BgeBase class.
    """
    assert BgeBase.size == 768


@pytest.fixture()
def setup():
    """
    Create an instance of the BgeBase class.
    """
    return BgeBase(devices=[0])


def test_bgeBase_forward(embedding: TextEmbedding):
    """
    Test the forward method of the BgeBase class.
    """
    texts = ["This is not a test", "This isn't a test"]
    vectors = embedding.forward(texts)
    assert vectors.shape == (2, 768) and vectors.dtype == np.float32
    assert np.all(np.isfinite(vectors))
