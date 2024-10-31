"""
Unit tests for the BgeBase class.
"""

import pytest
import numpy as np
from src.interface import TextEmbedding
from src.embedding.bgeBase import BgeBase


def test_bgeBase_getName():
    assert BgeBase.name == "BgeBase"


def test_bgeBase_getSize():
    assert BgeBase.size == 768


def test_bgeBase_getType():
    assert BgeBase.type == np.float32


@pytest.fixture
def embedding():
    return BgeBase(devices=[0])


def test_bgeBase_forward(embedding: TextEmbedding):
    documents = ["This is not a test", "This isn't a test"]
    vectors = embedding.forward(documents)
    assert vectors.shape == (2, 768) and vectors.dtype == np.float32
    assert np.all(np.isfinite(vectors))
