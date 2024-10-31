"""
Unit tests for the MiniCPM class.
"""

import pytest
import numpy as np
from src.interface import TextEmbedding
from src.embedding.miniCPM import MiniCPM


def test_miniCPM_getName():
    assert MiniCPM.name == "MiniCPM"


def test_miniCPM_getSize():
    assert MiniCPM.size == 2304


@pytest.fixture
def embedding():
    return MiniCPM(devices=[0])


def test_miniCPM_forward(embedding: TextEmbedding):
    texts = ["This is not a test", "This isn't a test"]
    vectors = embedding.forward(texts)
    assert vectors.shape == (2, 2304) and vectors.dtype == np.float32
    assert np.all(np.isfinite(vectors))
