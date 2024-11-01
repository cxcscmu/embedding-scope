"""
Unit tests for the MiniCPM class.
"""

import pytest
import numpy as np
from source.interface import TextEmbedding
from source.embedding.miniCPM import MiniCPM


def test_miniCPM_name():
    """
    Test the name of the MiniCPM class.
    """
    assert MiniCPM.name == "MiniCPM"


def test_miniCPM_size():
    """
    Test the size of the MiniCPM class.
    """
    assert MiniCPM.size == 2304


@pytest.fixture
def embedding():
    """
    Create an instance of the MiniCPM class.
    """
    return MiniCPM(devices=[0])


def test_miniCPM_forward(model: TextEmbedding):
    """
    Test the forward method of the MiniCPM class.
    """
    texts = ["This is not a test", "This isn't a test"]
    vectors = model.forward(texts)
    assert vectors.shape == (2, 2304) and vectors.dtype == np.float32
    assert np.all(np.isfinite(vectors))
