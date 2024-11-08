"""
Unit tests for the MiniCPM class.
"""

import pytest
import numpy as np
from source.embedding.miniCPM import MiniCPM


def test_name():
    """
    Test the name of the MiniCPM class.
    """
    assert MiniCPM.name == "miniCPM"


def test_size():
    """
    Test the size of the MiniCPM class.
    """
    assert MiniCPM.size == 2304


@pytest.fixture(name="setup")
def setup_fixture():
    """
    Create an instance of the MiniCPM class.
    """
    return MiniCPM(devices=[0])


def test_forward(setup: MiniCPM):
    """
    Test the forward method of the MiniCPM class.
    """
    texts = ["This is not a test", "This isn't a test"]
    vectors = setup.forward(texts)
    assert vectors.shape == (2, 2304) and vectors.dtype == np.float32
    assert np.all(np.isfinite(vectors))
