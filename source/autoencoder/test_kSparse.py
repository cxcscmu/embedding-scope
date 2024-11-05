"""
Unit tests for the KSparseAutoencoder class.
"""

import torch
from source.autoencoder.kSparse import KSparseAutoencoder


def test_kSparseAutoencoder_forward():
    """
    Test the forward method of the KSparseAutoencoder class.
    """
    x = torch.rand(8, 32, device="cuda")
    autoencoder = KSparseAutoencoder(32, 16, 4).cuda()
    f, xhat = autoencoder.forward(x)

    # Check the latent features.
    assert f.shape == (8, 16)
    assert torch.nonzero(f).shape == (8 * 4, 2)

    # Check the reconstructed vectors.
    assert xhat.shape == (8, 32)
