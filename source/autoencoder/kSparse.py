"""
Implement the k-sparce autoencoder.
"""

from typing import Tuple
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from source.interface.autoencoder import AutoEncoder


class KSparseAutoencoder(AutoEncoder, nn.Module):
    """
    Implementation of the k-sparse autoencoder.
    """

    def __init__(self, vectorSize: int, latentSize: int, latentTopK: int):
        """
        Initialize the autoencoder.

        :param vectorSize: The size of the input vectors.
        :param latentSize: The size of the latent features.
        :param latentTopK: The number of top-k features to keep.
        """
        nn.Module.__init__(self)
        self.encoder = nn.Linear(vectorSize, latentSize)
        self.decoder = nn.Linear(latentSize, vectorSize)
        self.latentTopK = latentTopK

    def decode(self, f: Tensor) -> Tensor:
        return self.decoder.forward(f)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        xbar = x - self.decoder.bias
        a = self.encoder.forward(xbar)
        pack = torch.topk(a, self.latentTopK)
        f = torch.zeros_like(a)
        f.scatter_(1, pack.indices, F.relu(pack.values))
        xhat = self.decoder.forward(f)
        return f, xhat
