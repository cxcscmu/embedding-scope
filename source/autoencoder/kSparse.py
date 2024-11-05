"""
Implementation of the KSparseAutoencoder.
"""

from typing import Tuple
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from source.interface import AutoEncoder


class KSparseAutoencoder(AutoEncoder, nn.Module):
    """
    Implementation of the KSparseAutoencoder.
    """

    def __init__(self, vectorSize: int, latentSize: int, topK: int) -> None:
        """
        Initialize the KSparseAutoencoder.

        :param vectorSize: The size of the input vectors.
        :param latentSize: The size of the latent features.
        :param topK: The number of non-zero elements in the latent features.
        """
        nn.Module.__init__(self)
        self.encoder = nn.Linear(vectorSize, latentSize)
        self.decoder = nn.Linear(latentSize, vectorSize)
        self.topK = topK

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        xbar = x - self.decoder.bias
        a = self.encoder.forward(xbar)
        pack = torch.topk(a, self.topK)
        f = torch.zeros_like(a)
        f.scatter_(1, pack.indices, F.relu(pack.values))
        xhat = self.decoder.forward(f)
        return f, xhat
