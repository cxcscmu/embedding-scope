"""
Specify the autoencoder interface.
"""

from typing import Tuple
from abc import ABC, abstractmethod
from torch import Tensor


class AutoEncoder(ABC):
    """
    Base class for all autoencoders.
    """

    @abstractmethod
    def __init__(self, vectorSize: int, latentSize: int) -> None:
        """
        Initialize the autoencoder.

        :param vectorSize: The size of the input vectors.
        :param latentSize: The size of the latent features.
        """

    @abstractmethod
    def decode(self, f: Tensor) -> Tensor:
        """
        Decode the latent features.

        :param f: The latent features.
        :return: The reconstructed tensor.
        """

    @abstractmethod
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the autoencoder.

        :param x: The original tensor.
        :return: The latent features and the reconstructed tensor.
        """
