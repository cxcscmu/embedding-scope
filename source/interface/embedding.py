"""
Specify the embedding interface.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
from numpy.typing import NDArray


class Embedding(ABC):
    """
    Base class for all embeddings.

    Attributes:
        name: The name of the embedding.
        size: The size of the embedding vector.
    """

    name: str
    size: int


class TextEmbedding(Embedding):
    """
    Base class for text embeddings.
    """

    @abstractmethod
    def __init__(self, devices: List[int]):
        """
        Initialize the TextEmbedding with the given devices.

        :param devices: List of device IDs to use for computation.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, texts: List[str]) -> NDArray[np.float32]:
        """
        Compute the embeddings for the given documents.

        :param texts: List of texts to embed.
        :return: The embeddings for the given texts.
        """
        raise NotImplementedError
