"""
Specify the retriever interface.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray


class DenseRetriever(ABC):
    """
    Base class for dense retrievers.
    """

    @abstractmethod
    def __init__(self, size: int):
        """
        Initialize the retriever.

        :param size: The size of the vectors.
        """
        raise NotImplementedError

    @abstractmethod
    def add(self, ids: List[str], vectors: NDArray[np.float32]):
        """
        Add the given vectors to the retriever.

        :param ids: List of vector IDs.
        :param vectors: The vectors to add.
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self, vectors: NDArray[np.float32], topK: int
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Search for the most similar vectors to the given query vectors.

        :param vectors: The query vectors.
        :param topK: The number of most similar vectors to return.
        :return: A tuple containing the IDs and scores of the most similar vectors.
        """
        raise NotImplementedError
