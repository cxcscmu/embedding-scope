"""
Specifies the dataset interface.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Type
import numpy as np
from numpy.typing import NDArray
from source.interface.embedding import TextEmbedding


class Dataset(ABC):
    """
    Base class for all datasets.

    Attributes:
        name: The name of the dataset.
    """

    name: str


class TextRetrievalDataset(Dataset):
    """
    Base class for text retrieval datasets.
    """

    @abstractmethod
    def getPassages(self) -> List[Tuple[str, str]]:
        """
        Get the passages in the dataset.

        :return: List of passage IDs and texts.
        """
        raise NotImplementedError

    @abstractmethod
    def getPassageEmbeddings(
        self, embedding: Type[TextEmbedding]
    ) -> List[Tuple[str, NDArray[np.float32]]]:
        """
        Get the embeddings of the passages in the dataset.

        :param embedding: The embedding to use.
        :return: List of passage IDs and embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def getQueries(self, partition: str) -> List[Tuple[str, str]]:
        """
        Get the queries in the dataset.

        :param partition: The partition to get queries from.
        :return: List of query IDs and texts.
        """
        raise NotImplementedError

    @abstractmethod
    def getQueryEmbeddings(
        self, partition: str, embedding: Type[TextEmbedding]
    ) -> List[Tuple[str, NDArray[np.float32]]]:
        """
        Get the embeddings of the queries in the dataset.

        :param partition: The partition to get queries from.
        :param embedding: The embedding to use.
        :return: List of query IDs and embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def getRelevantPassages(self, queryId: str) -> List[Tuple[str, int]]:
        """
        Get the relevant passages for the given query.

        :param queryId: The ID of the query.
        :return: List of passage IDs and relevance labels.
        """
        raise NotImplementedError

    @abstractmethod
    def getNeighborPassages(
        self, queryId: str, embedding: Type[TextEmbedding], k: int
    ) -> List[str]:
        """
        Get the neighbor passages for the given query. This is similar to
        getRelevancePassages, but instead of passages based on the relevance
        labels collected by the dataset, this method returns the k-nearest
        neighbors based on the embedding.

        :param queryId: The ID of the query.
        :param embedding: The embedding to use.
        :param k: The number of neighbors to return.
        :return: List of passage IDs.
        """
        raise NotImplementedError
