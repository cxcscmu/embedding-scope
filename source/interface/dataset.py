"""
Specify the dataset interface.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Literal, Type, Dict, List
import numpy as np
from numpy import ndarray as NDArray
from source.interface import TextEmbedding


class Dataset(ABC):
    """
    Base class for all datasets.

    Attributes:
        name: The name of the dataset.
    """

    name: str


PartitionType = Literal["train", "dev"]


class TextRetrievalDataset(Dataset):
    """
    Base class for text retrieval datasets.
    """

    @abstractmethod
    def getPassages(self) -> Iterator[Tuple[str, str]]:
        """
        Get the passages in the dataset.

        :return: Iterator over passage IDs and texts.
        """

    @abstractmethod
    def getPassageEmbeddings(
        self, embedding: Type[TextEmbedding]
    ) -> Iterator[Tuple[str, NDArray[np.float32]]]:
        """
        Get the embeddings of the passages in the dataset.

        :param embedding: The embedding to use.
        :return: Iterator over passage IDs and embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def getQueries(self, partition: PartitionType) -> Iterator[Tuple[str, str]]:
        """
        Get the queries in the dataset.

        :param partition: The partition to get queries from.
        :return: Iterator over query IDs and texts.
        """
        raise NotImplementedError

    @abstractmethod
    def getQueryEmbeddings(
        self, partition: PartitionType, embedding: Type[TextEmbedding]
    ) -> Iterator[Tuple[str, NDArray[np.float32]]]:
        """
        Get the embeddings of the queries in the dataset.

        :param partition: The partition to get queries from.
        :param embedding: The embedding to use.
        :return: Iterator over query IDs and embeddings.
        """
        raise NotImplementedError

    @abstractmethod
    def getRelevantPassages(
        self, partition: PartitionType
    ) -> Dict[str, Dict[str, int]]:
        """
        Get the relevant passages for each query in the dataset.

        :param partition: The partition to get relevant passages from.
        :return: Mapping from query IDs to mapping from passage IDs to relevance.
        """
        raise NotImplementedError

    @abstractmethod
    def getNeighborPassages(
        self, partition: PartitionType, k: int
    ) -> Dict[str, List[str]]:
        """
        Get the k nearest neighbor passages for each query in the dataset.

        :param partition: The partition to get neighbor passages from.
        :param k: The number of neighbor passages to get.
        :return: Mapping from query IDs to nearest passage IDs in descending order.
        """
        raise NotImplementedError
