"""
Specify the dataset interface.
"""

from abc import ABC, abstractmethod
from typing import Type, Literal, Dict
from collections import OrderedDict
from torch.utils.data import DataLoader
from source.interface.embedding import TextEmbedding


PartitionType = Literal["train", "dev", "eval"]


class TextRetrievalDataset(ABC):
    """
    Base class for text retrieval datasets.
    """

    @staticmethod
    @abstractmethod
    def newPassageLoader(batchSize: int, shuffle: bool, numWorkers: int) -> DataLoader:
        """
        Create a new passage loader.

        :param batchSize: The batch size.
        :param shuffle: Whether to shuffle the data.
        :param numWorkers: The number of workers.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def newPassageEmbeddingLoader(
        embedding: Type[TextEmbedding], batchSize: int, shuffle: bool, numWorkers: int
    ) -> DataLoader:
        """
        Create a new passage embedding loader.

        :param embedding: The embedding to use.
        :param batchSize: The batch size.
        :param shuffle: Whether to shuffle the data.
        :param numWorkers: The number of workers.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def newQueryLoader(
        partition: PartitionType, batchSize: int, shuffle: bool, numWorkers: int
    ) -> DataLoader:
        """
        Create a new query loader.

        :param partition: The partition.
        :param batchSize: The batch size.
        :param shuffle: Whether to shuffle the data.
        :param numWorkers: The number of workers.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def newQueryEmbeddingLoader(
        embedding: Type[TextEmbedding],
        partition: PartitionType,
        batchSize: int,
        shuffle: bool,
        numWorkers: int,
    ) -> DataLoader:
        """
        Create a new query embedding loader.

        :param embedding: The embedding to use.
        :param partition: The partition.
        :param batchSize: The batch size.
        :param shuffle: Whether to shuffle the data.
        :param numWorkers: The number of workers.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def getQueryRelevance(partition: PartitionType) -> Dict[str, Dict[str, int]]:
        """
        Get the query relevance judgments.

        :param partition: The partition.
        :return: Mapping from query ID to mapping from passage ID to relevance.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def getQueryNeighbors(
        embedding: Type[TextEmbedding], partition: PartitionType
    ) -> Dict[str, OrderedDict[str, float]]:
        """
        Get the query nearest neighbors using the embedding.

        :param embedding: The embedding to use.
        :param partition: The partition.
        :return: Mapping from query ID to mapping from passage ID to similarity.
        """
        raise NotImplementedError
