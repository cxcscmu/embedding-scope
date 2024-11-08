"""
Specify the dataset interface.
"""

from abc import ABC, abstractmethod
from typing import Type, Literal, Dict
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
        partition: PartitionType,
        embedding: Type[TextEmbedding],
        batchSize: int,
        shuffle: bool,
        numWorkers: int,
    ) -> DataLoader:
        """
        Create a new query embedding loader.

        :param partition: The partition.
        :param embedding: The embedding to use.
        :param batchSize: The batch size.
        :param shuffle: Whether to shuffle the data.
        :param numWorkers: The number of workers.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def getRelevance(partition: PartitionType) -> Dict[str, Dict[str, int]]:
        """
        Get the relevance judgments.

        :param partition: The partition.
        :return: Mapping from query ID to mapping from passage ID to relevance.
        """
        raise NotImplementedError
