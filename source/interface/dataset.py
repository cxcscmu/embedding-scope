"""
Specify the dataset interface.
"""

from abc import ABC, abstractmethod
from typing import Type, Literal, Dict, List, Tuple
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
    ) -> List[Tuple[List[int], List[float]]]:
        """
        Get the query nearest neighbors using the embedding.
        Depending on the dataset, the number of neighbors may vary.

        :param embedding: The embedding to use.
        :param partition: The partition.
        :return: Mapping from query offset to ordered mapping from passage offset to similarity.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def newMixEmbeddingLoader(
        embedding: Type[TextEmbedding],
        partition: PartitionType,
        numPassages: int,
        batchSize: int,
        shuffle: bool,
        numWorkers: int,
    ) -> DataLoader:
        """
        Create a new loader over queries and their nearest neighbors.

        :param embedding: The embedding to use.
        :param partition: The partition.
        :param numPassages: The number of passages to include.
        :param batchSize: The batch size.
        :param shuffle: Whether to shuffle the data.
        :param numWorkers: The number of workers.
        """
        raise NotImplementedError
