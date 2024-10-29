"""
@brief: Specify the project interface.
@author: Hao Kang <haok@andrew.cmu.edu>
"""

from pathlib import Path
from abc import ABC, abstractmethod
from typing import Iterator, Literal, List, Type, Tuple
from torch import Tensor


# Define the type aliases.
EmbeddingName = Literal["BgeBase", "MiniCPM"]
DatasetName = Literal["MsMarco", "Beir"]
PartitionName = Literal["Train", "Validate"]


class Embedding(ABC):
    """
    The interface for an embedding model.

    Attributes:
        name (EmbeddingName): The name of the embedding.
        size (int): The size of the embedding.
    """

    name: EmbeddingName
    size: int

    @abstractmethod
    def __init__(self, devices: List[int]) -> None:
        """
        Initialize the embedding model.

        :type devices: List[int]
        :param devices: The devices to use for embedding.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, passages: List[str]) -> Tensor:
        """
        Forward pass to embed the given passages.

        :type passages: List[str]
        :param passages: The list of passages to embed.
        :rtype: torch.Tensor
        :return: The computed embeddings in a tensor of shape (N, D), where N
            is the number of passages and D is the embedding size.
        """
        raise NotImplementedError


class Dataset(ABC):
    """
    The interface for a dataset.

    Attributes:
        name (DatasetName): The name of the dataset.
    """

    name: DatasetName

    @abstractmethod
    def didIter(self, batchSize: int) -> Iterator[List[str]]:
        """
        Iterate over the document IDs in batches.

        :type batchSize: int
        :param batchSize: The batch size for each iteration.
        :rtype: Iterator[List[str]]
        :return: An iterator over the document IDs. Each iteration yields a
            list of document IDs.
        """
        raise NotImplementedError

    @abstractmethod
    def docIter(self, batchSize: int) -> Iterator[List[str]]:
        """
        Iterate over the document texts in batches.

        :type batchSize: int
        :param batchSize: The batch size for each iteration.
        :rtype: Iterator[List[str]]
        :return: The iterator over the document texts. Each iteration yields a
            list of document texts.
        """
        raise NotImplementedError

    @abstractmethod
    def docEmbIter(
        self,
        embedding: Type[Embedding],
        batchSize: int,
        numWorkers: int,
        shuffle: bool,
    ) -> Iterator[Tensor]:
        """
        Iterate over the document embeddings in batches.

        :type embedding: Type[Embedding]
        :param embedding: The embedding model to use.
        :type batchSize: int
        :param batchSize: The batch size for each iteration.
        :type numWorkers: int
        :param numWorkers: The number of workers for data loading.
        :type shuffle: bool
        :param shuffle: Whether to shuffle the data during loading.
        :rtype: Iterator[Tensor]
        :return: The iterator over the document embeddings. Each tensor has
            shape (N, D), where N is the batch size, or less for the last
            batch, and D is the embedding size.
        """
        raise NotImplementedError

    @abstractmethod
    def getDocLen(self) -> int:
        """
        Get the number of documents.

        :rtype: int
        :return: The number of documents.
        """
        raise NotImplementedError

    @abstractmethod
    def qidIter(
        self, split: PartitionName, batchSize: int
    ) -> Iterator[List[str]]:
        """
        Iterate over the query IDs in batches.

        :type split: PartitionName
        :param split: Whether to use the training or validation split.
        :type batchSize: int
        :param batchSize: The batch size for each iteration.
        :rtype: Iterator[List[str]]
        :return: The iterator over the query IDs. Each iteration yields a list
            of query IDs.
        """
        raise NotImplementedError

    @abstractmethod
    def qryIter(
        self, split: PartitionName, batchSize: int
    ) -> Iterator[List[str]]:
        """
        Iterate over the query texts in batches.

        :type split: PartitionName
        :param split: Whether to use the training or validation split.
        :type batchSize: int
        :param batchSize: The batch size for each iteration.
        :rtype: Iterator[List[str]]
        :return: The iterator over the query texts. Each iteration yields a
            list of query texts.
        """
        raise NotImplementedError

    @abstractmethod
    def qryEmbIter(
        self,
        split: PartitionName,
        embedding: Type[Embedding],
        batchSize: int,
        numWorkers: int,
        shuffle: bool,
    ) -> Iterator[Tensor]:
        """
        Iterate over the query embeddings in batches.

        :type split: PartitionName
        :param split: Whether to use the training or validation split.
        :type embedding: Type[Embedding]
        :param embedding: The embedding class to use.
        :type batchSize: int
        :param batchSize: The batch size for each iteration.
        :type numWorkers: int
        :param numWorkers: The number of workers for data loading.
        :type shuffle: bool
        :param shuffle: Whether to shuffle the data.
        :rtype: Iterator[Tensor]
        :return: The iterator over the query embeddings. Each tensor has shape
            (N, D), where N is the batch size, or less for the last batch, and
            D is the embedding size.
        """
        raise NotImplementedError

    @abstractmethod
    def getQryLen(self, split: PartitionName) -> int:
        """
        Get the number of queries.

        :type split: PartitionName
        :param split: Whether to use the training or validation split.
        :rtype: int
        :return: The number of queries.
        """
        raise NotImplementedError

    @abstractmethod
    def getQryRel(self, split: PartitionName) -> Path:
        """
        Get the path to the query relevance file.

        :type split: PartitionName
        :param split: Whether to use the training or validation split.
        :rtype: Path
        :return: The path to the query relevance file.
        """

    @abstractmethod
    def mixEmbIter(
        self,
        split: PartitionName,
        embedding: Type[Embedding],
        relevant: int,
        batchSize: int,
        numWorkers: int,
        shuffle: bool,
    ) -> Iterator[Tuple[Tensor, Tensor]]:
        """
        Iterate over the embeddings of query and its retrieved documents in
            batches.

        :type split: PartitionName
        :param split: Whether to use the training or validation split.
        :type embedding: Type[Embedding]
        :param embedding: The embedding class to use.
        :type relevant: int
        :param relevant: The number of documents to include for each query.
        :type batchSize: int
        :param batchSize: The batch size for each iteration.
        :type numWorkers: int
        :param numWorkers: The number of workers for data loading.
        :type shuffle: bool
        :param shuffle: Whether to shuffle the data.
        :rtype: Iterator[Tuple[Tensor, Tensor]]
        :return: The iterator over the query and document embeddings. The
            first tensor is the query embeddings and has shape (N, D), where N
            is the batch size, or less for the last batch, and D is the
            embedding size. The second tensor is the document embeddings and
            has shape (N, K, D), where K is the number of relevant documents.
        """
        raise NotImplementedError

    @abstractmethod
    def getMixLen(self, split: PartitionName) -> int:
        """
        Get the number of query-document pairs.
        This function is equival to getQryLen.

        :type split: PartitionName
        :param split: Whether to use the training or validation split.
        :rtype: int
        :return: The number of query-document pairs.
        """
        raise NotImplementedError


class SAE(ABC):
    """
    The interface for a sparse autoencoder.
    """

    def __init__(self, features: int, expandBy: int) -> None:
        """
        Initialize the sparse autoencoder.

        :type features: int
        :param features: The embedding size.
        :type expandBy: int
        :param expandBy: Expand factor for the dictionary.
        """
        raise NotImplementedError

    def forward(self, x: Tensor, activate: int) -> Tuple[Tensor, Tensor]:
        """
        Forward pass to reconstruct the embedding.

        :type x: Tensor
        :param x: The original embedding. The tensor has shape (N, D), where N
            is the batch size and D is the embedding size.
        :type K: int
        :param activate: The number of features to activate. This is the
            sparsity constraint. Only the top-K features are activated. The
            rest are set to zero.
        :rtype: Tuple[Tensor, Tensor]
        :return: The latent features and the reconstructed embedding. The
            latent features have shape (N, D), where D is the dictionary size.
            The reconstructed embedding has shape (N, E), where E is the
            embedding size. N is the batch size in both cases.
        """
        raise NotImplementedError
