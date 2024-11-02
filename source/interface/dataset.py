"""
Specify the dataset interface.
"""

from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Literal


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
    def getPassages(self) -> Iterable[Tuple[str, str]]:
        """
        Get the passages in the dataset.

        :return: Iterable of passage IDs and texts.
        """

    @abstractmethod
    def getQueries(self, partition: PartitionType) -> Iterable[Tuple[str, str]]:
        """
        Get the queries in the dataset.

        :param partition: The partition to get queries from.
        :return: Iterable of query IDs and texts.
        """
        raise NotImplementedError
