"""
Specify the dataset interface.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Literal


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
    def getQueries(self, partition: PartitionType) -> Iterator[Tuple[str, str]]:
        """
        Get the queries in the dataset.

        :param partition: The partition to get queries from.
        :return: Iterator over query IDs and texts.
        """
        raise NotImplementedError
