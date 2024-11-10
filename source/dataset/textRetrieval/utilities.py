"""
Utilities for text retrieval dataset.
"""

from typing import List, Tuple
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset


class PassageDataset(Dataset):
    """
    Dataset for passages.
    """

    def __init__(self, base: Path) -> None:
        """
        Initialize the dataset.

        :param base: The base path where all the passage shards are stored.
        """
        super().__init__()
        self.shards = []
        for file in sorted(base.glob("*.parquet")):
            data = pq.read_table(file, memory_map=True)
            self.shards.append(data)
        self.length = sum(len(x) for x in self.shards)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[str, str]:
        shardOff, shardIdx = divmod(index, len(self.shards))
        pid = self.shards[shardIdx]["pid"][shardOff].as_py()
        passage = self.shards[shardIdx]["passage"][shardOff].as_py()
        return pid, passage


def newPassageLoaderFrom(
    base: Path, batchSize: int, shuffle: bool, numWorkers: int
) -> DataLoader:
    """
    Create a new passage loader from the base path.

    :param base: The base path.
    :param batchSize: The batch size.
    :param shuffle: Whether to shuffle the data.
    :param numWorkers: The number of workers.
    :return: The passage loader.
    """
    return DataLoader(
        PassageDataset(base),
        batch_size=batchSize,
        shuffle=shuffle,
        num_workers=numWorkers,
    )


class PassageEmbeddingDataset(Dataset):
    """
    Dataset for passage embeddings.
    """

    def __init__(self, base: Path) -> None:
        super().__init__()
        self.shards: List[NDArray[np.float32]] = []
        for file in sorted(base.glob("*.npy")):
            data = np.load(file, mmap_mode="r")
            self.shards.append(data)
        self.length = sum(len(x) for x in self.shards)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> NDArray[np.float32]:
        shardOff, shardIdx = divmod(index, len(self.shards))
        return self.shards[shardIdx][shardOff]


def newPassageEmbeddingLoaderFrom(
    base: Path, batchSize: int, shuffle: bool, numWorkers: int
) -> DataLoader:
    """
    Create a new passage embedding loader from the base path.

    :param base: The base path.
    :param batchSize: The batch size.
    :param shuffle: Whether to shuffle the data.
    :param numWorkers: The number of workers.
    :return: The passage embedding loader.
    """
    return DataLoader(
        PassageEmbeddingDataset(base),
        batch_size=batchSize,
        shuffle=shuffle,
        num_workers=numWorkers,
    )


class QueryDataset(Dataset):
    """
    Dataset for queries.
    """

    def __init__(self, file: Path) -> None:
        super().__init__()
        self.file = pq.read_table(file, memory_map=True)

    def __len__(self) -> int:
        return len(self.file)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        qid = self.file["qid"][index].as_py()
        query = self.file["query"][index].as_py()
        return qid, query


def newQueryLoaderFrom(
    file: Path, batchSize: int, shuffle: bool, numWorkers: int
) -> DataLoader:
    """
    Create a new query loader from the file.

    :param file: The file.
    :param batchSize: The batch size.
    :param shuffle: Whether to shuffle the data.
    :param numWorkers: The number of workers.
    :return: The query loader.
    """
    return DataLoader(
        QueryDataset(file),
        batch_size=batchSize,
        shuffle=shuffle,
        num_workers=numWorkers,
    )


class QueryEmbeddingDataset(Dataset):
    """
    Dataset for query embeddings.
    """

    def __init__(self, base: Path) -> None:
        super().__init__()
        self.shards: List[NDArray[np.float32]] = []
        for file in sorted(base.glob("*.npy")):
            data = np.load(file, mmap_mode="r")
            self.shards.append(data)
        self.length = sum(len(x) for x in self.shards)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> NDArray[np.float32]:
        shardOff, shardIdx = divmod(index, len(self.shards))
        return self.shards[shardIdx][shardOff]


def newQueryEmbeddingLoaderFrom(
    base: Path, batchSize: int, shuffle: bool, numWorkers: int
) -> DataLoader:
    """
    Create a new query embedding loader from the base path.

    :param base: The base path.
    :param batchSize: The batch size.
    :param shuffle: Whether to shuffle the data.
    :param numWorkers: The number of workers.
    :return: The query embedding loader.
    """
    return DataLoader(
        QueryEmbeddingDataset(base),
        batch_size=batchSize,
        shuffle=shuffle,
        num_workers=numWorkers,
    )
