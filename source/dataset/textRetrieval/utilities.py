"""
Utilities for text retrieval dataset.
"""

from typing import List, Tuple
from pathlib import Path
from tqdm import tqdm
import pyarrow.parquet as pq
import numpy as np
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset
from source.utilities import TqdmFile


class PassageDataset(Dataset):
    """
    Dataset for passages.
    """

    def __init__(self, base: Path) -> None:
        super().__init__()
        self.pids: List[str] = []
        self.passages: List[str] = []

        glob = list(base.iterdir())
        with tqdm(total=len(glob), mininterval=3, ncols=80, file=TqdmFile) as progress:
            for path in glob:
                file = pq.read_table(path)
                self.pids.extend(str(x) for x in file["pid"])
                self.passages.extend(str(x) for x in file["passage"])
                progress.update()

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        return self.pids[index], self.passages[index]


class PassageEmbeddingDataset(Dataset):
    """
    Dataset for passage embeddings.
    """

    def __init__(self, base: Path) -> None:
        super().__init__()
        self.base = base

        glob = list(base.iterdir())
        self.numShards = len(glob)
        self.length = 0
        with tqdm(total=len(glob), mininterval=3, ncols=80, file=TqdmFile) as progress:
            for path in glob:
                self.length += sum(1 for _ in path.iterdir())
                progress.update()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> NDArray[np.float32]:
        shardIndex = index % self.numShards
        shardBase = Path(self.base, f"{shardIndex:08d}")
        path = Path(shardBase, f"{index:08d}.npy")
        return np.load(path)


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
