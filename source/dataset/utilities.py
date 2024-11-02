"""
Utilities for dataset module.
"""

from pathlib import Path
from typing import Iterable, Tuple
import pyarrow.parquet as pq


def textRetrievalGetPassages(base: Path, N: int) -> Iterable[Tuple[str, str]]:
    """
    Getting passages from a text retrieval dataset.

    :param base: The base path for the passages.
    :param N: The number of partitions.
    :return: Iterable of passage IDs and texts.
    """
    for i in range(N):
        name = f"partition-{i:08d}.parquet"
        file = pq.ParquetFile(Path(base, name))
        for xs, ys in file.iter_batches():
            for x, y in zip(xs, ys):
                yield x.as_py(), y.as_py()


def textRetrievalGetQueries(base: Path, N: int) -> Iterable[Tuple[str, str]]:
    """
    Getting queries from a text retrieval dataset.

    :param base: The base path for the queries.
    :param N: The number of partitions.
    :return: Iterable of query IDs and texts.
    """
    return textRetrievalGetPassages(base, N)
