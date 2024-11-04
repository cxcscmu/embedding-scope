"""
Utilities for dataset module.
"""

from pathlib import Path
from typing import Iterator, Tuple
import pyarrow.parquet as pq
import numpy as np
from numpy import ndarray as NDArray


def textRetrievalGetPassages(base: Path) -> Iterator[Tuple[str, str]]:
    """
    Getting passages from a text retrieval dataset.

    :param base: The base path for the passages.
    :return: Iterator over passage IDs and texts.
    """
    for path in sorted(base.iterdir()):
        file = pq.ParquetFile(path)
        for xs, ys in file.iter_batches():
            for x, y in zip(xs, ys):
                yield x.as_py(), y.as_py()


def textRetrievalGetPassageEmbeddings(
    base: Path,
) -> Iterator[Tuple[str, NDArray[np.float32]]]:
    """
    Getting passage embeddings from a text retrieval dataset.

    :param base: The base path for the embeddings.
    :return: Iterator over passage IDs and embeddings.
    """
    for path in sorted(base.iterdir()):
        data = np.load(path)
        for x, y in zip(data["ids"], data["vectors"]):
            yield str(x), y


def textRetrievalGetQueries(base: Path) -> Iterator[Tuple[str, str]]:
    """
    Getting queries from a text retrieval dataset.

    :param base: The base path for the queries.
    :return: Iterator over query IDs and texts.
    """
    return textRetrievalGetPassages(base)


def textRetrievalGetQueryEmbeddings(
    base: Path,
) -> Iterator[Tuple[str, NDArray[np.float32]]]:
    """
    Getting query embeddings from a text retrieval dataset.

    :param base: The base path for the embeddings.
    :return: Iterator over query IDs and embeddings.
    """
    return textRetrievalGetPassageEmbeddings(base)
