"""
Utilities for dataset module.
"""

from pathlib import Path
from typing import Iterator, Tuple
import pyarrow.parquet as pq


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


def textRetrievalGetQueries(base: Path) -> Iterator[Tuple[str, str]]:
    """
    Getting queries from a text retrieval dataset.

    :param base: The base path for the queries.
    :return: Iterator over query IDs and texts.
    """
    return textRetrievalGetPassages(base)
