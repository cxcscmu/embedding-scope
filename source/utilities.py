"""
Utilities for the project.
"""

from tqdm import tqdm as _tqdm
from source import logger


class TqdmFile:
    """
    A file-like object for tqdm.
    """

    @staticmethod
    def write(message: str) -> None:
        """
        Write the message.
        """
        if message := message.strip():
            logger.info(message)

    @staticmethod
    def flush() -> None:
        """
        Flush the message.
        """


def tqdm(*args, **kwargs):
    """
    Wrapper for tqdm.
    """
    kwargs.setdefault("file", TqdmFile)
    kwargs.setdefault("mininterval", 3)
    kwargs.setdefault("ncols", 80)
    kwargs.setdefault("ascii", False)
    return _tqdm(*args, **kwargs)
