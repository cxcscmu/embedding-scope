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


def tqdm(**kwargs):
    """
    Wrapper for tqdm.
    """
    return _tqdm(**kwargs, file=TqdmFile, mininterval=3, ncols=80, ascii=False)
