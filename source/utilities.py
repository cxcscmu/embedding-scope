"""
Utilities for the project.
"""

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
