"""
Utilities for the project.
"""

import requests
from pathlib import Path
from tqdm import tqdm as _tqdm
from source import logger


class TqdmFile:
    """
    A file-like object for tqdm.

    This is helpful for redirecting the output of tqdm to a file on disk.
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
        pass


def tqdm(*args, **kwargs):
    """
    Wrapper for tqdm with logger.

    This wrapper redirects the output of tqdm to the logger. We've also set the
    default parameters for tqdm, such as the update interval, to make it more
    suitable for logging.
    """
    kwargs.setdefault("file", TqdmFile)
    kwargs.setdefault("mininterval", 3)
    kwargs.setdefault("ncols", 80)
    kwargs.setdefault("ascii", False)
    return _tqdm(*args, **kwargs)


def parseInt(value: str) -> int:
    """
    Parse the integer.

    This function parses the integer from the string. The string could be in
    the format of "123", "123k", or "123m". The function will convert them to
    the corresponding integer. The function will raise an error if the format
    is not recognized.
    """
    if value.isdigit():
        return int(value)
    if value.endswith(("k", "K")):
        return int(value[:-1]) * 1_000
    if value.endswith(("m", "M")):
        return int(value[:-1]) * 1_000_000
    raise NotImplementedError()


def download_file(link: str, path: Path, timeout: int = 60 * 60):
    """
    Download the file.

    Parameters
    ----------
    link : str
        The link to the file.
    path : Path
        The path to save the file.
    timeout : int
        The timeout for the download. Default is 1 hour.
    """

    with requests.get(
        link, stream=True, allow_redirects=True, timeout=timeout
    ) as response:
        response.raise_for_status()
        with path.open("wb") as file, tqdm(
            total=int(response.headers.get("Content-Length", 0)),
            unit="B",
            unit_scale=True,
        ) as progress:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                progress.update(len(chunk))
