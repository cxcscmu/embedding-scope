"""
Implementation of the MS MARCO dataset.
"""

import argparse
import subprocess
from pathlib import Path
from typing import Iterator, Tuple, List
from hashlib import md5
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from source import console
from source.interface import TextRetrievalDataset
from source.dataset import workspace
from source.dataset.utilities import textRetrievalGetPassages


class MsMarco(TextRetrievalDataset):
    """
    Implementation of the MS MARCO dataset.
    """

    name = "MsMarco"

    def __init__(self) -> None:
        pass

    def getPassages(self) -> Iterator[Tuple[str, str]]:
        N, base = GetPassagesInit.N, GetPassagesInit.base
        return textRetrievalGetPassages(base, N)


class GetPassagesInit:
    """
    Prepare the passages in the dataset.

    Attributes:
        N: The number of partitions.
        base: The base path for the passages.
    """

    N = 2
    base = Path(workspace, f"{MsMarco.name}/passages")

    def __init__(self) -> None:
        self.base.mkdir(mode=0o770, parents=True, exist_ok=True)
        self.dispatch()

    def step1(self):
        """
        Dispatch the download of the passages.
        """
        host = "https://msmarco.z22.web.core.windows.net"
        link = f"{host}/msmarcoranking/collection.tar.gz"
        console.log("Downloading the passages")
        with requests.get(link, stream=True, timeout=1800) as response:
            with Path(self.base, "collection.tar.gz").open("wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

    def step2(self):
        """
        Dispatch the extraction of the passages.
        """
        console.log("Extracting the passages")
        subprocess.run(
            ["tar", "-xzvf", "collection.tar.gz"],
            cwd=self.base,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        Path(self.base, "collection.tar.gz").unlink()

    def step3(self):
        """
        Dispatch the partitioning of the passages.
        """
        ids: List[List[str]] = [[] for _ in range(self.N)]
        texts: List[List[str]] = [[] for _ in range(self.N)]
        path = Path(self.base, "collection.tsv")
        console.log("Partitioning the passages")
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                x0, x1 = [p.strip() for p in line.split("\t")]
                i = int(md5(x0.encode()).hexdigest(), 16) % self.N
                ids[i].append(x0)
                texts[i].append(x1)
        console.log("Writing the partitions")
        for i in range(self.N):
            table = pa.Table.from_pydict({"id": ids[i], "text": texts[i]})
            pq.write_table(table, Path(self.base, f"partition-{i:08d}.parquet"))
        Path(self.base, "collection.tsv").unlink()

    def dispatch(self):
        """
        Dispatch the steps.
        """
        self.step1()
        self.step2()
        self.step3()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", type=str, choices=["getPassages"])
    parser.add_argument("params", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    match args.type:
        case "getPassages":
            GetPassagesInit()
