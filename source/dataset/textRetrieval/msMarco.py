"""
Implementation for the MS MARCO dataset.
"""

import argparse
import subprocess
from hashlib import md5
from typing import Type, List
from pathlib import Path
import requests
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from numpy import ndarray as NDArray
from torch import cuda
from torch.utils.data import DataLoader
from source import logger
from source.utilities import tqdm
from source.dataset.textRetrieval import workspace
from source.interface.embedding import TextEmbedding
from source.interface.dataset import TextRetrievalDataset
from source.embedding.miniCPM import MiniCPM
from source.dataset.textRetrieval.utilities import (
    newPassageLoaderFrom,
    newPassageEmbeddingLoaderFrom,
)


class MsMarco(TextRetrievalDataset):
    """
    Implementation for the MS MARCO dataset.
    """

    @staticmethod
    def newPassageLoader(batchSize: int, shuffle: bool, numWorkers: int) -> DataLoader:
        base = Path(workspace, "msMarco/passages")
        return newPassageLoaderFrom(base, batchSize, shuffle, numWorkers)

    @staticmethod
    def newPassageEmbeddingLoader(
        embedding: Type[TextEmbedding], batchSize: int, shuffle: bool, numWorkers: int
    ) -> DataLoader:
        base = Path(workspace, f"msMarco/passageEmbeddings/{embedding.name}")
        return newPassageEmbeddingLoaderFrom(base, batchSize, shuffle, numWorkers)


def preparePassages(numShards: int):
    """
    Prepare the passage loader.
    """
    logger.info("Preparing the passages")
    base = Path(workspace, "msMarco/passages")
    base.mkdir(mode=0o770, parents=True, exist_ok=True)

    logger.info("Downloading the passages")
    host = "https://msmarco.z22.web.core.windows.net"
    link = f"{host}/msmarcoranking/collection.tar.gz"
    path = Path(base, "collection.tar.gz")
    with requests.get(link, stream=True, timeout=1800) as response:
        response.raise_for_status()
        with tqdm(
            total=int(response.headers["Content-Length"]),
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            with path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    progress.update(len(chunk))

    logger.info("Extracting the passages")
    subprocess.run(
        ["tar", "-xzvf", "collection.tar.gz"],
        cwd=base,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    path.unlink()

    logger.info("Sharding the passages")
    path = Path(base, "collection.tsv")
    pids: List[List[str]] = [[] for _ in range(numShards)]
    passages: List[List[str]] = [[] for _ in range(numShards)]
    with tqdm(
        total=path.stat().st_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                pid, passage = line.split("\t")
                index = int(md5(pid.encode()).hexdigest(), 16) % numShards
                pids[index].append(pid)
                passages[index].append(passage)
                progress.update(len(line.encode()))

    logger.info("Writing the shards")
    with tqdm(total=numShards) as progress:
        for i in range(numShards):
            pidsShard, passagesShard = pids[i], passages[i]
            table = pa.Table.from_pydict({"pid": pidsShard, "passage": passagesShard})
            pq.write_table(table, Path(base, f"{i:08d}.parquet"))
            progress.update()
    path.unlink()


def preparePassageEmbeddings(
    embedding: TextEmbedding,
    batchSize: int,
    numShards: int,
    workerCnt: int,
    workerIdx: int,
):
    """
    Prepare the passage embedding loader.
    """
    base = Path(workspace, f"msMarco/passageEmbeddings/{embedding.name}")
    base.mkdir(mode=0o770, parents=True, exist_ok=True)

    logger.info("Load the passages")
    loader = MsMarco.newPassageLoader(batchSize, shuffle=False, numWorkers=1)

    logger.info("Split the shards with co-workers")
    shards: List[List[NDArray[np.float32]]] = [[] for _ in range(numShards)]
    batchIdx, batchPsg = [], []

    def compute():
        vectors = embedding.forward(batchPsg)
        for j, x in zip(batchIdx, vectors):
            _, shardIdx = divmod(j, numShards)
            shards[shardIdx].append(x)
        batchIdx.clear()
        batchPsg.clear()
        cuda.empty_cache()

    logger.info("Generate the embeddings")
    for i, (_, passage) in enumerate(tqdm(iterable=loader.dataset)):
        _, shardIdx = divmod(i, numShards)
        assert len(batchIdx) == len(batchPsg)
        if shardIdx % workerCnt == workerIdx:
            batchIdx.append(i)
            batchPsg.append(passage)
            if len(batchIdx) >= batchSize:
                compute()
    if batchIdx:
        compute()

    logger.info("Write the shards to disk")
    for i, shard in enumerate(shards):
        if i % workerCnt == workerIdx:
            buffer = np.stack(shard, dtype=np.float32)
            np.save(Path(base, f"{i:08d}.npy"), buffer)


def main():
    """
    The entry point.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    # fmt: off
    preparePassagesParser = subparsers.add_parser("preparePassages")
    preparePassagesParser.add_argument("--numShards", type=int, required=True)
    preparePassageEmbeddingsParser = subparsers.add_parser("preparePassageEmbeddings")
    preparePassageEmbeddingsParser.add_argument("--embedding", type=str, required=True)
    preparePassageEmbeddingsParser.add_argument("--gpuDevice", type=int, nargs="+", required=True)
    preparePassageEmbeddingsParser.add_argument("--batchSize", type=int, required=True)
    preparePassageEmbeddingsParser.add_argument("--numShards", type=int, required=True)
    preparePassageEmbeddingsParser.add_argument("--workerCnt", type=int, required=True)
    preparePassageEmbeddingsParser.add_argument("--workerIdx", type=int, required=True)
    parsed = parser.parse_args()
    # fmt: on

    match parsed.command:
        case "preparePassages":
            preparePassages(parsed.numShards)
        case "preparePassageEmbeddings":
            match parsed.embedding:
                case "miniCPM":
                    embedding = MiniCPM(parsed.gpuDevice)
                case _:
                    raise NotImplementedError
            preparePassageEmbeddings(
                embedding,
                parsed.batchSize,
                parsed.numShards,
                parsed.workerCnt,
                parsed.workerIdx,
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
