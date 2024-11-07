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
from tqdm import tqdm
from torch import cuda
from torch.utils.data import DataLoader
from source import logger
from source.utilities import TqdmFile
from source.dataset.textRetrieval import workspace
from source.interface.embedding import TextEmbedding
from source.interface.dataset import TextRetrievalDataset, PartitionType
from source.embedding.miniCPM import MiniCPM
from source.dataset.textRetrieval.utilities import (
    newPassageLoaderFrom,
    newPassageEmbeddingLoaderFrom,
    newQueryLoaderFrom,
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

    @staticmethod
    def newQueryLoader(
        partition: PartitionType, batchSize: int, shuffle: bool, numWorkers: int
    ) -> DataLoader:
        base = Path(workspace, "msMarco/queries", partition)
        return newQueryLoaderFrom(base, batchSize, shuffle, numWorkers)


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
            mininterval=3,
            ncols=80,
            file=TqdmFile,
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
        mininterval=3,
        ncols=80,
        file=TqdmFile,
    ) as progress:
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                pid, passage = line.split("\t")
                index = int(md5(pid.encode()).hexdigest(), 16) % numShards
                pids[index].append(pid)
                passages[index].append(passage)
                progress.update(len(line.encode()))

    logger.info("Writing the shards")
    with tqdm(total=numShards, mininterval=3, ncols=80, file=TqdmFile) as progress:
        for i in range(numShards):
            pidsShard, passagesShard = pids[i], passages[i]
            table = pa.Table.from_pydict({"pid": pidsShard, "passage": passagesShard})
            pq.write_table(table, Path(base, f"{i:08d}.parquet"))
            progress.update()
    path.unlink()


def preparePassageEmbeddings(
    embedding: TextEmbedding,
    numShards: int,
    numWorkers: int,
    workerSeed: int,
    batchSize: int,
):
    """
    Prepare the passage embedding loader.
    """
    logger.info("Preparing the passage embeddings")
    base = Path(workspace, f"msMarco/passageEmbeddings/{embedding.name}")
    base.mkdir(mode=0o770, parents=True, exist_ok=True)

    logger.info("Sharding the directories")
    for i in range(numShards):
        if i % numWorkers != workerSeed:
            continue
        shardBase = Path(base, f"{i:08d}")
        shardBase.mkdir(mode=0o770, parents=True, exist_ok=True)

    logger.info("Loading the passages")
    loader = MsMarco.newPassageLoader(1, False, 1)

    logger.info("Computing the embeddings")
    batchI: List[int] = []
    batchP: List[str] = []
    with tqdm(
        total=len(loader.dataset),
        mininterval=3,
        ncols=80,
        file=TqdmFile,
    ) as progress:
        for i, (_, passage) in enumerate(loader.dataset):
            progress.update()
            if i % numShards % numWorkers != workerSeed:
                continue
            batchI.append(i)
            batchP.append(passage)
            if len(batchI) >= batchSize or i == len(loader.dataset) - 1:
                vectors = embedding.forward(batchP)
                for j, x in zip(batchI, vectors):
                    index = j % numShards
                    shardBase = Path(base, f"{index:08d}")
                    np.save(Path(shardBase, f"{j:08d}.npy"), x)
                batchI.clear()
                batchP.clear()
                cuda.empty_cache()


def prepareQueries(numShards: int):
    """
    Prepare the query loader.
    """
    logger.info("Preparing the queries")
    base = Path(workspace, "msMarco/queries")
    base.mkdir(mode=0o770, parents=True, exist_ok=True)

    logger.info("Downloading the queries")
    host = "https://msmarco.z22.web.core.windows.net"
    link = f"{host}/msmarcoranking/queries.tar.gz"
    path = Path(base, "queries.tar.gz")
    with requests.get(link, stream=True, timeout=1800) as response:
        response.raise_for_status()
        with tqdm(
            total=int(response.headers["Content-Length"]),
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            mininterval=3,
            ncols=80,
            file=TqdmFile,
        ) as progress:
            with path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    progress.update(len(chunk))

    logger.info("Extracting the queries")
    subprocess.run(
        ["tar", "-xzvf", "queries.tar.gz"],
        cwd=base,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    path.unlink()

    choices: List[PartitionType] = ["train", "dev", "eval"]
    for partition in choices:
        partitionBase = Path(base, partition)
        partitionBase.mkdir(mode=0o770, parents=True, exist_ok=True)

        logger.info("Sharding the queries, %s", partition)
        path = Path(base, f"queries.{partition}.tsv")
        qids: List[List[str]] = [[] for _ in range(numShards)]
        queries: List[List[str]] = [[] for _ in range(numShards)]
        with tqdm(
            total=path.stat().st_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            mininterval=3,
            ncols=80,
            file=TqdmFile,
        ) as progress:
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    qid, query = line.split("\t")
                    index = int(md5(qid.encode()).hexdigest(), 16) % numShards
                    qids[index].append(qid)
                    queries[index].append(query)
                    progress.update(len(line.encode()))

        logger.info("Writing the shards, %s", partition)
        with tqdm(total=numShards, mininterval=3, ncols=80, file=TqdmFile) as progress:
            for i in range(numShards):
                qidsShard, queriesShard = qids[i], queries[i]
                table = pa.Table.from_pydict({"qid": qidsShard, "query": queriesShard})
                pq.write_table(table, Path(partitionBase, f"{i:08d}.parquet"))
                progress.update()
        path.unlink()


def prepareQueryEmbeddings(
    embedding: TextEmbedding,
    numShards: int,
    numWorkers: int,
    workerSeed: int,
    batchSize: int,
):
    """
    Prepare the query embedding loader.
    """
    logger.info("Preparing the query embeddings")
    base = Path(workspace, f"msMarco/queryEmbeddings/{embedding.name}")
    base.mkdir(mode=0o770, parents=True, exist_ok=True)

    choices: List[PartitionType] = ["train", "dev", "eval"]
    for partition in choices:
        partitionBase = Path(workspace, f"msMarco/queries/{partition}")
        partitionBase.mkdir(mode=0o770, parents=True, exist_ok=True)

        logger.info("Sharding the directories, %s", partition)
        for i in range(numShards):
            if i % numWorkers == workerSeed:
                shardBase = Path(partitionBase, f"{i:08d}")
                shardBase.mkdir(mode=0o770, parents=True, exist_ok=True)

        logger.info("Loading the queries, %s", partition)
        loader = MsMarco.newQueryLoader(partition, 1, False, 1)

        logger.info("Computing the embeddings, %s", partition)
        batchI: List[int] = []
        batchQ: List[str] = []
        with tqdm(
            total=len(loader.dataset),
            mininterval=3,
            ncols=80,
            file=TqdmFile,
        ) as progress:
            for i, (_, query) in enumerate(loader.dataset):
                progress.update()
                if i % numShards % numWorkers != workerSeed:
                    continue
                batchI.append(i)
                batchQ.append(query)
                if len(batchI) >= batchSize or i == len(loader.dataset) - 1:
                    vectors = embedding.forward(batchQ)
                    for j, x in zip(batchI, vectors):
                        index = j % numShards
                        shardBase = Path(partitionBase, f"{index:08d}")
                        np.save(Path(shardBase, f"{j:08d}.npy"), x)
                    batchI.clear()
                    batchQ.clear()
                    cuda.empty_cache()


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
    preparePassageEmbeddingsParser.add_argument("--numShards", type=int, required=True)
    preparePassageEmbeddingsParser.add_argument("--numWorkers", type=int, required=True)
    preparePassageEmbeddingsParser.add_argument("--workerSeed", type=int, required=True)
    preparePassageEmbeddingsParser.add_argument("--batchSize", type=int, required=True)
    preparePassageEmbeddingsParser.add_argument("--devices", type=int, nargs="+", required=True)
    prepareQueriesParser = subparsers.add_parser("prepareQueries")
    prepareQueriesParser.add_argument("--numShards", type=int, required=True)
    prepareQueryEmbeddingsParser = subparsers.add_parser("prepareQueryEmbeddings")
    prepareQueryEmbeddingsParser.add_argument("--embedding", type=str, required=True)
    prepareQueryEmbeddingsParser.add_argument("--numShards", type=int, required=True)
    prepareQueryEmbeddingsParser.add_argument("--numWorkers", type=int, required=True)
    prepareQueryEmbeddingsParser.add_argument("--workerSeed", type=int, required=True)
    prepareQueryEmbeddingsParser.add_argument("--batchSize", type=int, required=True)
    prepareQueryEmbeddingsParser.add_argument("--devices", type=int, nargs="+", required=True)
    parsed = parser.parse_args()
    # fmt: on

    match parsed.command:
        case "preparePassages":
            preparePassages(parsed.numShards)
        case "preparePassageEmbeddings":
            match parsed.embedding:
                case "miniCPM":
                    embedding = MiniCPM(parsed.devices)
                case _:
                    raise NotImplementedError
            preparePassageEmbeddings(
                embedding,
                parsed.numShards,
                parsed.numWorkers,
                parsed.workerSeed,
                parsed.batchSize,
            )
        case "prepareQueries":
            prepareQueries(parsed.numShards)
        case "prepareQueryEmbeddings":
            match parsed.embedding:
                case "miniCPM":
                    embedding = MiniCPM(parsed.devices)
                case _:
                    raise NotImplementedError
            prepareQueryEmbeddings(
                embedding,
                parsed.numShards,
                parsed.numWorkers,
                parsed.workerSeed,
                parsed.batchSize,
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
