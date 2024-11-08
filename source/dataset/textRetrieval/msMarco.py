"""
Implementation for the MS MARCO dataset.
"""

import pickle
import argparse
import subprocess
from hashlib import md5
from typing import Type, List, Dict
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
from source.interface.dataset import TextRetrievalDataset, PartitionType
from source.embedding.miniCPM import MiniCPM
from source.embedding.bgeBase import BgeBase
from source.dataset.textRetrieval.utilities import (
    newPassageLoaderFrom,
    newPassageEmbeddingLoaderFrom,
    newQueryLoaderFrom,
    newQueryEmbeddingLoaderFrom,
)


class MsMarcoDataset(TextRetrievalDataset):
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
        file = Path(workspace, f"msMarco/queries/{partition}.parquet")
        return newQueryLoaderFrom(file, batchSize, shuffle, numWorkers)

    @staticmethod
    def newQueryEmbeddingLoader(
        partition: PartitionType,
        embedding: Type[TextEmbedding],
        batchSize: int,
        shuffle: bool,
        numWorkers: int,
    ) -> DataLoader:
        base = Path(workspace, f"msMarco/queryEmbeddings/{embedding.name}/{partition}")
        return newQueryEmbeddingLoaderFrom(base, batchSize, shuffle, numWorkers)

    @staticmethod
    def getRelevance(partition: PartitionType) -> Dict[str, Dict[str, int]]:
        base = Path(workspace, "msMarco/queryRelevance")
        with Path(base, f"{partition}.pickle").open("rb") as file:
            return pickle.load(file)


def preparePassages(numShards: int):
    """
    Prepare the passage loader.
    """
    base = Path(workspace, "msMarco/passages")
    base.mkdir(mode=0o770, parents=True, exist_ok=True)

    logger.info("Download the passages")
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

    logger.info("Extract the passages from tarball")
    subprocess.run(
        ["tar", "-xzvf", "collection.tar.gz"],
        cwd=base,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    path.unlink()

    logger.info("Split the passages into shards")
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

    logger.info("Write the shards to disk")
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
    loader = MsMarcoDataset.newPassageLoader(1, False, 1)

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
        assert len(batchIdx) == len(batchPsg)
        _, shardIdx = divmod(i, numShards)
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


def prepareQueries():
    """
    Prepare the query loader.
    """
    base = Path(workspace, "msMarco/queries")
    base.mkdir(mode=0o770, parents=True, exist_ok=True)

    logger.info("Download the queries")
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
        ) as progress:
            with path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    progress.update(len(chunk))

    logger.info("Extract the queries from tarball")
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
        logger.info("Refactor the %s queries", partition)
        path = Path(base, f"queries.{partition}.tsv")
        qids, queries = [], []
        with tqdm(
            total=path.stat().st_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    pid, query = line.split("\t")
                    qids.append(pid)
                    queries.append(query)
                    progress.update(len(line.encode()))
        table = pa.Table.from_pydict({"qid": qids, "query": queries})
        pq.write_table(table, Path(base, f"{partition}.parquet"))
        path.unlink()


def prepareQueryEmbeddings(
    partition: PartitionType,
    embedding: TextEmbedding,
    batchSize: int,
    numShards: int,
    workerCnt: int,
    workerIdx: int,
):
    """
    Prepare the query embedding loader.
    """
    base = Path(workspace, f"msMarco/queryEmbeddings/{embedding.name}/{partition}")
    base.mkdir(mode=0o770, parents=True, exist_ok=True)

    batchIdx, batchQry = [], []

    def compute():
        vectors = embedding.forward(batchQry)
        for j, x in zip(batchIdx, vectors):
            _, shardIdx = divmod(j, numShards)
            shards[shardIdx].append(x)
        batchIdx.clear()
        batchQry.clear()
        cuda.empty_cache()

    logger.info("Load the queries")
    loader = MsMarcoDataset.newQueryLoader(partition, 1, False, 1)

    logger.info("Split the shards with co-workers")
    shards: List[List[NDArray[np.float32]]] = [[] for _ in range(numShards)]

    logger.info("Generate the embeddings")
    for i, (_, query) in enumerate(tqdm(iterable=loader.dataset)):
        assert len(batchIdx) == len(batchQry)
        _, shardIdx = divmod(i, numShards)
        if shardIdx % workerCnt == workerIdx:
            batchIdx.append(i)
            batchQry.append(query)
            if len(batchIdx) >= batchSize:
                compute()
    if batchIdx:
        compute()

    logger.info("Write the shards to disk")
    for i, shard in enumerate(shards):
        if i % workerCnt == workerIdx:
            buffer = np.stack(shard, dtype=np.float32)
            np.save(Path(base, f"{i:08d}.npy"), buffer)


def prepareRelevance(partition: PartitionType):
    """
    Prepare the relevance judgments.
    """
    base = Path(workspace, "msMarco/queryRelevance")
    base.mkdir(mode=0o770, parents=True, exist_ok=True)

    logger.info("Download the relevance judgments")
    host = "https://msmarco.z22.web.core.windows.net"
    link = f"{host}/msmarcoranking/qrels.{partition}.tsv"
    path = Path(base, f"{partition}.tsv")
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

    logger.info("Refactor the relevance judgments")
    data: Dict[str, Dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            qid, _, pid, rel = line.split()
            if qid not in data:
                data[qid] = {}
            data[qid][pid] = int(rel)

    logger.info("Write the relevance judgments to disk")
    with path.with_suffix(".pickle").open("wb") as file:
        pickle.dump(data, file)
    path.unlink()


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
    subparsers.add_parser("prepareQueries")
    prepareQueryEmbeddingsParser = subparsers.add_parser("prepareQueryEmbeddings")
    prepareQueryEmbeddingsParser.add_argument("--partition", type=str, required=True)
    prepareQueryEmbeddingsParser.add_argument("--embedding", type=str, required=True)
    prepareQueryEmbeddingsParser.add_argument("--gpuDevice", type=int, nargs="+", required=True)
    prepareQueryEmbeddingsParser.add_argument("--batchSize", type=int, required=True)
    prepareQueryEmbeddingsParser.add_argument("--numShards", type=int, required=True)
    prepareQueryEmbeddingsParser.add_argument("--workerCnt", type=int, required=True)
    prepareQueryEmbeddingsParser.add_argument("--workerIdx", type=int, required=True)
    prepareRelevanceParser = subparsers.add_parser("prepareRelevance")
    prepareRelevanceParser.add_argument("--partition", type=str, required=True)
    parsed = parser.parse_args()
    # fmt: on

    match parsed.command:
        case "preparePassages":
            preparePassages(parsed.numShards)
        case "preparePassageEmbeddings":
            match parsed.embedding:
                case "miniCPM":
                    embedding = MiniCPM(parsed.gpuDevice)
                case "bgeBase":
                    embedding = BgeBase(parsed.gpuDevice)
                case _:
                    raise NotImplementedError
            preparePassageEmbeddings(
                embedding,
                parsed.batchSize,
                parsed.numShards,
                parsed.workerCnt,
                parsed.workerIdx,
            )
        case "prepareQueries":
            prepareQueries()
        case "prepareQueryEmbeddings":
            match parsed.embedding:
                case "miniCPM":
                    embedding = MiniCPM(parsed.gpuDevice)
                case "bgeBase":
                    embedding = BgeBase(parsed.gpuDevice)
                case _:
                    raise NotImplementedError
            prepareQueryEmbeddings(
                parsed.partition,
                embedding,
                parsed.batchSize,
                parsed.numShards,
                parsed.workerCnt,
                parsed.workerIdx,
            )
        case "prepareRelevance":
            prepareRelevance(parsed.partition)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
