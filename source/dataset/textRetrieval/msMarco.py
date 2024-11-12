"""
Implement the MS MARCO dataset.
"""

import pickle
import argparse
import subprocess
from typing import Type, List, Dict, Tuple
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
from source.retriever.dense import DotProductRetriever
from source.embedding.miniCPM import MiniCPM
from source.embedding.bgeBase import BgeBase
from source.dataset.textRetrieval.utilities import (
    newPassageLoaderFrom,
    newPassageEmbeddingLoaderFrom,
    newQueryLoaderFrom,
    newQueryEmbeddingLoaderFrom,
    newMixEmbeddingLoaderFrom,
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
        embedding: Type[TextEmbedding],
        partition: PartitionType,
        batchSize: int,
        shuffle: bool,
        numWorkers: int,
    ) -> DataLoader:
        base = Path(workspace, f"msMarco/queryEmbeddings/{embedding.name}/{partition}")
        return newQueryEmbeddingLoaderFrom(base, batchSize, shuffle, numWorkers)

    @staticmethod
    def getQueryRelevance(partition: PartitionType) -> Dict[str, Dict[str, int]]:
        base = Path(workspace, "msMarco/queryRelevance")
        with Path(base, f"{partition}.pickle").open("rb") as file:
            return pickle.load(file)

    @staticmethod
    def getQueryNeighbors(
        embedding: Type[TextEmbedding], partition: PartitionType
    ) -> List[Tuple[List[int], List[float]]]:
        base = Path(workspace, f"msMarco/queryNeighbors/{embedding.name}")
        with Path(base, f"{partition}.pickle").open("rb") as file:
            return pickle.load(file)

    @staticmethod
    def newMixEmbeddingLoader(
        embedding: Type[TextEmbedding],
        partition: PartitionType,
        numPassages: int,
        batchSize: int,
        shuffle: bool,
        numWorkers: int,
    ) -> DataLoader:
        queryBase = Path(
            workspace,
            f"msMarco/queryEmbeddings/{embedding.name}/{partition}",
        )
        passageBase = Path(
            workspace,
            f"msMarco/passageEmbeddings/{embedding.name}",
        )
        queryNeighbors = [
            indices[:numPassages]
            for indices, _ in MsMarcoDataset.getQueryNeighbors(embedding, partition)
        ]
        return newMixEmbeddingLoaderFrom(
            queryBase, passageBase, queryNeighbors, batchSize, shuffle, numWorkers
        )


def preparePassages(numShards: int):
    """
    Prepare the passage loader.
    """
    base = Path(workspace, "msMarco/passages")
    base.mkdir(mode=0o770, parents=True, exist_ok=True)

    logger.info("Download the passages from the official website")
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
    logger.info("Completed!")

    logger.info("Extract the passages from the tarball.")
    subprocess.run(
        ["tar", "-xzvf", "collection.tar.gz"],
        cwd=base,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    path.unlink()
    logger.info("Completed!")

    logger.info("Split the passages into shards.")
    path = Path(base, "collection.tsv")
    shards = [([], []) for _ in range(numShards)]
    with tqdm(
        total=path.stat().st_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress:
        with path.open("r", encoding="utf-8") as file:
            for i, line in enumerate(file):
                pid, passage = line.strip().split("\t")
                _, shardIdx = divmod(i, numShards)
                shards[shardIdx][0].append(pid)
                shards[shardIdx][1].append(passage)
                progress.update(len(line.encode()))
    logger.info("Completed!")

    logger.info("Write the shards to disk.")
    with tqdm(total=numShards) as progress:
        for i in range(numShards):
            pids, passages = shards[i]
            table = pa.Table.from_pydict({"pid": pids, "passage": passages})
            pq.write_table(table, Path(base, f"{i:08d}.parquet"))
            progress.update()
    path.unlink()
    logger.info("Completed!")


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

    logger.info("Load the passages from disk.")
    loader = MsMarcoDataset.newPassageLoader(1, False, 1)
    logger.info("Completed!")

    logger.info("Split the shards with co-workers.")
    shards: List[List[NDArray[np.float32]]] = [[] for _ in range(numShards)]
    batchIdx, batchPsg = [], []
    logger.info("Completed!")

    logger.info("Generate the embeddings.")

    def batchCompute():
        vectors = embedding.forward(batchPsg)
        for j, x in zip(batchIdx, vectors):
            _, shardIdx = divmod(j, numShards)
            shards[shardIdx].append(x)
        batchIdx.clear()
        batchPsg.clear()
        cuda.empty_cache()

    for i, (_, passage) in enumerate(tqdm(loader.dataset)):
        assert len(batchIdx) == len(batchPsg)
        _, shardIdx = divmod(i, numShards)
        if shardIdx % workerCnt == workerIdx:
            batchIdx.append(i)
            batchPsg.append(passage)
            if len(batchIdx) >= batchSize:
                batchCompute()
    if batchIdx:
        batchCompute()
    logger.info("Completed!")

    logger.info("Write the shards to disk.")
    for i, shard in enumerate(shards):
        if i % workerCnt == workerIdx:
            buffer = np.stack(shard, dtype=np.float32)
            np.save(Path(base, f"{i:08d}.npy"), buffer)
    logger.info("Completed!")


def prepareQueries():
    """
    Prepare the query loader.
    """
    base = Path(workspace, "msMarco/queries")
    base.mkdir(mode=0o770, parents=True, exist_ok=True)

    logger.info("Download the queries from the official website")
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
    logger.info("Completed!")

    logger.info("Extract the queries from the tarball.")
    subprocess.run(
        ["tar", "-xzvf", "queries.tar.gz"],
        cwd=base,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    path.unlink()
    logger.info("Completed!")

    logger.info("Refactor the queries into parquet files.")
    for partition in ["train", "dev", "eval"]:
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
                    pid, query = line.strip().split("\t")
                    qids.append(pid)
                    queries.append(query)
                    progress.update(len(line.encode()))
        table = pa.Table.from_pydict({"qid": qids, "query": queries})
        pq.write_table(table, Path(base, f"{partition}.parquet"))
        path.unlink()
    logger.info("Completed!")


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

    logger.info("Load the queries from disk")
    loader = MsMarcoDataset.newQueryLoader(partition, 1, False, 1)

    logger.info("Split the shards with co-workers")
    shards: List[List[NDArray[np.float32]]] = [[] for _ in range(numShards)]

    match embedding.name:
        case "miniCPM":
            instruction = "Query: {x}"
        case "bgeBase":
            instruction = "Represent this sentence for searching relevant passages: {x}"
        case _:
            instruction = "{x}"

    logger.info("Generate the embeddings")
    for i, (_, query) in enumerate(tqdm(loader.dataset)):
        assert len(batchIdx) == len(batchQry)
        _, shardIdx = divmod(i, numShards)
        if shardIdx % workerCnt == workerIdx:
            batchIdx.append(i)
            batchQry.append(instruction.format(x=query))
            if len(batchIdx) >= batchSize:
                compute()
    if batchIdx:
        compute()

    logger.info("Write the shards to disk")
    for i, shard in enumerate(shards):
        if i % workerCnt == workerIdx:
            buffer = np.stack(shard, dtype=np.float32)
            np.save(Path(base, f"{i:08d}.npy"), buffer)


def prepareQueryRelevance(partition: PartitionType):
    """
    Prepare the query relevance judgments.
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


def prepareQueryNeighbors(
    partition: PartitionType,
    embedding: Type[TextEmbedding],
    retriever: DotProductRetriever,
    batchSize: int,
    topK: int,
):
    """
    Prepare the query neighbors.
    """
    base = Path(workspace, f"msMarco/queryNeighbors/{embedding.name}")
    base.mkdir(mode=0o770, parents=True, exist_ok=True)

    logger.info("Load the passage embeddings")
    loader = MsMarcoDataset.newPassageEmbeddingLoader(embedding, batchSize, False, 4)
    for batch in tqdm(loader):
        retriever.add(batch)

    logger.info("Compute the query neighbors")
    data: List[Tuple[List[int], List[float]]] = []
    loader = MsMarcoDataset.newQueryEmbeddingLoader(
        embedding, partition, batchSize, False, 4
    )
    for batch in tqdm(loader):
        indices, scores = retriever.search(batch, topK)
        data.extend(zip(indices, scores))

    logger.info("Write the query neighbors to disk")
    with Path(base, f"{partition}.pickle").open("wb") as file:
        pickle.dump(data, file)


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
    prepareQueryRelevanceParser = subparsers.add_parser("prepareQueryRelevance")
    prepareQueryRelevanceParser.add_argument("--partition", type=str, required=True)
    prepareQueryNeighborsParser = subparsers.add_parser("prepareQueryNeighbors")
    prepareQueryNeighborsParser.add_argument("--partition", type=str, required=True)
    prepareQueryNeighborsParser.add_argument("--embedding", type=str, required=True)
    prepareQueryNeighborsParser.add_argument("--gpuDevice", type=int, nargs="+", required=True)
    prepareQueryNeighborsParser.add_argument("--batchSize", type=int, required=True)
    prepareQueryNeighborsParser.add_argument("--topK", type=int, required=True)
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
        case "prepareQueryRelevance":
            prepareQueryRelevance(parsed.partition)
        case "prepareQueryNeighbors":
            match parsed.embedding:
                case "miniCPM":
                    embedding = MiniCPM
                case "bgeBase":
                    embedding = BgeBase
                case _:
                    raise NotImplementedError
            retriever = DotProductRetriever(embedding.size, parsed.gpuDevice)
            prepareQueryNeighbors(
                parsed.partition,
                embedding,
                retriever,
                parsed.batchSize,
                parsed.topK,
            )
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
