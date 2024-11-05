"""
Implementation of the MS MARCO dataset.
"""

import pickle
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Type
from hashlib import md5
import requests
import pyarrow as pa
import pyarrow.parquet as pq
import torch.cuda as cuda
import numpy as np
from source import console
from source.interface import PartitionType, TextRetrievalDataset, TextEmbedding
from source.dataset import workspace
from source.dataset.utilities import (
    textRetrievalGetPassages,
    textRetrievalGetPassageEmbeddings,
    textRetrievalGetQueries,
    textRetrievalGetQueryEmbeddings,
    textRetrievalGetRelevantPassages,
)
from source.embedding.miniCPM import MiniCPM
from source.embedding.bgeBase import BgeBase
from source.retriever.faiss import FaissRetriever


class MsMarco(TextRetrievalDataset):
    """
    Implementation of the MS MARCO dataset.
    """

    name = "MsMarco"

    def __init__(self) -> None:
        pass

    def getPassages(self):
        base = GetPassagesInit.base
        return textRetrievalGetPassages(base)

    def getPassageEmbeddings(self, embedding):
        base = GetPassageEmbeddingsInit.base
        base = Path(base, embedding.name)
        return textRetrievalGetPassageEmbeddings(base)

    def getQueries(self, partition):
        base = GetQueriesInit.base
        base = Path(base, partition)
        return textRetrievalGetQueries(base)

    def getQueryEmbeddings(self, partition, embedding):
        base = GetQueryEmbeddingsInit.base
        base = Path(base, partition, embedding.name)
        return textRetrievalGetQueryEmbeddings(base)

    def getRelevantPassages(self, partition):
        base = GetRelevantPassagesInit.base
        file = Path(base, f"{partition}.pkl")
        return textRetrievalGetRelevantPassages(file)

    def getNeighborPassages(self, partition, embedding):
        raise NotImplementedError


class GetPassagesInit:
    """
    Prepare the passages in the dataset.

    Attributes:
        N: The number of partitions.
        base: The base path for the passages.
    """

    base = Path(workspace, "MsMarco/passages")

    def __init__(self, numPartitions: int) -> None:
        self.numPartitions = numPartitions
        self.base.mkdir(mode=0o770, parents=True, exist_ok=True)
        self.dispatch()

    def step1(self) -> None:
        """
        Dispatch the download of the passages.
        """
        console.log("Downloading the passages")
        host = "https://msmarco.z22.web.core.windows.net"
        link = f"{host}/msmarcoranking/collection.tar.gz"
        with requests.get(link, stream=True, timeout=1800) as response:
            with Path(self.base, "collection.tar.gz").open("wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

    def step2(self) -> None:
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

    def step3(self) -> None:
        """
        Dispatch the partitioning of the passages.
        """
        console.log("Partitioning the passages")
        N = self.numPartitions
        ids: List[List[str]] = [[] for _ in range(N)]
        texts: List[List[str]] = [[] for _ in range(N)]
        path = Path(self.base, "collection.tsv")
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                x0, x1 = [p.strip() for p in line.split("\t")]
                i = int(md5(x0.encode()).hexdigest(), 16) % N
                ids[i].append(x0)
                texts[i].append(x1)
        for i in range(N):
            table = pa.Table.from_pydict({"id": ids[i], "text": texts[i]})
            pq.write_table(table, Path(self.base, f"partition-{i:08d}.parquet"))
        Path(self.base, "collection.tsv").unlink()

    def dispatch(self) -> None:
        """
        Dispatch the steps.
        """
        self.step1()
        self.step2()
        self.step3()


class GetQueriesInit:
    """
    Prepare the queries in the dataset.

    Attributes:
        base: The base path for the queries.
    """

    base = Path(workspace, "MsMarco/queries")

    def __init__(self, numPartitions: int) -> None:
        self.numPartitions = numPartitions
        self.base.mkdir(mode=0o770, parents=True, exist_ok=True)
        self.dispatch()

    def step1(self) -> None:
        """
        Dispatch the download of the queries.
        """
        console.log("Downloading the queries")
        host = "https://msmarco.z22.web.core.windows.net"
        link = f"{host}/msmarcoranking/queries.tar.gz"
        with requests.get(link, stream=True, timeout=1800) as response:
            with Path(self.base, "queries.tar.gz").open("wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

    def step2(self) -> None:
        """
        Dispatch the extraction of the queries.
        """
        console.log("Extracting the queries")
        subprocess.run(
            ["tar", "-xzvf", "queries.tar.gz"],
            cwd=self.base,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        Path(self.base, "queries.tar.gz").unlink()
        Path(self.base, "queries.eval.tsv").unlink()

    def step3(self) -> None:
        """
        Dispatch the partitioning of the queries.
        """
        console.log("Partitioning the queries")
        N = self.numPartitions
        choices: List[PartitionType] = ["train", "dev"]
        for partition in choices:
            base = Path(self.base, partition)
            base.mkdir(mode=0o770, exist_ok=True)
            ids: List[List[str]] = [[] for _ in range(N)]
            texts: List[List[str]] = [[] for _ in range(N)]
            path = Path(self.base, f"queries.{partition}.tsv")
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    x0, x1 = [p.strip() for p in line.split("\t")]
                    i = int(md5(x0.encode()).hexdigest(), 16) % N
                    ids[i].append(x0)
                    texts[i].append(x1)
            for i in range(N):
                table = pa.Table.from_pydict({"id": ids[i], "text": texts[i]})
                pq.write_table(table, Path(base, f"partition-{i:08d}.parquet"))
            path.unlink()

    def dispatch(self) -> None:
        """
        Dispatch the steps.
        """
        self.step1()
        self.step2()
        self.step3()


class GetPassageEmbeddingsInit:
    """
    Prepare the embeddings of the passages in the dataset.

    Attributes:
        base: The base path for the embeddings.
    """

    base = Path(workspace, "MsMarco/passageEmbeddings")

    def __init__(
        self,
        embedding: TextEmbedding,
        numPartitions: int,
        partitionIndex: int,
        batchSize: int,
    ) -> None:
        self.embedding = embedding
        self.numPartitions = numPartitions
        self.partitionIndex = partitionIndex
        self.batchSize = batchSize
        self.base.mkdir(mode=0o770, parents=True, exist_ok=True)
        self.dispatch()

    def dispatch(self) -> None:
        """
        Dispatch the steps.
        """
        console.log("Loading the passages")
        base = Path(self.base, self.embedding.name)
        base.mkdir(mode=0o770, exist_ok=True)
        passages = list(MsMarco().getPassages())
        I, N = self.partitionIndex, self.numPartitions
        M = len(passages) // N
        s = I * M
        t = len(passages) if I == N - 1 else (I + 1) * M
        passages = passages[s:t]

        console.log("Embedding the passages")
        ids = np.empty((len(passages),), dtype="U16")
        vectors = np.empty((len(passages), self.embedding.size), dtype=np.float32)
        B = self.batchSize
        for i in range(0, len(passages), B):
            console.log(f"Progress: {i}/{len(passages)}")
            parts = passages[i : i + B]
            batchIDs, batchTexts = zip(*parts)
            batchVectors = self.embedding.forward(list(batchTexts))
            ids[i : i + B] = batchIDs
            vectors[i : i + B] = batchVectors
        np.savez_compressed(
            Path(base, f"partition-{I:08d}.npz"),
            ids=ids,
            vectors=vectors,
        )


class GetQueryEmbeddingsInit:
    """
    Prepare the embeddings of the queries in the dataset.

    Attributes:
        base: The base path for the embeddings.
    """

    base = Path(workspace, "MsMarco/queryEmbeddings")

    def __init__(
        self,
        embedding: TextEmbedding,
        numPartitions: int,
        partitionIndex: int,
        batchSize: int,
    ) -> None:
        self.embedding = embedding
        self.numPartitions = numPartitions
        self.partitionIndex = partitionIndex
        self.batchSize = batchSize
        self.base.mkdir(mode=0o770, parents=True, exist_ok=True)
        self.dispatch()

    def dispatch(self) -> None:
        """
        Dispatch the steps.
        """
        choices: List[PartitionType] = ["train", "dev"]
        for partition in choices:
            console.log(f"Loading the queries from {partition}")
            base = Path(self.base, partition, self.embedding.name)
            base.mkdir(mode=0o770, parents=True, exist_ok=True)
            queries = list(MsMarco().getQueries(partition))
            I, N = self.partitionIndex, self.numPartitions
            M = len(queries) // N
            s = I * M
            t = len(queries) if I == N - 1 else (I + 1) * M
            queries = queries[s:t]

            console.log(f"Embedding the queries from {partition}")
            ids = np.empty((len(queries),), dtype="U16")
            vectors = np.empty((len(queries), self.embedding.size), dtype=np.float32)
            B = self.batchSize
            for i in range(0, len(queries), B):
                console.log(f"Progress: {i}/{len(queries)}")
                parts = queries[i : i + B]
                batchIDs, batchTexts = zip(*parts)
                batchVectors = self.embedding.forward(list(batchTexts))
                ids[i : i + B] = batchIDs
                vectors[i : i + B] = batchVectors
            np.savez_compressed(
                Path(base, f"partition-{I:08d}.npz"),
                ids=ids,
                vectors=vectors,
            )


class GetRelevantPassagesInit:
    """
    Prepare the relevant passages in the dataset.

    Attributes:
        base: The base path for the relevant passages.
    """

    base = Path(workspace, "MsMarco/relevantPassages")

    def __init__(self) -> None:
        self.base.mkdir(mode=0o770, parents=True, exist_ok=True)
        self.dispatch()

    def step1(self) -> None:
        """
        Dispatch the download of the relevance.
        """
        console.log("Downloading the relevance")
        host = "https://msmarco.z22.web.core.windows.net"
        link = f"{host}/msmarcoranking/qrels.train.tsv"
        with requests.get(link, stream=True, timeout=1800) as response:
            with Path(self.base, "qrels.train.tsv").open("wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
        link = f"{host}/msmarcoranking/qrels.dev.tsv"
        with requests.get(link, stream=True, timeout=1800) as response:
            with Path(self.base, "qrels.dev.tsv").open("wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

    def step2(self) -> None:
        """
        Dispatch the extraction of the relevance.
        """
        console.log("Extracting the relevance")
        choices: List[PartitionType] = ["train", "dev"]
        for partition in choices:
            data: Dict[str, Dict[str, int]] = {}
            path = Path(self.base, f"qrels.{partition}.tsv")
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    queryID, _, passageID, relevance = line.split()
                    if queryID not in data:
                        data[queryID] = {}
                    data[queryID][passageID] = int(relevance)
            with Path(self.base, f"{partition}.pkl").open("wb") as file:
                pickle.dump(data, file)
            path.unlink()

    def dispatch(self) -> None:
        """
        Dispatch the steps.
        """
        self.step1()
        self.step2()


class GetNeighborPassagesInit:
    """
    Prepare the neighbor passages in the dataset.

    Attributes:
        base: The base path for the neighbor passages.
    """

    base = Path(workspace, "MsMarco/neighborPassages")

    def __init__(self, embedding: Type[TextEmbedding], devices: List[int]) -> None:
        self.embedding = embedding
        self.retriever = FaissRetriever(embedding.size, devices)
        self.base.mkdir(mode=0o770, parents=True, exist_ok=True)
        self.dispatch()

    def dispatch(self) -> None:
        """
        Dispatch the steps.
        """
        console.log("Loading the passages")
        dataset, topK = MsMarco(), 256
        iterator = dataset.getPassageEmbeddings(self.embedding)
        ids: List[str] = []
        vectors: List[np.ndarray] = []
        for passageID, passageEmbedding in iterator:
            ids.append(passageID)
            vectors.append(passageEmbedding)
        self.retriever.add(ids, np.array(vectors))

        batchSize = 256
        choices: List[PartitionType] = ["train", "dev"]
        for partition in choices:
            data: Dict[str, Dict[str, float]] = {}
            console.log(f"Loading the queries from {partition}")
            iterator = dataset.getQueryEmbeddings(partition, self.embedding)
            ids: List[str] = []
            vectors: List[np.ndarray] = []
            for i, (queryID, queryEmbedding) in enumerate(iterator):
                ids.append(queryID)
                vectors.append(queryEmbedding)
                if len(vectors) == batchSize:
                    vectors = np.array(vectors)
                    results, scores = self.retriever.search(vectors, topK)
                    for j, (result, score) in enumerate(zip(results, scores)):
                        data[ids[j]] = {r: s for r, s in zip(result, score)}
                    ids = []
                    vectors = []
            if len(vectors) > 0:
                vectors = np.array(vectors)
                results, scores = self.retriever.search(vectors, topK)
                for j, (result, score) in enumerate(zip(results, scores)):
                    data[ids[j]] = {r: s for r, s in zip(result, score)}
            with Path(self.base, f"{partition}.pkl").open("wb") as file:
                pickle.dump(data, file)


def main() -> None:
    """
    The entry point.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    getPassages = subparsers.add_parser("getPassages")
    getPassages.add_argument("--numPartitions", type=int, required=True)
    getQueries = subparsers.add_parser("getQueries")
    getQueries.add_argument("--numPartitions", type=int, required=True)
    getPassageEmbeddings = subparsers.add_parser("getPassageEmbeddings")
    getPassageEmbeddings.add_argument("--embedding", type=str, required=True)
    getPassageEmbeddings.add_argument("--numPartitions", type=int, required=True)
    getPassageEmbeddings.add_argument("--partitionIndex", type=int, required=True)
    getPassageEmbeddings.add_argument("--batchSize", type=int, required=True)
    getQueryEmbeddings = subparsers.add_parser("getQueryEmbeddings")
    getQueryEmbeddings.add_argument("--embedding", type=str, required=True)
    getQueryEmbeddings.add_argument("--numPartitions", type=int, required=True)
    getQueryEmbeddings.add_argument("--partitionIndex", type=int, required=True)
    getQueryEmbeddings.add_argument("--batchSize", type=int, required=True)
    subparsers.add_parser("getRelevantPassages")
    getNeighborPassages = subparsers.add_parser("getNeighborPassages")
    getNeighborPassages.add_argument("--embedding", type=str, required=True)
    getNeighborPassages.add_argument("--devices", type=int, nargs="+", required=True)

    parsed = parser.parse_args()
    match parsed.command:
        case "getPassages":
            GetPassagesInit(parsed.numPartitions)
        case "getQueries":
            GetQueriesInit(parsed.numPartitions)
        case "getPassageEmbeddings":
            devices = list(range(cuda.device_count()))
            match parsed.embedding:
                case "MiniCPM":
                    embedding = MiniCPM(devices)
                case "BgeBase":
                    embedding = BgeBase(devices)
                case _:
                    raise NotImplementedError(parsed.embedding)
            GetPassageEmbeddingsInit(
                embedding,
                parsed.numPartitions,
                parsed.partitionIndex,
                parsed.batchSize,
            )
        case "getQueryEmbeddings":
            devices = list(range(cuda.device_count()))
            match parsed.embedding:
                case "MiniCPM":
                    embedding = MiniCPM(devices)
                case "BgeBase":
                    embedding = BgeBase(devices)
                case _:
                    raise NotImplementedError(parsed.embedding)
            GetQueryEmbeddingsInit(
                embedding,
                parsed.numPartitions,
                parsed.partitionIndex,
                parsed.batchSize,
            )
        case "getRelevantPassages":
            GetRelevantPassagesInit()
        case "getNeighborPassages":
            match parsed.embedding:
                case "MiniCPM":
                    embedding = MiniCPM
                case "BgeBase":
                    embedding = BgeBase
                case _:
                    raise NotImplementedError(parsed.embedding)
            GetNeighborPassagesInit(embedding, parsed.devices)


if __name__ == "__main__":
    main()
