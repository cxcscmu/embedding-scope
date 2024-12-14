"""
Evaluate the retrieval performance using the sparse latent features.
"""

import argparse
from pathlib import Path
import torch
from elasticsearch import Elasticsearch, BadRequestError
from elasticsearch.helpers import bulk
from source.embedding import BgeBase, MiniCPM
from source.dataset.textRetrieval import MsMarcoDataset
from source.utilities import parseInt, tqdm
from source.autoencoder import KSparseAutoencoder
from source.trainer import workspace as trainerWorkspace
from source.evaluator.utilities import evaluateRetrieval


class Evaluator:
    """
    The evaluator for the retrieval task.

    It first computes the latent features for each document in the dataset
    and build the index with Elasticsearch. Then, it retrieves the top-k
    documents for each query and computes the retrieval performance. For
    simpliciy, we've disabled the security features of Elasticsearch.

    References:
    -----------
    - https://www.elastic.co/guide/en/elasticsearch/reference/current/sparse-vector.html
    - https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-sparse-vector-query.html
    """

    def __init__(self) -> None:
        """
        Initialize the evaluator.
        """

        # Parse the arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument("--embedding", type=str, required=True)
        parser.add_argument("--latentSize", type=parseInt, required=True)
        parser.add_argument("--latentTopK", type=int, required=True)
        parser.add_argument("--modelName", type=str, required=True)
        parser.add_argument("--modelDevice", type=int)
        parser.add_argument("--indexBatchSize", type=int, default=2048)
        parser.add_argument("--queryBatchSize", type=int, default=64)
        parser.add_argument("--retrieveTopK", type=int, default=100)
        parsed = parser.parse_args()

        # Match the retrieval dataset.
        match parsed.dataset:
            case "msMarco":
                self.dataset = MsMarcoDataset
            case _:
                raise NotImplementedError()

        # Match the embedding model.
        match parsed.embedding:
            case "bgeBase":
                self.embedding = BgeBase
            case "miniCPM":
                self.embedding = MiniCPM
            case _:
                raise NotImplementedError()

        # Load back the autoencoder.
        self.model = KSparseAutoencoder(
            self.embedding.size,
            parsed.latentSize,
            parsed.latentTopK,
        )
        snapshotFile = Path(trainerWorkspace, parsed.modelName, "snapshot-best.pth")
        snapshotDict = torch.load(snapshotFile, map_location="cpu")
        self.model.load_state_dict(snapshotDict["model"])
        self.model = self.model.to(parsed.modelDevice)

        # Connect to the Elasticsearch.
        self.client = Elasticsearch(
            hosts=[{"host": "127.0.0.1", "port": 9200, "scheme": "http"}],
            request_timeout=600,
        )

        # Set the attributes.
        self.latentTopK: int = parsed.latentTopK
        self.modelName: str = parsed.modelName
        self.modelDevice: int = parsed.modelDevice
        self.indexBatchSize: int = parsed.indexBatchSize
        self.queryBatchSize: int = parsed.queryBatchSize
        self.indexName = self.modelName.lower()
        self.retrieveTopK: int = parsed.retrieveTopK

    def buildIndex(self) -> None:
        """
        Build the index with Elasticsearch.
        """
        # Create the index.
        try:
            self.client.indices.create(
                index=self.indexName,
                mappings={
                    "properties": {
                        "features": {
                            "type": "sparse_vector",
                        }
                    }
                },
            )
        except BadRequestError as e:
            if e.message != "resource_already_exists_exception":
                raise e

        # Build the lookup from index to passage ID.
        lookup, i = {}, 0
        iterable = self.dataset.newPassageLoader(
            self.indexBatchSize,
            shuffle=False,
            numWorkers=4,
        )
        for pids, _ in tqdm(iterable):
            for x in pids:
                lookup[i] = x
                i += 1

        # Index the sparse latent features.
        batch, i = [], 0
        iterable = self.dataset.newPassageEmbeddingLoader(
            self.embedding,
            batchSize=self.indexBatchSize,
            shuffle=False,
            numWorkers=4,
        )
        for passages in tqdm(iterable):
            passages = passages.to(self.modelDevice)
            latent = self.model.encode(passages)
            pack = torch.topk(latent, self.latentTopK)
            for idx, val in zip(pack.indices, pack.values):
                batch.append(
                    {
                        "_index": self.indexName,
                        "_id": lookup[i],
                        "_source": {
                            "features": {
                                str(x.item()): y.item()
                                for x, y in zip(idx, val)
                                if y > 0
                            }
                        },
                    }
                )
                i += 1
            bulk(self.client, batch)
            batch.clear()

    def evaluateRetrieval(self) -> None:
        """
        Evaluate the retrieval performance.
        """
        # Specify the retrieved storage.
        relevance = self.dataset.getQueryRelevance(partition="dev")
        retrieved = {}

        # Build the lookup from index to query ID.
        lookup, i = {}, 0
        iterable = self.dataset.newQueryLoader(
            partition="dev",
            batchSize=self.queryBatchSize,
            shuffle=False,
            numWorkers=4,
        )
        for qids, _ in tqdm(iterable):
            for x in qids:
                lookup[i] = x
                i += 1

        # Retrieve the top-k passages.
        batch, i = [], 0
        iterable = self.dataset.newQueryEmbeddingLoader(
            self.embedding,
            partition="dev",
            batchSize=self.queryBatchSize,
            shuffle=False,
            numWorkers=4,
        )
        for queries in tqdm(iterable):
            queries = queries.to(self.modelDevice)
            latent = self.model.encode(queries)
            pack = torch.topk(latent, self.retrieveTopK)
            for idx, val in zip(pack.indices, pack.values):
                batch.append({"index": self.indexName})
                batch.append(
                    {
                        "query": {
                            "sparse_vector": {
                                "field": "features",
                                "query_vector": {
                                    str(x.item()): y.item()
                                    for x, y in zip(idx, val)
                                    if y > 0
                                },
                            }
                        },
                        "size": self.retrieveTopK,
                    }
                )
            response = self.client.msearch(
                body=batch,
                max_concurrent_searches=32,
            )
            for result in response["responses"]:
                retrieved[lookup[i]] = [x["_id"] for x in result["hits"]["hits"]]
                i += 1
            batch.clear()

        # Save the evaluation results.
        evaluated = Path(f"logging/{self.modelName}.log")
        evaluateRetrieval(relevance, retrieved, evaluated)

    @torch.no_grad()
    def run(self) -> None:
        """
        Run the evaluation.
        """
        self.buildIndex()
        self.evaluateRetrieval()


if __name__ == "__main__":
    E = Evaluator()
    E.run()
