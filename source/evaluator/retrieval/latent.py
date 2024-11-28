"""
Evaluate the retrieval performance on the reconstructed embeddings.
"""

import argparse
from pathlib import Path
import torch
from source import logger
from source.autoencoder import KSparseAutoencoder
from source.trainer import workspace as trainerWorkspace
from source.utilities import parseInt, tqdm
from source.embedding import MiniCPM, BgeBase
from source.dataset.textRetrieval import MsMarcoDataset
from source.retriever.dense import DotProductRetriever
from source.retriever.utilities import evaluateRetrieval
from elasticsearch import Elasticsearch, helpers, BadRequestError


class Evaluator:
    """
    The reconstructed baseline.
    """

    def __init__(self):
        # Parse the arguments.
        parser = argparse.ArgumentParser()
        parser.add_argument("--embedding", type=str, required=True)
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument("--latentSize", type=parseInt, required=True)
        parser.add_argument("--latentTopK", type=int, required=True)
        parser.add_argument("--modelGpuDevice", type=int, required=True)
        parser.add_argument("--modelName", type=str, required=True)
        parser.add_argument("--retrieveTopK", type=int, required=True)
        parsed = parser.parse_args()

        # Match the embedding model.
        match parsed.embedding:
            case "miniCPM":
                self.embedding = MiniCPM
            case "bgeBase":
                self.embedding = BgeBase
            case _:
                raise NotImplementedError()

        # Match the retrieval dataset.
        match parsed.dataset:
            case "msMarco":
                self.dataset = MsMarcoDataset
            case _:
                raise NotImplementedError()

        # Create the autoencoder.
        self.model = KSparseAutoencoder(
            self.embedding.size,
            parsed.latentSize,
            parsed.latentTopK,
        )

        # Load back the weights.
        snapFile = Path(trainerWorkspace, parsed.modelName, "snapshot-best.pth")
        snapShot = torch.load(snapFile)
        logger.info("%s Iteration: %d", parsed.modelName, snapShot["lastEpoch"])
        self.model.load_state_dict(snapShot["model"])
        self.model = self.model.to(parsed.modelGpuDevice)

        # Map from passage index to passage id.
        logger.info("Build the passage lookup.")
        self.passageLookup, i = dict(), 0
        for pids, _ in tqdm(MsMarcoDataset.newPassageLoader(4096, False, 4)):
            for x in pids:
                self.passageLookup[i] = x
                i += 1

        # Build the sparse retriever.
        logger.info("Build the sparse retriever.")
        self.passageLoader = self.dataset.newPassageEmbeddingLoader(
            self.embedding,
            batchSize=4096,
            shuffle=False,
            numWorkers=4,
        )
        self.retriever = Elasticsearch(
            hosts=[{"host": "127.0.0.1", "port": 9200, "scheme": "http"}],
            request_timeout=600,
        )
        try:
            self.retriever.indices.create(
                index=parsed.modelName.lower(),
                body={
                    "mappings": {"properties": {"vector": {"type": "sparse_vector"}}}
                },
            )
        except BadRequestError as e:
            if e.message != "resource_already_exists_exception":
                raise e

        # Index the latent vectors.
        i, batch = 0, []
        with torch.no_grad():
            for passages in tqdm(self.passageLoader):
                passages = passages.to(parsed.modelGpuDevice)
                latent = self.model.encode(passages)
                pack = torch.topk(latent, parsed.latentTopK)
                index, value = pack.indices.cpu().numpy(), pack.values.cpu().numpy()
                for idx, val in zip(index, value):
                    batch.append(
                        {
                            "_index": parsed.modelName.lower(),
                            "_id": self.passageLookup[i],
                            "_source": {
                                "vector": {str(x): y for x, y in zip(idx, val) if y > 0}
                            },
                        }
                    )
                    i += 1
                helpers.bulk(self.retriever, batch)
                batch.clear()

        # Map from query index to query id.
        logger.info("Build the query lookup.")
        self.queryLookup, i = {}, 0
        for qids, _ in tqdm(MsMarcoDataset.newQueryLoader("dev", 4096, False, 4)):
            for x in qids:
                self.queryLookup[i] = x
                i += 1

        # Create the query loader.
        self.queryLoader = self.dataset.newQueryEmbeddingLoader(
            self.embedding,
            partition="dev",
            batchSize=512,
            shuffle=False,
            numWorkers=4,
        )

        # Set the attributes.
        self.parsedDataset = parsed.dataset
        self.parsedModelName = parsed.modelName
        self.retrieveTopK = parsed.retrieveTopK
        self.modelGpuDevice = parsed.modelGpuDevice

    def dispatch(self):
        """
        Dispatch the experiment.
        """
        # fmt: off
        relevance = self.dataset.getQueryRelevance("dev")
        retrieved, i, batchQry, batchQid = {}, 0, [], []
        logger.info("Retrieve with sparse latents.")
        with torch.no_grad():
            for queries in tqdm(self.queryLoader):
                queries = queries.to(self.modelGpuDevice)
                latent = self.model.encode(queries)
                pack = torch.topk(latent, self.retrieveTopK)
                index, value = pack.indices.cpu().numpy(), pack.values.cpu().numpy()
                for idx, val in zip(index, value):
                    batchQry.append({"index": self.parsedModelName.lower()})
                    batchQry.append({"query": {
                        "sparse_vector": {
                            "field": "vector",
                            "query_vector": {str(x): y for x, y in zip(idx, val) if y > 0},
                        }}, 
                        "size": self.retrieveTopK
                    })
                    batchQid.append(self.queryLookup[i])
                    i += 1
                rsp = self.retriever.msearch(
                    body=batchQry,
                    max_concurrent_searches=32,
                )
                for j, res in enumerate(rsp["responses"]):
                    qid = batchQid[j]
                    retrieved[qid] = [hit["_id"] for hit in res["hits"]["hits"]]
                batchQry.clear()
                batchQid.clear()
        evaluated = Path(f"{self.parsedDataset}.{self.parsedModelName}.log")
        evaluateRetrieval(relevance, retrieved, evaluated)


if __name__ == "__main__":
    E = Evaluator()
    E.dispatch()
