"""
Evaluate the retrieval performance on the reconstructed embeddings.

@author: Hao Kang <haok@andrew.cmu.edu>
@date: December 27, 2024
"""

import argparse
from pathlib import Path

import torch

from source import logger
from source.utilities import parseInt, tqdm
from source.dataset.textRetrieval import MsMarcoDataset
from source.embedding import BgeBase, MiniCPM
from source.autoencoder import KSparseAutoencoder
from source.trainer import workspace as trainerWorkspace
from source.retriever import DenseRetriever
from source.evaluator import workspace as evaluatorWorkspace
from source.evaluator.utilities import evaluateRetrieval


class Pipeline:
    """
    The evaluation pipeline.

    This pipeline evaluates the retrieval performance on the reconstructed
    embeddings. It first reconstructs the embeddings associated with each
    documents and indexes them using the dense retriever. Then, it retrieves
    the top-k documents for each query and evaluates the retrieval performance.
    """

    def __init__(self):
        """
        Run the evaluation pipeline.
        """

        # Parse the arguments
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument("--embedding", type=str, required=True)
        parser.add_argument("--latentSize", type=parseInt, required=True)
        parser.add_argument("--latentTopK", type=int, required=True)
        parser.add_argument("--modelName", type=str, required=True)
        parser.add_argument("--modelDevice", type=int, required=True)
        parser.add_argument("--indexDevice", type=int, nargs="+", required=True)
        parser.add_argument("--indexBatchSize", type=int, default=2048)
        parser.add_argument("--queryBatchSize", type=int, default=512)
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

        # Load back the sparse autoencoder.
        self.model = KSparseAutoencoder(
            self.embedding.size, parsed.latentSize, parsed.latentTopK
        )
        snapshotFile = Path(trainerWorkspace, parsed.modelName, "snapshot-best.pth")
        snapshotDict = torch.load(snapshotFile, map_location="cpu")
        self.model.load_state_dict(snapshotDict["model"])
        self.model = self.model.to(parsed.modelDevice)
        self.model.eval()

        # Set the attributes.
        self.latentTopK: int = parsed.latentTopK
        self.modelName: str = parsed.modelName
        self.modelDevice: int = parsed.modelDevice
        self.indexBatchSize: int = parsed.indexBatchSize
        self.queryBatchSize: int = parsed.queryBatchSize

        # Build the index and evaluate the retrieval.
        with DenseRetriever(size=self.embedding.size, devices=parsed.indexDevice) as self.retriever:
            self.build_index()
            self.evaluate_retrieval()

    @torch.no_grad()
    def build_index(self) -> None:
        """
        Build the index with the latent features.
        """
        logger.info("Index the latent features.")

        # Iterator over passage IDs and embeddings.
        iter0 = self.dataset.newPassageLoader(
            self.indexBatchSize,
            shuffle=False,
            numWorkers=4,
        )
        iter1 = self.dataset.newPassageEmbeddingLoader(
            self.embedding,
            batchSize=self.indexBatchSize,
            shuffle=False,
            numWorkers=4,
        )

        for (pids, _), passages in tqdm(zip(iter0, iter1), total=len(iter0)):
            passages = passages.to(self.modelDevice)
            _, xhat = self.model.forward(passages)

            # Map from passage ID to dense vector.
            self.retriever.batch_index(
                {
                    pid: xhat[i].cpu().numpy()
                    for i, pid in enumerate(pids)
                }
            )

    @torch.no_grad()
    def evaluate_retrieval(self) -> None:
        """
        Evaluate the retrieval performance.
        """
        logger.info("Evaluate the retrieval performance.")

        # Iterator over query IDs and embeddings.
        iter0 = self.dataset.newQueryLoader(
            partition="dev",
            batchSize=self.queryBatchSize,
            shuffle=False,
            numWorkers=4,
        )
        iter1 = self.dataset.newQueryEmbeddingLoader(
            self.embedding,
            partition="dev",
            batchSize=self.queryBatchSize,
            shuffle=False,
            numWorkers=4,
        )

        retrieved = {}
        for (qids, _), queries in tqdm(zip(iter0, iter1), total=len(iter0)):
            queries = queries.to(self.modelDevice)
            _, xhat = self.model.forward(queries)

            # Map from query ID to top-k retrieved passage IDs.
            indices, _ = self.retriever.batch_query(
                [
                    xhat[i].cpu().numpy()
                    for i in range(len(qids))
                ],
                top_k=100,
            )
            for qid, idx in zip(qids, indices):
                retrieved[qid] = idx

        # Run the evaluation
        relevance = self.dataset.getQueryRelevance(partition="dev")
        evaluated = Path(evaluatorWorkspace, f"reconstructed.{self.modelName}.log")
        evaluateRetrieval(relevance, retrieved, evaluated)


if __name__ == "__main__":
    Pipeline()
