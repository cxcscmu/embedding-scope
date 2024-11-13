"""
Implement baseline.
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


class Reconstructed:
    """
    The reconstructed baseline.
    """

    def __init__(self):
        # Parse the arguments.
        parser = argparse.ArgumentParser()
        parser.add_argument("--embedding", type=str, required=True)
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument("--indexGpuDevice", type=int, nargs="+", required=True)
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
        logger.info(f"{parsed.modelName} Iteration: {snapShot['lastEpoch']}")
        self.model.load_state_dict(snapShot["model"])
        self.model = self.model.to(parsed.modelGpuDevice)

        # Build the dense retriever.
        logger.info("Build the dense retriever.")
        self.passageLoader = self.dataset.newPassageEmbeddingLoader(
            self.embedding,
            batchSize=4096,
            shuffle=False,
            numWorkers=4,
        )
        self.retriever = DotProductRetriever(
            self.embedding.size,
            parsed.indexGpuDevice,
        )
        with torch.no_grad():
            for passages in tqdm(self.passageLoader):
                passages = passages.to(parsed.modelGpuDevice)
                _, reconstructed = self.model.forward(passages)
                reconstructed = reconstructed.detach().cpu()
                self.retriever.add(reconstructed)

        # Create the query loader.
        self.queryLoader = self.dataset.newQueryEmbeddingLoader(
            self.embedding,
            partition="dev",
            batchSize=256,
            shuffle=False,
            numWorkers=4,
        )

        # Map from passage index to passage id.
        logger.info("Build the passage lookup.")
        self.passageLookup, i = dict(), 0
        for pids, _ in tqdm(MsMarcoDataset.newPassageLoader(4096, False, 4)):
            for x in pids:
                self.passageLookup[i] = x
                i += 1

        # Map from query index to query id.
        logger.info("Build the query lookup.")
        self.queryLookup, i = dict(), 0
        for qids, _ in tqdm(MsMarcoDataset.newQueryLoader("dev", 4096, False, 4)):
            for x in qids:
                self.queryLookup[i] = x
                i += 1

        # Set the attributes.
        self.retrieveTopK = parsed.retrieveTopK
        self.modelGpuDevice = parsed.modelGpuDevice

    def dispatch(self):
        """
        Dispatch the experiment.
        """
        # fmt: off
        relevance = self.dataset.getQueryRelevance("dev")
        retrieved, i = dict(), 0
        logger.info("Retrieve with reconstructed.")
        with torch.no_grad():
            for queries in tqdm(self.queryLoader):
                # Retrieve initial passages for query expansion.
                queries = queries.to(self.modelGpuDevice)
                _, queries = self.model.forward(queries)
                queries = queries.detach().cpu()
                indices, _ = self.retriever.search(queries, self.retrieveTopK)
                for xs in indices:
                    retrieved[self.queryLookup[i]] = [self.passageLookup[x] for x in xs]
                    i += 1
        evaluated = Path("evaluated_baseline.log")
        evaluateRetrieval(relevance, retrieved, evaluated)


if __name__ == "__main__":
    R = Reconstructed()
    R.dispatch()
