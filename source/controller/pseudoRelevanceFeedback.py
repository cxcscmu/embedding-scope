"""
Implement pseduo relevance feedback.
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


class PseudoRelevanceFeedback:
    """
    Implementation of pseudo relevance feedback using sparse representations
    extracted from the dense embedding vectors of the documents.

    The algorithm is as follows:
    1. Find `feedbackTopK` passages using the reconstructed embeddings.
    2. Using `feedbackAlpha` sparse features, extend the query by `feedbackDelta`.
    3. Retrieve `retrieveTopK` passages using the modified query.
    """

    def __init__(self):
        logger.info("Parsing the arguments.")
        parser = argparse.ArgumentParser()
        parser.add_argument("--embedding", type=str, required=True)
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument("--indexGpuDevice", type=int, nargs="+", required=True)
        parser.add_argument("--latentSize", type=parseInt, required=True)
        parser.add_argument("--latentTopK", type=int, required=True)
        parser.add_argument("--modelGpuDevice", type=int, required=True)
        parser.add_argument("--modelName", type=str, required=True)
        parser.add_argument("--feedbackTopK", type=int, required=True)
        parser.add_argument("--retrieveTopK", type=int, required=True)
        parser.add_argument("--feedbackAlpha", type=float, required=True)
        parser.add_argument("--feedbackDelta", type=float, required=True)
        parsed = parser.parse_args()

        # Match the embedding.
        match parsed.embedding:
            case "miniCPM":
                self.embedding = MiniCPM
            case "bgeBase":
                self.embedding = BgeBase
            case _:
                raise NotImplementedError()

        # Match the dataset.
        match parsed.dataset:
            case "msMarco":
                self.dataset = MsMarcoDataset
            case _:
                raise NotImplementedError()

        logger.info("Creating the embedding loader.")
        self.passageLoader = self.dataset.newPassageEmbeddingLoader(
            self.embedding,
            batchSize=4096,
            shuffle=False,
            numWorkers=4,
        )
        self.queryLoader = self.dataset.newQueryEmbeddingLoader(
            self.embedding,
            partition="dev",
            batchSize=512,
            shuffle=False,
            numWorkers=4,
        )

        logger.info("Building the dense retriever.")
        self.retriever = DotProductRetriever(
            self.embedding.size,
            parsed.indexGpuDevice,
        )
        for batch in tqdm(self.passageLoader):
            self.retriever.add(batch)

        logger.info("Loading the autoencoder.")
        self.model = KSparseAutoencoder(
            self.embedding.size,
            parsed.latentSize,
            parsed.latentTopK,
        )
        snapFile = Path(trainerWorkspace, parsed.modelName, "snapshot-best.pth")
        self.model.load_state_dict(torch.load(snapFile)["model"])
        self.model = self.model.to(parsed.modelGpuDevice)

        # Set the attributes.
        self.feedbackTopK = parsed.feedbackTopK
        self.retrieveTopK = parsed.retrieveTopK

    def dispatch(self):
        """
        Dispatch the control experiment.
        """
        for queries in tqdm(self.queryLoader):
            # Retrieve feedbackTopK passages.
            # Extract the latent features.
            # Expand the queries using the passage features.
            # Retrieve retrievedTopK passages.
            pass
        # Evaluate the performance.


if __name__ == "__main__":
    P = PseudoRelevanceFeedback()
    P.dispatch()
