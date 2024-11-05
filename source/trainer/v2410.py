"""
Implementation of the trainer interface.
"""

import argparse
from typing import Type, Dict, Optional, List, Tuple
import torch
from torch import nn, amp
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import Tensor
import numpy as np
from numpy.typing import NDArray
from source import console
from source.interface import Trainer
from source.interface.dataset import TextRetrievalDataset
from source.interface.embedding import TextEmbedding
from source.autoencoder.kSparse import KSparseAutoencoder
from source.dataset.msMarco import MsMarco
from source.embedding.miniCPM import MiniCPM


class V2410(Trainer):
    """
    Implementation of the trainer interface.

    This trainer uses a combination of MSE and KLD loss.
    """

    def __init__(
        self,
        dataset: TextRetrievalDataset,
        embedding: Type[TextEmbedding],
        autoencoder: KSparseAutoencoder,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        numEpochs: int,
        batchSize: int,
        numNeighbors: int,
        devices: List[int],
    ):
        self.dataset = dataset
        self.embedding = embedding
        autoencoder.to(devices[0])
        self.autoencoder = nn.DataParallel(autoencoder, devices)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.numEpochs = numEpochs
        self.batchSize = batchSize
        self.numNeighbors = numNeighbors
        self.devices = devices
        self.scaler = amp.GradScaler()

        console.log("Loading the passage embeddings...")
        self.passages: Dict[str, NDArray[np.float32]] = {}
        iterator = self.dataset.getPassageEmbeddings(self.embedding)
        for passageID, passageEmbedding in iterator:
            self.passages[passageID] = passageEmbedding
        console.log("Loading the training queries...")
        self.trainQueries: Dict[str, NDArray[np.float32]] = {}
        iterator = self.dataset.getQueryEmbeddings("train", self.embedding)
        for queryID, queryEmbedding in iterator:
            self.trainQueries[queryID] = queryEmbedding
        console.log("Loading the validation queries...")
        self.validQueries: Dict[str, NDArray[np.float32]] = {}
        iterator = self.dataset.getQueryEmbeddings("dev", self.embedding)
        for queryID, queryEmbedding in iterator:
            self.validQueries[queryID] = queryEmbedding
        console.log("Loading the training neighbors...")
        self.trainNeighbors: Dict[str, Dict[str, float]] = {}
        iterator = self.dataset.getNeighborPassages("train", self.embedding)
        for queryID, neighbors in iterator.items():
            self.trainNeighbors[queryID] = neighbors
        console.log("Loading the validation neighbors...")
        self.validNeighbors: Dict[str, Dict[str, float]] = {}
        iterator = self.dataset.getNeighborPassages("dev", self.embedding)
        for queryID, neighbors in iterator.items():
            self.validNeighbors[queryID] = neighbors
        console.log("Building the training dataset...")
        self.trainDataset: List[Tuple[NDArray[np.float32], NDArray[np.float32]]] = []
        for queryID, neighbors in self.trainNeighbors.items():
            neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
            neighbors = neighbors[: self.numNeighbors]
            buffer = np.empty((len(neighbors), self.embedding.size), dtype=np.float32)
            for i, (passageID, _) in enumerate(neighbors):
                buffer[i] = self.passages[passageID]
            queryEmbedding = self.trainQueries[queryID]
            self.trainDataset.append((queryEmbedding, buffer))
        console.log("Building the validation dataset...")
        self.validDataset: List[Tuple[NDArray[np.float32], NDArray[np.float32]]] = []
        for queryID, neighbors in self.validNeighbors.items():
            neighbors = sorted(neighbors.items(), key=lambda x: x[1], reverse=True)
            neighbors = neighbors[: self.numNeighbors]
            buffer = np.empty((len(neighbors), self.embedding.size), dtype=np.float32)
            for i, (passageID, _) in enumerate(neighbors):
                buffer[i] = self.passages[passageID]
            queryEmbedding = self.validQueries[queryID]
            self.validDataset.append((queryEmbedding, buffer))

    def computeLoss(
        self, qrys: Tensor, docs: Tensor, qryHat: Tensor, docsHat: Tensor
    ) -> Dict[str, Tensor]:
        """
        Compute the loss for a batch, MSE + KLD.

        :param qrys: the query embeddings
        :param docs: the document embeddings
        :param qryHat: the reconstructed query embeddings
        :param docsHat: the reconstructed document embeddings
        """
        loss = dict()
        prefix = "Train" if self.autoencoder.training else "Valid"
        baseMSE = torch.tensor(0.0, requires_grad=self.autoencoder.training)
        loss[f"{prefix}.MSE"] = baseMSE
        loss[f"{prefix}.MSE"] = loss[f"{prefix}.MSE"] + F.mse_loss(qryHat, qrys)
        loss[f"{prefix}.MSE"] = loss[f"{prefix}.MSE"] + F.mse_loss(docsHat, docs)
        baseKLD = torch.tensor(0.0, requires_grad=self.autoencoder.training)
        loss[f"{prefix}.KLD"] = baseKLD
        buf = torch.exp(
            torch.matmul(
                qrys.unsqueeze(1),
                docs.transpose(1, 2),
            ).squeeze(1)
        )
        bufSum = buf.sum(dim=1).view(-1, 1)
        bufHat = torch.exp(
            torch.matmul(
                qryHat.unsqueeze(1),
                docsHat.transpose(1, 2),
            ).squeeze(1)
        )
        bufHatSum = bufHat.sum(dim=1).view(-1, 1)
        tar = buf / (buf + bufSum)
        ins = torch.log(bufHat / (bufHat + bufHatSum))
        loss[f"{prefix}.KLD"] = F.kl_div(ins, tar, reduction="batchmean")
        return loss

    def train(self):
        self.autoencoder.train()
        # shuffle the indices then group into batches
        indices = np.arange(len(self.trainDataset))
        np.random.shuffle(indices)
        batches = np.array_split(indices, len(indices) // self.batchSize)
        for i, batch in enumerate(batches):
            console.log(f"Train    : {i:08d}/{len(batches):08d}")
            # combine queries and neighbors into buffers
            N, D, K = len(batch), self.numNeighbors, self.embedding.size
            qrys = np.empty((N, K), dtype=np.float32)
            docs = np.empty((N, D, K), dtype=np.float32)
            for j, index in enumerate(batch):
                qrys[j], docs[j] = self.trainDataset[index]
            # convert to tensors on the device
            qrys = torch.from_numpy(qrys).to(self.devices[0])
            docs = torch.from_numpy(docs).to(self.devices[0])
            # clear the gradients
            self.optimizer.zero_grad()
            # forward pass under autocast
            with amp.autocast("cuda"):
                _, qryHat = self.autoencoder.forward(qrys)
                _, docsHat = self.autoencoder.forward(docs.view(-1, docs.size(-1)))
                loss = self.computeLoss(qrys, docs, qryHat, docsHat)
            # backward pass
            self.scaler.scale(sum(loss.values())).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    @torch.no_grad()
    def validate(self):
        self.autoencoder.eval()
        # shuffle the indices then group into batches
        indices = np.arange(len(self.validDataset))
        np.random.shuffle(indices)
        batches = np.array_split(indices, len(indices) // self.batchSize)
        for i, batch in enumerate(batches):
            console.log(f"Validate : {i:08d}/{len(batches):08d}")
            # combine queries and neighbors into buffers
            N, D, K = len(batch), self.numNeighbors, self.embedding.size
            qrys = np.empty((N, K), dtype=np.float32)
            docs = np.empty((N, D, K), dtype=np.float32)
            for j, index in enumerate(batch):
                qrys[j], docs[j] = self.validDataset[index]
            # convert to tensors on the device
            qrys = torch.from_numpy(qrys).to(self.devices[0])
            docs = torch.from_numpy(docs).to(self.devices[0])
            # forward pass
            _, qryHat = self.autoencoder.forward(qrys)
            _, docsHat = self.autoencoder.forward(docs.view(-1, docs.size(-1)))
            loss = self.computeLoss(qrys, docs, qryHat, docsHat)


def main():
    """
    The entrypoint.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--embedding", type=str, required=True)
    parser.add_argument("--latentSize", type=int, required=True)
    parser.add_argument("--latentTopK", type=int, required=True)
    parser.add_argument("--learnRate", type=float, required=True)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--numEpochs", type=int, required=True)
    parser.add_argument("--scheduler", type=str, required=True)
    parser.add_argument("--batchSize", type=int, required=True)
    parser.add_argument("--numNeighbors", type=int, required=True)
    parser.add_argument("--devices", type=int, nargs="+", required=True)
    parsed = parser.parse_args()

    dataset: Optional[TextRetrievalDataset] = None
    match parsed.dataset:
        case "MsMarco":
            dataset = MsMarco()
    assert dataset is not None

    embedding: Optional[Type[TextEmbedding]] = None
    match parsed.embedding:
        case "MiniCPM":
            embedding = MiniCPM
    assert embedding is not None

    autoencoder = KSparseAutoencoder(
        embedding.size,
        parsed.latentSize,
        parsed.latentTopK,
    )

    optimizer: Optional[Optimizer] = None
    match parsed.optimizer:
        case "Adam":
            optimizer = Adam(autoencoder.parameters(), lr=parsed.learnRate)
    assert optimizer is not None

    scheduler: Optional[LRScheduler] = None
    match parsed.scheduler:
        case "CosineAnnealing":
            scheduler = CosineAnnealingLR(optimizer, T_max=parsed.numEpochs)
    assert scheduler is not None

    trainer = V2410(
        dataset,
        embedding,
        autoencoder,
        optimizer,
        scheduler,
        parsed.numEpochs,
        parsed.batchSize,
        parsed.numNeighbors,
        parsed.devices,
    )

    trainer.train()
    trainer.validate()


if __name__ == "__main__":
    main()
