"""
Dispatch the training.
"""

import argparse
from pathlib import Path
from typing import DefaultDict, Dict
from collections import defaultdict
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import functional as F
from torch import amp
from torch import Tensor
import wandb
from source import logger
from source.utilities import tqdm, parseInt
from source.trainer import workspace
from source.autoencoder import KSparseAutoencoder
from source.dataset.textRetrieval import MsMarcoDataset
from source.embedding import MiniCPM, BgeBase


class Trainer:
    """
    Dispatch the training.
    """

    def __init__(self) -> None:
        # Parse the arguments.
        parser = argparse.ArgumentParser()
        parser.add_argument("--name", type=str, required=True)
        parser.add_argument("--embedding", type=str, required=True)
        parser.add_argument("--latentSize", type=parseInt, required=True)
        parser.add_argument("--latentTopK", type=int, required=True)
        parser.add_argument("--nearbyTopK", type=int, required=True)
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument("--optimizer", type=str, required=True)
        parser.add_argument("--learningRate", type=float, required=True)
        parser.add_argument("--scheduler", type=str, required=True)
        parser.add_argument("--numEpochs", type=int, required=True)
        parser.add_argument("--batchSize", type=int, required=True)
        parsed = parser.parse_args()
        wandb.init(project="scope", name=parsed.name)

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

        # Create the embedding loader.
        self.trainLoader = self.dataset.newMixEmbeddingLoader(
            self.embedding,
            "train",
            parsed.nearbyTopK,
            parsed.batchSize,
            shuffle=True,
            numWorkers=4,
        )
        self.devLoader = self.dataset.newMixEmbeddingLoader(
            self.embedding,
            "dev",
            parsed.nearbyTopK,
            parsed.batchSize,
            shuffle=False,
            numWorkers=4,
        )

        # Create the autoencoder.
        self.model = KSparseAutoencoder(
            self.embedding.size,
            parsed.latentSize,
            parsed.latentTopK,
        )
        self.model = self.model.cuda()

        # Match the optimizer.
        match parsed.optimizer:
            case "Adam":
                self.optimizer = Adam(
                    self.model.parameters(), lr=parsed.learningRate
                )
            case _:
                raise NotImplementedError()

        # Match the scheduler.
        match parsed.scheduler:
            case "CosineAnnealing":
                self.scheduler = CosineAnnealingLR(
                    self.optimizer, parsed.numEpochs
                )
            case _:
                raise NotImplementedError()

        # Training parameters.
        self.lastEpoch = 0
        self.vLossBest = float("inf")
        self.numEpochs = parsed.numEpochs
        self.batchSize = parsed.batchSize

        # Create the workspace.
        self.workspace = Path(workspace, parsed.name)
        self.workspace.mkdir(mode=0o770, parents=True, exist_ok=True)

        # Create the scaler.
        self.scaler = amp.GradScaler()

    def save(self, mode: str) -> None:
        """
        Save the last snapshot.
        """
        snapshot = dict()
        snapshot["model"] = self.model.state_dict()
        snapshot["optimizer"] = self.optimizer.state_dict()
        snapshot["scheduler"] = self.scheduler.state_dict()
        snapshot["lastEpoch"] = self.lastEpoch
        snapshot["vLossBest"] = self.vLossBest
        snapfile = Path(self.workspace, f"snapshot-{mode}.pth")
        torch.save(snapshot, snapfile)

    def load(self, mode: str) -> None:
        """
        Load the snapshot.
        """
        snapfile = Path(self.workspace, f"snapshot-{mode}.pth")
        if snapfile.exists():
            snapshot = torch.load(snapfile)
            self.model.load_state_dict(snapshot["model"])
            self.optimizer.load_state_dict(snapshot["optimizer"])
            self.scheduler.load_state_dict(snapshot["scheduler"])
            self.lastEpoch = snapshot["lastEpoch"] + 1
            self.vLossBest = snapshot["vLossBest"]

    def measure(
        self, qrys: Tensor, docs: Tensor, qrys_hat: Tensor, docs_hat: Tensor
    ) -> Dict[str, Tensor]:
        """
        Measure the loss.
        """
        loss = dict()
        loss["MSE"] = F.mse_loss(qrys_hat, qrys) + F.mse_loss(docs_hat, docs)

        lhs = qrys.unsqueeze(1)  # (N, 1, D)
        rhs = docs.transpose(1, 2)  # (N, D, K)
        mat = torch.matmul(lhs, rhs).squeeze(1)  # (N, K)
        mat = mat / qrys.size(1) ** 0.5  # (N, K)
        score = torch.exp(mat)  # (N, K)
        norms = score.sum(dim=1, keepdim=True)  # (N, 1)

        lhs_hat = qrys_hat.unsqueeze(1)  # (N, 1, D)
        rhs_hat = docs_hat.transpose(1, 2)  # (N, D, K)
        mat_hat = torch.matmul(lhs_hat, rhs_hat).squeeze(1)  # (N, K)
        mat_hat = mat_hat / qrys.size(1) ** 0.5  # (N, K)
        score_hat = torch.exp(mat_hat)  # (N, K)
        norms_hat = score_hat.sum(dim=1, keepdim=True)  # (N, 1)

        tar = score / (score + norms)  # (N, K)
        ins = torch.log(score_hat / (score_hat + norms_hat))  # (N, K)
        loss["KLD"] = F.kl_div(ins, tar, reduction="sum")

        return loss

    def train(self) -> DefaultDict[str, float]:
        """
        Train the model.
        """
        self.model.train()
        tLoss: DefaultDict[str, float] = defaultdict(float)
        for qrys, docs in tqdm(self.trainLoader, mininterval=10):
            self.optimizer.zero_grad()
            with amp.autocast("cuda"):
                qrys, docs = qrys.cuda(), docs.cuda()
                _, qrysHat = self.model.forward(qrys)
                _, docsHat = self.model.forward(
                    docs.view(-1, self.embedding.size)
                )
                loss = self.measure(qrys, docs, qrysHat, docsHat.view_as(docs))
            self.scaler.scale(sum(loss.values())).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            for key, val in loss.items():
                tLoss[key] += val.item()
        for key, val in tLoss.items():
            tLoss[key] = val / len(self.trainLoader)
        return tLoss

    def validate(self) -> DefaultDict[str, float]:
        """
        Validate the model.
        """
        self.model.eval()
        vLoss: DefaultDict[str, float] = defaultdict(float)
        with torch.no_grad():
            for qrys, docs in tqdm(self.devLoader, mininterval=10):
                qrys, docs = qrys.cuda(), docs.cuda()
                _, qrysHat = self.model.forward(qrys)
                _, docsHat = self.model.forward(
                    docs.view(-1, self.embedding.size)
                )
                loss = self.measure(qrys, docs, qrysHat, docsHat.view_as(docs))
                for key, val in loss.items():
                    vLoss[key] += val.item()
            for key, val in vLoss.items():
                vLoss[key] = val / len(self.devLoader)
        return vLoss

    def dispatch(self):
        """
        Dispatch the training.
        """
        self.load("last")
        for epoch in range(self.lastEpoch, self.numEpochs):
            self.lastEpoch = epoch
            logger.info("Epoch: %03d/%03d", epoch, self.numEpochs)
            tLoss = self.train()
            tLossStr = ", ".join(
                f"{key}={val:.7f}" for key, val in tLoss.items()
            )
            logger.info("Train    : %s", tLossStr)
            vLoss = self.validate()
            vLossStr = ", ".join(
                f"{key}={val:.7f}" for key, val in vLoss.items()
            )
            logger.info("Validate : %s", vLossStr)
            self.scheduler.step()
            self.save("last")
            if sum(vLoss.values()) < self.vLossBest:
                self.vLossBest = sum(vLoss.values())
                self.save("best")
            health = dict()
            for key, val in tLoss.items():
                health[f"t{key}"] = val
            for key, val in vLoss.items():
                health[f"v{key}"] = val
            health["lr"] = self.optimizer.param_groups[0]["lr"]
            wandb.log(health)


if __name__ == "__main__":
    T = Trainer()
    T.dispatch()
