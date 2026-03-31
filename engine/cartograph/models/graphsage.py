"""
GraphSAGE-based module discovery.

Given a code graph, learn node embeddings that capture both structural
topology and semantic similarity. Nodes that belong to the same logical
module should cluster together in the embedding space.

Architecture:
  3-layer GraphSAGE with neighborhood sampling → 64-dim node embeddings
  → hierarchical agglomerative clustering → module assignments

Training objective: supervised contrastive loss on directory-based labels.
Nodes in the same directory are positive pairs; cross-directory are negatives.
At inference, the model generalizes to projects without directory structure
(e.g., vibe-coded flat-file projects) using learned graph patterns.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)

from cartograph.config import settings

logger = logging.getLogger(__name__)


class GraphSAGEEncoder(nn.Module):
    """
    Multi-layer GraphSAGE encoder.
    Produces node embeddings by aggregating neighborhood information.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout

        # first layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # middle layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # output layer
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        self.bns.append(nn.BatchNorm1d(output_dim))

        # projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return F.normalize(x, p=2, dim=1)

    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project embeddings through the contrastive head."""
        return F.normalize(self.projector(embeddings), p=2, dim=1)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.
    Nodes in the same module (directory) are pulled together;
    nodes in different modules are pushed apart.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]

        # similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature

        # mask: 1 where labels match (positive pairs)
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # exclude self-similarity
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # log softmax
        exp_logits = torch.exp(sim) * logits_mask
        log_prob = sim - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # mean of log-prob over positive pairs
        pos_count = mask.sum(dim=1)
        pos_count = torch.clamp(pos_count, min=1.0)
        mean_log_prob = (mask * log_prob).sum(dim=1) / pos_count

        loss = -mean_log_prob.mean()
        return loss


class ModuleDiscovery:
    """
    End-to-end module discovery pipeline.

    Train: GraphSAGE with supervised contrastive loss on labeled graphs.
    Infer: Encode nodes → hierarchical clustering → module assignments.
    """

    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = GraphSAGEEncoder(
            input_dim=input_dim,
            hidden_dim=settings.sage_hidden_dim,
            output_dim=settings.sage_output_dim,
            num_layers=settings.sage_num_layers,
            dropout=settings.sage_dropout,
        )
        self.loss_fn = SupConLoss(temperature=0.07)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(
        self,
        data: Data,
        labels: torch.Tensor,
        epochs: int = None,
        lr: float = None,
    ) -> list[float]:
        """
        Train GraphSAGE on a labeled graph.
        labels: int tensor of module labels per node (from directory structure).
        Returns: list of losses per epoch.
        """
        epochs = epochs or settings.sage_epochs
        lr = lr or settings.sage_lr

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        data = data.to(self.device)
        labels = labels.to(self.device)

        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()

            embeddings = self.model(data.x, data.edge_index)
            projections = self.model.project(embeddings)
            loss = self.loss_fn(projections, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

            if (epoch + 1) % 20 == 0:
                logger.info(
                    "Epoch %d/%d — loss: %.4f — lr: %.6f",
                    epoch + 1, epochs, loss.item(), scheduler.get_last_lr()[0],
                )

        return losses

    @torch.no_grad()
    def encode(self, data: Data) -> np.ndarray:
        """Get node embeddings for a graph."""
        self.model.eval()
        data = data.to(self.device)
        embeddings = self.model(data.x, data.edge_index)
        return embeddings.cpu().numpy()

    def discover_modules(
        self,
        data: Data,
        min_modules: int = None,
        max_modules: int = None,
    ) -> tuple[np.ndarray, int, float]:
        """
        Run module discovery on a graph.
        Returns: (labels, num_modules, silhouette_score)
        """
        min_k = min_modules or settings.min_modules
        max_k = max_modules or settings.max_modules

        embeddings = self.encode(data)

        # hierarchical agglomerative clustering
        Z = linkage(embeddings, method="ward")

        # search for optimal k using silhouette score
        best_k = min_k
        best_score = -1.0
        best_labels = None

        for k in range(min_k, max_k + 1):
            labels = fcluster(Z, t=k, criterion="maxclust")
            if len(set(labels)) < 2:
                continue

            score = silhouette_score(embeddings, labels, metric="cosine")
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        if best_labels is None:
            best_labels = fcluster(Z, t=min_k, criterion="maxclust")
            best_score = 0.0

        logger.info(
            "Module discovery: k=%d, silhouette=%.3f",
            best_k, best_score,
        )

        # shift labels to 0-indexed
        best_labels = best_labels - 1

        return best_labels, best_k, best_score

    def evaluate(
        self,
        predicted_labels: np.ndarray,
        true_labels: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate module discovery against ground truth."""
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        ari = adjusted_rand_score(true_labels, predicted_labels)
        return {"nmi": round(nmi, 4), "ari": round(ari, 4)}

    def save(self, path: str) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "input_dim": self.input_dim,
            "config": {
                "hidden_dim": settings.sage_hidden_dim,
                "output_dim": settings.sage_output_dim,
                "num_layers": settings.sage_num_layers,
                "dropout": settings.sage_dropout,
            },
        }, path)
        logger.info("Saved GraphSAGE model to %s", path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        logger.info("Loaded GraphSAGE model from %s", path)
