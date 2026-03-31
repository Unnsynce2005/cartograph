"""
GAT-based change impact prediction.

Given a code graph and a "modified node" signal, predict which other
nodes will need to be co-changed (i.e., will also be modified).

Architecture:
  2-layer GAT with 8 attention heads → node pair scoring
  Input: code graph + one-hot modification signal per node
  Output: per-node probability of being affected

Training:
  Positive pairs: nodes co-changed in the same git commit
  Negative pairs: random node pairs NOT co-changed
  Loss: binary cross-entropy on pair predictions

The attention mechanism learns which types of dependencies
(calls, imports, data_flow, temporal co-change) are most
predictive of co-change propagation.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
)

from cartograph.config import settings

logger = logging.getLogger(__name__)


class GATImpactPredictor(nn.Module):
    """
    Graph Attention Network for change impact prediction.

    The model takes:
    - Node features x (from GraphSAGE encoder or raw features)
    - Edge index and edge types
    - A modification signal: which node was modified (one-hot or multi-hot)

    And outputs a per-node "affected probability".
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 32,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.4,
        num_edge_types: int = 9,
    ):
        super().__init__()
        self.num_edge_types = num_edge_types

        # modification signal is concatenated to node features: +1 dim
        actual_input = input_dim + 1

        # edge type embedding
        self.edge_type_embedding = nn.Embedding(num_edge_types, num_heads)

        # GAT layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # first layer: multi-head attention
        self.convs.append(GATConv(
            actual_input, hidden_dim, heads=num_heads, dropout=dropout, concat=True
        ))
        self.bns.append(nn.BatchNorm1d(hidden_dim * num_heads))

        # middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(
                hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout, concat=True
            ))
            self.bns.append(nn.BatchNorm1d(hidden_dim * num_heads))

        # final layer: single head for output
        self.convs.append(GATConv(
            hidden_dim * num_heads, output_dim, heads=1, dropout=dropout, concat=False
        ))
        self.bns.append(nn.BatchNorm1d(output_dim))

        # scoring head: from node embedding to impact probability
        self.scorer = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, 1),
        )

        self.dropout = dropout

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        mod_signal: torch.Tensor,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode all nodes given modification signal."""
        # concatenate modification signal
        h = torch.cat([x, mod_signal.unsqueeze(-1).float()], dim=-1)

        # compute edge attention bias from edge types
        if edge_type is not None:
            edge_attn = self.edge_type_embedding(edge_type)  # [E, num_heads]
        else:
            edge_attn = None

        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = conv(h, edge_index)
            h = bn(h)
            if i < len(self.convs) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        return h  # [N, output_dim]

    def predict_impact(
        self,
        node_embeddings: torch.Tensor,
        modified_idx: int,
    ) -> torch.Tensor:
        """
        Predict impact probability for all nodes given the modified node.
        Returns: [N] tensor of probabilities.
        """
        num_nodes = node_embeddings.shape[0]
        src_emb = node_embeddings[modified_idx].unsqueeze(0).expand(num_nodes, -1)
        pair_features = torch.cat([src_emb, node_embeddings], dim=-1)
        scores = self.scorer(pair_features).squeeze(-1)
        return torch.sigmoid(scores)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        mod_signal: torch.Tensor,
        modified_idx: int,
        edge_type: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full forward: encode + predict."""
        embeddings = self.encode(x, edge_index, mod_signal, edge_type)
        return self.predict_impact(embeddings, modified_idx)


class ImpactPredictor:
    """
    End-to-end impact prediction pipeline.

    Train on co-change data from GitHub repos.
    Infer on user's project to predict modification impact.
    """

    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = GATImpactPredictor(
            input_dim=input_dim,
            hidden_dim=settings.gat_hidden_dim,
            output_dim=settings.gat_output_dim,
            num_heads=settings.gat_num_heads,
            num_layers=settings.gat_num_layers,
            dropout=settings.gat_dropout,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(
        self,
        data: Data,
        positive_pairs: list[tuple[int, int]],
        epochs: int = None,
        lr: float = None,
        neg_ratio: int = 3,
    ) -> list[float]:
        """
        Train GAT on co-change supervision.

        positive_pairs: list of (node_a, node_b) that were co-changed.
        neg_ratio: number of negative samples per positive sample.
        """
        epochs = epochs or settings.gat_epochs
        lr = lr or settings.gat_lr
        num_nodes = data.num_nodes

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        data = data.to(self.device)

        # build positive set for quick lookup
        pos_set = set()
        for a, b in positive_pairs:
            pos_set.add((min(a, b), max(a, b)))

        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            # sample a batch of positive pairs
            np.random.shuffle(positive_pairs)
            batch_size = min(len(positive_pairs), 256)

            for batch_start in range(0, len(positive_pairs), batch_size):
                batch_pairs = positive_pairs[batch_start:batch_start + batch_size]
                if not batch_pairs:
                    continue

                optimizer.zero_grad()
                batch_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                for src, tgt in batch_pairs:
                    # create modification signal
                    mod_signal = torch.zeros(num_nodes, device=self.device)
                    mod_signal[src] = 1.0

                    # forward pass
                    probs = self.model(
                        data.x, data.edge_index, mod_signal, src,
                        edge_type=data.edge_type if hasattr(data, "edge_type") else None,
                    )

                    # positive: target node should have high probability
                    pos_loss = F.binary_cross_entropy(
                        probs[tgt].unsqueeze(0),
                        torch.ones(1, device=self.device),
                    )

                    # negative: random non-co-changed nodes should have low probability
                    neg_indices = []
                    attempts = 0
                    while len(neg_indices) < neg_ratio and attempts < neg_ratio * 10:
                        neg = np.random.randint(0, num_nodes)
                        if neg != src and (min(src, neg), max(src, neg)) not in pos_set:
                            neg_indices.append(neg)
                        attempts += 1

                    if neg_indices:
                        neg_probs = probs[neg_indices]
                        neg_loss = F.binary_cross_entropy(
                            neg_probs,
                            torch.zeros(len(neg_indices), device=self.device),
                        )
                    else:
                        neg_loss = torch.tensor(0.0, device=self.device)

                    batch_loss = batch_loss + pos_loss + neg_loss

                batch_loss = batch_loss / max(len(batch_pairs), 1)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += batch_loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)

            if (epoch + 1) % 20 == 0:
                logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, epochs, avg_loss)

        return losses

    @torch.no_grad()
    def predict(
        self,
        data: Data,
        modified_node_idx: int,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Predict which nodes are most likely affected by modifying the given node.
        Returns: list of (node_idx, probability) sorted by probability descending.
        """
        self.model.eval()
        data = data.to(self.device)

        mod_signal = torch.zeros(data.num_nodes, device=self.device)
        mod_signal[modified_node_idx] = 1.0

        probs = self.model(
            data.x, data.edge_index, mod_signal, modified_node_idx,
            edge_type=data.edge_type if hasattr(data, "edge_type") else None,
        )

        probs = probs.cpu().numpy()

        # exclude the modified node itself
        probs[modified_node_idx] = -1.0

        # top-k
        top_indices = np.argsort(probs)[::-1][:top_k]
        results = [(int(idx), float(probs[idx])) for idx in top_indices if probs[idx] > 0]

        return results

    def evaluate(
        self,
        data: Data,
        test_pairs: list[tuple[int, int]],
        all_nodes: int,
    ) -> dict[str, float]:
        """
        Evaluate impact prediction on held-out co-change pairs.
        Metrics: precision@5, precision@10, recall@10, MRR.
        """
        self.model.eval()

        # group by source node
        src_to_targets: dict[int, set[int]] = {}
        for a, b in test_pairs:
            src_to_targets.setdefault(a, set()).add(b)
            src_to_targets.setdefault(b, set()).add(a)

        p5_scores, p10_scores, r10_scores, mrr_scores = [], [], [], []

        for src, true_targets in src_to_targets.items():
            predictions = self.predict(data, src, top_k=10)
            pred_nodes = [idx for idx, _ in predictions]

            # precision@5
            top5 = set(pred_nodes[:5])
            p5 = len(top5 & true_targets) / 5
            p5_scores.append(p5)

            # precision@10
            top10 = set(pred_nodes[:10])
            p10 = len(top10 & true_targets) / 10
            p10_scores.append(p10)

            # recall@10
            r10 = len(top10 & true_targets) / max(len(true_targets), 1)
            r10_scores.append(r10)

            # MRR
            rr = 0.0
            for rank, node in enumerate(pred_nodes):
                if node in true_targets:
                    rr = 1.0 / (rank + 1)
                    break
            mrr_scores.append(rr)

        return {
            "precision_at_5": round(np.mean(p5_scores), 4) if p5_scores else 0.0,
            "precision_at_10": round(np.mean(p10_scores), 4) if p10_scores else 0.0,
            "recall_at_10": round(np.mean(r10_scores), 4) if r10_scores else 0.0,
            "mrr": round(np.mean(mrr_scores), 4) if mrr_scores else 0.0,
        }

    def save(self, path: str) -> None:
        torch.save({
            "model_state": self.model.state_dict(),
            "input_dim": self.input_dim,
        }, path)
        logger.info("Saved GAT model to %s", path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        logger.info("Loaded GAT model from %s", path)
