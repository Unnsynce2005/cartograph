"""
Training pipeline for Cartograph models.

Usage:
  python -m cartograph.train.run --repos repos.txt --output models/

Steps:
  1. Clone repos listed in repos.txt
  2. Run parser on each → code graphs
  3. Extract directory labels → GraphSAGE training data
  4. Extract git co-change pairs → GAT training data
  5. Compute SBERT embeddings for all nodes
  6. Train GraphSAGE with supervised contrastive loss
  7. Train GAT with co-change supervision
  8. Save model checkpoints
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from cartograph.config import settings
from cartograph.data.graph import build_nx_graph, build_node_features, to_pyg_data, load_graph_json
from cartograph.data.features import NodeEmbedder
from cartograph.data.dataset import (
    extract_directory_labels,
    extract_cochange_pairs,
    build_temporal_edges,
)
from cartograph.models.graphsage import ModuleDiscovery
from cartograph.models.gat import ImpactPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)


def clone_repo(url: str, dest: str) -> bool:
    """Shallow clone a repo."""
    try:
        subprocess.run(
            ["git", "clone", "--depth", "200", url, dest],
            capture_output=True, timeout=120, check=True,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning("Failed to clone %s: %s", url, e)
        return False


def parse_repo(repo_dir: str, output_json: str) -> bool:
    """Run the TypeScript parser on a repo."""
    cmd = f"{settings.parser_bin} {repo_dir} -o {output_json} --snippets"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def train_models(
    repo_urls: list[str],
    output_dir: str,
    max_repos: int = 100,
):
    """Full training pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    embedder = NodeEmbedder(settings.sbert_model)

    all_graphs = []
    all_labels = []
    all_cochange_pairs = []
    all_features = []
    cumulative_offset = 0

    tmpdir = tempfile.mkdtemp(prefix="cartograph_train_")

    for i, url in enumerate(repo_urls[:max_repos]):
        repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
        repo_dir = os.path.join(tmpdir, repo_name)
        graph_json_path = os.path.join(tmpdir, f"{repo_name}.json")

        logger.info("[%d/%d] Processing %s", i + 1, len(repo_urls), repo_name)

        # clone
        if not clone_repo(url, repo_dir):
            continue

        # parse
        if not parse_repo(repo_dir, graph_json_path):
            logger.warning("Parse failed for %s", repo_name)
            continue

        # load graph
        try:
            graph_json = load_graph_json(graph_json_path)
        except Exception as e:
            logger.warning("Failed to load graph for %s: %s", repo_name, e)
            continue

        if len(graph_json.get("nodes", [])) < 10:
            logger.info("Skipping %s (too few nodes: %d)", repo_name, len(graph_json.get("nodes", [])))
            continue

        G, node_id_to_idx, node_metadata = build_nx_graph(graph_json)
        idx_to_id = {v: k for k, v in node_id_to_idx.items()}

        # SBERT embeddings
        ordered_nodes = [node_metadata[idx_to_id[i]] for i in range(len(node_id_to_idx))]
        sbert_embs = embedder.embed_nodes(ordered_nodes)

        # directory labels
        labels = extract_directory_labels(ordered_nodes)

        # co-change pairs
        file_map: dict[str, list[int]] = {}
        for idx, nid in idx_to_id.items():
            fp = node_metadata[nid].get("filePath", "")
            file_map.setdefault(fp, []).append(idx + cumulative_offset)

        pairs = extract_cochange_pairs(repo_dir, file_map)

        # accumulate
        node_features = build_node_features(G, sbert_embs)
        all_features.append(node_features)
        all_labels.append(labels)
        all_cochange_pairs.extend(
            [(p.node_a_idx, p.node_b_idx) for p in pairs]
        )

        cumulative_offset += len(node_id_to_idx)
        logger.info("  → %d nodes, %d labels, %d co-change pairs",
                     len(node_id_to_idx), len(set(labels)), len(pairs))

    if not all_features:
        logger.error("No valid repos processed. Exiting.")
        return

    # merge all graphs into one big training graph
    logger.info("Merging %d repos into training set...", len(all_features))
    merged_features = torch.cat(all_features, dim=0)
    merged_labels = torch.from_numpy(np.concatenate(all_labels))

    # create a simple data object (no cross-repo edges)
    # Each repo's internal edges would need to be merged with offset — simplified here
    data = Data(
        x=merged_features,
        edge_index=torch.zeros((2, 0), dtype=torch.long),  # placeholder
        num_nodes=merged_features.shape[0],
    )

    input_dim = merged_features.shape[1]

    # train GraphSAGE
    logger.info("Training GraphSAGE (input_dim=%d, nodes=%d)...", input_dim, data.num_nodes)
    sage = ModuleDiscovery(input_dim)
    losses = sage.train(data, merged_labels, epochs=settings.sage_epochs)
    sage.save(os.path.join(output_dir, "graphsage.pt"))
    logger.info("GraphSAGE training complete. Final loss: %.4f", losses[-1] if losses else 0)

    # train GAT
    if all_cochange_pairs:
        logger.info("Training GAT (%d co-change pairs)...", len(all_cochange_pairs))
        gat = ImpactPredictor(input_dim)
        gat_losses = gat.train(data, all_cochange_pairs, epochs=settings.gat_epochs)
        gat.save(os.path.join(output_dir, "gat.pt"))
        logger.info("GAT training complete. Final loss: %.4f", gat_losses[-1] if gat_losses else 0)
    else:
        logger.warning("No co-change pairs found. Skipping GAT training.")

    logger.info("All models saved to %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Train Cartograph models")
    parser.add_argument("--repos", required=True, help="File with one GitHub URL per line")
    parser.add_argument("--output", default="models/", help="Output directory for model checkpoints")
    parser.add_argument("--max-repos", type=int, default=100)
    args = parser.parse_args()

    with open(args.repos, "r") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    logger.info("Loaded %d repo URLs", len(urls))
    train_models(urls, args.output, args.max_repos)


if __name__ == "__main__":
    main()
