"""
Training dataset construction from open-source GitHub repositories.

Two training signals:
1. Module discovery (GraphSAGE): directory structure → module labels
   If two functions are in /components/auth/, they belong to the same module.

2. Impact prediction (GAT): git commit co-change → temporal edges
   If function A and function B are modified in the same commit,
   they form a positive (co-changed) pair.

This module handles:
- Cloning repos in batch
- Running the parser on each
- Extracting labels and co-change pairs
- Assembling into PyG datasets
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

from cartograph.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CoChangePair:
    """A pair of node indices that were modified in the same commit."""
    node_a_idx: int
    node_b_idx: int
    commit_hash: str
    timestamp: int  # unix epoch


def extract_directory_labels(nodes: list[dict]) -> np.ndarray:
    """
    Assign module labels based on directory structure.
    Nodes in the same top-level directory (under src/) get the same label.

    This is the ground truth for GraphSAGE module discovery training.
    """
    dir_to_label: dict[str, int] = {}
    labels = []

    for node in nodes:
        file_path = node.get("filePath", "")
        parts = file_path.replace("\\", "/").split("/")

        # find the meaningful grouping directory
        # skip common prefixes: src, app, lib
        skip = {"src", "app", "lib", "pages", ""}
        meaningful = [p for p in parts[:-1] if p not in skip]

        if meaningful:
            module_dir = meaningful[0]  # first meaningful directory = module
        else:
            module_dir = "__root__"

        if module_dir not in dir_to_label:
            dir_to_label[module_dir] = len(dir_to_label)

        labels.append(dir_to_label[module_dir])

    return np.array(labels, dtype=np.int64)


def extract_cochange_pairs(
    repo_path: str,
    node_file_map: dict[str, list[int]],  # filePath -> list of node indices
    max_commits: int = 500,
) -> list[CoChangePair]:
    """
    Extract co-change pairs from git history.

    For each commit, find which files were modified. For each pair of
    modified files that both have parsed nodes, create co-change pairs
    between all nodes in those files.
    """
    pairs: list[CoChangePair] = []

    try:
        result = subprocess.run(
            [
                "git", "log",
                f"--max-count={max_commits}",
                "--name-only",
                "--pretty=format:%H %ct",
                "--diff-filter=M",
            ],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return pairs

    if result.returncode != 0:
        return pairs

    # parse git log output
    current_hash = ""
    current_ts = 0
    current_files: list[str] = []

    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            # end of a commit block: process accumulated files
            if current_hash and len(current_files) >= 2:
                _add_cochange(pairs, current_files, node_file_map, current_hash, current_ts)
            current_files = []
            continue

        parts = line.split(" ", 1)
        if len(parts) == 2 and len(parts[0]) == 40:
            # new commit header
            if current_hash and len(current_files) >= 2:
                _add_cochange(pairs, current_files, node_file_map, current_hash, current_ts)
            current_hash = parts[0]
            try:
                current_ts = int(parts[1])
            except ValueError:
                current_ts = 0
            current_files = []
        else:
            # file path
            current_files.append(line)

    # last commit
    if current_hash and len(current_files) >= 2:
        _add_cochange(pairs, current_files, node_file_map, current_hash, current_ts)

    return pairs


def _add_cochange(
    pairs: list[CoChangePair],
    files: list[str],
    node_file_map: dict[str, list[int]],
    commit_hash: str,
    timestamp: int,
) -> None:
    """Add co-change pairs for all node pairs across modified files."""
    file_nodes: list[list[int]] = []
    for f in files:
        indices = node_file_map.get(f, [])
        if indices:
            file_nodes.append(indices)

    if len(file_nodes) < 2:
        return

    # create pairs between nodes in different files
    for i in range(len(file_nodes)):
        for j in range(i + 1, len(file_nodes)):
            for a in file_nodes[i]:
                for b in file_nodes[j]:
                    pairs.append(CoChangePair(a, b, commit_hash, timestamp))


def build_temporal_edges(
    cochange_pairs: list[CoChangePair],
    num_nodes: int,
    decay: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build temporal co-change edge index and weights.

    More recent co-changes get higher weight. Weight = decay^(age_rank).
    Returns: (edge_index [2, E], edge_weight [E])
    """
    if not cochange_pairs:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros(0)

    # sort by timestamp (most recent first)
    sorted_pairs = sorted(cochange_pairs, key=lambda p: -p.timestamp)

    # count co-change frequency per pair, weighted by recency
    pair_weights: dict[tuple[int, int], float] = {}
    for rank, pair in enumerate(sorted_pairs):
        key = (min(pair.node_a_idx, pair.node_b_idx),
               max(pair.node_a_idx, pair.node_b_idx))
        w = decay ** rank
        pair_weights[key] = pair_weights.get(key, 0.0) + w

    # normalize weights to [0, 1]
    max_w = max(pair_weights.values()) if pair_weights else 1.0
    if max_w == 0:
        max_w = 1.0

    src, tgt, weights = [], [], []
    for (a, b), w in pair_weights.items():
        src.extend([a, b])  # bidirectional
        tgt.extend([b, a])
        norm_w = w / max_w
        weights.extend([norm_w, norm_w])

    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)

    return edge_index, edge_weight
