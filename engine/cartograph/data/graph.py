"""
Code graph representation and conversion to PyTorch Geometric Data objects.

The parser outputs JSON with nodes and edges. This module:
1. Loads the JSON into a NetworkX DiGraph (for analysis and traversal)
2. Converts to torch_geometric.data.Data (for GNN training/inference)
3. Handles node feature normalization and edge type encoding
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

EDGE_TYPE_MAP = {
    "calls": 0,
    "imports": 1,
    "data_flow": 2,
    "jsx_renders": 3,
    "type_ref": 4,
    "prop_passes": 5,
    "state_reads": 6,
    "state_writes": 7,
    "temporal": 8,  # co-change edges from git history
}

NODE_KIND_MAP = {
    "function": 0,
    "arrow_function": 1,
    "component": 2,
    "hook": 3,
    "class": 4,
    "variable": 5,
    "type_alias": 6,
    "interface": 7,
    "enum": 8,
    "api_route": 9,
    "page": 10,
    "middleware": 11,
}

# structural features extracted from parser JSON
STRUCTURAL_FEATURE_KEYS = [
    "loc",
    "cyclomaticComplexity",
    "paramCount",
    "returnCount",
    "calleeCount",
    "callerCount",
    "depthInFileTree",
    "hasJSX",
    "hasTryCatch",
    "hasAwait",
    "importCount",
    "jsxChildCount",
    "propsCount",
]


@dataclass
class CodeGraphData:
    """Container for a parsed code graph with both NetworkX and PyG representations."""

    project_name: str
    nx_graph: nx.DiGraph
    node_id_to_idx: dict[str, int]
    idx_to_node_id: dict[int, str]
    node_metadata: dict[str, dict]  # id -> full node JSON
    pyg_data: Optional[Data] = None

    @property
    def num_nodes(self) -> int:
        return self.nx_graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.nx_graph.number_of_edges()


def load_graph_json(json_path: str | Path) -> dict:
    with open(json_path, "r") as f:
        return json.load(f)


def build_nx_graph(graph_json: dict) -> tuple[nx.DiGraph, dict[str, int], dict[str, dict]]:
    """
    Build a NetworkX directed graph from parser JSON output.
    Returns (graph, node_id_to_idx, node_metadata).
    """
    G = nx.DiGraph()
    node_id_to_idx: dict[str, int] = {}
    node_metadata: dict[str, dict] = {}

    for idx, node in enumerate(graph_json["nodes"]):
        nid = node["id"]
        node_id_to_idx[nid] = idx
        node_metadata[nid] = node

        # node attributes for NetworkX
        kind_onehot = [0] * len(NODE_KIND_MAP)
        kind_idx = NODE_KIND_MAP.get(node["kind"], 0)
        kind_onehot[kind_idx] = 1

        features = node.get("features", {})
        structural = [float(features.get(k, 0)) for k in STRUCTURAL_FEATURE_KEYS]

        G.add_node(
            idx,
            node_id=nid,
            name=node["name"],
            kind=node["kind"],
            file_path=node["filePath"],
            structural_features=structural,
            kind_onehot=kind_onehot,
        )

    for edge in graph_json["edges"]:
        src = node_id_to_idx.get(edge["sourceId"])
        tgt = node_id_to_idx.get(edge["targetId"])
        if src is None or tgt is None:
            continue

        edge_type = EDGE_TYPE_MAP.get(edge["kind"], 0)
        weight = edge.get("weight", 1.0)
        G.add_edge(src, tgt, edge_type=edge_type, weight=weight)

    return G, node_id_to_idx, node_metadata


def build_node_features(
    G: nx.DiGraph,
    sbert_embeddings: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Assemble the full node feature matrix.

    Features per node:
    - structural features (13 dims, from parser)
    - node kind one-hot (12 dims)
    - graph-derived features (4 dims: in-degree, out-degree, pagerank, clustering coeff)
    - SBERT semantic embedding (384 dims, optional)

    Total: 13 + 12 + 4 = 29 (without SBERT) or 29 + 384 = 413 (with SBERT)
    """
    num_nodes = G.number_of_nodes()
    sorted_nodes = sorted(G.nodes())

    # structural features
    structural = np.zeros((num_nodes, len(STRUCTURAL_FEATURE_KEYS)), dtype=np.float32)
    for node in sorted_nodes:
        data = G.nodes[node]
        structural[node] = data.get("structural_features", [0] * len(STRUCTURAL_FEATURE_KEYS))

    # normalize structural features (z-score per column)
    means = structural.mean(axis=0)
    stds = structural.std(axis=0)
    stds[stds == 0] = 1.0
    structural = (structural - means) / stds

    # kind one-hot
    kind_onehot = np.zeros((num_nodes, len(NODE_KIND_MAP)), dtype=np.float32)
    for node in sorted_nodes:
        data = G.nodes[node]
        kind_onehot[node] = data.get("kind_onehot", [0] * len(NODE_KIND_MAP))

    # graph-derived features
    in_degrees = np.array([G.in_degree(n) for n in sorted_nodes], dtype=np.float32)
    out_degrees = np.array([G.out_degree(n) for n in sorted_nodes], dtype=np.float32)

    try:
        pagerank = nx.pagerank(G, max_iter=100)
        pr_values = np.array([pagerank.get(n, 0.0) for n in sorted_nodes], dtype=np.float32)
    except nx.NetworkXError:
        pr_values = np.zeros(num_nodes, dtype=np.float32)

    undirected = G.to_undirected()
    clustering = nx.clustering(undirected)
    cl_values = np.array([clustering.get(n, 0.0) for n in sorted_nodes], dtype=np.float32)

    graph_features = np.stack([in_degrees, out_degrees, pr_values, cl_values], axis=1)
    # normalize
    gf_means = graph_features.mean(axis=0)
    gf_stds = graph_features.std(axis=0)
    gf_stds[gf_stds == 0] = 1.0
    graph_features = (graph_features - gf_means) / gf_stds

    # concatenate all
    parts = [structural, kind_onehot, graph_features]
    if sbert_embeddings is not None:
        assert sbert_embeddings.shape[0] == num_nodes
        parts.append(sbert_embeddings.astype(np.float32))

    feature_matrix = np.concatenate(parts, axis=1)
    return torch.from_numpy(feature_matrix)


def to_pyg_data(
    G: nx.DiGraph,
    node_features: torch.Tensor,
    edge_type_tensor: Optional[torch.Tensor] = None,
) -> Data:
    """Convert NetworkX graph + feature matrix to PyG Data object."""
    sorted_nodes = sorted(G.nodes())

    # build edge index
    edges = list(G.edges())
    if len(edges) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_types = torch.zeros(0, dtype=torch.long)
        edge_weights = torch.zeros(0, dtype=torch.float)
    else:
        src = [e[0] for e in edges]
        tgt = [e[1] for e in edges]
        edge_index = torch.tensor([src, tgt], dtype=torch.long)
        edge_attr_types = torch.tensor(
            [G.edges[e].get("edge_type", 0) for e in edges], dtype=torch.long
        )
        edge_weights = torch.tensor(
            [G.edges[e].get("weight", 1.0) for e in edges], dtype=torch.float
        )

    data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_type=edge_attr_types,
        edge_weight=edge_weights,
        num_nodes=len(sorted_nodes),
    )
    return data


def load_and_convert(
    json_path: str | Path,
    sbert_embeddings: Optional[np.ndarray] = None,
) -> CodeGraphData:
    """Full pipeline: JSON -> NetworkX -> PyG Data."""
    graph_json = load_graph_json(json_path)
    G, node_id_to_idx, node_metadata = build_nx_graph(graph_json)

    idx_to_node_id = {v: k for k, v in node_id_to_idx.items()}

    node_features = build_node_features(G, sbert_embeddings)
    pyg_data = to_pyg_data(G, node_features)

    return CodeGraphData(
        project_name=graph_json.get("projectName", "unknown"),
        nx_graph=G,
        node_id_to_idx=node_id_to_idx,
        idx_to_node_id=idx_to_node_id,
        node_metadata=node_metadata,
        pyg_data=pyg_data,
    )
