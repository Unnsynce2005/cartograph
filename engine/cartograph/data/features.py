"""
Semantic feature engineering for code graph nodes.

Each node gets a 384-dim SBERT embedding based on:
  - function/component name (camelCase split into words)
  - file path context
  - source snippet (if available)
  - parameter names

This embedding captures the *semantic intent* of a code entity,
complementing the structural features from the parser.
"""
from __future__ import annotations

import re
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def split_identifier(name: str) -> str:
    """Split camelCase/PascalCase/snake_case into words."""
    # insert space before uppercase letters
    s = re.sub(r"([A-Z])", r" \1", name)
    # replace underscores and hyphens with spaces
    s = re.sub(r"[_\-]", " ", s)
    return s.strip().lower()


def build_node_text(node: dict) -> str:
    """
    Build a natural language description of a node for SBERT encoding.
    Combine name, kind, file path context, and snippet.
    """
    parts = []

    # name
    name = node.get("name", "unknown")
    readable_name = split_identifier(name)
    kind = node.get("kind", "function")
    parts.append(f"{kind}: {readable_name}")

    # file path context
    file_path = node.get("filePath", "")
    path_parts = file_path.replace("\\", "/").split("/")
    meaningful_parts = [
        p for p in path_parts
        if p not in ("src", "lib", "app", "index.ts", "index.tsx", "index.js")
        and not p.startswith(".")
    ]
    if meaningful_parts:
        parts.append("in " + " / ".join(meaningful_parts[-3:]))

    # source snippet (first 200 chars for embedding quality)
    snippet = node.get("sourceSnippet", "")
    if snippet:
        # strip boilerplate
        clean = re.sub(r"//.*", "", snippet)  # remove comments
        clean = re.sub(r"/\*.*?\*/", "", clean, flags=re.DOTALL)
        clean = re.sub(r"\s+", " ", clean).strip()
        if len(clean) > 200:
            clean = clean[:200]
        if clean:
            parts.append(clean)

    # features as context
    features = node.get("features", {})
    if features.get("hasJSX"):
        parts.append("renders JSX")
    if features.get("hasAwait"):
        parts.append("async operation")
    if features.get("paramCount", 0) > 3:
        parts.append(f"{features['paramCount']} parameters")

    return ". ".join(parts)


class NodeEmbedder:
    """Compute SBERT embeddings for code graph nodes."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SBERT model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_nodes(
        self,
        nodes: list[dict],
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Generate SBERT embeddings for a list of node dicts.
        Returns: np.ndarray of shape (len(nodes), 384)
        """
        texts = [build_node_text(n) for n in nodes]

        logger.info("Encoding %d nodes with SBERT...", len(texts))
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
        )

        return np.array(embeddings, dtype=np.float32)

    def embed_single(self, node: dict) -> np.ndarray:
        """Embed a single node."""
        text = build_node_text(node)
        emb = self.model.encode([text], normalize_embeddings=True)
        return np.array(emb[0], dtype=np.float32)
