"""
Full analysis pipeline.

Orchestrates: Parser → Graph Construction → SBERT Embedding → Module Discovery
→ Risk Detection → Impact Prediction → API-ready output.

This is the main entry point for analyzing a project.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from cartograph.config import settings
from cartograph.data.graph import (
    CodeGraphData,
    load_graph_json,
    build_nx_graph,
    build_node_features,
    to_pyg_data,
)
from cartograph.data.features import NodeEmbedder
from cartograph.data.dataset import extract_directory_labels
from cartograph.models.graphsage import ModuleDiscovery
from cartograph.models.gat import ImpactPredictor
from cartograph.models.risk import RiskDetector, Risk

logger = logging.getLogger(__name__)


@dataclass
class Module:
    """A discovered semantic module (a 'LEGO block')."""
    id: int
    name: str
    description: str
    node_ids: list[str]
    node_count: int
    file_paths: list[str]
    centroid_features: Optional[list[float]] = None


@dataclass
class AnalysisResult:
    """Complete analysis output for a project."""
    project_name: str
    analysis_time_sec: float

    # layer 1: structure
    total_nodes: int
    total_edges: int
    modules: list[Module]
    module_quality: float  # silhouette score

    # layer 2: risks
    risks: list[dict]
    risk_summary: dict[str, int]  # category -> count

    # layer 3: impact prediction ready
    impact_model_loaded: bool

    # raw data for frontend
    graph_nodes: list[dict]
    graph_edges: list[dict]

    def to_dict(self) -> dict:
        return {
            "project_name": self.project_name,
            "analysis_time_sec": self.analysis_time_sec,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "modules": [
                {
                    "id": m.id,
                    "name": m.name,
                    "description": m.description,
                    "node_ids": m.node_ids,
                    "node_count": m.node_count,
                    "file_paths": m.file_paths,
                }
                for m in self.modules
            ],
            "module_quality": self.module_quality,
            "risks": self.risks,
            "risk_summary": self.risk_summary,
            "impact_model_loaded": self.impact_model_loaded,
            "graph_nodes": self.graph_nodes,
            "graph_edges": self.graph_edges,
        }


class AnalysisPipeline:
    """
    Full Cartograph analysis pipeline.

    Usage:
        pipeline = AnalysisPipeline()
        result = pipeline.analyze("/path/to/project")
    """

    def __init__(
        self,
        sage_model_path: Optional[str] = None,
        gat_model_path: Optional[str] = None,
    ):
        self.embedder = NodeEmbedder(settings.sbert_model)
        self.sage_model_path = sage_model_path
        self.gat_model_path = gat_model_path
        self._module_discovery: Optional[ModuleDiscovery] = None
        self._impact_predictor: Optional[ImpactPredictor] = None

    def analyze(self, project_path: str) -> AnalysisResult:
        """Run full analysis on a project directory."""
        start = time.time()
        project_path = os.path.abspath(project_path)
        logger.info("Starting analysis of %s", project_path)

        # step 1: parse
        graph_json = self._parse_project(project_path)

        # step 2: build graph
        G, node_id_to_idx, node_metadata = build_nx_graph(graph_json)
        idx_to_node_id = {v: k for k, v in node_id_to_idx.items()}

        logger.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

        # step 3: SBERT embeddings
        ordered_nodes = [
            node_metadata[idx_to_node_id[i]]
            for i in range(len(node_id_to_idx))
        ]
        sbert_embeddings = self.embedder.embed_nodes(ordered_nodes)

        # step 4: build PyG data
        node_features = build_node_features(G, sbert_embeddings)
        pyg_data = to_pyg_data(G, node_features)
        input_dim = node_features.shape[1]

        # step 5: module discovery (Layer 1)
        modules, module_quality = self._discover_modules(
            pyg_data, input_dim, ordered_nodes, idx_to_node_id, node_metadata
        )

        # step 6: risk detection (Layer 2)
        module_labels = np.array([
            next((m.id for m in modules if nid in m.node_ids), -1)
            for nid in [idx_to_node_id[i] for i in range(len(node_id_to_idx))]
        ])
        risks = self._detect_risks(G, node_metadata, idx_to_node_id, module_labels)

        # step 7: prepare impact predictor (Layer 3)
        impact_ready = self._prepare_impact(pyg_data, input_dim)

        # step 8: build output
        graph_nodes = self._build_graph_output_nodes(
            ordered_nodes, node_id_to_idx, modules, risks
        )
        graph_edges = self._build_graph_output_edges(graph_json)

        elapsed = time.time() - start
        logger.info("Analysis complete in %.1fs", elapsed)

        risk_dicts = [
            {
                "category": r.category.value,
                "severity": r.severity.value,
                "affected_node_ids": r.affected_node_ids,
                "title": r.title,
                "explanation": r.explanation,
                "suggestion": r.suggestion,
            }
            for r in risks
        ]

        risk_summary: dict[str, int] = {}
        for r in risks:
            risk_summary[r.severity.value] = risk_summary.get(r.severity.value, 0) + 1

        return AnalysisResult(
            project_name=graph_json.get("projectName", "unknown"),
            analysis_time_sec=round(elapsed, 2),
            total_nodes=G.number_of_nodes(),
            total_edges=G.number_of_edges(),
            modules=modules,
            module_quality=round(module_quality, 4),
            risks=risk_dicts,
            risk_summary=risk_summary,
            impact_model_loaded=impact_ready,
            graph_nodes=graph_nodes,
            graph_edges=graph_edges,
        )

    def predict_impact(
        self,
        project_path: str,
        modified_node_id: str,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Predict which nodes are affected by modifying the given node.
        Must call analyze() first.
        """
        if self._impact_predictor is None:
            raise RuntimeError("Impact predictor not loaded. Call analyze() first.")

        # re-parse (in production, cache the PyG data)
        graph_json = self._parse_project(project_path)
        G, node_id_to_idx, node_metadata = build_nx_graph(graph_json)
        idx_to_node_id = {v: k for k, v in node_id_to_idx.items()}

        ordered_nodes = [
            node_metadata[idx_to_node_id[i]]
            for i in range(len(node_id_to_idx))
        ]
        sbert_embeddings = self.embedder.embed_nodes(ordered_nodes)
        node_features = build_node_features(G, sbert_embeddings)
        pyg_data = to_pyg_data(G, node_features)

        # find node index
        mod_idx = node_id_to_idx.get(modified_node_id)
        if mod_idx is None:
            raise ValueError(f"Node {modified_node_id} not found in graph")

        predictions = self._impact_predictor.predict(pyg_data, mod_idx, top_k=top_k)

        results = []
        for idx, prob in predictions:
            nid = idx_to_node_id.get(idx, "")
            meta = node_metadata.get(nid, {})
            results.append({
                "node_id": nid,
                "name": meta.get("name", "?"),
                "kind": meta.get("kind", "?"),
                "file_path": meta.get("filePath", "?"),
                "impact_probability": round(prob, 4),
            })

        return results

    def _parse_project(self, project_path: str) -> dict:
        """Run the TypeScript parser on a project."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            cmd = f"{settings.parser_bin} {project_path} -o {output_path} --snippets"
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                logger.error("Parser failed: %s", result.stderr)
                raise RuntimeError(f"Parser failed: {result.stderr[:500]}")

            return load_graph_json(output_path)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def _discover_modules(
        self,
        pyg_data,
        input_dim: int,
        ordered_nodes: list[dict],
        idx_to_node_id: dict[int, str],
        node_metadata: dict[str, dict],
    ) -> tuple[list[Module], float]:
        """Run GraphSAGE module discovery."""
        if self._module_discovery is None:
            self._module_discovery = ModuleDiscovery(input_dim)
            if self.sage_model_path and os.path.exists(self.sage_model_path):
                self._module_discovery.load(self.sage_model_path)

        labels, num_modules, silhouette = self._module_discovery.discover_modules(pyg_data)

        # build Module objects
        modules: list[Module] = []
        for mod_id in range(num_modules):
            member_indices = np.where(labels == mod_id)[0]
            member_ids = [idx_to_node_id[int(i)] for i in member_indices]
            member_files = list(set(
                node_metadata.get(nid, {}).get("filePath", "")
                for nid in member_ids
            ))

            # generate module name from member names
            member_names = [
                node_metadata.get(nid, {}).get("name", "")
                for nid in member_ids
            ]
            name = self._generate_module_name(member_names, member_files)

            modules.append(Module(
                id=mod_id,
                name=name,
                description=f"Module with {len(member_ids)} nodes across {len(member_files)} files",
                node_ids=member_ids,
                node_count=len(member_ids),
                file_paths=member_files,
            ))

        return modules, silhouette

    def _generate_module_name(
        self,
        member_names: list[str],
        member_files: list[str],
    ) -> str:
        """Heuristic module naming from member names and file paths."""
        # try to find common directory
        dirs = set()
        for fp in member_files:
            parts = fp.replace("\\", "/").split("/")
            meaningful = [
                p for p in parts[:-1]
                if p not in ("src", "app", "lib", "pages", "")
            ]
            if meaningful:
                dirs.add(meaningful[0])

        if len(dirs) == 1:
            return dirs.pop().replace("-", " ").replace("_", " ").title()

        # fall back to common prefix of member names
        kinds: dict[str, int] = {}
        for name in member_names:
            # extract prefix (e.g., handleSubmit -> handle, UserProfile -> User)
            import re
            parts = re.findall(r"[A-Z][a-z]+|[a-z]+", name)
            if parts:
                kinds[parts[0].lower()] = kinds.get(parts[0].lower(), 0) + 1

        if kinds:
            top = sorted(kinds.items(), key=lambda x: -x[1])[0][0]
            return top.title() + " Logic"

        return f"Module"

    def _detect_risks(
        self,
        G,
        node_metadata,
        idx_to_node_id,
        module_labels,
    ) -> list[Risk]:
        """Run risk detection."""
        detector = RiskDetector(G, node_metadata, idx_to_node_id)
        risks = detector.detect_all()
        # also check for god modules
        god_risks = detector._detect_god_modules(module_labels)
        risks.extend(god_risks)
        return risks

    def _prepare_impact(self, pyg_data, input_dim: int) -> bool:
        """Load or initialize impact predictor."""
        try:
            self._impact_predictor = ImpactPredictor(input_dim)
            if self.gat_model_path and os.path.exists(self.gat_model_path):
                self._impact_predictor.load(self.gat_model_path)
                return True
            # without pretrained weights, impact prediction still works
            # (just with random weights — user should train first)
            return False
        except Exception as e:
            logger.warning("Failed to load impact predictor: %s", e)
            return False

    def _build_graph_output_nodes(
        self,
        ordered_nodes,
        node_id_to_idx,
        modules,
        risks,
    ) -> list[dict]:
        """Build frontend-ready node list."""
        # build risk lookup
        risk_by_node: dict[str, list[dict]] = {}
        for r in risks:
            for nid in r.affected_node_ids:
                risk_by_node.setdefault(nid, []).append({
                    "category": r.category.value,
                    "severity": r.severity.value,
                    "title": r.title,
                })

        # build module lookup
        module_by_node: dict[str, int] = {}
        for m in modules:
            for nid in m.node_ids:
                module_by_node[nid] = m.id

        nodes = []
        for node in ordered_nodes:
            nid = node["id"]
            nodes.append({
                "id": nid,
                "name": node["name"],
                "kind": node["kind"],
                "filePath": node["filePath"],
                "startLine": node["startLine"],
                "endLine": node["endLine"],
                "moduleId": module_by_node.get(nid, -1),
                "risks": risk_by_node.get(nid, []),
                "features": node.get("features", {}),
            })

        return nodes

    def _build_graph_output_edges(self, graph_json: dict) -> list[dict]:
        """Build frontend-ready edge list."""
        return [
            {
                "source": e["sourceId"],
                "target": e["targetId"],
                "kind": e["kind"],
                "weight": e.get("weight", 1.0),
            }
            for e in graph_json.get("edges", [])
        ]
