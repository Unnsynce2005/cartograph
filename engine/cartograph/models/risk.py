"""
Risk detection engine (Layer 2).

Hybrid approach:
  1. Rule-based detectors for high-confidence patterns
     (no auth, hardcoded secrets, circular deps)
  2. Graph-feature classifier for structural anti-patterns
     (excessive coupling, orphan modules, state sync issues)

Each detected risk has:
  - category (auth, secret, coupling, cycle, error_handling, state_sync)
  - severity (critical, high, medium, low)
  - affected node IDs
  - explanation text
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import networkx as nx
import numpy as np

from cartograph.config import settings

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskCategory(str, Enum):
    AUTH_MISSING = "auth_missing"
    SECRET_EXPOSED = "secret_exposed"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    EXCESSIVE_COUPLING = "excessive_coupling"
    UNHANDLED_ERROR = "unhandled_error"
    STATE_SYNC = "state_sync"
    ORPHAN_NODE = "orphan_node"
    GOD_MODULE = "god_module"


@dataclass
class Risk:
    category: RiskCategory
    severity: Severity
    affected_node_ids: list[str]
    title: str
    explanation: str
    suggestion: str
    metadata: dict = field(default_factory=dict)


class RiskDetector:
    """Run all risk detectors on a code graph."""

    def __init__(
        self,
        nx_graph: nx.DiGraph,
        node_metadata: dict[str, dict],
        idx_to_node_id: dict[int, str],
    ):
        self.G = nx_graph
        self.meta = node_metadata
        self.idx_to_id = idx_to_node_id
        self.id_to_idx = {v: k for k, v in idx_to_node_id.items()}

    def detect_all(self) -> list[Risk]:
        risks: list[Risk] = []
        risks.extend(self._detect_auth_missing())
        risks.extend(self._detect_secrets())
        risks.extend(self._detect_circular_deps())
        risks.extend(self._detect_excessive_coupling())
        risks.extend(self._detect_unhandled_errors())
        risks.extend(self._detect_orphan_nodes())
        risks.extend(self._detect_god_modules())

        logger.info("Detected %d risks across %d categories",
                     len(risks), len(set(r.category for r in risks)))
        return risks

    def _detect_auth_missing(self) -> list[Risk]:
        """Find API route handlers without auth middleware references."""
        risks = []
        auth_patterns = settings.risk_auth_patterns

        for nid, meta in self.meta.items():
            if meta.get("kind") != "api_route":
                continue

            # check if this route or any of its callers reference auth
            idx = self.id_to_idx.get(nid)
            if idx is None:
                continue

            has_auth = False
            # check predecessors (middleware chain)
            for pred in self.G.predecessors(idx):
                pred_id = self.idx_to_id.get(pred, "")
                pred_meta = self.meta.get(pred_id, {})
                pred_name = pred_meta.get("name", "").lower()
                if any(p in pred_name for p in auth_patterns):
                    has_auth = True
                    break

            # check snippet for auth references
            snippet = meta.get("sourceSnippet", "").lower()
            if any(p in snippet for p in auth_patterns):
                has_auth = True

            if not has_auth:
                risks.append(Risk(
                    category=RiskCategory.AUTH_MISSING,
                    severity=Severity.HIGH,
                    affected_node_ids=[nid],
                    title=f"API route `{meta['name']}` has no authentication",
                    explanation=(
                        f"The endpoint at {meta.get('filePath', '?')} does not reference "
                        "any authentication or authorization middleware. Anyone with the "
                        "URL can access this endpoint."
                    ),
                    suggestion=(
                        "Add authentication middleware (e.g., session check, JWT validation) "
                        "before this route handler."
                    ),
                ))

        return risks

    def _detect_secrets(self) -> list[Risk]:
        """Find hardcoded secrets in source snippets."""
        risks = []
        patterns = [re.compile(p) for p in settings.risk_secret_patterns]

        for nid, meta in self.meta.items():
            snippet = meta.get("sourceSnippet", "")
            if not snippet:
                continue

            for pattern in patterns:
                matches = pattern.findall(snippet)
                if matches:
                    risks.append(Risk(
                        category=RiskCategory.SECRET_EXPOSED,
                        severity=Severity.CRITICAL,
                        affected_node_ids=[nid],
                        title=f"Hardcoded secret in `{meta['name']}`",
                        explanation=(
                            f"Found what appears to be a hardcoded credential or API key "
                            f"in {meta.get('filePath', '?')}. This will be exposed if "
                            "the code is pushed to a public repository."
                        ),
                        suggestion="Move secrets to environment variables (.env file).",
                        metadata={"pattern": pattern.pattern},
                    ))
                    break  # one risk per node

        return risks

    def _detect_circular_deps(self) -> list[Risk]:
        """Find cycles in the dependency graph."""
        risks = []

        try:
            cycles = list(nx.simple_cycles(self.G))
        except nx.NetworkXError:
            return risks

        # filter to meaningful cycles (length >= 2)
        for cycle in cycles:
            if len(cycle) < 2:
                continue
            if len(cycle) > 6:
                continue  # skip very long cycles (noisy)

            node_ids = [self.idx_to_id.get(idx, "") for idx in cycle]
            node_names = [self.meta.get(nid, {}).get("name", "?") for nid in node_ids]

            risks.append(Risk(
                category=RiskCategory.CIRCULAR_DEPENDENCY,
                severity=Severity.MEDIUM,
                affected_node_ids=[nid for nid in node_ids if nid],
                title=f"Circular dependency: {' → '.join(node_names[:4])}{'...' if len(node_names) > 4 else ''}",
                explanation=(
                    f"These {len(cycle)} nodes form a dependency cycle. "
                    "Circular dependencies make code harder to test, refactor, "
                    "and reason about. Changes to any node in the cycle may "
                    "require changes to all others."
                ),
                suggestion="Break the cycle by extracting shared logic into a separate module.",
                metadata={"cycle_length": len(cycle)},
            ))

        return risks

    def _detect_excessive_coupling(self) -> list[Risk]:
        """Find nodes with unusually high connectivity."""
        risks = []
        num_nodes = self.G.number_of_nodes()
        if num_nodes < 5:
            return risks

        threshold = max(num_nodes * 0.4, 5)

        for idx in self.G.nodes():
            total_degree = self.G.in_degree(idx) + self.G.out_degree(idx)
            if total_degree >= threshold:
                nid = self.idx_to_id.get(idx, "")
                meta = self.meta.get(nid, {})
                risks.append(Risk(
                    category=RiskCategory.EXCESSIVE_COUPLING,
                    severity=Severity.MEDIUM,
                    affected_node_ids=[nid],
                    title=f"`{meta.get('name', '?')}` is coupled to {total_degree} other nodes",
                    explanation=(
                        f"This node has {self.G.in_degree(idx)} incoming and "
                        f"{self.G.out_degree(idx)} outgoing dependencies, touching "
                        f"{total_degree}/{num_nodes} nodes in the project. Any change "
                        "here has a high probability of cascading effects."
                    ),
                    suggestion="Consider splitting this into smaller, more focused modules.",
                    metadata={"in_degree": self.G.in_degree(idx), "out_degree": self.G.out_degree(idx)},
                ))

        return risks

    def _detect_unhandled_errors(self) -> list[Risk]:
        """Find async functions without try-catch."""
        risks = []
        for nid, meta in self.meta.items():
            features = meta.get("features", {})
            if features.get("hasAwait") and not features.get("hasTryCatch"):
                if meta.get("kind") in ("api_route", "function", "arrow_function", "hook"):
                    risks.append(Risk(
                        category=RiskCategory.UNHANDLED_ERROR,
                        severity=Severity.MEDIUM,
                        affected_node_ids=[nid],
                        title=f"Async function `{meta['name']}` has no error handling",
                        explanation=(
                            f"This async function in {meta.get('filePath', '?')} uses await "
                            "but has no try-catch block. Unhandled promise rejections can crash "
                            "the application or produce confusing error messages."
                        ),
                        suggestion="Wrap async operations in try-catch and handle errors explicitly.",
                    ))
        return risks

    def _detect_orphan_nodes(self) -> list[Risk]:
        """Find nodes with zero connections (isolated code)."""
        risks = []
        for idx in self.G.nodes():
            if self.G.in_degree(idx) == 0 and self.G.out_degree(idx) == 0:
                nid = self.idx_to_id.get(idx, "")
                meta = self.meta.get(nid, {})
                if meta.get("kind") in ("type_alias", "interface", "enum"):
                    continue  # type definitions are often standalone
                if meta.get("exported"):
                    continue  # exported symbols may be used externally

                risks.append(Risk(
                    category=RiskCategory.ORPHAN_NODE,
                    severity=Severity.LOW,
                    affected_node_ids=[nid],
                    title=f"`{meta.get('name', '?')}` is unused (dead code)",
                    explanation=(
                        f"This {meta.get('kind', 'node')} in {meta.get('filePath', '?')} "
                        "has no incoming or outgoing dependencies. It may be dead code "
                        "left over from AI generation iterations."
                    ),
                    suggestion="Remove it if unused, or connect it to the rest of the codebase.",
                ))
        return risks

    def _detect_god_modules(self, module_labels: Optional[np.ndarray] = None) -> list[Risk]:
        """Find modules that are disproportionately large."""
        risks = []
        if module_labels is None:
            return risks

        unique, counts = np.unique(module_labels, return_counts=True)
        total = len(module_labels)
        mean_size = total / len(unique) if len(unique) > 0 else total

        for mod_id, count in zip(unique, counts):
            if count > mean_size * 2.5 and count > 10:
                # find representative nodes
                member_indices = np.where(module_labels == mod_id)[0]
                member_ids = [self.idx_to_id.get(int(i), "") for i in member_indices[:5]]

                risks.append(Risk(
                    category=RiskCategory.GOD_MODULE,
                    severity=Severity.LOW,
                    affected_node_ids=[nid for nid in member_ids if nid],
                    title=f"Module {mod_id} has {count} nodes ({count/total*100:.0f}% of project)",
                    explanation=(
                        f"This module contains {count} out of {total} nodes. "
                        "Oversized modules are a sign of poor separation of concerns "
                        "and make changes risky because they affect a large surface area."
                    ),
                    suggestion="Consider splitting into smaller, more cohesive modules.",
                    metadata={"module_id": int(mod_id), "node_count": int(count)},
                ))

        return risks
