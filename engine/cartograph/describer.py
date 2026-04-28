"""
Module description generator.

Takes a cluster of code nodes and generates a human-readable
description using an LLM, grounded in the GNN-discovered structure.

This is what makes Cartograph different from "paste code into ChatGPT":
- ChatGPT sees raw text and hallucinates structure
- Cartograph sees GNN-derived semantic clusters and asks LLM to name them

The prompt explicitly tells the LLM: this grouping was determined
by graph topology, not by the LLM. LLM's job is just to *describe*
what the structure represents in plain language.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import httpx

from cartograph.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ModuleDescription:
    """Human-readable description of a discovered module."""
    short_name: str           # 2-4 words, e.g., "User Authentication"
    one_liner: str            # one sentence summary
    responsibility: str       # what this module is responsible for
    connections: list[str]    # which other modules it talks to
    representative_files: list[str]


class ModuleDescriber:
    """
    Generates plain-language descriptions of GNN-discovered modules.
    Uses Claude (or falls back to heuristic naming if no API key).
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.anthropic_api_key
        self.model = settings.llm_model
        self._enabled = bool(self.api_key)
        if not self._enabled:
            logger.warning("No Anthropic API key — falling back to heuristic naming")

    def describe_module(
        self,
        module_id: int,
        node_metadata: list[dict],
        connections_to: dict[int, int],  # other_module_id -> edge count
        all_modules: dict[int, list[dict]],  # for context about other modules
    ) -> ModuleDescription:
        """Generate a description for a single module."""
        if not self._enabled or len(node_metadata) == 0:
            return self._heuristic_describe(module_id, node_metadata, connections_to, all_modules)

        try:
            return self._llm_describe(module_id, node_metadata, connections_to, all_modules)
        except Exception as e:
            logger.warning("LLM description failed: %s. Falling back to heuristic.", e)
            return self._heuristic_describe(module_id, node_metadata, connections_to, all_modules)

    def _llm_describe(
        self,
        module_id: int,
        nodes: list[dict],
        connections_to: dict[int, int],
        all_modules: dict[int, list[dict]],
    ) -> ModuleDescription:
        """Call Claude API to generate description."""
        # build context
        node_summary = self._summarize_nodes(nodes)
        files = sorted(set(n.get("filePath", "") for n in nodes))[:8]

        connection_summary = []
        for other_id, edge_count in sorted(connections_to.items(), key=lambda x: -x[1])[:5]:
            other_nodes = all_modules.get(other_id, [])
            if other_nodes:
                other_files = list(set(n.get("filePath", "") for n in other_nodes))[:3]
                connection_summary.append(
                    f"  - Module {other_id} ({edge_count} connections): files include {', '.join(other_files)}"
                )

        prompt = f"""You are analyzing one module of a TypeScript/React codebase. The module was automatically grouped by a graph neural network based on call patterns and dependencies. Your job is ONLY to describe what this group of code does, in plain language a non-developer can understand.

This module contains {len(nodes)} code entities:
{node_summary}

Files in this module:
{chr(10).join('  - ' + f for f in files)}

This module connects to other modules:
{chr(10).join(connection_summary) if connection_summary else '  (no connections to other modules)'}

Respond in EXACTLY this format (no markdown, no extra text):

NAME: [2-4 word descriptive name, e.g. "User Authentication" or "Shopping Cart"]
SUMMARY: [one sentence describing what this module does, in plain language a non-coder understands]
RESPONSIBILITY: [2-3 sentences explaining what this module is responsible for and why it exists]
"""

        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": 400,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30.0,
        )
        response.raise_for_status()
        text = response.json()["content"][0]["text"]

        # parse response
        name_match = re.search(r"NAME:\s*(.+?)(?:\n|$)", text)
        summary_match = re.search(r"SUMMARY:\s*(.+?)(?:\n|$)", text)
        resp_match = re.search(r"RESPONSIBILITY:\s*(.+?)(?=\n[A-Z]+:|\Z)", text, re.DOTALL)

        return ModuleDescription(
            short_name=(name_match.group(1).strip() if name_match else f"Module {module_id}"),
            one_liner=(summary_match.group(1).strip() if summary_match else ""),
            responsibility=(resp_match.group(1).strip() if resp_match else ""),
            connections=[f"Module {oid}" for oid in connections_to],
            representative_files=files,
        )

    def _summarize_nodes(self, nodes: list[dict]) -> str:
        """Summarize a list of nodes for LLM context."""
        # group by kind
        by_kind: dict[str, list[str]] = {}
        for n in nodes:
            by_kind.setdefault(n.get("kind", "function"), []).append(n.get("name", "?"))

        lines = []
        for kind, names in sorted(by_kind.items()):
            display_names = names[:8]
            suffix = f" (+{len(names)-8} more)" if len(names) > 8 else ""
            lines.append(f"  {kind}: {', '.join(display_names)}{suffix}")
        return "\n".join(lines)

    def _heuristic_describe(
        self,
        module_id: int,
        nodes: list[dict],
        connections_to: dict[int, int],
        all_modules: dict[int, list[dict]],
    ) -> ModuleDescription:
        """Fallback: name modules by directory + node kind heuristics."""
        if not nodes:
            return ModuleDescription(
                short_name=f"Module {module_id}",
                one_liner="Empty module",
                responsibility="",
                connections=[],
                representative_files=[],
            )

        # find dominant directory
        dirs: dict[str, int] = {}
        for n in nodes:
            path = n.get("filePath", "").replace("\\", "/")
            parts = [p for p in path.split("/") if p and p not in ("src", "app", "lib", "pages")]
            if parts:
                dirs[parts[0]] = dirs.get(parts[0], 0) + 1

        if dirs:
            top_dir = max(dirs.items(), key=lambda x: x[1])[0]
            name = top_dir.replace("-", " ").replace("_", " ").title()
        else:
            name = f"Module {module_id}"

        # describe contents
        kinds: dict[str, int] = {}
        for n in nodes:
            kinds[n.get("kind", "?")] = kinds.get(n.get("kind", "?"), 0) + 1

        kind_summary = ", ".join(
            f"{count} {kind}{'s' if count > 1 else ''}"
            for kind, count in sorted(kinds.items(), key=lambda x: -x[1])[:3]
        )

        files = sorted(set(n.get("filePath", "") for n in nodes))[:8]

        return ModuleDescription(
            short_name=name,
            one_liner=f"Contains {kind_summary} across {len(files)} files",
            responsibility=f"Code grouped by graph topology in the {top_dir if dirs else 'core'} area of the project.",
            connections=[f"Module {oid}" for oid in connections_to],
            representative_files=files,
        )
