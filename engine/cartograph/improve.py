"""
Improvement suggester (Layer 4 lite).

Given a module + a desired change, generate a precise, context-aware
prompt the user can paste into Cursor/Claude/Copilot.

Uses GAT impact prediction to determine which other modules will be
affected, and constrains the prompt to avoid breaking unrelated parts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from cartograph.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ImprovementSuggestion:
    summary: str
    affected_modules: list[dict]  # [{module_id, name, probability}]
    constraints: list[str]
    generated_prompt: str


def generate_improvement_prompt(
    user_intent: str,           # e.g., "add a share button"
    target_module: dict,        # the module the user wants to modify
    affected_predictions: list[dict],  # from GAT
    all_modules: list[dict],
    api_key: Optional[str] = None,
) -> ImprovementSuggestion:
    """
    Generate a precise modification prompt.
    """
    # group predictions by module
    affected_by_module: dict[int, list[dict]] = {}
    for pred in affected_predictions:
        mid = pred.get("module_id", -1)
        if mid >= 0:
            affected_by_module.setdefault(mid, []).append(pred)

    # rank by max impact probability per module
    affected_modules_ranked = []
    for mid, preds in affected_by_module.items():
        max_prob = max(p.get("impact_probability", 0) for p in preds)
        mod = next((m for m in all_modules if m["id"] == mid), None)
        if mod:
            affected_modules_ranked.append({
                "module_id": mid,
                "name": mod.get("name", f"Module {mid}"),
                "probability": round(max_prob, 3),
                "affected_nodes": [p.get("name", "?") for p in preds[:3]],
            })

    affected_modules_ranked.sort(key=lambda x: -x["probability"])

    high_impact = [m for m in affected_modules_ranked if m["probability"] >= 0.5]
    low_impact = [m for m in affected_modules_ranked if m["probability"] < 0.3]

    # build constraints
    constraints = [
        f"Only modify files in the '{target_module.get('name', 'target')}' module.",
    ]
    if low_impact:
        names = [m["name"] for m in low_impact[:5]]
        constraints.append(
            f"Do NOT modify files in these modules — they should be unaffected: {', '.join(names)}"
        )
    if high_impact:
        names = [m["name"] for m in high_impact[:3]]
        constraints.append(
            f"You may need to update interfaces in: {', '.join(names)} — but be conservative."
        )

    # representative files
    files = target_module.get("files", [])[:5]

    # build the prompt
    prompt = f"""Modification request: {user_intent}

Target scope:
- Module: {target_module.get('name')}
- Responsibility: {target_module.get('responsibility', '')}
- Primary files to modify: {', '.join(files) if files else 'see module nodes'}

Constraints (verified by static analysis):
{chr(10).join('- ' + c for c in constraints)}

Predicted cascade (from GNN impact analysis):
{chr(10).join(f'- {m["name"]} (impact prob: {m["probability"]:.2f}): may need updates in {", ".join(m["affected_nodes"][:2])}' for m in affected_modules_ranked[:5]) if affected_modules_ranked else '- None predicted'}

Please implement the change while respecting these constraints. If you need to modify code outside the target scope, list the changes explicitly before making them.
"""

    summary = (
        f"Modifying '{target_module.get('name')}' will most likely cascade to: "
        + (", ".join(m["name"] for m in high_impact[:3]) if high_impact else "no other modules")
    )

    return ImprovementSuggestion(
        summary=summary,
        affected_modules=affected_modules_ranked,
        constraints=constraints,
        generated_prompt=prompt,
    )
