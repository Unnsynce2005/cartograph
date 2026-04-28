"""
Blueprint endpoint additions.

Returns the project as a card-based blueprint instead of a force-directed graph.
"""
from fastapi import HTTPException
from pydantic import BaseModel


class BlueprintCard(BaseModel):
    module_id: int
    name: str
    summary: str
    responsibility: str
    node_count: int
    file_count: int
    files: list[str]
    risks: list[dict]
    connects_to: list[dict]  # [{module_id, name, strength}]
    nodes: list[dict]


class Blueprint(BaseModel):
    project_name: str
    overview: str
    cards: list[BlueprintCard]
    total_risks: int
    risk_severity_counts: dict[str, int]


def build_blueprint(analysis: dict, describer) -> Blueprint:
    """Convert raw analysis result into a card-based blueprint."""
    modules = analysis.get("modules", [])
    nodes = analysis.get("graph_nodes", [])
    edges = analysis.get("graph_edges", [])
    risks = analysis.get("risks", [])

    # group nodes by module
    nodes_by_module: dict[int, list[dict]] = {}
    for node in nodes:
        mid = node.get("moduleId", -1)
        nodes_by_module.setdefault(mid, []).append(node)

    # compute inter-module connection strength
    node_to_module = {n["id"]: n.get("moduleId", -1) for n in nodes}
    inter_module: dict[int, dict[int, int]] = {}
    for edge in edges:
        src_mod = node_to_module.get(edge["source"], -1)
        tgt_mod = node_to_module.get(edge["target"], -1)
        if src_mod != tgt_mod and src_mod >= 0 and tgt_mod >= 0:
            inter_module.setdefault(src_mod, {})
            inter_module[src_mod][tgt_mod] = inter_module[src_mod].get(tgt_mod, 0) + 1

    # group risks by module
    risks_by_module: dict[int, list[dict]] = {}
    for risk in risks:
        for nid in risk.get("affected_node_ids", []):
            mid = node_to_module.get(nid, -1)
            if mid >= 0:
                risks_by_module.setdefault(mid, []).append(risk)
                break

    # build cards
    cards = []
    for mod in modules:
        mid = mod["id"]
        mod_nodes = nodes_by_module.get(mid, [])
        connections = inter_module.get(mid, {})

        # describe via LLM or heuristic
        desc = describer.describe_module(
            mid, mod_nodes, connections, nodes_by_module
        )

        # build connection list
        conn_list = []
        for other_mid, count in sorted(connections.items(), key=lambda x: -x[1])[:5]:
            other_mod = next((m for m in modules if m["id"] == other_mid), None)
            if other_mod:
                conn_list.append({
                    "module_id": other_mid,
                    "name": other_mod.get("name", f"Module {other_mid}"),
                    "strength": count,
                })

        files = sorted(set(n.get("filePath", "") for n in mod_nodes))

        cards.append(BlueprintCard(
            module_id=mid,
            name=desc.short_name,
            summary=desc.one_liner,
            responsibility=desc.responsibility,
            node_count=len(mod_nodes),
            file_count=len(files),
            files=files,
            risks=risks_by_module.get(mid, []),
            connects_to=conn_list,
            nodes=[
                {
                    "id": n["id"],
                    "name": n["name"],
                    "kind": n["kind"],
                    "filePath": n.get("filePath", ""),
                }
                for n in mod_nodes
            ],
        ))

    # severity counts
    sev_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for r in risks:
        sev = r.get("severity", "low")
        sev_counts[sev] = sev_counts.get(sev, 0) + 1

    overview = (
        f"This project contains {len(nodes)} code entities organized into "
        f"{len(modules)} semantic modules. "
        f"Cartograph detected {len(risks)} potential issues."
    )

    return Blueprint(
        project_name=analysis.get("project_name", "unknown"),
        overview=overview,
        cards=cards,
        total_risks=len(risks),
        risk_severity_counts=sev_counts,
    )
