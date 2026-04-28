"""
Cartograph API server v2 — blueprint-first.

Endpoints:
  POST /api/analyze        — upload project zip, run full pipeline
  GET  /api/blueprint/{id} — get blueprint cards (the main UI data)
  POST /api/impact         — predict change impact for a node
  POST /api/improve        — generate context-aware modification prompt
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
import zipfile

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cartograph.pipeline import AnalysisPipeline
from cartograph.describer import ModuleDescriber
from cartograph.blueprint import build_blueprint
from cartograph.improve import generate_improvement_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cartograph",
    description="Code X-ray for AI-generated codebases",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyses: dict[str, dict] = {}
analysis_status: dict[str, str] = {}
blueprints: dict[str, dict] = {}
pipeline = AnalysisPipeline(
    sage_model_path="models/graphsage.pt",
    gat_model_path="models/gat.pt",
)
describer = ModuleDescriber()


class AnalyzeResponse(BaseModel):
    id: str
    status: str


class ImpactRequest(BaseModel):
    project_id: str
    node_id: str
    top_k: int = 10


class ImproveRequest(BaseModel):
    project_id: str
    module_id: int
    user_intent: str


def _run_analysis(analysis_id: str, project_dir: str):
    try:
        analysis_status[analysis_id] = "running"
        result = pipeline.analyze(project_dir)
        result_dict = result.to_dict()
        result_dict["_project_dir"] = project_dir
        analyses[analysis_id] = result_dict

        # build blueprint
        bp = build_blueprint(result_dict, describer)
        blueprints[analysis_id] = bp.dict()

        analysis_status[analysis_id] = "done"
        logger.info(
            "Analysis %s complete: %d nodes, %d modules, %d risks",
            analysis_id, result.total_nodes, len(result.modules), len(result.risks),
        )
    except Exception as e:
        logger.exception("Analysis %s failed", analysis_id)
        analysis_status[analysis_id] = "error"
        analyses[analysis_id] = {"error": str(e)}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_project(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(400, "Please upload a .zip file")

    analysis_id = str(uuid.uuid4())[:8]
    tmp_dir = tempfile.mkdtemp(prefix=f"cartograph_{analysis_id}_")
    zip_path = os.path.join(tmp_dir, "project.zip")

    try:
        content = await file.read()
        with open(zip_path, "wb") as f:
            f.write(content)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)
        os.unlink(zip_path)

        project_dir = _find_project_root(tmp_dir)
        if not project_dir:
            raise HTTPException(400, "No TypeScript/React project found in zip")

    except zipfile.BadZipFile:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, "Invalid zip file")

    analysis_status[analysis_id] = "pending"
    background_tasks.add_task(_run_analysis, analysis_id, project_dir)

    return AnalyzeResponse(id=analysis_id, status="pending")


@app.post("/api/analyze-local")
async def analyze_local(payload: dict, background_tasks: BackgroundTasks):
    path = payload.get("path", "")
    if not os.path.isdir(path):
        raise HTTPException(400, f"Directory not found: {path}")
    analysis_id = str(uuid.uuid4())[:8]
    analysis_status[analysis_id] = "pending"
    background_tasks.add_task(_run_analysis, analysis_id, path)
    return {"id": analysis_id, "status": "pending"}


@app.get("/api/status/{analysis_id}")
async def get_status(analysis_id: str):
    status = analysis_status.get(analysis_id)
    if not status:
        raise HTTPException(404, "Analysis not found")
    return {"id": analysis_id, "status": status}


@app.get("/api/blueprint/{analysis_id}")
async def get_blueprint(analysis_id: str):
    """Get the blueprint cards — the main UI data."""
    status = analysis_status.get(analysis_id)
    if not status:
        raise HTTPException(404, "Analysis not found")
    if status != "done":
        return {"id": analysis_id, "status": status}

    bp = blueprints.get(analysis_id)
    if not bp:
        raise HTTPException(404, "Blueprint not yet built")

    return {"id": analysis_id, "status": "done", **bp}


@app.get("/api/projects/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Full raw analysis result (for debugging / advanced views)."""
    status = analysis_status.get(analysis_id)
    if not status:
        raise HTTPException(404, "Analysis not found")
    if status != "done":
        return {"id": analysis_id, "status": status}

    result = analyses.get(analysis_id, {})
    output = {k: v for k, v in result.items() if not k.startswith("_")}
    output["status"] = "done"
    output["id"] = analysis_id
    return output


@app.post("/api/impact")
async def predict_impact(req: ImpactRequest):
    result = analyses.get(req.project_id)
    if not result:
        raise HTTPException(404, "Project not found")
    project_dir = result.get("_project_dir")
    try:
        predictions = pipeline.predict_impact(project_dir, req.node_id, req.top_k)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {
        "modified_node_id": req.node_id,
        "affected_nodes": predictions,
    }


@app.post("/api/improve")
async def improve(req: ImproveRequest):
    """Generate a context-aware modification prompt."""
    result = analyses.get(req.project_id)
    if not result:
        raise HTTPException(404, "Project not found")

    bp = blueprints.get(req.project_id)
    if not bp:
        raise HTTPException(404, "Blueprint not built")

    target = next((c for c in bp["cards"] if c["module_id"] == req.module_id), None)
    if not target:
        raise HTTPException(404, f"Module {req.module_id} not found")

    # get GAT predictions for the most central node in this module
    project_dir = result.get("_project_dir")
    target_nodes = target.get("nodes", [])
    if not target_nodes:
        raise HTTPException(400, "Target module has no nodes")

    # pick the node with most connections (highest degree)
    nodes_full = result.get("graph_nodes", [])
    edges_full = result.get("graph_edges", [])
    degree: dict[str, int] = {}
    for e in edges_full:
        degree[e["source"]] = degree.get(e["source"], 0) + 1
        degree[e["target"]] = degree.get(e["target"], 0) + 1
    target_node_ids = {n["id"] for n in target_nodes}
    central = max(
        target_node_ids,
        key=lambda nid: degree.get(nid, 0),
        default=target_nodes[0]["id"],
    )

    try:
        predictions = pipeline.predict_impact(project_dir, central, top_k=20)
    except Exception as e:
        logger.warning("Impact prediction failed for improve: %s", e)
        predictions = []

    # tag each prediction with its module_id
    node_to_module = {n["id"]: n.get("moduleId", -1) for n in nodes_full}
    for p in predictions:
        p["module_id"] = node_to_module.get(p["node_id"], -1)

    suggestion = generate_improvement_prompt(
        user_intent=req.user_intent,
        target_module=target,
        affected_predictions=predictions,
        all_modules=bp["cards"],
    )

    return {
        "summary": suggestion.summary,
        "affected_modules": suggestion.affected_modules,
        "constraints": suggestion.constraints,
        "generated_prompt": suggestion.generated_prompt,
    }


def _find_project_root(base_dir: str) -> str | None:
    markers = ["package.json", "tsconfig.json", "next.config.js", "next.config.ts"]
    for marker in markers:
        if os.path.exists(os.path.join(base_dir, marker)):
            return base_dir
    for entry in os.listdir(base_dir):
        sub = os.path.join(base_dir, entry)
        if os.path.isdir(sub):
            for marker in markers:
                if os.path.exists(os.path.join(sub, marker)):
                    return sub
    return None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
