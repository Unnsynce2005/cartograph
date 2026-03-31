"""
Cartograph API server.

Endpoints:
  POST /api/analyze        — upload project zip, run full pipeline
  GET  /api/projects/{id}  — get analysis result
  POST /api/impact         — predict change impact for a node
  GET  /api/modules/{id}   — get module detail with semantic zoom
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cartograph.pipeline import AnalysisPipeline, AnalysisResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Cartograph",
    description="Code X-ray: GNN-powered architecture understanding for AI-generated codebases",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory store (swap for DB in production)
analyses: dict[str, dict] = {}
analysis_status: dict[str, str] = {}  # id -> "pending" | "running" | "done" | "error"
pipeline = AnalysisPipeline()


class AnalyzeResponse(BaseModel):
    id: str
    status: str


class ImpactRequest(BaseModel):
    project_id: str
    node_id: str
    top_k: int = 10


class ImpactResult(BaseModel):
    node_id: str
    name: str
    kind: str
    file_path: str
    impact_probability: float


class ImpactResponse(BaseModel):
    modified_node_id: str
    affected_nodes: list[ImpactResult]


class ModuleDetail(BaseModel):
    id: int
    name: str
    description: str
    node_count: int
    file_paths: list[str]
    nodes: list[dict]
    sub_modules: list[dict] | None = None


def _run_analysis(analysis_id: str, project_dir: str):
    """Background task: run the full analysis pipeline."""
    try:
        analysis_status[analysis_id] = "running"
        logger.info("Starting analysis %s on %s", analysis_id, project_dir)

        result = pipeline.analyze(project_dir)
        analyses[analysis_id] = result.to_dict()
        analyses[analysis_id]["_project_dir"] = project_dir
        analysis_status[analysis_id] = "done"

        logger.info("Analysis %s complete: %d nodes, %d modules",
                     analysis_id, result.total_nodes, len(result.modules))
    except Exception as e:
        logger.exception("Analysis %s failed", analysis_id)
        analysis_status[analysis_id] = "error"
        analyses[analysis_id] = {"error": str(e)}


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_project(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload a project zip and start analysis."""
    if not file.filename or not file.filename.endswith(".zip"):
        raise HTTPException(400, "Please upload a .zip file")

    analysis_id = str(uuid.uuid4())[:8]

    # extract zip to temp directory
    tmp_dir = tempfile.mkdtemp(prefix=f"cartograph_{analysis_id}_")
    zip_path = os.path.join(tmp_dir, "project.zip")

    try:
        content = await file.read()
        with open(zip_path, "wb") as f:
            f.write(content)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        os.unlink(zip_path)

        # find project root (first directory with package.json or tsconfig.json)
        project_dir = _find_project_root(tmp_dir)
        if not project_dir:
            raise HTTPException(400, "No TypeScript/React project found in zip")

    except zipfile.BadZipFile:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(400, "Invalid zip file")

    analysis_status[analysis_id] = "pending"
    background_tasks.add_task(_run_analysis, analysis_id, project_dir)

    return AnalyzeResponse(id=analysis_id, status="pending")


@app.get("/api/status/{analysis_id}")
async def get_status(analysis_id: str):
    """Check analysis status."""
    status = analysis_status.get(analysis_id)
    if not status:
        raise HTTPException(404, "Analysis not found")
    return {"id": analysis_id, "status": status}


@app.get("/api/projects/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get full analysis result."""
    status = analysis_status.get(analysis_id)
    if not status:
        raise HTTPException(404, "Analysis not found")
    if status == "pending" or status == "running":
        return {"id": analysis_id, "status": status}
    if status == "error":
        return {"id": analysis_id, "status": "error", "error": analyses.get(analysis_id, {}).get("error")}

    result = analyses.get(analysis_id)
    if not result:
        raise HTTPException(404, "Analysis result not found")

    # strip internal fields
    output = {k: v for k, v in result.items() if not k.startswith("_")}
    output["status"] = "done"
    output["id"] = analysis_id
    return output


@app.post("/api/impact", response_model=ImpactResponse)
async def predict_impact(req: ImpactRequest):
    """Predict change impact for a specific node."""
    result = analyses.get(req.project_id)
    if not result:
        raise HTTPException(404, "Project not found")

    project_dir = result.get("_project_dir")
    if not project_dir:
        raise HTTPException(500, "Project directory not available")

    try:
        predictions = pipeline.predict_impact(project_dir, req.node_id, req.top_k)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(500, str(e))

    return ImpactResponse(
        modified_node_id=req.node_id,
        affected_nodes=[ImpactResult(**p) for p in predictions],
    )


@app.get("/api/modules/{analysis_id}/{module_id}", response_model=ModuleDetail)
async def get_module_detail(analysis_id: str, module_id: int):
    """Get detailed info about a specific module (semantic zoom)."""
    result = analyses.get(analysis_id)
    if not result:
        raise HTTPException(404, "Analysis not found")

    modules = result.get("modules", [])
    module = next((m for m in modules if m["id"] == module_id), None)
    if not module:
        raise HTTPException(404, f"Module {module_id} not found")

    # get all nodes belonging to this module
    all_nodes = result.get("graph_nodes", [])
    module_nodes = [n for n in all_nodes if n.get("moduleId") == module_id]

    # group into sub-clusters by file
    file_groups: dict[str, list[dict]] = {}
    for node in module_nodes:
        fp = node.get("filePath", "unknown")
        file_groups.setdefault(fp, []).append(node)

    sub_modules = [
        {
            "file_path": fp,
            "node_count": len(nodes),
            "nodes": [{"id": n["id"], "name": n["name"], "kind": n["kind"]} for n in nodes],
        }
        for fp, nodes in file_groups.items()
    ]

    return ModuleDetail(
        id=module["id"],
        name=module["name"],
        description=module["description"],
        node_count=module["node_count"],
        file_paths=module["file_paths"],
        nodes=module_nodes,
        sub_modules=sub_modules,
    )


# demo endpoint: analyze from local path (dev only)
class AnalyzeLocalRequest(BaseModel):
    path: str


@app.post("/api/analyze-local")
async def analyze_local(req: AnalyzeLocalRequest, background_tasks: BackgroundTasks):
    """Analyze a local project directory (dev mode)."""
    if not os.path.isdir(req.path):
        raise HTTPException(400, f"Directory not found: {req.path}")

    analysis_id = str(uuid.uuid4())[:8]
    analysis_status[analysis_id] = "pending"
    background_tasks.add_task(_run_analysis, analysis_id, req.path)
    return {"id": analysis_id, "status": "pending"}


def _find_project_root(base_dir: str) -> str | None:
    """Find the actual project root inside an extracted zip."""
    markers = ["package.json", "tsconfig.json", "next.config.js", "next.config.ts"]

    # check base dir first
    for marker in markers:
        if os.path.exists(os.path.join(base_dir, marker)):
            return base_dir

    # check one level deep (zip often has a wrapper directory)
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
