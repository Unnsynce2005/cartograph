# Cartograph

A static analysis engine for AI-generated codebases. Uses graph neural networks to automatically discover semantic modules, detect structural anti-patterns, and predict change impact in TypeScript/React/Next.js projects.

## What it does

You give it a project directory. It gives you back:

1. **Module map** — GraphSAGE clusters the code graph into 5–12 semantic modules. Not based on directory structure (vibe-coded projects often have none), but on learned patterns of which functions, components, and variables belong together based on their graph topology and semantic features.

2. **Risk flags** — Hybrid rule-based + graph-feature detection of: missing auth on API routes, hardcoded secrets, circular dependencies, excessive coupling, unhandled async errors, dead code, oversized modules.

3. **Impact prediction** — GAT predicts which nodes will need to co-change when you modify a given function or component. Trained on commit co-change data from open-source repos. Outputs per-node probability of being affected.

The frontend renders all three layers as an interactive force-directed graph with semantic zoom.

## Architecture

```
source code
    │
    ▼
┌──────────────┐
│  ts-morph     │  AST parse → extract functions, components, imports,
│  Parser       │  call sites, JSX renders, data flow
└──────┬───────┘
       │  JSON graph (nodes + edges)
       ▼
┌──────────────┐
│  SBERT        │  Encode function names + context → 384-dim vectors
│  Embedder     │
└──────┬───────┘
       │  node feature matrix (structural + semantic)
       ▼
┌──────────────┐
│  GraphSAGE    │  3-layer, supervised contrastive loss
│  Module       │  → node embeddings → hierarchical clustering
│  Discovery    │  → 5-12 semantic modules
└──────┬───────┘
       │
       ├──► Risk Detector (rule-based + graph features)
       │
       ▼
┌──────────────┐
│  GAT Impact   │  2-layer, 8 attention heads
│  Predictor    │  trained on git co-change data
│               │  → per-node affected probability
└──────┬───────┘
       │
       ▼
   FastAPI → React + D3.js
```

## Node features (29 + 384 dimensions)

**Structural (13):** LOC, cyclomatic complexity, param count, return count, callee count, caller count, file tree depth, has JSX, has try-catch, has await, import count, JSX child count, props count.

**Kind one-hot (12):** function, arrow_function, component, hook, class, variable, type_alias, interface, enum, api_route, page, middleware.

**Graph-derived (4):** in-degree, out-degree, PageRank, clustering coefficient.

**Semantic (384):** SBERT embedding of function name + file path context + source snippet.

## Edge types

| Type | Meaning |
|------|---------|
| `calls` | function A invokes function B |
| `imports` | file A imports from file B |
| `data_flow` | variable defined in A, read in B |
| `jsx_renders` | component A renders component B |
| `type_ref` | A references type defined in B |
| `prop_passes` | parent passes prop to child |
| `state_reads` | component reads shared state |
| `state_writes` | component writes shared state |
| `temporal` | co-changed in git history (training only) |

## Models

### GraphSAGE (Layer 1 — Module Discovery)

- 3 layers, hidden dim 128, output dim 64
- Neighborhood sampling: [25, 10, 5]
- Supervised contrastive loss (SupCon) on directory-based labels
- Hierarchical agglomerative clustering on node embeddings
- Optimal k selected by silhouette score
- Evaluation: NMI, ARI against ground truth directory structure

### GAT (Layer 3 — Impact Prediction)

- 2 layers, 8 attention heads, hidden dim 64
- Edge type embedding (9 types → per-head bias)
- Modification signal: one-hot concatenated to node features
- Training: binary CE on co-change pairs from git history
- Negative sampling: 3:1 ratio, excluding known co-change pairs
- Temporal edge weighting: exponential decay on older commits
- Evaluation: precision@5, precision@10, recall@10, MRR

### Risk Detector (Layer 2)

- Rule-based: regex + AST pattern matching for auth, secrets, error handling
- Graph-structural: cycle detection, degree centrality, data flow taint analysis
- No ML training required (deterministic)

## Training data

Module discovery labels: directory structure of well-organized open-source Next.js/React projects. `src/components/auth/Login.tsx` → module label "auth".

Impact prediction labels: git commit co-change pairs. If functions A and B are modified in the same commit, that's a positive training sample.

## Quick start

```bash
cp .env.example .env
docker compose up --build
```

Frontend: `http://localhost:5173`
API docs: `http://localhost:8000/docs`

Upload a .zip of any Next.js/React/TypeScript project. Analysis takes 10–60 seconds depending on project size.

## API

```
POST /api/analyze          Upload zip, returns { id, status }
GET  /api/status/{id}      Poll analysis status
GET  /api/projects/{id}    Full analysis result (modules, risks, graph)
POST /api/impact           { project_id, node_id } → affected nodes with probabilities
GET  /api/modules/{id}/{m} Module detail with sub-structure
POST /api/analyze-local    Dev mode: analyze a local directory path
```

## Stack

- **Parser:** TypeScript, ts-morph
- **ML:** Python, PyTorch, PyTorch Geometric, sentence-transformers, scikit-learn, NetworkX
- **Backend:** FastAPI, PostgreSQL, pgvector
- **Frontend:** React, D3.js, Vite
- **Infra:** Docker Compose

## Project structure

```
cartograph/
├── parser/                 # TypeScript AST parser
│   └── src/
│       ├── types.ts        # CodeNode, CodeEdge, CodeGraph types
│       ├── ast-extractor.ts # ts-morph parsing + edge extraction
│       ├── graph-builder.ts # assemble CodeGraph, output JSON
│       └── index.ts        # CLI entry point
├── engine/                 # ML engine
│   └── cartograph/
│       ├── config.py       # settings (model hyperparams, paths)
│       ├── data/
│       │   ├── graph.py    # NetworkX ↔ PyG conversion, feature matrix
│       │   ├── features.py # SBERT node embedding
│       │   └── dataset.py  # training data construction from GitHub
│       ├── models/
│       │   ├── graphsage.py # module discovery (GraphSAGE + SupCon + HAC)
│       │   ├── gat.py      # impact prediction (GAT + temporal edges)
│       │   └── risk.py     # risk detection (rules + graph features)
│       └── pipeline.py     # full analysis orchestrator
├── backend/
│   └── app/main.py         # FastAPI server
├── frontend/
│   └── src/
│       ├── App.jsx         # full UI: upload, graph viz, panels
│       └── lib/api.js      # API client
├── docker-compose.yml
├── Dockerfile.backend
└── .env.example
```

## Scope limitations

- **Next.js/React/TypeScript only.** The parser uses ts-morph which handles .ts/.tsx/.js/.jsx. Python, Java, Go etc. are not supported.
- **GraphSAGE and GAT need training data.** Without pretrained weights, module discovery still works (unsupervised clustering on raw features) but quality is lower. Impact prediction requires trained weights to be useful.
- **Projects up to ~500 files.** Larger projects may hit memory limits on the GNN forward pass.

## Design decisions

**Why GraphSAGE over GCN?** GraphSAGE uses neighborhood sampling, which means it generalizes to unseen graphs (user projects) without retraining on the full graph. GCN is transductive — it needs to see all nodes during training. Since every user project is a new graph, inductive learning is necessary.

**Why GAT for impact, not GraphSAGE?** The attention mechanism in GAT learns which edge types matter most for co-change prediction. In code graphs, a `calls` edge between two functions is more predictive of co-change than an `imports` edge. GAT discovers this weighting from data. GraphSAGE's mean/max aggregation treats all neighbors equally.

**Why supervised contrastive loss over cross-entropy?** SupCon pulls same-module nodes together in embedding space without forcing a fixed number of classes. This matters because the number of modules varies per project (3–12). Cross-entropy requires knowing k in advance; SupCon + post-hoc clustering does not.

**Why hierarchical clustering over k-means?** HAC produces a dendrogram that directly supports semantic zoom — different cut heights give different granularities. k-means gives a flat partition with no hierarchy.
