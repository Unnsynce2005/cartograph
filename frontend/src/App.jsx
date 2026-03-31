import React, { useState, useEffect, useRef, useCallback } from 'react'
import * as d3 from 'd3'
import { Upload, AlertTriangle, Zap, ChevronRight, X, Loader2, Shield, GitBranch, Box, ArrowLeft } from 'lucide-react'
import * as api from './lib/api'

/* ── palette ── */
const MODULE_COLORS = [
  '#6366f1', '#f59e0b', '#10b981', '#ef4444',
  '#8b5cf6', '#06b6d4', '#f97316', '#ec4899',
  '#14b8a6', '#a855f7', '#84cc16', '#e11d48',
]
const KIND_SHAPE = {
  component: d3.symbolSquare,
  hook: d3.symbolDiamond,
  api_route: d3.symbolTriangle,
  page: d3.symbolStar,
  function: d3.symbolCircle,
  arrow_function: d3.symbolCircle,
  class: d3.symbolCross,
  variable: d3.symbolWye,
}
const SEVERITY_COLOR = { critical: '#dc2626', high: '#f59e0b', medium: '#f97316', low: '#6b7280' }

/* ── styles ── */
const STYLES = `
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'DM Sans', sans-serif;
    background: #0a0a0f;
    color: #e4e4e7;
    overflow: hidden;
  }
  .mono { font-family: 'JetBrains Mono', monospace; }
  .app { display: flex; height: 100vh; width: 100vw; }
  .sidebar {
    width: 340px;
    background: #111118;
    border-right: 1px solid #1e1e2a;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    flex-shrink: 0;
  }
  .sidebar-header {
    padding: 20px 20px 16px;
    border-bottom: 1px solid #1e1e2a;
  }
  .sidebar-header h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 18px;
    font-weight: 600;
    letter-spacing: -0.5px;
    color: #f4f4f5;
  }
  .sidebar-header .sub {
    font-size: 11px;
    color: #52525b;
    margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
  }
  .sidebar-body { flex: 1; overflow-y: auto; padding: 12px; }
  .main { flex: 1; position: relative; overflow: hidden; }
  .graph-canvas { width: 100%; height: 100%; background: #0a0a0f; }

  /* upload */
  .upload-overlay {
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    background: #0a0a0f;
    z-index: 50;
  }
  .drop-zone {
    width: 460px;
    border: 2px dashed #27272a;
    border-radius: 16px;
    padding: 48px 32px;
    text-align: center;
    cursor: pointer;
    transition: all 0.2s;
  }
  .drop-zone:hover, .drop-zone.drag-over {
    border-color: #6366f1;
    background: rgba(99, 102, 241, 0.04);
  }
  .drop-zone h2 { font-size: 20px; font-weight: 600; margin-bottom: 8px; }
  .drop-zone p { font-size: 13px; color: #71717a; line-height: 1.5; }
  .drop-zone .icon { margin-bottom: 16px; color: #3f3f46; }

  /* loading */
  .loading-overlay {
    position: absolute; inset: 0;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    background: #0a0a0f; z-index: 50;
  }
  .loading-overlay .spinner { animation: spin 1s linear infinite; color: #6366f1; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .loading-overlay p { margin-top: 16px; font-size: 14px; color: #a1a1aa; }

  /* module list */
  .module-item {
    padding: 10px 12px;
    border-radius: 8px;
    cursor: pointer;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 10px;
    transition: background 0.15s;
    font-size: 13px;
  }
  .module-item:hover { background: #18181f; }
  .module-item.active { background: #1e1e2e; }
  .module-dot {
    width: 10px; height: 10px;
    border-radius: 2px;
    flex-shrink: 0;
  }
  .module-count {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #52525b;
  }

  /* risk list */
  .risk-item {
    padding: 8px 12px;
    border-radius: 6px;
    margin-bottom: 4px;
    font-size: 12px;
    border-left: 3px solid;
    background: #111118;
    cursor: pointer;
  }
  .risk-item:hover { background: #18181f; }
  .risk-title { font-weight: 500; margin-bottom: 2px; }
  .risk-sev {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  /* detail panel */
  .detail-panel {
    position: absolute;
    top: 0; right: 0;
    width: 380px; height: 100%;
    background: #111118;
    border-left: 1px solid #1e1e2a;
    z-index: 40;
    display: flex;
    flex-direction: column;
    animation: slideIn 0.2s ease-out;
  }
  @keyframes slideIn { from { transform: translateX(100%); } }
  .detail-header {
    padding: 16px 20px;
    border-bottom: 1px solid #1e1e2a;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .detail-header h3 { font-size: 15px; font-weight: 600; flex: 1; }
  .detail-close {
    background: none; border: none; color: #71717a;
    cursor: pointer; padding: 4px;
  }
  .detail-close:hover { color: #e4e4e7; }
  .detail-body { flex: 1; overflow-y: auto; padding: 16px 20px; }
  .detail-section { margin-bottom: 20px; }
  .detail-section h4 {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #52525b;
    margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
  }
  .detail-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    padding: 4px 0;
  }
  .detail-row .label { color: #71717a; }
  .detail-row .value { font-family: 'JetBrains Mono', monospace; font-size: 12px; }

  /* impact results */
  .impact-node {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 10px;
    border-radius: 6px;
    margin-bottom: 4px;
    font-size: 12px;
    background: #0f0f17;
  }
  .impact-bar {
    height: 4px;
    border-radius: 2px;
    background: #6366f1;
    transition: width 0.3s;
  }
  .impact-prob {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    margin-left: auto;
    flex-shrink: 0;
  }

  /* breadcrumb */
  .breadcrumb {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 8px 12px;
    font-size: 12px;
    color: #71717a;
    border-bottom: 1px solid #1e1e2a;
    font-family: 'JetBrains Mono', monospace;
  }
  .breadcrumb span { cursor: pointer; }
  .breadcrumb span:hover { color: #e4e4e7; }
  .breadcrumb .current { color: #e4e4e7; cursor: default; }

  /* stats bar */
  .stats-bar {
    display: flex;
    gap: 16px;
    padding: 10px 20px;
    border-bottom: 1px solid #1e1e2a;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #52525b;
  }
  .stat-val { color: #a1a1aa; margin-left: 4px; }

  /* section header */
  .section-head {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #52525b;
    padding: 12px 0 8px;
    font-family: 'JetBrains Mono', monospace;
  }

  /* impact button */
  .btn-impact {
    display: flex; align-items: center; gap: 6px;
    padding: 7px 12px;
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 6px;
    color: #818cf8;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
    font-family: 'DM Sans', sans-serif;
    margin-top: 8px;
  }
  .btn-impact:hover {
    background: rgba(99, 102, 241, 0.2);
    border-color: #6366f1;
  }

  /* tooltip */
  .tooltip {
    position: absolute;
    pointer-events: none;
    background: #1e1e2e;
    border: 1px solid #2e2e3e;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    z-index: 100;
    max-width: 280px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  }
  .tooltip .tt-name { font-weight: 600; font-size: 13px; margin-bottom: 2px; }
  .tooltip .tt-kind { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #6366f1; }
  .tooltip .tt-file { font-size: 11px; color: #71717a; margin-top: 4px; }

  /* tab bar */
  .tab-bar {
    display: flex;
    border-bottom: 1px solid #1e1e2a;
  }
  .tab {
    flex: 1;
    padding: 10px;
    text-align: center;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    color: #71717a;
    transition: all 0.15s;
  }
  .tab:hover { color: #a1a1aa; }
  .tab.active { color: #e4e4e7; border-bottom-color: #6366f1; }
`

export default function App() {
  const [analysis, setAnalysis] = useState(null)
  const [analysisId, setAnalysisId] = useState(null)
  const [status, setStatus] = useState('idle') // idle | uploading | analyzing | done | error
  const [error, setError] = useState(null)
  const [selectedNode, setSelectedNode] = useState(null)
  const [selectedModule, setSelectedModule] = useState(null)
  const [impactResults, setImpactResults] = useState(null)
  const [impactLoading, setImpactLoading] = useState(false)
  const [tab, setTab] = useState('modules') // modules | risks
  const [tooltip, setTooltip] = useState(null)
  const [zoomPath, setZoomPath] = useState([]) // breadcrumb trail
  const [highlightedNodes, setHighlightedNodes] = useState(new Set())
  const svgRef = useRef(null)
  const simRef = useRef(null)

  // poll for analysis status
  useEffect(() => {
    if (!analysisId || status !== 'analyzing') return
    const interval = setInterval(async () => {
      try {
        const res = await api.getStatus(analysisId)
        if (res.status === 'done') {
          const full = await api.getAnalysis(analysisId)
          setAnalysis(full)
          setStatus('done')
        } else if (res.status === 'error') {
          setStatus('error')
          setError('Analysis failed')
        }
      } catch (e) {
        console.error(e)
      }
    }, 1000)
    return () => clearInterval(interval)
  }, [analysisId, status])

  // handle file upload
  const handleUpload = useCallback(async (file) => {
    setStatus('uploading')
    setError(null)
    try {
      const res = await api.uploadProject(file)
      setAnalysisId(res.id)
      setStatus('analyzing')
    } catch (e) {
      setStatus('error')
      setError(e.message)
    }
  }, [])

  // handle impact prediction
  const handlePredictImpact = useCallback(async (nodeId) => {
    if (!analysisId) return
    setImpactLoading(true)
    try {
      const res = await api.predictImpact(analysisId, nodeId)
      setImpactResults(res)
      setHighlightedNodes(new Set(res.affected_nodes.map(n => n.node_id)))
    } catch (e) {
      console.error('Impact prediction failed:', e)
    } finally {
      setImpactLoading(false)
    }
  }, [analysisId])

  // clear impact
  const clearImpact = useCallback(() => {
    setImpactResults(null)
    setHighlightedNodes(new Set())
  }, [])

  return (
    <>
      <style>{STYLES}</style>
      <div className="app">
        {/* left sidebar */}
        <div className="sidebar">
          <div className="sidebar-header">
            <h1>Cartograph</h1>
            <div className="sub">code x-ray engine</div>
          </div>

          {analysis && (
            <>
              <div className="stats-bar">
                <span>nodes<span className="stat-val">{analysis.total_nodes}</span></span>
                <span>edges<span className="stat-val">{analysis.total_edges}</span></span>
                <span>modules<span className="stat-val">{analysis.modules?.length || 0}</span></span>
                <span>risks<span className="stat-val">{analysis.risks?.length || 0}</span></span>
              </div>

              <div className="tab-bar">
                <div className={`tab ${tab === 'modules' ? 'active' : ''}`} onClick={() => setTab('modules')}>
                  <Box size={12} style={{ display: 'inline', marginRight: 4, verticalAlign: -1 }} />
                  Modules
                </div>
                <div className={`tab ${tab === 'risks' ? 'active' : ''}`} onClick={() => setTab('risks')}>
                  <Shield size={12} style={{ display: 'inline', marginRight: 4, verticalAlign: -1 }} />
                  Risks
                  {analysis.risks?.length > 0 && (
                    <span style={{ marginLeft: 4, color: '#f59e0b', fontSize: 10 }}>
                      {analysis.risks.length}
                    </span>
                  )}
                </div>
              </div>
            </>
          )}

          <div className="sidebar-body">
            {analysis && tab === 'modules' && (
              <>
                <div className="section-head"><Box size={12} /> discovered modules</div>
                {analysis.modules?.map((mod, i) => (
                  <div
                    key={mod.id}
                    className={`module-item ${selectedModule === mod.id ? 'active' : ''}`}
                    onClick={() => {
                      setSelectedModule(selectedModule === mod.id ? null : mod.id)
                      setSelectedNode(null)
                      clearImpact()
                    }}
                  >
                    <div className="module-dot" style={{ background: MODULE_COLORS[i % MODULE_COLORS.length] }} />
                    <span>{mod.name}</span>
                    <span className="module-count">{mod.node_count}</span>
                  </div>
                ))}
              </>
            )}

            {analysis && tab === 'risks' && (
              <>
                <div className="section-head"><AlertTriangle size={12} /> detected risks</div>
                {analysis.risks?.length === 0 && (
                  <p style={{ fontSize: 13, color: '#52525b', padding: '8px 0' }}>No risks detected.</p>
                )}
                {analysis.risks?.map((risk, i) => (
                  <div
                    key={i}
                    className="risk-item"
                    style={{ borderColor: SEVERITY_COLOR[risk.severity] || '#6b7280' }}
                    onClick={() => {
                      if (risk.affected_node_ids?.[0]) {
                        const node = analysis.graph_nodes?.find(n => n.id === risk.affected_node_ids[0])
                        if (node) setSelectedNode(node)
                      }
                    }}
                  >
                    <div className="risk-title">{risk.title}</div>
                    <div className="risk-sev" style={{ color: SEVERITY_COLOR[risk.severity] }}>
                      {risk.severity}
                    </div>
                  </div>
                ))}
              </>
            )}
          </div>
        </div>

        {/* main area */}
        <div className="main">
          {status === 'idle' && <UploadOverlay onUpload={handleUpload} />}
          {(status === 'uploading' || status === 'analyzing') && <LoadingOverlay status={status} />}
          {status === 'error' && <ErrorOverlay error={error} onRetry={() => setStatus('idle')} />}

          {status === 'done' && analysis && (
            <GraphView
              ref={svgRef}
              simRef={simRef}
              nodes={analysis.graph_nodes || []}
              edges={analysis.graph_edges || []}
              modules={analysis.modules || []}
              selectedModule={selectedModule}
              selectedNode={selectedNode}
              highlightedNodes={highlightedNodes}
              impactSource={impactResults?.modified_node_id}
              onSelectNode={(node) => {
                setSelectedNode(node)
                clearImpact()
              }}
              onTooltip={setTooltip}
            />
          )}

          {tooltip && (
            <div className="tooltip" style={{ left: tooltip.x + 16, top: tooltip.y - 10 }}>
              <div className="tt-name">{tooltip.name}</div>
              <div className="tt-kind">{tooltip.kind}</div>
              <div className="tt-file">{tooltip.filePath}</div>
            </div>
          )}

          {/* detail panel */}
          {selectedNode && (
            <div className="detail-panel">
              <div className="detail-header">
                <div style={{
                  width: 8, height: 8, borderRadius: 2,
                  background: MODULE_COLORS[selectedNode.moduleId % MODULE_COLORS.length] || '#6b7280'
                }} />
                <h3 className="mono">{selectedNode.name}</h3>
                <button className="detail-close" onClick={() => { setSelectedNode(null); clearImpact() }}>
                  <X size={16} />
                </button>
              </div>
              <div className="detail-body">
                <div className="detail-section">
                  <h4>identity</h4>
                  <div className="detail-row"><span className="label">Kind</span><span className="value">{selectedNode.kind}</span></div>
                  <div className="detail-row"><span className="label">File</span><span className="value">{selectedNode.filePath}</span></div>
                  <div className="detail-row"><span className="label">Lines</span><span className="value">{selectedNode.startLine}–{selectedNode.endLine}</span></div>
                  <div className="detail-row"><span className="label">Module</span><span className="value">
                    {analysis?.modules?.find(m => m.id === selectedNode.moduleId)?.name || 'Unknown'}
                  </span></div>
                </div>

                <div className="detail-section">
                  <h4>features</h4>
                  <div className="detail-row"><span className="label">LOC</span><span className="value">{selectedNode.features?.loc || 0}</span></div>
                  <div className="detail-row"><span className="label">Complexity</span><span className="value">{selectedNode.features?.cyclomaticComplexity || 0}</span></div>
                  <div className="detail-row"><span className="label">Params</span><span className="value">{selectedNode.features?.paramCount || 0}</span></div>
                  <div className="detail-row"><span className="label">Async</span><span className="value">{selectedNode.features?.hasAwait ? 'Yes' : 'No'}</span></div>
                  <div className="detail-row"><span className="label">JSX</span><span className="value">{selectedNode.features?.hasJSX ? 'Yes' : 'No'}</span></div>
                </div>

                {selectedNode.risks?.length > 0 && (
                  <div className="detail-section">
                    <h4>risks</h4>
                    {selectedNode.risks.map((r, i) => (
                      <div key={i} className="risk-item" style={{ borderColor: SEVERITY_COLOR[r.severity] }}>
                        <div className="risk-title">{r.title}</div>
                        <div className="risk-sev" style={{ color: SEVERITY_COLOR[r.severity] }}>{r.severity}</div>
                      </div>
                    ))}
                  </div>
                )}

                <div className="detail-section">
                  <h4>impact prediction</h4>
                  <button
                    className="btn-impact"
                    onClick={() => handlePredictImpact(selectedNode.id)}
                    disabled={impactLoading}
                  >
                    {impactLoading ? <Loader2 size={14} className="spinner" /> : <Zap size={14} />}
                    {impactLoading ? 'Predicting...' : 'What breaks if I change this?'}
                  </button>

                  {impactResults && impactResults.modified_node_id === selectedNode.id && (
                    <div style={{ marginTop: 12 }}>
                      {impactResults.affected_nodes.map((n, i) => (
                        <div key={i} className="impact-node">
                          <div style={{ flex: 1 }}>
                            <div style={{ fontWeight: 500, marginBottom: 2 }}>{n.name}</div>
                            <div style={{ fontSize: 10, color: '#6366f1' }} className="mono">{n.kind}</div>
                            <div className="impact-bar" style={{ width: `${n.impact_probability * 100}%`, marginTop: 4 }} />
                          </div>
                          <div className="impact-prob" style={{
                            color: n.impact_probability > 0.7 ? '#ef4444' :
                                   n.impact_probability > 0.4 ? '#f59e0b' : '#52525b'
                          }}>
                            {(n.impact_probability * 100).toFixed(0)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  )
}


/* ── Upload Overlay ── */
function UploadOverlay({ onUpload }) {
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef(null)

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) onUpload(file)
  }

  return (
    <div className="upload-overlay">
      <div
        className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
      >
        <div className="icon"><Upload size={40} /></div>
        <h2>Drop your project here</h2>
        <p>
          Upload a .zip of your Next.js / React / TypeScript project.
          <br />Cartograph will X-ray its architecture with graph neural networks.
        </p>
        <input
          ref={inputRef}
          type="file"
          accept=".zip"
          style={{ display: 'none' }}
          onChange={(e) => e.target.files[0] && onUpload(e.target.files[0])}
        />
      </div>
    </div>
  )
}


/* ── Loading Overlay ── */
function LoadingOverlay({ status }) {
  return (
    <div className="loading-overlay">
      <Loader2 size={36} className="spinner" />
      <p>{status === 'uploading' ? 'Uploading project...' : 'Analyzing codebase with GNN...'}</p>
      <p style={{ fontSize: 12, color: '#52525b', marginTop: 4 }}>
        {status === 'analyzing' && 'Parsing AST → building graph → running GraphSAGE → detecting risks'}
      </p>
    </div>
  )
}


/* ── Error Overlay ── */
function ErrorOverlay({ error, onRetry }) {
  return (
    <div className="upload-overlay">
      <div className="drop-zone" onClick={onRetry}>
        <div className="icon" style={{ color: '#ef4444' }}><AlertTriangle size={40} /></div>
        <h2>Analysis failed</h2>
        <p style={{ color: '#ef4444' }}>{error}</p>
        <p style={{ marginTop: 12 }}>Click to try again</p>
      </div>
    </div>
  )
}


/* ── Graph View (D3 force-directed) ── */
const GraphView = React.forwardRef(function GraphView(
  { nodes, edges, modules, selectedModule, selectedNode, highlightedNodes, impactSource, onSelectNode, onTooltip },
  svgRef
) {
  const containerRef = useRef(null)
  const simRef = useRef(null)

  useEffect(() => {
    if (!nodes.length || !containerRef.current) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = container.clientHeight

    // clear previous
    d3.select(container).selectAll('svg').remove()

    const svg = d3.select(container)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('class', 'graph-canvas')

    const g = svg.append('g')

    // zoom
    const zoom = d3.zoom()
      .scaleExtent([0.15, 5])
      .on('zoom', (e) => g.attr('transform', e.transform))
    svg.call(zoom)

    // build lookup
    const nodeMap = new Map(nodes.map(n => [n.id, { ...n }]))
    const simNodes = Array.from(nodeMap.values())
    const simEdges = edges
      .filter(e => nodeMap.has(e.source) && nodeMap.has(e.target))
      .map(e => ({ ...e }))

    // simulation
    const sim = d3.forceSimulation(simNodes)
      .force('link', d3.forceLink(simEdges).id(d => d.id).distance(80).strength(0.3))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(18))
      .force('x', d3.forceX(width / 2).strength(0.03))
      .force('y', d3.forceY(height / 2).strength(0.03))

    simRef.current = sim

    // module grouping force
    if (modules.length > 0) {
      const moduleCenter = {}
      const angle = (2 * Math.PI) / modules.length
      modules.forEach((m, i) => {
        moduleCenter[m.id] = {
          x: width / 2 + Math.cos(angle * i) * 200,
          y: height / 2 + Math.sin(angle * i) * 200,
        }
      })
      sim.force('module', d3.forceX(d => moduleCenter[d.moduleId]?.x || width / 2).strength(0.08))
      sim.force('moduleY', d3.forceY(d => moduleCenter[d.moduleId]?.y || height / 2).strength(0.08))
    }

    // edges
    const link = g.append('g')
      .selectAll('line')
      .data(simEdges)
      .join('line')
      .attr('stroke', '#1e1e2a')
      .attr('stroke-width', d => 0.5 + (d.weight || 1) * 0.8)
      .attr('stroke-opacity', 0.4)

    // nodes
    const node = g.append('g')
      .selectAll('g')
      .data(simNodes)
      .join('g')
      .attr('cursor', 'pointer')
      .call(d3.drag()
        .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y })
        .on('drag', (e, d) => { d.fx = e.x; d.fy = e.y })
        .on('end', (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null })
      )

    // shape per kind
    node.append('path')
      .attr('d', d => {
        const sym = d3.symbol().type(KIND_SHAPE[d.kind] || d3.symbolCircle).size(140)
        return sym()
      })
      .attr('fill', d => MODULE_COLORS[d.moduleId % MODULE_COLORS.length] || '#52525b')
      .attr('fill-opacity', 0.85)
      .attr('stroke', '#0a0a0f')
      .attr('stroke-width', 1.5)

    // label
    node.append('text')
      .text(d => d.name?.length > 18 ? d.name.slice(0, 16) + '…' : d.name)
      .attr('x', 10)
      .attr('y', 3)
      .attr('fill', '#71717a')
      .attr('font-size', '9px')
      .attr('font-family', "'JetBrains Mono', monospace")
      .attr('pointer-events', 'none')

    // interactions
    node.on('click', (e, d) => {
      e.stopPropagation()
      onSelectNode(d)
    })

    node.on('mouseenter', function (e, d) {
      d3.select(this).select('path')
        .transition().duration(100)
        .attr('d', d3.symbol().type(KIND_SHAPE[d.kind] || d3.symbolCircle).size(280)())

      link.attr('stroke-opacity', l =>
        l.source.id === d.id || l.target.id === d.id ? 0.8 : 0.06
      ).attr('stroke', l =>
        l.source.id === d.id || l.target.id === d.id ? '#6366f1' : '#1e1e2a'
      )

      onTooltip({ name: d.name, kind: d.kind, filePath: d.filePath, x: e.clientX, y: e.clientY })
    })

    node.on('mouseleave', function (e, d) {
      d3.select(this).select('path')
        .transition().duration(100)
        .attr('d', d3.symbol().type(KIND_SHAPE[d.kind] || d3.symbolCircle).size(140)())

      link.attr('stroke-opacity', 0.4).attr('stroke', '#1e1e2a')
      onTooltip(null)
    })

    svg.on('click', () => { onSelectNode(null); onTooltip(null) })

    // tick
    sim.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y)
      node.attr('transform', d => `translate(${d.x},${d.y})`)
    })

    return () => sim.stop()
  }, [nodes, edges, modules])

  // update visual state when selection/highlighting changes
  useEffect(() => {
    if (!containerRef.current) return
    const svg = d3.select(containerRef.current).select('svg')
    if (svg.empty()) return

    svg.selectAll('g g g').each(function (d) {
      const el = d3.select(this)
      const path = el.select('path')

      let opacity = 0.85
      let strokeColor = '#0a0a0f'
      let strokeWidth = 1.5

      // dim nodes not in selected module
      if (selectedModule !== null && d.moduleId !== selectedModule) {
        opacity = 0.15
      }

      // highlight impact nodes
      if (highlightedNodes.size > 0) {
        if (highlightedNodes.has(d.id)) {
          strokeColor = '#ef4444'
          strokeWidth = 3
          opacity = 1
        } else if (d.id === impactSource) {
          strokeColor = '#6366f1'
          strokeWidth = 3
          opacity = 1
        } else {
          opacity = 0.12
        }
      }

      // selected node
      if (selectedNode?.id === d.id) {
        strokeColor = '#ffffff'
        strokeWidth = 2.5
        opacity = 1
      }

      path.transition().duration(150)
        .attr('fill-opacity', opacity)
        .attr('stroke', strokeColor)
        .attr('stroke-width', strokeWidth)

      el.select('text').transition().duration(150)
        .attr('fill-opacity', opacity < 0.3 ? 0.15 : 1)
    })
  }, [selectedModule, selectedNode, highlightedNodes, impactSource])

  return <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
})
