import React, { useState, useEffect, useCallback, useRef } from 'react'
import {
  Upload, AlertTriangle, ChevronRight, X, Loader2, Shield, FileCode,
  ArrowRight, Sparkles, Copy, Check, GitBranch, Box, AlertOctagon
} from 'lucide-react'
import * as api from './lib/api'

const SEVERITY = {
  critical: { color: '#dc2626', bg: 'rgba(220, 38, 38, 0.08)', border: '#7f1d1d', label: 'CRITICAL' },
  high:     { color: '#ea580c', bg: 'rgba(234, 88, 12, 0.08)',  border: '#7c2d12', label: 'HIGH' },
  medium:   { color: '#ca8a04', bg: 'rgba(202, 138, 4, 0.08)',  border: '#713f12', label: 'MEDIUM' },
  low:      { color: '#737373', bg: 'rgba(115, 115, 115, 0.06)', border: '#404040', label: 'LOW' },
}

const STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@300;400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }
  html, body, #root { height: 100%; }
  body {
    font-family: 'Inter', sans-serif;
    background: #0a0a0a;
    color: #e8e6e3;
    overflow-x: hidden;
    -webkit-font-smoothing: antialiased;
    background-image:
      radial-gradient(at 20% 0%, rgba(99, 102, 241, 0.06) 0%, transparent 50%),
      radial-gradient(at 80% 100%, rgba(245, 158, 11, 0.04) 0%, transparent 50%);
  }
  .serif { font-family: 'Instrument Serif', serif; font-style: italic; }
  .mono { font-family: 'JetBrains Mono', monospace; }

  /* page wrapper */
  .page { min-height: 100vh; max-width: 1200px; margin: 0 auto; padding: 64px 48px; }

  /* header */
  .header { margin-bottom: 64px; display: flex; justify-content: space-between; align-items: flex-end; }
  .brand .name {
    font-family: 'Instrument Serif', serif;
    font-size: 56px;
    line-height: 1;
    color: #f5f3f0;
    letter-spacing: -1px;
  }
  .brand .name .accent { font-style: italic; color: #fbbf24; }
  .brand .tagline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    color: #737373;
    margin-top: 8px;
  }
  .header-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #525252;
    text-align: right;
  }
  .header-meta .ver { color: #737373; margin-bottom: 4px; }

  /* upload */
  .upload-zone {
    margin-top: 80px;
    padding: 80px 48px;
    border: 1px dashed #262626;
    border-radius: 2px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    background: rgba(20, 20, 20, 0.3);
  }
  .upload-zone:hover, .upload-zone.drag {
    border-color: #fbbf24;
    background: rgba(251, 191, 36, 0.02);
  }
  .upload-zone .upload-icon {
    width: 48px; height: 48px;
    border: 1px solid #404040;
    border-radius: 2px;
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 24px;
    color: #737373;
    transition: all 0.3s;
  }
  .upload-zone:hover .upload-icon, .upload-zone.drag .upload-icon {
    border-color: #fbbf24; color: #fbbf24;
  }
  .upload-zone h2 {
    font-family: 'Instrument Serif', serif;
    font-size: 32px; font-weight: 400; color: #f5f3f0;
    margin-bottom: 12px;
  }
  .upload-zone p { font-size: 14px; color: #a3a3a3; line-height: 1.6; max-width: 420px; margin: 0 auto; }
  .upload-zone .hint {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; text-transform: uppercase; letter-spacing: 2px;
    color: #525252; margin-top: 16px;
  }

  /* loading */
  .loading {
    margin-top: 80px;
    padding: 64px;
    text-align: center;
    border: 1px solid #1f1f1f;
    border-radius: 2px;
  }
  .loading h2 {
    font-family: 'Instrument Serif', serif;
    font-size: 28px; color: #f5f3f0; margin-bottom: 12px;
  }
  .loading p { color: #a3a3a3; font-size: 14px; }
  .loading .pipeline {
    margin-top: 32px;
    display: flex; gap: 0; justify-content: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; text-transform: uppercase; letter-spacing: 2px;
    color: #525252;
  }
  .loading .pipeline span { padding: 0 12px; }
  .loading .pipeline span:not(:last-child)::after {
    content: '→'; margin-left: 24px; color: #404040;
  }
  .loading .pipeline span.active { color: #fbbf24; }
  .spin { animation: spin 1.2s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* blueprint header */
  .bp-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 48px;
    padding-bottom: 24px;
    border-bottom: 1px solid #1f1f1f;
  }
  .bp-title {
    font-family: 'Instrument Serif', serif;
    font-size: 48px; line-height: 1.1; color: #f5f3f0;
    letter-spacing: -1px;
  }
  .bp-overview { font-size: 15px; color: #a3a3a3; max-width: 520px; margin-top: 8px; line-height: 1.6; }
  .bp-stats { display: flex; gap: 32px; }
  .bp-stat { text-align: right; }
  .bp-stat .num { font-family: 'Instrument Serif', serif; font-size: 32px; color: #f5f3f0; line-height: 1; }
  .bp-stat .lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; text-transform: uppercase; letter-spacing: 2.5px;
    color: #525252; margin-top: 4px;
  }

  /* section header */
  .section-h {
    display: flex; align-items: baseline; gap: 16px; margin-bottom: 24px;
  }
  .section-h h3 {
    font-family: 'Instrument Serif', serif;
    font-size: 24px; color: #f5f3f0; font-weight: 400;
  }
  .section-h .count {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: #525252;
  }
  .section-h .line { flex: 1; height: 1px; background: #1f1f1f; }

  /* card grid */
  .card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
    gap: 16px;
    margin-bottom: 64px;
  }
  .card {
    background: #111111;
    border: 1px solid #1f1f1f;
    border-radius: 2px;
    padding: 28px;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
  }
  .card::before {
    content: '';
    position: absolute; top: 0; left: 0; bottom: 0;
    width: 3px;
    background: var(--accent);
    transition: width 0.3s ease;
  }
  .card:hover {
    border-color: #404040;
    background: #161616;
    transform: translateY(-2px);
  }
  .card:hover::before { width: 5px; }
  .card-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #525252;
    margin-bottom: 8px;
    text-transform: uppercase; letter-spacing: 2px;
  }
  .card-title {
    font-family: 'Instrument Serif', serif;
    font-size: 28px; color: #f5f3f0;
    line-height: 1.1; margin-bottom: 12px;
    font-weight: 400;
  }
  .card-summary {
    font-size: 14px; color: #d4d4d4;
    line-height: 1.6;
    margin-bottom: 20px;
    min-height: 44px;
  }
  .card-meta {
    display: flex; gap: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #737373;
    text-transform: uppercase; letter-spacing: 1.5px;
    padding-top: 16px;
    border-top: 1px solid #1f1f1f;
  }
  .card-meta .num { color: #f5f3f0; }
  .card-risk-pill {
    position: absolute; top: 20px; right: 20px;
    padding: 4px 8px;
    background: var(--risk-bg);
    border: 1px solid var(--risk-border);
    border-radius: 2px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px;
    color: var(--risk-color);
    letter-spacing: 1.5px;
    display: flex; align-items: center; gap: 4px;
  }

  /* detail */
  .detail-overlay {
    position: fixed; inset: 0;
    background: rgba(10, 10, 10, 0.85);
    backdrop-filter: blur(8px);
    z-index: 100;
    display: flex; justify-content: center; align-items: flex-start;
    padding: 64px 32px;
    overflow-y: auto;
    animation: fade-in 0.2s ease-out;
  }
  @keyframes fade-in { from { opacity: 0; } to { opacity: 1; } }
  .detail-card {
    background: #0f0f0f;
    border: 1px solid #2a2a2a;
    border-radius: 2px;
    width: 100%;
    max-width: 800px;
    padding: 48px;
    position: relative;
    animation: slide-up 0.3s cubic-bezier(0.16, 1, 0.3, 1);
  }
  @keyframes slide-up {
    from { transform: translateY(20px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
  }
  .detail-close {
    position: absolute; top: 24px; right: 24px;
    background: none; border: 1px solid #2a2a2a;
    color: #737373; padding: 6px; cursor: pointer;
    border-radius: 2px;
    transition: all 0.2s;
  }
  .detail-close:hover { border-color: #fbbf24; color: #fbbf24; }
  .detail-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--accent);
    text-transform: uppercase; letter-spacing: 2.5px;
  }
  .detail-title {
    font-family: 'Instrument Serif', serif;
    font-size: 56px; color: #f5f3f0;
    line-height: 1; margin: 8px 0 16px;
    font-weight: 400; letter-spacing: -1.5px;
  }
  .detail-summary { font-size: 17px; color: #d4d4d4; line-height: 1.6; margin-bottom: 32px; max-width: 600px; }
  .detail-section { margin-bottom: 32px; }
  .detail-section h4 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #525252;
    text-transform: uppercase; letter-spacing: 2.5px;
    margin-bottom: 12px;
  }
  .detail-section p { color: #a3a3a3; line-height: 1.7; font-size: 14px; }

  /* file list */
  .file-list {
    display: flex; flex-direction: column;
    border-top: 1px solid #1f1f1f;
  }
  .file-list .file {
    padding: 10px 0;
    border-bottom: 1px solid #1f1f1f;
    display: flex; align-items: center; gap: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: #a3a3a3;
  }
  .file-list .file svg { color: #525252; flex-shrink: 0; }

  /* connections */
  .connections-list { display: flex; flex-direction: column; gap: 8px; }
  .connection {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 16px;
    background: #161616;
    border: 1px solid #1f1f1f;
    border-radius: 2px;
    cursor: pointer;
    transition: all 0.2s;
  }
  .connection:hover { border-color: #404040; background: #1a1a1a; }
  .connection-name { font-size: 14px; color: #d4d4d4; display: flex; align-items: center; gap: 8px; }
  .connection-strength {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: #737373;
    text-transform: uppercase; letter-spacing: 1.5px;
  }

  /* risks */
  .risk-list { display: flex; flex-direction: column; gap: 8px; }
  .risk-item {
    padding: 14px 16px;
    background: var(--risk-bg);
    border: 1px solid var(--risk-border);
    border-radius: 2px;
    border-left: 3px solid var(--risk-color);
  }
  .risk-item .risk-head {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 6px;
  }
  .risk-item .risk-title { font-size: 13px; color: #e8e6e3; font-weight: 500; }
  .risk-item .risk-sev {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 1.5px; color: var(--risk-color);
  }
  .risk-item .risk-explain { font-size: 12px; color: #a3a3a3; line-height: 1.5; }

  /* improve form */
  .improve-form { margin-top: 16px; }
  .improve-form input {
    width: 100%;
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 2px;
    padding: 12px 16px;
    color: #f5f3f0;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    transition: border 0.2s;
  }
  .improve-form input:focus { outline: none; border-color: #fbbf24; }
  .improve-form input::placeholder { color: #525252; }
  .btn {
    display: flex; align-items: center; gap: 8px;
    padding: 10px 16px;
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 2px;
    color: #d4d4d4;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s;
    margin-top: 12px;
  }
  .btn:hover { border-color: #fbbf24; color: #fbbf24; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn.primary {
    background: #fbbf24;
    border-color: #fbbf24;
    color: #0a0a0a;
    font-weight: 500;
  }
  .btn.primary:hover { background: #f59e0b; border-color: #f59e0b; color: #0a0a0a; }

  /* prompt result */
  .prompt-result { margin-top: 24px; }
  .prompt-summary {
    padding: 16px;
    background: rgba(251, 191, 36, 0.05);
    border: 1px solid rgba(251, 191, 36, 0.2);
    border-radius: 2px;
    font-size: 14px; color: #fbbf24; line-height: 1.5;
    margin-bottom: 16px;
  }
  .prompt-cascade { margin-bottom: 16px; }
  .cascade-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 12px;
    border-bottom: 1px solid #1f1f1f;
    font-size: 13px;
  }
  .cascade-bar {
    height: 3px;
    background: var(--prob-color);
    border-radius: 1px;
    margin-top: 4px;
  }
  .cascade-name { color: #d4d4d4; flex: 1; }
  .cascade-prob {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--prob-color);
    margin-left: 12px;
  }
  .prompt-box {
    background: #060606;
    border: 1px solid #1f1f1f;
    border-radius: 2px;
    padding: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #d4d4d4;
    line-height: 1.7;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 400px;
    overflow-y: auto;
    position: relative;
  }
  .copy-btn {
    position: absolute; top: 12px; right: 12px;
    background: #1a1a1a; border: 1px solid #2a2a2a;
    padding: 6px 10px; border-radius: 2px;
    color: #a3a3a3; cursor: pointer;
    font-size: 11px; display: flex; align-items: center; gap: 6px;
    transition: all 0.2s;
  }
  .copy-btn:hover { border-color: #fbbf24; color: #fbbf24; }

  /* error */
  .err-box {
    padding: 32px; border: 1px solid #7f1d1d;
    background: rgba(220, 38, 38, 0.05);
    border-radius: 2px; margin-top: 80px; text-align: center;
  }

  /* scrollbar */
  ::-webkit-scrollbar { width: 8px; height: 8px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #2a2a2a; border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: #3a3a3a; }
`

const ACCENT_COLORS = [
  '#fbbf24', '#a3e635', '#34d399', '#22d3ee',
  '#a78bfa', '#f472b6', '#fb923c', '#facc15',
  '#4ade80', '#60a5fa', '#c084fc', '#f87171',
]

function colorForModule(id) {
  return ACCENT_COLORS[id % ACCENT_COLORS.length]
}

function topRiskSeverity(risks) {
  if (!risks || risks.length === 0) return null
  const order = ['critical', 'high', 'medium', 'low']
  for (const sev of order) {
    if (risks.find(r => r.severity === sev)) return sev
  }
  return null
}

export default function App() {
  const [status, setStatus] = useState('idle')
  const [analysisId, setAnalysisId] = useState(null)
  const [blueprint, setBlueprint] = useState(null)
  const [error, setError] = useState(null)
  const [selectedCard, setSelectedCard] = useState(null)
  const [pipelineStage, setPipelineStage] = useState(0)

  // poll for status
  useEffect(() => {
    if (!analysisId || status !== 'analyzing') return

    let stage = 0
    const stageInterval = setInterval(() => {
      stage = (stage + 1) % 5
      setPipelineStage(stage)
    }, 1500)

    const poll = setInterval(async () => {
      try {
        const s = await api.getStatus(analysisId)
        if (s.status === 'done') {
          clearInterval(poll)
          clearInterval(stageInterval)
          const bp = await fetch(`/api/blueprint/${analysisId}`).then(r => r.json())
          setBlueprint(bp)
          setStatus('done')
        } else if (s.status === 'error') {
          clearInterval(poll)
          clearInterval(stageInterval)
          setStatus('error')
          setError('Analysis failed')
        }
      } catch (e) { console.error(e) }
    }, 1000)

    return () => { clearInterval(poll); clearInterval(stageInterval) }
  }, [analysisId, status])

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

  const reset = () => {
    setStatus('idle')
    setAnalysisId(null)
    setBlueprint(null)
    setSelectedCard(null)
    setError(null)
  }

  return (
    <>
      <style>{STYLES}</style>
      <div className="page">
        <header className="header">
          <div className="brand">
            <div className="name">Carto<span className="accent">graph</span></div>
            <div className="tagline">— Code X-Ray Engine —</div>
          </div>
          <div className="header-meta">
            <div className="ver">v0.2 · gnn-powered</div>
            <div>
              {status === 'done' && blueprint
                ? `${blueprint.cards?.length || 0} modules · ${blueprint.total_risks} risks`
                : status === 'idle' ? 'awaiting input' : 'processing'}
            </div>
          </div>
        </header>

        {status === 'idle' && <UploadZone onUpload={handleUpload} />}
        {(status === 'uploading' || status === 'analyzing') && <Loading stage={pipelineStage} />}
        {status === 'error' && <ErrorBox error={error} onRetry={reset} />}
        {status === 'done' && blueprint && (
          <Blueprint
            data={blueprint}
            onSelectCard={setSelectedCard}
            onReset={reset}
          />
        )}

        {selectedCard && (
          <CardDetail
            card={selectedCard}
            analysisId={analysisId}
            allCards={blueprint?.cards || []}
            onClose={() => setSelectedCard(null)}
            onSelectOther={setSelectedCard}
          />
        )}
      </div>
    </>
  )
}


function UploadZone({ onUpload }) {
  const [drag, setDrag] = useState(false)
  const inputRef = useRef()

  const handleDrop = (e) => {
    e.preventDefault()
    setDrag(false)
    const file = e.dataTransfer.files[0]
    if (file) onUpload(file)
  }

  return (
    <div
      className={`upload-zone ${drag ? 'drag' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
      onDragLeave={() => setDrag(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <div className="upload-icon"><Upload size={24} strokeWidth={1.5} /></div>
      <h2>Drop your project</h2>
      <p>
        Upload a <span className="mono" style={{ color: '#fbbf24' }}>.zip</span> of any
        Next.js, React, or TypeScript project. Cartograph will read it the way an
        architect reads a blueprint — finding the rooms, the wiring, the load-bearing walls.
      </p>
      <div className="hint">— supports projects up to 500 files —</div>
      <input
        ref={inputRef}
        type="file"
        accept=".zip"
        style={{ display: 'none' }}
        onChange={(e) => e.target.files[0] && onUpload(e.target.files[0])}
      />
    </div>
  )
}

function Loading({ stage }) {
  const stages = ['parse', 'embed', 'cluster', 'detect', 'describe']
  return (
    <div className="loading">
      <Loader2 size={32} className="spin" style={{ color: '#fbbf24', marginBottom: 24 }} strokeWidth={1.5} />
      <h2>Reading your codebase</h2>
      <p>Building the graph, finding the modules, flagging the risks.</p>
      <div className="pipeline">
        {stages.map((s, i) => (
          <span key={s} className={i === stage ? 'active' : ''}>{s}</span>
        ))}
      </div>
    </div>
  )
}

function ErrorBox({ error, onRetry }) {
  return (
    <div className="err-box">
      <AlertOctagon size={32} style={{ color: '#dc2626', marginBottom: 16 }} />
      <h2 className="serif" style={{ fontSize: 28, marginBottom: 12 }}>Analysis failed</h2>
      <p style={{ color: '#a3a3a3', marginBottom: 24 }}>{error}</p>
      <button className="btn" onClick={onRetry}>Try again</button>
    </div>
  )
}

function Blueprint({ data, onSelectCard, onReset }) {
  return (
    <>
      <div className="bp-header">
        <div>
          <div className="bp-title">{data.project_name}</div>
          <div className="bp-overview">{data.overview}</div>
        </div>
        <div className="bp-stats">
          <div className="bp-stat">
            <div className="num">{data.cards?.length || 0}</div>
            <div className="lbl">modules</div>
          </div>
          <div className="bp-stat">
            <div className="num" style={{ color: data.total_risks > 0 ? '#fbbf24' : '#f5f3f0' }}>
              {data.total_risks}
            </div>
            <div className="lbl">risks</div>
          </div>
          <div className="bp-stat">
            <button className="btn" onClick={onReset} style={{ marginTop: 8 }}>
              <X size={14} strokeWidth={1.5} /> new project
            </button>
          </div>
        </div>
      </div>

      <div className="section-h">
        <h3>The Modules</h3>
        <span className="count">— {data.cards?.length || 0} discovered by graphsage —</span>
        <div className="line"></div>
      </div>

      <div className="card-grid">
        {data.cards?.map((card, i) => {
          const accent = colorForModule(card.module_id)
          const sev = topRiskSeverity(card.risks)
          const sevConfig = sev ? SEVERITY[sev] : null

          return (
            <div
              key={card.module_id}
              className="card"
              style={{
                '--accent': accent,
                '--risk-color': sevConfig?.color,
                '--risk-bg': sevConfig?.bg,
                '--risk-border': sevConfig?.border,
              }}
              onClick={() => onSelectCard(card)}
            >
              {sev && (
                <div className="card-risk-pill">
                  <AlertTriangle size={9} strokeWidth={2} />
                  {sevConfig.label}
                </div>
              )}
              <div className="card-num" style={{ color: accent }}>
                №{String(card.module_id + 1).padStart(2, '0')}
              </div>
              <div className="card-title">{card.name}</div>
              <div className="card-summary">{card.summary || 'A coherent group of code entities discovered by graph topology.'}</div>
              <div className="card-meta">
                <span><span className="num">{card.node_count}</span> entities</span>
                <span><span className="num">{card.file_count}</span> files</span>
                {card.connects_to?.length > 0 && (
                  <span><span className="num">{card.connects_to.length}</span> connections</span>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </>
  )
}


function CardDetail({ card, analysisId, allCards, onClose, onSelectOther }) {
  const [intent, setIntent] = useState('')
  const [improving, setImproving] = useState(false)
  const [improvement, setImprovement] = useState(null)
  const [copied, setCopied] = useState(false)
  const accent = colorForModule(card.module_id)

  const handleImprove = async () => {
    if (!intent.trim()) return
    setImproving(true)
    setImprovement(null)
    try {
      const res = await fetch('/api/improve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          project_id: analysisId,
          module_id: card.module_id,
          user_intent: intent,
        }),
      })
      setImprovement(await res.json())
    } catch (e) { console.error(e) }
    setImproving(false)
  }

  const handleCopy = () => {
    if (improvement?.generated_prompt) {
      navigator.clipboard.writeText(improvement.generated_prompt)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  return (
    <div className="detail-overlay" onClick={(e) => e.target.classList.contains('detail-overlay') && onClose()}>
      <div className="detail-card" style={{ '--accent': accent }}>
        <button className="detail-close" onClick={onClose}><X size={16} strokeWidth={1.5} /></button>

        <div className="detail-num">№{String(card.module_id + 1).padStart(2, '0')} · module</div>
        <h2 className="detail-title">{card.name}</h2>
        <p className="detail-summary">{card.summary}</p>

        {card.responsibility && (
          <div className="detail-section">
            <h4>What it does</h4>
            <p>{card.responsibility}</p>
          </div>
        )}

        {card.connects_to?.length > 0 && (
          <div className="detail-section">
            <h4>Connections to other modules</h4>
            <div className="connections-list">
              {card.connects_to.map(conn => {
                const other = allCards.find(c => c.module_id === conn.module_id)
                return (
                  <div
                    key={conn.module_id}
                    className="connection"
                    onClick={() => other && onSelectOther(other)}
                  >
                    <span className="connection-name">
                      <ArrowRight size={14} strokeWidth={1.5} style={{ color: colorForModule(conn.module_id) }} />
                      {conn.name}
                    </span>
                    <span className="connection-strength">{conn.strength} edges</span>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {card.risks?.length > 0 && (
          <div className="detail-section">
            <h4>Issues found ({card.risks.length})</h4>
            <div className="risk-list">
              {card.risks.slice(0, 5).map((risk, i) => {
                const sev = SEVERITY[risk.severity] || SEVERITY.low
                return (
                  <div
                    key={i}
                    className="risk-item"
                    style={{
                      '--risk-color': sev.color,
                      '--risk-bg': sev.bg,
                      '--risk-border': sev.border,
                    }}
                  >
                    <div className="risk-head">
                      <div className="risk-title">{risk.title}</div>
                      <div className="risk-sev">{sev.label}</div>
                    </div>
                    <div className="risk-explain">{risk.explanation}</div>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        <div className="detail-section">
          <h4>Files in this module</h4>
          <div className="file-list">
            {card.files?.slice(0, 12).map((f, i) => (
              <div key={i} className="file">
                <FileCode size={12} strokeWidth={1.5} />
                {f}
              </div>
            ))}
          </div>
        </div>

        <div className="detail-section">
          <h4>What if I want to change this?</h4>
          <p style={{ marginBottom: 16 }}>
            Describe a change in plain English. Cartograph will use its impact prediction model to figure out
            what else might break, then generate a precise prompt you can paste into Cursor or Claude.
          </p>
          <div className="improve-form">
            <input
              type="text"
              placeholder='e.g., "add a search bar" or "let users filter by category"'
              value={intent}
              onChange={(e) => setIntent(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleImprove()}
            />
            <button className="btn primary" onClick={handleImprove} disabled={improving || !intent.trim()}>
              {improving ? <Loader2 size={14} className="spin" /> : <Sparkles size={14} strokeWidth={1.5} />}
              {improving ? 'Analyzing...' : 'Generate modification plan'}
            </button>
          </div>

          {improvement && (
            <div className="prompt-result">
              <div className="prompt-summary">{improvement.summary}</div>

              {improvement.affected_modules?.length > 0 && (
                <div className="prompt-cascade">
                  <h4 style={{ marginBottom: 8 }}>predicted cascade (gat impact)</h4>
                  {improvement.affected_modules.slice(0, 5).map(m => {
                    const probColor = m.probability >= 0.7 ? '#dc2626'
                      : m.probability >= 0.4 ? '#fbbf24' : '#737373'
                    return (
                      <div key={m.module_id} className="cascade-item" style={{ flexDirection: 'column', alignItems: 'stretch' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span className="cascade-name">{m.name}</span>
                          <span className="cascade-prob" style={{ '--prob-color': probColor, color: probColor }}>
                            {(m.probability * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="cascade-bar" style={{
                          '--prob-color': probColor,
                          background: probColor,
                          width: `${m.probability * 100}%`,
                        }} />
                      </div>
                    )
                  })}
                </div>
              )}

              <div className="prompt-box">
                <button className="copy-btn" onClick={handleCopy}>
                  {copied ? <Check size={11} strokeWidth={2} /> : <Copy size={11} strokeWidth={1.5} />}
                  {copied ? 'copied' : 'copy'}
                </button>
                {improvement.generated_prompt}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
