const API_BASE = '/api'

export async function uploadProject(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: form })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function analyzeLocal(path) {
  const res = await fetch(`${API_BASE}/analyze-local`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getStatus(id) {
  const res = await fetch(`${API_BASE}/status/${id}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getAnalysis(id) {
  const res = await fetch(`${API_BASE}/projects/${id}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getModuleDetail(analysisId, moduleId) {
  const res = await fetch(`${API_BASE}/modules/${analysisId}/${moduleId}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function predictImpact(projectId, nodeId, topK = 10) {
  const res = await fetch(`${API_BASE}/impact`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ project_id: projectId, node_id: nodeId, top_k: topK }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}
