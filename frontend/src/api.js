// Thin client for the FastAPI backend.
export const API_BASE = 'http://localhost:8000'

export async function fetchFilters() {
  const res = await fetch(`${API_BASE}/api/filters`)
  if (!res.ok) throw new Error(`filters ${res.status}`)
  return res.json()
}

// Consume the SSE stream from POST /api/chat/stream.
// EventSource is GET-only, so we use fetch + a manual reader and buffer partial
// events (network chunks don't align to "\n\n" SSE boundaries).
export async function streamChat(body, { onSources, onToken, onDone }) {
  const res = await fetch(`${API_BASE}/api/chat/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok || !res.body) throw new Error(`chat ${res.status}`)

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    const events = buffer.split('\n\n')
    buffer = events.pop() ?? '' // carry the trailing partial event over

    for (const evt of events) {
      const line = evt.trim()
      if (!line.startsWith('data:')) continue
      const payload = JSON.parse(line.slice(5).trim())
      if (payload.type === 'sources') onSources?.(payload)
      else if (payload.type === 'token') onToken?.(payload.text)
      else if (payload.type === 'done') onDone?.()
    }
  }
}
