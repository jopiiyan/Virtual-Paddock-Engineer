import { useEffect, useRef, useState } from 'react'
import { fetchFilters, streamChat } from './api'

export default function App() {
  const [filters, setFilters] = useState({ grands_prix: [], session_types: [], drivers: [] })
  const [gp, setGp] = useState('')
  const [session, setSession] = useState('')
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState([]) // {role, text, driver?, sources?}
  const [answer, setAnswer] = useState('')      // in-progress streamed answer
  const [sources, setSources] = useState([])    // receipts for the latest answer
  const [driver, setDriver] = useState(null)     // detected driver for latest answer
  const [streaming, setStreaming] = useState(false)
  const endRef = useRef(null)

  useEffect(() => {
    fetchFilters()
      .then((f) => {
        setFilters(f)
        setGp(f.grands_prix?.[0] ?? '')
        setSession(f.session_types?.[0] ?? '')
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, answer])

  async function send(e) {
    e.preventDefault()
    const q = input.trim()
    if (!q || streaming) return

    setMessages((m) => [...m, { role: 'user', text: q }])
    setInput('')
    setStreaming(true)
    setAnswer('')
    setSources([])
    setDriver(null)

    let acc = ''
    let srcs = []
    let drv = null
    try {
      await streamChat(
        { message: q, grand_prix: gp || null, session_type: session || null },
        {
          onSources: (p) => { srcs = p.sources; drv = p.driver; setSources(p.sources); setDriver(p.driver) },
          onToken: (t) => { acc += t; setAnswer(acc) },
        }
      )
    } catch (err) {
      acc = acc || `⚠️ ${err.message}. Is the backend running on :8000?`
    }
    setMessages((m) => [...m, { role: 'assistant', text: acc, driver: drv, sources: srcs }])
    setAnswer('')
    setStreaming(false)
  }

  return (
    <div className="app">
      <header className="header">
        <h1>🏎️ Virtual Paddock Engineer</h1>
        <div className="controls">
          <label>
            Grand Prix
            <select value={gp} onChange={(e) => setGp(e.target.value)}>
              {filters.grands_prix.map((g) => <option key={g} value={g}>{g}</option>)}
            </select>
          </label>
          <label>
            Session
            <select value={session} onChange={(e) => setSession(e.target.value)}>
              {filters.session_types.map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </label>
        </div>
      </header>

      <main className="layout">
        <section className="chat">
          <div className="messages">
            {messages.length === 0 && (
              <p className="hint">Ask about a stint — e.g. “How is Hamilton so fast?” or
                “Who had the worst tyre degradation?”. Name a driver and it filters automatically.</p>
            )}
            {messages.map((m, i) => (
              <div key={i} className={`msg ${m.role}`}>
                {m.role === 'assistant' && m.driver && <span className="chip">{m.driver}</span>}
                <div className="bubble">{m.text}</div>
              </div>
            ))}
            {streaming && (
              <div className="msg assistant">
                {driver && <span className="chip">{driver}</span>}
                <div className="bubble">{answer || <span className="typing">…</span>}</div>
              </div>
            )}
            <div ref={endRef} />
          </div>

          <form className="composer" onSubmit={send}>
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask the race engineer…"
              disabled={streaming}
            />
            <button type="submit" disabled={streaming || !input.trim()}>
              {streaming ? '…' : 'Send'}
            </button>
          </form>
        </section>

        <aside className="receipts">
          <h2>Receipts {sources.length > 0 && <span className="count">{sources.length}</span>}</h2>
          {sources.length === 0 && <p className="hint">Retrieved stint data appears here.</p>}
          {sources.map((s, i) => (
            <div key={i} className="receipt">
              <div className="receipt-meta">
                <span>{s.metadata.driver}</span>
                <span>Stint {s.metadata.stint}</span>
                <span>{s.metadata.compound}</span>
              </div>
              <p>{s.content}</p>
            </div>
          ))}
        </aside>
      </main>
    </div>
  )
}
