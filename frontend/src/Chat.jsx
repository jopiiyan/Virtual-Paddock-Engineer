import { useEffect, useRef, useState } from 'react'
import { streamChat } from './api'

export default function Chat({ races, ingested }) {
  const [gp, setGp] = useState('')
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState([]) // {role, text, driver?, sources?}
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState([])
  const [driver, setDriver] = useState(null)
  const [streaming, setStreaming] = useState(false)
  const endRef = useRef(null)

  // Default to an ingested race so chat works out of the box.
  useEffect(() => {
    if (!gp && (ingested.length || races.length)) {
      setGp(ingested[0] || races[0]?.location || '')
    }
  }, [ingested, races]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, answer])

  const isIngested = ingested.includes(gp)

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

    let acc = '', srcs = [], drv = null
    try {
      await streamChat(
        { message: q, grand_prix: gp || null, session_type: 'R' },
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
    <>
      <div className="controls">
        <label>
          Grand Prix
          <select value={gp} onChange={(e) => setGp(e.target.value)}>
            {races.map((r) => (
              <option key={r.round} value={r.location}>{r.name}</option>
            ))}
          </select>
        </label>
        {!isIngested && gp && (
          <span className="warn">No ingested data for this race — chat will say so.</span>
        )}
      </div>

      <main className="layout">
        <section className="chat">
          <div className="messages">
            {messages.length === 0 && (
              <p className="hint">Ask about a stint or result — e.g. “How is Hamilton so fast?”,
                “Who won this race?”, “Worst tyre degradation?”. Name a driver and it filters automatically.</p>
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
          {sources.length === 0 && <p className="hint">Retrieved stint / result data appears here.</p>}
          {sources.map((s, i) => (
            <div key={i} className="receipt">
              <div className="receipt-meta">
                <span>{s.metadata.driver}</span>
                {s.metadata.doc_type === 'result'
                  ? <span>Result</span>
                  : <>{s.metadata.stint != null && <span>Stint {s.metadata.stint}</span>}
                    {s.metadata.compound && <span>{s.metadata.compound}</span>}</>}
              </div>
              <p>{s.content}</p>
            </div>
          ))}
        </aside>
      </main>
    </>
  )
}
