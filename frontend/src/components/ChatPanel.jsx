import { useEffect, useRef } from 'react'
import { Markdown, sourceLabel } from '../format'

const SUGGESTIONS = ['Who won this race?', 'Who was on the podium?', 'Worst tyre degradation?', 'How did Hamilton do?']

function BotMessage({ msg, active, onCite }) {
  const { text, sources = [], latency, driver, error, pending } = msg
  return (
    <div className={`bot-card${error ? ' error' : ''}`}>
      <span className="bot-label">// race engineer</span>
      <div className="bot-text">
        {error
          ? <p>{error}</p>
          : text
            ? <Markdown text={text} />
            : pending && <span className="typing" aria-label="Thinking"><i /><i /><i /></span>}
      </div>
      {sources.length > 0 && (
        <div className="cites">
          {sources.map((s, i) => (
            <button key={i} className={`cite${active === i ? ' active' : ''}`} onClick={() => onCite(i)}>
              {i + 1} · {sourceLabel(s).label}
            </button>
          ))}
        </div>
      )}
      {!pending && !error && (
        <div className="bot-metrics">
          {latency != null && <span>{latency} ms</span>}
          <span>{sources.length} sources</span>
          <span>{driver ? `driver: ${driver}` : 'all drivers'}</span>
        </div>
      )}
    </div>
  )
}

export default function ChatPanel({ raceName, messages, busy, input, setInput, onSend, activeMsg, activeReceipt, onCite }) {
  const endRef = useRef(null)
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [messages])

  const submit = (e) => { e.preventDefault(); onSend(input) }

  return (
    <section className="chat">
      <div className="messages">
        {messages.length === 0 && (
          <div className="hero">
            <span className="hero-kicker">I want to know</span>
            <h1>ask your<br />race engineer.</h1>
            <div className="hero-row">
              <p>Questions about stints and results for the {raceName || 'selected race'}. Name a driver and it filters automatically.</p>
              <span className="hero-note">// every answer<br />// shows its receipts</span>
            </div>
            <div className="suggestions">
              {SUGGESTIONS.map((q) => (
                <button key={q} className="suggestion" onClick={() => onSend(q)} disabled={busy}>{q}</button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className="msg">
            {m.role === 'user'
              ? <div className="user-bubble">{m.text}</div>
              : <BotMessage msg={m} active={activeMsg === i ? activeReceipt : null} onCite={(r) => onCite(i, r)} />}
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <form className="composer" onSubmit={submit}>
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask the race engineer…"
          aria-label="Ask the race engineer" disabled={busy} />
        <button type="submit" className="send" aria-label="Send" disabled={busy || !input.trim()}>↗</button>
      </form>
    </section>
  )
}
