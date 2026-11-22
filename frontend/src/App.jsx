import { useEffect, useMemo, useState } from 'react'
import { fetchFilters, fetchSchedule, streamChat } from './api'
import { applyMode, initialMode, switchMode } from './theme'
import Header from './components/Header'
import Road from './components/Road'
import ChatPanel from './components/ChatPanel'
import Receipts from './components/Receipts'

export default function App() {
  const [mode, setMode] = useState(initialMode)
  const [schedule, setSchedule] = useState([])   // full season: {round, name, location, date}
  const [ingested, setIngested] = useState(null) // race locations that have chat data
  const [race, setRace] = useState('')
  const [messages, setMessages] = useState([])   // user: {role, text}; bot: {role, text, sources, ...}
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [active, setActive] = useState({ msg: null, receipt: null }) // which answer's receipts are shown

  useEffect(() => { applyMode(mode) }, [mode])

  useEffect(() => {
    fetchSchedule().then((d) => setSchedule(d.races || [])).catch(() => setSchedule([]))
    fetchFilters().then((f) => setIngested(f.grands_prix || [])).catch(() => setIngested([]))
  }, [])

  // All rounds of the season, flagged by whether we have data for them. If the schedule
  // can't load, fall back to just the ingested races so the app still works.
  const races = useMemo(() => {
    const have = new Set(ingested || [])
    if (schedule.length) return schedule.map((r) => ({ ...r, hasData: have.has(r.location) }))
    return [...have].map((loc, i) => ({ round: i + 1, name: loc, location: loc, hasData: true }))
  }, [schedule, ingested])

  // Default to the most recent race that has data.
  useEffect(() => {
    if (!race) {
      const withData = races.filter((r) => r.hasData)
      if (withData.length) setRace(withData[withData.length - 1].location)
    }
  }, [races, race])

  const current = races.find((r) => r.location === race)

  const changeRace = (loc) => {
    if (loc === race) return
    setRace(loc)
    setMessages([])
    setActive({ msg: null, receipt: null })
  }

  const toggleTheme = (originEl) => {
    const next = mode === 'dark' ? 'light' : 'dark'
    switchMode(next, originEl)
    setMode(next)
  }

  async function send(text) {
    const q = text.trim()
    if (!q || busy) return
    const history = messages
      .filter((m) => m.text && !m.error && !m.pending)
      .slice(-6)
      .map((m) => ({ role: m.role === 'user' ? 'user' : 'assistant', content: m.text }))

    const botIndex = messages.length + 1
    setMessages((ms) => [...ms, { role: 'user', text: q }, { role: 'bot', text: '', sources: [], pending: true }])
    setInput('')
    setBusy(true)

    const update = (patch) => setMessages((ms) => ms.map((m, i) => (i === botIndex ? { ...m, ...patch } : m)))
    const started = performance.now()
    let acc = ''
    let latency = null

    try {
      await streamChat(
        { message: q, grand_prix: race || null, session_type: 'R', history },
        {
          onSources: (p) => {
            update({ sources: p.sources || [], driver: p.driver })
            setActive({ msg: botIndex, receipt: null })
          },
          onToken: (t) => {
            if (latency == null) latency = Math.round(performance.now() - started)
            acc += t
            update({ text: acc })
          },
        },
      )
      update({ pending: false, latency, text: acc || "That isn't in the data I have." })
    } catch (err) {
      update({
        pending: false,
        error: `Couldn't get an answer (${err.message}). The model may be over its usage limit — try again in a little while.`,
      })
    } finally {
      setBusy(false)
    }
  }

  const shownSources = active.msg != null ? messages[active.msg]?.sources || [] : []

  return (
    <div className="page">
      <div className="bg" aria-hidden="true">
        <div className="bg-dark" /><div className="bg-glow" /><div className="bg-light" />
        <div className="bg-lines" /><div className="bg-sheen" />
      </div>

      <Header dark={mode === 'dark'} onToggleTheme={toggleTheme} races={races} race={race} onRaceChange={changeRace} />
      <Road thinking={busy} />

      <main className="main">
        <ChatPanel
          raceName={current?.name}
          messages={messages}
          busy={busy}
          input={input}
          setInput={setInput}
          onSend={send}
          activeMsg={active.msg}
          activeReceipt={active.receipt}
          onCite={(msg, receipt) => setActive({ msg, receipt })}
        />
        <Receipts sources={shownSources} active={active.receipt} />
      </main>
    </div>
  )
}
