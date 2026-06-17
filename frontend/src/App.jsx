import { useEffect, useState } from 'react'
import { fetchFilters, fetchSchedule } from './api'
import Chat from './Chat'
import Compare from './Compare'

export default function App() {
  const [tab, setTab] = useState('chat')
  const [races, setRaces] = useState([])       // full season schedule
  const [drivers, setDrivers] = useState([])    // driver codes for the compare dropdowns
  const [ingested, setIngested] = useState([])  // races that actually have chat data

  useEffect(() => {
    fetchSchedule().then((d) => setRaces(d.races)).catch(() => {})
    fetchFilters()
      .then((f) => { setDrivers(f.drivers || []); setIngested(f.grands_prix || []) })
      .catch(() => {})
  }, [])

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <span className="brand-mark">F1</span>
          <span className="brand-title">Virtual Paddock Engineer</span>
        </div>
        <nav className="tabs">
          <button className={tab === 'chat' ? 'active' : ''} onClick={() => setTab('chat')}>
            Chat
          </button>
          <button className={tab === 'compare' ? 'active' : ''} onClick={() => setTab('compare')}>
            Compare Telemetry
          </button>
        </nav>
      </header>

      {tab === 'chat'
        ? <Chat races={races} ingested={ingested} />
        : <Compare races={races} drivers={drivers} />}
    </div>
  )
}
