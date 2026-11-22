import { useEffect, useRef, useState } from 'react'
import { MiniCar } from './CarIcon'

const SunPath = () => (
  <><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>
)
const MoonPath = () => <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />

function Icon({ className, size, children }) {
  return (
    <svg className={className} viewBox="0 0 24 24" width={size} height={size} fill="none" stroke="currentColor"
      strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{children}</svg>
  )
}

function ThemeToggle({ dark, onToggle }) {
  const ref = useRef(null)
  return (
    <button ref={ref} className="toggle" role="switch" aria-checked={dark} aria-label="Dark mode"
      onClick={() => onToggle(ref.current)}>
      <Icon className="toggle-ghost sun" size={15}><SunPath /></Icon>
      <Icon className="toggle-ghost moon" size={15}><MoonPath /></Icon>
      <span className="toggle-knob">
        <Icon className="knob-icon sun" size={17}><SunPath /></Icon>
        <Icon className="knob-icon moon" size={17}><MoonPath /></Icon>
      </span>
    </button>
  )
}

const pad = (n) => String(n).padStart(2, '0')

// races: [{round, name, location, date, hasData}]
function RacePicker({ races, value, onChange }) {
  const [open, setOpen] = useState(false)
  const [focus, setFocus] = useState(-1)
  const btnRef = useRef(null)
  const current = races.find((r) => r.location === value)

  useEffect(() => {
    if (!open) return
    setFocus(Math.max(0, races.findIndex((r) => r.location === value)))
    const onKey = (e) => { if (e.key === 'Escape') { setOpen(false); btnRef.current?.focus() } }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open]) // eslint-disable-line react-hooks/exhaustive-deps

  const pick = (r) => {
    if (!r.hasData) return
    onChange(r.location)
    setOpen(false)
    btnRef.current?.focus()
  }

  const onListKey = (e) => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault()
      const step = e.key === 'ArrowDown' ? 1 : -1
      setFocus((f) => (f + step + races.length) % races.length)
    } else if (e.key === 'Enter' && races[focus]) {
      e.preventDefault()
      pick(races[focus])
    }
  }

  return (
    <div className="picker">
      <button ref={btnRef} className="picker-btn" aria-haspopup="listbox" aria-expanded={open}
        onClick={() => setOpen((o) => !o)} onKeyDown={(e) => open && onListKey(e)}>
        <span className="picker-text">
          <span className="picker-meta">
            {current ? `Round ${pad(current.round)} · ${current.location}` : 'Loading races'}
          </span>
          <span className="picker-name">{current?.name ?? '—'}</span>
        </span>
        <span className="picker-car"><MiniCar /></span>
      </button>
      {open && (
        <>
          <div className="picker-scrim" onClick={() => setOpen(false)} />
          <div className="picker-list" role="listbox" aria-label="Race weekend">
            <span className="picker-heading">// pick a race weekend</span>
            {races.map((r, i) => (
              <button key={r.location} role="option" aria-selected={r.location === value} disabled={!r.hasData}
                className={`race-opt${r.location === value ? ' selected' : ''}${i === focus ? ' focused' : ''}`}
                onClick={() => pick(r)} onMouseEnter={() => setFocus(i)}>
                <span className="race-round">{pad(r.round)}</span>
                <span className="race-body">
                  <span className="race-name">{r.name}</span>
                  <span className="race-sub">
                    {r.location}{r.date ? ` · ${r.date}` : ''}{r.hasData ? '' : ' · no data yet'}
                  </span>
                </span>
                <span className="race-dot" />
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

// Season selector. Only seasons with ingested data are clickable; the rest are
// blurred placeholders until their data is added.
const YEARS = [2023, 2024, 2025, 2026]
export const AVAILABLE_YEARS = [2025]

function YearSelect({ value }) {
  return (
    <div className="years" role="radiogroup" aria-label="Season">
      {YEARS.map((y) => {
        const available = AVAILABLE_YEARS.includes(y)
        return (
          <button key={y} type="button" role="radio" aria-checked={y === value} aria-disabled={!available}
            disabled={!available} className={`year${y === value ? ' selected' : ''}`}
            title={available ? `${y} season` : `${y} season — data coming soon`}>
            {y}
          </button>
        )
      })}
    </div>
  )
}

export default function Header({ dark, onToggleTheme, races, race, onRaceChange, year = 2025 }) {
  return (
    <header className="header">
      <div className="brand">
        <span className="wordmark">virtual paddock engineer*</span>
        <span className="tagline">formula 1 race data<br />with receipts</span>
      </div>
      <ThemeToggle dark={dark} onToggle={onToggleTheme} />
      <YearSelect value={year} />
      <RacePicker races={races} value={race} onChange={onRaceChange} />
    </header>
  )
}
