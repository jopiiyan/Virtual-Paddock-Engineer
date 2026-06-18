import { useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine, ResponsiveContainer,
} from 'recharts'
import { fetchTelemetry } from './api'
import Dropdown from './Dropdown'

const COLORS = ['#e10600', '#00d2be'] // driver A (red), driver B (teal)

// Merge two drivers' telemetry into Recharts rows, aligned by sample index
// (each driver is resampled to the same number of points across their lap).
function mergeChannel(drivers, channel) {
  if (!drivers.length) return []
  const n = Math.min(...drivers.map((d) => d[channel].length))
  const rows = []
  for (let i = 0; i < n; i++) {
    const row = { distance: drivers[0].distance[i] }
    drivers.forEach((d) => { row[d.driver] = d[channel][i] })
    rows.push(row)
  }
  return rows
}

function Trace({ title, drivers, channel, domain, step }) {
  const rows = mergeChannel(drivers, channel)
  return (
    <div className="chart-card">
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={rows} margin={{ top: 5, right: 12, left: -12, bottom: 0 }}>
          <CartesianGrid stroke="#2c2f3a" strokeDasharray="3 3" />
          <XAxis dataKey="distance" type="number" domain={['dataMin', 'dataMax']}
            tick={{ fontSize: 11, fill: '#9aa0ac' }} tickFormatter={(v) => Math.round(v)}
            unit="m" />
          <YAxis domain={domain} tick={{ fontSize: 11, fill: '#9aa0ac' }} width={42} allowDecimals={false} />
          <Tooltip contentStyle={{ background: '#1a1c22', border: '1px solid #2c2f3a', borderRadius: 8 }}
            labelStyle={{ color: '#9aa0ac' }} labelFormatter={(v) => `${Math.round(v)} m`} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {drivers.map((d, i) => (
            <Line key={d.driver} type={step ? 'stepAfter' : 'monotone'} dataKey={d.driver}
              stroke={COLORS[i]} dot={false} strokeWidth={2} isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

// Speed difference between the two drivers along the lap: A.speed − B.speed.
// Above 0 = Driver A faster at that point; below 0 = Driver B faster.
function SpeedDelta({ drivers }) {
  if (drivers.length < 2) {
    return (
      <div className="chart-card">
        <h3>Speed delta</h3>
        <p className="hint">Pick two different drivers to see the speed delta.</p>
      </div>
    )
  }
  const [A, B] = drivers
  const n = Math.min(A.speed.length, B.speed.length)
  const rows = []
  for (let i = 0; i < n; i++) {
    rows.push({ distance: A.distance[i], delta: +(A.speed[i] - B.speed[i]).toFixed(1) })
  }
  return (
    <div className="chart-card">
      <h3>Speed delta — {A.driver} vs {B.driver} (km/h)</h3>
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={rows} margin={{ top: 5, right: 12, left: -12, bottom: 0 }}>
          <CartesianGrid stroke="#2c2f3a" strokeDasharray="3 3" />
          <XAxis dataKey="distance" type="number" domain={['dataMin', 'dataMax']}
            tick={{ fontSize: 11, fill: '#9aa0ac' }} tickFormatter={(v) => Math.round(v)} unit="m" />
          <YAxis tick={{ fontSize: 11, fill: '#9aa0ac' }} width={42} />
          <Tooltip contentStyle={{ background: '#1a1c22', border: '1px solid #2c2f3a', borderRadius: 8 }}
            labelStyle={{ color: '#9aa0ac' }} labelFormatter={(v) => `${Math.round(v)} m`}
            formatter={(v) => [`${v} km/h`, `${A.driver} − ${B.driver}`]} />
          <ReferenceLine y={0} stroke="#9aa0ac" strokeDasharray="4 4" />
          <Line type="monotone" dataKey="delta" stroke="#ffd166" dot={false} strokeWidth={2} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
      <p className="caption">
        Above 0: <span style={{ color: COLORS[0] }}>{A.driver}</span> faster ·
        Below 0: <span style={{ color: COLORS[1] }}>{B.driver}</span> faster
      </p>
    </div>
  )
}

function TrackMap({ drivers }) {
  const pts = drivers.flatMap((d) => d.x.map((x, i) => [x, d.y[i]]))
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1])
  const minX = Math.min(...xs), maxX = Math.max(...xs)
  const minY = Math.min(...ys), maxY = Math.max(...ys)
  const W = 560, H = 320, pad = 16
  const s = Math.min((W - 2 * pad) / (maxX - minX), (H - 2 * pad) / (maxY - minY))
  const tx = (x) => pad + (x - minX) * s
  const ty = (y) => H - (pad + (y - minY) * s) // invert Y so it isn't upside down

  return (
    <div className="chart-card">
      <h3>Racing line (track map)</h3>
      <svg viewBox={`0 0 ${W} ${H}`} className="trackmap">
        {drivers.map((d, idx) => {
          const path = d.x.map((x, i) => `${i === 0 ? 'M' : 'L'} ${tx(x).toFixed(1)} ${ty(d.y[i]).toFixed(1)}`).join(' ')
          return <path key={d.driver} d={path} fill="none" stroke={COLORS[idx]} strokeWidth={2.5} opacity={0.85} />
        })}
      </svg>
    </div>
  )
}

export default function Compare({ races, drivers }) {
  const [race, setRace] = useState('')
  const [a, setA] = useState('')
  const [b, setB] = useState('')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!race && races.length) setRace(races.find((r) => r.location === 'Silverstone')?.location || races[0].location)
  }, [races]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (drivers.length && !a && !b) { setA(drivers[0]); setB(drivers[1] || drivers[0]) }
  }, [drivers]) // eslint-disable-line react-hooks/exhaustive-deps

  async function run() {
    if (!race || !a) return
    setLoading(true); setError(''); setData(null)
    const picks = a === b ? [a] : [a, b]
    try {
      const res = await fetchTelemetry({ drivers: picks, grand_prix: race, session_type: 'R' })
      setData(res)
      if (Object.keys(res.errors || {}).length) {
        setError(Object.entries(res.errors).map(([d, m]) => `${d}: ${m}`).join('  •  '))
      }
    } catch (err) {
      setError(`${err.message}. First load of a race downloads telemetry and can take a while.`)
    }
    setLoading(false)
  }

  const tels = data?.drivers || []

  return (
    <main className="compare">
      <div className="controls compare-controls">
        <label>
          Grand Prix
          <Dropdown className="wide" ariaLabel="Grand Prix" value={race} onChange={setRace}
            options={races.map((r) => ({ value: r.location, label: r.name }))} />
        </label>
        <label>
          Driver A
          <Dropdown className="narrow" ariaLabel="Driver A" value={a} onChange={setA}
            options={drivers.map((d) => ({ value: d, label: d }))} />
        </label>
        <label>
          Driver B
          <Dropdown className="narrow" ariaLabel="Driver B" value={b} onChange={setB}
            options={drivers.map((d) => ({ value: d, label: d }))} />
        </label>
        <button className="primary" onClick={run} disabled={loading}>
          {loading ? 'Loading telemetry…' : 'Compare'}
        </button>
      </div>

      {error && <p className="warn">{error}</p>}

      {tels.length > 0 && (
        <>
          <div className="legend-row">
            {tels.map((d, i) => (
              <span key={d.driver} className="legend-item">
                <span className="swatch" style={{ background: COLORS[i] }} />
                <strong>{d.driver}</strong> — fastest lap {d.lap_time} (lap {d.lap_number}, {d.compound})
              </span>
            ))}
          </div>
          <div className="charts-grid">
            <Trace title="Speed (km/h)" drivers={tels} channel="speed" domain={['auto', 'auto']} />
            <Trace title="Throttle (%)" drivers={tels} channel="throttle" domain={[0, 100]} />
            <SpeedDelta drivers={tels} />
            <TrackMap drivers={tels} />
          </div>
        </>
      )}

      {!tels.length && !loading && !error && (
        <p className="hint">Pick a race and two drivers, then “Compare”. Telemetry is each driver’s
          fastest lap, loaded live from FastF1 — works for any of the 24 races.</p>
      )}
    </main>
  )
}
