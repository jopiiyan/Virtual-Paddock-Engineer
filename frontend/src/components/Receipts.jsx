import { useEffect, useRef } from 'react'
import { sourceLabel } from '../format'

// The retrieved documents behind the current answer, shown as-is.
export default function Receipts({ sources, active }) {
  const refs = useRef([])

  useEffect(() => {
    if (active != null) refs.current[active]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [active, sources])

  return (
    <aside className="receipts" aria-label="Receipts">
      <div className="receipts-head">
        <h2>receipts</h2>
        <span className="receipts-count">{sources.length} sources</span>
      </div>
      {sources.length === 0 && (
        <p className="receipts-empty">The stint, result and race-summary data behind each answer appears here.</p>
      )}
      <div className="receipt-list">
        {sources.map((s, i) => {
          const { label, type } = sourceLabel(s)
          return (
            <div key={i} ref={(el) => { refs.current[i] = el }} className={`receipt${active === i ? ' active' : ''}`}>
              <div className="receipt-top">
                <span className="receipt-n">{i + 1}</span>
                <span className="receipt-title">{label}</span>
                <span className="receipt-type">{type}</span>
              </div>
              <p className="receipt-raw">{s.content}</p>
            </div>
          )
        })}
      </div>
    </aside>
  )
}
