import { useEffect, useRef, useState } from 'react'

// Custom dropdown — native <select> popups can't be themed, so we render our
// own list to keep the dark dashboard consistent across browsers/OSes.
export default function Dropdown({ value, onChange, options, ariaLabel, className = '' }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    function onDoc(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', onDoc)
    return () => document.removeEventListener('mousedown', onDoc)
  }, [])

  const selected = options.find((o) => o.value === value)

  return (
    <div className={`dd ${className} ${open ? 'open' : ''}`} ref={ref}>
      <button type="button" className="dd-trigger" aria-label={ariaLabel}
        aria-haspopup="listbox" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <span className="dd-value">{selected ? selected.label : 'Select…'}</span>
        <svg className="dd-arrow" width="12" height="12" viewBox="0 0 24 24" aria-hidden="true">
          <path d="M6 9l6 6 6-6" fill="none" stroke="currentColor" strokeWidth="2.5"
            strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {open && (
        <ul className="dd-menu" role="listbox">
          {options.map((o) => (
            <li key={o.value} role="option" aria-selected={o.value === value}
              className={`dd-option ${o.value === value ? 'active' : ''}`}
              onClick={() => { onChange(o.value); setOpen(false) }}>
              {o.label}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
