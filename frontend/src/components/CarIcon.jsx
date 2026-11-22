// The two car SVGs from the v5 design, coloured entirely through CSS variables
// so they follow light/dark mode.

export function MiniCar() {
  return (
    <svg viewBox="0 0 120 48" width="26" height="11" aria-hidden="true">
      <rect x="2" y="6" width="9" height="36" rx="1.5" style={{ fill: 'var(--c23)' }} />
      <rect x="14" y="1" width="18" height="11" rx="3" style={{ fill: 'var(--carTire)' }} />
      <rect x="14" y="36" width="18" height="11" rx="3" style={{ fill: 'var(--carTire)' }} />
      <rect x="78" y="4" width="15" height="9" rx="3" style={{ fill: 'var(--carTire)' }} />
      <rect x="78" y="35" width="15" height="9" rx="3" style={{ fill: 'var(--carTire)' }} />
      <path d="M10 19 L40 14 L72 18 L106 21.5 L106 26.5 L72 30 L40 34 L10 29 Z" style={{ fill: 'var(--c23)' }} />
      <ellipse cx="44" cy="24" rx="19" ry="10.5" style={{ fill: 'var(--c23)' }} />
      <rect x="104" y="8" width="8" height="32" rx="1.5" style={{ fill: 'var(--c23)' }} />
    </svg>
  )
}

export function RaceCar() {
  const v = (name) => ({ fill: `var(${name})` })
  const tyre = { fill: '#0A0A12' }
  return (
    <svg viewBox="0 0 120 48" width="92" height="37" style={{ display: 'block', overflow: 'visible' }} aria-hidden="true">
      <rect x="1" y="5" width="10" height="38" rx="1.5" style={v('--carWing')} />
      <rect x="1" y="5" width="10" height="5" style={v('--carWingTip')} />
      <rect x="1" y="38" width="10" height="5" style={v('--carWingTip')} />
      <rect x="14" y="0" width="19" height="12" rx="3.5" style={tyre} />
      <rect x="14" y="36" width="19" height="12" rx="3.5" style={tyre} />
      <rect x="79" y="3" width="15" height="10" rx="3" style={tyre} />
      <rect x="79" y="35" width="15" height="10" rx="3" style={tyre} />
      <rect x="18" y="4" width="11" height="1.6" style={v('--carStripe')} />
      <rect x="18" y="42.4" width="11" height="1.6" style={v('--carStripe')} />
      <path d="M10 19 L40 13.5 L74 18 L104 21.8 L104 26.2 L74 30 L40 34.5 L10 29 Z" style={v('--carBody')} />
      <path d="M24 13 C34 9 52 10 60 15 L60 33 C52 38 34 39 24 35 Z" style={v('--carSide')} />
      <path d="M30 14.5 C38 12 48 12.5 55 16 L50 19 C44 17 37 17 30 18.5 Z" style={v('--carStripe')} />
      <path d="M30 33.5 C38 36 48 35.5 55 32 L50 29 C44 31 37 31 30 29.5 Z" style={v('--carStripe')} />
      <path d="M78 19.6 L104 21.8 L104 26.2 L78 28.4 Z" style={v('--carNose')} />
      <ellipse cx="60" cy="24" rx="8" ry="4.4" style={tyre} />
      <path d="M53 19.4 Q65 24 53 28.6" fill="none" strokeWidth="1.8" style={{ stroke: 'var(--carHalo)' }} />
      <rect x="46" y="22.6" width="6" height="2.8" rx="1" style={v('--carNose')} />
      <rect x="103" y="7" width="9" height="34" rx="1.5" style={v('--carWing')} />
      <rect x="103" y="7" width="9" height="5" style={v('--carWingTip')} />
      <rect x="103" y="36" width="9" height="5" style={v('--carWingTip')} />
    </svg>
  )
}
