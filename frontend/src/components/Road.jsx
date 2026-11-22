import { useEffect, useRef, useState } from 'react'
import { RaceCar } from './CarIcon'

// Sine-wave road with a car driving along it. The car cruises normally and speeds up
// (with a motion-blur trail) while `thinking` is true — i.e. while an answer is
// being retrieved and streamed. Parked and static under prefers-reduced-motion.

const AMP = 30
const K = (Math.PI * 2) / 380
const MID = 66
const roadY = (x) => MID + Math.sin(x * K) * AMP

function roadPath(width) {
  let d = ''
  for (let x = -60; x <= width + 60; x += 6) d += `${d ? ' L' : 'M'}${x} ${roadY(x).toFixed(1)}`
  return d
}

export default function Road({ thinking }) {
  const roadRef = useRef(null)
  const carRef = useRef(null)
  const trailRef = useRef(null)
  const thinkingRef = useRef(thinking)
  const [width, setWidth] = useState(1200)

  thinkingRef.current = thinking

  useEffect(() => {
    const measure = () => roadRef.current && setWidth(roadRef.current.clientWidth)
    measure()
    window.addEventListener('resize', measure)

    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches
    let x = reduce ? 240 : -120
    let last = performance.now()
    let speed = 140
    let raf

    const tick = (now) => {
      const dt = Math.min(0.05, (now - last) / 1000)
      last = now
      const road = roadRef.current, car = carRef.current, trail = trailRef.current
      if (road && car) {
        const W = road.clientWidth
        const target = thinkingRef.current ? 640 : 140
        speed += (target - speed) * Math.min(1, dt * 3)
        if (!reduce) x += speed * dt
        if (x > W + 140) x = -140
        const y = roadY(x)
        const ang = (Math.atan(Math.cos(x * K) * AMP * K) * 180) / Math.PI
        const f = Math.max(0, (speed - 140) / 500)
        car.style.transform = `translate(${x}px, ${y}px) rotate(${ang}deg) scaleX(${1 + f * 0.12})`
        car.style.filter = `blur(${(0.3 + f * 1.4).toFixed(2)}px)`
        if (trail) {
          trail.style.transform = `translate(${x - 150}px, ${y}px) rotate(${ang}deg)`
          trail.style.opacity = (0.25 + f * 0.75).toFixed(2)
        }
      }
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', measure) }
  }, [])

  const d = roadPath(width)
  return (
    <div ref={roadRef} className={`road${thinking ? ' thinking' : ''}`} aria-hidden="true">
      <svg className="road-svg" width="100%" height="132">
        <path d={d} fill="none" strokeWidth="66" strokeLinecap="round" style={{ stroke: 'var(--c18)', filter: 'blur(10px)' }} />
        <path d={d} fill="none" strokeWidth="58" style={{ stroke: 'var(--c9)' }} />
        <path d={d} fill="none" strokeWidth="58" strokeDasharray="14 14" style={{ stroke: 'var(--c2)' }} />
        <path d={d} fill="none" strokeWidth="46" style={{ stroke: 'var(--c19)' }} />
        <path d={d} fill="none" strokeWidth="2" strokeDasharray="18 16" style={{ stroke: 'var(--c20)' }} />
      </svg>
      <div ref={trailRef} className="road-trail" />
      <div ref={carRef} className="road-car"><RaceCar /></div>
      <span className="road-label">// retrieving</span>
    </div>
  )
}
