// Light/dark mode, as in the v5 design: stored on <html data-vpe-mode>, remembered in
// localStorage (best-effort — it can throw in private windows), defaulting to the OS setting.

const KEY = 'vpe-mode'

export function initialMode() {
  try {
    const saved = localStorage.getItem(KEY)
    if (saved === 'light' || saved === 'dark') return saved
  } catch { /* storage unavailable */ }
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function applyMode(mode) {
  document.documentElement.setAttribute('data-vpe-mode', mode)
}

// Switch mode with a circular reveal growing out of `originEl` (View Transitions API);
// falls back to a short cross-fade where that API or motion isn't available.
export function switchMode(next, originEl) {
  try { localStorage.setItem(KEY, next) } catch { /* ignore */ }
  const root = document.documentElement
  const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

  if (document.startViewTransition && !reduce) {
    const r = originEl?.getBoundingClientRect() ?? { left: innerWidth, top: 0, width: 0, height: 0 }
    const x = r.left + r.width / 2
    const y = r.top + r.height / 2
    const radius = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y))
    const vt = document.startViewTransition(() => applyMode(next))
    vt.ready
      .then(() => root.animate(
        { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${radius}px at ${x}px ${y}px)`] },
        { duration: 700, easing: 'cubic-bezier(.65,0,.25,1)', pseudoElement: '::view-transition-new(root)' },
      ))
      .catch(() => {})
  } else {
    root.classList.add('vpe-fade')
    applyMode(next)
    setTimeout(() => root.classList.remove('vpe-fade'), 600)
  }
}
