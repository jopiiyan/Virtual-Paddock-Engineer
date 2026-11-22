// Small helpers shared by the chat and receipts.

// Short label + type for a retrieved source, from its metadata.
export function sourceLabel(src) {
  const m = src.metadata || {}
  const type = m.doc_type || (m.stint != null ? 'stint' : 'doc')
  if (type === 'summary') return { label: 'Race summary', type: 'SUMMARY' }
  if (type === 'result') {
    return { label: m.position ? `P${m.position} ${m.driver}` : `${m.driver} result`, type: 'RESULT' }
  }
  if (type === 'stint') {
    const compound = m.compound ? ` · ${m.compound.toLowerCase()}` : ''
    return { label: `${m.driver} stint ${m.stint}${compound}`, type: 'STINT' }
  }
  return { label: m.driver || 'Source', type: String(type).toUpperCase() }
}

// Minimal, safe renderer for the model's markdown: paragraphs, "- " / "* " bullets and
// **bold**. Produces React elements only (no innerHTML), so model output can't inject markup.
function inline(text, keyBase) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith('**') && part.endsWith('**') && part.length > 4
      ? <strong key={`${keyBase}-${i}`}>{part.slice(2, -2)}</strong>
      : part,
  )
}

export function Markdown({ text }) {
  const blocks = []
  let list = null
  text.split('\n').forEach((raw, i) => {
    const line = raw.trimEnd()
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/)
    const heading = line.match(/^\s*#{1,6}\s+(.*)$/)
    if (heading) {
      list = null
      blocks.push({ type: 'h', text: heading[1].replace(/\*\*/g, '') })
    } else if (bullet) {
      if (!list) { list = []; blocks.push({ type: 'ul', items: list }) }
      list.push(bullet[1])
    } else {
      list = null
      if (line.trim()) blocks.push({ type: 'p', text: line.trim(), key: i })
    }
  })
  return blocks.map((b, i) => {
    if (b.type === 'ul') return <ul key={i}>{b.items.map((it, j) => <li key={j}>{inline(it, `${i}-${j}`)}</li>)}</ul>
    if (b.type === 'h') return <p key={i} className="md-h">{b.text}</p>
    return <p key={i}>{inline(b.text, i)}</p>
  })
}
