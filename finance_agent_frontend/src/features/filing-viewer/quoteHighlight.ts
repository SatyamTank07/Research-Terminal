/**
 * Marks a verbatim quote inside rendered filing text without mutating React-owned DOM, using the CSS
 * Custom Highlight API. Matching mirrors the backend's quote verification: case-insensitive, curly quotes
 * and dashes folded, whitespace runs collapsed. Returns false when the quote is not found or the browser
 * lacks the API (the chunk-level highlight still applies).
 */
const HIGHLIGHT_NAME = 'cite-quote'
const FOLD: Record<string, string> = {
  '\u2018': "'",
  '\u2019': "'",
  '\u201c': '"',
  '\u201d': '"',
  '\u2013': '-',
  '\u2014': '-',
  '\u00a0': ' ',
}

interface HighlightRegistryLike {
  set(name: string, highlight: object): unknown
  delete(name: string): boolean
}
type HighlightCtor = new (...ranges: Range[]) => object

function highlightApi(): { registry: HighlightRegistryLike; Highlight: HighlightCtor } | null {
  const registry = (CSS as unknown as { highlights?: HighlightRegistryLike }).highlights
  const Highlight = (globalThis as unknown as { Highlight?: HighlightCtor }).Highlight
  return registry && Highlight ? { registry, Highlight } : null
}

function fold(ch: string): string {
  return (FOLD[ch] ?? ch).toLowerCase()
}

export function clearQuoteHighlight(): void {
  highlightApi()?.registry.delete(HIGHLIGHT_NAME)
}

export function highlightQuote(container: HTMLElement, quote: string): boolean {
  const api = highlightApi()
  clearQuoteHighlight()
  const needle = Array.from(quote).map(fold).join('').replace(/\s+/g, ' ').trim()
  if (!api || !needle) return false

  let flat = ''
  const positions: Array<{ node: Text; offset: number }> = []
  let lastWasSpace = true
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT)
  for (let node = walker.nextNode() as Text | null; node; node = walker.nextNode() as Text | null) {
    const text = node.data
    for (let i = 0; i < text.length; i++) {
      const ch = fold(text[i])
      if (/\s/.test(ch)) {
        if (lastWasSpace) continue
        flat += ' '
        lastWasSpace = true
      } else {
        flat += ch
        lastWasSpace = false
      }
      positions.push({ node, offset: i })
    }
  }

  const start = flat.indexOf(needle)
  if (start < 0) return false
  const end = positions[start + needle.length - 1]
  const range = document.createRange()
  range.setStart(positions[start].node, positions[start].offset)
  range.setEnd(end.node, end.offset + 1)
  api.registry.set(HIGHLIGHT_NAME, new api.Highlight(range))
  return true
}
