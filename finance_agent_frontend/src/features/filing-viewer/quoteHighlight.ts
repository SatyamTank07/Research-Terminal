/**
 * Marks a verbatim quote inside rendered filing text without mutating React-owned DOM, using the CSS
 * Custom Highlight API. Matching mirrors the backend's quote verification: case-insensitive, curly quotes
 * and dashes folded, whitespace runs collapsed, near-verbatim quotes anchored by prefix/suffix. Returns
 * false when the quote is not found or the browser lacks the API (the chunk-level highlight still applies).
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

  const span = locateQuote(flat, needle)
  if (!span) return false
  const first = positions[span[0]]
  const last = positions[span[1] - 1]
  if (!first || !last) return false
  const range = document.createRange()
  range.setStart(first.node, first.offset)
  range.setEnd(last.node, last.offset + 1)
  api.registry.set(HIGHLIGHT_NAME, new api.Highlight(range))
  return true
}

const MIN_ANCHOR_WORDS = 6

/**
 * [start, end) of the quote in the flattened text. Model quotes are near-verbatim: they often end with
 * a period where the filing's sentence continues, or skip a parenthetical. So: exact match without
 * trailing punctuation first, else the longest matching word prefix and suffix, spanning both when the
 * gap is plausible (at most 1.5x the quote) and falling back to the prefix alone.
 */
function locateQuote(flat: string, rawNeedle: string): [number, number] | null {
  const needle = rawNeedle.replace(/[.;:,!?]+$/, '')
  const exact = flat.indexOf(needle)
  if (exact >= 0) return [exact, exact + needle.length]

  const words = needle.split(' ')
  if (words.length < MIN_ANCHOR_WORDS) return null
  let start = -1
  let prefixEnd = -1
  for (let k = words.length - 1; k >= MIN_ANCHOR_WORDS; k--) {
    const prefix = words.slice(0, k).join(' ')
    const at = flat.indexOf(prefix)
    if (at >= 0) {
      start = at
      prefixEnd = at + prefix.length
      break
    }
  }
  let end = -1
  for (let k = words.length - 1; k >= MIN_ANCHOR_WORDS; k--) {
    const suffix = words.slice(words.length - k).join(' ')
    const at = flat.indexOf(suffix, Math.max(start, 0))
    if (at >= 0) {
      end = at + suffix.length
      if (start < 0) start = at
      break
    }
  }
  if (start < 0) return null
  if (end < 0 || end - start > needle.length * 1.5) end = prefixEnd > 0 ? prefixEnd : end
  return [start, end]
}
