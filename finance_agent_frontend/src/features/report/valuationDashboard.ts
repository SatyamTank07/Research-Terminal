/**
 * Parses the "Executive Valuation Dashboard" section that the backend's `report_blocks.j2`
 * `dashboard` macro renders (full_10k_report and dcf_valuation_only routes).
 *
 * The structured `final_report` only exists on a live SSE result and is not persisted, so the
 * markdown table is the one source that also works for conversation history. Row labels are
 * fixed by the template; values keep their raw text and are parsed leniently ("n/a" -> null).
 */

const DASHBOARD_HEADING = /^##\s+Executive Valuation Dashboard\s*$/im
const NEXT_SECTION = /^#{1,2}\s/m
const CITE = /\[\[(\d+)\]\]\(#cite-\d+\)/g

export type ValuationStance = 'Undervalued' | 'Fairly Valued' | 'Overvalued'

export interface DashboardRow {
  label: string
  value: string
  cites: number[]
}

export interface ValuationDashboard {
  fairValue: number
  fairValueCites: number[]
  bear: number | null
  bull: number | null
  probabilityWeighted: number | null
  price: number | null
  priceCites: number[]
  upsidePct: number | null
  stance: ValuationStance | null
  /** Raw stance text, e.g. "Not stated (no market price)". */
  stanceText: string | null
  waccPct: string | null
  waccNote: string | null
  waccCites: number[]
  terminalGrowthPct: string | null
  terminalValueShare: string | null
  marketImpliedGrowthPct: string | null
  revenueCagrPct: string | null
  rowCount: number
}

export interface SplitReport {
  /** Markdown before the dashboard heading (title + filing banner). */
  before: string
  /** The dashboard table alone, without its heading. */
  table: string
  /** Everything after the dashboard section. */
  after: string
  dashboard: ValuationDashboard
}

/** "$111.21" -> 111.21, "$-33.0M" -> -33, "n/a" -> null. */
export function parseNumber(text: string | undefined): number | null {
  if (!text) return null
  const match = /([-−]?)\s*\$?\s*([-−]?)\s*(\d[\d,]*(?:\.\d+)?)/.exec(text)
  if (!match) return null
  const sign = match[1] || match[2] ? -1 : 1
  const value = Number(match[3].replace(/,/g, ''))
  return Number.isFinite(value) ? sign * value : null
}

/** First percentage in the text, kept as written ("9.75%"). */
function firstPct(text: string | undefined): string | null {
  const match = text ? /[-−]?\d[\d,]*(?:\.\d+)?%/.exec(text) : null
  return match ? match[0] : null
}

function stripMarkup(text: string): string {
  return text.replace(/\*\*/g, '').replace(/\s+/g, ' ').trim()
}

function parseRows(table: string): DashboardRow[] {
  const rows: DashboardRow[] = []
  for (const line of table.split('\n')) {
    const trimmed = line.trim()
    if (!trimmed.startsWith('|')) continue
    const cells = trimmed.replace(/^\||\|$/g, '').split('|')
    if (cells.length < 2) continue
    const [rawLabel, rawValue] = cells
    if (/^\s*:?-{3,}:?\s*$/.test(rawLabel)) continue // separator row
    const cites = [...rawLabel.matchAll(CITE)].map((m) => Number(m[1]))
    const label = stripMarkup(rawLabel.replace(CITE, ''))
    if (label.toLowerCase() === 'metric') continue // header row
    rows.push({ label, value: stripMarkup(rawValue.replace(CITE, '')), cites })
  }
  return rows
}

function toStance(text: string | null): ValuationStance | null {
  if (!text) return null
  const t = text.toLowerCase()
  if (t.startsWith('undervalued')) return 'Undervalued'
  if (t.startsWith('overvalued')) return 'Overvalued'
  if (t.startsWith('fairly valued')) return 'Fairly Valued'
  return null
}

function buildDashboard(rows: DashboardRow[]): ValuationDashboard | null {
  const find = (prefix: string) => rows.find((r) => r.label.toLowerCase().startsWith(prefix))
  const fair = find('base-case dcf fair value')
  const fairValue = parseNumber(fair?.value)
  if (!fair || fairValue === null) return null

  const bearBull = find('bear / bull fair value')?.value.split('/')
  const price = find('current share price')
  const wacc = find('discount rate')
  const stanceText = find('valuation stance')?.value ?? null
  const forecast = find('forecast')?.value

  return {
    fairValue,
    fairValueCites: fair.cites,
    bear: parseNumber(bearBull?.[0]),
    bull: parseNumber(bearBull?.[1]),
    probabilityWeighted: parseNumber(find('probability-weighted fair value')?.value),
    price: parseNumber(price?.value),
    priceCites: price?.cites ?? [],
    upsidePct: parseNumber(find('base-case upside / downside')?.value),
    stance: toStance(stanceText),
    stanceText,
    waccPct: firstPct(wacc?.value),
    waccNote: wacc ? (/\(([^)]*)\)/.exec(wacc.value)?.[1] ?? null) : null,
    waccCites: wacc?.cites ?? [],
    terminalGrowthPct: firstPct(find('terminal growth')?.value),
    terminalValueShare: firstPct(find('terminal value % of ev')?.value),
    marketImpliedGrowthPct: firstPct(find('market-implied revenue growth')?.value),
    revenueCagrPct: forecast ? (/([-−]?\d[\d.]*%)\s+revenue CAGR/i.exec(forecast)?.[1] ?? null) : null,
    rowCount: rows.length,
  }
}

/** Splits a report around its valuation dashboard; null when there is no parseable dashboard. */
export function splitValuationReport(markdown: string): SplitReport | null {
  const heading = DASHBOARD_HEADING.exec(markdown)
  if (!heading) return null
  const sectionStart = heading.index + heading[0].length
  const rest = markdown.slice(sectionStart)
  const next = NEXT_SECTION.exec(rest)
  const table = (next ? rest.slice(0, next.index) : rest).trim()
  const dashboard = buildDashboard(parseRows(table))
  if (!dashboard) return null
  return {
    before: markdown.slice(0, heading.index),
    table,
    after: next ? rest.slice(next.index) : '',
    dashboard,
  }
}
