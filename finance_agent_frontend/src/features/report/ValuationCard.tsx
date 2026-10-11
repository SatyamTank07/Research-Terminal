import React, { useState } from 'react'
import { ChevronRight } from 'lucide-react'
import { CitationMarker, MarkdownRenderer } from '@/shared/components/MarkdownRenderer'
import type { SourceRef } from '@/shared/types/citations'
import type { ValuationDashboard, ValuationStance } from './valuationDashboard'

interface ValuationCardProps {
  dashboard: ValuationDashboard
  /** The dashboard table markdown, shown on demand below the card. */
  table: string
  sources?: SourceRef[]
  onCite?: (ref: SourceRef) => void
}

const STANCE_STYLE: Record<ValuationStance, string> = {
  Overvalued: 'text-neg border-neg-dim bg-neg-bg',
  'Fairly Valued': 'text-acc border-acc-dim bg-acc-bg',
  Undervalued: 'text-pos border-pos-dim bg-pos-bg',
}

const LABEL = 'font-mono text-[10.5px] uppercase tracking-[0.08em] text-dim flex items-center gap-0.5'

const money = (v: number) =>
  `${v < 0 ? '−' : ''}$${Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`

const Cites: React.FC<{ ns: number[]; sources?: SourceRef[]; onCite?: (ref: SourceRef) => void }> = ({
  ns,
  sources,
  onCite,
}) => (
  <>
    {ns.map((n) => (
      <CitationMarker key={n} n={String(n)} source={sources?.find((s) => s.n === n)} onCite={onCite} />
    ))}
  </>
)

/** Bear-to-bull band with base-case tick and market-price marker; values sit in a legend so labels never collide. */
const ScenarioRange: React.FC<{ d: ValuationDashboard }> = ({ d }) => {
  const { bear, bull, fairValue, price } = d
  if (bear === null || bull === null) return null
  const lo0 = Math.min(bear, fairValue, price ?? bear)
  const hi0 = Math.max(bull, fairValue, price ?? bull)
  const pad = (hi0 - lo0) * 0.08 || Math.abs(hi0) * 0.1 || 1
  const lo = lo0 - pad
  const hi = hi0 + pad
  const pos = (v: number) => `${(((v - lo) / (hi - lo)) * 100).toFixed(2)}%`

  return (
    <div className="flex flex-col h-full">
      <div className={LABEL}>Scenario range vs market price</div>
      <div className="relative h-9 mt-5 mx-1">
        <div className="absolute inset-x-0 top-1/2 h-px bg-line-strong" />
        <div
          className="absolute top-1/2 -translate-y-1/2 h-3 bg-acc-bg border border-acc-dim"
          style={{ left: pos(bear), width: `calc(${pos(bull)} - ${pos(bear)})` }}
        />
        <div className="absolute top-1/2 -translate-y-1/2 w-0.5 h-6 bg-acc" style={{ left: pos(fairValue) }} />
        {price !== null && (
          <div className="absolute top-0 w-0.5 h-full bg-fg-strong" style={{ left: pos(price) }} />
        )}
      </div>
      <div className="mt-auto pt-4 flex flex-wrap gap-x-4 gap-y-1.5 font-mono text-[11px] font-tabular [&>span]:whitespace-nowrap">
        <span className="text-faint">
          <i className="inline-block w-2.5 h-2 mr-1.5 bg-acc-bg border border-acc-dim" />
          BEAR <span className="text-fg">{money(bear)}</span>
        </span>
        <span className="text-faint">
          <i className="inline-block w-2.5 h-2 mr-1.5 bg-acc" />
          BASE <span className="text-acc">{money(fairValue)}</span>
        </span>
        <span className="text-faint">
          <i className="inline-block w-2.5 h-2 mr-1.5 bg-acc-bg border border-acc-dim" />
          BULL <span className="text-fg">{money(bull)}</span>
        </span>
        {price !== null && (
          <span className="text-faint">
            <i className="inline-block w-2.5 h-2 mr-1.5 bg-fg-strong" />
            PRICE <span className="text-fg-strong">{money(price)}</span>
          </span>
        )}
      </div>
    </div>
  )
}

export const ValuationCard: React.FC<ValuationCardProps> = ({ dashboard: d, table, sources, onCite }) => {
  const [showTable, setShowTable] = useState(false)
  const upside = d.upsidePct ?? (d.price ? (d.fairValue / d.price - 1) * 100 : null)
  const hasRange = d.bear !== null && d.bull !== null

  const tiles = [
    d.waccPct && {
      label: 'WACC',
      value: d.waccPct,
      note: d.waccNote,
      cites: d.waccCites,
    },
    d.terminalGrowthPct && {
      label: 'Terminal g',
      value: d.terminalGrowthPct,
      note: d.terminalValueShare ? `TV ${d.terminalValueShare} of EV` : null,
      cites: [],
    },
    d.probabilityWeighted !== null && {
      label: 'Prob-weighted',
      value: money(d.probabilityWeighted),
      note: 'bear / base / bull',
      cites: [],
    },
    d.marketImpliedGrowthPct && {
      label: 'Market-implied g',
      value: d.marketImpliedGrowthPct,
      note: d.revenueCagrPct ? `vs ${d.revenueCagrPct} modeled CAGR` : 'reverse DCF',
      cites: [],
    },
  ].filter((t): t is { label: string; value: string; note: string | null; cites: number[] } => Boolean(t))

  return (
    <div className="my-4 border border-line-strong bg-surface">
      <div className={hasRange ? 'grid grid-cols-1 md:grid-cols-[1fr_1.35fr]' : ''}>
        <div className={`px-5 py-4 ${hasRange ? 'border-b md:border-b-0 md:border-r border-line' : ''}`}>
          <div className={LABEL}>
            Base-case DCF fair value / share
            <Cites ns={d.fairValueCites} sources={sources} onCite={onCite} />
          </div>
          <div className="font-mono font-tabular text-3xl sm:text-[34px] font-semibold text-fg-strong leading-tight mt-1.5">
            {money(d.fairValue)}
          </div>
          {d.price !== null && (
            <div className="font-mono font-tabular text-[12.5px] text-dim mt-1 flex flex-wrap items-center gap-x-1">
              <span>vs price</span>
              <span className="text-fg">{money(d.price)}</span>
              <Cites ns={d.priceCites} sources={sources} onCite={onCite} />
              {upside !== null && (
                <span className={`font-semibold ${upside >= 0 ? 'text-pos' : 'text-neg'}`}>
                  · {upside >= 0 ? '+' : '−'}
                  {Math.abs(upside).toFixed(1)}% {upside >= 0 ? 'upside' : 'downside'}
                </span>
              )}
            </div>
          )}
          {d.stance ? (
            <div
              className={`inline-block mt-3 px-2.5 py-1 border font-mono text-[11.5px] font-semibold uppercase tracking-[0.08em] ${STANCE_STYLE[d.stance]}`}
            >
              {d.stance}
            </div>
          ) : (
            d.stanceText && <div className="mt-3 font-mono text-[11px] text-faint">{d.stanceText}</div>
          )}
          {d.stance && <div className="font-mono text-[10.5px] text-faint mt-1.5">Stance band: ±10% around fair value</div>}
        </div>
        {hasRange && (
          <div className="px-5 py-4">
            <ScenarioRange d={d} />
          </div>
        )}
      </div>

      {tiles.length > 0 && (
        <div className="grid grid-cols-2 md:grid-flow-col md:auto-cols-fr border-t border-line">
          {tiles.map((t, i) => (
            <div
              key={t.label}
              className={`px-3.5 py-2.5 border-line ${i % 2 === 0 ? 'border-r' : ''} ${
                i < tiles.length - 2 ? 'border-b md:border-b-0' : ''
              } md:border-r md:last:border-r-0`}
            >
              <div className={LABEL}>
                {t.label}
                <Cites ns={t.cites} sources={sources} onCite={onCite} />
              </div>
              <div className="font-mono font-tabular text-[15px] font-semibold text-fg-strong mt-0.5">{t.value}</div>
              {t.note && <div className="font-mono text-[10px] text-faint mt-0.5 truncate">{t.note}</div>}
            </div>
          ))}
        </div>
      )}

      <button
        type="button"
        onClick={() => setShowTable((v) => !v)}
        aria-expanded={showTable}
        className="w-full flex items-center gap-2 px-3.5 py-2 border-t border-line font-mono text-[11px] tracking-wider text-dim hover:text-acc hover:bg-acc-bg transition-colors cursor-pointer"
      >
        <ChevronRight className={`w-3.5 h-3.5 transition-transform ${showTable ? 'rotate-90' : ''}`} />
        FULL VALUATION DASHBOARD · {d.rowCount} ROWS
      </button>
      {showTable && (
        <div className="border-t border-line px-3 pb-1 [&_.overflow-x-auto]:my-2">
          <MarkdownRenderer content={table} sources={sources} onCite={onCite} />
        </div>
      )}
    </div>
  )
}
