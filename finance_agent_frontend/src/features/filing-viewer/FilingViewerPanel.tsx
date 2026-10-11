import React, { useEffect, useRef } from 'react'
import { AlertTriangle, FileText, Loader2, X } from 'lucide-react'
import { MarkdownRenderer } from '@/shared/components/MarkdownRenderer'
import type { FilingViewer } from './hooks/useFilingViewer'
import { clearQuoteHighlight, highlightQuote } from './quoteHighlight'

interface FilingViewerPanelProps {
  viewer: FilingViewer
}

export const FilingViewerPanel: React.FC<FilingViewerPanelProps> = ({ viewer }) => {
  const bodyRef = useRef<HTMLDivElement>(null)
  const { source, filing, targetChunkId, approximate, loading, error, chunkPos, openCount } = viewer

  useEffect(() => {
    if (!filing || !targetChunkId || !bodyRef.current) return
    const el = bodyRef.current.querySelector<HTMLElement>(`[data-chunk-id="${CSS.escape(targetChunkId)}"]`)
    if (!el) return
    el.scrollIntoView({ block: 'center' })
    if (source?.kind === 'verbatim' && source.quote) {
      highlightQuote(el, source.quote)
    } else {
      clearQuoteHighlight()
    }
  }, [filing, targetChunkId, source, openCount])

  useEffect(() => clearQuoteHighlight, [])

  if (!viewer.isOpen || !source) return null

  const jumpTo = (chunkIndex: number) => {
    bodyRef.current
      ?.querySelector<HTMLElement>(`[data-chunk-index="${chunkIndex}"]`)
      ?.scrollIntoView({ block: 'start' })
  }
  const doc = filing?.document

  return (
    <aside className="fixed inset-0 z-40 flex flex-col bg-surface md:static md:inset-auto md:z-auto md:w-[45%] md:max-w-[760px] md:border-l md:border-line h-screen">
      <header className="flex items-start justify-between gap-3 px-4 py-3 border-b border-line border-t-2 border-t-acc">
        <div className="min-w-0">
          <div className="flex items-center gap-2 font-mono text-xs font-semibold text-fg-strong">
            <FileText className="w-3.5 h-3.5 text-acc shrink-0" />
            {doc ? `${doc.ticker} ${doc.filing_type} · FY${doc.fiscal_year}` : 'SEC filing'}
            <span className="text-acc">[{source.n}]</span>
          </div>
          <div className="text-xs text-dim truncate mt-0.5">{source.label}</div>
        </div>
        <button
          type="button"
          onClick={viewer.close}
          className="p-1 rounded-xs hover:bg-raised text-dim hover:text-fg-strong cursor-pointer"
          aria-label="Close filing viewer"
        >
          <X className="w-4 h-4" />
        </button>
      </header>

      {source.kind === 'verbatim' && source.quote && (
        <blockquote className="mx-4 mt-3 border-l-2 border-acc bg-acc-bg pl-3 pr-2 py-2 text-xs text-[#f5d3ad] italic leading-relaxed">
          “{source.quote}”
        </blockquote>
      )}
      {source.kind === 'derived' && (
        <div className="mx-4 mt-3 text-xs text-fg">
          <div className="mb-1.5">
            Derived from {source.chunks.length} filing table{source.chunks.length === 1 ? '' : 's'}
            {source.derived_from ? ` — ${source.derived_from}` : ''}
          </div>
          {source.chunks.length > 1 && (
            <div className="flex flex-wrap gap-1">
              {source.chunks.map((c, i) => (
                <button
                  key={c.chunk_id}
                  type="button"
                  onClick={() => viewer.selectChunk(i)}
                  className={`px-2 py-0.5 rounded-xs border font-mono text-[10px] cursor-pointer ${
                    i === chunkPos
                      ? 'border-acc text-acc bg-acc-bg'
                      : 'border-line-strong text-dim hover:border-acc-dim'
                  }`}
                >
                  {c.item ?? 'Table'} · FY{c.fiscal_year}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      {approximate && (
        <div className="mx-4 mt-3 flex items-center gap-1.5 text-xs text-acc">
          <AlertTriangle className="w-3.5 h-3.5" />
          Approximate location: the filing was re-ingested after this answer was written.
        </div>
      )}

      <div className="flex flex-1 min-h-0 mt-3 border-t border-line">
        {filing && filing.toc.length > 0 && (
          <nav className="hidden lg:block w-44 shrink-0 overflow-y-auto border-r border-line px-1.5 py-2">
            {filing.toc.map((t) => (
              <button
                key={`${t.part}-${t.item}`}
                type="button"
                onClick={() => jumpTo(t.chunk_index)}
                className="block w-full text-left font-mono text-[10.5px] text-faint hover:text-acc hover:bg-acc-bg px-1.5 py-1 truncate cursor-pointer"
                title={t.title}
              >
                {t.title}
              </button>
            ))}
          </nav>
        )}
        <div ref={bodyRef} className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
          {loading && (
            <div className="flex items-center gap-2 text-xs text-dim">
              <Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading filing…
            </div>
          )}
          {error && !loading && (
            <div className="flex items-center gap-2 text-xs text-neg">
              <AlertTriangle className="w-3.5 h-3.5" /> {error}
            </div>
          )}
          {filing &&
            !error &&
            filing.chunks.map((c) => (
              <div
                key={c.id}
                data-chunk-id={c.id}
                data-chunk-index={c.chunk_index}
                className={
                  c.id === targetChunkId
                    ? 'ring-1 ring-acc-dim bg-acc/5 p-2 -mx-2 scroll-mt-4'
                    : 'scroll-mt-4'
                }
              >
                <MarkdownRenderer content={c.content} />
              </div>
            ))}
        </div>
      </div>
    </aside>
  )
}
