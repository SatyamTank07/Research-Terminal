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
  const { source, filing, targetChunkId, approximate, loading, error, chunkPos } = viewer

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
  }, [filing, targetChunkId, source])

  useEffect(() => clearQuoteHighlight, [])

  if (!viewer.isOpen || !source) return null

  const jumpTo = (chunkIndex: number) => {
    bodyRef.current
      ?.querySelector<HTMLElement>(`[data-chunk-index="${chunkIndex}"]`)
      ?.scrollIntoView({ block: 'start' })
  }
  const doc = filing?.document

  return (
    <aside className="fixed inset-0 z-40 flex flex-col bg-[#070a12] md:static md:inset-auto md:z-auto md:w-[45%] md:max-w-[760px] md:border-l md:border-slate-800 h-screen">
      <header className="flex items-start justify-between gap-3 px-4 py-3 border-b border-slate-800">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-100">
            <FileText className="w-3.5 h-3.5 text-sky-400 shrink-0" />
            {doc ? `${doc.ticker} ${doc.filing_type} · FY${doc.fiscal_year}` : 'SEC filing'}
            <span className="font-mono text-sky-400">[{source.n}]</span>
          </div>
          <div className="text-[11px] text-slate-400 truncate">{source.label}</div>
        </div>
        <button
          type="button"
          onClick={viewer.close}
          className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-100"
          aria-label="Close filing viewer"
        >
          <X className="w-4 h-4" />
        </button>
      </header>

      {source.kind === 'verbatim' && source.quote && (
        <blockquote className="mx-4 mt-3 border-l-2 border-amber-400 bg-amber-400/5 pl-3 py-1.5 text-[11px] text-amber-100/90 italic">
          “{source.quote}”
        </blockquote>
      )}
      {source.kind === 'derived' && (
        <div className="mx-4 mt-3 text-[11px] text-slate-300">
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
                  className={`px-2 py-0.5 rounded border text-[10px] ${
                    i === chunkPos
                      ? 'border-sky-500 text-sky-300 bg-sky-500/10'
                      : 'border-slate-700 text-slate-400 hover:border-slate-500'
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
        <div className="mx-4 mt-3 flex items-center gap-1.5 text-[11px] text-amber-300">
          <AlertTriangle className="w-3.5 h-3.5" />
          Approximate location: the filing was re-ingested after this answer was written.
        </div>
      )}

      <div className="flex flex-1 min-h-0 mt-3">
        {filing && filing.toc.length > 0 && (
          <nav className="hidden lg:block w-44 shrink-0 overflow-y-auto border-r border-slate-800 px-2 py-2">
            {filing.toc.map((t) => (
              <button
                key={`${t.part}-${t.item}`}
                type="button"
                onClick={() => jumpTo(t.chunk_index)}
                className="block w-full text-left text-[11px] text-slate-400 hover:text-sky-300 py-1 truncate"
                title={t.title}
              >
                {t.title}
              </button>
            ))}
          </nav>
        )}
        <div ref={bodyRef} className="flex-1 overflow-y-auto px-4 py-2 space-y-3">
          {loading && (
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading filing…
            </div>
          )}
          {error && !loading && (
            <div className="flex items-center gap-2 text-xs text-red-400">
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
                    ? 'rounded-md ring-1 ring-amber-400/60 bg-amber-400/5 p-2 -mx-2 scroll-mt-4'
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
