import React from 'react'
import { BookOpen, LineChart } from 'lucide-react'
import type { SourceRef } from '@/shared/types/citations'

interface SourceListProps {
  sources: SourceRef[]
  onOpen?: (ref: SourceRef) => void
}

export const SourceList: React.FC<SourceListProps> = ({ sources, onOpen }) => {
  if (sources.length === 0) return null
  return (
    <div className="mt-6 pt-3.5 border-t border-line flex flex-col gap-2">
      <span className="font-mono text-[10.5px] tracking-[0.1em] uppercase text-dim flex items-center gap-1.5">
        <BookOpen className="w-3 h-3 text-acc" />
        Cited filings & intelligence
      </span>
      <ol className="flex flex-col gap-1">
        {sources.map((s) => (
          <li key={s.n} className="text-xs">
            {s.kind === 'market' ? (
              <div className="flex items-center gap-2 px-2.5 py-1.5 border border-line text-dim">
                <span className="font-mono text-faint">[{s.n}]</span>
                <LineChart className="w-3 h-3 shrink-0" />
                <span className="truncate">
                  {s.label} · {s.provider}
                </span>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => onOpen?.(s)}
                className="w-full text-left px-2.5 py-1.5 bg-canvas border border-line hover:border-acc-dim hover:bg-acc-bg transition-colors cursor-pointer"
              >
                <span className="font-mono text-acc mr-2">[{s.n}]</span>
                <span className="text-fg-strong">{s.label}</span>
                {s.kind === 'verbatim' && s.quote && (
                  <span className="block truncate text-faint italic mt-0.5">“{s.quote}”</span>
                )}
                {s.kind === 'derived' && s.derived_from && (
                  <span className="block truncate text-faint mt-0.5">{s.derived_from}</span>
                )}
              </button>
            )}
          </li>
        ))}
      </ol>
    </div>
  )
}
