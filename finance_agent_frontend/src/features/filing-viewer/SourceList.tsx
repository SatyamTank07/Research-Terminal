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
    <div className="mt-4 pt-3.5 border-t border-slate-800/80 flex flex-col gap-2">
      <span className="text-[11px] font-semibold tracking-wider uppercase text-slate-400 flex items-center gap-1">
        <BookOpen className="w-3 h-3 text-sky-400" />
        Cited Filings & Intelligence
      </span>
      <ol className="flex flex-col gap-1">
        {sources.map((s) => (
          <li key={s.n} className="text-[11px]">
            {s.kind === 'market' ? (
              <div className="flex items-center gap-1.5 px-2 py-1 text-slate-400">
                <span className="font-mono text-slate-500">[{s.n}]</span>
                <LineChart className="w-3 h-3 shrink-0" />
                <span className="truncate">
                  {s.label} · {s.provider}
                </span>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => onOpen?.(s)}
                className="w-full text-left px-2 py-1 rounded bg-[#070a12] border border-slate-800 hover:border-sky-500/40 transition-colors"
              >
                <span className="font-mono text-sky-400 mr-1.5">[{s.n}]</span>
                <span className="text-slate-200">{s.label}</span>
                {s.kind === 'verbatim' && s.quote && (
                  <span className="block truncate text-slate-500 italic">“{s.quote}”</span>
                )}
                {s.kind === 'derived' && s.derived_from && (
                  <span className="block truncate text-slate-500">{s.derived_from}</span>
                )}
              </button>
            )}
          </li>
        ))}
      </ol>
    </div>
  )
}
