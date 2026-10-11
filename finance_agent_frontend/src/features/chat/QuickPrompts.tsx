import React from 'react'

interface QuickPromptsProps {
  /** Receives a prompt prefix; the analyst completes it with a company before sending. */
  onSelectPrompt: (prompt: string) => void
  disabled?: boolean
}

// One entry per backend research route. Prefixes stop before the company name: the supervisor
// resolves tickers from the inquiry, so the UI never supplies a default company.
const PROMPTS = [
  { title: 'Full 10-K report', hint: 'Complete 7-agent research memorandum', text: 'Full 10-K research report on ' },
  { title: 'DCF valuation', hint: 'WACC, UFCF schedule, scenarios, sensitivity', text: 'DCF valuation of ' },
  { title: 'Financial audit', hint: 'Ratios, earnings quality, forensic red flags', text: 'Financial statement audit of ' },
  { title: 'Business & moat', hint: 'Segments, moat type, competitive threats', text: 'Business model and moat analysis of ' },
  { title: 'Risk factors', hint: 'Item 1A matrix and YoY disclosure changes', text: 'Risk factor analysis of ' },
  { title: 'Ask a question', hint: 'Quick Q&A on a filing or a concept', text: '' },
]

export const QuickPrompts: React.FC<QuickPromptsProps> = ({
  onSelectPrompt,
  disabled,
}) => {
  return (
    <div className="w-full">
      <div className="font-mono text-[10.5px] uppercase tracking-[0.1em] text-faint mb-2.5">
        Start an analysis
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
        {PROMPTS.map((item, idx) => (
          <button
            key={item.title}
            type="button"
            onClick={() => onSelectPrompt(item.text)}
            disabled={disabled}
            className="text-left flex gap-3 px-3.5 py-3 border border-line bg-panel hover:border-acc-dim hover:bg-acc-bg transition-colors cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <span className="font-mono text-[10px] text-acc tracking-wider pt-0.5 w-6 shrink-0">
              F{idx + 1}
            </span>
            <span className="min-w-0">
              <span className="block text-[13px] font-semibold text-fg-strong">{item.title}</span>
              <span className="block text-xs text-dim">{item.hint}</span>
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}
