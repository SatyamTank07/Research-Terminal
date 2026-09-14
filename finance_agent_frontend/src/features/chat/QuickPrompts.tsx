import React from 'react'
import { Card } from '@/shared/ui/card'
import {
  PieChart,
  DollarSign,
  Briefcase,
  ShieldCheck,
  TrendingDown,
  Sparkles,
} from 'lucide-react'

interface QuickPromptsProps {
  onSelectPrompt: (prompt: string) => void
  disabled?: boolean
}

const PROMPTS = [
  {
    icon: PieChart,
    category: 'Budgeting',
    text: 'Explain the 50/30/20 budgeting rule with a simple monthly example.',
  },
  {
    icon: DollarSign,
    category: 'Investing',
    text: 'What is dollar-cost averaging and why is it recommended for beginners?',
  },
  {
    icon: Briefcase,
    category: 'Retirement',
    text: 'Compare Roth IRA vs Traditional 401(k) tax advantages.',
  },
  {
    icon: ShieldCheck,
    category: 'Risk Management',
    text: 'How should I build a 6-month emergency fund safely?',
  },
  {
    icon: TrendingDown,
    category: 'Macroeconomics',
    text: 'How do central bank interest rate cuts affect stock and bond valuations?',
  },
]

export const QuickPrompts: React.FC<QuickPromptsProps> = ({
  onSelectPrompt,
  disabled,
}) => {
  return (
    <div className="w-full max-w-3xl mx-auto my-6 px-4">
      <div className="flex items-center gap-2 mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
        <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
        Suggested Financial Inquiries
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
        {PROMPTS.map((item, idx) => {
          const Icon = item.icon
          return (
            <button
              key={idx}
              onClick={() => onSelectPrompt(item.text)}
              disabled={disabled}
              className="text-left group transition-all duration-200 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Card className="h-full p-3.5 bg-slate-900/60 border-slate-800 hover:border-emerald-500/50 hover:bg-slate-800/80 transition-all cursor-pointer flex flex-col justify-between gap-2 shadow-sm hover:shadow-emerald-500/5">
                <div className="flex items-center gap-2">
                  <div className="p-1.5 rounded-md bg-emerald-500/10 text-emerald-400 group-hover:bg-emerald-500/20 transition-colors">
                    <Icon className="w-3.5 h-3.5" />
                  </div>
                  <span className="text-xs font-medium text-emerald-400">
                    {item.category}
                  </span>
                </div>
                <p className="text-xs text-slate-300 line-clamp-2 group-hover:text-white leading-relaxed">
                  {item.text}
                </p>
              </Card>
            </button>
          )
        })}
      </div>
    </div>
  )
}
