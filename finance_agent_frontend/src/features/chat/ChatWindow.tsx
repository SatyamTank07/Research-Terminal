import React, { useRef, useEffect, useState } from 'react'
import type { AgentMilestone, Message } from './types'
import { MarkdownRenderer } from '@/shared/components/MarkdownRenderer'
import {
  TrendingUp,
  User,
  Copy,
  Check,
  AlertCircle,
  Clock,
  ExternalLink,
  FileText,
  Activity,
} from 'lucide-react'

const AGENT_META: Record<string, { label: string; badgeColor: string }> = {
  supervisor: {
    label: 'Supervisor Router',
    badgeColor: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20',
  },
  business_strategist: {
    label: 'Business & Moat Strategist',
    badgeColor: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
  },
  financial_auditor: {
    label: 'Financial Statement Auditor',
    badgeColor: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
  },
  risk_analyst: {
    label: 'Risk & Red Flag Analyst',
    badgeColor: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
  },
  forecasting_analyst: {
    label: 'Financial Forecasting Analyst',
    badgeColor: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',
  },
  valuation_specialist: {
    label: 'DCF Valuation Specialist',
    badgeColor: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
  },
  lead_synthesizer: {
    label: 'Lead Synthesizer',
    badgeColor: 'bg-sky-500/10 text-sky-400 border-sky-500/20',
  },
}

interface ChatWindowProps {
  messages: Message[]
  isLoading: boolean
  activeMilestones?: AgentMilestone[]
}

export const ChatWindow: React.FC<ChatWindowProps> = ({
  messages,
  isLoading,
  activeMilestones = [],
}) => {

  const bottomRef = useRef<HTMLDivElement>(null)
  const [copiedId, setCopiedId] = useState<string | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  const copyToClipboard = (id: string, text: string) => {
    navigator.clipboard.writeText(text)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 2000)
  }

  const formatTime = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    }).format(date)
  }

  return (
    <div className="flex-1 w-full overflow-y-auto min-h-0">
      <div className="w-full max-w-4xl mx-auto px-4 py-6 space-y-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center min-h-[480px] text-center px-4">
            <div className="w-14 h-14 rounded-xl bg-sky-500/10 border border-sky-500/20 flex items-center justify-center mb-5 text-sky-400 shadow-sm shadow-sky-500/5">
              <TrendingUp className="w-7 h-7 stroke-[2]" />
            </div>

            <h2 className="text-xl sm:text-2xl font-semibold text-white tracking-tight">
              Research Terminal
            </h2>
            <p className="text-slate-400 text-xs sm:text-sm max-w-lg mt-2.5 leading-relaxed">
              Conduct fundamental analysis, evaluate valuation models, review earnings consensus, and analyze regulatory disclosures across equity markets.
            </p>
          </div>
        ) : (
        messages.map((message) => {
          const isUser = message.role === 'user'

          return (
            <div key={message.id} className="w-full group">
              {isUser ? (
                /* Analyst Inquiry Card */
                <div className="flex flex-col items-end mb-4">
                  <div className="max-w-[90%] sm:max-w-[82%]">
                    <div className="flex items-center justify-end gap-2 mb-1.5 px-1">
                      <span className="text-[10px] text-slate-500 font-tabular flex items-center gap-1">
                        <Clock className="w-2.5 h-2.5" />
                        {formatTime(message.timestamp)}
                      </span>
                      <span className="text-[11px] font-semibold tracking-wider uppercase text-sky-400 flex items-center gap-1">
                        <User className="w-3 h-3" />
                        Analyst Inquiry
                      </span>
                    </div>

                    <div className="p-3.5 sm:p-4 rounded-xl bg-[#0e1424] border border-slate-800 text-slate-100 text-xs sm:text-sm leading-relaxed shadow-sm font-medium whitespace-pre-wrap">
                      {message.content}
                    </div>
                  </div>
                </div>
              ) : (
                /* Research Memorandum Card */
                <div className="flex flex-col items-start w-full mb-6">
                  <div className="w-full">
                    {/* Memorandum Top Bar */}
                    <div className="flex items-center justify-between mb-1.5 px-1">
                      <div className="flex items-center gap-2">
                        <span className="text-[11px] font-semibold tracking-wider uppercase text-slate-300 flex items-center gap-1.5">
                          <FileText className="w-3.5 h-3.5 text-sky-400" />
                          Research Memorandum
                        </span>
                        <span className="text-[10px] text-slate-500 font-tabular flex items-center gap-1">
                          <Clock className="w-2.5 h-2.5" />
                          {formatTime(message.timestamp)}
                        </span>
                      </div>

                      {!message.isError && (
                        <button
                          onClick={() => copyToClipboard(message.id, message.content)}
                          className="flex items-center gap-1 text-[11px] px-2 py-0.5 rounded text-slate-400 hover:text-white hover:bg-slate-800/80 transition-all cursor-pointer"
                          title="Copy research memorandum"
                        >
                          {copiedId === message.id ? (
                            <>
                              <Check className="w-3 h-3 text-emerald-400" />
                              <span className="text-emerald-400">Copied</span>
                            </>
                          ) : (
                            <>
                              <Copy className="w-3 h-3" />
                              <span>Copy Memo</span>
                            </>
                          )}
                        </button>
                      )}
                    </div>

                    {/* Memorandum Body */}
                    <div
                      className={`p-4 sm:p-6 rounded-xl text-xs sm:text-sm leading-relaxed shadow-md transition-all ${
                        message.isError
                          ? 'bg-red-950/20 border border-red-900/60 text-red-200 whitespace-pre-wrap'
                          : 'bg-[#0a0f1d] border border-slate-800/90 text-slate-100'
                      }`}
                    >
                      {message.isError && (
                        <div className="flex items-center gap-1.5 text-red-400 font-semibold mb-2 text-xs">
                          <AlertCircle className="w-3.5 h-3.5" />
                          <span>Inquiry Execution Error</span>
                        </div>
                      )}

                      {message.isError ? (
                        <div>{message.content}</div>
                      ) : (
                        <>
                          <MarkdownRenderer content={message.content} />

                          {message.sources && message.sources.length > 0 && (
                            <div className="mt-4 pt-3.5 border-t border-slate-800/80 flex flex-col gap-2">
                              <span className="text-[11px] font-semibold tracking-wider uppercase text-slate-400 flex items-center gap-1">
                                <ExternalLink className="w-3 h-3 text-sky-400" />
                                Cited Filings & Intelligence
                              </span>
                              <div className="flex flex-wrap gap-1.5">
                                {message.sources.map((src, idx) => (
                                  <a
                                    key={idx}
                                    href={src.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="inline-flex items-center gap-1 text-[11px] px-2.5 py-1 rounded bg-[#070a12] border border-slate-800 text-sky-300 hover:text-sky-200 hover:border-sky-500/40 transition-colors"
                                    title={src.snippet || src.title}
                                  >
                                    <span className="truncate max-w-[240px]">
                                      {src.title || src.url}
                                    </span>
                                    <ExternalLink className="w-2.5 h-2.5 opacity-60 shrink-0" />
                                  </a>
                                ))}
                              </div>
                            </div>
                          )}
                        </>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )
        })
      )}

      {/* Live Multi-Agent Pipeline Progress Card */}
      {isLoading && (
        <div className="w-full mb-6 p-4 sm:p-5 rounded-xl bg-[#0a0f1d] border border-slate-800/90 shadow-lg text-slate-100 animate-in fade-in duration-300">
          {/* Header */}
          <div className="flex items-center justify-between pb-3 mb-3 border-b border-slate-800/80">
            <div className="flex items-center gap-2">
              <span className="relative flex h-2.5 w-2.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-sky-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-sky-500"></span>
              </span>
              <span className="text-[11px] font-semibold tracking-wider uppercase text-slate-300 flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-sky-400" />
                Multi-Agent Research Pipeline Active
              </span>
            </div>
            {activeMilestones.length > 0 && (
              <span className="text-[10px] text-slate-500 font-mono">
                {activeMilestones.length} milestone{activeMilestones.length !== 1 ? 's' : ''} reached
              </span>
            )}
          </div>

          {/* Milestones list */}
          {activeMilestones.length > 0 ? (
            <div className="space-y-2">
              {activeMilestones.map((ms, index) => {
                const isLatest = index === activeMilestones.length - 1
                const meta = AGENT_META[ms.node] || {
                  label: ms.node.replace('_', ' ').toUpperCase(),
                  badgeColor: 'bg-slate-800 text-slate-300 border-slate-700',
                }

                return (
                  <div
                    key={ms.id}
                    className={`flex items-start gap-2.5 p-2.5 rounded-lg transition-all ${
                      isLatest
                        ? 'bg-sky-950/20 border border-sky-500/30 shadow-sm'
                        : 'bg-slate-900/40 border border-slate-800/40 opacity-75'
                    }`}
                  >
                    <div className="mt-0.5 shrink-0">
                      {isLatest ? (
                        <div className="w-3.5 h-3.5 border-2 border-sky-400 border-t-transparent rounded-full animate-spin" />
                      ) : (
                        <Check className="w-3.5 h-3.5 text-emerald-400" />
                      )}
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-0.5 flex-wrap">
                        <span
                          className={`text-[10px] font-semibold tracking-wider uppercase px-1.5 py-0.5 rounded border ${meta.badgeColor}`}
                        >
                          {meta.label}
                        </span>
                        <span className="text-[10px] text-slate-500 font-tabular">
                          {formatTime(ms.timestamp)}
                        </span>
                      </div>
                      <p className="text-xs text-slate-200 leading-snug">
                        {ms.message}
                      </p>
                    </div>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="flex items-center gap-2.5 py-1 text-slate-400 text-xs">
              <div className="w-3.5 h-3.5 border-2 border-sky-400 border-t-transparent rounded-full animate-spin shrink-0" />
              <span>Initializing multi-agent pipeline & triaging inquiry...</span>
            </div>
          )}
        </div>
      )}


      <div ref={bottomRef} />
      </div>
    </div>
  )
}
