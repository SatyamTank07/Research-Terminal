import React, { useRef, useEffect, useState } from 'react'
import type { AgentMilestone, Message } from './types'
import { MarkdownRenderer } from '@/shared/components/MarkdownRenderer'
import type { SourceRef } from '@/shared/types/citations'
import { SourceList } from '@/features/filing-viewer/SourceList'
import { QuickPrompts } from './QuickPrompts'
import {
  TrendingUp,
  User,
  Copy,
  Check,
  AlertCircle,
  FileText,
  Sparkles,
} from 'lucide-react'

const AGENT_LABELS: Record<string, string> = {
  supervisor: 'Supervisor',
  conversational_analyst: 'Research assistant',
  business_strategist: 'Business strategist',
  financial_auditor: 'Financial auditor',
  risk_analyst: 'Risk analyst',
  forecasting_analyst: 'Forecasting analyst',
  valuation_specialist: 'Valuation specialist',
  lead_synthesizer: 'Lead synthesizer',
}

interface ChatWindowProps {
  messages: Message[]
  isLoading: boolean
  activeMilestones?: AgentMilestone[]
  onOpenSource?: (ref: SourceRef) => void
  onSelectPrompt?: (prompt: string) => void
}

const META_ROW = 'flex items-center gap-2.5 font-mono text-[10.5px] uppercase tracking-[0.08em] text-faint mb-1.5'

export const ChatWindow: React.FC<ChatWindowProps> = ({
  messages,
  isLoading,
  activeMilestones = [],
  onOpenSource,
  onSelectPrompt,
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
      <div className="w-full max-w-[900px] mx-auto px-4 sm:px-5 py-6 space-y-6">
        {messages.length === 0 ? (
          <div className="flex flex-col justify-center min-h-[calc(100vh-240px)] py-6">
            <div className="w-11 h-11 rounded-xs bg-acc text-black flex items-center justify-center mb-5">
              <TrendingUp className="w-6 h-6 stroke-[2.2]" />
            </div>

            <h2 className="font-mono text-xl sm:text-2xl font-semibold tracking-[0.12em] text-fg-strong">
              RESEARCH TERMINAL
            </h2>
            <p className="text-dim text-sm max-w-xl mt-2 leading-relaxed">
              Run a multi-agent equity research pipeline on any ingested SEC filing: 3-statement audit, moat assessment,
              risk matrix, forecast and DCF valuation, with every figure cited back to the 10-K.
            </p>

            {onSelectPrompt && (
              <div className="mt-8">
                <QuickPrompts onSelectPrompt={onSelectPrompt} disabled={isLoading} />
              </div>
            )}
          </div>
        ) : (
        messages.map((message) => {
          const isUser = message.role === 'user'
          const isReport =
            message.content.includes('# ') &&
            (message.content.includes('Research Report') ||
              message.content.includes('Executive Summary') ||
              message.content.includes('Investment Thesis') ||
              message.content.includes('Valuation Dashboard'))

          return (
            <div key={message.id} className="w-full group">
              {isUser ? (
                /* Analyst Inquiry */
                <div className="flex flex-col items-end">
                  <div className="max-w-[92%] sm:max-w-[78%]">
                    <div className={`${META_ROW} justify-end`}>
                      <span className="font-tabular">{formatTime(message.timestamp)}</span>
                      <span className="font-semibold text-acc flex items-center gap-1.5">
                        <User className="w-3 h-3" />
                        Analyst inquiry
                      </span>
                    </div>

                    <div className="px-3.5 py-2.5 bg-raised border border-line-strong border-l-2 border-l-acc text-fg-strong text-sm leading-relaxed whitespace-pre-wrap">
                      {message.content}
                    </div>
                  </div>
                </div>
              ) : (
                /* Assistant Response */
                <div className="w-full">
                  <div className={META_ROW}>
                    {isReport ? (
                      <span className="font-semibold text-fg flex items-center gap-1.5">
                        <FileText className="w-3 h-3 text-acc" />
                        Research memorandum
                      </span>
                    ) : (
                      <span className="font-semibold text-acc flex items-center gap-1.5">
                        <Sparkles className="w-3 h-3" />
                        Research assistant
                      </span>
                    )}
                    <span className="font-tabular">{formatTime(message.timestamp)}</span>

                    {!message.isError && (
                      <button
                        onClick={() => copyToClipboard(message.id, message.content)}
                        className="ml-auto flex items-center gap-1.5 px-1.5 py-0.5 rounded-xs text-dim hover:text-fg-strong hover:bg-raised transition-colors cursor-pointer"
                        title={isReport ? 'Copy research memorandum' : 'Copy response'}
                      >
                        {copiedId === message.id ? (
                          <>
                            <Check className="w-3 h-3 text-pos" />
                            <span className="text-pos">Copied</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3 h-3" />
                            <span>{isReport ? 'Copy memo' : 'Copy'}</span>
                          </>
                        )}
                      </button>
                    )}
                  </div>

                  <div
                    className={`border ${
                      isReport ? 'px-4 py-5 sm:px-6 sm:py-6' : 'px-4 py-3.5 sm:px-5'
                    } ${
                      message.isError
                        ? 'bg-neg-bg border-neg-dim text-neg text-sm whitespace-pre-wrap'
                        : 'bg-panel border-line'
                    }`}
                  >
                    {message.isError && (
                      <div className="flex items-center gap-1.5 font-mono font-semibold mb-2 text-[11px] uppercase tracking-wider">
                        <AlertCircle className="w-3.5 h-3.5" />
                        <span>Inquiry execution error</span>
                      </div>
                    )}

                    {message.isError ? (
                      <div className="text-fg">{message.content}</div>
                    ) : (
                      <>
                        <MarkdownRenderer
                          content={message.content}
                          sources={message.sources}
                          onCite={onOpenSource}
                        />
                        <SourceList sources={message.sources ?? []} onOpen={onOpenSource} />
                      </>
                    )}
                  </div>
                </div>
              )}
            </div>
          )
        })
      )}

      {/* Live Multi-Agent Pipeline Progress */}
      {isLoading && (
        <div className="w-full bg-panel border border-line border-t-2 border-t-acc">
          <div className="flex items-center gap-2.5 px-4 py-3 border-b border-line font-mono text-[11px] uppercase tracking-[0.08em]">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-acc opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-acc"></span>
            </span>
            <span className="text-fg-strong">Multi-agent pipeline active</span>
            {activeMilestones.length > 0 && (
              <span className="ml-auto text-faint">
                {activeMilestones.length} milestone{activeMilestones.length !== 1 ? 's' : ''}
              </span>
            )}
          </div>

          {activeMilestones.length > 0 ? (
            <div>
              {activeMilestones.map((ms, index) => {
                const isLatest = index === activeMilestones.length - 1
                const label = AGENT_LABELS[ms.node] || ms.node.replace(/_/g, ' ')

                return (
                  <div
                    key={ms.id}
                    className={`grid grid-cols-[16px_1fr_auto] sm:grid-cols-[16px_170px_1fr_auto] gap-x-2.5 gap-y-0.5 items-start px-4 py-2.5 border-b border-[#1a1a1a] last:border-b-0 text-[12.5px] ${
                      isLatest ? 'bg-acc-bg' : ''
                    }`}
                  >
                    <div className="mt-0.5">
                      {isLatest ? (
                        <div className="w-3 h-3 border-2 border-acc border-t-transparent rounded-full animate-spin" />
                      ) : (
                        <Check className="w-3.5 h-3.5 text-pos" />
                      )}
                    </div>
                    <span
                      className={`font-mono text-[10.5px] uppercase tracking-wider pt-px ${
                        isLatest ? 'text-acc' : 'text-dim'
                      }`}
                    >
                      {label}
                    </span>
                    <span
                      className={`col-start-2 row-start-2 sm:col-start-auto sm:row-start-auto leading-snug ${
                        isLatest ? 'text-fg-strong' : 'text-fg'
                      }`}
                    >
                      {ms.message}
                    </span>
                    <span className="col-start-3 row-start-1 sm:col-start-auto sm:row-start-auto font-mono text-[10.5px] text-faint text-right font-tabular">
                      {formatTime(ms.timestamp)}
                    </span>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="flex items-center gap-2.5 px-4 py-3 text-dim text-xs">
              <div className="w-3 h-3 border-2 border-acc border-t-transparent rounded-full animate-spin shrink-0" />
              <span>Initializing multi-agent pipeline and triaging inquiry…</span>
            </div>
          )}
        </div>
      )}


      <div ref={bottomRef} />
      </div>
    </div>
  )
}
