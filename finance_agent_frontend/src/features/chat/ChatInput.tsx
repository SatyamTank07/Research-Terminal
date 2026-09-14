import React, { useState, useRef, useEffect } from 'react'
import { Button } from '@/shared/ui/button'
import { Textarea } from '@/shared/ui/textarea'
import { CornerDownLeft, ArrowUp } from 'lucide-react'

interface ChatInputProps {
  onSendMessage: (message: string) => void
  disabled?: boolean
  isLoading?: boolean
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  disabled,
  isLoading,
}) => {
  const [input, setInput] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (!isLoading && textareaRef.current) {
      textareaRef.current.focus()
    }
  }, [isLoading])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleSend = () => {
    const trimmed = input.trim()
    if (!trimmed || disabled || isLoading) return
    onSendMessage(trimmed)
    setInput('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    const target = e.target
    target.style.height = 'auto'
    target.style.height = `${Math.min(target.scrollHeight, 140)}px`
  }

  return (
    <div className="w-full max-w-4xl mx-auto px-4 pb-5 pt-2">
      <div className="relative rounded-xl border border-slate-800 bg-[#0a0f1d] shadow-2xl p-2.5 focus-within:border-sky-500/70 focus-within:ring-1 focus-within:ring-sky-500/20 transition-all">
        <Textarea
          ref={textareaRef}
          value={input}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder="Enter ticker, company name, or research inquiry (e.g. Analyze NVDA gross margin trajectory)..."
          disabled={disabled || isLoading}
          rows={1}
          className="min-h-[48px] max-h-[140px] border-0 bg-transparent p-2 text-xs sm:text-sm text-slate-100 placeholder:text-slate-500 focus-visible:ring-0 focus-visible:ring-offset-0 resize-none leading-relaxed"
        />

        <div className="flex items-center justify-between pt-2 px-1 border-t border-slate-800/80 mt-1">
          <div className="flex items-center gap-1.5 text-[11px] text-slate-500">
            <CornerDownLeft className="w-3 h-3 text-slate-500" />
            <span>Enter to execute inquiry, Shift+Enter for new line</span>
          </div>

          <Button
            onClick={handleSend}
            disabled={!input.trim() || disabled || isLoading}
            size="sm"
            className="h-7.5 px-3 rounded-md gap-1.5 text-xs font-semibold bg-sky-500 hover:bg-sky-400 text-slate-950 shadow-sm shadow-sky-500/20 disabled:opacity-40 transition-all cursor-pointer"
          >
            {isLoading ? (
              <>
                <div className="w-3 h-3 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                <span>Compiling...</span>
              </>
            ) : (
              <>
                <span>Analyze</span>
                <ArrowUp className="w-3.5 h-3.5 stroke-[2.5]" />
              </>
            )}
          </Button>
        </div>
      </div>
      <p className="text-[10px] text-center text-slate-500 mt-2 tracking-wide">
        Confidential • For Institutional Equity Research and Fundamental Analysis Only.
      </p>
    </div>
  )
}
