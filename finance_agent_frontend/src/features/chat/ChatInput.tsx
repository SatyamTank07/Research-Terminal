import React, { useState, useRef, useEffect } from 'react'
import { Button } from '@/shared/ui/button'
import { Textarea } from '@/shared/ui/textarea'
import { CircularProgress } from '@/shared/components/CircularProgress'
import { uploadFilingWithSSE, type IngestionProgressEvent } from '@/features/documents/documentService'
import {
  CornerDownLeft,
  ArrowUp,
  Paperclip,
  FileText,
  X,
} from 'lucide-react'

interface ChatInputProps {
  onSendMessage: (message: string) => void
  disabled?: boolean
  isLoading?: boolean
  onDocumentIngested?: () => void
  /** Text to load into the input (e.g. a quick-prompt prefix); a new nonce re-applies the same text. */
  prefill?: { text: string; nonce: number } | null
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  disabled,
  isLoading,
  onDocumentIngested,
  prefill,
}) => {
  const [input, setInput] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Document Upload & Circular Progress State
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null)
  const [uploadProgress, setUploadProgress] = useState<number>(0)
  const [uploadStage, setUploadStage] = useState<
    'idle' | 'uploaded' | 'uploading' | 'parsing' | 'chunking' | 'embedding' | 'storing' | 'completed' | 'error'
  >('idle')
  const [uploadMessage, setUploadMessage] = useState<string>('')

  useEffect(() => {
    if (!isLoading && textareaRef.current) {
      textareaRef.current.focus()
    }
  }, [isLoading])

  useEffect(() => {
    const el = textareaRef.current
    if (!prefill || !el) return
    setInput(prefill.text)
    el.focus()
    requestAnimationFrame(() => el.setSelectionRange(prefill.text.length, prefill.text.length))
  }, [prefill])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleSend = () => {
    const trimmed = input.trim()
    if (!trimmed || disabled || isLoading || uploadStage === 'uploading' || uploadStage === 'parsing' || uploadStage === 'embedding') return
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

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    // Reset input so the same file can be re-selected if desired
    e.target.value = ''

    setSelectedFileName(file.name)
    setUploadProgress(10)
    setUploadStage('uploading')
    setUploadMessage('Uploading...')

    try {
      await uploadFilingWithSSE(file, (event: IngestionProgressEvent) => {
        setUploadProgress(event.progress)
        setUploadStage(event.stage)
        setUploadMessage(event.message)
      })

      setUploadProgress(100)
      setUploadStage('completed')
      setUploadMessage('Ready for Analysis')

      if (onDocumentIngested) {
        onDocumentIngested()
      }
    } catch (err: unknown) {
      console.error('File upload error:', err)
      setUploadStage('error')
      setUploadMessage(err instanceof Error ? err.message : 'Upload failed')
    }
  }

  const handleClearFile = () => {
    setSelectedFileName(null)
    setUploadProgress(0)
    setUploadStage('idle')
    setUploadMessage('')
  }

  const isIngesting = ['uploading', 'parsing', 'chunking', 'embedding', 'storing'].includes(uploadStage)

  return (
    <div className="w-full max-w-[880px] mx-auto px-3 sm:px-5 pb-3.5 pt-2 shrink-0">
      <div className="relative border border-line-strong bg-panel focus-within:border-acc focus-within:ring-1 focus-within:ring-acc-dim transition-colors">
        {/* Hidden File Input for .htm only */}
        <input
          type="file"
          ref={fileInputRef}
          accept=".htm"
          onChange={handleFileSelect}
          className="hidden"
        />

        <div className="flex items-start gap-2 px-3.5 pt-2.5">
        <span className="font-mono text-sm font-semibold text-acc leading-[22px] pt-px select-none">&gt;</span>
        <Textarea
          ref={textareaRef}
          value={input}
          onChange={handleInputChange}
          onKeyDown={handleKeyDown}
          placeholder="Enter ticker, company name, or research inquiry (e.g. Analyze NVDA gross margin trajectory)..."
          disabled={disabled || isLoading}
          rows={1}
          className="min-h-[44px] max-h-[140px] border-0 bg-transparent px-0 py-0.5 font-mono text-[13px] text-fg-strong placeholder:text-faint focus-visible:ring-0 focus-visible:ring-offset-0 resize-none leading-[22px]"
        />
        </div>

        <div className="flex items-center justify-between py-1.5 pl-2 pr-1.5 border-t border-line mt-1 gap-2">
          {/* Left section: Upload trigger + File name & Circular Progress Bar */}
          <div className="flex items-center gap-2 min-w-0 flex-1 overflow-hidden">
            {/* Attachment Button */}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled || isIngesting}
              className="inline-flex items-center gap-1.5 px-1.5 py-1 rounded-xs font-mono text-[11px] text-dim hover:text-acc hover:bg-raised transition-colors cursor-pointer shrink-0 disabled:opacity-40"
              title="Upload SEC .htm filing (e.g. Form 10-K, 10-Q)"
            >
              <Paperclip className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">ATTACH .HTM</span>
            </button>

            {/* If a file is selected: Show file name + round circular progress bar right here! */}
            {selectedFileName ? (
              <div
                className={`inline-flex items-center gap-2 px-2 py-0.5 rounded-xs font-mono text-[11px] border max-w-full truncate transition-colors ${
                  uploadStage === 'error'
                    ? 'bg-neg-bg border-neg-dim text-neg'
                    : uploadStage === 'completed'
                    ? 'bg-pos-bg border-pos-dim text-pos'
                    : 'bg-acc-bg border-acc-dim text-fg'
                }`}
              >
                <FileText className="w-3.5 h-3.5 text-acc shrink-0" />
                <span
                  className="truncate max-w-[120px] sm:max-w-[200px]"
                  title={selectedFileName}
                >
                  {selectedFileName}
                </span>

                {/* Round Circular Progress Indicator */}
                <div className="flex items-center gap-1 shrink-0">
                  <CircularProgress
                    progress={uploadProgress}
                    size={18}
                    strokeWidth={2.5}
                    status={uploadStage}
                  />
                  <span className="text-[10px] font-tabular text-dim hidden sm:inline">
                    {uploadStage === 'completed' ? 'Indexed' : uploadStage === 'error' ? 'Error' : `${uploadProgress}%`}
                  </span>
                </div>

                {uploadMessage && (
                  <span className="text-[10px] text-dim truncate hidden md:inline max-w-[150px]" title={uploadMessage}>
                    • {uploadMessage}
                  </span>
                )}

                {/* Clear / Dismiss button */}
                <button
                  type="button"
                  onClick={handleClearFile}
                  disabled={isIngesting}
                  className="text-faint hover:text-fg-strong p-0.5 ml-0.5 rounded-xs cursor-pointer disabled:opacity-20"
                  title="Remove attached filing"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ) : (
              /* Default Helper Text */
              <div className="flex items-center gap-1.5 font-mono text-[10.5px] text-faint truncate">
                <CornerDownLeft className="w-3 h-3 shrink-0" />
                <span className="truncate">ENTER to run · SHIFT+ENTER new line</span>
              </div>
            )}
          </div>

          {/* Right section: Analyze Send Button */}
          <Button
            onClick={handleSend}
            disabled={!input.trim() || disabled || isLoading || isIngesting}
            size="sm"
            className="h-7 px-3 gap-1.5 text-[11.5px] shrink-0"
          >
            {isLoading ? (
              <>
                <div className="w-3 h-3 border-2 border-black border-t-transparent rounded-full animate-spin" />
                <span>RUNNING</span>
              </>
            ) : isIngesting ? (
              <>
                <div className="w-3 h-3 border-2 border-black border-t-transparent rounded-full animate-spin" />
                <span>INGESTING</span>
              </>
            ) : (
              <>
                <span>ANALYZE</span>
                <ArrowUp className="w-3.5 h-3.5 stroke-[2.5]" />
              </>
            )}
          </Button>
        </div>
      </div>
      <p className="font-mono text-[10px] text-center text-[#444] mt-1.5 tracking-wider">
        CONFIDENTIAL · FOR INSTITUTIONAL EQUITY RESEARCH ONLY
      </p>
    </div>
  )
}
