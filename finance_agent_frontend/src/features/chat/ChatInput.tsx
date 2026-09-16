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
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSendMessage,
  disabled,
  isLoading,
  onDocumentIngested,
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
    <div className="w-full max-w-4xl mx-auto px-4 pb-5 pt-2">
      <div className="relative rounded-xl border border-slate-800 bg-[#0a0f1d] shadow-2xl p-2.5 focus-within:border-sky-500/70 focus-within:ring-1 focus-within:ring-sky-500/20 transition-all">
        {/* Hidden File Input for .htm only */}
        <input
          type="file"
          ref={fileInputRef}
          accept=".htm"
          onChange={handleFileSelect}
          className="hidden"
        />

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

        <div className="flex items-center justify-between pt-2 px-1 border-t border-slate-800/80 mt-1 gap-2">
          {/* Left section: Upload trigger + File name & Circular Progress Bar */}
          <div className="flex items-center gap-2 min-w-0 flex-1 overflow-hidden">
            {/* Attachment Button */}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled || isIngesting}
              className="inline-flex items-center gap-1 px-2 py-1 rounded text-slate-400 hover:text-sky-300 hover:bg-slate-800/60 transition-colors cursor-pointer text-xs shrink-0 disabled:opacity-40"
              title="Upload SEC .htm filing (e.g. Form 10-K, 10-Q)"
            >
              <Paperclip className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Attach .htm</span>
            </button>

            {/* If a file is selected: Show file name + round circular progress bar right here! */}
            {selectedFileName ? (
              <div
                className={`inline-flex items-center gap-2 px-2.5 py-1 rounded-md text-xs border max-w-full truncate transition-all ${
                  uploadStage === 'error'
                    ? 'bg-red-950/30 border-red-800/60 text-red-300'
                    : uploadStage === 'completed'
                    ? 'bg-emerald-950/20 border-emerald-800/50 text-emerald-300'
                    : 'bg-slate-900 border-sky-500/40 text-slate-200'
                }`}
              >
                <FileText className="w-3.5 h-3.5 text-sky-400 shrink-0" />
                <span
                  className="truncate font-mono text-[11px] max-w-[120px] sm:max-w-[200px]"
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
                  <span className="text-[10px] font-tabular text-slate-400 hidden sm:inline">
                    {uploadStage === 'completed' ? 'Indexed' : uploadStage === 'error' ? 'Error' : `${uploadProgress}%`}
                  </span>
                </div>

                {uploadMessage && (
                  <span className="text-[10px] text-slate-400 truncate hidden md:inline max-w-[150px]" title={uploadMessage}>
                    • {uploadMessage}
                  </span>
                )}

                {/* Clear / Dismiss button */}
                <button
                  type="button"
                  onClick={handleClearFile}
                  disabled={isIngesting}
                  className="text-slate-500 hover:text-slate-300 p-0.5 ml-0.5 rounded cursor-pointer disabled:opacity-20"
                  title="Remove attached filing"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ) : (
              /* Default Helper Text */
              <div className="flex items-center gap-1.5 text-[11px] text-slate-500 truncate">
                <CornerDownLeft className="w-3 h-3 text-slate-500 shrink-0" />
                <span className="truncate">Enter to execute inquiry, Shift+Enter for new line</span>
              </div>
            )}
          </div>

          {/* Right section: Analyze Send Button */}
          <Button
            onClick={handleSend}
            disabled={!input.trim() || disabled || isLoading || isIngesting}
            size="sm"
            className="h-7.5 px-3 rounded-md gap-1.5 text-xs font-semibold bg-sky-500 hover:bg-sky-400 text-slate-950 shadow-sm shadow-sky-500/20 disabled:opacity-40 transition-all cursor-pointer shrink-0"
          >
            {isLoading ? (
              <>
                <div className="w-3 h-3 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                <span>Compiling...</span>
              </>
            ) : isIngesting ? (
              <>
                <div className="w-3 h-3 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                <span>Ingesting...</span>
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
