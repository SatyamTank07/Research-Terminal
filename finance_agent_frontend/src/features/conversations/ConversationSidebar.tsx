import React from 'react'
import type { Conversation } from './types'
import { Button } from '@/shared/ui/button'
import {
  Plus,
  FileText,
  Trash2,
  Clock,
  Archive,
  PanelLeftClose,
} from 'lucide-react'

interface ConversationSidebarProps {
  conversations: Conversation[]
  activeConversationId: string | null
  isOpen: boolean
  onToggle: () => void
  onSelectConversation: (id: string) => void
  onNewChat: () => void
  onDeleteConversation: (id: string, e: React.MouseEvent) => void
}

function formatRelativeTime(dateString: string): string {
  try {
    const date = new Date(dateString)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMins / 60)
    const diffDays = Math.floor(diffHours / 24)

    if (diffMins < 1) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays === 1) return 'Yesterday'
    if (diffDays < 7) return `${diffDays}d ago`
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  } catch {
    return ''
  }
}

export const ConversationSidebar: React.FC<ConversationSidebarProps> = ({
  conversations,
  activeConversationId,
  isOpen,
  onToggle,
  onSelectConversation,
  onNewChat,
  onDeleteConversation,
}) => {
  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          onClick={onToggle}
          className="fixed inset-0 bg-black/70 z-40 md:hidden"
          aria-hidden="true"
        />
      )}

      {/* Sidebar container */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-40 flex flex-col w-68 bg-surface border-r border-line transition-all duration-300 ease-in-out shrink-0 ${
          isOpen
            ? 'translate-x-0'
            : '-translate-x-full md:translate-x-0 md:w-0 md:border-r-0 md:overflow-hidden'
        }`}
      >
        {/* Sidebar Header */}
        <div className="p-2.5 border-b border-line flex items-center justify-between gap-1.5">
          <Button
            onClick={onNewChat}
            className="flex-1 justify-start gap-2 h-[34px] text-xs"
          >
            <Plus className="w-4 h-4 stroke-[2.5]" />
            <span>NEW DOSSIER</span>
          </Button>

          <Button
            variant="ghost"
            size="sm"
            onClick={onToggle}
            className="h-[34px] w-[34px] p-0"
            title="Collapse dossiers"
          >
            <PanelLeftClose className="w-4 h-4" />
          </Button>
        </div>

        {/* Dossiers List */}
        <div className="flex-1 overflow-y-auto px-1.5 pb-2">
          <div className="px-2 pt-3 pb-1.5 font-mono text-[10.5px] text-faint uppercase tracking-[0.1em] flex justify-between">
            <span>Research Dossiers</span>
            {conversations.length > 0 && <span>{conversations.length}</span>}
          </div>

          {conversations.length === 0 ? (
            <div className="p-6 text-center text-faint text-xs">
              <Archive className="w-5 h-5 mx-auto mb-2" />
              <p>No research dossiers recorded.</p>
              <p className="text-[11px] mt-1">Initiate coverage to populate archives.</p>
            </div>
          ) : (
            conversations.map((conv) => {
              const isActive = conv.id === activeConversationId
              return (
                <div
                  key={conv.id}
                  onClick={() => onSelectConversation(conv.id)}
                  className={`group relative flex items-center gap-2.5 px-2.5 py-2 border-l-2 text-sm transition-colors cursor-pointer ${
                    isActive
                      ? 'bg-acc-bg border-acc text-fg-strong'
                      : 'border-transparent text-fg hover:bg-raised hover:text-fg-strong'
                  }`}
                >
                  <FileText
                    className={`w-3.5 h-3.5 shrink-0 transition-colors ${
                      isActive ? 'text-acc' : 'text-faint group-hover:text-dim'
                    }`}
                  />

                  <div className="flex-1 min-w-0">
                    <p className="truncate text-[12.5px] leading-snug">{conv.title}</p>
                    <div className="flex items-center gap-1 font-mono text-[10px] text-faint mt-0.5 font-tabular">
                      <Clock className="w-2.5 h-2.5" />
                      <span>{formatRelativeTime(conv.updated_at)}</span>
                    </div>
                  </div>

                  {/* Delete conversation button */}
                  <button
                    onClick={(e) => onDeleteConversation(conv.id, e)}
                    className="opacity-0 group-hover:opacity-100 p-1.5 rounded-xs text-faint hover:text-neg hover:bg-neg-bg transition-all cursor-pointer"
                    title="Delete dossier"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              )
            })
          )}
        </div>
      </aside>
    </>
  )
}
