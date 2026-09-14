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
          className="fixed inset-0 bg-black/70 backdrop-blur-xs z-40 md:hidden"
          aria-hidden="true"
        />
      )}

      {/* Sidebar container */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-40 flex flex-col w-72 bg-[#090d18] border-r border-slate-800/80 transition-all duration-300 ease-in-out shrink-0 ${
          isOpen
            ? 'translate-x-0'
            : '-translate-x-full md:translate-x-0 md:w-0 md:border-r-0 md:overflow-hidden'
        }`}
      >
        {/* Sidebar Header */}
        <div className="p-3 border-b border-slate-800/80 flex items-center justify-between gap-2">
          <Button
            onClick={onNewChat}
            className="flex-1 justify-start gap-2 bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold h-9 shadow-sm shadow-sky-500/20 cursor-pointer text-xs"
          >
            <Plus className="w-4 h-4 stroke-[2.5]" />
            <span>New Dossier</span>
          </Button>

          <Button
            variant="ghost"
            size="sm"
            onClick={onToggle}
            className="h-9 w-9 p-0 text-slate-400 hover:text-slate-100 hover:bg-slate-800/60 cursor-pointer"
            title="Collapse dossiers"
          >
            <PanelLeftClose className="w-4 h-4" />
          </Button>
        </div>

        {/* Dossiers List */}
        <div className="flex-1 overflow-y-auto p-2 space-y-1">
          <div className="px-2 py-1.5 text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
            <span>Research Dossiers</span>
          </div>

          {conversations.length === 0 ? (
            <div className="p-6 text-center text-slate-500 text-xs">
              <Archive className="w-5 h-5 mx-auto mb-2 text-slate-600" />
              <p>No research dossiers recorded.</p>
              <p className="text-[11px] text-slate-500 mt-1">Initiate coverage to populate archives.</p>
            </div>
          ) : (
            conversations.map((conv) => {
              const isActive = conv.id === activeConversationId
              return (
                <div
                  key={conv.id}
                  onClick={() => onSelectConversation(conv.id)}
                  className={`group relative flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-sm transition-all cursor-pointer ${
                    isActive
                      ? 'bg-sky-500/10 text-white font-medium border border-sky-500/30'
                      : 'text-slate-300 hover:bg-slate-800/50 hover:text-white'
                  }`}
                >
                  <FileText
                    className={`w-3.5 h-3.5 shrink-0 transition-colors ${
                      isActive ? 'text-sky-400' : 'text-slate-500 group-hover:text-slate-300'
                    }`}
                  />

                  <div className="flex-1 min-w-0">
                    <p className="truncate text-xs leading-snug">{conv.title}</p>
                    <div className="flex items-center gap-1 text-[10px] text-slate-500 mt-0.5 font-tabular">
                      <Clock className="w-2.5 h-2.5" />
                      <span>{formatRelativeTime(conv.updated_at)}</span>
                    </div>
                  </div>

                  {/* Delete conversation button */}
                  <button
                    onClick={(e) => onDeleteConversation(conv.id, e)}
                    className="opacity-0 group-hover:opacity-100 p-1.5 rounded-md text-slate-400 hover:text-red-400 hover:bg-red-950/40 transition-all cursor-pointer"
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
