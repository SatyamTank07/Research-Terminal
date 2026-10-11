import React from 'react'
import { Button } from '@/shared/ui/button'
import type { BackendStatus } from '@/shared/types/common'
import type { UserProfile } from '@/shared/types/user'
import {
  TrendingUp,
  User,
  PanelLeft,
  FileText,
} from 'lucide-react'

interface HeaderProps {
  user?: UserProfile | null
  onToggleSidebar?: () => void
  isSidebarOpen?: boolean
  documentsCount?: number
  backendStatus?: BackendStatus
}

const STATUS_STYLE: Record<BackendStatus, { label: string; dot: string; text: string }> = {
  online: { label: 'ONLINE', dot: 'bg-pos shadow-[0_0_6px_var(--color-pos)]', text: 'text-pos' },
  offline: { label: 'OFFLINE', dot: 'bg-neg', text: 'text-neg' },
  checking: { label: 'CONNECTING', dot: 'bg-acc animate-pulse', text: 'text-acc' },
}

export const Header: React.FC<HeaderProps> = ({
  user,
  onToggleSidebar,
  isSidebarOpen,
  documentsCount,
  backendStatus,
}) => {
  const status = backendStatus ? STATUS_STYLE[backendStatus] : null

  return (
    <header className="@container w-full h-12 shrink-0 border-b border-line border-t-2 border-t-acc bg-surface px-4 sm:px-5 flex items-center justify-between gap-3 sticky top-0 z-30">
      {/* Brand & Expand Toggle (only shown when sidebar is collapsed) */}
      <div className="flex items-center gap-2.5 min-w-0">
        {!isSidebarOpen && onToggleSidebar && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggleSidebar}
            className="h-8 w-8 p-0"
            title="Open research dossiers"
          >
            <PanelLeft className="w-4 h-4" />
          </Button>
        )}

        <div className="w-7 h-7 rounded-xs bg-acc text-black flex items-center justify-center shrink-0">
          <TrendingUp className="w-4 h-4 stroke-[2.4]" />
        </div>
        <div className="min-w-0">
          <h1 className="font-mono text-[13px] font-semibold tracking-[0.12em] text-fg-strong leading-tight whitespace-nowrap">
            RESEARCH TERMINAL
          </h1>
          <p className="text-[11px] text-faint hidden @2xl:block leading-tight whitespace-nowrap">
            Institutional fundamental intelligence
          </p>
        </div>
      </div>

      {/* Right Controls: backend status, indexed filings, analyst profile */}
      <div className="flex items-center gap-2 font-mono text-[11px] whitespace-nowrap shrink-0">
        {status && (
          <div
            className={`hidden @xl:flex items-center gap-1.5 h-[26px] px-2 rounded-xs border border-line-strong ${status.text}`}
            title="FastAPI backend status"
          >
            <span className={`w-1.5 h-1.5 rounded-full ${status.dot}`} />
            {status.label}
          </div>
        )}

        {documentsCount !== undefined && documentsCount > 0 && (
          <div
            className="hidden @lg:flex items-center gap-1.5 h-[26px] px-2 rounded-xs bg-pos-bg border border-pos-dim text-pos"
            title={`${documentsCount} SEC 10-K/10-Q filings active in pgvector database`}
          >
            <FileText className="w-3 h-3" />
            <span className="font-semibold">
              {documentsCount} FILING{documentsCount > 1 ? 'S' : ''} READY
            </span>
          </div>
        )}

        {user ? (
          <div
            title={`Analyst: ${user.full_name || user.username} (${user.email || 'Authorized'})`}
            className="flex items-center gap-1.5 h-[26px] px-2 rounded-xs border border-line-strong text-dim"
          >
            <div className="w-4 h-4 rounded-xs bg-acc-bg text-acc flex items-center justify-center text-[10px] font-semibold">
              {(user.username || 'A')[0].toUpperCase()}
            </div>
            <span className="text-fg hidden @md:inline">{user.full_name || user.username}</span>
            <span className="text-faint hidden @3xl:inline">· ANALYST</span>
          </div>
        ) : (
          <div className="flex items-center gap-1.5 h-[26px] px-2 rounded-xs border border-line-strong text-dim">
            <User className="w-3.5 h-3.5" />
            <span>ANALYST</span>
          </div>
        )}
      </div>
    </header>
  )
}
