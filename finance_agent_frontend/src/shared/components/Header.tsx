import React from 'react'
import { Button } from '@/shared/ui/button'
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
}

export const Header: React.FC<HeaderProps> = ({
  user,
  onToggleSidebar,
  isSidebarOpen,
  documentsCount,
}) => {
  return (
    <header className="w-full border-b border-slate-800/80 bg-[#070a12]/90 backdrop-blur-md px-4 sm:px-6 py-2.5 flex items-center justify-between gap-3 sticky top-0 z-30">
      {/* Brand & Expand Toggle (only shown when sidebar is collapsed) */}
      <div className="flex items-center gap-2.5">
        {!isSidebarOpen && onToggleSidebar && (
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggleSidebar}
            className="h-8 w-8 p-0 text-slate-400 hover:text-white hover:bg-slate-800/60 cursor-pointer"
            title="Open research dossiers"
          >
            <PanelLeft className="w-4 h-4" />
          </Button>
        )}

        <div className="w-8 h-8 rounded-lg bg-sky-500/10 border border-sky-500/30 flex items-center justify-center text-sky-400 shadow-sm shadow-sky-500/10">
          <TrendingUp className="w-4 h-4 stroke-[2.2]" />
        </div>
        <div>
          <h1 className="text-sm sm:text-base font-semibold tracking-tight text-white">
            Research Terminal
          </h1>
          <p className="text-[11px] text-slate-400 hidden sm:block">
            Institutional Fundamental Intelligence & Market Coverage
          </p>
        </div>
      </div>

      {/* Right Controls: Indexed Filings Badge + Analyst Profile */}
      <div className="flex items-center gap-2.5">
        {documentsCount !== undefined && documentsCount > 0 && (
          <div
            className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-950/30 border border-emerald-800/60 text-xs text-emerald-300 shadow-xs"
            title={`${documentsCount} SEC 10-K/10-Q filings active in pgvector database`}
          >
            <FileText className="w-3.5 h-3.5 text-emerald-400" />
            <span className="font-semibold">{documentsCount} Filing{documentsCount > 1 ? 's' : ''} Ready</span>
          </div>
        )}

        {user ? (
          <div
            title={`Analyst: ${user.full_name || user.username} (${user.email || 'Authorized'})`}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-xs text-slate-300"
          >
            <div className="w-4 h-4 rounded-full bg-sky-500/20 text-sky-400 flex items-center justify-center text-[10px] font-semibold">
              {(user.username || 'A')[0].toUpperCase()}
            </div>
            <span className="font-medium text-slate-200">{user.full_name || user.username}</span>
            <span className="text-[10px] text-slate-500 hidden md:inline">• Analyst</span>
          </div>
        ) : (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-xs text-slate-300">
            <User className="w-3.5 h-3.5 text-slate-400" />
            <span className="font-medium text-slate-300">Senior Research Analyst</span>
          </div>
        )}
      </div>
    </header>
  )
}
