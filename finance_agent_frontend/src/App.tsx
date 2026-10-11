import { useState, useEffect } from 'react'
import { Header } from '@/shared/components/Header'
import { useBackendHealth } from '@/shared/hooks/useBackendHealth'
import { ConversationSidebar } from '@/features/conversations/ConversationSidebar'
import { useConversations } from '@/features/conversations/hooks/useConversations'
import { ChatWindow } from '@/features/chat/ChatWindow'
import { ChatInput } from '@/features/chat/ChatInput'
import { useChat } from '@/features/chat/hooks/useChat'
import { useDocuments } from '@/features/documents/hooks/useDocuments'
import { FilingViewerPanel } from '@/features/filing-viewer/FilingViewerPanel'
import { useFilingViewer } from '@/features/filing-viewer/hooks/useFilingViewer'

export function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(() =>
    typeof window !== 'undefined' ? window.innerWidth >= 768 : true
  )

  // 1. Centralized user identity & documents
  const { user } = useBackendHealth()
  const { documents, refreshDocuments } = useDocuments()

  // 2. Conversation management and URL state
  const {
    conversations,
    activeConversationId,
    setActiveConversationId,
    loadConversations,
    deleteConversationItem,
  } = useConversations()

  // 3. Active chat session, message sending, and loading states
  const { messages, activeMilestones, isLoading, sendMessage } = useChat({
    activeConversationId,
    onConversationUpdated: async (convId) => {
      await loadConversations(convId)
    },
  })
  const filingViewer = useFilingViewer()

  // Initial load of conversations
  useEffect(() => {
    loadConversations()
  }, [loadConversations])

  // Navigation handlers
  const handleSelectConversation = (convId: string) => {
    filingViewer.close()
    setActiveConversationId(convId)
    if (window.innerWidth < 768) {
      setIsSidebarOpen(false)
    }
  }

  const handleNewChat = () => {
    filingViewer.close()
    setActiveConversationId(null)
    if (window.innerWidth < 768) {
      setIsSidebarOpen(false)
    }
  }

  return (
    <div className="flex h-screen bg-[#070a12] text-slate-100 selection:bg-sky-500/30 selection:text-sky-200 overflow-hidden">
      {/* Left Research Dossiers Sidebar */}
      <ConversationSidebar
        conversations={conversations}
        activeConversationId={activeConversationId}
        isOpen={isSidebarOpen}
        onToggle={() => setIsSidebarOpen((prev) => !prev)}
        onSelectConversation={handleSelectConversation}
        onNewChat={handleNewChat}
        onDeleteConversation={deleteConversationItem}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-screen overflow-hidden min-w-0">
        {/* Top Header */}
        <Header
          user={user}
          onToggleSidebar={() => setIsSidebarOpen((prev) => !prev)}
          isSidebarOpen={isSidebarOpen}
          documentsCount={documents.length}
        />

        {/* Main Research Workspace */}
        <main className="flex-1 flex flex-col justify-between overflow-hidden relative">
          {/* Memorandum Feed */}
          <ChatWindow
            messages={messages}
            isLoading={isLoading}
            activeMilestones={activeMilestones}
            onOpenSource={filingViewer.openSource}
          />


          {/* Bottom Analyst Command Bar */}
          <ChatInput
            onSendMessage={sendMessage}
            isLoading={isLoading}
            onDocumentIngested={refreshDocuments}
          />
        </main>
      </div>

      {/* Right-hand 10-K viewer for clicked citations */}
      <FilingViewerPanel viewer={filingViewer} />
    </div>
  )
}

export default App
