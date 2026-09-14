import { useState, useCallback, useRef } from 'react'
import type { Conversation } from '../types'
import {
  getConversations,
  deleteConversation,
} from '../conversationService'

export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null)
  const isInitialMount = useRef(true)

  // Sync URL query param ?c=<id>
  const syncUrlQuery = useCallback((conversationId: string | null) => {
    const url = new URL(window.location.href)
    if (conversationId) {
      url.searchParams.set('c', conversationId)
    } else {
      url.searchParams.delete('c')
    }
    window.history.replaceState({}, '', url.toString())
  }, [])

  // Select a conversation and sync URL
  const selectConversation = useCallback((convId: string | null) => {
    setActiveConversationId(convId)
    syncUrlQuery(convId)
  }, [syncUrlQuery])

  // Fetch conversations list and optionally activate a target or the latest thread
  const loadConversations = useCallback(async (preferConvId?: string) => {
    try {
      const list = await getConversations()
      setConversations(list)

      const urlParamId = new URLSearchParams(window.location.search).get('c')
      const targetId = preferConvId || urlParamId

      if (targetId && list.some((c) => c.id === targetId)) {
        selectConversation(targetId)
        return targetId
      } else if (list.length > 0 && !targetId && isInitialMount.current) {
        // On initial page load, activate the most recent conversation if present
        selectConversation(list[0].id)
        return list[0].id
      }
    } catch (err) {
      console.error('Failed to load conversations:', err)
    } finally {
      isInitialMount.current = false
    }
    return null
  }, [selectConversation])

  // Delete a conversation thread
  const deleteConversationItem = useCallback(async (id: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation()
    try {
      await deleteConversation(id)
      setConversations((prev) => {
        const remaining = prev.filter((c) => c.id !== id)
        if (activeConversationId === id) {
          if (remaining.length > 0) {
            selectConversation(remaining[0].id)
          } else {
            selectConversation(null)
          }
        }
        return remaining
      })
    } catch (err) {
      console.error('Failed to delete conversation:', err)
    }
  }, [activeConversationId, selectConversation])

  return {
    conversations,
    activeConversationId,
    setActiveConversationId: selectConversation,
    loadConversations,
    deleteConversationItem,
  }
}
