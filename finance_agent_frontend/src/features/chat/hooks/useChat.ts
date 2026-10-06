import { useState, useCallback, useEffect } from 'react'
import type { AgentMilestone, Message } from '../types'
import {
  sendChatMessageStream,
  getConversationMessages,
  clearConversationMessages,
} from '../chatService'

interface UseChatOptions {
  activeConversationId: string | null
  onConversationUpdated?: (conversationId: string) => Promise<void> | void
}

export function useChat({
  activeConversationId,
  onConversationUpdated,
}: UseChatOptions) {
  const [messages, setMessages] = useState<Message[]>([])
  const [activeMilestones, setActiveMilestones] = useState<AgentMilestone[]>([])
  const [isLoading, setIsLoading] = useState(false)

  // Load messages whenever activeConversationId changes
  useEffect(() => {
    if (!activeConversationId) {
      setMessages([])
      return
    }

    let isMounted = true
    const fetchMessages = async () => {
      try {
        const msgs = await getConversationMessages(activeConversationId)
        if (isMounted) {
          setMessages(msgs)
        }
      } catch (err) {
        console.error('Failed to load conversation messages:', err)
      }
    }

    fetchMessages()
    return () => {
      isMounted = false
    }
  }, [activeConversationId])

  // Send message
  const sendMessage = useCallback(
    async (userPrompt: string) => {
      if (!userPrompt.trim() || isLoading) return

      const tempUserMessage: Message = {
        id: crypto.randomUUID(),
        role: 'user',
        content: userPrompt,
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, tempUserMessage])
      setActiveMilestones([])
      setIsLoading(true)

      const streamingMsgId = crypto.randomUUID()

      try {
        const replyData = await sendChatMessageStream(
          userPrompt,
          activeConversationId || undefined,
          (milestone) => {
            setActiveMilestones((prev) => [...prev, milestone])
          },
          (delta) => {
            setMessages((prev) => {
              const idx = prev.findIndex((m) => m.id === streamingMsgId)
              if (idx !== -1) {
                const next = [...prev]
                next[idx] = {
                  ...next[idx],
                  content: next[idx].content + delta,
                }
                return next
              }
              return [
                ...prev,
                {
                  id: streamingMsgId,
                  role: 'assistant',
                  content: delta,
                  timestamp: new Date(),
                },
              ]
            })
          }
        )

        const agentMessage: Message = {
          id: replyData.message_id || streamingMsgId,
          conversation_id: replyData.conversation_id,
          role: 'assistant',
          content: replyData.response,
          timestamp: new Date(),
          sources: replyData.sources || [],
        }

        setMessages((prev) => {
          const filtered = prev.filter((m) => m.id !== streamingMsgId)
          return [...filtered, agentMessage]
        })

        if (onConversationUpdated) {
          await onConversationUpdated(replyData.conversation_id)
        }
      } catch (err: unknown) {
        const errorMessage =
          err instanceof Error
            ? err.message
            : 'Unable to communicate with the finance agent. Ensure the backend is running.'

        const errorReply: Message = {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: errorMessage,
          timestamp: new Date(),
          isError: true,
        }
        setMessages((prev) => {
          const filtered = prev.filter((m) => m.id !== streamingMsgId)
          return [...filtered, errorReply]
        })
      } finally {
        setIsLoading(false)
        setActiveMilestones([])
      }
    },
    [activeConversationId, isLoading, onConversationUpdated]
  )

  // Clear messages within active conversation
  const clearChat = useCallback(async () => {
    if (!activeConversationId) {
      setMessages([])
      setActiveMilestones([])
      return
    }

    try {
      await clearConversationMessages(activeConversationId)
      setMessages([])
      setActiveMilestones([])
      if (onConversationUpdated) {
        await onConversationUpdated(activeConversationId)
      }
    } catch (err) {
      console.error('Failed to clear conversation messages:', err)
      setMessages([])
      setActiveMilestones([])
    }
  }, [activeConversationId, onConversationUpdated])

  return {
    messages,
    activeMilestones,
    isLoading,
    sendMessage,
    clearChat,
    setMessages,
  }
}

