import { useState, useCallback, useEffect } from 'react'
import type { Message } from '../types'
import {
  sendChatMessage,
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
      setIsLoading(true)

      try {
        const replyData = await sendChatMessage(
          userPrompt,
          activeConversationId || undefined
        )

        const agentMessage: Message = {
          id: replyData.message_id || crypto.randomUUID(),
          conversation_id: replyData.conversation_id,
          role: 'assistant',
          content: replyData.response,
          timestamp: new Date(),
          sources: replyData.sources || [],
        }

        setMessages((prev) => [...prev, agentMessage])

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
        setMessages((prev) => [...prev, errorReply])
      } finally {
        setIsLoading(false)
      }
    },
    [activeConversationId, isLoading, onConversationUpdated]
  )

  // Clear messages within active conversation
  const clearChat = useCallback(async () => {
    if (!activeConversationId) {
      setMessages([])
      return
    }

    try {
      await clearConversationMessages(activeConversationId)
      setMessages([])
      if (onConversationUpdated) {
        await onConversationUpdated(activeConversationId)
      }
    } catch (err) {
      console.error('Failed to clear conversation messages:', err)
      setMessages([])
    }
  }, [activeConversationId, onConversationUpdated])

  return {
    messages,
    isLoading,
    sendMessage,
    clearChat,
    setMessages,
  }
}
