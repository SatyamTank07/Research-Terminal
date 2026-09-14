import { request } from '@/shared/services/apiClient'
import type {
  ChatMessageResponse,
  ChatResponse,
  Message,
} from './types'

export async function sendChatMessage(
  message: string,
  conversationId?: string
): Promise<ChatResponse> {
  return request<ChatResponse>('/chat', {
    method: 'POST',
    body: JSON.stringify({
      message,
      conversation_id: conversationId || undefined,
    }),
  })
}

export async function getConversationMessages(
  conversationId: string
): Promise<Message[]> {
  const rawMessages = await request<ChatMessageResponse[]>(
    `/conversations/${conversationId}/messages`
  )
  return rawMessages.map((m) => ({
    id: m.id,
    conversation_id: m.conversation_id,
    role: m.role,
    content: m.content,
    timestamp: new Date(m.created_at),
    isError: m.is_error,
    sources: m.sources || [],
  }))
}

export async function clearConversationMessages(
  conversationId: string
): Promise<void> {
  return request<void>(`/conversations/${conversationId}/messages`, {
    method: 'DELETE',
  })
}
