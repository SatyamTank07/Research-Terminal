import { request } from '@/shared/services/apiClient'
import type { Conversation } from './types'

export async function getConversations(): Promise<Conversation[]> {
  return request<Conversation[]>('/conversations')
}

export async function createConversation(title?: string): Promise<Conversation> {
  return request<Conversation>('/conversations', {
    method: 'POST',
    body: JSON.stringify({ title: title || 'New Chat' }),
  })
}

export async function deleteConversation(conversationId: string): Promise<void> {
  return request<void>(`/conversations/${conversationId}`, {
    method: 'DELETE',
  })
}

export async function updateConversationTitle(
  conversationId: string,
  title: string
): Promise<Conversation> {
  return request<Conversation>(`/conversations/${conversationId}`, {
    method: 'PATCH',
    body: JSON.stringify({ title }),
  })
}
