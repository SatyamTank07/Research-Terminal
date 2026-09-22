import { API_BASE_URL } from '@/shared/config/env'
import { request } from '@/shared/services/apiClient'
import type {
  AgentMilestone,
  ChatMessageResponse,
  ChatResponse,
  Message,
  StreamingChatEvent,
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

export async function sendChatMessageStream(
  message: string,
  conversationId?: string,
  onMilestone?: (milestone: AgentMilestone) => void
): Promise<ChatResponse> {
  const url = `${API_BASE_URL}/chat/stream`
  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({
      message,
      conversation_id: conversationId || undefined,
    }),
  })

  if (!response.ok) {
    let errMsg = `Request failed (${response.status})`
    try {
      const errJson = await response.json()
      if (errJson.detail) {
        errMsg = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail)
      } else if (errJson.message) {
        errMsg = errJson.message
      }
    } catch {
      // Non-JSON response body
    }
    throw new Error(errMsg)
  }

  if (!response.body) {
    throw new Error('ReadableStream not supported by browser.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let finalResult: ChatResponse | null = null

  while (true) {
    const { value, done } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || trimmed.startsWith(':')) continue

      if (trimmed.startsWith('data: ')) {
        const jsonStr = trimmed.substring(6)
        try {
          const parsed = JSON.parse(jsonStr) as StreamingChatEvent
          if (parsed.type === 'status') {
            if (onMilestone) {
              onMilestone({
                id: crypto.randomUUID(),
                node: parsed.node,
                message: parsed.message,
                timestamp: new Date(),
                details: parsed.details,
              })
            }
          } else if (parsed.type === 'result') {
            finalResult = {
              response: parsed.response,
              conversation_id: parsed.conversation_id,
              message_id: parsed.message_id || crypto.randomUUID(),
              sources: parsed.sources || [],
            }
          } else if (parsed.type === 'error') {
            throw new Error(parsed.message || 'Stream encountered an error')
          }
        } catch (e) {
          if (e instanceof Error && (e.message.includes('Stream encountered an error') || !e.message.includes('JSON'))) {
            throw e
          }
          console.warn('Failed to parse SSE JSON line:', trimmed, e)
        }
      }
    }
  }

  if (!finalResult) {
    throw new Error('Research pipeline completed without emitting a final memorandum.')
  }

  return finalResult
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
