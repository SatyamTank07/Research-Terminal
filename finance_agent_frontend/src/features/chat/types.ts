import type { SourceRef } from '@/shared/types/citations'

export type MessageRole = 'user' | 'assistant' | 'system'

export interface Message {
  id: string
  conversation_id?: string
  role: MessageRole
  content: string
  timestamp: Date
  isError?: boolean
  sources?: SourceRef[]
}

export interface ChatRequest {
  message: string
  conversation_id?: string
  stream?: boolean
}

export interface ChatResponse {
  response: string
  conversation_id: string
  message_id: string
  sources?: SourceRef[]
  /** Structured Final10KResearchReport (streamed research routes only). */
  final_report?: Record<string, any>
  /** Conversation session state persisted by the backend after this turn (streamed responses only). */
  updated_session_state?: Record<string, any>
}

export interface ChatMessageResponse {
  id: string
  conversation_id: string
  user_id?: number | null
  role: MessageRole
  content: string
  sources?: SourceRef[]
  token_count?: number | null
  is_error?: boolean
  created_at: string
}

export interface AgentMilestone {
  id: string
  node: string
  message: string
  timestamp: Date
  details?: Record<string, any>
}

export type StreamingChatEvent =
  | {
      type: 'status'
      node: string
      message: string
      details?: Record<string, any>
    }
  | {
      type: 'token'
      delta: string
    }
  | {
      type: 'result'
      response: string
      conversation_id: string
      message_id?: string
      sources?: SourceRef[]
      final_report?: Record<string, any>
      updated_session_state?: Record<string, any>
    }
  | {
      type: 'error'
      message: string
      conversation_id?: string
      message_id?: string
    }
