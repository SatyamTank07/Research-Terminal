export type MessageRole = 'user' | 'assistant' | 'system'

export interface SourceItem {
  title?: string
  url?: string
  snippet?: string
}

export interface Message {
  id: string
  conversation_id?: string
  role: MessageRole
  content: string
  timestamp: Date
  isError?: boolean
  sources?: SourceItem[]
}

export interface ChatRequest {
  message: string
  conversation_id?: string
}

export interface ChatResponse {
  response: string
  conversation_id: string
  message_id: string
  sources?: SourceItem[]
}

export interface ChatMessageResponse {
  id: string
  conversation_id: string
  user_id?: number | null
  role: MessageRole
  content: string
  sources?: SourceItem[]
  token_count?: number | null
  is_error?: boolean
  created_at: string
}
