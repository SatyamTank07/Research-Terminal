export interface Conversation {
  id: string
  user_id: number
  title: string
  is_archived: boolean
  created_at: string
  updated_at: string
  message_count?: number
}
