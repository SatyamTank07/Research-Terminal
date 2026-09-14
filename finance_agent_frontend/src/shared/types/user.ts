export interface UserProfile {
  id: number
  username: string
  email: string | null
  full_name: string | null
  is_active: boolean
  created_at: string
}
