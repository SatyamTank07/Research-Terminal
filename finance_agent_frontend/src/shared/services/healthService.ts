import { request } from './apiClient'
import type { UserProfile } from '@/shared/types/user'
import type { HealthCheckResponse } from '@/shared/types/common'

export async function checkBackendHealth(): Promise<HealthCheckResponse> {
  try {
    const data = await request<{ message?: string }>('/')
    return { ok: true, message: data.message }
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : 'Connection failed'
    return { ok: false, message }
  }
}

export async function getCurrentUser(): Promise<UserProfile | null> {
  try {
    return await request<UserProfile>('/user/me')
  } catch {
    return null
  }
}
