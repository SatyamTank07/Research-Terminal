export type BackendStatus = 'checking' | 'online' | 'offline'

export interface HealthCheckResponse {
  ok: boolean
  message?: string
}
