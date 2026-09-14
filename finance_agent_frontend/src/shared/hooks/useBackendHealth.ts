import { useState, useEffect, useCallback } from 'react'
import type { BackendStatus } from '@/shared/types/common'
import type { UserProfile } from '@/shared/types/user'
import { checkBackendHealth, getCurrentUser } from '@/shared/services/healthService'

export function useBackendHealth() {
  const [status, setStatus] = useState<BackendStatus>('checking')
  const [statusMessage, setStatusMessage] = useState<string>('Checking backend...')
  const [user, setUser] = useState<UserProfile | null>(null)

  const verifyHealth = useCallback(async () => {
    setStatus('checking')
    const [res, currentUser] = await Promise.all([
      checkBackendHealth(),
      getCurrentUser(),
    ])

    if (res.ok) {
      setStatus('online')
      setStatusMessage(res.message || 'Connected to FastAPI backend')
      setUser(currentUser)
    } else {
      setStatus('offline')
      setStatusMessage(res.message || 'Backend unreachable')
      setUser(null)
    }
  }, [])

  useEffect(() => {
    verifyHealth()
    const interval = setInterval(verifyHealth, 30000)
    return () => clearInterval(interval)
  }, [verifyHealth])

  return {
    status,
    statusMessage,
    user,
    verifyHealth,
    setStatus,
  }
}
