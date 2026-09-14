import { useState, useEffect, useCallback } from 'react'
import { getDocuments, type IngestedDocument } from '../documentService'

export function useDocuments() {
  const [documents, setDocuments] = useState<IngestedDocument[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadDocuments = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const data = await getDocuments()
      setDocuments(data)
    } catch (err) {
      console.error('Failed to load documents:', err)
      setError(err instanceof Error ? err.message : 'Failed to fetch documents')
    } finally {
      setIsLoading(false)
    }
  }, [])

  useEffect(() => {
    loadDocuments()
  }, [loadDocuments])

  return {
    documents,
    isLoading,
    error,
    refreshDocuments: loadDocuments,
  }
}
