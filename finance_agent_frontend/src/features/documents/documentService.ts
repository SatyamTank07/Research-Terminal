import { API_BASE_URL } from '@/shared/config/env'
import { request } from '@/shared/services/apiClient'

export interface IngestedDocument {
  id: string
  ticker: string
  company_name: string
  filing_type: string
  fiscal_year: number
  period_end_date: string | null
  source_filename: string
  total_chunks: number
  created_at: string
}

export interface IngestionProgressEvent {
  stage: 'uploaded' | 'parsing' | 'chunking' | 'embedding' | 'storing' | 'completed' | 'error'
  progress: number
  message: string
  data?: any
}

export async function uploadFilingWithSSE(
  file: File,
  onProgress: (event: IngestionProgressEvent) => void
): Promise<IngestionProgressEvent> {
  const url = `${API_BASE_URL}/api/documents/upload`
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(url, {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    let errMsg = `Upload failed with status ${response.status}`
    try {
      const errJson = await response.json()
      if (errJson.detail) errMsg = errJson.detail
    } catch {
      // Non-JSON error
    }
    throw new Error(errMsg)
  }

  if (!response.body) {
    throw new Error('ReadableStream not supported by browser.')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let lastEvent: IngestionProgressEvent = {
    stage: 'uploaded',
    progress: 10,
    message: 'File received...',
  }
  let buffer = ''

  while (true) {
    const { value, done } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || trimmed.startsWith(':')) continue // Skip empty or keep-alive lines

      if (trimmed.startsWith('data: ')) {
        const jsonStr = trimmed.substring(6)
        try {
          const parsed = JSON.parse(jsonStr) as IngestionProgressEvent
          lastEvent = parsed
          onProgress(parsed)
        } catch (e) {
          console.warn('Failed to parse SSE JSON line:', trimmed, e)
        }
      }
    }
  }

  return lastEvent
}

export async function getDocuments(): Promise<IngestedDocument[]> {
  return request<IngestedDocument[]>('/api/documents/')
}

export async function deleteDocument(documentId: string): Promise<{ status: string }> {
  return request<{ status: string }>(`/api/documents/${documentId}`, {
    method: 'DELETE',
  })
}
