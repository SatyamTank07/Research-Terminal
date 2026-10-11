import { request } from '@/shared/services/apiClient'
import type { FilingContent, LocateParams, LocateResult } from './types'

export async function getFilingContent(documentId: string): Promise<FilingContent> {
  return request<FilingContent>(`/api/documents/${encodeURIComponent(documentId)}/content`)
}

export async function locateChunk(params: LocateParams): Promise<LocateResult> {
  const query = new URLSearchParams({ ticker: params.ticker, fiscal_year: String(params.fiscal_year) })
  if (params.breadcrumb) query.set('breadcrumb', params.breadcrumb)
  if (params.item) query.set('item', params.item)
  if (params.chunk_index !== undefined) query.set('chunk_index', String(params.chunk_index))
  return request<LocateResult>(`/api/documents/locate?${query.toString()}`)
}
