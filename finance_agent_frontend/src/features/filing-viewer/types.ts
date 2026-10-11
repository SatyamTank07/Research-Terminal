import type { IngestedDocument } from '@/features/documents/documentService'

export interface FilingChunk {
  id: string
  chunk_index: number
  part?: string | null
  item?: string | null
  breadcrumb?: string | null
  chunk_type: 'narrative' | 'table' | string
  content: string
}

export interface TocEntry {
  part?: string | null
  item: string
  title: string
  chunk_index: number
}

export interface FilingContent {
  document: IngestedDocument
  toc: TocEntry[]
  chunks: FilingChunk[]
}

export interface LocateResult {
  document_id: string
  chunk_id: string
  exact: boolean
}

export interface LocateParams {
  ticker: string
  fiscal_year: number
  breadcrumb?: string
  item?: string
  chunk_index?: number
}
