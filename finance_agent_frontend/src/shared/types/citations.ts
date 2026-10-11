/** Numbered filing source attached to an assistant message (mirrors backend SourceRef). */
export type SourceKind = 'verbatim' | 'derived' | 'market'

export interface SourceChunk {
  chunk_id: string
  document_id: string
  chunk_index: number
  ticker: string
  fiscal_year: number
  item?: string | null
  breadcrumb?: string | null
}

export interface SourceRef {
  n: number
  kind: SourceKind
  label: string
  document_id?: string | null
  ticker?: string | null
  fiscal_year?: number | null
  chunks: SourceChunk[]
  quote?: string | null
  derived_from?: string | null
  provider?: string | null
}

export function isSourceRef(value: unknown): value is SourceRef {
  if (typeof value !== 'object' || value === null) return false
  const v = value as Partial<SourceRef>
  return typeof v.n === 'number' && typeof v.kind === 'string' && Array.isArray(v.chunks)
}

/** Keeps only numbered sources; legacy citation dicts from older messages are dropped. */
export function toSourceRefs(raw: unknown): SourceRef[] {
  return Array.isArray(raw) ? raw.filter(isSourceRef) : []
}
