import { useCallback, useRef, useState } from 'react'
import { ApiError } from '@/shared/services/apiClient'
import type { SourceRef } from '@/shared/types/citations'
import { getFilingContent, locateChunk } from '../filingService'
import type { FilingContent } from '../types'

interface ViewerState {
  source: SourceRef | null
  chunkPos: number
  filing: FilingContent | null
  targetChunkId: string | null
  approximate: boolean
  loading: boolean
  error: string | null
}

const CLOSED: ViewerState = {
  source: null,
  chunkPos: 0,
  filing: null,
  targetChunkId: null,
  approximate: false,
  loading: false,
  error: null,
}

function errorMessage(e: unknown): string {
  if (e instanceof ApiError && e.status === 404) return 'Source filing is no longer available.'
  return e instanceof Error ? e.message : 'Failed to load the filing.'
}

export function useFilingViewer() {
  const [state, setState] = useState<ViewerState>(CLOSED)
  const cache = useRef(new Map<string, FilingContent>())
  const latestRequest = useRef(0)

  const fetchFiling = useCallback(async (documentId: string): Promise<FilingContent> => {
    const cached = cache.current.get(documentId)
    if (cached) return cached
    const content = await getFilingContent(documentId)
    cache.current.set(documentId, content)
    return content
  }, [])

  const show = useCallback(
    async (source: SourceRef, chunkPos: number) => {
      const chunk = source.chunks[chunkPos]
      if (!chunk) return
      const requestId = ++latestRequest.current
      setState((prev) => ({ ...prev, source, chunkPos, loading: true, error: null, approximate: false }))
      try {
        let filing: FilingContent | null = null
        try {
          filing = await fetchFiling(chunk.document_id)
        } catch (e) {
          if (!(e instanceof ApiError && e.status === 404)) throw e
        }
        let targetChunkId = chunk.chunk_id
        let approximate = false
        if (!filing || !filing.chunks.some((c) => c.id === chunk.chunk_id)) {
          // The filing was re-ingested since this answer was written: open the closest passage instead.
          const located = await locateChunk({
            ticker: chunk.ticker,
            fiscal_year: chunk.fiscal_year,
            breadcrumb: chunk.breadcrumb ?? undefined,
            item: chunk.item ?? undefined,
            chunk_index: chunk.chunk_index,
          })
          filing = await fetchFiling(located.document_id)
          targetChunkId = located.chunk_id
          approximate = true
        }
        if (requestId !== latestRequest.current) return
        setState({ source, chunkPos, filing, targetChunkId, approximate, loading: false, error: null })
      } catch (e) {
        if (requestId !== latestRequest.current) return
        setState((prev) => ({ ...prev, filing: null, targetChunkId: null, loading: false, error: errorMessage(e) }))
      }
    },
    [fetchFiling]
  )

  const openSource = useCallback(
    (source: SourceRef) => {
      if (source.kind !== 'market') void show(source, 0)
    },
    [show]
  )

  const selectChunk = useCallback(
    (pos: number) => {
      if (state.source) void show(state.source, pos)
    },
    [show, state.source]
  )

  const close = useCallback(() => {
    latestRequest.current++
    setState(CLOSED)
  }, [])

  return { ...state, isOpen: state.source !== null, openSource, selectChunk, close }
}

export type FilingViewer = ReturnType<typeof useFilingViewer>
