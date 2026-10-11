import React, { useMemo } from 'react'
import { MarkdownRenderer } from '@/shared/components/MarkdownRenderer'
import type { SourceRef } from '@/shared/types/citations'
import { splitValuationReport } from './valuationDashboard'
import { ValuationCard } from './ValuationCard'

interface ReportBodyProps {
  content: string
  sources?: SourceRef[]
  onCite?: (ref: SourceRef) => void
}

/**
 * Renders an assistant message. Reports with a valuation dashboard get the headline card in place
 * of the dashboard section; everything else renders as plain markdown.
 */
export const ReportBody: React.FC<ReportBodyProps> = ({ content, sources, onCite }) => {
  const split = useMemo(() => splitValuationReport(content), [content])

  if (!split) {
    return <MarkdownRenderer content={content} sources={sources} onCite={onCite} />
  }
  return (
    <>
      {split.before.trim() && <MarkdownRenderer content={split.before} sources={sources} onCite={onCite} />}
      <ValuationCard dashboard={split.dashboard} table={split.table} sources={sources} onCite={onCite} />
      {split.after.trim() && <MarkdownRenderer content={split.after} sources={sources} onCite={onCite} />}
    </>
  )
}
