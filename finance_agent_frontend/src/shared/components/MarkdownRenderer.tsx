import React, { useMemo } from 'react'
import ReactMarkdown, { type Components, type Options } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { SourceRef } from '@/shared/types/citations'

interface MarkdownRendererProps {
  content: string
  /** Numbered sources of this message; `#cite-n` links render as citation markers. */
  sources?: SourceRef[]
  onCite?: (ref: SourceRef) => void
}

const CITE_HREF = /^#cite-(\d+)$/

/**
 * Reports mix single-dollar LaTeX ("$R_f$", "$K_d \times (1 - t)$") with dollar amounts in prose
 * ("price of $336.64 ... value of $111.21"), so remark-math's single-dollar mode would turn prose
 * into math. Math spans open with a letter, backslash or paren (amounts open with a digit); those
 * are rewritten to `$$...$$`, the only inline form remark-math accepts with singleDollarTextMath off.
 */
const INLINE_MATH = /(^|[^\w$\\])\$(?=[A-Za-z\\(])([^$\n]{0,160}?[^\s$])\$(?![\w$])/g
function markInlineMath(markdown: string): string {
  return markdown.replace(INLINE_MATH, (_m, pre: string, body: string) => `${pre}$$${body}$$`)
}
const REMARK_PLUGINS: Options['remarkPlugins'] = [remarkGfm, [remarkMath, { singleDollarTextMath: false }]]
const REHYPE_PLUGINS: Options['rehypePlugins'] = [[rehypeKatex, { throwOnError: false, strict: 'ignore' }]]

/** One figure: optional paren/sign/currency, digits, optional unit ("$412.30", "(1.2)", "-20.3%", "$4,912.0M", "12.5x"). */
const FIGURE = String.raw`\(?[-+−–]?\s*[$€£]?\s*[-−–]?\s*\d[\d,]*(?:\.\d+)?\s*(?:%|x|bps|pp|bn|mm|[KMBT])?\)?`
/** Whole cell is figures: "a", "a / b", or "a (source note)". Rejects text like "10-K balance sheet". */
const NUMERIC_CELL = new RegExp(String.raw`^${FIGURE}(?:\s*/\s*${FIGURE})*(?:\s*\([^)]*\))?$`, 'i')
/** Explicitly signed percentage or point change: "+9.0%", "−20.3%", "-1.5pp". */
const SIGNED_PCT = /^\s*([-+−–])\s*\d[\d.,]*\s*(%|pp|bps)\s*$/i

/** Whole-cell rating words emitted by the risk matrix and forensic findings tables. */
const RATING_TAG: Record<string, string> = {
  severe: 'text-neg border-neg-dim bg-neg-bg',
  critical: 'text-neg border-neg-dim bg-neg-bg',
  high: 'text-acc border-acc-dim bg-acc-bg',
  moderate: 'text-dim border-line-strong',
  medium: 'text-dim border-line-strong',
  low: 'text-faint border-line',
}

/** Plain text of rendered markdown children (descends through <strong>, <em>, ...). */
function textOf(node: React.ReactNode): string {
  if (node == null || typeof node === 'boolean') return ''
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(textOf).join('')
  if (React.isValidElement<{ children?: React.ReactNode }>(node)) return textOf(node.props.children)
  return ''
}

/** Cell treatment derived from its text: rating tag, signed-change color, or mono numerics. */
function renderCell(children: React.ReactNode): { className: string; content: React.ReactNode } {
  const text = textOf(children).trim()
  const rating = RATING_TAG[text.toLowerCase()]
  if (rating) {
    return {
      className: '',
      content: (
        <span className={`inline-block font-mono text-[10px] font-semibold uppercase tracking-wider px-1.5 py-0.5 border rounded-xs ${rating}`}>
          {text}
        </span>
      ),
    }
  }
  const signed = SIGNED_PCT.exec(text)
  if (signed) {
    const tone = signed[1] === '+' ? 'text-pos' : 'text-neg'
    return { className: `font-mono font-tabular whitespace-nowrap ${tone}`, content: children }
  }
  if (NUMERIC_CELL.test(text)) {
    return { className: 'font-mono font-tabular whitespace-nowrap', content: children }
  }
  return { className: '', content: children }
}

export const CitationMarker: React.FC<{ source?: SourceRef; n: string; onCite?: (ref: SourceRef) => void }> = ({
  source,
  n,
  onCite,
}) => {
  const chip = 'font-mono text-[10px] font-medium px-1 ml-0.5 rounded-xs leading-none'
  if (!source) {
    return <sup className={`${chip} text-faint`}>{n}</sup>
  }
  if (source.kind === 'market' || !onCite) {
    return (
      <sup
        className={`${chip} bg-raised text-dim cursor-help`}
        title={source.kind === 'market' ? `Market data · ${source.provider ?? 'unknown'}` : source.label}
      >
        {n}
      </sup>
    )
  }
  return (
    <sup>
      <button
        type="button"
        onClick={() => onCite(source)}
        title={source.quote ? `“${source.quote}”` : source.derived_from || source.label}
        className={`${chip} py-0.5 bg-acc-bg text-acc hover:bg-acc hover:text-black transition-colors cursor-pointer`}
      >
        {n}
      </button>
    </sup>
  )
}

function MarkdownRendererImpl({ content, sources, onCite }: MarkdownRendererProps) {
  // Stable component map: an inline object would give every element a new component type on each
  // parent render, remounting the whole tree (slow for a full 10-K, and it detaches quote highlights).
  const components = useMemo<Components>(
    () => ({
      a: ({ href, children }) => {
        const cite = href ? CITE_HREF.exec(href) : null
        if (cite) {
          const source = sources?.find((s) => s.n === Number(cite[1]))
          return <CitationMarker source={source} n={cite[1]} onCite={onCite} />
        }
        return (
          <a href={href} className="text-acc hover:underline">
            {children}
          </a>
        )
      },
      h1: ({ children }) => (
        <h1 className="text-lg sm:text-xl font-semibold text-fg-strong leading-snug tracking-tight mb-3">
          {children}
        </h1>
      ),
      h2: ({ children }) => (
        <h2 className="font-mono text-xs font-semibold tracking-[0.1em] uppercase text-acc mt-7 mb-3 pb-2 border-b border-line">
          {children}
        </h2>
      ),
      h3: ({ children }) => (
        <h3 className="text-sm font-semibold text-fg-strong mt-5 mb-1.5">
          {children}
        </h3>
      ),
      h4: ({ children }) => (
        <h4 className="font-mono text-[11px] font-semibold tracking-wider uppercase text-dim mt-4 mb-1">
          {children}
        </h4>
      ),
      p: ({ children }) => (
        <p className="text-fg leading-[1.7] mb-2.5 last:mb-0">
          {children}
        </p>
      ),
      strong: ({ children }) => (
        <strong className="font-semibold text-fg-strong">
          {children}
        </strong>
      ),
      em: ({ children }) => (
        <em className="text-fg-strong italic">
          {children}
        </em>
      ),
      ul: ({ children }) => (
        <ul className="list-disc pl-5 space-y-1 my-2 text-fg marker:text-acc">
          {children}
        </ul>
      ),
      ol: ({ children }) => (
        <ol className="list-decimal pl-5 space-y-1.5 my-2 text-fg marker:text-acc marker:font-mono marker:text-xs">
          {children}
        </ol>
      ),
      li: ({ children }) => (
        <li className="leading-[1.65] pl-0.5">
          {children}
        </li>
      ),
      blockquote: ({ children }) => (
        <blockquote className="border-l-2 border-acc bg-acc-bg pl-3 pr-2 py-2 my-3 font-mono text-xs text-fg [&_strong]:text-acc [&_p]:text-fg [&_p]:leading-relaxed">
          {children}
        </blockquote>
      ),
      code: ({ className, children, ...props }) => {
        const match = /language-(\w+)/.exec(className || '')
        const isInline = !match && typeof children === 'string' && !children.includes('\n')
        return isInline ? (
          <code
            className="px-1 py-0.5 rounded-xs bg-raised border border-line font-mono text-[11px] text-acc"
            {...props}
          >
            {children}
          </code>
        ) : (
          <div className="overflow-x-auto my-3 rounded-xs border border-line bg-canvas p-3">
            <code className="font-mono text-xs text-fg block" {...props}>
              {children}
            </code>
          </div>
        )
      },
      table: ({ children }) => (
        <div className="overflow-x-auto my-3 border border-line">
          <table className="w-full text-left border-collapse text-[12.5px] leading-snug">
            {children}
          </table>
        </div>
      ),
      thead: ({ children }) => (
        <thead className="bg-surface border-b border-line-strong">
          {children}
        </thead>
      ),
      tbody: ({ children }) => (
        <tbody className="divide-y divide-[#1c1c1c]">
          {children}
        </tbody>
      ),
      tr: ({ children }) => (
        <tr className="hover:bg-[#151515] transition-colors">
          {children}
        </tr>
      ),
      th: ({ children, style }) => (
        <th
          style={style}
          className="px-3 py-2 font-mono text-[10.5px] font-medium uppercase tracking-wider text-dim whitespace-nowrap"
        >
          {children}
        </th>
      ),
      td: ({ children, style }) => {
        const cell = renderCell(children)
        return (
          <td style={style} className={`px-3 py-2 text-fg align-top ${cell.className}`}>
            {cell.content}
          </td>
        )
      },
      hr: () => <hr className="my-4 border-line" />,
    }),
    [sources, onCite]
  )
  const markedContent = useMemo(() => markInlineMath(content), [content])
  return (
    <div className="markdown-content text-sm leading-[1.7] break-words text-fg">
      <ReactMarkdown
        remarkPlugins={REMARK_PLUGINS}
        rehypePlugins={REHYPE_PLUGINS}
        components={components}
      >
        {markedContent}
      </ReactMarkdown>
    </div>
  )
}

export const MarkdownRenderer = React.memo(MarkdownRendererImpl)
