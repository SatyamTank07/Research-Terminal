import React from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface MarkdownRendererProps {
  content: string
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content }) => {
  return (
    <div className="markdown-content text-xs sm:text-sm space-y-3 leading-relaxed break-words text-slate-200">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => (
            <h1 className="text-base sm:text-lg font-semibold text-white mt-4 mb-2 pb-1 border-b border-slate-800 flex items-center gap-2">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-xs sm:text-sm font-semibold tracking-wider text-slate-100 uppercase mt-3.5 mb-1.5 flex items-center gap-1.5">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-xs font-semibold tracking-wider uppercase text-sky-400 mt-3 mb-1">
              {children}
            </h3>
          ),
          p: ({ children }) => (
            <p className="text-slate-200 leading-relaxed mb-2 last:mb-0">
              {children}
            </p>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-white">
              {children}
            </strong>
          ),
          em: ({ children }) => (
            <em className="text-slate-100 font-medium italic">
              {children}
            </em>
          ),
          ul: ({ children }) => (
            <ul className="list-disc pl-5 space-y-1 my-2 text-slate-200 marker:text-sky-400">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal pl-5 space-y-1.5 my-2 text-slate-200 marker:text-sky-400 marker:font-semibold">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="leading-relaxed pl-0.5">
              {children}
            </li>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-sky-400 bg-sky-950/20 pl-3.5 py-2 my-2.5 rounded-r-md text-xs sm:text-sm text-sky-200/90">
              {children}
            </blockquote>
          ),
          code: ({ className, children, ...props }) => {
            const match = /language-(\w+)/.exec(className || '')
            const isInline = !match && typeof children === 'string' && !children.includes('\n')
            return isInline ? (
              <code
                className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 font-mono text-[11px] text-sky-300"
                {...props}
              >
                {children}
              </code>
            ) : (
              <div className="overflow-x-auto my-2.5 rounded-md border border-slate-800 bg-[#070a12] p-3">
                <code className="font-mono text-xs text-slate-200 block" {...props}>
                  {children}
                </code>
              </div>
            )
          },
          table: ({ children }) => (
            <div className="overflow-x-auto my-3 rounded-lg border border-slate-800 bg-[#090e1a] shadow-sm">
              <table className="w-full text-left border-collapse text-xs">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-slate-900 text-slate-300 font-semibold border-b border-slate-800 text-[11px] uppercase tracking-wider">
              {children}
            </thead>
          ),
          tbody: ({ children }) => (
            <tbody className="divide-y divide-slate-800/60 font-tabular text-xs">
              {children}
            </tbody>
          ),
          tr: ({ children }) => (
            <tr className="hover:bg-sky-500/5 transition-colors even:bg-slate-900/40">
              {children}
            </tr>
          ),
          th: ({ children }) => (
            <th className="px-3.5 py-2.5 text-slate-300 font-medium">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-3.5 py-2 text-slate-300 font-tabular">
              {children}
            </td>
          ),
          hr: () => <hr className="my-3 border-slate-800/80" />,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}
