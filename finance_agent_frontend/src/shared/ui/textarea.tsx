import * as React from "react"
import { cn } from "@/shared/lib/utils"

export interface TextareaProps
  extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {}

const Textarea = React.forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ className, ...props }, ref) => {
    return (
      <textarea
        className={cn(
          "flex min-h-[60px] w-full rounded-xs border border-line-strong bg-raised px-3 py-2 text-sm text-fg-strong placeholder:text-faint focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-acc-dim focus-visible:border-acc disabled:cursor-not-allowed disabled:opacity-50 resize-none transition-all",
          className
        )}
        ref={ref}
        {...props}
      />
    )
  }
)
Textarea.displayName = "Textarea"

export { Textarea }
