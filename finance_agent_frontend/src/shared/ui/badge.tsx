import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/shared/lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-xs border px-1.5 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wider transition-colors",
  {
    variants: {
      variant: {
        default:
          "bg-acc-bg text-acc border-acc-dim",
        secondary:
          "bg-raised text-fg border-line-strong",
        destructive:
          "bg-neg-bg text-neg border-neg-dim",
        outline:
          "text-dim border-line-strong",
        success:
          "bg-pos-bg text-pos border-pos-dim",
        warning:
          "bg-acc-bg text-acc border-acc-dim",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <div className={cn(badgeVariants({ variant }), className)} {...props} />
  )
}

export { Badge, badgeVariants }
