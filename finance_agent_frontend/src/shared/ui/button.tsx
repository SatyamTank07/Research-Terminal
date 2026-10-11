import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"
import { cn } from "@/shared/lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap rounded-xs text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-acc disabled:pointer-events-none disabled:opacity-40 select-none cursor-pointer",
  {
    variants: {
      variant: {
        default:
          "bg-acc text-black font-mono font-semibold tracking-wide hover:bg-acc-hover",
        destructive:
          "bg-neg-bg text-neg border border-neg-dim hover:bg-neg hover:text-black",
        outline:
          "border border-line-strong bg-panel text-fg hover:border-acc-dim hover:text-fg-strong",
        secondary:
          "bg-raised text-fg-strong hover:bg-line",
        ghost:
          "text-dim hover:bg-raised hover:text-fg-strong",
        link:
          "text-acc underline-offset-4 hover:underline",
      },
      size: {
        default: "h-9 px-4 py-2",
        sm: "h-8 px-3 text-xs",
        lg: "h-10 px-8 text-base",
        icon: "h-9 w-9",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = "Button"

export { Button, buttonVariants }
