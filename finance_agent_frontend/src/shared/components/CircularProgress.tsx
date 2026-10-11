import React from 'react'

interface CircularProgressProps {
  progress: number // 0 to 100
  size?: number
  strokeWidth?: number
  className?: string
  status?: 'idle' | 'uploaded' | 'uploading' | 'parsing' | 'chunking' | 'embedding' | 'storing' | 'completed' | 'error'
}

export const CircularProgress: React.FC<CircularProgressProps> = ({
  progress,
  size = 20,
  strokeWidth = 2.5,
  className = '',
  status,
}) => {
  const radius = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const clampedProgress = Math.min(Math.max(progress, 0), 100)
  const offset = circumference - (clampedProgress / 100) * circumference

  const isCompleted = status === 'completed' || clampedProgress >= 100
  const isError = status === 'error'

  const strokeColor = isError
    ? 'stroke-neg'
    : isCompleted
    ? 'stroke-pos'
    : 'stroke-acc'

  return (
    <div
      className={`relative inline-flex items-center justify-center ${className}`}
      style={{ width: size, height: size }}
      title={isCompleted ? '100% Ingested' : `${clampedProgress}% Ingested`}
    >
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        className="transform -rotate-90 origin-center"
      >
        {/* Track Background */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
          className="stroke-line-strong fill-none"
        />

        {/* Dynamic Progress Fill */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="butt"
          className={`${strokeColor} fill-none transition-all duration-300 ease-out`}
        />
      </svg>

      {/* Center checkmark when finished */}
      {isCompleted && (
        <span className="absolute inset-0 flex items-center justify-center text-[9px] font-bold text-pos">
          ✓
        </span>
      )}
    </div>
  )
}
