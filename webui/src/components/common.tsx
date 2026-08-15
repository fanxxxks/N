import { Card, Skeleton, Typography } from 'antd'
import type { ReactNode } from 'react'

export function Panel({
  title,
  extra,
  children,
  loading = false,
}: {
  title: string
  extra?: ReactNode
  children: ReactNode
  loading?: boolean
}) {
  return (
    <Card
      size="small"
      className="panel"
      title={title}
      extra={extra}
      styles={{ body: { padding: 16 } }}
    >
      {loading ? <Skeleton active paragraph={{ rows: 4 }} /> : children}
    </Card>
  )
}

export function ErrorText({ message }: { message: string }) {
  return <Typography.Text type="danger">加载失败：{message}</Typography.Text>
}

export function MetricCard({
  title,
  value,
  suffix,
  precision = 2,
  color,
}: {
  title: string
  value: number | string | null | undefined
  suffix?: string
  precision?: number
  color?: string
}) {
  const formatted =
    value === null || value === undefined || value === ''
      ? '—'
      : typeof value === 'number'
        ? value.toFixed(precision)
        : value
  return (
    <div className="metric-card">
      <div className="metric-card-title">{title}</div>
      <div className="metric-card-value" style={color ? { color } : undefined}>
        {formatted}
        {suffix ? <span className="metric-card-suffix">{suffix}</span> : null}
      </div>
    </div>
  )
}

export function fmtPercent(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

export function fmtMoney(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  return value.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
}

export function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`
}
