import { useCallback, useEffect, useRef, useState } from 'react'
import type {
  BacktestData,
  DataStatus,
  LogContent,
  LogFile,
  OverviewData,
  PositionPage,
  SimConfigData,
  SimDayData,
  SimDays,
  SimRunStatus,
  SimState,
  StrategyData,
} from '../types'

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, init)
  if (!resp.ok) {
    let detail = resp.statusText
    try {
      const body = await resp.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      /* ignore */
    }
    throw new Error(`${resp.status} ${detail}`)
  }
  return (await resp.json()) as T
}

export const api = {
  health: () => request<{ status: string; time: string }>('/api/health'),
  overview: () => request<OverviewData>('/api/overview'),
  backtest: () => request<BacktestData>('/api/backtest'),
  positions: (offset = 0, limit = 20) =>
    request<PositionPage>(`/api/backtest/positions?offset=${offset}&limit=${limit}`),
  strategy: () => request<StrategyData>('/api/strategy'),
  sim: () => request<SimState>('/api/sim'),
  simDays: () => request<SimDays>('/api/sim/days'),
  simDay: (date: string) => request<SimDayData>(`/api/sim/day/${date}`),
  simStop: () =>
    request<{ ok: boolean; reason?: string }>('/api/sim/stop', {
      method: 'POST',
    }),
  simStatus: () => request<SimRunStatus>('/api/sim/status'),
  simStart: (body: { reset?: boolean; start?: string | null; end?: string | null } = {}) =>
    request<SimRunStatus>('/api/sim/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),
  simReset: () =>
    request<{ ok: boolean; reason?: string; archive?: string }>('/api/sim/reset', {
      method: 'POST',
    }),
  simConfig: () => request<SimConfigData>('/api/sim/config'),
  simConfigPut: (patch: Record<string, number | null>) =>
    request<SimConfigData>('/api/sim/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    }),
  dataStatus: () => request<DataStatus>('/api/data-status'),
  logs: () => request<LogFile[]>('/api/logs'),
  logContent: (name: string, tail = 1000) =>
    request<LogContent>(`/api/logs/${encodeURIComponent(name)}?tail=${tail}`),
}

export interface ApiState<T> {
  data: T | null
  loading: boolean
  error: string | null
  reload: () => void
}

/** Fetch on mount (and on interval, when given) with manual reload. */
export function useApi<T>(
  fn: () => Promise<T>,
  deps: unknown[] = [],
  intervalMs?: number,
): ApiState<T> {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)
  const fnRef = useRef(fn)
  fnRef.current = fn

  const reload = useCallback(() => setTick((t) => t + 1), [])

  useEffect(() => {
    let alive = true
    setLoading(true)
    fnRef
      .current()
      .then((value) => {
        if (alive) {
          setData(value)
          setError(null)
        }
      })
      .catch((err: Error) => {
        if (alive) setError(err.message)
      })
      .finally(() => {
        if (alive) setLoading(false)
      })
    return () => {
      alive = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick])

  useEffect(() => {
    if (!intervalMs) return
    const timer = setInterval(reload, intervalMs)
    return () => clearInterval(timer)
  }, [intervalMs, reload])

  return { data, loading, error, reload }
}
