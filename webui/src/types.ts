export interface BacktestMetrics {
  total_return: number
  annual_return: number
  annual_volatility: number
  sharpe: number
  sortino: number
  max_drawdown: number
  calmar: number
  average_turnover?: number
}

export interface BacktestData {
  formula?: number[]
  formula_text?: string
  metrics?: BacktestMetrics
  dates?: string[]
  equity_curve?: number[]
  benchmark?: string
  benchmark_equity?: number[] | null
  daily_returns?: number[]
  turnover?: number[]
  positions_count?: number
}

export interface TrainHistoryPoint {
  step: number
  avg_reward: number
  best_reward: number
  loss: number
  value_loss?: number
}

export interface StrategyData {
  formula?: number[]
  formula_text?: string
  best_reward?: number
  history?: TrainHistoryPoint[]
}

export interface SimPosition {
  ts_code: string
  name: string
  quantity: number
  available_quantity: number
  avg_cost: number | null
  last_price: number | null
  last_date?: string
  market_value: number
}

export interface SimState {
  initial_capital: number | null
  cash: number | null
  trade_count: number
  market_value: number
  total_equity: number
  positions: SimPosition[]
  equity_history: { trade_date: string; equity: number }[]
}

export interface HoldingRow {
  ts_code: string
  name: string
  weight: number
}

export interface PositionSnapshot {
  signal_date: string
  exec_date: string
  count: number
  rows: HoldingRow[]
}

export interface PositionPage {
  items: PositionSnapshot[]
  total: number
}

export interface SimDays {
  total: number
  dates: string[]
}

export interface SimDayData {
  date: string
  orders: Record<string, unknown>[]
  trades: Record<string, unknown>[]
}

export interface Artifact {
  name: string
  size: number
  mtime: string
  exists: boolean
}

export interface DataStatus {
  ready?: boolean
  reason?: string
  db: {
    ready: boolean
    reason?: string
    path?: string
    stocks?: number
    daily_rows?: number
    first_trade_date?: string
    last_trade_date?: string
  }
  artifacts: Artifact[]
  config: {
    date_range?: { start?: string; end?: string }
    universe?: { indexes?: string[]; min_listed_days?: number }
    model?: Record<string, number | null>
    backtest?: Record<string, number | string | null>
  }
}

export interface LogFile {
  name: string
  size: number
  mtime: string
  kind: string
}

export interface LogContent {
  name: string
  size?: number
  lines?: number
  content?: string
  truncated?: boolean
  error?: string
}

export interface OverviewData {
  backtest: BacktestData
  strategy: StrategyData
  sim: SimState
  status: DataStatus
}
