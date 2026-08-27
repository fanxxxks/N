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
  best_val_reward?: number
  best_reward?: number
  loss: number
  value_loss?: number
}

export interface StrategyData {
  formula?: number[]
  formula_text?: string
  val_reward?: number
  val_icir?: number
  full_window_reward?: number
  full_window_icir?: number
  best_reward?: number
  history?: TrainHistoryPoint[]
  /** P0-04: legacy artifacts (old reward/protocol generation) are flagged. */
  legacy?: boolean
  legacy_reason?: string[]
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
  entry_date: string
  exit_date: string
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

export interface SimRunStatus {
  state: 'idle' | 'starting' | 'running' | 'stopping' | 'stopped' | 'finished' | 'error'
  pid: number | null
  started_at: string | null
  stopping_at: string | null
  ended_at: string | null
  exit_code: number | null
  error: string | null
  reset: boolean
  start_date: string | null
  end_date: string | null
  log_path: string | null
  phase: string | null
  current_date: string | null
  equity: number | null
  progress_updated_at: string | null
}

export interface SimStartResult extends SimRunStatus {
  ok: boolean
  action: 'started' | 'resumed' | 'reset_and_started'
  message: string
  archive: string | null
  args: string[]
}

export interface SimConfigData {
  effective: {
    initial_capital: number
    max_positions: number
    single_weight_cap: number
    commission_rate: number
    min_commission: number
    stamp_tax_rate: number
    transfer_fee_rate: number
    slippage_rate: number
  }
  overrides_path: string
  overrides: Record<string, unknown>
  state_initial_capital: number | null
  pending_reset: boolean
  execution_config_consistent: boolean
  execution_config_mismatches: Record<string, { backtest: number; sim: number }>
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
    universe?: { indexes?: string[]; min_listed_sessions?: number }
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
