import { Button, Col, Row, Space, Tag, Tooltip } from 'antd'
import { ReloadOutlined, InfoCircleOutlined } from '@ant-design/icons'
import { api, useApi } from '../api/client'
import { ErrorText, MetricCard, Panel, fmtMoney, fmtPercent } from '../components/common'
import { EquityChart, LineChart } from '../components/charts'

export default function Overview() {
  const { data, loading, error, reload } = useApi(api.overview, [], 30000)

  const metrics = data?.backtest?.metrics
  const sim = data?.sim
  const strategy = data?.strategy
  const history = strategy?.history ?? []
  const valReward = strategy?.val_reward ?? strategy?.best_reward

  return (
    <div>
      <div className="page-head">
        <h2 className="page-title">概览</h2>
        <Space>
          <Tooltip title="每 30 秒自动刷新">
            <InfoCircleOutlined style={{ color: '#8b949e' }} />
          </Tooltip>
          <Button icon={<ReloadOutlined />} onClick={reload}>
            刷新
          </Button>
        </Space>
      </div>

      {error && <ErrorText message={error} />}

      <Row gutter={[12, 12]}>
        <Col span={4}>
          <MetricCard
            title="累计收益"
            value={metrics ? fmtPercent(metrics.total_return) : undefined}
            color={positiveColor(metrics?.total_return)}
          />
        </Col>
        <Col span={4}>
          <MetricCard
            title="年化收益"
            value={metrics ? fmtPercent(metrics.annual_return) : undefined}
            color={positiveColor(metrics?.annual_return)}
          />
        </Col>
        <Col span={4}>
          <MetricCard
            title="Sharpe"
            value={metrics?.sharpe}
            color={positiveColor(metrics?.sharpe)}
          />
        </Col>
        <Col span={4}>
          <MetricCard
            title="Sortino"
            value={metrics?.sortino}
            color={positiveColor(metrics?.sortino)}
          />
        </Col>
        <Col span={4}>
          <MetricCard
            title="最大回撤"
            value={metrics ? fmtPercent(metrics.max_drawdown) : undefined}
            color="#ff7b72"
          />
        </Col>
        <Col span={4}>
          <MetricCard
            title="Calmar"
            value={metrics?.calmar}
            color={positiveColor(metrics?.calmar)}
          />
        </Col>
      </Row>

      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col span={16}>
          <Panel
            title="回测净值曲线"
            loading={loading}
            extra={
              metrics ? (
                <Tag color="blue">平均换手 {fmtPercent(metrics.average_turnover ?? 0)}</Tag>
              ) : null
            }
          >
            {data?.backtest?.equity_curve && data.backtest.dates ? (
              <EquityChart
                dates={data.backtest.dates}
                equity={data.backtest.equity_curve}
                benchmarkEquity={data.backtest.benchmark_equity}
                benchmarkName={data.backtest.benchmark}
                height={400}
              />
            ) : (
              <div style={{ color: '#8b949e', padding: 24 }}>
                暂无回测结果，请先运行 python -m ashare_model.backtest
              </div>
            )}
          </Panel>
        </Col>
        <Col span={8}>
          <Panel
            title="最优因子公式"
            loading={loading}
            extra={
              <>
                {strategy?.legacy ? (
                  <Tooltip title={(strategy.legacy_reason ?? []).join('；') || '旧代产物，仅存档参考'}>
                    <Tag color="orange">LEGACY</Tag>
                  </Tooltip>
                ) : null}
                {valReward != null ? (
                  <Tag color="green">验证集奖励 {valReward.toFixed(3)}</Tag>
                ) : null}
              </>
            }
          >
            {strategy?.formula_text ? (
              <div className="formula-box">{strategy.formula_text}</div>
            ) : (
              <div style={{ color: '#8b949e' }}>暂无策略，请先运行 python -m ashare_model.train</div>
            )}
          </Panel>
          <Panel title="模拟盘摘要" loading={loading}>
            <table className="kv-table">
              <tbody>
                <tr>
                  <td>总资产</td>
                  <td>{fmtMoney(sim?.total_equity)}</td>
                </tr>
                <tr>
                  <td>可用资金</td>
                  <td>{fmtMoney(sim?.cash)}</td>
                </tr>
                <tr>
                  <td>持仓市值</td>
                  <td>{fmtMoney(sim?.market_value)}</td>
                </tr>
                <tr>
                  <td>持仓只数</td>
                  <td>{sim?.positions.length ?? '—'}</td>
                </tr>
                <tr>
                  <td>成交笔数</td>
                  <td>{sim?.trade_count ?? '—'}</td>
                </tr>
              </tbody>
            </table>
          </Panel>
        </Col>
      </Row>

      {history.length > 0 && (
        <Panel title="训练过程（平均奖励 / 最优验证奖励 / 损失）" loading={loading}>
          <LineChart
            x={history.map((h) => String(h.step))}
            series={[
              { name: '平均奖励', data: history.map((h) => h.avg_reward) },
              { name: '最优验证奖励', data: history.map((h) => h.best_val_reward ?? h.best_reward ?? 0) },
              { name: '损失', data: history.map((h) => h.loss), yAxisIndex: 1 },
            ]}
            height={320}
          />
        </Panel>
      )}
    </div>
  )
}

function positiveColor(value: number | null | undefined): string | undefined {
  if (value === null || value === undefined) return undefined
  return value >= 0 ? '#3fb950' : '#ff7b72'
}
