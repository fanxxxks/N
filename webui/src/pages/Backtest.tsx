import { useMemo, useState } from 'react'
import { Button, Col, Descriptions, Pagination, Row, Table, Tag } from 'antd'
import { ReloadOutlined } from '@ant-design/icons'
import { api, useApi } from '../api/client'
import { ErrorText, Panel, fmtPercent } from '../components/common'
import { EquityChart, LineChart } from '../components/charts'
import type { BacktestData, PositionPage } from '../types'

function usePositions() {
  const [offset, setOffset] = useState(0)
  const { data, loading, error, reload } = useApi(
    () => api.positions(offset, 20),
    [offset],
  )
  return { data, loading, error, reload, offset, setOffset }
}

export default function BacktestPage() {
  const bt = useApi<BacktestData>(api.backtest)
  const pos = usePositions()

  const metrics = bt.data?.metrics
  const drawdown = useMemo(() => {
    const curve = bt.data?.equity_curve ?? []
    if (curve.length < 2) return []
    let peak = curve[0]
    return curve.map((v) => {
      peak = Math.max(peak, v)
      return peak > 0 ? (v / peak - 1) : 0
    })
  }, [bt.data?.equity_curve])

  return (
    <div>
      <div className="page-head">
        <h2 className="page-title">回测</h2>
        <Button icon={<ReloadOutlined />} onClick={bt.reload}>
          刷新
        </Button>
      </div>
      {bt.error && <ErrorText message={bt.error} />}

      <Panel title="净值曲线" loading={bt.loading}>
        {bt.data?.equity_curve && bt.data.dates ? (
          <EquityChart
            dates={bt.data.dates}
            equity={bt.data.equity_curve}
            benchmarkEquity={bt.data.benchmark_equity}
            benchmarkName={bt.data.benchmark}
            height={420}
          />
        ) : (
          <div style={{ color: '#8b949e', padding: 24 }}>暂无回测结果</div>
        )}
      </Panel>

      <Row gutter={16}>
        <Col span={12}>
          <Panel title="回撤曲线" loading={bt.loading}>
            {drawdown.length > 1 && bt.data?.dates ? (
              <LineChart
                x={bt.data.dates}
                series={[{ name: '回撤', data: drawdown }]}
                height={260}
                percent
                legend={false}
              />
            ) : (
              <div style={{ color: '#8b949e' }}>暂无数据</div>
            )}
          </Panel>
        </Col>
        <Col span={12}>
          <Panel title="日收益与换手" loading={bt.loading}>
            {bt.data?.daily_returns && bt.data.dates ? (
              <LineChart
                x={bt.data.dates.slice(1)}
                series={[
                  { name: '日收益', data: bt.data.daily_returns },
                  { name: '换手', data: bt.data.turnover ?? [], yAxisIndex: 1 },
                ]}
                height={260}
                percent
                legend={false}
              />
            ) : (
              <div style={{ color: '#8b949e' }}>暂无数据</div>
            )}
          </Panel>
        </Col>
      </Row>

      <Panel title="绩效指标" loading={bt.loading}>
        {metrics ? (
          <Descriptions
            size="small"
            column={{ xs: 2, sm: 2, md: 4 }}
            items={[
              { key: '1', label: '累计收益', children: fmtPercent(metrics.total_return) },
              { key: '2', label: '年化收益', children: fmtPercent(metrics.annual_return) },
              { key: '3', label: '年化波动', children: fmtPercent(metrics.annual_volatility) },
              { key: '4', label: 'Sharpe', children: metrics.sharpe.toFixed(3) },
              { key: '5', label: 'Sortino', children: metrics.sortino.toFixed(3) },
              { key: '6', label: 'Calmar', children: metrics.calmar.toFixed(3) },
              { key: '7', label: '最大回撤', children: fmtPercent(metrics.max_drawdown) },
              {
                key: '8',
                label: '平均换手',
                children: fmtPercent(metrics.average_turnover ?? 0),
              },
            ]}
          />
        ) : (
          <div style={{ color: '#8b949e' }}>暂无指标</div>
        )}
      </Panel>

      <Panel
        title="每日持仓快照（信号日 → 执行日）"
        loading={pos.loading}
        extra={
          pos.data && pos.data.total > 20 ? (
            <Pagination
              simple
              current={pos.offset / 20 + 1}
              pageSize={20}
              total={pos.data.total}
              onChange={(page) => pos.setOffset((page - 1) * 20)}
            />
          ) : null
        }
      >
        {pos.error && <ErrorText message={pos.error} />}
        <PositionsTable page={pos.data} />
      </Panel>
    </div>
  )
}

function PositionsTable({ page }: { page: PositionPage | null }) {
  if (!page || page.items.length === 0) {
    return <div style={{ color: '#8b949e' }}>暂无持仓快照</div>
  }
  const rows = page.items.flatMap((snap) =>
    snap.rows.map((r) => ({
      ...r,
      signal_date: snap.signal_date,
      entry_date: snap.entry_date,
      exit_date: snap.exit_date,
    })),
  )
  return (
    <Table
      size="small"
      rowKey={(r) => `${r.signal_date}-${r.ts_code}`}
      dataSource={rows}
      pagination={false}
      scroll={{ y: 480 }}
      columns={[
        { title: '信号日', dataIndex: 'signal_date', width: 110 },
        { title: '入场日', dataIndex: 'entry_date', width: 110 },
        { title: '退出日', dataIndex: 'exit_date', width: 110 },
        { title: '代码', dataIndex: 'ts_code', width: 110 },
        { title: '名称', dataIndex: 'name', width: 110 },
        {
          title: '权重',
          dataIndex: 'weight',
          render: (w: number) => <Tag color="blue">{fmtPercent(w)}</Tag>,
        },
      ]}
    />
  )
}
