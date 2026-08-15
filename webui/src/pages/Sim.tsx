import { useState } from 'react'
import {
  Button,
  Col,
  Modal,
  Row,
  Select,
  Space,
  Table,
  Tag,
  Tooltip,
  message,
} from 'antd'
import { ReloadOutlined, StopOutlined } from '@ant-design/icons'
import { api, useApi } from '../api/client'
import { ErrorText, MetricCard, Panel, fmtMoney, fmtPercent } from '../components/common'
import { EquityChart } from '../components/charts'
import type { SimDayData } from '../types'

export default function SimPage() {
  const state = useApi(api.sim, [], 15000)
  const days = useApi(api.simDays)
  const [selectedDay, setSelectedDay] = useState<string | null>(null)
  const day = useApi<SimDayData | null>(
    () => (selectedDay ? api.simDay(selectedDay) : Promise.resolve(null)),
    [selectedDay],
  )
  const [stopping, setStopping] = useState(false)

  const equityDates = state.data?.equity_history.map((h) => h.trade_date) ?? []
  const equityValues = state.data?.equity_history.map((h) => h.equity) ?? []
  const init = state.data?.initial_capital ?? 0
  const totalReturn = init > 0 && equityValues.length ? equityValues[equityValues.length - 1] / init - 1 : null

  const handleStop = () => {
    Modal.confirm({
      title: '写入紧急停止信号？',
      content: '运行中的模拟盘将在下一个交易日循环检查到 STOP_SIGNAL 后停止。',
      okText: '确认写入',
      cancelText: '取消',
      okButtonProps: { danger: true },
      onOk: async () => {
        setStopping(true)
        try {
          await api.simStop()
          message.success('已写入 STOP_SIGNAL')
        } catch (err) {
          message.error(`写入失败：${(err as Error).message}`)
        } finally {
          setStopping(false)
        }
      },
    })
  }

  return (
    <div>
      <div className="page-head">
        <h2 className="page-title">模拟盘</h2>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={() => { state.reload(); days.reload() }}>
            刷新
          </Button>
          <Tooltip title="在项目根目录写入 STOP_SIGNAL 文件，模拟盘循环将停止">
            <Button danger icon={<StopOutlined />} loading={stopping} onClick={handleStop}>
              紧急停止
            </Button>
          </Tooltip>
        </Space>
      </div>
      {state.error && <ErrorText message={state.error} />}

      <Row gutter={[12, 12]}>
        <Col span={4}>
          <MetricCard title="总资产" value={fmtMoney(state.data?.total_equity)} />
        </Col>
        <Col span={4}>
          <MetricCard title="可用资金" value={fmtMoney(state.data?.cash)} />
        </Col>
        <Col span={4}>
          <MetricCard title="持仓市值" value={fmtMoney(state.data?.market_value)} />
        </Col>
        <Col span={4}>
          <MetricCard title="累计收益" value={totalReturn === null ? undefined : fmtPercent(totalReturn)} color={positiveColor(totalReturn)} />
        </Col>
        <Col span={4}>
          <MetricCard title="持仓只数" value={state.data?.positions.length ?? '—'} />
        </Col>
        <Col span={4}>
          <MetricCard title="成交笔数" value={state.data?.trade_count ?? '—'} />
        </Col>
      </Row>

      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col span={24}>
          <Panel title="模拟盘资金曲线" loading={state.loading}>
            {equityValues.length > 1 ? (
              <EquityChart
                dates={equityDates}
                equity={equityValues}
                height={360}
              />
            ) : (
              <div style={{ color: '#8b949e', padding: 24 }}>暂无资金曲线</div>
            )}
          </Panel>
        </Col>
      </Row>

      <Panel title={`当前持仓（${state.data?.positions.length ?? 0} 只）`} loading={state.loading}>
        {state.data && state.data.positions.length > 0 ? (
          <Table
            size="small"
            rowKey="ts_code"
            dataSource={state.data.positions}
            pagination={false}
            scroll={{ y: 420 }}
            columns={[
              { title: '代码', dataIndex: 'ts_code', width: 120 },
              { title: '名称', dataIndex: 'name', width: 130 },
              { title: '持仓', dataIndex: 'quantity', align: 'right' as const },
              { title: '可用', dataIndex: 'available_quantity', align: 'right' as const },
              { title: '成本', dataIndex: 'avg_cost', align: 'right' as const, render: (v: number | null) => fmtMoney(v) },
              { title: '现价', dataIndex: 'last_price', align: 'right' as const, render: (v: number | null) => fmtMoney(v) },
              { title: '市值', dataIndex: 'market_value', align: 'right' as const, render: (v: number) => fmtMoney(v) },
              {
                title: '盈亏',
                key: 'pnl',
                align: 'right' as const,
                render: (_: unknown, row: { avg_cost: number | null; last_price: number | null; quantity: number }) => {
                  if (row.avg_cost == null || row.last_price == null) return '—'
                  const pnl = (row.last_price - row.avg_cost) * row.quantity
                  return <span style={{ color: positiveColor(pnl) }}>{fmtMoney(pnl)}</span>
                },
              },
            ]}
          />
        ) : (
          <div style={{ color: '#8b949e' }}>暂无持仓</div>
        )}
      </Panel>

      <Panel
        title="订单 / 成交流水"
        loading={days.loading}
        extra={
          days.data && days.data.dates.length > 0 ? (
            <Select
              showSearch
              placeholder="选择交易日"
              style={{ width: 200 }}
              value={selectedDay ?? undefined}
              onChange={setSelectedDay}
              options={days.data.dates.map((d) => ({ value: d, label: formatDate(d) }))}
            />
          ) : null
        }
      >
        {days.error && <ErrorText message={days.error} />}
        {!days.data || days.data.dates.length === 0 ? (
          <div style={{ color: '#8b949e' }}>暂无订单流水</div>
        ) : !selectedDay ? (
          <div style={{ color: '#8b949e' }}>共 {days.data.total} 个交易日有流水，请选择日期查看</div>
        ) : day.loading ? (
          <div style={{ color: '#8b949e' }}>加载中…</div>
        ) : (
          <DayFlow data={day.data} />
        )}
      </Panel>
    </div>
  )
}

function DayFlow({ data }: { data: SimDayData | null }) {
  if (!data) return <div style={{ color: '#8b949e' }}>无数据</div>
  const orders = data.orders ?? []
  const trades = data.trades ?? []
  return (
    <Row gutter={16}>
      <Col span={12}>
        <div style={{ marginBottom: 8, color: '#8b949e' }}>
          订单（{orders.length}）
        </div>
        <Table
          size="small"
          rowKey="order_id"
          dataSource={orders}
          pagination={false}
          scroll={{ y: 380 }}
          columns={[
            { title: '代码', dataIndex: 'ts_code', width: 110 },
            { title: '方向', dataIndex: 'side', width: 70, render: (s: string) => (s === 'buy' ? <Tag color="red">买入</Tag> : <Tag color="green">卖出</Tag>) },
            { title: '数量', dataIndex: 'quantity', align: 'right' as const },
            { title: '价格', dataIndex: 'price', align: 'right' as const },
            { title: '状态', dataIndex: 'status', width: 90, render: (s: string) => <Tag>{s}</Tag> },
            { title: '原因', dataIndex: 'reason', ellipsis: true },
          ]}
        />
      </Col>
      <Col span={12}>
        <div style={{ marginBottom: 8, color: '#8b949e' }}>
          成交（{trades.length}）
        </div>
        <Table
          size="small"
          rowKey="trade_id"
          dataSource={trades}
          pagination={false}
          scroll={{ y: 380 }}
          columns={[
            { title: '代码', dataIndex: 'ts_code', width: 110 },
            { title: '方向', dataIndex: 'side', width: 70, render: (s: string) => (s === 'buy' ? <Tag color="red">买入</Tag> : <Tag color="green">卖出</Tag>) },
            { title: '数量', dataIndex: 'quantity', align: 'right' as const },
            { title: '价格', dataIndex: 'price', align: 'right' as const },
            { title: '金额', dataIndex: 'amount', align: 'right' as const, render: (v: number) => fmtMoney(v) },
            { title: '佣金', dataIndex: 'commission', align: 'right' as const, render: (v: number) => fmtMoney(v) },
            { title: '印花税', dataIndex: 'stamp_tax', align: 'right' as const, render: (v: number) => fmtMoney(v) },
          ]}
        />
      </Col>
    </Row>
  )
}

function formatDate(d: string): string {
  return `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6, 8)}`
}

function positiveColor(value: number | null): string | undefined {
  if (value === null) return undefined
  return value >= 0 ? '#3fb950' : '#ff7b72'
}
