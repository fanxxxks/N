import { useState } from 'react'
import {
  Alert,
  Button,
  Col,
  Form,
  InputNumber,
  Modal,
  Row,
  Select,
  Space,
  Table,
  Tag,
  Tooltip,
  message,
} from 'antd'
import {
  PlayCircleOutlined,
  RedoOutlined,
  ReloadOutlined,
  SettingOutlined,
  StopOutlined,
} from '@ant-design/icons'
import { api, useApi } from '../api/client'
import { ErrorText, MetricCard, Panel, fmtMoney, fmtPercent } from '../components/common'
import { EquityChart } from '../components/charts'
import type { SimConfigData, SimDayData, SimRunStatus } from '../types'

const ACTIVE_STATES = new Set(['starting', 'running', 'stopping'])

const STATE_META: Record<string, { color: string; label: string }> = {
  idle: { color: 'default', label: '空闲' },
  starting: { color: 'processing', label: '启动中' },
  running: { color: 'processing', label: '运行中' },
  stopping: { color: 'warning', label: '停止中' },
  stopped: { color: 'warning', label: '已停止' },
  finished: { color: 'success', label: '已完成' },
  error: { color: 'error', label: '出错' },
}

const PHASE_LABEL: Record<string, string> = {
  loading: '加载数据与因子中…',
  executing: '逐日撮合中',
  stopped: '收到停止信号',
  finished: '运行完成',
  error: '运行出错',
}

export default function SimPage() {
  const state = useApi(api.sim, [], 15000)
  const days = useApi(api.simDays)
  const status = useApi(api.simStatus, [], 3000)
  const [selectedDay, setSelectedDay] = useState<string | null>(null)
  const day = useApi<SimDayData | null>(
    () => (selectedDay ? api.simDay(selectedDay) : Promise.resolve(null)),
    [selectedDay],
  )
  const [stopping, setStopping] = useState(false)
  const [starting, setStarting] = useState(false)
  const [resetting, setResetting] = useState(false)
  const [configOpen, setConfigOpen] = useState(false)

  const equityDates = state.data?.equity_history.map((h) => h.trade_date) ?? []
  const equityValues = state.data?.equity_history.map((h) => h.equity) ?? []
  const init = state.data?.initial_capital ?? 0
  const totalReturn = init > 0 && equityValues.length ? equityValues[equityValues.length - 1] / init - 1 : null
  const runState = status.data?.state ?? 'idle'
  const active = ACTIVE_STATES.has(runState)
  const hasHistory = (state.data?.trade_count ?? 0) > 0 || equityValues.length > 0

  const reloadAll = () => {
    state.reload()
    days.reload()
    status.reload()
  }

  const handleStart = (reset: boolean) => {
    const doStart = async () => {
      setStarting(true)
      try {
        await api.simStart({ reset })
        message.success(reset ? '已重置并启动，从数据集起点重放' : '已启动，将续跑或从头重放')
        reloadAll()
      } catch (err) {
        message.error(`启动失败：${(err as Error).message}`)
      } finally {
        setStarting(false)
      }
    }
    if (reset) {
      Modal.confirm({
        title: '重置并重新开始？',
        content: '旧状态将先归档到 experiments/，随后清空持仓并从头重放全部历史。该操作不可撤销。',
        okText: '确认重置并启动',
        cancelText: '取消',
        okButtonProps: { danger: true },
        onOk: doStart,
      })
    } else {
      void doStart()
    }
  }

  const handleStop = () => {
    Modal.confirm({
      title: '停止模拟盘？',
      content: '先写入 STOP_SIGNAL（下一个交易日循环退出），宽限期后自动终止进程。',
      okText: '确认停止',
      cancelText: '取消',
      okButtonProps: { danger: true },
      onOk: async () => {
        setStopping(true)
        try {
          await api.simStop()
          message.success('已发送停止信号')
          reloadAll()
        } catch (err) {
          message.error(`停止失败：${(err as Error).message}`)
        } finally {
          setStopping(false)
        }
      },
    })
  }

  const handleReset = () => {
    Modal.confirm({
      title: '重置模拟盘？',
      content: '当前状态将先归档到 experiments/（归档失败会中止重置），随后清空持仓与资金曲线。',
      okText: '确认重置',
      cancelText: '取消',
      okButtonProps: { danger: true },
      onOk: async () => {
        setResetting(true)
        try {
          const res = await api.simReset()
          if (res.ok) {
            message.success('模拟盘已重置（旧状态已归档）')
          } else {
            message.error(`重置失败：${res.reason ?? '未知错误'}`)
          }
          reloadAll()
        } catch (err) {
          message.error(`重置失败：${(err as Error).message}`)
        } finally {
          setResetting(false)
        }
      },
    })
  }

  return (
    <div>
      <div className="page-head">
        <h2 className="page-title">模拟盘</h2>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={reloadAll}>
            刷新
          </Button>
          <Button
            icon={<SettingOutlined />}
            disabled={active}
            onClick={() => setConfigOpen(true)}
          >
            配置
          </Button>
          <Button
            icon={<RedoOutlined />}
            danger
            loading={resetting}
            disabled={active}
            onClick={handleReset}
          >
            重置
          </Button>
          <Button
            type="primary"
            icon={<PlayCircleOutlined />}
            loading={starting}
            disabled={active}
            onClick={() => handleStart(false)}
          >
            {hasHistory ? '继续运行' : '启动'}
          </Button>
          <Tooltip title="写入 STOP_SIGNAL，宽限期后自动终止进程">
            <Button danger icon={<StopOutlined />} loading={stopping} disabled={!active} onClick={handleStop}>
              停止
            </Button>
          </Tooltip>
        </Space>
      </div>

      <RunStatusBar status={status.data} hasHistory={hasHistory} onRestart={() => handleStart(true)} />

      {state.error && <ErrorText message={state.error} />}
      {status.error && <ErrorText message={status.error} />}

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

      <ConfigModal open={configOpen} onClose={() => setConfigOpen(false)} onSaved={reloadAll} />
    </div>
  )
}

function RunStatusBar({
  status,
  hasHistory,
  onRestart,
}: {
  status: SimRunStatus | null
  hasHistory: boolean
  onRestart: () => void
}) {
  if (!status) return null
  const meta = STATE_META[status.state] ?? { color: 'default', label: status.state }
  const detail = statusDetail(status)
  if (status.state === 'error') {
    return (
      <Alert
        style={{ marginBottom: 12 }}
        type="error"
        showIcon
        message={`模拟盘运行出错${status.pid ? `（pid=${status.pid}）` : ''}`}
        description={
          <Space direction="vertical" size={2}>
            <span>{status.error ?? '未知错误'}</span>
            {status.log_path && <span>日志：{status.log_path}</span>}
            <Button size="small" danger onClick={onRestart}>
              重置并重新开始
            </Button>
          </Space>
        }
      />
    )
  }
  return (
    <div className="sim-status-bar" style={{ marginBottom: 12 }}>
      <Space size="middle" wrap>
        <Tag color={meta.color}>{meta.label}</Tag>
        {status.pid ? <span>pid={status.pid}</span> : null}
        <span>{detail}</span>
        {status.current_date ? <span>当前日期：{formatDate(status.current_date)}</span> : null}
        {status.equity != null ? <span>净值：{fmtMoney(status.equity)}</span> : null}
        {status.state === 'idle' && hasHistory && (
          <span style={{ color: '#8b949e' }}>存在历史状态，点击“继续运行”将自动续跑</span>
        )}
      </Space>
    </div>
  )
}

function statusDetail(status: SimRunStatus): string {
  if (status.phase && PHASE_LABEL[status.phase]) return PHASE_LABEL[status.phase]
  switch (status.state) {
    case 'idle':
      return '未在运行'
    case 'starting':
      return '进程启动中'
    case 'stopping':
      return '等待进程退出…'
    case 'stopped':
      return '已被停止'
    case 'finished':
      return '已运行完毕'
    default:
      return ''
  }
}

function ConfigModal({
  open,
  onClose,
  onSaved,
}: {
  open: boolean
  onClose: () => void
  onSaved: () => void
}) {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [config, setConfig] = useState<SimConfigData | null>(null)

  const load = async () => {
    setLoading(true)
    try {
      const data = await api.simConfig()
      setConfig(data)
      form.setFieldsValue({
        initial_capital: data.effective.initial_capital,
        max_positions: data.effective.max_positions,
        commission_rate: data.effective.commission_rate,
        min_commission: data.effective.min_commission,
        stamp_tax_rate: data.effective.stamp_tax_rate,
        transfer_fee_rate: data.effective.transfer_fee_rate,
        slippage_rate: data.effective.slippage_rate,
      })
    } catch (err) {
      message.error(`读取配置失败：${(err as Error).message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    try {
      const values = await form.validateFields()
      setSaving(true)
      const patch: Record<string, number> = {
        initial_capital: values.initial_capital,
        max_positions: values.max_positions,
        commission_rate: values.commission_rate,
        min_commission: values.min_commission,
        stamp_tax_rate: values.stamp_tax_rate,
        transfer_fee_rate: values.transfer_fee_rate,
        slippage_rate: values.slippage_rate,
      }
      const data = await api.simConfigPut(patch)
      setConfig(data)
      message.success('配置已保存到运行时覆盖文件')
      onSaved()
      onClose()
    } catch (err) {
      if (err && typeof err === 'object' && 'errorFields' in (err as Record<string, unknown>)) {
        return // antd renders field-level validation errors
      }
      message.error(`保存失败：${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setSaving(false)
    }
  }

  const pendingReset = config?.pending_reset ?? false

  return (
    <Modal
      title="模拟盘配置"
      open={open}
      onCancel={onClose}
      afterOpenChange={(visible) => {
        if (visible) void load()
      }}
      onOk={handleSave}
      okText="保存"
      cancelText="取消"
      confirmLoading={saving}
      destroyOnClose
    >
      <div style={{ marginBottom: 12 }}>
        {pendingReset && (
          <Alert
            type="warning"
            showIcon
            message="初始资金与当前状态不一致，将在下次重置时生效"
          />
        )}
        {!pendingReset && (
          <Alert
            type="info"
            showIcon
            message="费用（佣金/印花税/过户费/滑点）为全项目单一口径，保存后对回测与模拟盘同时生效"
          />
        )}
      </div>
      <Form form={form} layout="vertical" disabled={loading}>
        <Row gutter={12}>
          <Col span={12}>
            <Form.Item name="initial_capital" label="初始资金（重置时生效）" rules={[{ required: true }]}>
              <InputNumber min={1} step={1000} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item name="max_positions" label="最大持仓数" rules={[{ required: true }]}>
              <InputNumber min={1} max={500} step={1} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
        </Row>
        <Row gutter={12}>
          <Col span={12}>
            <Form.Item name="commission_rate" label="佣金率（0.00025 = 万2.5）" rules={[{ required: true }]}>
              <InputNumber min={0} max={1} step={0.00001} stringMode style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item name="min_commission" label="最低佣金（元）" rules={[{ required: true }]}>
              <InputNumber min={0} step={0.5} style={{ width: '100%' }} />
            </Form.Item>
          </Col>
        </Row>
        <Row gutter={12}>
          <Col span={12}>
            <Form.Item name="stamp_tax_rate" label="印花税率（卖出）" rules={[{ required: true }]}>
              <InputNumber min={0} max={1} step={0.00001} stringMode style={{ width: '100%' }} />
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item name="transfer_fee_rate" label="过户费率" rules={[{ required: true }]}>
              <InputNumber min={0} max={1} step={0.00001} stringMode style={{ width: '100%' }} />
            </Form.Item>
          </Col>
        </Row>
        <Form.Item name="slippage_rate" label="滑点率" rules={[{ required: true }]}>
          <InputNumber min={0} max={1} step={0.00001} stringMode style={{ width: '100%' }} />
        </Form.Item>
      </Form>
    </Modal>
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
