import { Button, Col, Row, Space, Table, Tag } from 'antd'
import { CheckCircleOutlined, CloseCircleOutlined, ReloadOutlined } from '@ant-design/icons'
import { api, useApi } from '../api/client'
import { ErrorText, MetricCard, Panel, fmtSize } from '../components/common'
import type { Artifact } from '../types'

export default function DataStatusPage() {
  const { data, loading, error, reload } = useApi(api.dataStatus, [], 30000)

  const db = data?.db
  const artifacts = data?.artifacts ?? []
  const config = data?.config

  return (
    <div>
      <div className="page-head">
        <h2 className="page-title">数据状态</h2>
        <Button icon={<ReloadOutlined />} onClick={reload}>
          刷新
        </Button>
      </div>
      {error && <ErrorText message={error} />}

      <Row gutter={[12, 12]}>
        <Col span={4}>
          <MetricCard
            title="数据库"
            value={db?.ready ? '就绪' : '不可用'}
            suffix={db?.ready ? undefined : db?.reason ? '' : ''}
            color={db?.ready ? '#3fb950' : '#ff7b72'}
          />
        </Col>
        <Col span={5}>
          <MetricCard title="股票数" value={db?.stocks ?? '—'} />
        </Col>
        <Col span={5}>
          <MetricCard title="日线行数" value={db?.daily_rows != null ? db.daily_rows.toLocaleString() : '—'} />
        </Col>
        <Col span={5}>
          <MetricCard title="最早交易日" value={db?.first_trade_date ?? '—'} />
        </Col>
        <Col span={5}>
          <MetricCard title="最新交易日" value={db?.last_trade_date ?? '—'} />
        </Col>
      </Row>

      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col span={14}>
          <Panel title="产出文件" loading={loading}>
            <Table<Artifact>
              size="small"
              rowKey="name"
              dataSource={artifacts}
              pagination={false}
              columns={[
                {
                  title: '文件',
                  dataIndex: 'name',
                  render: (name: string, row) => (
                    <Space>
                      {row.exists ? (
                        <CheckCircleOutlined style={{ color: '#3fb950' }} />
                      ) : (
                        <CloseCircleOutlined style={{ color: '#6e7681' }} />
                      )}
                      <span style={{ fontFamily: 'monospace' }}>{name}</span>
                    </Space>
                  ),
                },
                { title: '大小', dataIndex: 'size', align: 'right', render: (s: number) => (s > 0 ? fmtSize(s) : '—') },
                { title: '更新时间', dataIndex: 'mtime', align: 'right', render: (v: string) => v || '—' },
              ]}
            />
          </Panel>
        </Col>
        <Col span={10}>
          <Panel title="运行配置" loading={loading}>
            <table className="kv-table">
              <tbody>
                <tr>
                  <td>数据区间</td>
                  <td>
                    {config?.date_range?.start ?? '—'} ~ {config?.date_range?.end ?? '—'}
                  </td>
                </tr>
                <tr>
                  <td>股票池指数</td>
                  <td>
                    {(config?.universe?.indexes ?? []).join('、') || '—'}
                  </td>
                </tr>
                <tr>
                  <td>模型结构</td>
                  <td>
                    {config?.model?.d_model != null
                      ? `d=${config.model.d_model} · heads=${config.model.nhead} · layers=${config.model.num_layers}`
                      : '—'}
                  </td>
                </tr>
                <tr>
                  <td>训练配置</td>
                  <td>
                    {config?.model?.train_steps != null
                      ? `${config.model.train_steps} 步 · batch ${config.model.batch_size} · 公式长度 ≤ ${config.model.max_formula_len}`
                      : '—'}
                  </td>
                </tr>
                <tr>
                  <td>训练截止日</td>
                  <td>{config?.backtest?.train_end_date ?? '—'}</td>
                </tr>
                <tr>
                  <td>回测持仓数</td>
                  <td>{config?.backtest?.top_n ?? '—'} 只 · 单票上限 {(Number(config?.backtest?.single_weight_cap ?? 0) * 100).toFixed(1)}%</td>
                </tr>
                <tr>
                  <td>费用模型</td>
                  <td>
                    佣金 {(Number(config?.backtest?.commission_rate ?? 0) * 10000).toFixed(1)}‱ · 印花税 {(Number(config?.backtest?.stamp_tax_rate ?? 0) * 100).toFixed(2)}% · 滑点 {(Number(config?.backtest?.slippage_rate ?? 0) * 100).toFixed(2)}%
                  </td>
                </tr>
              </tbody>
            </table>
            <div style={{ marginTop: 12 }}>
              <Tag color="blue">基准：{config?.backtest?.benchmark ?? '全市场等权'}</Tag>
              <Tag color="purple">A 股规则：T+1 · 整手买入 · 涨跌停限制</Tag>
            </div>
          </Panel>
        </Col>
      </Row>
    </div>
  )
}
