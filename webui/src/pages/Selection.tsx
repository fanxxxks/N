import { Button, Space, Table, Tag, Tooltip } from 'antd'
import { ReloadOutlined } from '@ant-design/icons'
import { api, useApi } from '../api/client'
import { ErrorText, Panel, fmtPercent } from '../components/common'

export default function SelectionPage() {
  const positions = useApi(() => api.positions(0, 1))
  const strategy = useApi(api.strategy)

  const latest = positions.data?.items[0]

  return (
    <div>
      <div className="page-head">
        <h2 className="page-title">选股</h2>
        <Space>
          <Tooltip title="展示最新一期的选股快照与生成它的因子公式">
            <span style={{ color: '#8b949e', fontSize: 12 }}>最新一期快照</span>
          </Tooltip>
          <Button
            icon={<ReloadOutlined />}
            onClick={() => {
              positions.reload()
              strategy.reload()
            }}
          >
            刷新
          </Button>
        </Space>
      </div>
      {(positions.error || strategy.error) && (
        <ErrorText message={positions.error ?? strategy.error ?? ''} />
      )}

      <Panel
        title="最优因子公式"
        loading={strategy.loading}
        extra={
          (strategy.data?.val_reward ?? strategy.data?.best_reward) != null ? (
            <Tag color="green">验证集奖励 {(strategy.data?.val_reward ?? strategy.data?.best_reward ?? 0).toFixed(3)}</Tag>
          ) : null
        }
      >
        {strategy.data?.formula_text ? (
          <div className="formula-box">{strategy.data.formula_text}</div>
        ) : (
          <div style={{ color: '#8b949e' }}>
            暂无策略文件 data/best_ashare_strategy.json，请先运行 python -m ashare_model.train
          </div>
        )}
      </Panel>

      <Panel
        title={
          latest
            ? `最新持仓快照（信号日 ${latest.signal_date} · 入场日 ${latest.entry_date} · 退出日 ${latest.exit_date} · ${latest.count} 只）`
            : '最新持仓快照'
        }
        loading={positions.loading}
      >
        {latest ? (
          <Table
            size="small"
            rowKey="ts_code"
            dataSource={latest.rows}
            pagination={false}
            scroll={{ y: 480 }}
            columns={[
              { title: '代码', dataIndex: 'ts_code', width: 120 },
              { title: '名称', dataIndex: 'name', width: 140 },
              {
                title: '权重',
                dataIndex: 'weight',
                render: (w: number) => <Tag color="blue">{fmtPercent(w)}</Tag>,
              },
            ]}
          />
        ) : (
          <div style={{ color: '#8b949e' }}>暂无持仓快照</div>
        )}
      </Panel>
    </div>
  )
}
