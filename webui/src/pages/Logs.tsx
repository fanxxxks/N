import { useMemo, useState } from 'react'
import { Button, Col, Input, List, Row, Select, Space, Switch, Tag, Typography } from 'antd'
import { FileTextOutlined, ReloadOutlined } from '@ant-design/icons'
import { api, useApi } from '../api/client'
import { ErrorText, Panel, fmtSize } from '../components/common'
import type { LogContent, LogFile } from '../types'

const KIND_COLOR: Record<string, string> = {
  train: 'geekblue',
  backtest: 'blue',
  sim: 'purple',
  sync: 'cyan',
  pytest: 'default',
  other: 'default',
}

const KIND_LABEL: Record<string, string> = {
  train: '训练',
  backtest: '回测',
  sim: '模拟盘',
  sync: '数据同步',
  pytest: '测试',
  other: '其他',
}

export default function LogsPage() {
  const list = useApi(api.logs, [], 15000)
  const [selected, setSelected] = useState<string | null>(null)
  const [tail, setTail] = useState(1000)
  const [search, setSearch] = useState('')
  const [autoRefresh, setAutoRefresh] = useState(true)
  const content = useApi<LogContent | null>(
    () => (selected ? api.logContent(selected, tail) : Promise.resolve(null)),
    [selected, tail],
    autoRefresh && selected ? 10000 : undefined,
  )

  const filtered = useMemo(() => {
    const files = list.data ?? []
    if (!search.trim()) return files
    const q = search.trim().toLowerCase()
    return files.filter((f) => f.name.toLowerCase().includes(q) || f.kind.includes(q))
  }, [list.data, search])

  const lines = useMemo(() => {
    const text = content.data?.content ?? ''
    if (!text) return []
    return text.split('\n').map((line, i) => (
      <div key={i} className={lineClass(line)}>
        {line || '\u00a0'}
      </div>
    ))
  }, [content.data?.content])

  return (
    <div>
      <div className="page-head">
        <h2 className="page-title">运行日志</h2>
        <Space>
          <Switch
            checked={autoRefresh}
            onChange={setAutoRefresh}
            checkedChildren="自动刷新"
            unCheckedChildren="手动"
          />
          <Button icon={<ReloadOutlined />} onClick={() => { list.reload(); content.reload() }}>
            刷新
          </Button>
        </Space>
      </div>
      {list.error && <ErrorText message={list.error} />}

      <Row gutter={16}>
        <Col span={7}>
          <Panel
            title={`日志文件（${filtered.length}）`}
            loading={list.loading}
            extra={
              <Input
                size="small"
                placeholder="搜索文件名"
                allowClear
                style={{ width: 140 }}
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            }
          >
            <List
              size="small"
              dataSource={filtered.slice(0, 200)}
              style={{ height: 620, overflow: 'auto' }}
              renderItem={(item: LogFile) => (
                <List.Item
                  className={item.name === selected ? 'log-file-active' : ''}
                  onClick={() => setSelected(item.name)}
                  style={{ cursor: 'pointer', padding: '8px 10px', borderRadius: 6 }}
                >
                  <List.Item.Meta
                    avatar={<FileTextOutlined style={{ color: '#8b949e' }} />}
                    title={
                      <Space size={4}>
                        <span style={{ fontFamily: 'monospace', fontSize: 12 }}>{item.name}</span>
                        <Tag color={KIND_COLOR[item.kind]} style={{ fontSize: 11 }}>
                          {KIND_LABEL[item.kind] ?? item.kind}
                        </Tag>
                      </Space>
                    }
                    description={
                      <Typography.Text type="secondary" style={{ fontSize: 11 }}>
                        {fmtSize(item.size)} · {item.mtime}
                      </Typography.Text>
                    }
                  />
                </List.Item>
              )}
            />
          </Panel>
        </Col>
        <Col span={17}>
          <Panel
            title={
              selected
                ? `${selected}${content.data?.lines != null ? ` · 已展示 ${content.data.lines} 行` : ''}${content.data?.truncated ? ' · 仅尾部 16MB' : ''}`
                : '日志内容'
            }
            loading={content.loading && Boolean(selected)}
            extra={
              <Space>
                <Select
                  size="small"
                  value={tail}
                  style={{ width: 110 }}
                  onChange={setTail}
                  options={[
                    { value: 300, label: '尾部 300 行' },
                    { value: 1000, label: '尾部 1000 行' },
                    { value: 5000, label: '尾部 5000 行' },
                    { value: 20000, label: '尾部 20000 行' },
                  ]}
                />
              </Space>
            }
          >
            {!selected ? (
              <div style={{ color: '#8b949e', padding: 24 }}>从左侧选择一个日志文件</div>
            ) : content.error ? (
              <ErrorText message={content.error} />
            ) : (
              <div className="log-viewer">{lines}</div>
            )}
          </Panel>
        </Col>
      </Row>
    </div>
  )
}

function lineClass(line: string): string {
  if (line.includes('| ERROR') || line.includes('ERROR |')) return 'log-line-error'
  if (line.includes('| WARNING') || line.includes('WARNING |')) return 'log-line-warning'
  if (line.includes('| SUCCESS') || line.includes('SUCCESS |')) return 'log-line-success'
  if (line.includes('| DEBUG') || line.includes('DEBUG |')) return 'log-line-debug'
  return ''
}
