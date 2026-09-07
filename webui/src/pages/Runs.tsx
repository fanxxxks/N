import { useState } from 'react'
import { Table, Tag, Typography } from 'antd'
import { ReloadOutlined } from '@ant-design/icons'
import { Button, Col, Row, Space } from 'antd'
import { api, useApi } from '../api/client'
import { ErrorText, Panel } from '../components/common'
import type { RunInventoryEntry } from '../types'

const { Text } = Typography

/** Lifecycle-state colours: terminal non-success states must never look
 * like success (status visibility rule — unavailable/failed are not
 * rendered healthy). */
export function stateTag(state: string | null, error?: string | null) {
  if (error) return <Tag color="error">replay error</Tag>
  if (!state) return <Tag color="default">unknown</Tag>
  const palette: Record<string, string> = {
    IDEA: 'default',
    SPEC_LOCKED: 'default',
    DATA_QUALIFIED: 'processing',
    FACTOR_SET_QUALIFIED: 'processing',
    SEARCH_PLAN_ADMITTED: 'processing',
    OOS_QUALIFIED: 'processing',
    PAPER_OBSERVING: 'processing',
    PROMOTED: 'success',
    REJECTED: 'warning',
    FAILED: 'error',
    RETIRED: 'default',
  }
  return <Tag color={palette[state] ?? 'default'}>{state}</Tag>
}

export default function RunsPage() {
  const { data, loading, error, reload } = useApi(api.runs, [], 30000)
  const [selected, setSelected] = useState<RunInventoryEntry | null>(null)
  const detail = useApi(
    () =>
      selected
        ? api.runDetail(selected.spec_id, selected.run_id)
        : Promise.resolve(null),
    [selected?.spec_id, selected?.run_id],
  )

  const columns = [
    { title: 'spec', dataIndex: 'spec_id', key: 'spec_id' },
    { title: 'run', dataIndex: 'run_id', key: 'run_id' },
    {
      title: 'lifecycle state',
      key: 'state',
      render: (_: unknown, record: RunInventoryEntry) =>
        stateTag(record.state, record.error),
    },
    { title: 'artifacts', dataIndex: 'artifact_count', key: 'artifact_count' },
    { title: 'dataset', dataIndex: 'dataset_id', key: 'dataset_id' },
  ]

  return (
    <div>
      <div className="page-head">
        <h2 className="page-title">Runs (RunStore)</h2>
        <Button icon={<ReloadOutlined />} onClick={reload}>
          Refresh
        </Button>
      </div>
      {error && <ErrorText message={error} />}
      <Row gutter={[12, 12]}>
        <Col span={selected ? 12 : 24}>
          <Panel title="Run inventory (append-only RunStore replay)">
            <Table
              rowKey={(r) => `${r.spec_id}/${r.run_id}`}
              loading={loading}
              dataSource={data?.runs ?? []}
              columns={columns}
              pagination={false}
              rowClassName={(r) =>
                selected?.run_id === r.run_id ? 'ant-table-row-selected' : ''
              }
              onRow={(record) => ({
                onClick: () => setSelected(record),
                style: { cursor: 'pointer' },
              })}
              locale={{ emptyText: 'No runs recorded yet' }}
            />
          </Panel>
        </Col>
        {selected && (
          <Col span={12}>
            <Panel
              title={`Run detail: ${selected.spec_id}/${selected.run_id}`}
              extra={
                <Space>
                  <Text type="secondary">{selected.dataset_id}</Text>
                  <Button size="small" onClick={() => setSelected(null)}>
                    Close
                  </Button>
                </Space>
              }
            >
              {detail.error && <ErrorText message={detail.error} />}
              {detail.data && (
                <>
                  <p>
                    state: {stateTag(detail.data.state)} · trials:{' '}
                    {detail.data.trials.total} · artifacts:{' '}
                    {detail.data.artifacts.length}
                  </p>
                  <Table
                    rowKey={(e) => String(e.seq)}
                    size="small"
                    dataSource={detail.data.events}
                    pagination={false}
                    columns={[
                      { title: 'seq', dataIndex: 'seq', width: 60 },
                      { title: 'from', dataIndex: 'from_state' },
                      {
                        title: 'to',
                        dataIndex: 'to_state',
                        render: (s: string) => stateTag(s),
                      },
                      {
                        title: 'reason',
                        dataIndex: 'reason',
                        render: (r: string | null) => r ?? '—',
                      },
                    ]}
                  />
                </>
              )}
            </Panel>
          </Col>
        )}
      </Row>
    </div>
  )
}
