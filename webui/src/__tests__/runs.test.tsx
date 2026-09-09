import { beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import RunsPage, { stateTag } from '../pages/Runs'

/** Contract: the Runs page reads the RunStore-backed /api/runs view and
 * renders every lifecycle state with visibility rules — terminal
 * non-success states (REJECTED/FAILED) and replay errors must never
 * render as success (E2/t26 re-dispatch). */

const RUNS_PAYLOAD = {
  runs: [
    {
      spec_id: 'spec-a',
      run_id: 'run-1',
      state: 'PROMOTED',
      artifact_count: 3,
      dataset_id: 'ds-1',
      error: null,
    },
    {
      spec_id: 'spec-b',
      run_id: 'run-2',
      state: 'FAILED',
      artifact_count: 1,
      dataset_id: 'ds-2',
      error: null,
    },
  ],
  runs_dir: '/tmp/runs',
}

const DETAIL_PAYLOAD = {
  spec_id: 'spec-a',
  run_id: 'run-1',
  state: 'PROMOTED',
  dataset_id: 'ds-1',
  research_domain: null,
  git_commit: 'abc',
  artifacts: [{ artifact_id: 'a1', artifact_type: 'runspec' }],
  events: [
    {
      seq: 1,
      from_state: null,
      to_state: 'IDEA',
      completed: true,
      reason: null,
      ts: '2026-01-01T00:00:00',
    },
    {
      seq: 2,
      from_state: 'IDEA',
      to_state: 'SPEC_LOCKED',
      completed: true,
      reason: null,
      ts: '2026-01-01T00:00:01',
    },
    {
      seq: 3,
      from_state: 'SPEC_LOCKED',
      to_state: 'PROMOTED',
      completed: true,
      reason: null,
      ts: '2026-01-01T00:00:02',
    },
  ],
  trials: { total: 2, by_status: { completed: 2 } },
}

function mockFetch(routes: Record<string, unknown>) {
  const keys = Object.keys(routes).sort((a, b) => b.length - a.length)
  return vi.fn(async (url: string) => {
    const match = keys.find((key) => url.startsWith(key))
    if (!match) throw new Error(`unmocked url ${url}`)
    return {
      ok: true,
      status: 200,
      json: async () => routes[match],
    } as Response
  })
}

beforeEach(() => {
  vi.stubGlobal('fetch', mockFetch({}))
})

describe('stateTag visibility rules', () => {
  it('renders PROMOTED as success', () => {
    const { container } = render(stateTag('PROMOTED'))
    expect(container.textContent).toBe('PROMOTED')
  })

  it('renders FAILED and REJECTED as distinct non-success tags', () => {
    const failed = render(stateTag('FAILED'))
    expect(failed.container.textContent).toBe('FAILED')
    const rejected = render(stateTag('REJECTED'))
    expect(rejected.container.textContent).toBe('REJECTED')
  })

  it('renders replay errors as errors, never as a healthy state', () => {
    const { container } = render(stateTag(null, 'replay broke'))
    expect(container.textContent).toBe('replay error')
  })
})

describe('RunsPage', () => {
  it('renders the run inventory from the RunStore-backed endpoint', async () => {
    vi.stubGlobal('fetch', mockFetch({ '/api/runs': RUNS_PAYLOAD }))
    render(<RunsPage />)
    await waitFor(() => {
      expect(screen.getByText('spec-a')).toBeInTheDocument()
    })
    expect(screen.getByText('spec-b')).toBeInTheDocument()
    expect(screen.getByText('ds-1')).toBeInTheDocument()
  })

  it('loads the detail panel on row click and shows the event trail', async () => {
    const fetchMock = mockFetch({
      '/api/runs': RUNS_PAYLOAD,
      '/api/runs/spec-a/run-1': DETAIL_PAYLOAD,
    })
    vi.stubGlobal('fetch', fetchMock)
    render(<RunsPage />)
    await waitFor(() => screen.getByText('spec-a'))
    '// antd onRow: dispatch synchronously on the row'
    fireEvent.click(screen.getByText('spec-a'))
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith('/api/runs/spec-a/run-1', undefined)
    })
    // The detail panel renders the trial summary from the RunStore view.
    await waitFor(() => {
      expect(screen.getByText(/trials: 2/)).toBeInTheDocument()
    })
  })
})
