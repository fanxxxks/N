import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'

const AXIS_STYLE = {
  axisLine: { lineStyle: { color: '#30363d' } },
  axisLabel: { color: '#8b949e' },
  splitLine: { lineStyle: { color: '#21262d' } },
}

export interface SeriesDef {
  name: string
  data: (number | null)[]
  dashed?: boolean
  yAxisIndex?: number
}

export function LineChart({
  x,
  series,
  title,
  height = 380,
  percent = false,
  legend = true,
}: {
  x: string[]
  series: SeriesDef[]
  title?: string
  height?: number
  percent?: boolean
  legend?: boolean
}) {
  const hasSecondAxis = series.some((s) => s.yAxisIndex === 1)
  const option: EChartsOption = {
    backgroundColor: 'transparent',
    title: title
      ? { text: title, textStyle: { color: '#c9d1d9', fontSize: 14 } }
      : undefined,
    tooltip: { trigger: 'axis' },
    legend: legend
      ? { top: 28, textStyle: { color: '#8b949e' }, itemWidth: 18 }
      : undefined,
    grid: { left: 56, right: hasSecondAxis ? 56 : 24, top: legend ? 60 : 36, bottom: 36 },
    xAxis: {
      type: 'category',
      data: x,
      ...AXIS_STYLE,
      axisLabel: { ...AXIS_STYLE.axisLabel, formatter: (v: string) => v.slice(2, 10) },
    },
    yAxis: [
      {
        type: 'value',
        ...AXIS_STYLE,
        scale: true,
        axisLabel: {
          ...AXIS_STYLE.axisLabel,
          formatter: (v: number) =>
            percent ? `${(v * 100).toFixed(1)}%` : v.toFixed(3),
        },
      },
      ...(hasSecondAxis
        ? [
            {
              type: 'value' as const,
              ...AXIS_STYLE,
              scale: true,
            },
          ]
        : []),
    ],
    dataZoom: x.length > 300 ? [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 8 }] : undefined,
    series: series.map((s) => ({
      name: s.name,
      type: 'line',
      data: s.data,
      showSymbol: false,
      smooth: false,
      yAxisIndex: s.yAxisIndex ?? 0,
      lineStyle: { width: 1.6, type: s.dashed ? 'dashed' : 'solid' },
      itemStyle: { color: undefined },
    })),
  }
  return (
    <ReactECharts
      option={option}
      style={{ height, width: '100%' }}
      notMerge
      opts={{ renderer: 'canvas' }}
    />
  )
}

export function EquityChart({
  dates,
  equity,
  benchmarkEquity,
  benchmarkName = '基准',
  title,
  height = 380,
}: {
  dates: string[]
  equity: number[]
  benchmarkEquity?: number[] | null
  benchmarkName?: string
  title?: string
  height?: number
}) {
  const series: SeriesDef[] = [
    { name: '策略净值', data: equity, dashed: false },
  ]
  if (benchmarkEquity && benchmarkEquity.length === equity.length) {
    series.push({ name: benchmarkName, data: benchmarkEquity, dashed: true })
  }
  return (
    <LineChart
      x={dates}
      series={series}
      title={title}
      height={height}
      legend={Boolean(benchmarkEquity?.length)}
    />
  )
}
