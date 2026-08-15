import { useEffect, useState } from 'react'
import { Layout, Menu, Tag, Typography } from 'antd'
import {
  DatabaseOutlined,
  ExperimentOutlined,
  FileTextOutlined,
  HomeOutlined,
  LineChartOutlined,
  StockOutlined,
} from '@ant-design/icons'
import { HashRouter, NavLink, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import Overview from './pages/Overview'
import BacktestPage from './pages/Backtest'
import SelectionPage from './pages/Selection'
import SimPage from './pages/Sim'
import DataStatusPage from './pages/DataStatus'
import LogsPage from './pages/Logs'
import { api } from './api/client'

const { Sider, Content, Header } = Layout

const MENU_ITEMS = [
  { key: '/', icon: <HomeOutlined />, label: '概览' },
  { key: '/backtest', icon: <LineChartOutlined />, label: '回测' },
  { key: '/selection', icon: <StockOutlined />, label: '选股' },
  { key: '/sim', icon: <ExperimentOutlined />, label: '模拟盘' },
  { key: '/data', icon: <DatabaseOutlined />, label: '数据状态' },
  { key: '/logs', icon: <FileTextOutlined />, label: '运行日志' },
]

function HealthBadge() {
  const [status, setStatus] = useState<'checking' | 'up' | 'down'>('checking')
  useEffect(() => {
    let alive = true
    const check = () =>
      api
        .health()
        .then(() => alive && setStatus('up'))
        .catch(() => alive && setStatus('down'))
    check()
    const timer = setInterval(check, 15000)
    return () => {
      alive = false
      clearInterval(timer)
    }
  }, [])
  const color = status === 'up' ? 'green' : status === 'down' ? 'red' : 'orange'
  const text = status === 'up' ? 'API 正常' : status === 'down' ? 'API 离线' : '检测中'
  return <Tag color={color}>{text}</Tag>
}

function Shell() {
  const location = useLocation()
  const selected = MENU_ITEMS.map((m) => m.key).includes(location.pathname)
    ? location.pathname
    : '/'
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sider width={200} theme="dark" className="app-sider">
        <div className="app-logo">
          <span className="app-logo-mark">α</span>
          <span className="app-logo-text">AlphaGPT</span>
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[selected]}
          items={MENU_ITEMS.map((m) => ({
            key: m.key,
            icon: m.icon,
            label: <NavLink to={m.key}>{m.label}</NavLink>,
          }))}
        />
        <div className="app-sider-footer">纯 A 股多因子研究</div>
      </Sider>
      <Layout>
        <Header className="app-header">
          <Typography.Text strong className="app-header-title">
            纯 A 股多因子模拟盘看板
          </Typography.Text>
          <HealthBadge />
        </Header>
        <Content className="app-content">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/backtest" element={<BacktestPage />} />
            <Route path="/selection" element={<SelectionPage />} />
            <Route path="/sim" element={<SimPage />} />
            <Route path="/data" element={<DataStatusPage />} />
            <Route path="/logs" element={<LogsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Content>
      </Layout>
    </Layout>
  )
}

export default function App() {
  return (
    <HashRouter>
      <Shell />
    </HashRouter>
  )
}
