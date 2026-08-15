# AlphaGPT 纯 A 股多因子模拟盘

将原 AlphaGPT 从 Solana meme 链上因子系统改造为纯 A 股横截面多因子量化研究与模拟盘工具。保留 Transformer 可解释因子公式生成、StackVM 解释执行和回测评分训练；数据使用 AkShare，本地 DuckDB/Parquet 存储，Streamlit 看板。

## 目录结构

- `ashare_data/`：AkShare 数据获取、交易日历、DuckDB/Parquet、清洗复权、股票池。
- `ashare_model/`：因子、公式词表、算子、StackVM、Transformer 生成器、训练与回测。
- `ashare_trading/`：模拟券商撮合、组合管理、风控、日频模拟运行器。
- `dashboard/`：Streamlit 研究/模拟盘看板。
- `config/ashare_config.yaml`：非敏感配置。
- `_obsolete_crypto/`：原加密链路与 `times.py` 等遗留脚本归档，不再参与运行。

## 安装

```bash
python -m pip install -r requirements.txt
```

复制 `config/.env.example` 为 `config/.env` 并按需填写。

## 运行入口

```bash
python -m ashare_data.sync
python -m ashare_model.train
python -m ashare_model.backtest
python -m ashare_trading.run_sim
streamlit run dashboard/app.py
```

### 模拟盘的启动 / 续跑 / 重置

`run_sim` 的每个交易日是一个事务：订单/成交流水、资金曲线与 `last_exec_date`
水位线在同一份原子快照中落盘。因此：

```bash
python -m ashare_trading.run_sim            # 首次运行：从数据集起点重放
python -m ashare_trading.run_sim --resume   # 从状态文件最后处理日期续跑
python -m ashare_trading.run_sim --reset    # 清空状态，从头重放
```

当状态文件已含历史（`has_history`）时，不带 `--reset` / `--resume` 直接启动会
报错退出，防止把历史订单在现有持仓上重放一遍污染状态。`--start` 早于
`last_exec_date` 且非 `--reset` 同样被拒绝。运行进度（phase / 当前日期 / 净值）
实时写入 `data/sim_progress.json`，供前端状态条轮询。

## Web 前端（React + FastAPI）

除 Streamlit 看板外，仓库还提供一套现代 Web 前端：FastAPI 后端读取本地
DuckDB / JSON 产物与日志，React (Vite + Ant Design + ECharts) 前端展示
概览、回测、选股、模拟盘、数据状态与运行日志六个页面。

开发模式（前后端分离，Vite 自动代理 `/api`）：

```bash
# 终端 1：后端（需先 pip install fastapi，uvicorn 已在 requirements 中）
python -m uvicorn webapi.app:app --host 127.0.0.1 --port 8000 --reload

# 终端 2：前端（首次需 cd webui && npm install）
cd webui && npm run dev
# 浏览器打开 http://127.0.0.1:5173
```

生产模式（单服务，后端托管构建产物）：

```bash
cd webui && npm install && npm run build && cd ..
python -m uvicorn webapi.app:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

前端数据以只读展示为主；模拟盘页提供“紧急停止”（写入 `STOP_SIGNAL`）与
配置编辑（`PUT /api/sim/config`）。配置编辑写入全局运行时覆盖文件
`config/runtime_overrides.yaml`（已被 gitignore，不污染 YAML 基线），
`run_sim` / `backtest` / `train` 所有入口都会在 YAML 之上合并它。费用
（佣金 / 最低佣金 / 印花税 / 过户费 / 滑点）是全项目单一口径（`backtest`
段），修改后对回测与模拟盘同时生效；初始资金在下次 reset 时生效。日志页
仅展示 `logs/` 与 `data/` 下的 `.log` / `.txt` 文件。

首次数据同步会拉取沪深 300、中证 500、中证 1000 成分股及日线数据。为避免重复请求 AkShare，可先使用 `--offline` 测试本地流程，或使用 `--limit N` 限制股票数量。日线缓存落后于交易日历时会自动刷新；全量（不带 `--limit`）同步会清理不在股票池中的历史行。

## 测试

```bash
python -m pytest -q tests
```

## 实验留档（experiments/）

每次训练 / 回测 / 模拟盘跑完后，用 `scripts/archive_run.py` 把 {配置、公式、模型、指标、日期}
快照到 `experiments/<YYYYMMDD>_<公式>/` 并提交，让"哪个公式 + 哪份配置 + 什么结果"永久可追溯：

```bash
python scripts/archive_run.py --mode backtest --commit
python scripts/archive_run.py --mode train --commit
python scripts/archive_run.py --mode sim --commit
```

每个实验目录包含 `manifest.json`（运行模式、代码 commit、脏工作区标记、数据末端日期、
各产物 SHA-256）、`formula.json`、`config.yaml`、`metrics_summary.json` 与 `metrics.json`
（超过 `--max-metrics-size-mb` 时仅保留摘要）、`model.*`（超过 `--max-model-size-mb` 时只记录
哈希、权重留在本地 `data/`）。`--commit` 只提交本次实验目录，不会 push，也不触碰其他改动。
详见 `experiments/README.md`。

## 与现实对应的关键设计

- **无未来泄漏**：因子只使用当前及历史截面；`open_to_open_returns` 以 t+1 开盘买入、t+2 开盘卖出为目标收益；停牌/未上市日被掩码为 0，绝不产生虚假收益。
- **交易规则**：A 股 T+1、买入 100 股整手、卖出可零股清仓、涨停不买/跌停不卖、一字板判定、ST 股 5% 涨跌停（按股票名称识别）。
- **费用模型**：佣金万 2.5（最低 5 元）、印花税卖出 0.05%、过户费 0.001%、滑点 0.05%；回测与训练奖励使用同一套费用口径。
- **涨跌停事件因子**：`LIMIT_UP_EVENT`/`LIMIT_DOWN_EVENT` 由一字板真实计算（创业板/科创板 20%，其余 10%）。
- **训练**：REINFORCE + value baseline；最佳公式在训练窗尾部的验证集上选取（`model.validation_fraction`，默认 0.2），避免纯样本内过拟合。
- **回测输出**：包含持仓快照与全市场等权基准（与策略同一 open-to-open 口径），供看板展示。

## 已知局限（有意保留）

- **成分股为当前快照**：AkShare 免费接口无逐日历史成分，`constituents` 使用当前成分并统一标记为全区间成员，存在幸存者偏差。
- **中性占位因子**：`NORTHBOUND_CHG`/`MARGIN_BALANCE_CHG`/`INDUSTRY_MOMENTUM` 需要北向、两融与行业历史数据，暂保持中性（0）；词表保持不变以兼容已保存的公式 token。
- **换手率缺失即缺失**：换手率依赖流通股本，无法从 OHLCV 反推；缺失时保持中性而非伪造常数。
- **基本面快照**（PE/PB/ROE 等）仅在数据末端日期参与截面（点-in-time 安全），其余日期中性。
- **离线日历**：`--offline` 模式用工作日近似，包含节假日，仅用于开发与测试。
