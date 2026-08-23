# AlphaGPT 纯 A 股多因子模拟盘

将原 AlphaGPT 从 Solana meme 链上因子系统改造为纯 A 股横截面多因子量化研究与模拟盘工具。保留 Transformer 可解释因子公式生成、StackVM 解释执行和回测评分训练；数据使用 AkShare，本地 DuckDB/Parquet 存储，Streamlit 看板。

## 目录结构

- `ashare_data/`：AkShare 数据获取、交易日历、DuckDB/Parquet、清洗复权、股票池。
- `ashare_model/`：因子、公式词表、算子、StackVM、Transformer 生成器、训练与回测。
- `ashare_trading/`：模拟券商撮合、组合管理、风控、日频模拟运行器。
- `dashboard/`：Streamlit 研究/模拟盘看板。
- `config/ashare_config.yaml`：非敏感配置。
- 原加密链路（`times.py` 等）与独立 grokking 实验（`lord/`）已从主线移除，可在 tag `archive/lord-and-crypto` 检回。

## 安装

```bash
python -m pip install -r requirements.txt
```

复制 `config/.env.example` 为 `config/.env` 并按需填写。

### 可选：启用 NVIDIA GPU 加速

训练/协议入口默认 `--device auto`（有 CUDA 用 CUDA，否则 CPU，CI 无需 GPU）。
有 NVIDIA 显卡时可用 CUDA 版 torch 替代 CPU 版（训练窗口因子张量约 650MB，
6GB 显存即可运行）：

```bash
python -m pip install "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
python -m ashare_model.train --device auto   # 或显式 --device cuda / cpu
```

GPU 只加速 VM 因子算子；**策略模型与采样始终在 CPU**（torch 的 CPU 随机流与
dropout 跨设备一致），因此同一 `(init_seed, seed)` 在任何机器上采样出相同的
公式序列（有测试固化该不变量）。仅 VM 的 float32 算术在 GPU 上与 CPU 有
~1e-7 的差异。训练产物记录 `init_seed` 与 `device` 字段供归档溯源。

## 运行入口

```bash
python -m ashare_data.sync
python -m ashare_model.diagnostics   # 因子质量报告（覆盖率/IC/相关性 → data/factor_report.json）
python -m ashare_model.train
python -m ashare_model.backtest
python -m ashare_model.evaluation --tier screening  # 测量协议（见下）
python -m ashare_trading.run_sim
python scripts/analyze_sim.py               # 模拟盘费用拖累/毛盈亏/现金核对
streamlit run dashboard/app.py
```

### 因子诊断与家族消融

新增因子族（或怀疑某族退化）时，先看证据再训练：

```bash
python -m ashare_model.diagnostics                        # 覆盖率 / rank-IC / 相关性
python scripts/ablate_families.py --steps 50 --batch-size 256   # 逐族消融（同 seed）
```

消融把每个家族轮流中性化（tensor 形状不变，token id 不变），对比同 seed 下
验证集奖励相对基线的变化（`data/ablation_results.json`）；奖励掉的多的族就是
"真正在出力"的族，掉得少的族可以安全精简。

### 测量协议（walk-forward 评价）

在改动奖励语义之前，先固定"怎么证明公式变好"的测量协议；协议裁决与 RL reward
完全解耦，所以不同 `reward_version` 代的产物仍然可比：

```bash
python -m ashare_model.evaluation --tier screening     # 快速筛选档（50 步 x 256）
python -m ashare_model.evaluation --tier confirmation  # 确认档（200 步 x 512）
python -m ashare_model.evaluation --selfcheck          # 空转验收：纯噪声候选，DS/max-t 必须不显著
# 确认档可并入此前筛选档的全部试错，计入多重检验校正：
python -m ashare_model.evaluation --tier confirmation \
    --trials experiments/20260816_protocol_screening/metrics.json
python scripts/archive_run.py --mode protocol --commit # 结果归档进 experiments/
```

- **折**：`protocol.folds` 按**绝对日期**锚定（默认 5 个日历年测试窗 2021–2025），
  数据持续增长不会移动折边界；每折训练至 `train_end`，在 `(train_end, test_end]`
  上做 OOS 打分。
- **种子**：`protocol.seeds` 每折多种子独立训练（默认 3 个），聚合报中位数 ± IQR
  （reward 有 clip，均值不可信）。
- **基线**：`protocol.baseline_signals` 按因子诊断 ICIR 选取（默认
  REVERSAL_5 / RSQ_60 / ILLIQ_20 / OVERNIGHT_RET / MOMENTUM_20 / ROE / TURNOVER），
  每个裸因子与训练公式走同一回测引擎路径，给出"什么水平算好"的标尺。
- **裁决指标**：完整回测引擎的净收益 / Sharpe / Sortino / 最大回撤 / 换手 +
  rank-IC / ICIR；**不用** `best_reward` / `fast_basket_reward` 裁决，训练侧
  `val_reward` 只归档、不参与排序。
- **多重检验**：Deflated Sharpe（Bailey & López de Prado 2014，含偏度/峰度修正）
  与中心化学生化 max-t 块自助法（White 现实检验风格）共享同一试验矩阵——每个
  非失败候选行是一个试验（excess 逐日收益）；DSR > 0.95 / max-t p ≤ 0.05 才
  判显著。此前跑批的试错用 `--trials` 并入校正。
- 产物 `data/protocol_result.json` 记录 `protocol_version` / `reward_version` /
  `frequency` / `horizon` 与逐折逐种子原始行（含逐日收益序列，便于下钻）；
  `frequency` / `horizon` 目前只是记录字段（周频 / 多周期目标留待后续阶段）。
- `batch_size` 不要低于 256：advantage 归一化（`rewards.std()`）在更小批次下有
  退化风险。

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

模拟盘每个 signal date 都按 PIT 资格掩码（signal 日与执行日同时 eligible）选股，
与回测引擎共用同一套 top-N 与涨跌停判定：未进入 universe 的股票不会产生新买单，
退出 universe 的持仓按普通卖出路径减仓，跌停无法卖出时按撮合规则继续持有；卖出单
先于买入单撮合，卖出资金当日即可用于买入。涨跌停价幅在历史回放中只按板块基准
（主板 10%、创业板/科创板 20%）判定，当前 `*ST` 名称只用于展示；执行日期等于今天
的当日模拟才额外使用 `stocks.is_st` 当前快照按 5% 判定，且仅限当日，不会回写历史。

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

生产模式（单服务，后端托管构建产物；API 可启停模拟盘，故仅绑定本机）：

```bash
cd webui && npm install && npm run build && cd ..
python -m uvicorn webapi.app:app --host 127.0.0.1 --port 8000
# 浏览器打开 http://127.0.0.1:8000
```

前端数据以只读展示为主；模拟盘页提供启动/继续、停止、重置按钮组、运行状态条
（3 秒轮询）与配置弹窗（`PUT /api/sim/config`），危险操作均二次确认。
配置编辑写入全局运行时覆盖文件
`config/runtime_overrides.yaml`（已被 gitignore，不污染 YAML 基线），
`run_sim` / `backtest` / `train` 所有入口都会在 YAML 之上合并它。费用
（佣金 / 最低佣金 / 印花税 / 过户费 / 滑点）是全项目单一口径（`backtest`
段），修改后对回测与模拟盘同时生效；初始资金在下次 reset 时生效。日志页
仅展示 `logs/` 与 `data/` 下的 `.log` / `.txt` 文件。

模拟盘的启停由后端进程管理器（`ashare_trading/manager.py`）托管，状态全部
落盘（`data/sim_run.json` + 锁文件），API 进程重启不会丢失或孤儿化子进程：

| 端点 | 说明 |
| --- | --- |
| `POST /api/sim/start` | 启动 `run_sim`（body：`reset` / `start` / `end`）；有状态时自动 `--resume`，运行中返回 409 |
| `POST /api/sim/stop` | 写 `STOP_SIGNAL` 后立即返回 `stopping`，宽限期后升级为终止进程树 |
| `POST /api/sim/reset` | 先 `archive_run.py --mode sim --commit` 归档旧状态（失败则中止），再重置并移走订单/成交目录 |
| `GET /api/sim/status` | 轮询状态机（idle/starting/running/stopping/stopped/finished/error）、PID、当前日期与净值 |

首次数据同步会读取沪深 300、中证 500 的**当前**成分快照来决定需要同步
哪些股票的日线，但快照不会写入 `constituents`、也不会伪造成历史有效期。PIT 历史成员
区间由 `scripts/import_pit_universe.py` 从 BaoStock 逐日成分查询按月末采样压缩成
`[in_date, out_date)` 半开区间（边界精度为月），上市日期由 `scripts/import_pit_universe.py`
经交易所批量股票资料（含退市表）回填；中证 1000（000852.SH）无免费历史成分来源，已从
配置移除，universe 为沪深 300 + 中证 500。正式训练、
协议、诊断、回测、模拟和归档统一执行生产数据门禁：`constituents` 必须提供
`index_code/ts_code/in_date/out_date` 的非重叠半开区间，成员股票必须有有效
`stocks.list_date`，日期轴必须来自 `trade_calendar.is_open = true`。退出后重新加入用
同一股票的多个区间表达，主键为 `(index_code, ts_code, in_date)`；跨指数同期重叠和
`[a,b)`/`[b,c)` 相邻区间合法。`scripts/check_production_gates.py` 显式核对全部门禁，
任何门禁失败即停止正式运行。
仅程序化构造 `AshareDataLoader(..., allow_development_universe_fallback=True)` 时允许
全期成员开发降级，并同时产生 warning 与可检查的状态；CLI、环境变量和测试检测均不能
开启该降级。

`AshareDataLoader` 是逐日资格掩码的唯一构建者：它在因子计算前把成员区间、完整开市
日历、上市日期和 signal-date bar 是否存在统一交给 `UniverseMask`，下游只消费与
`ts_codes/dates` 精确对齐的只读 `eligible/reasons` 数组。上市满 `N` 日按交易会话而非
自然日计算；同步会从交易所批量股票资料保留真实上市日期，并保存截至配置结束日的完整
历史开市日历。当前 `stocks.is_st` 只是当前快照，不会删除股票或被外推到历史资格；在
增加日期化 ST 数据前，每个历史单元带非阻断的 `STATUS_UNKNOWN` 审计位。缺 bar、未上市、
上市会话不足或当日非成员仍分别以可叠加原因码阻断 eligibility。涨跌停判定同样与名称
解耦：历史回放/回测一律使用板块基准价幅（主板 10%、创业板/科创板 20%），股票名称仅
用于展示，不会用当前 `*ST` 名称反推历史日期的 5% 限制；只有真实当日模拟（执行日期
等于今天）才把 `stocks.is_st` 快照按 5% 价幅用于当日撮合，且绝不用于历史日期。

日线同步后会继续同步**逐期
point-in-time 财报**（东财业绩报表按季度全市场拉取，披露季节末日对齐；新浪财务指标
补充 ROA/负债率；东财分红送配补充股息率，按除权除息日对齐），写入 `fundamental_pit`
表，供 11 个基本面因子（PE_TTM/PB/PS_TTM/ROE/ROA/毛利率/净利率/营收与利润增速/负债率/
股息率）在整个训练窗口使用。接着同步**两融与申万行业**：沪深交易所按交易日的全市场
融资余额截面（`margin_balance` 表，供 MARGIN_BALANCE_CHG 因子；首次全量回填约 2×交易日
次请求，之后仅刷新近 30 天）、申万一级行业指数与成分快照（`sw_industry_index` /
`sw_industry_member` 表，供 INDUSTRY_MOMENTUM 因子与行业中性因子 IND_REL_*，按成分
快照做行业去均值）。`--no-fundamentals` /
`--no-capital-flow` 可分别跳过。为避免重复请求 AkShare，可先使用 `--offline` 测试本地
流程，或使用 `--limit N` 限制股票数量。日线缓存落后于交易日历时会自动刷新；全量
（不带 `--limit`）同步会清理不在股票池中的历史行。

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

- **无未来泄漏**：因子只使用当前及历史截面；`open_to_open_returns` 以 t+1 开盘买入、t+2 开盘卖出为目标收益；缺 bar、未上市或非成员单元的 target 为 `NaN`，绝不产生虚假收益。基本面因子按**公告日期**（不是报告期）进入截面，公告前保持中性。
- **交易规则**：A 股 T+1、买入 100 股整手、卖出可零股清仓、涨停不买/跌停不卖、一字板判定。涨跌停价幅与名称解耦：历史回放与回测只按板块价幅（主板 10%、创业板/科创板 20%），股票名称仅用于展示；真实当日模拟（执行日期等于今天）才额外按 `stocks.is_st` 当前快照使用 5% 价幅，且仅限当日。
- **费用模型**：佣金万 2.5（最低 5 元）、印花税卖出 0.05%、过户费 0.001%、滑点 0.05%；回测与训练奖励使用同一套费用口径。
- **涨跌停事件因子**：`LIMIT_UP_EVENT`/`LIMIT_DOWN_EVENT` 由一字板真实计算（创业板/科创板 20%，其余 10%）。
- **训练**：REINFORCE + value baseline + 熵正则（advantage 裁剪防数值爆炸）；训练
  奖励为**截面 rank-ICIR 减去连续换手成本**（`reward.py` v5：成本按费率比例计入，
  无阈值跳变；篮子模拟按执行日（t+1 开盘）对齐回测引擎的**可交易性屏蔽**——
  停牌/一字涨停不买、停牌/一字跌停持仓强制保留），验证段按
  `model.validation_splits`（默认 3）个独立子窗口取**中位数**
  选择最佳公式；不含算子的裸因子公式减 `reward.complexity_penalty`（默认 0.2），
  最佳公式的验证奖励须达到 `reward.min_val_reward`（默认 0.0）才保存，避免把
  负质量公式回测/归档。v7 起，IC/ICIR、候选打分、RL 训练、随机搜索与裸因子
  baseline 的全部质量统计只使用 **signal-date eligible** 股票
  （`universe_mask[:, t]`；买入可交易性仍用 `blocked_buy[:, t+1]`、卖出
  `blocked_sell[:, t+1]`），near-constant 拒绝与方向打平时的 canonical
  orientation 也只扫描 eligible 观测——未来成分在加入前的有限极端值既不能改变
  排名 IC/ICIR/方向/拒绝原因，也不能进入奖励篮子；退出成员池后目标权重经正常
  卖出路径归零（执行日无法卖出则 force-hold，绝不凭空消失）。
- **公式输出终标准化**：VM 执行的每个公式信号按日做**截面 z-score** 后再进入
  评分/回测/模拟盘（单调变换，不改变 rank-IC 与 top-n 排序），消除叠加运算的
  尺度漂移，使 GATE/JUMP 的 0 阈值语义稳定，并为多公式合成提供统一尺度。
- **横截面算子与参数化窗口**（v6 词表）：公式栈内可直接表达截面语义——
  `CS_RANK`（平均秩百分位，ties 共享平均秩、CPU/GPU 一致）、`CS_ZSCORE`、
  `CS_DEMEAN`（减截面均值）、`CS_NEUTRALIZE`（按申万一级行业去均值，分组张量
  由数据加载器注入 VM；无行业数据时退化为全市场去均值）；时序窗口算子由
  硬编码 20 日扩展为 5/10/60 日枚举（MA/STD/TS_RANK/CORR/DOWNVOL）及
  DELTA10/20，采样语法不变。截面算子只消费当日截面，无未来泄漏。
- **PIT 成分约束贯穿因子与 VM 截面语义**：`vm.universe_mask`
  （`[stock × date]` 布尔，随窗口与计算设备移动）使 `CS_RANK`/`CS_ZSCORE`/
  `CS_DEMEAN`/`CS_NEUTRALIZE` 与 VM 终端 z-score 的参考集合只含**当日
  eligible 且 finite** 的单元——未来成分既不能被选中，也不能改变过去截面的
  均值/标准差/排名/行业均值；非 eligible 单元在最终信号中保持 `NaN`
  （不可参与），绝不作为 0 进入排序。基础因子层同样接入：winsorize
  分位点与 median/MAD、CAPM 等权市场收益、AMOUNT_SHARE 当日分母、行业相对
  因子的行业均值都只统计 eligible 单元，而原始行情与个股自身历史不被删除或
  置零（加入当天的动量仍使用加入前的自身行情）。全日无 eligible 股票时截面
  稳定映射为中性 0，仅一只 eligible 股票时 z-score/demean 为中性但该股票仍
  是唯一有效成员。
- **回测与研究报告共用同一 PIT universe**：回测引擎 `run()` 显式接收
  `[stock × date]` mask，选股要求 signal-date 与 entry-date
  （`mask[:, t] & mask[:, t+1]`）同时 eligible；默认等权基准只平均当日
  eligible 且目标收益有效的股票，无 eligible 股票的日期基准与空仓策略收益
  均为 0；退出成员池的持仓经正常卖出路径减仓（计费），持仓快照不会新增
  ineligible 股票。研究报告的 coverage 分母、rank IC（复用统一的
  `rank_ic_series` 实现，无第二套 Spearman）与每日相关矩阵只使用 eligible
  单元；协议 benchmark 行复用引擎的基准路径，IC 与回测消费同一个 mask。
  协议（`PROTOCOL_VERSION` v8）与回测产物记录实际生效的 universe policy
  字段（index_codes / min_listed_sessions / membership_end_inclusive /
  degraded），不记录数据 hash、不建立 lineage。**所有正式路径都强制传入
  mask**——回测引擎、奖励/IC、候选打分、VM/截面算子、因子预处理、诊断与
  模拟盘均无 `None`/无约束的兼容旁路；`tests/test_universe.py` 的集中式
  未来成员哨兵契约把一个全历史、极端行情的未来成员 F 同时放进（与移除出）
  全链路（因子预处理 → VM 截面算子 → 终端 z-score → IC/ICIR →
  CandidateScore → reward → 随机搜索最优候选 → baseline → diagnostics →
  benchmark → top-N → 回测 → 模拟盘），断言加入前所有结果与删除 F 时完全
  一致，且差异只允许从加入日起出现。
- **词表版本化**：训练产物记录 `feature_names`/`operator_names`/`feature_version`；加载公式时按**名称**重映射 token，词表新增特征不会错位旧公式；无元数据的旧公式对照首发词表（v1：34 特征/16 算子）重映射，退役重复特征经 `FEATURE_ALIASES`（`RET_20` → `MOMENTUM_20`，两者原为同一计算）解析，语义永不漂移。
- **回测输出**：包含持仓快照与全市场等权基准（与策略同一 open-to-open 口径），供看板展示。

## 已知局限（有意保留）

- **历史成分区间的月粒度近似**：`constituents` 的 300/500 历史区间由 BaoStock 月末快照压缩而来，边界精度为月（真实调样日前后最多一个月的偏差）；BaoStock 无中证 1000 历史成分，故 universe 为沪深 300 + 中证 500。申万行业成分映射仍为当前快照（行业指数行情本身是完整历史），只用于行业类因子的开发性近似。当前成分快照只用于选择同步标的，不写入 PIT `constituents`。
- **中性占位因子**：仅 `NORTHBOUND_CHG` 保持中性（0）——北向每日净流入及个股明细自 2024-08-19 起停止披露，免费渠道无可用历史。`MARGIN_BALANCE_CHG`（融资余额 20 日变化，收盘后披露、次日可用）与 `INDUSTRY_MOMENTUM`（申万一级行业指数 20 日收益映射到成分股）已有真实数据；行业中性因子 `IND_REL_RET_5/20`、`IND_REL_VOL_20`、`IND_REL_TURNOVER` 用同一当前快照成员映射去行业均值，无行业映射或单成员行业的股票保持中性。新特征按"代"追加在词表末尾（v1 的 token id 永不偏移），旧公式经按名重映射后继续有效。
- **换手率缺失即缺失**：换手率依赖流通股本，无法从 OHLCV 反推；缺失时保持中性而非伪造常数。
- **基本面 PIT 的近似口径**：`MARKET_CAP` 为流通市值近似（成交额/换手率，每日可得）；`PS_TTM = PE_TTM × 扣非净利TTM/营收TTM`（避免依赖总股本历史）；PE/PS 在 TTM 亏损时保持中性；ROE/ROA/毛利率/净利率/增速为累计（YTD）报告口径；**披露时点为法定披露季节末日**（Q1→4/30、中报→8/31、三季报→10/31、年报→次年 4/30，保守方向、绝不提前可见；免费接口无逐股首发公告日，东财业绩报表的"最新公告日期"实为重述日期，不可用）；股息率按精确除权除息日对齐；财报修正不追踪（以最新披露值为准）。
- **离线日历**：`--offline` 模式用工作日近似，包含节假日，仅用于开发与测试。
