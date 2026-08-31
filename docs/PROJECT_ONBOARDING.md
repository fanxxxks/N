# AlphaGPT 项目架构指南

> 分析日期：2026-08-27（Asia/Shanghai）  
> 分析基线：<code>main</code> @ <code>c5b801e936f6ef6cdba4c80ff1e81d12a7387ca6</code>（原始基线）
> 核验更新：2026-08-28（Asia/Shanghai）；核验基线 <code>main</code> @ <code>80918fa</code>（c5b801e 之后合入 P0/P1/P2 三个阶段，见 §0.3；本文已按当前仓库状态修订）  
> 分析方式：源码、配置、测试、历史实验、现有 DuckDB 与本地运行时产物的静态/只读核查。除创建本文档外，没有执行数据同步、训练、回测、模拟盘、归档、部署或任何项目业务数据/运行时写操作。  
> 文档定位：帮助刚接手项目的开发者在不误用旧产物、不破坏本地数据的前提下，快速建立正确的系统心智模型。

## 0. 先读结论

AlphaGPT 当前是一套本地优先、单用户、纯 A 股横截面因子研究与模拟盘系统。它不直接预测价格，而是：

1. 从 AkShare/BaoStock 拉取 A 股行情、指数历史成分、财务和资金数据；
2. 构造 62 个基础因子；
3. 在 39 个算子组成的后缀公式语言中搜索可解释公式；
4. 用统一的 PIT 股票池、交易可行性和费用模型评分；
5. 产出策略 JSON，随后用于回测、研究协议和纸面模拟交易；
6. 通过 Streamlit 或 React + FastAPI 看板展示本地产物。

最重要的接手判断如下。

| 结论 | 接手含义 | 证据 |
|---|---|---|
| 这是模块化单体/分层批处理系统，不是微服务，也不是在线预测服务 | 主流程由 CLI 驱动，模块之间大量通过 DuckDB、Parquet、JSON 和文件锁衔接 | [README.md](../README.md#L45)、[webapi/app.py](../webapi/app.py#L1)、[ashare_data/sync.py](../ashare_data/sync.py#L197) |
| 当前生产默认搜索器是强类型 GP，不是 Transformer/RL | RL 保留为实验选项，但准入实验失败；不要把项目简称为“RL 生产系统” | [config/ashare_config.yaml](../config/ashare_config.yaml#L42)、[docs/phase2_measurement_log.md](phase2_measurement_log.md#L130)、[ashare_model/train.py](../ashare_model/train.py#L1223) |
| 当前奖励实现是 v14：稀疏因果标签驱动 IC/质量门，逐日组合收益独立消费相邻 open 收益；主项仍是组合主动 IR 减精确年化费用 | P3 将标签、全局调仓日历和统一 PortfolioConstructor 接入训练/验证；仍以源码与 P3 契约为准 | [ashare_model/reward.py](../ashare_model/reward.py)、[docs/p3_portfolio_contract.md](p3_portfolio_contract.md)、[config/ashare_config.yaml](../config/ashare_config.yaml) |
| 本机数据库的 G1-G7 生产门禁当前全部通过 | PIT 成分、上市日、日历、最小股票数和历史成员 bar 覆盖目前可用；这不等价于策略或产物可用于决策 | [ashare_data/gates.py](../ashare_data/gates.py#L1)、[scripts/check_production_gates.py](../scripts/check_production_gates.py) |
| 本机现有策略、回测和协议产物彼此不一致且全部落后于当前源码代际 | 看板目前可能把不同公式、不同版本、不同数据时间的结果拼在一起；接手后不能直接引用这些数字 | [data/best_ashare_strategy.json](../data/best_ashare_strategy.json)、[data/backtest_result.json](../data/backtest_result.json)、[data/protocol_result.json](../data/protocol_result.json) |
| 当前没有可称为最终 holdout 的历史区间，也没有当前 v22/v14 的显著 alpha 证据 | 2021-2026 已被反复查看，只能算开发/验证数据；v20 selfcheck 曾显示 DSR=0.000、max-t p=1.0000（无显著候选） | [docs/phase4_measurement_log.md](phase4_measurement_log.md#L15)、[docs/phase5_measurement_log.md](phase5_measurement_log.md#L17) |
| P3 已把 QP optimizer 作为 `PortfolioConstructor` 的可选方法接入 reward、回测和模拟主链；生产默认仍是 equal_weight | 方法由 `backtest.portfolio_method` 显式选择；两种方法共享排名缓冲、阻断和后处理，不存在第二套生产 constructor | [ashare_portfolio/constructor.py](../ashare_portfolio/constructor.py)、[ashare_portfolio/optimizer.py](../ashare_portfolio/optimizer.py)、[docs/p3_portfolio_contract.md](p3_portfolio_contract.md) |
| 项目面向研究和 paper trading，不连接真实券商 | 所有成交由本地模拟撮合器生成，状态保存在 JSON；没有实盘下单适配器 | [ashare_trading/matching.py](../ashare_trading/matching.py)、[ashare_trading/run_sim.py](../ashare_trading/run_sim.py#L83) |

### 0.1 当前本机快照

以下是分析时的只读观测，不是仓库承诺的固定数据。

| 项目 | 当前值 |
|---|---|
| Git | <code>main</code> 与 <code>origin/main</code> 对齐（P0/P1/P2 共 28 个提交已推送）；工作区仅本文档有未提交修订 |
| Python | 项目说明要求 3.10+；CI 使用 3.12；本机可用项目环境为 <code>D:\minequant\.venv\Scripts\python.exe</code>，Python 3.13.12 |
| 默认 shell Python | <code>C:\ProgramData\miniconda3\python.exe</code>；它不是当前依赖完整的项目环境 |
| Node / npm | Node 24.14.0 / npm 11.9.0 |
| DuckDB | 786,444,288 bytes；日线 4,874,595 行，2015-01-05 至 2026-08-21 |
| 股票/成分 | 5,539 只股票元数据（P2-01 清除 7 个 900xxx B 股脏行）；2,574 条 PIT 成分区间 |
| 生产门禁 | G1-G7 全部通过；每年最少 eligible 股票数 473；2,426 个有效成员区间中零 bar 区间为 0 |
| 数据集清单 | <code>dataset_manifest</code> 1 行 + <code>dataset_manifest_cache</code> 129 行；dataset_id <code>b927074a455a…</code>（11,003,350 行 / 8 表，P0 构建，P2 purge 后沿用） |
| STOP 信号 | 根目录现有被忽略的 <code>STOP_SIGNAL</code>，内容为 <code>STOP</code> |
| 协议治理文件 | <code>holdout_registry.json</code>、<code>paper_windows.json</code>、<code>promotion_verdict.json</code> 不存在；<code>experiment_ledger.jsonl</code> 已存在（10 行 selfcheck 记录，见 P0/P1 日志） |

数据库表的当前行数：

| 表 | 行数 | 说明 |
|---|---:|---|
| <code>stocks</code> | 5,539 | 股票快照、上市日、当前 ST 标识（P2-01 已清除 7 个 B 股脏行） |
| <code>daily_bar</code> | 4,874,595 | 前复权日线 |
| <code>constituents</code> | 2,574 | 沪深 300/中证 500 PIT 成分半开区间 |
| <code>trade_calendar</code> | 8,797 | 交易日历 |
| <code>fundamental_pit</code> | 205,589 | 财务 PIT 近似数据（P2-01 范围治理后） |
| <code>margin_balance</code> | 5,642,388 | 融资余额 |
| <code>sw_industry_index</code> | 141,378 | 申万行业指数历史 |
| <code>sw_industry_member</code> | 5,196 | 当前申万行业成分映射 |
| <code>factor_cache</code> | 0 | 预留表；当前因子仍在加载时重新计算 |

### 0.2 当前运行时产物一致性审计

| 产物 | 当前内容 | 与当前源码的差异 |
|---|---|---|
| [data/best_ashare_strategy.json](../data/best_ashare_strategy.json) | 公式为 <code>(VOL_20 CORR60 (ATR_14 ADD MAX3((RET_10 ADD (PS_TTM MUL TS_RANK20(DIVIDEND_YIELD))))))</code>，方向 -1，reward v10 | 已由 <code>scripts/stamp_legacy_artifacts.py</code> 盖章 legacy（无 searcher / reward v10≠14 / 无 protocol/execution/constructor/config provenance / 无 model_version / 无 dataset_id，2026-08-27T10:31:32Z）；当前 reward v14 |
| [data/backtest_result.json](../data/backtest_result.json) | 公式为 <code>LIMIT_BREAK</code>，2015-01-06 至 2026-08-14，累计收益 -100%，Sharpe -1.956 | 公式与当前策略 JSON 不同；无 <code>dataset_id</code>；属于旧 schema |
| [data/protocol_result.json](../data/protocol_result.json) | protocol v12、reward v10、60 行候选 | 已盖章 legacy（protocol v12≠22 / reward v10≠14 / 无 execution/constructor/config provenance / 无 dataset_id / 无 stitched / 无 ledger）；当前 protocol v22、reward v14；没有 stitched OOS、dataset、ledger 或 data-regime 块 |
| [data/sim_portfolio_state.json](../data/sim_portfolio_state.json) | 2,822 个权益点、28,179 笔成交、最后日期 2026-08-14 | 旧状态没有 <code>last_exec_date</code>、公式、配置版本或 <code>dataset_id</code>；续跑时可能把新策略接到旧权益曲线上 |

结论：当前 UI 只能视为“历史文件查看器”，不能被当作一组同源、同版本、可复现的研究结果。任何策略判断前，应先核对数据 manifest（当前已存在，dataset_id <code>b927074a455a…</code>），再按当前版本重新训练、回测、执行 v22 协议，并通过六道晋级门禁。

### 0.3 自基线以来的变更（P0 / P1 / P2）

文档基线 c5b801e 之后，main 又合入三个阶段（核验基线 <code>80918fa</code>，2026-08-28 已推送 origin/main）：

| 阶段 | 主题 | 主要改动 |
|---|---|---|
| P0（2026-08-27） | GP 默认统一、doctor、legacy 盖章、CI web 构建、CPU/CUDA 依赖拆分 | 默认搜索器统一为 GP（deap 移入基础依赖）；新增只读 <code>research_doctor</code> 与 manifest CLI；旧策略/协议产物盖章 legacy 并在消费端防护（web API 补章、backtest/run_sim 警告）；CI 新增 React 构建 job；<code>requirements.txt</code> 改 pin CPU torch，GPU 机器用 [requirements-cuda.txt](../requirements-cuda.txt) 替换 |
| P1（2026-08-27） | 低成本测量与成本诊断 | [cost_matrix.py](../ashare_model/cost_matrix.py)（费用矩阵）、[bare_factor_backtest.py](../ashare_model/bare_factor_backtest.py)（七裸因子固定回测）、[searcher_bench.py](../ashare_model/searcher_bench.py)（四搜索器成本测量）、selfcheck 空转验收；P1 收尾 949 passed |
| P2（2026-08-28） | 免费数据可信度分层 | Tier A/B/C 定义与可用时间规则（契约见 [docs/p2_data_tier_contract.md](p2_data_tier_contract.md)）、feature registry v2、协议 v21 逐行记录 data_tier、promotion 第六道 data_tier 门（默认 A-only、B 单独对照、C 永不晋级）、分层诊断/消融报告、fundamental_pit/stocks 范围治理（purge 117,287 行）；P2 收尾 981 passed |

各阶段测量记录见 [docs/phase0_measurement_log.md](phase0_measurement_log.md) 至 [docs/phase6_measurement_log.md](phase6_measurement_log.md)。本文其余章节已按当前状态修订。

## 1. 项目概览

### 1.1 项目做什么

AlphaGPT 的核心目标是自动发现可解释的 A 股横截面选股公式。公式由基础特征和算子组成，表示为带独立 EOS 的后缀 token 序列；AST 是语义事实来源，StackVM 负责执行。搜索器不直接输出“明天涨跌概率”，而是在约束公式空间内生成或优化表达式，再用实际可交易、带排名缓冲与执行约束的组合表现评分。

主链路可概括为：

<code>数据同步 → PIT 股票池 → 因子张量 → 公式搜索/执行 → 候选评分与筛选 → 策略产物 → 回测/协议/模拟盘 → 看板</code>

核心定位来自 [README.md](../README.md#L1)、[CATREADME.md](../CATREADME.md#L18) 和 [ashare_model/train.py](../ashare_model/train.py#L846)。

### 1.2 面向用户

- 量化研究员：研究新因子、算子、奖励、搜索算法和评价协议。
- 量化开发者：维护数据同步、PIT 约束、回测/撮合一致性和实验追踪。
- 本地模拟盘操作者：通过 React/FastAPI 看板启动、停止、重置和查看 paper account。
- 不面向多租户 SaaS 用户，也没有用户、账户、权限、审计租户等领域模型。

### 1.3 解决的问题

- 把手工编写因子变为有类型、有语法约束、可执行、可版本化的公式搜索问题。
- 把 A 股上市日、历史成分、停牌、涨跌停、T+1、整手、费用等现实约束带入研究链路。
- 用 matched-budget 的 RL、随机、GP、TPE 和单因子基线比较搜索器，而不是只展示最优样本。
- 用 nested walk-forward、拼接 OOS、Deflated Sharpe、max-t、hash-chain ledger 和晋级门禁降低数据窥探风险。
- 在真实券商接入之前，以本地纸面撮合验证订单、费用、状态恢复和操作流程。

### 1.4 明确不做什么

- 不连接真实券商，不产生实盘订单。
- 不提供毫秒/分钟级交易；当前频率固定为日频。
- <code>protocol.frequency</code> 和 <code>horizon</code> 目前只记录到产物，不驱动实际调仓机制，见 [ashare_data/config.py](../ashare_data/config.py#L187)。
- 不提供服务端数据库、多用户认证或远程任务队列。
- 不保证已有策略有统计显著性，更不应把历史回测当作收益承诺。

### 1.5 项目历史边界

仓库从 Solana meme/加密因子系统迁移为纯 A 股系统；原链上主线和 <code>lord/</code> 实验已从主分支移除，可从 tag <code>archive/lord-and-crypto</code> 追溯，见 [CATREADME.md](../CATREADME.md#L3)。

[paper/20251226.pdf](../paper/20251226.pdf) 经逐页核对，内容是 “Defense in Predatory Markets: A Differential Game Framework for AMM Liquidity via Uniswap V4 Hooks”，讨论 AMM、JIT 流动性攻击、HJI 方程和动态费率。仓库源码、README 和配置均未引用它，它也不解释当前 A 股 AlphaGPT 的设计。应将其视为历史/误置参考资产，等待维护者确认，而不是系统设计文档。

[showcase.png](../showcase.png) 是一条“转债增强”界面截图，[assets/backtest.png](../assets/backtest.png) 和 [assets/backtest_2.png](../assets/backtest_2.png) 是无 manifest 的旧回测图片；三者同样未被代码或文档引用，不能作为当前结果证据。

## 2. 技术栈

### 2.1 后端与研究栈

| 类别 | 技术 | 用途 |
|---|---|---|
| 语言 | Python 3.10+ | 数据、模型、搜索、回测、模拟盘、API、旧看板和脚本 |
| 数值/表格 | NumPy、Pandas | 稠密因子矩阵、横截面/时序计算、数据整理 |
| 深度学习 | PyTorch | Looped Transformer、actor/critic、公式采样、GPU VM 张量执行 |
| 搜索 | DEAP、Optuna | 强类型 GP 和 TPE 基线/搜索器 |
| 优化 | CVXPY + OSQP | 独立组合 QP 优化器 |
| 数据源 | AkShare、BaoStock | A 股行情、股票、指数成分、财务、历史成员和兜底日线 |
| 本地存储 | DuckDB、Parquet/PyArrow、JSON | 规范化数据、逐股缓存、策略/回测/模拟状态和实验产物 |
| API | FastAPI、Uvicorn、Pydantic/Starlette | 本地 Web API、静态前端托管、模拟盘控制 |
| 旧 UI | Streamlit、Plotly | 单进程研究看板 |
| 日志 | Loguru | 控制台、内存队列、旋转文件和导出文本 |
| 配置 | PyYAML、python-dotenv | YAML 基线、运行时覆盖和少量环境变量 |
| 系统 | psutil | 模拟子进程存活检测、终止和锁恢复 |
| 测试 | pytest、SciPy | 单元/集成/统计检验 |

直接 Python 依赖以 [requirements.in](../requirements.in) 为人读清单、[requirements.txt](../requirements.txt) 为精确 pin；可选依赖位于 [requirements-optional.in](../requirements-optional.in) 和 [requirements-optional.txt](../requirements-optional.txt)。[requirements.lock](../requirements.lock) 是生成机器的完整环境快照，不是跨平台 lock。

### 2.2 前端栈

| 技术 | 用途 |
|---|---|
| React 18 + React DOM | SPA |
| React Router 6 | HashRouter 和六个页面 |
| Ant Design 5 + Icons | 布局、表单、表格、状态组件 |
| ECharts 5 + echarts-for-react | 净值、回撤、收益、换手图 |
| dayjs | 日期展示 |
| TypeScript 5.6 | 类型约束 |
| Vite 5 | 开发服务器、API 代理和生产构建 |

前端清单和构建脚本见 [webui/package.json](../webui/package.json)，开发代理见 [webui/vite.config.ts](../webui/vite.config.ts#L4)。

### 2.3 构建、打包和部署现状

- Python 代码没有 <code>pyproject.toml</code>、<code>setup.py</code> 或 <code>setup.cfg</code>，不是可安装包；通常必须从仓库根目录运行。
- 没有 Dockerfile、Compose、Kubernetes、Makefile、tox、正式迁移工具或部署清单。
- 唯一 CI 是 GitHub Actions：Python 3.12 job（安装两份 pin 文件、执行 pip check、lock check 和全量 pytest；freeze_lock 还校验 <code>requirements-cuda.txt</code> 与基础 torch 版本一致）加 Node 22 web job（npm ci、npm ls --depth=0、npm run build），见 [.github/workflows/ci.yml](../.github/workflows/ci.yml)。
- 前端没有测试和 lint；CI 已有构建步骤（P0-05 新增）。
- Web 的“生产模式”只是先构建 <code>webui/dist</code>，再由单个 Uvicorn/FastAPI 进程提供 API 和静态文件，见 [webapi/app.py](../webapi/app.py#L7)。

## 3. 整体架构

### 3.1 架构风格

项目采用以下组合风格：

- 分层架构：数据层、研究/模型层、组合与执行层、模拟交易层、接口展示层。
- 模块化单体：所有 Python 包在同一仓库、同一进程空间内直接 import；没有网络化内部服务。
- 批处理/管道架构：同步、训练、回测、协议和模拟都由 CLI 批次运行。
- 文件/产物驱动：模块间通过 DuckDB、Parquet、策略 JSON、结果 JSON、状态 JSON、逐日订单/成交 JSON 和日志衔接。
- 部分事件/控制信号：<code>STOP_SIGNAL</code> 和磁盘锁承担模拟盘进程的跨进程控制，但系统不是通用事件驱动架构。

### 3.2 系统上下文与组件图

~~~mermaid
flowchart LR
    Researcher[量化研究员/开发者]
    Operator[模拟盘操作者]
    AkShare[AkShare / 交易所与行情接口]
    BaoStock[BaoStock]

    subgraph AlphaGPT[AlphaGPT 本地模块化单体]
        Sync[ashare_data.sync]
        DB[(DuckDB)]
        PQ[(Parquet 缓存)]
        Gates[ProductionGateRunner G1-G7]
        Loader[AshareDataLoader]
        Factor[62 因子 + PIT 预处理]
        Search[GP / Random / RL / TPE]
        VM[AST + StackVM]
        Score[Reward v14 + CandidateSelector]
        Backtest[回测与 v22 评价协议]
        Optimizer[独立 QP 组合优化器]
        Sim[SimulationRunner + SimBroker]
        Artifacts[(JSON / PT / 日志 / 实验档案)]
        API[FastAPI]
        React[React 看板]
        Streamlit[Streamlit 看板]
    end

    AkShare --> Sync
    BaoStock --> Sync
    Sync --> DB
    Sync --> PQ
    DB --> Gates
    Gates --> Loader
    DB --> Loader
    PQ --> Sync
    Loader --> Factor
    Factor --> VM
    Search <--> VM
    VM --> Score
    Score --> Artifacts
    Artifacts --> Backtest
    Factor --> Backtest
    Backtest --> Artifacts
    Factor --> Sim
    Artifacts --> Sim
    Sim --> Artifacts
    Optimizer -. 当前未接入默认主链 .-> Backtest
    Artifacts --> API
    DB --> API
    API --> React
    Artifacts --> Streamlit
    DB --> Streamlit
    Researcher --> Sync
    Researcher --> Search
    Researcher --> Backtest
    Operator --> React
    Operator --> Streamlit
~~~

### 3.3 包级依赖关系

~~~mermaid
flowchart TD
    Config[config/ashare_config.yaml]
    Data[ashare_data]
    Model[ashare_model]
    Exec[ashare_execution.py]
    Portfolio[ashare_portfolio]
    Trading[ashare_trading]
    API[webapi]
    React[webui]
    Dashboard[dashboard]
    Scripts[scripts]
    Tests[tests]
    Logging[ashare_logging.py]

    Config --> Data
    Data --> Model
    Data --> Exec
    Data --> Trading
    Data --> Portfolio
    Model --> Trading
    Model --> Portfolio
    Exec --> Model
    Exec --> Trading
    Exec --> Portfolio
    Trading --> Portfolio
    Trading --> API
    Data --> API
    Exec --> API
    API --> React
    Data --> Dashboard
    Trading --> Dashboard
    Data --> Scripts
    Model --> Scripts
    Trading --> Scripts
    Logging --> Data
    Logging --> Model
    Logging --> Trading
    Tests -. 覆盖 .-> Data
    Tests -. 覆盖 .-> Model
    Tests -. 覆盖 .-> Portfolio
    Tests -. 覆盖 .-> Trading
    Tests -. 覆盖 .-> API
~~~

静态 import 扫描显示，最强依赖方向是 <code>ashare_model → ashare_data</code>，其次是 <code>ashare_trading → ashare_data/ashare_model</code>。反向依赖很少，层次总体清楚；但 [ashare_portfolio/golden.py](../ashare_portfolio/golden.py) 为一致性测试同时依赖回测、执行和交易层，是有意的集成验证边界。

### 3.4 从数据到策略的核心数据流

~~~mermaid
flowchart TD
    A[首次同步：日历、股票、当前成分快照、当前成员日线] --> B[导入 BaoStock 月末历史成分与上市日]
    B --> C[再次同步：当前成员 + PIT 历史成员 + 本地缓存并集]
    C --> D[回填零 bar 历史成员]
    D --> E[最终同步/manifest]
    E --> F{G1-G7 生产门禁}
    F -- 失败 --> X[正式入口拒绝]
    F -- 通过 --> G[AshareDataLoader]
    G --> H[PIT universe mask 与 reason codes]
    G --> I[OHLCV / 财务 / 融资 / 行业宽表]
    H --> J[62 x 股票 x 日期 因子张量]
    I --> J
    J --> K[AST/后缀公式 + StackVM]
    K --> L[候选信号按日横截面 z-score]
    L --> M[信号 t]
    M --> N{全局调仓日历到期?}
    N -- 是 --> O[开盘 t+1 调仓；开盘 t+1+horizon 形成稀疏标签]
    N -- 否 --> P[原样持仓；无订单/换手/成本]
    O --> Q[相邻 open 逐日资金曲线 + 主动 IR - 精确费用]
    P --> Q
    Q --> S[稀疏 IC/质量门 + 复杂度/容量门禁]
    S --> T[策略 JSON + execution provenance]
    T --> R[回测 / v22 协议 / 模拟盘]
~~~

同步入口见 [ashare_data/sync.py](../ashare_data/sync.py#L197)，PIT 导入见 [scripts/import_pit_universe.py](../scripts/import_pit_universe.py#L1)，统一门禁见 [ashare_data/gates.py](../ashare_data/gates.py#L1)，加载和目标构造见 [ashare_model/data_loader.py](../ashare_model/data_loader.py#L184)。

### 3.5 训练/搜索时序

~~~mermaid
sequenceDiagram
    actor Dev as 开发者
    participant CLI as ashare_model.train
    participant Gate as ProductionGateRunner
    participant Loader as AshareDataLoader
    participant Factor as FactorEngine
    participant Search as GP/Random/RL
    participant Cache as SemanticCache
    participant VM as StackVM
    participant Scorer as CandidateScorer
    participant Selector as CandidateSelector
    participant File as best_ashare_strategy.json

    Dev->>CLI: train --device auto
    CLI->>Gate: require_production()
    Gate-->>CLI: strict PIT status
    CLI->>Loader: load_data()
    Loader->>Factor: 计算 62 因子
    Factor-->>Loader: factor_tensor
    CLI->>Search: steps x batch_size 预算
    loop 每批候选
        Search->>Cache: canonical AST / 数值指纹去重
        Cache->>VM: 只执行未计费语义
        VM-->>Scorer: 标准化信号
        Scorer->>Scorer: IS 奖励 + 验证窗中位数 + 质量/复杂度/容量
        Scorer-->>Search: CandidateScore
    end
    Search->>Selector: eligible 候选 + Pareto/排序
    Selector-->>CLI: selected formula
    CLI->>File: 写版本、完整组合 provenance、词表、搜索器、dataset_id、历史
~~~

生产默认 <code>model.searcher: gp</code>，因此默认训练不会写 <code>ashare_model.pt</code>；只有 RL 搜索才保存 policy checkpoint，见 [ashare_model/train.py](../ashare_model/train.py#L825)。

### 3.6 模拟盘控制与执行时序

~~~mermaid
sequenceDiagram
    actor User as 操作者
    participant UI as React Sim 页面
    participant API as FastAPI
    participant Auth as mutation auth
    participant Gate as ProductionGateRunner
    participant Manager as SimJobManager
    participant Proc as run_sim 子进程
    participant Runner as SimulationRunner
    participant Broker as SimBroker
    participant Disk as 状态/订单/成交 JSON

    User->>UI: 启动/续跑/重置
    UI->>API: POST /api/sim/start
    API->>Auth: token 或 loopback 检查
    API->>Gate: require_production()
    API->>Manager: start(reset,start,end)
    Manager->>Manager: 文件锁、清 STOP、可选归档
    Manager->>Proc: subprocess.Popen
    Proc->>Runner: load formula + factors + resume watermark
    loop 每个执行日
        Runner->>Runner: signal t / open t+1
        Runner->>Broker: sells first, then whole-lot buys
        Broker-->>Runner: fills / skipped reasons / exact fees
        Runner->>Disk: 原子写 orders、trades、portfolio、progress
    end
    UI->>API: GET /api/sim/status
    API->>Manager: status()
    Manager-->>UI: state / pid / phase / date / equity
~~~

API 管理器启动时会清理旧 STOP；直接 CLI 只有带 <code>--resume</code> 或 <code>--reset</code> 才清理，见 [ashare_trading/run_sim.py](../ashare_trading/run_sim.py#L468) 和 [ashare_trading/manager.py](../ashare_trading/manager.py#L228)。

### 3.7 关键时间与形状契约

| 契约 | 说明 | 实现 |
|---|---|---|
| 因子张量 | <code>[feature, stock, date]</code>，当前为 62 个 feature | [ashare_model/factors.py](../ashare_model/factors.py#L828) |
| 信号/目标/universe | <code>[stock, date]</code> | [ashare_model/data_loader.py](../ashare_model/data_loader.py#L299) |
| 公式 | stack-only postfix；独立 EOS 终止，PAD 仅在 EOS 后；当前词表 103 token | [ashare_model/vocab.py](../ashare_model/vocab.py#L162)、[ashare_model/ir.py](../ashare_model/ir.py) |
| 成分区间 | <code>[in_date, out_date)</code> 半开区间，可多次进出指数 | [ashare_data/universe.py](../ashare_data/universe.py#L205) |
| 收益标签与资金曲线 | 调仓 signal 日 t；t+1 开盘进入；t+1+horizon 开盘退出，稀疏目标为 <code>open[t+1+horizon]/open[t+1]-1</code>；资金曲线始终消费相邻 open 日收益 | [ashare_model/targets.py](../ashare_model/targets.py)、[ashare_model/time_contract.py](../ashare_model/time_contract.py)、[ashare_portfolio/rebalance.py](../ashare_portfolio/rebalance.py) |
| 训练/验证 | 策略梯度只读 IS 头部；验证尾部切成 4 个子窗，以中位数选公式 | [ashare_model/train.py](../ashare_model/train.py#L880) |
| 评价 trial | v20 起（当前 v22）一个 trial 是一个 <code>(candidate, seed)</code> 跨折拼接 OOS 序列，不是一折一行 | [docs/phase4_measurement_log.md](phase4_measurement_log.md#L23) |
| PIT 选择 | 信号日和入场日必须 eligible；退出成员通过正常卖出路径处理 | [ashare_model/reward.py](../ashare_model/reward.py#L17)、[ashare_model/backtest.py](../ashare_model/backtest.py#L65) |
| no-signal | 可选截面少于两个不同值时保持原仓，不做信号驱动换手 | [ashare_model/reward.py](../ashare_model/reward.py#L36)、[ashare_trading/run_sim.py](../ashare_trading/run_sim.py#L283) |

## 4. 目录结构详解

### 4.1 总览

~~~text
AlphaGPT/
├─ ashare_data/          数据源、清洗、PIT 股票池、DuckDB、manifest、生产门禁
├─ ashare_model/         因子、公式语言、搜索、奖励、回测、评价与晋级治理
├─ ashare_portfolio/     统一组合构造、可选 QP 后端、调仓契约与黄金一致性规范
├─ ashare_trading/       订单、撮合、组合状态、模拟运行器与子进程管理
├─ webapi/               FastAPI 读取服务和模拟盘控制 API
├─ webui/                React/TypeScript 现代看板
├─ dashboard/            Streamlit 旧看板
├─ scripts/              数据修复、基线、消融、归档、依赖锁等运维脚本
├─ tests/                Python 测试与离线 fixtures
├─ config/               版本化 YAML 基线与环境变量示例
├─ experiments/          已提交的实验快照和准入结果
├─ docs/                 历史评估、阶段测量日志和本指南
├─ assets/               未被主线引用的旧回测图片
├─ paper/                未被主线引用、主题不匹配的 PDF
├─ .github/              GitHub Actions
├─ data/                 本地运行时数据与产物，gitignored
└─ logs/                 本地运行日志，gitignored
~~~

仓库跟踪文件分布：<code>ashare_data</code> 14 个、<code>ashare_model</code> 35 个、<code>ashare_portfolio</code> 3 个、<code>ashare_trading</code> 7 个、<code>webapi</code> 4 个、<code>webui</code> 18 个、<code>scripts</code> 12 个、<code>tests</code> 73 个（65 个测试模块 + conftest + 7 fixtures）、<code>experiments</code> 95 个。<code>experiments</code> 中的大 JSON 占绝大多数行数，做源码搜索或统计时应排除它。

### 4.2 根目录

| 级别 | 文件 | 职责 |
|---|---|---|
| 核心 | [ashare_execution.py](../ashare_execution.py) | 回测、训练奖励、组合黄金规范和模拟撮合共享的唯一费用模型；佣金最低额、印花税、过户费、滑点、可买股数 |
| 辅助 | [ashare_logging.py](../ashare_logging.py) | Loguru 控制台/文件/内存配置；10 MB rotation、14 份 retention、最多 10,000 行内存、文本导出 |
| 文档 | [README.md](../README.md) | 主运行说明；已更新 reward v14、P3 组合/因果标签、P2 分层与 CPU/CUDA 安装说明，历史测量以各 phase log 为准 |
| 文档 | [CATREADME.md](../CATREADME.md) | 仓库速读；已更新为 39 个算子并含 P2 分层说明（98405a7） |
| 依赖 | [requirements.in](../requirements.in)、[requirements.txt](../requirements.txt) | 直接依赖的人读清单和精确 pin |
| 依赖 | [requirements-optional.in](../requirements-optional.in)、[requirements-optional.txt](../requirements-optional.txt) | 测试/统计/GP/TPE 可选依赖 |
| 依赖 | [requirements.lock](../requirements.lock) | 当前开发机完整环境快照，含平台特定包 |
| 依赖 | [requirements-cuda.txt](../requirements-cuda.txt) | GPU 机器的 torch CUDA wheel 替换清单（P0-06）；与基础 pin 同版本，CI 校验一致性 |
| 配置 | [.gitignore](../.gitignore) | 忽略 data、logs、token、runtime overrides、node_modules、dist 等 |
| 法务 | [LICENSE](../LICENSE) | Apache License 2.0 |
| 遗留资产 | [showcase.png](../showcase.png) | 未引用的转债界面截图，与当前主线没有可验证关系 |

### 4.3 ashare_data：数据与 PIT 治理

| 级别 | 文件 | 职责 |
|---|---|---|
| 核心 | [config.py](../ashare_data/config.py) | Data/Model/Reward/Protocol/Backtest/Sim dataclass；YAML、runtime override、.env 合并和校验 |
| 核心 | [akshare_client.py](../ashare_data/akshare_client.py) | 外部数据客户端；重试、离线 fixtures、Eastmoney/Sina/Tencent/BaoStock 兜底和代码/日期标准化 |
| 核心 | [sync.py](../ashare_data/sync.py) | 数据同步 CLI；创建 schema、刷新日历/股票/行情/财务/资金、Parquet 缓存、purge 和 dataset manifest |
| 核心 | [db.py](../ashare_data/db.py) | DuckDB 封装、建表和各表 upsert；包含 constituents 主键兼容迁移 |
| 核心 | [universe.py](../ashare_data/universe.py) | 严格 PIT universe contract、半开成员区间、上市满 60 个交易日、bar presence 和 reason code |
| 核心 | [gates.py](../ashare_data/gates.py) | 正式入口统一 G1-G7 门禁；formal fail-closed、dev 带 degraded |
| 核心 | [manifest.py](../ashare_data/manifest.py) | 数据表分区指纹、Merkle/dataset_id、缓存、持久化和验证 |
| 核心 | [processor.py](../ashare_data/processor.py) | 行情清洗、宽表 pivot、前向收益、涨跌停/停牌/板块价幅等共享规则 |
| 核心 | [fundamentals.py](../ashare_data/fundamentals.py) | 财报同步、PIT 可见日期近似、估值/质量/成长/股息宽表 |
| 核心 | [capital_flow.py](../ashare_data/capital_flow.py) | 融资余额和申万行业同步、行业映射、资金类宽表 |
| 辅助 | [pit_import.py](../ashare_data/pit_import.py) | BaoStock 代码转换、月末采样、时间线压缩为成员区间、上市日合并的纯函数 |
| 辅助 | [io_utils.py](../ashare_data/io_utils.py) | 原子 JSON/Parquet 写、容错读取 |
| 数据结构 | [schemas.py](../ashare_data/schemas.py) | <code>BacktestResult</code>、<code>SimOrder</code>、<code>SimTrade</code> |
| 包入口 | [__init__.py](../ashare_data/__init__.py) | 包标识 |

数据库 schema 见 [ashare_data/db.py](../ashare_data/db.py#L48)。其中 <code>constituents</code> 的主键为 <code>(index_code, ts_code, in_date)</code>，允许同一股票多次进入同一指数；<code>out_date</code> 不包含在有效区间内。

Universe reason code 定义在 [ashare_data/universe.py](../ashare_data/universe.py#L47)：

- <code>NOT_MEMBER</code>：当日不是配置指数成员；
- <code>NOT_YET_LISTED</code>：尚未上市；
- <code>LISTING_AGE_INSUFFICIENT</code>：上市交易日不足；
- <code>STATUS_UNKNOWN</code>：元数据不足；
- <code>MISSING_BAR</code>：当日没有 bar。

### 4.4 ashare_model：研究与策略发现

| 级别 | 文件 | 职责 |
|---|---|---|
| 核心 | [feature_registry.py](../ashare_model/feature_registry.py) | 62 个基础特征的稳定顺序、别名、家族、必需列和 warm-up 元数据 |
| 核心 | [factors.py](../ashare_model/factors.py) | 基础因子计算、PIT 截面预处理、滚动 CAPM、技术/行业/事件因子 |
| 核心 | [ops.py](../ashare_model/ops.py) | 39 个公式算子的实现、元数/窗口配置；JUMP 使用因果 trailing baseline |
| 核心 | [vocab.py](../ashare_model/vocab.py) | feature/operator token 布局、EOS/PAD、feature_version、旧词表按名迁移 |
| 核心 | [ir.py](../ashare_model/ir.py) | 公式 AST、解析、规范化、canonical hash；公式语义事实来源 |
| 核心 | [vm.py](../ashare_model/vm.py) | StackVM；执行后缀 token、PIT 截面算子和最终按日 z-score |
| 核心 | [alphagpt.py](../ashare_model/alphagpt.py)、[imitation.py](../ashare_model/imitation.py) | Looped Transformer、actor/critic；v3 先做 baseline-elite teacher forcing，随后重建 optimizer 进入 RL |
| 核心 | [train.py](../ashare_model/train.py) | RL/GP/random 统一训练窗口、语义预算、候选选择、策略/模型产物写入 |
| 核心 | [reward.py](../ashare_model/reward.py) | reward v14；稀疏因果标签用于 IC/质量门，相邻 open 逐日收益用于资金曲线，统一 constructor 与精确费用 |
| 核心 | [candidates.py](../ashare_model/candidates.py) | CandidateSpec/Score、方向对称评分、质量/复杂度/容量门禁、选择 |
| 核心 | [complexity.py](../ashare_model/complexity.py) | AST 节点、深度、最长窗口和操作成本的复杂度账单 |
| 核心 | [signal_quality.py](../ashare_model/signal_quality.py) | HAC 有效样本 ICIR、block bootstrap、覆盖率/活跃度/符号稳定性 |
| 核心 | [semantic_cache.py](../ashare_model/semantic_cache.py) | 规范 AST + 校准切片数值指纹；按 dataset/reward/protocol/window 隔离预算 |
| 搜索 | [gp_search.py](../ashare_model/gp_search.py) | DEAP 强类型 GP |
| 搜索 | [search_contract.py](../ashare_model/search_contract.py)、[search_backends.py](../ashare_model/search_backends.py) | GP/TPE/Random/RL 统一 `SearchBackend`、预算与 `SearchResult` 契约 |
| 搜索 | [elite_archive.py](../ashare_model/elite_archive.py) | GP/TPE/Random eligible elite 的 v1 确定性归档、合并、读写与版本拒绝 |
| RL 诊断 | [rl_diagnostics.py](../ashare_model/rl_diagnostics.py) | reward/拒绝/entropy/duplicate/advantage/gradient/公式长度/算子覆盖的 v1 纯观测指标 |
| 搜索 | [tpe_search.py](../ashare_model/tpe_search.py) | Optuna TPE（正式 `model.searcher` 后端） |
| 搜索 | [baseline_harness.py](../ashare_model/baseline_harness.py) | matched unique-semantic-evaluation 预算和统一搜索评价适配器 |
| 搜索治理 | [admission.py](../ashare_model/admission.py) | v2 配对种子规则：imitation RL 必须在 area/OOS IR 同时胜 random RL 与 GP；失败时禁用高级 RL |
| 评价 | [backtest.py](../ashare_model/backtest.py) | 消费统一 PortfolioConstructor 的连续权重回测、基准、费用、持仓快照和指标 |
| 评价 | [evaluation.py](../ashare_model/evaluation.py) | v24 nested walk-forward + P4 四搜索器统一比较语义、P6 研究域维度、全局日历/稀疏标签/拼接 OOS/DSR/max-t |
| 评价 | [pareto.py](../ashare_model/pareto.py) | 多目标 Pareto frontier 辅助 |
| 治理 | [ledger.py](../ashare_model/ledger.py) | append-only JSONL 试验账本、序列和 SHA-256 hash chain |
| 治理 | [regime.py](../ashare_model/regime.py) | dev cutoff、预锁 final slice、dataset 绑定和违规拒绝 |
| 治理 | [promotion.py](../ashare_model/promotion.py) | Champion/Challenger 六门晋级（+data_tier）与成本/容量压力网格 |
| 时间 | [time_contract.py](../ashare_model/time_contract.py) | t/t+1/t+2 和 fold 内标签边界 |
| 诊断 | [diagnostics.py](../ashare_model/diagnostics.py) | 因子覆盖率、rank-IC、相关性报告 |
| 实验 | [experiment_tracking.py](../ashare_model/experiment_tracking.py) | 可选 MLflow；无 URI/无包时结构化 no-op |
| 版本 | [artifact_versions.py](../ashare_model/artifact_versions.py) | MODEL/REWARD/PROTOCOL/DATA_TIER/TIER_REPORT 等版本常量唯一来源（P0 新增） |
| 研究域 | [research_domain.py](../ashare_model/research_domain.py) | RESEARCH_DOMAIN_VERSION=1；按预测周期拆分研究域：特征全量划分（24/25/12）、每域目标周期/执行周期/Reward 参数/换手约束、域限制张量与采样 token 集（P6 新增） |
| 数据分层 | [data_tier.py](../ashare_model/data_tier.py) | DATA_TIER_VERSION=1；PitLevel→DataTier 映射、各档可用时间规则、<code>formula_data_tier_report</code> 公式追溯 API（P2 新增） |
| 分层报告 | [tier_reports.py](../ashare_model/tier_reports.py) | TIER_REPORT_VERSION=1；A/A+B/all 分层诊断与消融报告（P2 新增） |
| 测量 | [cost_matrix.py](../ashare_model/cost_matrix.py) | FEE_MATRIX_VERSION=1；资金×持仓数×换手率费用矩阵（P1 新增） |
| 测量 | [bare_factor_backtest.py](../ashare_model/bare_factor_backtest.py) | BARE_FACTOR_BACKTEST_VERSION=3；固定 daily/weekly × equal_weight/optimizer 四象限，逐象限记录 P3 provenance、收益、风险、换手、订单与成本 |
| 测量 | [searcher_bench.py](../ashare_model/searcher_bench.py) | SEARCHER_BENCH_VERSION=2；四后端请求/实耗预算、终止/停滞、best-so-far、时间与峰值内存 |
| 诊断 | [research_doctor.py](../ashare_model/research_doctor.py) | 只读研究医生：门禁、依赖与运行量估算，输出 data/research_doctor.json（P0 新增） |
| 兼容 | [ir.py](../ashare_model/ir.py)、[vocab.py](../ashare_model/vocab.py) | 旧 token/裸因子迁移和别名解析 |
| 包入口 | [__init__.py](../ashare_model/__init__.py) | 包标识 |

当前语义版本：

| 组件 | 版本 |
|---|---:|
| 模型 | <code>MODEL_VERSION = 3</code>（v3 记录 elite-imitation 初始化；v2 checkpoint 明确拒绝晋级并重训） |
| 奖励 | <code>REWARD_VERSION = 14</code>（v14 分离稀疏研究标签与逐日组合收益） |
| 评价协议 | <code>PROTOCOL_VERSION = 25</code>（v25 用语义类型约束采样候选池（P7-E）；v24 增加研究域维度并记录 research_domain；v23 统一四搜索器预算、终止与 best-so-far 结果语义） |
| 公式语法 | <code>GRAMMAR_VERSION = 3</code>（v3 使 action mask 类型感知（P7-E）；v2 引入独立 EOS 与 stack-only postfix 语法） |
| feature registry | 3（v3 起记录携带作者研究元数据与 horizon/cost/depends_on 派生三元组（P7 D1，仅描述性）；v2 起逐特征记录 data_tier） |
| data tier | 1（ashare_model/data_tier.py，P2 新增） |
| tier report | 1（ashare_model/tier_reports.py，P2 新增） |
| fee matrix / searcher bench | 1 / 2（P1 新增测量模块；searcher bench v2 起统一四后端 SearchResult 与唯一语义预算口径，契约见 docs/p4_search_transformer_contract.md §3） |
| bare factor backtest | 3（v2 记录完整执行 provenance；v3 固定四象限 schema） |
| portfolio constructor | 1（ashare_portfolio/constructor.py） |
| rebalance policy | 2（v2 新增 every_20_days 与 monthly 频率，P6） |
| research domain | 1（ashare_model/research_domain.py，P6 新增） |
| execution spec | 2（ashare_portfolio/execution_spec.py） |
| semantic cache | 1 |
| dataset manifest | 1 |

### 4.5 ashare_portfolio：统一组合构造与黄金规范

| 级别 | 文件 | 职责 |
|---|---|---|
| 核心 | [constructor.py](../ashare_portfolio/constructor.py) | 信号到目标权重的唯一生产实现；排名缓冲、equal_weight/optimizer、阻断、阈值、最小金额与换手预算 |
| 核心 | [optimizer.py](../ashare_portfolio/optimizer.py) | constructor 的可选 CVXPY/OSQP 长仓 QP 后端；alpha、风险、换手、冲击、行业/beta/size 暴露、ADV 容量约束 |
| 契约 | [rebalance.py](../ashare_portfolio/rebalance.py) | 全局 daily/weekly/every-N/monthly 调仓日历及 frequency/horizon 非重叠约束（v2 起含 every_20_days 与 monthly，P6） |
| provenance | [execution_spec.py](../ashare_portfolio/execution_spec.py) | execution v2、constructor 版本和完整组合配置的统一记录/校验 |
| 集成测试 | [golden.py](../ashare_portfolio/golden.py) | 将当前 constructor 权重通过 lot-free/whole-lot 撮合重放，分解费用、阻塞、手数残差；外部权重必须携带匹配 provenance |
| 包入口 | [__init__.py](../ashare_portfolio/__init__.py) | 公开 constructor/optimizer/golden/provenance 类型 |

reward、完整回测、golden parity 和 simulation 都消费同一个 <code>PortfolioConstructor</code>。生产默认由 YAML 选择 <code>equal_weight</code>；将 <code>portfolio_method</code> 显式改为 <code>optimizer</code> 时，同一 constructor 调用 QP 后端，并继续执行相同的缓冲、阻断和后处理契约。

### 4.6 ashare_trading：纸面执行

| 级别 | 文件 | 职责 |
|---|---|---|
| 核心 | [run_sim.py](../ashare_trading/run_sim.py) | 日频回放、公式信号、resume watermark、逐日原子落盘、进度 |
| 核心 | [matching.py](../ashare_trading/matching.py) | SimBroker；停牌、涨跌停、T+1、现金、整手、精确费用和成交原因 |
| 核心 | [orders.py](../ashare_trading/orders.py) | 权重转股数、卖单优先、全退出、确定性订单 ID |
| 核心 | [portfolio.py](../ashare_trading/portfolio.py) | 现金、持仓、可卖数量、成本、权益历史、原子 JSON 状态 |
| 控制 | [manager.py](../ashare_trading/manager.py) | API 子进程启动/状态/停止/升级终止/重置/归档、磁盘锁 |
| 控制 | [signals.py](../ashare_trading/signals.py) | STOP/STOPPED 文件协议 |
| 包入口 | [__init__.py](../ashare_trading/__init__.py) | 包标识 |

回测是连续权重模型；模拟盘是真实股数、买入 100 股整手、可零股清仓。两者通过共享费用模型和 [ashare_portfolio/golden.py](../ashare_portfolio/golden.py) 检查可解释残差，而不是承诺逐分完全相同。

### 4.7 Web 与看板

#### webapi

| 文件 | 职责 |
|---|---|
| [app.py](../webapi/app.py) | FastAPI 路由、CORS、本地静态 dist 挂载 |
| [service.py](../webapi/service.py) | JSON/DB 读取、日志 tail、runtime override 写入、SimJobManager 适配 |
| [auth.py](../webapi/auth.py) | 变更接口的 token/loopback 保护 |
| [__init__.py](../webapi/__init__.py) | 包标识 |

#### webui

| 路径 | 职责 |
|---|---|
| [src/App.tsx](../webui/src/App.tsx) | HashRouter、菜单和 API 健康检查 |
| [src/api/client.ts](../webui/src/api/client.ts) | fetch 封装和 React 数据轮询 hook |
| [src/types.ts](../webui/src/types.ts) | API 响应 TypeScript 接口 |
| [src/pages/Overview.tsx](../webui/src/pages/Overview.tsx) | 绩效、策略、模拟摘要、训练曲线 |
| [src/pages/Backtest.tsx](../webui/src/pages/Backtest.tsx) | 净值、回撤、收益、换手、指标和持仓分页 |
| [src/pages/Selection.tsx](../webui/src/pages/Selection.tsx) | 最新选股快照和公式 |
| [src/pages/Sim.tsx](../webui/src/pages/Sim.tsx) | 启停/重置、状态、配置、持仓、订单/成交 |
| [src/pages/DataStatus.tsx](../webui/src/pages/DataStatus.tsx) | DB 行数、日期、产物和配置摘要 |
| [src/pages/Logs.tsx](../webui/src/pages/Logs.tsx) | 日志列表、搜索、tail 和自动刷新 |
| [src/components/charts.tsx](../webui/src/components/charts.tsx) | ECharts 公共图表 |
| [src/components/common.tsx](../webui/src/components/common.tsx) | Panel、MetricCard 等公共 UI |
| [src/styles.css](../webui/src/styles.css) | 全局样式 |
| [package.json](../webui/package.json)、[vite.config.ts](../webui/vite.config.ts) | 依赖、构建、代理和 chunk 拆分 |

#### dashboard

| 文件 | 职责 |
|---|---|
| [app.py](../dashboard/app.py) | 旧 Streamlit 五 tab 看板和紧急停止按钮 |
| [data_service.py](../dashboard/data_service.py) | 防御性读取 backtest/sim JSON 和 DuckDB 状态 |
| [visualizer.py](../dashboard/visualizer.py) | Plotly 净值/基准图 |
| [.gitkeep](../dashboard/.gitkeep) | 目录占位 |

React/FastAPI 是功能更完整的现代 UI；Streamlit 适合作为简单、低依赖的本地只读入口，但两套 UI 会增加 schema 漂移维护成本。

### 4.8 scripts：运维与研究脚本

| 文件 | 职责与副作用 |
|---|---|
| [import_pit_universe.py](../scripts/import_pit_universe.py) | 从 BaoStock 月末快照重建 300/500 历史成员区间并补上市日；写 DuckDB |
| [backfill_member_bars.py](../scripts/backfill_member_bars.py) | 审计并回填零 bar 历史成员；写 DuckDB/Parquet |
| [check_production_gates.py](../scripts/check_production_gates.py) | 运行 G1-G7 并输出 JSON；检查本身只读 |
| [baseline_harness.py](../scripts/baseline_harness.py) | 在统一预算下跑裸因子/随机基线 |
| [ablate_families.py](../scripts/ablate_families.py) | 同 seed 因子家族消融 |
| [admission_experiment.py](../scripts/admission_experiment.py) | 五个独立 pair 同 seed/同请求预算比较 GP/TPE/Random、random RL、imitation RL；失败行不丢弃 |
| [analyze_sim.py](../scripts/analyze_sim.py) | 汇总模拟盘日文件和交易表现 |
| [archive_run.py](../scripts/archive_run.py) | 归档公式、配置、指标、模型 hash 和 commit；带 <code>--commit</code> 会创建 Git commit |
| [freeze_lock.py](../scripts/freeze_lock.py) | 从当前解释器已安装包生成 pin/完整 lock；无参数会改写依赖文件，<code>--check</code> 才是只读核对 |
| [check_fundamental_scope.py](../scripts/check_fundamental_scope.py) | P2-01 基本面表范围审计与清理；<code>--report</code> 只读，<code>--purge</code> 删除范围外行并把前后计数写入 data/fundamental_scope.json |
| [stamp_legacy_artifacts.py](../scripts/stamp_legacy_artifacts.py) | P0-04 旧策略/协议产物盖章 legacy（幂等，可重复执行） |
| [tier_reports.py](../scripts/tier_reports.py) | P2-05 分层诊断与消融报告；写 data/tier_report.json 与 tier_report_diagnostics.json |

### 4.9 tests：测试结构

73 个跟踪文件中包含 65 个测试模块、1 个 <code>conftest.py</code> 和 7 个 JSON fixtures。命名基本与生产模块一一对应：

- 数据：<code>test_akshare_client</code>、<code>test_sync</code>、<code>test_db</code>、<code>test_universe</code>、<code>test_gates</code>、<code>test_manifest</code>、<code>test_fundamentals</code>、<code>test_capital_flow</code>。
- 公式/模型：<code>test_factors</code>、<code>test_ops</code>、<code>test_vm</code>、<code>test_vocab</code>、<code>test_ir</code>、<code>test_grammar</code>、<code>test_train</code>、<code>test_candidates</code>。
- 研究有效性：<code>test_evaluation</code>、<code>test_stitched_oos</code>、<code>test_ledger</code>、<code>test_regime</code>、<code>test_promotion</code>、<code>test_semantic_cache</code>、<code>test_admission</code>、<code>test_data_tier</code>、<code>test_tier_reports</code>、<code>test_fundamental_scope</code>、<code>test_artifact_versions</code>。
- 执行：<code>test_backtest</code>、<code>test_execution</code>、<code>test_trading</code>、<code>test_run_sim</code>、<code>test_jobmanager</code>、<code>test_golden_parity</code>。
- 测量（P1）：<code>test_cost_matrix</code>、<code>test_bare_factor_backtest</code>、<code>test_searcher_bench</code>、<code>test_research_doctor</code>。
- UI/API：<code>test_dashboard</code>、<code>test_webapi</code>。
- 完成性：<code>test_completion_gates</code> 聚合检查版本、文档/配置和关键契约。

[tests/conftest.py](../tests/conftest.py) 提供公共 fixtures/日志；[tests/fixtures](../tests/fixtures) 提供离线股票、日历、成分和两只股票的日线。

### 4.10 配置、文档、实验与遗留资产

| 目录 | 职责/现状 |
|---|---|
| [config](../config) | [ashare_config.yaml](../config/ashare_config.yaml) 是版本化基线；[.env.example](../config/.env.example) 只列三个数据路径变量；真实 <code>.env</code>、<code>.webapi_token</code> 和 <code>runtime_overrides.yaml</code> 被忽略 |
| [docs](.) | 本指南、P2 契约（[p2_data_tier_contract.md](p2_data_tier_contract.md)）与 Phase 0-6 七份测量日志；旧评估报告 evaluation_20260823.md 已删除（74f833e） |
| [experiments](../experiments) | 只增不改的研究快照；T2 固定-baseline-seed admission 仅为历史证据，P4 晋级必须使用配对独立种子 |
| [assets](../assets) | 两张无 provenance 的旧回测图片 |
| [paper](../paper) | 一篇与 A 股主线无关的 Uniswap V4 论文 |
| [.github](../.github) | 单一 Python CI workflow |
| <code>data</code> | gitignored，约含 DuckDB、Parquet、策略、模型、回测、协议、模拟状态、逐日订单/成交，以及 P0/P1/P2 产物（fee_matrix、bare_factor_backtest、searcher_bench、selfcheck、research_doctor、tier_report、fundamental_scope 等）和 purge 前备份 ashare.duckdb.p2bak |
| <code>logs</code> | gitignored，当前约 1,800 个历史文件；API 可读取其尾部 |

## 5. 功能清单与代码映射

| 功能 | 主要模块 | 输入 | 输出 |
|---|---|---|---|
| 交易日历/股票/日线同步 | [ashare_data/sync.py](../ashare_data/sync.py)、[akshare_client.py](../ashare_data/akshare_client.py) | 外部接口、YAML | DuckDB + per-code Parquet + manifest |
| PIT 历史成员和上市日 | [scripts/import_pit_universe.py](../scripts/import_pit_universe.py)、[pit_import.py](../ashare_data/pit_import.py) | BaoStock 月末快照/股票基本表 | <code>constituents</code>/<code>stocks</code> |
| 历史成员 bar 修复 | [scripts/backfill_member_bars.py](../scripts/backfill_member_bars.py)、[sync.py](../ashare_data/sync.py#L383) | G7 零 bar 审计 | DuckDB + Parquet |
| 正式生产门禁 | [gates.py](../ashare_data/gates.py) | DB + DataConfig | <code>GateResult</code>/<code>UniverseContractStatus</code> |
| PIT 股票池 | [universe.py](../ashare_data/universe.py)、[data_loader.py](../ashare_model/data_loader.py) | 成分区间、上市日、日历、bar presence | mask + reason codes |
| 基本面/资金数据 | [fundamentals.py](../ashare_data/fundamentals.py)、[capital_flow.py](../ashare_data/capital_flow.py) | AkShare/DB | PIT 宽表 |
| 62 基础因子 | [feature_registry.py](../ashare_model/feature_registry.py)、[factors.py](../ashare_model/factors.py) | 行情/PIT 宽表/universe | 因子张量 |
| 公式语言和执行 | [ir.py](../ashare_model/ir.py)、[vocab.py](../ashare_model/vocab.py)、[ops.py](../ashare_model/ops.py)、[vm.py](../ashare_model/vm.py) | token + factor tensor | 标准化信号 |
| GP 默认搜索 | [gp_search.py](../ashare_model/gp_search.py)、[train.py](../ashare_model/train.py#L979) | 统一语义预算 | 最优 CandidateScore |
| Random/RL/TPE 实验搜索 | [train.py](../ashare_model/train.py)、[tpe_search.py](../ashare_model/tpe_search.py)、[evaluation.py](../ashare_model/evaluation.py) | 同一搜索空间/预算 | 候选与协议行 |
| 质量/复杂度/容量门禁 | [candidates.py](../ashare_model/candidates.py)、[signal_quality.py](../ashare_model/signal_quality.py)、[complexity.py](../ashare_model/complexity.py) | 信号、目标、ADV、AST | eligible/reasons/objectives |
| 训练奖励 | [reward.py](../ashare_model/reward.py) + [ashare_execution.py](../ashare_execution.py) | basket、基准、交易阻塞、费用 | reward/ICIR/objectives |
| 因子诊断/消融 | [diagnostics.py](../ashare_model/diagnostics.py)、[scripts/ablate_families.py](../scripts/ablate_families.py) | 因子张量 | factor_report/ablation JSON |
| 回测 | [backtest.py](../ashare_model/backtest.py) | 策略信号、行情、mask | 净值、指标、持仓、基准 |
| Nested walk-forward 评价 | [evaluation.py](../ashare_model/evaluation.py) | 候选、fold、seed、全局 rebalance mask、因果 target | protocol v22 JSON（data_tier + execution/constructor/config provenance） |
| 试验账本/数据区间治理 | [ledger.py](../ashare_model/ledger.py)、[regime.py](../ashare_model/regime.py) | trial/fold/dataset | hash-chain ledger、registry |
| 策略晋级 | [promotion.py](../ashare_model/promotion.py) | v22/v14/v2 当前协议、paper windows、当前数据与组合 provenance | 六门 verdict（+data_tier，默认 A-only） |
| 统一组合构造 | [constructor.py](../ashare_portfolio/constructor.py) | signal、前仓、PIT/阻断 mask、资本、配置 | PortfolioOutput；equal_weight/optimizer 共用后处理 |
| QP 组合优化 | [optimizer.py](../ashare_portfolio/optimizer.py) | alpha、前仓、风险/暴露/ADV | PortfolioSolution；由 constructor 的 optimizer 方法消费 |
| 回测/撮合黄金一致性 | [golden.py](../ashare_portfolio/golden.py) | 回测目标权重和原始 bar | ParityReport |
| 模拟订单与撮合 | [orders.py](../ashare_trading/orders.py)、[matching.py](../ashare_trading/matching.py) | 目标权重、现金、bar | SimOrder/SimTrade |
| 模拟状态/续跑 | [portfolio.py](../ashare_trading/portfolio.py)、[run_sim.py](../ashare_trading/run_sim.py) | 策略、旧状态、日期区间 | portfolio/progress/逐日流水 |
| 模拟子进程控制 | [manager.py](../ashare_trading/manager.py)、[signals.py](../ashare_trading/signals.py) | API 操作 | run record、锁、STOP、子进程 |
| Web/API 看板 | [webapi](../webapi)、[webui](../webui) | DB/JSON/logs | 六页 SPA |
| 简版看板 | [dashboard](../dashboard) | DB/JSON | Streamlit 五 tab |
| 归档/复现 | [scripts/archive_run.py](../scripts/archive_run.py)、[experiments](../experiments) | 运行产物 | manifest/config/metrics/formula/model hash |
| 数据等级追溯 | [data_tier.py](../ashare_model/data_tier.py) | 公式 token / 裸因子名 | {max_tier, tiers_used, per_feature}（P2） |
| 分层诊断/消融 | [tier_reports.py](../ashare_model/tier_reports.py)、[scripts/tier_reports.py](../scripts/tier_reports.py) | 因子张量、训练预算 | data/tier_report*.json（P2） |
| 费用矩阵 | [cost_matrix.py](../ashare_model/cost_matrix.py) | 资金×持仓×换手网格 | data/fee_matrix.json（P1） |
| 裸因子固定回测 | [bare_factor_backtest.py](../ashare_model/bare_factor_backtest.py) | 七裸因子 | data/bare_factor_backtest.json（P1） |
| 搜索器成本测量 | [searcher_bench.py](../ashare_model/searcher_bench.py) | 四搜索器、统一语义预算 | data/searcher_bench.json（P1） |

### 5.1 62 个基础因子

当前稳定顺序和 feature version 来自 [ashare_model/feature_registry.py](../ashare_model/feature_registry.py) 与 [ashare_model/vocab.py](../ashare_model/vocab.py#L213)。当前 <code>feature_version = 29ac4001dd3c</code>。

逐特征元数据（家族、Tier、PIT 级别、语义类型、推荐周期、预期方向、计算成本、依赖、可用时间规则、经济假设）**由注册表生成**，见 [docs/feature_registry.md](feature_registry.md)——P7 D3 起手工名单已退役，禁止在文档中重建第二份名单（漂移守卫测试会拒绝）。

特殊语义：

- <code>NORTHBOUND_CHG</code> 是中性 0 占位，因为北向日度明细自 2024-08 起停止披露，见 [README.md](../README.md#L335)。
- 申万行业指数是历史序列，但股票到行业的映射是当前快照投射到历史。
- 财报可见日使用法定披露季末近似，不是逐股首发公告日；重述只保留当前最新值。
- <code>PS_TTM</code> 为近似口径，<code>MARKET_CAP</code> 为成交额/换手率得到的流通市值近似。

### 5.2 公式算子

全部算子（39 个）的 arity、类别、输入/输出语义类型、计算成本与数值稳定性备注同样**由注册表生成**，见 [docs/feature_registry.md](feature_registry.md) §2。

除零和非有限值有保护；窗口算子只使用当前及过去；JUMP 在 v8 修复为 trailing 60 日基线，见 [ashare_model/reward.py](../ashare_model/reward.py#L63) 和 [ashare_model/ops.py](../ashare_model/ops.py#L117)。

## 6. 接口文档

### 6.1 鉴权模型

FastAPI 的读取接口全部无鉴权。只有四类变更接口使用 [webapi/auth.py](../webapi/auth.py#L38)：

1. 如果 <code>config/.webapi_token</code> 存在且内容非空，请求必须携带完全相同的 <code>X-API-Token</code>。
2. 如果 token 文件不存在，仅 request.client 为 <code>127.0.0.1</code>、<code>::1</code> 或 <code>localhost</code> 时允许变更。
3. 如果 token 文件存在但为空，所有变更请求都会返回 401。
4. CORS 只允许本机 5173/8000 四个 origin；CORS 不是访问控制。

FastAPI 默认还提供 <code>/docs</code>、<code>/redoc</code> 和 <code>/openapi.json</code>。

### 6.2 对外 HTTP API

路由定义见 [webapi/app.py](../webapi/app.py#L61)，响应字段的前端契约见 [webui/src/types.ts](../webui/src/types.ts)。

| 方法与路由 | 入参 | 成功响应 | 鉴权 | 主要错误 |
|---|---|---|---|---|
| GET <code>/api/health</code> | 无 | <code>{status:"ok", time:ISO}</code> | 无 | 500 |
| GET <code>/api/overview</code> | 无 | <code>{backtest,strategy,sim,status}</code> | 无 | 200 + 空子对象；服务层防御性降级 |
| GET <code>/api/backtest</code> | 无 | BacktestData：公式、metrics、dates、equity、benchmark 等 | 无 | 缺文件返回空对象 |
| GET <code>/api/backtest/positions</code> | query <code>offset≥0</code>；<code>1≤limit≤200</code> | <code>{items,total}</code> | 无 | 422 参数校验 |
| GET <code>/api/strategy</code> | 无 | StrategyData：公式、方向、奖励、ICIR、history 等原始策略字段 | 无 | 缺文件返回空对象 |
| GET <code>/api/sim</code> | 无 | SimState：现金、总权益、持仓、权益历史、成交数 | 无 | 缺/坏状态返回空对象 |
| GET <code>/api/sim/days</code> | 无 | <code>{total, dates}</code>；dates 最多返回前 200 个 | 无 | 200 |
| GET <code>/api/sim/day/{date}</code> | 8 位 <code>YYYYMMDD</code> | <code>{date,orders,trades}</code> | 无 | 400 日期格式 |
| GET <code>/api/sim/status</code> | 无 | state、pid、时间、phase、current_date、equity、log_path | 无 | 服务层错误对象 |
| POST <code>/api/sim/start</code> | JSON <code>{reset?:bool,start?:date,end?:date}</code>；日期可带连字符 | SimStartResult：action/message/archive/args + status | 变更鉴权 | 400 门禁/配置；401/403；409 已运行；422 body 校验 |
| POST <code>/api/sim/stop</code> | 无 body | <code>{ok,state,pid?}</code> | 变更鉴权 | 401/403；500 I/O |
| POST <code>/api/sim/reset</code> | 无 body | <code>{ok,action,message,archive}</code> | 变更鉴权 | 401/403；409 运行中；500 归档/IO |
| GET <code>/api/sim/config</code> | 无 | effective、overrides、pending_reset、execution_config_consistent | 无 | 防御性空配置 |
| PUT <code>/api/sim/config</code> | SimConfigPatch，字段可为数值或 null | 更新后的 SimConfigData + <code>ok</code> | 变更鉴权 | 400/422；401/403 |
| GET <code>/api/data-status</code> | 无 | DB 状态、产物 stat、配置摘要 | 无 | 200 + ready=false |
| GET <code>/api/logs</code> | 无 | LogFile 数组 | 无 | 200 |
| GET <code>/api/logs/{name}</code> | query <code>1≤tail≤20000</code> | 文件尾部、size、lines、truncated/error | 无 | 422；未找到以 200 error 对象返回 |

日志读取只接受 basename，在 <code>logs</code> 和 <code>data</code> 下寻找 <code>.log/.txt</code>，单次最多读尾部 16 MiB，见 [webapi/service.py](../webapi/service.py#L322)。

<code>SimConfigPatch</code> 的限制见 [webapi/service.py](../webapi/service.py#L373)：

- <code>initial_capital &gt; 0</code>；
- <code>1 ≤ max_positions ≤ 500</code>；
- <code>0 &lt; single_weight_cap ≤ 1</code>；
- 费率 <code>0 ≤ rate &lt; 1</code>；
- <code>min_commission ≥ 0</code>；
- null 表示从 runtime override 删除该键，退回 YAML 基线。

配置写接口同时更新 <code>sim</code> 和 <code>backtest</code> 节；这意味着 Web “模拟盘配置”也会影响训练奖励和回测，见 [webapi/service.py](../webapi/service.py#L420)。

API 没有数据同步、训练、回测、协议评价或晋级路由，这些任务仍须从 CLI 执行。

### 6.3 主要响应对象

| 对象 | 关键字段 |
|---|---|
| BacktestMetrics | total_return、annual_return、annual_volatility、sharpe、sortino、max_drawdown、calmar、average_turnover? |
| BacktestData | formula、formula_text、metrics、dates、equity_curve、benchmark、benchmark_equity、daily_returns、turnover、positions_count |
| StrategyData | formula、formula_text、val_reward、val_icir、train/full reward、history |
| SimState | initial_capital、cash、trade_count、market_value、total_equity、positions、equity_history |
| SimRunStatus | state、pid、started/stopping/ended_at、exit_code、error、reset、date range、log_path、phase、current_date、equity |
| SimConfigData | effective、overrides_path、overrides、state_initial_capital、pending_reset、execution_config_mismatches |
| DataStatus | db、artifacts、config；顶层 ready 在当前 service 成功路径中没有返回 |

### 6.4 内部关键公开接口

以下是新开发者最常调用或扩展的内部接口；签名以当前源码为准。

| 接口 | 签名摘要 | 职责/调用方 |
|---|---|---|
| <code>load_config</code> | <code>(path=None, project_root=None, overrides_path=None) → dict</code> | 所有 CLI、API、看板的配置事实来源 |
| <code>sync_all</code> | <code>(config_path=None, offline=None, limit=None, sync_fundamentals=None, sync_capital_flow=None) → dict</code> | 同步 CLI/测试 |
| <code>AshareDB</code> | <code>(path, read_only=False)</code>；<code>query/execute/upsert_*</code> | 全部数据层 |
| <code>ProductionGateRunner.run</code> | <code>(mode="formal") → GateResult</code> | dev/formal 审计 |
| <code>ProductionGateRunner.require_production</code> | <code>() → UniverseContractStatus</code> | train/evaluation/diagnostics/backtest/sim/archive/API start |
| <code>resolve_universe_contract</code> | <code>(config, allow_development_fallback=False) → ResolvedUniverse</code> | loader 和 gates |
| <code>build_universe_mask</code> | <code>(codes, dates, sessions, constituents, list_dates, bar_presence, policy) → UniverseMask</code> | PIT eligibility 唯一构造器 |
| <code>AshareDataLoader.load_data</code> | <code>(ts_codes=None, dates=None) → self</code> | trainer/backtest/evaluation/sim/diagnostics |
| <code>AshareDataLoader.tradability_masks</code> | <code>() → (blocked_buy, blocked_sell)</code> | reward、backtest 对齐 |
| <code>AshareFactorEngine.compute_factor_tensor</code> | <code>(bars,codes,dates,universe_mask,pit_fundamentals=None,extra_frames=None,industry_frame=None) → ndarray</code> | loader |
| <code>StackVM.execute</code> | <code>(formula_tokens, factor_tensor) → Tensor or None</code> | 所有公式消费者 |
| <code>resolve_formula_tokens</code> | <code>(payload, vocab=None) → list[int]</code> | backtest/sim/旧产物迁移 |
| <code>AshareTrainer.train</code> | <code>(steps=None,batch_size=None,seed=42,save_artifacts=True,train_end_date=None,device=None,window_cap=None)</code> | RL 搜索 |
| <code>AshareTrainer.train_search</code> | <code>(searcher, steps=None, batch_size=None, ...)</code> | GP/random 生产搜索 |
| <code>CandidateScorer.score_many</code> | <code>(specs,signals,target,val_windows,universe_mask=...,blocked_*=...,adv=...) → list[CandidateScore]</code> | 所有搜索器统一评分 |
| <code>CandidateSelector.select</code> | <code>(scores, pareto_objectives=None) → SelectionResult</code> | trainer/search harness |
| <code>formula_reward</code> | <code>(signal,target,bt_cfg,reward_cfg,...,universe_mask,adv=None) → float</code> | 标量参考奖励 |
| <code>batched_basket_rewards</code> | <code>(signals,target,bt_cfg,reward_cfg,...,universe_mask,adv=None) → reward/val/ICIR/objectives</code> | 训练和批量搜索 |
| <code>AshareBacktestEngine.run</code> | <code>(factors,raw_cache,codes,dates,universe_mask,benchmark_returns=None,signal_range=None,execution_delay=1) → BacktestResult</code> | backtest/evaluation/golden |
| <code>run_protocol</code> | <code>(loader,data_cfg,model_cfg,bt_cfg,reward_cfg,proto_cfg,tier_name,...)</code> | evaluation CLI |
| <code>ExperimentLedger</code> | <code>(path, run_id=None)</code>；<code>trial/finalize</code> | protocol 自动试验账本 |
| <code>RegimeRegistry</code> | <code>(path, regime=None)</code>；<code>assert_folds_clear/assert_final_evaluation</code> | protocol/晋级数据边界 |
| <code>evaluate_challenger</code> | 读取协议产物 + current dataset + paper window；<code>allowed_data_tiers=("A",)</code> | 六门晋级（+data_tier；<code>--allow-tier-b</code> 单独对照） |
| <code>PortfolioConstructor.construct</code> | <code>(signal,prev_weights,eligible,buy_blocked,sell_blocked,stable_keys,capital,rebalance_due,... ) → PortfolioOutput</code> | reward/backtest/golden/simulation 唯一生产组合路径 |
| <code>PortfolioOptimizer.solve</code> | <code>(alpha,prev_weights,capital=...,cov=None,industries=None,beta=None,size=None,adv=None) → PortfolioSolution</code> | constructor 的 optimizer 后端及独立约束测试 |
| <code>ExecutionCostModel.rebalance_cost</code> | <code>(buy_weights,sell_weights,capital) → ExecutionCosts</code> | reward/backtest/golden |
| <code>build_orders</code> | <code>(exec_date,codes,open_prices,target_shares,selected,current_quantities,lot_size=100)</code> | sim/golden |
| <code>SimulationRunner.run</code> | <code>(start_date=None,end_date=None,resume=False) → dict</code> | CLI 子进程 |
| <code>SimJobManager.start/status/stop/reset</code> | 文件锁 + 子进程控制 | Web API |

### 6.5 重要产物契约

| 产物 | 默认路径 | 生产者 | 消费者 |
|---|---|---|---|
| 数据库 | <code>data/ashare.duckdb</code> | sync/PIT import/backfill | loader、gates、API/看板 |
| Parquet | <code>data/parquet</code> | sync/backfill | sync 缓存 |
| 策略 | <code>data/best_ashare_strategy.json</code> | trainer | backtest、sim、API、archive |
| RL 权重 | <code>data/ashare_model.pt</code> | 仅 RL trainer | 目前主要归档/展示，不是执行公式所必需 |
| 回测 | <code>data/backtest_result.json</code> | backtest CLI | API、两套看板、archive |
| 因子报告 | <code>data/factor_report.json</code> | diagnostics | 人工研究/归档 |
| 协议 | <code>data/protocol_result.json</code> | evaluation | promotion、archive、研究 |
| 账本 | <code>data/experiment_ledger.jsonl</code> | evaluation | protocol artifact、审计 |
| 数据区间 | <code>data/holdout_registry.json</code> | regime CLI | evaluation/promotion |
| paper windows | <code>data/paper_windows.json</code> | 人工注册 | promotion G5 |
| 模拟状态 | <code>data/sim_portfolio_state.json</code> | SimulationPortfolio | sim/API/看板/archive |
| 逐日订单/成交 | <code>data/sim_orders/YYYYMMDD.json</code>、<code>sim_trades</code> | SimulationRunner | API/分析/归档 |
| 运行记录/进度 | <code>data/sim_run.json</code>、<code>sim_progress.json</code> | Manager/Runner | API 状态 |
| 运行时覆盖 | <code>config/runtime_overrides.yaml</code> | Web config API | 所有 load_config 调用者 |
| 费用矩阵 | <code>data/fee_matrix.json</code> | cost_matrix CLI | 人工研究/归档（P1） |
| 裸因子回测 | <code>data/bare_factor_backtest.json</code> | bare_factor_backtest CLI | 人工研究/归档（P1） |
| 搜索器成本 | <code>data/searcher_bench.json</code> | searcher_bench CLI | 人工研究/归档（P1） |
| selfcheck | <code>data/selfcheck_result.json</code> | evaluation --selfcheck | 人工研究/归档（P1） |
| 研究医生 | <code>data/research_doctor.json</code> | research_doctor CLI | 只读健康检查（P0） |
| 分层报告 | <code>data/tier_report.json</code>、<code>tier_report_diagnostics.json</code> | tier_reports CLI | 研究/晋级准备（P2） |
| 基本面范围 | <code>data/fundamental_scope.json</code> | check_fundamental_scope.py | 治理审计（P2） |
| 数据库备份 | <code>data/ashare.duckdb.p2bak</code> | P2 purge 前备份 | 回滚参考（P2） |

## 7. 依赖关系

### 7.1 Python 直接依赖

当前精确版本来自 [requirements.txt](../requirements.txt)。

| 依赖 | 当前 pin | 用途 | 主要调用模块 |
|---|---:|---|---|
| torch | 2.11.0+cpu（基础 pin；GPU 按 [requirements-cuda.txt](../requirements-cuda.txt) 换 +cu128） | Transformer、采样、Tensor VM、可选 CUDA | <code>ashare_model</code> |
| numpy | 2.5.2 | 数值矩阵、回测、奖励、执行 | 全部核心层 |
| pandas | 3.0.5 | 表格、滚动窗口、DB frame、数据清洗 | data/model/dashboard |
| cvxpy | 1.6.5 | QP 建模；默认 solver OSQP 来自其依赖 | portfolio |
| akshare | 1.18.91 | A 股外部数据 | data |
| duckdb | 1.5.5 | 嵌入式分析数据库 | data/API/dashboard/scripts |
| pyarrow | 24.0.0 | Parquet | data |
| streamlit | 1.61.1 | 旧看板 | dashboard |
| plotly | 6.9.0 | 旧看板图表 | dashboard |
| loguru | 0.7.3 | 日志 | root/data/model/trading |
| tqdm | 4.70.0 | 长任务进度 | data/model/scripts |
| fastapi | 0.141.1 | Web API | webapi |
| uvicorn | 0.52.3 | ASGI server | 部署 |
| psutil | 7.2.2 | 模拟子进程存活/终止 | trading.manager |
| python-dotenv | 1.2.3 | <code>config/.env</code> | data.config |
| PyYAML | 6.0.3 | YAML 配置/覆盖/归档 | data/webapi/scripts |

### 7.2 Python 可选/测试依赖

| 依赖 | 当前 pin | 用途 |
|---|---:|---|
| pytest | 9.1.1 | 测试 |
| scipy | 1.18.0 | 统计/数值测试 |
| deap | 1.4.4 | 强类型 GP；P0-01 已移入基础依赖（GP 是生产默认搜索器） |
| optuna | 4.9.0 | TPE 协议基线 |
| httpx2 | 2.12.0 | starlette TestClient（test_webapi）依赖；P0 修复后已收录 |

存在两处依赖声明缺口：

1. <code>baostock==0.9.3</code> 只出现在完整 [requirements.lock](../requirements.lock#L9)，不在直接或 optional spec/pin 中，但 [ashare_data/akshare_client.py](../ashare_data/akshare_client.py#L514) 和 [scripts/import_pit_universe.py](../scripts/import_pit_universe.py#L138) 会动态 import；干净安装无法完成正式 PIT bootstrap/兜底。
2. <code>mlflow</code> 是 [ashare_model/experiment_tracking.py](../ashare_model/experiment_tracking.py) 的可选动态依赖，但没有出现在任何 requirements；模块在未安装时会 no-op。

此前 test_webapi 缺 <code>httpx2</code> 的第三处缺口已由 P0 修复：httpx2 现列于 optional pin 且本机已安装。

### 7.3 前端依赖

| 依赖 | package.json 范围 | 用途 |
|---|---:|---|
| react / react-dom | ^18.3.1 | SPA |
| react-router-dom | ^6.28.0 | HashRouter |
| antd / icons | ^5.22.5 / ^5.5.2 | UI 组件 |
| echarts / echarts-for-react | ^5.5.1 / ^3.0.2 | 图表 |
| dayjs | ^1.11.13 | 日期 |
| typescript | ^5.6.3 | 类型 |
| vite / react plugin | ^5.4.11 / ^4.3.4 | 构建/开发服务 |

[webui/package-lock.json](../webui/package-lock.json) 锁定实际 npm 依赖树；CI web job 执行 <code>npm ci</code>、<code>npm ls --depth=0</code> 和 <code>npm run build</code>（含 <code>tsc -b</code>），仍没有 lint 或前端测试。

### 7.4 内部依赖约束

- <code>ashare_data</code> 不依赖模型、交易或 Web，是底层。
- <code>ashare_execution.py</code> 只依赖配置和 NumPy，供 model/portfolio/trading 复用。
- <code>ashare_model</code> 依赖 data + execution，不依赖 Web。
- <code>ashare_trading</code> 依赖 data + model + execution。
- <code>webapi</code> 依赖 data + execution + trading，只读模型产物而不 import trainer。
- <code>webui</code> 只通过 HTTP 依赖 webapi。
- <code>dashboard</code> 直接读文件/DB，绕过 webapi。
- <code>ashare_portfolio.golden</code> 跨越 model/execution/trading，是有意的集成规范；optimizer 自身相对独立。

新增功能时应保持依赖向下，不要让 data 反向 import model/trading，也不要把 React 响应结构变成核心领域模型。

## 8. 配置、环境与运行

### 8.1 配置优先级

~~~mermaid
flowchart LR
    YAML[config/ashare_config.yaml 基线]
    Runtime[config/runtime_overrides.yaml]
    Dotenv[config/.env]
    Ambient[进程环境变量]
    Effective[effective raw config]
    Classes[Data/Model/Reward/Protocol/Backtest/Sim Config]

    YAML --> Effective
    Runtime -->|递归覆盖 YAML| Effective
    Dotenv -->|override=false，仅填未设置环境| Ambient
    Ambient -->|只覆盖三个数据路径| Effective
    Effective --> Classes
~~~

实现见 [ashare_data/config.py](../ashare_data/config.py#L351)。

注意：

- YAML 缺失会直接抛 <code>FileNotFoundError</code>，不会静默使用 dataclass 默认值。
- runtime overrides 对字典递归合并，标量和列表整体替换。
- 环境变量只覆盖 <code>data_dir</code>、<code>duckdb_path</code>、<code>parquet_dir</code>。
- <code>config/.env</code> 以 <code>override=False</code> 加载，所以进程已有同名环境变量优先。
- 相对路径实际按传入的 <code>project_root</code> 解析；源码 docstring 写“相对 YAML 所在目录”，两者在自定义外部 YAML 时不一致。

### 8.2 当前关键配置

| 分组 | 当前值 | 说明 |
|---|---|---|
| 数据 | 2015-01-01 至 2026-12-31；<code>qfq</code> | 日线范围/前复权 |
| 股票池 | 000300.SH + 000905.SH | 沪深 300 + 中证 500；无 CSI1000 历史来源 |
| 上市年龄 | 60 个 open sessions | PIT eligible 条件 |
| 日线 provider | auto | Eastmoney 优先、Sina 回退；部分路径再到 Tencent/BaoStock |
| 模型 | d=64、4 heads、2 layers、FF=128、3 loops、dropout=.1 | 小模型 |
| 默认搜索 | gp | RL 准入失败后的生产默认 |
| 默认预算 | 150 × 256；公式最长 12 | unique semantic evaluations 上限通常为 steps × batch |
| 验证 | 尾部 35%，4 子窗中位数 | 与 IS 学习窗隔离 |
| reward | clip [-1,1]、cost weight 1、complexity 0.02、max complexity 25 | YAML 未列字段继承 dataclass |
| 质量门禁 | val reward ≥0、val ICIR ≥.05、8 IC days、coverage .2、activity .05、sign stability .5 | 详见 RewardConfig |
| 容量 | position / execution-day ADV ≤ .25 | 需要 loader 提供 dollar volume |
| 协议 | 5 个年度 OOS fold × 3 seed | 2021-2025 OOS |
| 搜索基线 | 7 裸因子 + random + GP + TPE | matched budget |
| 回测/模拟本金 | 100,000 | Web 修改会写 runtime override |
| 组合 | 日频 horizon=1、equal_weight、买入 Top-20/跌出 Top-30、单票上限 5% | 首次 10 万资金最多 20 笔约 5,000 元；后续应用 1% 权重阈值、5,000 元最小交易额和 20% L1 换手预算 |
| 费用 | 佣金 .025%、最低 5；印花税卖出 .05%；过户 .001%；滑点 .05% | 训练/回测/模拟共享 |

配置基线见 [config/ashare_config.yaml](../config/ashare_config.yaml)；未显式列出的 reward 默认值见 [ashare_data/config.py](../ashare_data/config.py#L101)。

### 8.3 环境变量与本地秘密

| 名称 | 用途 | 读取位置 |
|---|---|---|
| <code>ASHARE_DATA_DIR</code> | 数据目录 | [ashare_data/config.py](../ashare_data/config.py#L400) |
| <code>ASHARE_DUCKDB_PATH</code> | DuckDB 路径 | 同上 |
| <code>ASHARE_PARQUET_DIR</code> | Parquet 路径 | 同上 |
| <code>ASHARE_OFFLINE</code> | 强制 AkShareClient 离线 fixtures | [ashare_data/akshare_client.py](../ashare_data/akshare_client.py#L159) |
| <code>ASHARE_VM_STRICT</code> | VM 出错时重抛，供测试/调试 | [ashare_model/vm.py](../ashare_model/vm.py#L171) |
| <code>MLFLOW_TRACKING_URI</code> | 可选 MLflow 后端 | [ashare_model/experiment_tracking.py](../ashare_model/experiment_tracking.py#L66) |
| <code>ALPHAGPT_DISABLE_TRACKING</code> | 强制禁用 MLflow | [ashare_model/experiment_tracking.py](../ashare_model/experiment_tracking.py#L41) |
| <code>ASHARE_WEBAPI_ROOT</code> | 仅测试中重定向 token root | [webapi/auth.py](../webapi/auth.py#L27) |

<code>ASHARE_ALLOW_DEVELOPMENT_UNIVERSE_FALLBACK</code> 在旧测试中出现，但正式代码有意不读取它；开发降级只能通过显式函数/CLI 参数，避免环境变量暗中放宽 PIT 约束，见 [tests/test_universe.py](../tests/test_universe.py#L530)。

秘密只有可选的 <code>config/.webapi_token</code>。外部数据源当前不需要 API key。不要把 token 放入 <code>.env</code> 后以为 API 会读取它；auth 只读独立文件。

### 8.4 解释器与安装

必须先确认解释器：

~~~powershell
Set-Location D:\minequant\AlphaGPT
python -c "import sys; print(sys.executable)"
~~~

本机应使用：

~~~powershell
& D:\minequant\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
python -m pip check
python scripts/freeze_lock.py --check
~~~

在新机器创建环境的建议基线：

~~~powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-optional.txt
python -m pip install baostock==0.9.3
# 有 NVIDIA 显卡时再把 torch 换成 CUDA wheel（与基础同版本，P0-06）
python -m pip install -r requirements-cuda.txt
~~~

风险说明：

- 基础 pin 已是 CPU torch（自带 PyTorch CPU index，P0-06 拆分），干净环境和 CI 不再需要 CUDA wheel；GPU 机器按 <code>requirements-cuda.txt</code> 显式替换，版本一致性由 freeze_lock --check 校验。
- <code>requirements.lock</code> 含 <code>win32_setctime</code> 等 Windows 特定包，不应直接拿到 Linux CI 安装。
- <code>baostock</code> 仍只存在于 lock，干净环境须手装（见 §7.2）。

### 8.5 数据初始化：全新环境

以下命令会写大量本地数据并访问外网。运行前先确认磁盘、网络和数据备份；不要在分析/代码 review 时顺手执行。

由于 PIT 导入依赖已存在的 schema、日历和 daily max，一个稳妥的 fresh bootstrap 是：

~~~powershell
# 1. 创建 schema、日历、股票和当前成员数据
python -m ashare_data.sync

# 2. 重建沪深300/中证500历史成员区间并补上市日
python scripts/import_pit_universe.py

# 3. 用“当前快照 + PIT成员 + 缓存”并集补齐历史成员数据
python -m ashare_data.sync

# 4. 修复仍然为零 bar 的退市/合并/长期停牌成员
python scripts/backfill_member_bars.py

# 5. 如果第4步写入了数据，再跑一次同步以刷新 dataset manifest
python -m ashare_data.sync

# 6. 正式门禁
python scripts/check_production_gates.py
~~~

重要副作用：

- 非 <code>--limit</code> 的完整 sync 会删除不在最终 universe 且未失败的旧 daily rows 和 Parquet，见 [ashare_data/sync.py](../ashare_data/sync.py#L313)。
- PIT import 和 backfill 会改 DuckDB；backfill 后若不重建 manifest，manifest 与实际数据可能不一致。
- <code>--offline</code> 使用工作日近似日历并只适合 fixtures/开发，不能产出正式结果。
- <code>--limit</code> 会限制股票数且跳过 purge，适合烟雾检查，但 G6/G7 正式门禁通常不会通过。
- P2-01 后，怀疑 <code>fundamental_pit</code>/<code>stocks</code> 混入范围外行时，先跑 <code>python scripts/check_fundamental_scope.py --report</code>（只读审计）再决定是否 <code>--purge</code>；purge 前已有备份 <code>data/ashare.duckdb.p2bak</code>。



### 8.6 研究与模拟 CLI

~~~powershell
# 因子诊断
python -m ashare_model.diagnostics

# 只读研究医生（门禁/依赖/运行量估算，P0）
python -m ashare_model.research_doctor

# P1 测量：费用矩阵 / 七裸因子固定回测 / 搜索器成本
python -m ashare_model.cost_matrix
python -m ashare_model.bare_factor_backtest
python -m ashare_model.searcher_bench --budget 128

# P2 分层诊断与消融（A / A+B / all）
python scripts/tier_reports.py --steps 50 --batch-size 256

# P2 基本面表范围审计（--purge 才会删除行）
python scripts/check_fundamental_scope.py --report

# 默认 GP 训练；GPU 只用于 VM，policy/采样始终 CPU
python -m ashare_model.train --device auto

# 当前策略回测
python -m ashare_model.backtest

# 快速测量协议与纯噪声自检
python -m ashare_model.evaluation --tier screening
python -m ashare_model.evaluation --selfcheck

# 更昂贵的确认档
python -m ashare_model.evaluation --tier confirmation

# 模拟：已有状态必须显式 resume 或 reset
python -m ashare_trading.run_sim --resume
python -m ashare_trading.run_sim --reset

# Streamlit 旧看板
streamlit run dashboard/app.py
~~~

模拟 CLI 当前根目录已有 STOP 文件；不带 <code>--resume</code>/<code>--reset</code> 的首次命令不会清除它，可能立即停止。不要手工删除状态文件来规避；应先按归档/重置流程处理。

### 8.7 React + FastAPI

开发模式：

~~~powershell
# 终端 1
python -m uvicorn webapi.app:app --host 127.0.0.1 --port 8000 --reload

# 终端 2
Set-Location webui
npm ci
npm run dev
~~~

浏览器访问 <code>http://127.0.0.1:5173</code>，Vite 把 <code>/api</code> 代理到 8000。

本地单进程模式：

~~~powershell
Set-Location webui
npm ci
npm run build
Set-Location ..
python -m uvicorn webapi.app:app --host 127.0.0.1 --port 8000
~~~

必须在启动 Uvicorn 前完成 build，因为 [webapi/app.py](../webapi/app.py#L166) 只在模块 import 时检查 <code>webui/dist</code> 是否存在。

部署建议：

- 默认只绑定 <code>127.0.0.1</code>。
- 如经反向代理或 LAN 暴露，必须创建强随机 token，并在前端/代理补齐 <code>X-API-Token</code> 传递；当前 React 客户端不发送该头。
- 读取接口和日志也应在代理层加认证。
- 反向代理把后端连接来源显示为 loopback 时，不能依赖“无 token 时只允许 loopback”的保护。
- 当前架构不适合多实例：DuckDB 单写、JSON 状态、进程锁和本地文件路径都假设单主机。

## 9. 测试、验证与质量现状

### 9.1 已核对结果

| 检查 | 结果 | 说明 |
|---|---|---|
| <code>pip check</code> | 通过 | 使用 <code>D:\minequant\.venv</code>（2026-08-28 核验复跑） |
| <code>freeze_lock.py --check</code> | 通过 | 只核对 direct/optional pin 与 CUDA 一致性，不核对完整平台 lock（2026-08-28 核验复跑） |
| TypeScript <code>tsc --noEmit</code> | 通过（基线时点） | 没有执行 Vite build，避免写 dist |
| pytest collection | 已无 blocker | P0 起 <code>httpx2</code> 已入 optional pin 且本机已装，test_webapi 可收集 |
| 最新阶段记录 | P0 921 → P1 949 → P2 最终 981 passed | P2 日志口径排除 test_webapi，见 [docs/phase6_measurement_log.md](phase6_measurement_log.md#L25) |
| 生产数据门禁 | G1-G7 全通过 | 2026-08-28 核验只读复跑：min eligible 473（2015）、2,426 个成员区间 0 个零 bar |

本次核验没有重跑完整 pytest（测试会生成被忽略的日志），数字以阶段日志为准；基线时点的“887 collected / 1 collection error”已因 httpx2 修复而过时。

### 9.2 推荐验证命令

~~~powershell
& D:\minequant\.venv\Scripts\Activate.ps1
python -m pip check
python scripts/freeze_lock.py --check

# 当前环境可运行的全量测试（httpx2 已入 pin，P0 起无需排除 test_webapi）
python -m pytest -q tests

Set-Location webui
npm ci
.\node_modules\.bin\tsc --noEmit
npm run build
~~~



