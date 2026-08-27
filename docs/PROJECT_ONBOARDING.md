# AlphaGPT 项目架构与新开发者上手指南

> 分析日期：2026-08-27（Asia/Shanghai）  
> 分析基线：<code>main</code> @ <code>c5b801e936f6ef6cdba4c80ff1e81d12a7387ca6</code>  
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
| 当前奖励实现是 v13：组合主动 IR 减精确年化费用；ICIR 是辅助指标 | README 和 YAML 注释仍描述旧的 rank-ICIR 主奖励，阅读时以源码为准 | [ashare_model/reward.py](../ashare_model/reward.py#L15)、[ashare_model/reward.py](../ashare_model/reward.py#L765)、[README.md](../README.md#L273)、[config/ashare_config.yaml](../config/ashare_config.yaml#L61) |
| 本机数据库的 G1-G7 生产门禁当前全部通过 | PIT 成分、上市日、日历、最小股票数和历史成员 bar 覆盖目前可用；这不等价于策略或产物可用于决策 | [ashare_data/gates.py](../ashare_data/gates.py#L1)、[scripts/check_production_gates.py](../scripts/check_production_gates.py) |
| 本机现有策略、回测和协议产物彼此不一致且全部落后于当前源码代际 | 看板目前可能把不同公式、不同版本、不同数据时间的结果拼在一起；接手后不能直接引用这些数字 | [data/best_ashare_strategy.json](../data/best_ashare_strategy.json)、[data/backtest_result.json](../data/backtest_result.json)、[data/protocol_result.json](../data/protocol_result.json) |
| 当前没有可称为最终 holdout 的历史区间，也没有当前 v20/v13 的显著 alpha 证据 | 2021-2026 已被反复查看，只能算开发/验证数据；旧协议曾显示无显著候选 | [docs/phase4_measurement_log.md](phase4_measurement_log.md#L15)、[docs/evaluation_20260823.md](evaluation_20260823.md#L163) |
| 组合优化器已经实现并有黄金一致性测试，但未接入默认训练、回测或模拟盘 | 当前生产策略仍是 top-N 等权加单票权重上限；不要从目录名推断已启用优化组合 | [ashare_portfolio/optimizer.py](../ashare_portfolio/optimizer.py#L105)、[ashare_portfolio/golden.py](../ashare_portfolio/golden.py#L180)、[ashare_model/backtest.py](../ashare_model/backtest.py#L339) |
| 项目面向研究和 paper trading，不连接真实券商 | 所有成交由本地模拟撮合器生成，状态保存在 JSON；没有实盘下单适配器 | [ashare_trading/matching.py](../ashare_trading/matching.py)、[ashare_trading/run_sim.py](../ashare_trading/run_sim.py#L83) |

### 0.1 当前本机快照

以下是分析时的只读观测，不是仓库承诺的固定数据。

| 项目 | 当前值 |
|---|---|
| Git | <code>main</code> 与 <code>origin/main</code> 对齐；分析开始时工作区干净 |
| Python | 项目说明要求 3.10+；CI 使用 3.12；本机可用项目环境为 <code>D:\minequant\.venv\Scripts\python.exe</code>，Python 3.13.12 |
| 默认 shell Python | <code>C:\ProgramData\miniconda3\python.exe</code>；它不是当前依赖完整的项目环境 |
| Node / npm | Node 24.14.0 / npm 11.9.0 |
| DuckDB | 786,444,288 bytes；日线 4,874,595 行，2015-01-05 至 2026-08-21 |
| 股票/成分 | 5,546 只股票元数据；2,574 条 PIT 成分区间 |
| 生产门禁 | G1-G7 全部通过；每年最少 eligible 股票数 473；2,426 个有效成员区间中零 bar 区间为 0 |
| 数据集清单 | 当前数据库没有 <code>dataset_manifest</code>/<code>dataset_manifest_cache</code> 表，因此加载器会把 <code>dataset_id</code> 降级为 <code>None</code> |
| STOP 信号 | 根目录现有被忽略的 <code>STOP_SIGNAL</code>，内容为 <code>STOP</code> |
| 协议治理文件 | <code>experiment_ledger.jsonl</code>、<code>holdout_registry.json</code>、<code>paper_windows.json</code>、<code>promotion_verdict.json</code> 均不存在 |

数据库表的当前行数：

| 表 | 行数 | 说明 |
|---|---:|---|
| <code>stocks</code> | 5,546 | 股票快照、上市日、当前 ST 标识 |
| <code>daily_bar</code> | 4,874,595 | 前复权日线 |
| <code>constituents</code> | 2,574 | 沪深 300/中证 500 PIT 成分半开区间 |
| <code>trade_calendar</code> | 8,797 | 交易日历 |
| <code>fundamental_pit</code> | 322,876 | 财务 PIT 近似数据 |
| <code>margin_balance</code> | 5,642,388 | 融资余额 |
| <code>sw_industry_index</code> | 141,378 | 申万行业指数历史 |
| <code>sw_industry_member</code> | 5,196 | 当前申万行业成分映射 |
| <code>factor_cache</code> | 0 | 预留表；当前因子仍在加载时重新计算 |

### 0.2 当前运行时产物一致性审计

| 产物 | 当前内容 | 与当前源码的差异 |
|---|---|---|
| [data/best_ashare_strategy.json](../data/best_ashare_strategy.json) | 公式为 <code>(VOL_20 CORR60 (ATR_14 ADD MAX3((RET_10 ADD (PS_TTM MUL TS_RANK20(DIVIDEND_YIELD))))))</code>，方向 -1，reward v10 | 当前 reward v13；缺少 <code>searcher</code>、<code>protocol_version</code>、<code>model_version</code>、<code>dataset_id</code> 和语义缓存元数据 |
| [data/backtest_result.json](../data/backtest_result.json) | 公式为 <code>LIMIT_BREAK</code>，2015-01-06 至 2026-08-14，累计收益 -100%，Sharpe -1.956 | 公式与当前策略 JSON 不同；无 <code>dataset_id</code>；属于旧 schema |
| [data/protocol_result.json](../data/protocol_result.json) | protocol v12、reward v10、60 行候选 | 当前 protocol v20、reward v13；没有 stitched OOS、dataset、ledger 或 data-regime 块 |
| [data/sim_portfolio_state.json](../data/sim_portfolio_state.json) | 2,822 个权益点、28,179 笔成交、最后日期 2026-08-14 | 旧状态没有 <code>last_exec_date</code>、公式、配置版本或 <code>dataset_id</code>；续跑时可能把新策略接到旧权益曲线上 |

结论：当前 UI 只能视为“历史文件查看器”，不能被当作一组同源、同版本、可复现的研究结果。任何策略判断前，应先生成数据 manifest，再按当前版本重新训练、回测、执行 v20 协议，并通过晋级门禁。

## 1. 项目概览

### 1.1 项目做什么

AlphaGPT 的核心目标是自动发现可解释的 A 股横截面选股公式。公式由基础特征和算子组成，表示为带独立 EOS 的后缀 token 序列；AST 是语义事实来源，StackVM 负责执行。搜索器不直接输出“明天涨跌概率”，而是在约束公式空间内生成或优化表达式，再用实际可交易的 top-N 组合表现评分。

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
- 唯一 CI 是 GitHub Actions：Ubuntu、Python 3.12、安装两份 pin 文件、执行 pip check、lock check 和全量 pytest，见 [.github/workflows/ci.yml](../.github/workflows/ci.yml)。
- 前端没有测试、lint 或 CI 构建步骤。
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
        Score[Reward v13 + CandidateSelector]
        Backtest[回测与 v20 评价协议]
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
    M --> N[开盘 t+1 建仓]
    N --> O[开盘 t+2 收益标签/退出]
    O --> P[主动 IR - 精确年化费用 + 质量/复杂度/容量门禁]
    P --> Q[策略 JSON]
    Q --> R[回测 / v20 协议 / 模拟盘]
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
    CLI->>File: 写版本、词表、搜索器、dataset_id、历史
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
| 收益标签 | signal 日 t；t+1 开盘进入；t+2 开盘退出，目标为 <code>open[t+2]/open[t+1]-1</code> | [ashare_model/data_loader.py](../ashare_model/data_loader.py#L299)、[ashare_model/time_contract.py](../ashare_model/time_contract.py) |
| 训练/验证 | 策略梯度只读 IS 头部；验证尾部切成 4 个子窗，以中位数选公式 | [ashare_model/train.py](../ashare_model/train.py#L880) |
| 评价 trial | v20 中一个 trial 是一个 <code>(candidate, seed)</code> 跨折拼接 OOS 序列，不是一折一行 | [docs/phase4_measurement_log.md](phase4_measurement_log.md#L23) |
| PIT 选择 | 信号日和入场日必须 eligible；退出成员通过正常卖出路径处理 | [ashare_model/reward.py](../ashare_model/reward.py#L17)、[ashare_model/backtest.py](../ashare_model/backtest.py#L65) |
| no-signal | 可选截面少于两个不同值时保持原仓，不做信号驱动换手 | [ashare_model/reward.py](../ashare_model/reward.py#L36)、[ashare_trading/run_sim.py](../ashare_trading/run_sim.py#L283) |

## 4. 目录结构详解

### 4.1 总览

~~~text
AlphaGPT/
├─ ashare_data/          数据源、清洗、PIT 股票池、DuckDB、manifest、生产门禁
├─ ashare_model/         因子、公式语言、搜索、奖励、回测、评价与晋级治理
├─ ashare_portfolio/     独立组合优化器与回测/撮合黄金一致性规范
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

仓库跟踪文件分布：<code>ashare_data</code> 14 个、<code>ashare_model</code> 28 个、<code>ashare_portfolio</code> 3 个、<code>ashare_trading</code> 7 个、<code>webapi</code> 4 个、<code>webui</code> 18 个、<code>scripts</code> 9 个、<code>tests</code> 65 个、<code>experiments</code> 95 个。<code>experiments</code> 中的大 JSON 占绝大多数行数，做源码搜索或统计时应排除它。

### 4.2 根目录

| 级别 | 文件 | 职责 |
|---|---|---|
| 核心 | [ashare_execution.py](../ashare_execution.py) | 回测、训练奖励、组合黄金规范和模拟撮合共享的唯一费用模型；佣金最低额、印花税、过户费、滑点、可买股数 |
| 辅助 | [ashare_logging.py](../ashare_logging.py) | Loguru 控制台/文件/内存配置；10 MB rotation、14 份 retention、最多 10,000 行内存、文本导出 |
| 文档 | [README.md](../README.md) | 主运行说明；覆盖面广但奖励、manifest 和部分版本描述已漂移 |
| 文档 | [CATREADME.md](../CATREADME.md) | 仓库速读；仍写 18 个算子，当前实际为 39 个 |
| 依赖 | [requirements.in](../requirements.in)、[requirements.txt](../requirements.txt) | 直接依赖的人读清单和精确 pin |
| 依赖 | [requirements-optional.in](../requirements-optional.in)、[requirements-optional.txt](../requirements-optional.txt) | 测试/统计/GP/TPE 可选依赖 |
| 依赖 | [requirements.lock](../requirements.lock) | 当前开发机完整环境快照，含平台特定包 |
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
| 核心 | [alphagpt.py](../ashare_model/alphagpt.py) | Looped Transformer、RMSNorm、QK norm、SwiGLU、actor/critic heads；MODEL_VERSION=2 |
| 核心 | [train.py](../ashare_model/train.py) | RL/GP/random 统一训练窗口、语义预算、候选选择、策略/模型产物写入 |
| 核心 | [reward.py](../ashare_model/reward.py) | reward v13；组合主动 IR、统一 basket、精确费用、ICIR 辅助统计 |
| 核心 | [candidates.py](../ashare_model/candidates.py) | CandidateSpec/Score、方向对称评分、质量/复杂度/容量门禁、选择 |
| 核心 | [complexity.py](../ashare_model/complexity.py) | AST 节点、深度、最长窗口和操作成本的复杂度账单 |
| 核心 | [signal_quality.py](../ashare_model/signal_quality.py) | HAC 有效样本 ICIR、block bootstrap、覆盖率/活跃度/符号稳定性 |
| 核心 | [semantic_cache.py](../ashare_model/semantic_cache.py) | 规范 AST + 校准切片数值指纹；按 dataset/reward/protocol/window 隔离预算 |
| 搜索 | [gp_search.py](../ashare_model/gp_search.py) | DEAP 强类型 GP |
| 搜索 | [tpe_search.py](../ashare_model/tpe_search.py) | Optuna TPE |
| 搜索 | [baseline_harness.py](../ashare_model/baseline_harness.py) | matched unique-semantic-evaluation 预算和统一搜索评价适配器 |
| 搜索治理 | [admission.py](../ashare_model/admission.py) | RL 与 random/GP/TPE 的预注册准入裁决 |
| 评价 | [backtest.py](../ashare_model/backtest.py) | 连续权重 top-N 回测、基准、费用、持仓快照和指标 |
| 评价 | [evaluation.py](../ashare_model/evaluation.py) | v20 nested walk-forward、基线、拼接 OOS、DSR、max-t、自检 |
| 评价 | [pareto.py](../ashare_model/pareto.py) | 多目标 Pareto frontier 辅助 |
| 治理 | [ledger.py](../ashare_model/ledger.py) | append-only JSONL 试验账本、序列和 SHA-256 hash chain |
| 治理 | [regime.py](../ashare_model/regime.py) | dev cutoff、预锁 final slice、dataset 绑定和违规拒绝 |
| 治理 | [promotion.py](../ashare_model/promotion.py) | Champion/Challenger 五门晋级与成本/容量压力网格 |
| 时间 | [time_contract.py](../ashare_model/time_contract.py) | t/t+1/t+2 和 fold 内标签边界 |
| 诊断 | [diagnostics.py](../ashare_model/diagnostics.py) | 因子覆盖率、rank-IC、相关性报告 |
| 实验 | [experiment_tracking.py](../ashare_model/experiment_tracking.py) | 可选 MLflow；无 URI/无包时结构化 no-op |
| 兼容 | [ir.py](../ashare_model/ir.py)、[vocab.py](../ashare_model/vocab.py) | 旧 token/裸因子迁移和别名解析 |
| 包入口 | [__init__.py](../ashare_model/__init__.py) | 包标识 |

当前语义版本：

| 组件 | 版本 |
|---|---:|
| 模型 | <code>MODEL_VERSION = 2</code> |
| 奖励 | <code>REWARD_VERSION = 13</code> |
| 评价协议 | <code>PROTOCOL_VERSION = 20</code> |
| 公式语法 | <code>GRAMMAR_VERSION = 2</code> |
| feature registry | 1 |
| semantic cache | 1 |
| dataset manifest | 1 |

### 4.5 ashare_portfolio：组合优化与黄金规范

| 级别 | 文件 | 职责 |
|---|---|---|
| 核心但未接线 | [optimizer.py](../ashare_portfolio/optimizer.py) | CVXPY/OSQP 长仓 QP；alpha、风险、换手、冲击、行业/beta/size 暴露、ADV 容量约束 |
| 集成测试 | [golden.py](../ashare_portfolio/golden.py) | 将连续回测权重通过 lot-free/whole-lot 撮合重放，分解费用、阻塞、手数残差 |
| 包入口 | [__init__.py](../ashare_portfolio/__init__.py) | 公开 optimizer/golden 类型 |

仓库内除包入口、测试和 golden harness 外，没有生产模块 import <code>PortfolioOptimizer</code>。默认 [ashare_model/backtest.py](../ashare_model/backtest.py#L339) 和 [ashare_trading/run_sim.py](../ashare_trading/run_sim.py#L317) 仍生成等权 top-N。

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
| [admission_experiment.py](../scripts/admission_experiment.py) | RL 与 random/GP/TPE 五 seed 准入实验 |
| [analyze_sim.py](../scripts/analyze_sim.py) | 汇总模拟盘日文件和交易表现 |
| [archive_run.py](../scripts/archive_run.py) | 归档公式、配置、指标、模型 hash 和 commit；带 <code>--commit</code> 会创建 Git commit |
| [freeze_lock.py](../scripts/freeze_lock.py) | 从当前解释器已安装包生成 pin/完整 lock；无参数会改写依赖文件，<code>--check</code> 才是只读核对 |

### 4.9 tests：测试结构

65 个跟踪文件中包含 57 个测试模块、1 个 <code>conftest.py</code> 和 7 个 JSON fixtures。命名基本与生产模块一一对应：

- 数据：<code>test_akshare_client</code>、<code>test_sync</code>、<code>test_db</code>、<code>test_universe</code>、<code>test_gates</code>、<code>test_manifest</code>、<code>test_fundamentals</code>、<code>test_capital_flow</code>。
- 公式/模型：<code>test_factors</code>、<code>test_ops</code>、<code>test_vm</code>、<code>test_vocab</code>、<code>test_ir</code>、<code>test_grammar</code>、<code>test_train</code>、<code>test_candidates</code>。
- 研究有效性：<code>test_evaluation</code>、<code>test_stitched_oos</code>、<code>test_ledger</code>、<code>test_regime</code>、<code>test_promotion</code>、<code>test_semantic_cache</code>、<code>test_admission</code>。
- 执行：<code>test_backtest</code>、<code>test_execution</code>、<code>test_trading</code>、<code>test_run_sim</code>、<code>test_jobmanager</code>、<code>test_golden_parity</code>。
- UI/API：<code>test_dashboard</code>、<code>test_webapi</code>。
- 完成性：<code>test_completion_gates</code> 聚合检查版本、文档/配置和关键契约。

[tests/conftest.py](../tests/conftest.py) 提供公共 fixtures/日志；[tests/fixtures](../tests/fixtures) 提供离线股票、日历、成分和两只股票的日线。

### 4.10 配置、文档、实验与遗留资产

| 目录 | 职责/现状 |
|---|---|
| [config](../config) | [ashare_config.yaml](../config/ashare_config.yaml) 是版本化基线；[.env.example](../config/.env.example) 只列三个数据路径变量；真实 <code>.env</code>、<code>.webapi_token</code> 和 <code>runtime_overrides.yaml</code> 被忽略 |
| [docs](.) | 2026-08-23 旧工程评估和 Phase 1-4 测量日志；旧评估基于更早 commit，部分缺陷已修复，不能照单全收 |
| [experiments](../experiments) | 只增不改的研究快照；当前有多个 2026-08-15 至 2026-08-23 归档和 [admission_experiment.json](../experiments/admission_experiment.json) |
| [assets](../assets) | 两张无 provenance 的旧回测图片 |
| [paper](../paper) | 一篇与 A 股主线无关的 Uniswap V4 论文 |
| [.github](../.github) | 单一 Python CI workflow |
| <code>data</code> | gitignored，约含 DuckDB、Parquet、策略、模型、回测、协议、模拟状态、逐日订单/成交 |
| <code>logs</code> | gitignored，当前约 1,600 个历史文件；API 可读取其尾部 |

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
| Nested walk-forward 评价 | [evaluation.py](../ashare_model/evaluation.py) | 候选、fold、seed | protocol v20 JSON |
| 试验账本/数据区间治理 | [ledger.py](../ashare_model/ledger.py)、[regime.py](../ashare_model/regime.py) | trial/fold/dataset | hash-chain ledger、registry |
| 策略晋级 | [promotion.py](../ashare_model/promotion.py) | v20 协议、paper windows、当前数据 | 五门 verdict |
| QP 组合优化 | [optimizer.py](../ashare_portfolio/optimizer.py) | alpha、前仓、风险/暴露/ADV | PortfolioSolution；默认主链未消费 |
| 回测/撮合黄金一致性 | [golden.py](../ashare_portfolio/golden.py) | 回测目标权重和原始 bar | ParityReport |
| 模拟订单与撮合 | [orders.py](../ashare_trading/orders.py)、[matching.py](../ashare_trading/matching.py) | 目标权重、现金、bar | SimOrder/SimTrade |
| 模拟状态/续跑 | [portfolio.py](../ashare_trading/portfolio.py)、[run_sim.py](../ashare_trading/run_sim.py) | 策略、旧状态、日期区间 | portfolio/progress/逐日流水 |
| 模拟子进程控制 | [manager.py](../ashare_trading/manager.py)、[signals.py](../ashare_trading/signals.py) | API 操作 | run record、锁、STOP、子进程 |
| Web/API 看板 | [webapi](../webapi)、[webui](../webui) | DB/JSON/logs | 六页 SPA |
| 简版看板 | [dashboard](../dashboard) | DB/JSON | Streamlit 五 tab |
| 归档/复现 | [scripts/archive_run.py](../scripts/archive_run.py)、[experiments](../experiments) | 运行产物 | manifest/config/metrics/formula/model hash |

### 5.1 62 个基础因子

当前稳定顺序和 feature version 来自 [ashare_model/feature_registry.py](../ashare_model/feature_registry.py) 与 [ashare_model/vocab.py](../ashare_model/vocab.py#L213)。当前 <code>feature_version = 29ac4001dd3c</code>。

| 家族 | 因子 |
|---|---|
| 收益/动量/反转 | RET_1、RET_5、RET_10、MOMENTUM_20、MOMENTUM_60、REVERSAL_5、HIGH_52W、RET_120、REVERSAL_60、REVERSAL_120 |
| 波动/分布 | VOL_20、VOL_60、SKEW_20、KURT_20 |
| 量价/换手 | TURNOVER、TURNOVER_CHG、VOLUME_RATIO、VOLUME_IMPACT、AMPLITUDE、CLOSE_POSITION、TURNOVER_MA5、TURNOVER_MA20、TURNOVER_STD20 |
| 基本面/估值 | PE_TTM、PB、PS_TTM、ROE、ROA、GROSS_MARGIN、NET_MARGIN、REVENUE_YOY、PROFIT_YOY、DEBT_RATIO、MARKET_CAP、DIVIDEND_YIELD |
| 外部资金/行业 | NORTHBOUND_CHG、MARGIN_BALANCE_CHG、INDUSTRY_MOMENTUM |
| 涨跌停事件 | LIMIT_UP_EVENT、LIMIT_DOWN_EVENT、LIMIT_STREAK、LIMIT_UP_CNT_20、LIMIT_BREAK |
| 日内分解 | OVERNIGHT_RET、INTRADAY_RET |
| 流动性/彩票 | ILLIQ_20、AMOUNT_SHARE、MAX_20 |
| 风险回归 | BETA_60、IVOL_60、RSQ_60 |
| 技术指标 | BIAS_20、RSI_14、ATR_14、MACD_DIF、MACD_DEA |
| 微观结构 | SUSPEND_DAYS_60、LIST_AGE |
| 行业相对 | IND_REL_RET_5、IND_REL_RET_20、IND_REL_VOL_20、IND_REL_TURNOVER |

特殊语义：

- <code>NORTHBOUND_CHG</code> 是中性 0 占位，因为北向日度明细自 2024-08 起停止披露，见 [README.md](../README.md#L335)。
- 申万行业指数是历史序列，但股票到行业的映射是当前快照投射到历史。
- 财报可见日使用法定披露季末近似，不是逐股首发公告日；重述只保留当前最新值。
- <code>PS_TTM</code> 为近似口径，<code>MARKET_CAP</code> 为成交额/换手率得到的流通市值近似。

### 5.2 39 个公式算子

| 类型 | 算子 |
|---|---|
| 二元算术 | ADD、SUB、MUL、DIV |
| 一元变换 | NEG、ABS、SIGN |
| 条件/动态 | GATE、JUMP、DECAY、DELAY1、MAX3 |
| 差分 | DELTA5、DELTA10、DELTA20 |
| 移动均值 | MA5、MA10、MA20、MA60 |
| 移动标准差 | STD5、STD10、STD20、STD60 |
| 时序排名 | TS_RANK5、TS_RANK10、TS_RANK20、TS_RANK60 |
| 滚动相关 | CORR5、CORR10、CORR20、CORR60 |
| 下行波动 | DOWNVOL5、DOWNVOL10、DOWNVOL20、DOWNVOL60 |
| 横截面 | CS_RANK、CS_ZSCORE、CS_DEMEAN、CS_NEUTRALIZE |

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
| <code>evaluate_challenger</code> | 读取 v20 artifact + current dataset + paper window | 五门晋级 |
| <code>PortfolioOptimizer.solve</code> | <code>(alpha,prev_weights,capital=...,cov=None,industries=None,beta=None,size=None,adv=None) → PortfolioSolution</code> | 当前仅测试/golden |
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

## 7. 依赖关系

### 7.1 Python 直接依赖

当前精确版本来自 [requirements.txt](../requirements.txt)。

| 依赖 | 当前 pin | 用途 | 主要调用模块 |
|---|---:|---|---|
| torch | 2.11.0+cu128 | Transformer、采样、Tensor VM、可选 CUDA | <code>ashare_model</code> |
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
| deap | 1.4.4 | 强类型 GP；由于 GP 是默认 searcher，实际已接近运行时必需 |
| optuna | 4.9.0 | TPE 协议基线 |

存在三处依赖声明缺口：

1. <code>baostock==0.9.3</code> 只出现在完整 [requirements.lock](../requirements.lock#L9)，不在直接或 optional spec/pin 中，但 [ashare_data/akshare_client.py](../ashare_data/akshare_client.py#L514) 和 [scripts/import_pit_universe.py](../scripts/import_pit_universe.py#L138) 会动态 import；干净安装无法完成正式 PIT bootstrap/兜底。
2. <code>mlflow</code> 是 [ashare_model/experiment_tracking.py](../ashare_model/experiment_tracking.py) 的可选动态依赖，但没有出现在任何 requirements；模块在未安装时会 no-op。
3. 当前 Starlette 1.3.1 的 TestClient 需要 <code>httpx2</code>，但 lock 和 pin 均没有它；因此 [tests/test_webapi.py](../tests/test_webapi.py#L10) 当前无法收集。

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

[webui/package-lock.json](../webui/package-lock.json) 锁定实际 npm 依赖树；CI 当前没有执行 <code>npm ci</code>、TypeScript 检查或 build。

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
| 组合 | top 30、单票上限 5% | 实际可能因上限/整手保留现金 |
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
python -m pip install "torch==2.11.0+cu128" --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt -r requirements-optional.txt
python -m pip install baostock==0.9.3
~~~

风险说明：

- <code>requirements.txt</code> 自己已经 pin <code>torch==2.11.0+cu128</code>，但没有声明 PyTorch index；在普通 PyPI/CI 环境可能无法解析或会下载巨大的 CUDA 构建。
- README 的“先安装 requirements，再单独安装 CUDA torch”顺序与现有 pin 冲突。
- CPU-only 机器仍会被要求安装 CUDA 构建；应由维护者拆分 CPU/CUDA constraints，而不是新开发者私自改 lock。
- <code>requirements.lock</code> 含 <code>win32_setctime</code> 等 Windows 特定包，不应直接拿到 Linux CI 安装。

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

### 8.6 现有本机数据的接手顺序

当前 G1-G7 已通过，但 manifest 表缺失、全部研究产物落后。建议在获准写运行时数据后按以下顺序恢复，而不是直接点“模拟盘续跑”：

1. 备份 <code>data</code> 和当前 Git commit 信息。
2. 清点/归档现有策略、回测、协议和模拟状态，明确它们只能作为 legacy evidence。
3. 运行一次当前 sync，生成 <code>dataset_manifest</code>；必要时 full verify。
4. 再跑 G1-G7，记录完整 JSON。
5. 用当前默认 GP、reward v13、protocol v20 重新训练。
6. 用同一策略和 dataset_id 重新回测。
7. 初始化 ledger 和 data regime，跑 screening/selfcheck，再决定是否付出 confirmation 成本。
8. 只有 v20 统计门禁和未来 paper window 都满足后，才讨论 challenger promotion。
9. 模拟盘应先归档并 reset，不能在当前 legacy equity history 上 resume 新策略。

本次分析没有执行上述任何写操作。

### 8.7 研究与模拟 CLI

~~~powershell
# 因子诊断
python -m ashare_model.diagnostics

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

### 8.8 React + FastAPI

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
| <code>pip check</code> | 通过 | 使用 <code>D:\minequant\.venv</code> |
| <code>freeze_lock.py --check</code> | 通过 | 只核对 direct/optional pin，不核对完整平台 lock |
| TypeScript <code>tsc --noEmit</code> | 通过 | 没有执行 Vite build，避免写 dist |
| pytest collection | 887 tests collected，1 collection error | <code>tests/test_webapi.py</code> 因缺 <code>httpx2</code> 失败 |
| 最新阶段记录 | 887 个非 Web API 测试通过 | Phase 4 明确排除了 test_webapi，见 [docs/phase4_measurement_log.md](phase4_measurement_log.md#L3) |
| 生产数据门禁 | G1-G7 全通过 | 本次只读运行当前 DB |

本次没有运行完整 pytest，因为测试会生成被忽略的日志，而且已在 collection 阶段发现依赖 blocker。不要把“887 collected”写成“887 当前全通过”；准确说法是“阶段日志记录 887 个非 Web API 测试通过，当前全套 collection 被 Web API 依赖阻塞”。

### 9.2 推荐验证命令

~~~powershell
& D:\minequant\.venv\Scripts\Activate.ps1
python -m pip check
python scripts/freeze_lock.py --check

# 当前环境可运行的非 Web 基线
python -m pytest -q tests --ignore=tests/test_webapi.py

# 修复依赖后必须恢复为真正全量
python -m pytest -q tests

Set-Location webui
npm ci
.\node_modules\.bin\tsc --noEmit
npm run build
~~~

### 9.3 质量保障的优点

- 测试与核心源码规模接近，模块覆盖广。
- 有 PIT 未来成员哨兵、t/t+1/t+2 时间契约、CPU/GPU VM、resume 等价、no-signal、语义 cache、ledger 篡改、locked holdout、promotion all-gates 等高价值测试。
- 回测、训练 reward 和模拟撮合共享费用与大部分可交易性语义。
- 语义变化有 MODEL/REWARD/PROTOCOL/GRAMMAR 版本。
- 协议从单折行升级为 stitched OOS trial，并有 Deflated Sharpe/max-t 多重检验。

### 9.4 质量保障缺口

- 声明的 CI 当前很可能在全套 collection 或 CUDA torch 安装处失败。
- CI 不跑前端 typecheck/build。
- 没有 Python lint、format、typecheck 或覆盖率门槛。
- 没有端到端浏览器测试。
- 阶段测量日志手工排除 Web API，与 CI 的全量 pytest 命令不一致。
- 没有针对“策略 + 回测 + protocol + sim 必须同一 dataset/formula/version”的产物集合一致性测试。

## 10. 潜在风险、技术债与注意事项

### 10.1 P0：会让研究/决策结论失真的风险

#### R-01 现有运行时产物不构成同一实验

现状：

- 策略是复杂公式、reward v10；
- 回测是 <code>LIMIT_BREAK</code>、-100%；
- protocol 是 v12/v10；
- 源码是 protocol v20/reward v13；
- 三者都没有有效 dataset_id；
- 模拟状态没有公式/数据/配置 provenance。

影响：概览页会把不相干的策略、回测和模拟历史并排展示，使用者很容易误以为它们来自同一次运行。

建议：

- 引入 <code>run_id/artifact_set_id</code>，策略、回测、协议、模拟状态都记录 formula hash、dataset_id、config hash、版本和 Git commit；
- API overview 在不一致时返回显式 red status，而不是静默组合；
- 当前所有 data JSON 标记 legacy，只读归档后重跑。

证据：[webapi/app.py](../webapi/app.py#L66)、[webapi/service.py](../webapi/service.py#L90)、当前 <code>data/*.json</code>。

#### R-02 当前没有最终 holdout，也没有当前代际的显著 alpha

2021-2026 已被反复查看，只能作为 dev/validation。2026-08-23 的旧 screening 结果为 DSR 0.043、max-t p=1.0，没有候选显著；其最佳非基准仍是已知因子。之后代码已升级，因此这个结果不能代表 v20/v13，但同样不能被升级本身推翻。

影响：任何“已发现可部署 alpha”的陈述都没有当前证据。

建议：

- 先创建 data regime registry；
- 把未来数据或预先锁定且无人查看的切片作为最终评价；
- 用 v20 完整记录所有试错并完成 future paper window；
- 只有 promotion 五门全过才允许进入 challenger。

证据：[docs/evaluation_20260823.md](evaluation_20260823.md#L163)、[docs/phase4_measurement_log.md](phase4_measurement_log.md#L15)、[ashare_model/promotion.py](../ashare_model/promotion.py#L1)。

#### R-03 dataset lineage 是 fail-open，且 manifest cache 不能可靠发现同分区值修正

三个层面：

1. 当前 DB 无 manifest 表；[AshareDataLoader.load_data](../ashare_model/data_loader.py#L304) 捕获所有异常并把 dataset_id 设为 None，正式门禁仍通过。
2. 回测和模拟加载策略时没有调用 <code>check_dataset_id</code>；即使 DB 有 manifest，策略与当前数据不匹配也不会阻止执行，见 [ashare_model/backtest.py](../ashare_model/backtest.py#L520) 和 [ashare_trading/run_sim.py](../ashare_trading/run_sim.py#L115)。
3. manifest cache 的复用键只有 partition 的 row_count 和 max key。上游更正同一日期的值、但行数和最大日期不变时，默认 <code>use_cache=True</code> 可能复用旧 hash；sync 因而可能生成不变的 dataset_id。只有 full verify 或禁用 cache 才会重新 hash，见 [ashare_data/manifest.py](../ashare_data/manifest.py#L364)。

影响：所谓 content-addressed dataset_id 可能缺失或对值级修正不敏感，实验可复现性和晋级 P0 门禁被削弱。

建议：

- 将“manifest 存在且可 full verify”加入 formal gate；
- backtest/sim/train/API start 强制 artifact dataset_id == current dataset_id，不再接受 None；
- sync 的最终 manifest 生成禁用 cache，或让 cache 额外绑定数据库变更版本/分区内容校验；
- backfill/PIT import 后强制重建 manifest。

### 10.2 P1：高优先级运行/安全风险

#### R-04 模拟盘 resume 没有绑定策略、数据和配置

[SimulationPortfolio](../ashare_trading/portfolio.py#L83) 状态只含本金、现金、持仓、交易数、last date 和权益历史；公式、direction、formula hash、dataset_id、reward/protocol、费用和 Git commit 都不在状态中。<code>run_sim --resume</code> 会读取当前 <code>best_ashare_strategy.json</code> 接着跑。

影响：更换策略/数据库/费用后续跑，会把多个制度拼进一条权益曲线，且没有可检测标志。当前 legacy state 已满足这个风险条件。

建议：状态增加 immutable run contract；任何不匹配默认拒绝 resume，只允许 archive + reset 或显式 migration。

#### R-05 状态损坏会被自动重置并覆盖

[SimulationPortfolio.load](../ashare_trading/portfolio.py#L60) 捕获任何异常后调用 <code>reset()</code>，后者立即原子写一个空状态。虽然避免崩溃，但会覆盖损坏文件，丢失现场和持仓恢复证据。

建议：把坏文件原子移动到 quarantine，停止运行并要求人工恢复；绝不能自动开一个空账户继续。

#### R-06 loader universe 外的旧持仓可能永久滞留

[build_orders](../ashare_trading/orders.py#L109) 对不在当前 <code>ts_codes</code> 的持仓直接跳过；[run_sim.py](../ashare_trading/run_sim.py#L394) 只能用 <code>last_price</code> 估值。若配置指数变化、数据缺失或股票从 loader 彻底消失，该仓位不会生成卖单。

建议：建立 orphan holdings reconciliation：独立查询持仓 bar/退市状态，显式强平、现金结算或阻止运行。

#### R-07 Web 安全边界只适合纯本机，且 React 与 token 模式不兼容

- 全部 GET，包括持仓、模拟状态、完整配置摘要和日志内容，都无鉴权。
- React [client.ts](../webui/src/api/client.ts#L33) 从不发送 <code>X-API-Token</code>，UI 也没有 token 输入；创建 token 后所有控制按钮会 401。
- 无 token 时基于 <code>request.client.host</code> 判断 loopback；反向代理会让远端请求看起来来自 127.0.0.1。
- CORS 不能阻止 curl、同机恶意进程或错误代理访问。

建议：保持 loopback；若远程使用，统一认证所有 API、让前端安全注入 token/会话、配置可信代理头并在代理层二次鉴权。

#### R-08 干净环境安装与 CI 不闭环

- 默认 GP 依赖 DEAP，但 DEAP 放在 optional；
- BaoStock 被运行时使用但未列 direct/optional；
- Web API 测试缺 httpx2；
- CUDA torch pin 没有 index；
- Linux CI 与 Windows full lock 不同；
- 阶段日志排除 Web API，CI 却不排除。

影响：新开发者很可能在安装或 test collection 就失败；main 的绿色状态不能从当前文件静态保证。

建议：用 pyproject/constraints 建立 base、research、web、dev、cpu/cuda extras；在干净 Windows/Linux job 实际安装；CI 增加前端。

#### R-09 数据质量门禁覆盖 universe/bar，但不覆盖因子源完整性

[build_pit_frames](../ashare_data/fundamentals.py#L269) 和 [build_capital_frames](../ashare_data/capital_flow.py#L194) 在表缺失/查询失败时降级为中性 frame；G1-G7 不检查财务、融资、行业的日期覆盖或缺失率。

影响：formal 训练可能在某个因子族全为中性的情况下继续，结果只在日志中留 warning。

建议：新增按因子族的数据 SLO 门禁和 artifact coverage 摘要；生产模式对必需源 fail-closed，开发模式才中性降级。

#### R-10 组合优化模块“存在但未部署”

QP optimizer 和黄金 harness 测试充分，但默认搜索、回测、模拟都不消费 optimizer 输出。

影响：开发者/文档可能误报“系统已有行业中性、风险约束、冲击优化的生产组合”；实际只有等权 top-N。

建议：要么明确标记 experimental，要么建立单一 PortfolioConstructor 接口并同时接入 reward/backtest/sim/golden。

#### R-11 完整 sync 和 Web reset 都包含易被忽略的破坏性副作用

- full sync 会 purge 不在 universe 的日线和 Parquet；
- Web sim reset 先运行 <code>archive_run.py --mode sim --commit</code>，会创建 Git commit；归档失败则 reset 失败；
- reset 会把 orders/trades 目录整体改名为 timestamp backup。

影响：运维操作会删除/移动本地数据或改变 Git 历史，与按钮名称表达的范围不完全一致。

建议：UI/CLI 显示 dry-run plan、精确目标和预计空间；归档与 Git commit 解耦；提供显式确认和恢复说明。

### 10.3 P2：中优先级技术债

#### R-12 文档和源码语义漂移

- README/YAML/RewardConfig docstring 仍写 rank-ICIR 主奖励或 v12；
- CATREADME 写 18 个算子，当前 39；
- README 的 manifest 段仍写“不记录数据 hash”，当前源码已记录 dataset_id；
- README 的旧策略描述和当前 data 文件不一致。

建议：把版本常量、因子/算子数、默认搜索器、配置 schema、CLI/API 表由代码生成并在 completion test 中比对。

#### R-13 Data Status 只说明“文件存在”，不说明“可用于生产”

[webapi/service.py](../webapi/service.py#L219) 只统计 stocks、daily rows、日期和 artifact stat，不运行 G1-G7、不检查 manifest、公式/版本/dataset 一致性，也不返回成功路径的顶层 <code>ready</code>。

建议：新增 readiness 聚合：gates、manifest verify、artifact set compatibility、STOP/run state、protocol/promotion 状态。

#### R-14 双 UI 与大量可选字段导致 schema 漂移

Streamlit 直接读原始 JSON；React 通过 service 做兼容映射；TypeScript 多数字段可选。缺少 schema version/Pydantic response model。

建议：定义版本化 artifact schemas 和 FastAPI <code>response_model</code>，两套 UI 共用同一 service 或退役旧 UI。

#### R-15 没有正式数据库迁移机制

<code>create_schema</code> 里做 IF NOT EXISTS 和 constituents 主键特例迁移；manifest 表又在 save 时创建。没有 schema version、upgrade/downgrade 或备份策略。

建议：建立 schema_version + 可重复 migration，启动前备份并验证；不要继续把迁移散落在业务函数。

#### R-16 全量加载是稠密且昂贵的

62 × 约 2,400 股票 × 2,800 日期的 float32 因子张量约 1.7 GiB，还不含 9 个原始矩阵、Pandas 宽表、industry、targets 和 reward float64 批次。虽然 reward chunk 约束为 512 MB，冷启动和内存峰值仍高。<code>factor_cache</code> 当前 0 行。

建议：按日期/因子分块、持久化版本化因子 cache、memory-map/Arrow、基于 registry 只算公式需要因子；记录峰值内存 benchmark。

#### R-17 本地文件状态缺少事务级一致性

单个 JSON 用原子替换，但一天的 orders、trades、portfolio、progress 是多文件顺序写；进程崩溃仍可能留下跨文件不一致。watermark 降低了重放风险，但没有 WAL/transaction id。

建议：每执行日写 transaction directory/commit marker，恢复时验证全套 hash。

#### R-18 运行状态的 exit_code 基本无法得到

Manager 启动后只存 PID，不保留 Popen 句柄；状态通过 psutil 判断死亡并看 progress phase，<code>exit_code</code> 没有赋值路径。

建议：增加 watcher 进程/线程 wait 子进程并持久化真实 return code。

#### R-19 日志保留策略不能有效控制跨运行文件增长

每次运行使用新的 timestamp 文件 handler；Loguru 的 rotation/retention 主要约束该 handler 的旋转文件，另有每次导出的 txt。当前已有约 1,600 个日志文件。

建议：全局日志清理任务、按 run_id 单文件或结构化日志数据库；API 列表分页。

#### R-20 前端缺测试，错误体与回测 exporter schema 不闭环

client 只读取 <code>body.detail</code>；许多 service 错误以 <code>{ok:false,reason}</code> 且 HTTP 200/400 返回，UI 可能只显示 statusText，丢失 reason。另一个具体断点是：<code>BacktestResult</code> 内有 <code>daily_returns</code> 和 <code>turnover</code>，React 回测页也绘制“日收益与换手”，但 [backtest.main](../ashare_model/backtest.py#L578) 没有把这两个字段写入 JSON；当前产物确实缺少它们，因此该图没有数据。

建议：统一 Problem Details/error schema；用版本化 response/artifact model 驱动 exporter 和 TypeScript；增加 API contract tests、React component tests 和 e2e。

#### R-21 当前行业/ST/财报近似可能影响历史有效性

- 历史指数成员边界只有月粒度；
- 当前申万行业成分被投射到历史；
- 没有日期化历史 ST，历史涨跌停只按板块 10%/20%；
- 财报采用法定披露季末近似且不追踪重述；
- NORTHBOUND 为 0；
- offline 日历包含节假日。

这些不是实现 bug，而是已知研究假设；产物必须记录并在结论中披露。证据见 [README.md](../README.md#L329)。

**P2 已落实**（`docs/p2_data_tier_contract.md`）：以上近似按数据可信度分层——
月粒度成员/日线属于 Tier A（晋级默认唯一准入）；财报季末近似与两融属于 Tier B
（晋级需单独对照）；当前行业快照/ST 近似/北向占位属于 Tier C（仅研究展示、永不
晋级）。任意公式经 `formula_data_tier_report` 追溯其数据等级，协议产物 v21 起
逐行记录；`fundamental_pit` 表范围治理见 P2-01（`scripts/check_fundamental_scope.py`）。

#### R-22 当前 STOP 文件与旧 sim state 容易造成误操作

直接 run_sim 不带 resume/reset 会先因已有 history 退出；带 resume 会清 STOP 并在旧状态上使用当前策略。UI 启动同样倾向自动 resume。

建议：UI 首屏显示 legacy/incompatible 状态并禁用 resume，直到人工 archive + reset。

### 10.4 P3：清理与可维护性

- [paper/20251226.pdf](../paper/20251226.pdf)、[showcase.png](../showcase.png) 和两张 assets 图片与主线无引用/无 provenance，应移到明确的 archive 或删除。
- 项目不是 installable package，依赖 cwd；建议引入 pyproject、console scripts 和 src/test 配置。
- <code>factor_cache</code> 表目前是死 schema；要么实现版本化缓存，要么移除以免误导。
- 日志、data、experiments 的保留/容量策略没有统一运维文档。
- <code>webapi/service.py</code> 直接返回宽松 dict，内部/外部边界难做静态检查。
- 多处 broad <code>except Exception</code> 为看板容错服务，但会把数据/状态错误变成空对象或中性数据；应按 formal/dev 分级。

### 10.5 风险修复优先顺序

~~~mermaid
flowchart TD
    A[1. 冻结并归档 legacy runtime artifacts] --> B[2. 修复依赖与 CI 可重复安装]
    B --> C[3. 强制 manifest + artifact dataset/formula/config 绑定]
    C --> D[4. 修复 sim resume/corrupt/orphan 状态安全]
    D --> E[5. 统一 API 认证与错误/schema]
    E --> F[6. 重新生成 v13/v20 同源产物]
    F --> G[7. 建立未来 holdout + paper window]
    G --> H[8. 决定是否把 optimizer 接入主链]
    H --> I[9. 性能、迁移、双 UI 与遗留资产清理]
~~~

## 11. 新开发者上手路线

### 11.1 第一天

1. 阅读本指南的第 0、3、8、10 节。
2. 确认 Git 分支和工作区，不要把 <code>data</code>/<code>logs</code> 当作版本化事实。
3. 激活正确 venv，执行 pip check 和 lock check。
4. 阅读 [config/ashare_config.yaml](../config/ashare_config.yaml)，特别是 universe、searcher、fold 和费用。
5. 运行只读 G1-G7 检查，理解每个 gate。
6. 不启动 sim、不 resume、不引用现有 backtest 指标。

### 11.2 建立代码心智模型的推荐阅读顺序

1. [ashare_data/config.py](../ashare_data/config.py)
2. [ashare_data/gates.py](../ashare_data/gates.py) 与 [universe.py](../ashare_data/universe.py)
3. [ashare_model/data_loader.py](../ashare_model/data_loader.py)
4. [feature_registry.py](../ashare_model/feature_registry.py)、[factors.py](../ashare_model/factors.py)
5. [ir.py](../ashare_model/ir.py)、[vocab.py](../ashare_model/vocab.py)、[ops.py](../ashare_model/ops.py)、[vm.py](../ashare_model/vm.py)
6. [reward.py](../ashare_model/reward.py)、[candidates.py](../ashare_model/candidates.py)
7. [train.py](../ashare_model/train.py)、[evaluation.py](../ashare_model/evaluation.py)
8. [backtest.py](../ashare_model/backtest.py)、[ashare_execution.py](../ashare_execution.py)
9. [ashare_trading/run_sim.py](../ashare_trading/run_sim.py)、[matching.py](../ashare_trading/matching.py)、[portfolio.py](../ashare_trading/portfolio.py)
10. [webapi/app.py](../webapi/app.py)、[service.py](../webapi/service.py)、[webui/src/types.ts](../webui/src/types.ts)

### 11.3 常见改动应该落在哪里

| 需求 | 首选修改点 | 必须联动 |
|---|---|---|
| 新基础因子 | feature_registry + factors | 因子测试、vocab feature version、diagnostics、必要的数据门禁 |
| 新公式算子 | ops + IR arity/metadata | grammar/vocab、VM CPU/GPU、因果性测试、复杂度成本 |
| 改 reward | reward + candidates | REWARD_VERSION、产物 schema、基线/准入、README/config 注释 |
| 改公式搜索 | gp/tpe/train/baseline harness | matched semantic budget、admission、protocol rows |
| 改交易费用/规则 | ashare_execution 或 processor/matching | reward、backtest、sim、golden parity、配置 API |
| 改 universe | universe/gates/loader | 全链路未来成员哨兵、benchmark、diagnostics、sim exit |
| 改回测 | backtest | reward/golden/evaluation 和 artifact schema |
| 接入 optimizer | 新 PortfolioConstructor 边界 | reward、backtest、sim、golden、capacity stress |
| 改 API | app/service + Pydantic model | webui types/client、test_webapi、认证 |
| 改模拟状态 | portfolio/manager/run_sim | archive/migration/API types、resume 黄金测试 |

### 11.4 提交前检查

- 是否引入未来数据、训练/验证混用或 universe 泄漏？
- 是否改变 reward/protocol/model/grammar/manifest/execution 语义？若是，版本是否 bump？
- 旧 artifact 的迁移或拒绝政策是否明确？
- data/formula/config/Git provenance 是否写入产物？
- 回测、reward、sim 和 golden 是否仍使用同一交易规则/费用？
- formal 是否 fail-closed，dev 降级是否显式 <code>degraded=true</code>？
- 单元、集成、统计自检、前端类型和 CI 是否都覆盖？
- 文档中的默认搜索器、因子/算子数、版本和命令是否同步？
- 是否意外改写/归档/删除本地 data、logs 或用户未提交内容？

## 12. 术语表

| 术语 | 含义 |
|---|---|
| PIT | Point-in-time；只使用当时可见的成员、上市状态和财务数据 |
| eligible | 当日属于配置指数、已上市满要求 session、状态有效且有 bar |
| signal date | 公式产生横截面分数的 t 日 |
| entry/exec date | t+1 开盘执行 |
| label/exit | t+2 开盘相对 t+1 开盘的收益 |
| active IR | 策略 gross basket return 相对当日 universe 等权基准的年化、有效样本收缩信息比 |
| robust ICIR | 日度横截面 rank IC 的 HAC 有效样本收缩统计；v13 为辅助/门禁指标 |
| semantic budget | 只对唯一规范 AST/数值语义计费的公式评价预算 |
| stitched OOS | 同一 candidate/seed 跨年度 fold 按时间拼接后的完整 OOS 序列 |
| DSR | Deflated Sharpe Ratio，校正候选数、样本长度、偏度和峰度 |
| max-t | 对多次试错的最大统计量校正 |
| degraded | 开发模式或门禁失败后的显式非生产标记 |
| golden parity | 连续回测、lot-free matcher 和 whole-lot 模拟间的可解释一致性规范 |
| paper window | 完成的未来纸面交易观察窗口，是 promotion G5 必需证据 |

## 13. 最终接手建议

把当前仓库视为“工程能力较完整、研究治理正在升级、运行时产物尚未迁移到最新代际”的量化研究平台最准确。

短期不要从“继续调模型”开始。最高价值顺序是：

1. 修复可重复安装和 CI；
2. 建立不可绕过的数据/策略/回测/模拟 provenance；
3. 安全处理 legacy sim state；
4. 生成同一 commit、dataset、formula、config 下的 v13/v20 全套产物；
5. 用真正未来数据和 paper window 证明策略；
6. 再决定是否投入搜索算法、因子扩展或组合优化接线。

这样能先保证“测到的东西是真的、同源的、可复现的”，再讨论它是否更赚钱。
