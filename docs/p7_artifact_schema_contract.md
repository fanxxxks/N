# P7-C 类型化产物契约：strategy / protocol / backtest / paper state

状态：预注册（实现前）
适用版本：新增 `ARTIFACT_SCHEMA_VERSION = 1`（`ashare_model/artifact_schemas.py`）。
**不 bump**：`PROTOCOL_VERSION = 24`、`REWARD_VERSION = 14`、`MODEL_VERSION`、
`GRAMMAR_VERSION = 2`、`SEARCH_CONTRACT_VERSION`、`EXECUTION_SPEC_VERSION = 2`、
`FEATURE_REGISTRY_VERSION = 2`、`DATA_TIER_VERSION = 1`——本契约不改变任何
测量、公式、Reward、执行或晋级语义，只给既有 JSON 产物加类型化校验与
schema 版本字段（纯增量字段）。

本文是 Phase C 测试断言与验收的来源。仲裁顺序：本文 > 测试 > 实现。

## 0. 问题陈述

四类正式边界产物目前是无校验的松散 dict：

- `data/best_ashare_strategy.json`（写：`AshareTrainer._write_artifact`；
  读：`backtest.py`、`run_sim.py`、`archive_run.py`、`webapi`）；
- `data/protocol_result.json`（写：`evaluation.run_protocol`；读：
  `promotion.py`、`research_doctor.py`、`load_trial_rows`、`archive_run.py`）；
- `data/backtest_result.json`（写：`backtest.py` CLI；读：`webapi`、
  `archive_run.py`）；
- `data/sim_portfolio_state.json`（写：`ashare_trading/portfolio.py`；
  读：同模块 resume 路径）。

后果：(a) 缺字段在读取侧被 `payload.get(...)` 悄悄变成 `None` 继续流通；
(b) `SimulationPortfolio.load()` 用宽泛 `except Exception` 吞掉一切读取失败
并 `reset()`——**一次损坏的读取会覆盖销毁用户状态文件**（fail-open）；
(c) 产物没有 schema 版本，artifact_versions.py 只能按研究版本间接推断
legacy，无法区分"同研究版本但字段形状不同"的产物。

## 1. 范围与非目标

范围：上述四个文件的写入侧 fail-closed 校验 + 读取侧显式 reject/legacy
矩阵 + `artifact_schema_version` 字段 + Pydantic 显式依赖（C0）。

非目标：

- 不改变任何现有字段的名称、类型或语义；不删除任何字段；
- 不把 rows/trials/equity_history 的每一内层字段都纳入强类型（见 §3
  "脊柱类型化"策略）；
- 不治理 `webapi/service.py` 的防御性 `.get(`（P7 主计划 §1 非目标）；
- 不迁移或重写任何既有 `data/` 产物文件（migration 只发生在"下一次
  正常写入"）；不重跑任何正式运行；
- `bare_factor_backtest.json`、`factor_report.json` 等其他 JSON 不在本期
  （`artifact_versions.py` 的 bare_factor 分类规则不变）。

## 2. Schema 版本与不变量

新增模块 `ashare_model/artifact_schemas.py`，常量 `ARTIFACT_SCHEMA_VERSION = 1`。
四个产物在写入时携带 `"artifact_schema_version": 1`。

不变量（测试直接断言）：

1. 写入侧 fail-closed：四处的写入路径在落盘前通过各自 Pydantic 模型
   校验；构造缺必填 provenance 的 payload 必须抛出 `ValidationError`，
   文件不出现。
2. 读取侧矩阵（§4）被每一处读取路径执行：current 校验通过；
   legacy 按既有规则；unknown/future 版本硬拒绝。
3. 既有产物字节级兼容：schema 只增字段；当前 `data/` 下的 legacy 产物
   （无 `artifact_schema_version`）读取行为与契约前完全一致。
4. `SimulationPortfolio` 永不因读取失败而覆盖既有状态文件（见 §5）。
5. `artifact_versions.py` 的分类规则不变（legacy 判定仍按研究版本）；
   schema 版本是独立的正交维度。

## 3. 类型化策略：脊柱类型化

Pydantic v2 模型（`BaseModel`，`model_config = ConfigDict(extra="allow")`
仅用于内层记录列表；**顶层一律 `extra="forbid"`**，新增顶层字段必须显式
修改 schema 并 bump）：

- `StrategyArtifact`（顶层 forbid）：`artifact_schema_version`、
  `formula: list[int]`、`formula_text: str`、`direction: int`、
  `searcher: str`、`feature_names/operator_names: list[str]`、
  `feature_version: str`、`grammar_version: int`、`model_version`、
  `reward_version`、`protocol_version`、`research_domain`、
  `research_domain_version`、`dataset_id: str | None`（必填键，可为
  null——pre-T1-01 数据库的合法取值）、`semantic_cache_version`、
  `search_contract_version`、`data_tier`、执行 provenance 块
  （`execution_spec_version`/`portfolio_config`/`portfolio_config_hash`，
  形状由 `execution_provenance()` 固定）。legacy 戳记字段
  （`legacy`/`legacy_reason`/`legacy_stamped_at`）为可选——只有被打戳的
  产物携带。`history`/`search_result`/`semantic_cache_stats` 为
  extra-allow 的内层记录。
- `ProtocolResultArtifact`（顶层 forbid）：`artifact_schema_version`、
  `protocol_version`、`reward_version`、`data_tier_version`、
  `research_domain`、`research_domain_version`、`dataset_id`、
  `frequency`、`horizon`、`tier`、`steps`、`batch_size`、`seeds`、
  `folds`、`baseline_signals`、`rows`（list，元素 extra-allow）、
  `aggregates`、`stitched`、`top_trial`、`dsr`、`max_t`。
- `BacktestResultArtifact`（顶层 forbid）：`artifact_schema_version` +
  `backtest.py` CLI 当前写出的全部顶层键（实现时逐一从写方代码著录，
  禁止从既有 JSON 文件反向猜测——文件可能是 legacy）。
- `PaperStateArtifact`（顶层 forbid）：`artifact_schema_version`、
  `initial_capital: float`、`cash: float`、`trade_count: int`、
  `last_exec_date: str | None`、`positions: dict[str, PositionState]`、
  `equity_history: list[EquityPoint]`（`trade_date`/`equity`）。
- 每个模型提供 `validate_payload(payload: dict) -> Model` 类方法与
  `to_payload() -> dict`（JSON 安全、保持既有键序无关语义）。

顶层 forbid 的摩擦是有意的：新增顶层 provenance 必须过 schema 修改 +
版本 bump 的显式决定，禁止悄悄扩散字段。

## 4. 读取侧 reject / legacy 矩阵

| 读取到的 payload | 判定 | 行为 |
|---|---|---|
| `artifact_schema_version == 1` | current | Pydantic 校验；失败即 `ArtifactSchemaError`（新增异常类型），拒绝使用 |
| 无 `artifact_schema_version` 键 | legacy（v0） | 与契约前完全一致的路径：`artifact_versions.py` 分类/打戳；strategy 可读取执行回测但不得冒充 current champion；paper state 可读入并在下一次 save 时自动带上版本（唯一允许的"隐式迁移"：发生在正常写入，不单独改写文件） |
| `artifact_schema_version` 为非整数或 `> 1` | unknown/future | 硬拒绝：抛 `ArtifactSchemaError`，禁止任何降级读取（更新代码再来读） |

提供单一入口 `classify_schema_version(payload) -> Literal["current", "legacy", "unknown"]`
（`ashare_model/artifact_schemas.py`），禁止读取方各自实现第二套判定。

## 5. Paper state 运行态变更（显式声明）

`SimulationPortfolio.load()` 当前 `except Exception → reset()` 会用空状态
**覆盖**损坏或未知版本的既有文件。本契约改为 fail-closed：

- JSON 损坏 / 非 dict / `artifact_schema_version > 1` / current 校验失败：
  抛 `ArtifactSchemaError`，**不写文件**；resume 中止，由人工处置
  （备份仍在，因为不再覆盖）。
- 无版本键（legacy v0）：按现行宽容逻辑读入（含 equity_history 尾部的
  resume watermark 回退），下一次 `save()` 自动携带
  `artifact_schema_version`——不单独迁移旧文件。
- 文件不存在：行为不变（新账户初始状态）。

这是对模拟运行态的语义修正，已按 AGENTS.md §2 任务类型声明；无晋级/
回测/reward 语义影响。既有 `test_trading.py`/`test_run_sim.py` 行为变化
仅限"损坏文件不再被静默重置"——相关断言如有依赖旧 fail-open 行为，按
仲裁顺序回本契约判定后修改（预期存在，属契约声明的合法测试变更）。

## 6. 预期 RED 测试（先写后实现）

1. 四个写入路径：构造缺 `dataset_id` 键（不是 None，是缺键）的 strategy
   payload → `ValidationError`，目标文件不存在。
2. 写入侧 round-trip：`_write_artifact` 产出的真实 payload 通过
   `StrategyArtifact` 校验且 `artifact_schema_version == 1`。
3. 读取矩阵：无版本键 → legacy 路径（与契约前逐字节一致）；`"2"` →
   `ArtifactSchemaError`；`"x"` → 同上。
4. Paper state：损坏 JSON 文件 → 抛错且文件内容未被改写；无版本 legacy
   文件 → 读入成功，save 后带版本；`>1` → 抛错。
5. `classify_schema_version` 是读取方唯一判定入口（grep 级测试：
   run_sim/backtest/webapi 不得各自实现版本比较——允许引用同一函数）。
6. 顶层 forbid：strategy payload 多一个未知顶层键 → `ValidationError`。

## 7. 测量方案与裁决

- 验证命令：聚焦 `tests/test_artifact_schemas.py`（新）+
  `tests/test_artifact_versions.py`、`test_trading.py`、`test_run_sim.py`、
  `test_backtest.py`、`test_evaluation.py`、`test_train.py`、`test_webapi.py`；
  全量 `python -m pytest -q tests`；compileall；`git diff --check`；
  C0 另需 `python -m pip check` 与 `python scripts/freeze_lock.py --check`。
- 不变量：全量 passed 较 Phase B 基线（1115/5/618）只增不减；
  warnings 不增。
- 裁决：任一 RED 测试未先红后绿、或既有产物读取行为出现契约外变化，
  即停止并回到本契约。
- 证据：既有 `data/best_ashare_strategy.json`（legacy 已打戳）与
  `data/sim_portfolio_state.json`（无版本键）在实现后读取行为不变的
  对照记录，写入 `docs/p7_measurement_log.md`。

## 8. 停止条件

1. 发现既有写方实际产出的顶层键与本契约 §3 清单冲突（以代码为准修订
   契约，禁止以既有 JSON 文件为准）；
2. 任一读取方无法在不改变行为的前提下接入 §4 矩阵；
3. C0 依赖固化失败（lock 冲突）——C0 单独 PR，失败即停，不挟带。

## 9. Retirement

- strategy/protocol/backtest/paper state 的未校验 dict 直写路径退役
  （写入必须过 schema）；
- `SimulationPortfolio.load()` 的 `except Exception → reset()` fail-open
  路径退役（被 §5 的显式判定替代）；
- legacy（v0）读取路径永久保留只读兼容，不删除。
