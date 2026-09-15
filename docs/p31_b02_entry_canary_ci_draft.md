# P31 B02 真实入口 canary 与 CI 质量预注册契约

**状态**：APPROVED（用户 2026-09-15 会话直接指令「阅读整改方案，完成 B02，开两个子 agent 审核工作，不
开多余子 agent」；依 p18 Captain 调度权条款与 p29/p30 先例（用户直接指令 + captain 代行签署）登记）。

**勘察基线**：分支 `codex/b01-pit-data-quality` @ `88a784e`（B01 交付后；A01–A05 与 B01 均已在该分支
完成交付并双审）。本契约起草时工作树除既有未跟踪 `reports/alphagpt-ab/`（历史治理记录，非本包范围）
外干净。

## 1. 问题陈述

整改方案（`D:/minequant/reports/AlphaGPT_整改与提升执行方案_2026-09-09.md` §四 B02）确认：
1. 项目没有任何走完「真实配置解析 → RunSpec/资格 → 搜索/固定候选 → 回测 → 产物登记 → API 读取」
   全链的小型确定性 canary；契约级错误只能在昂贵研究运行中暴露。
2. CI 的 web job 只有 `npm ci → npm ls → build`，没有执行 `npm test` / `npm run lint`（F10）；本地
   `node_modules` 过期导致 `npm test` 找不到 vitest（F11），前端测试从未成为合并门禁。

## 2. 范围与非目标

**范围内**：`tests/`（新增 canary 测试文件）、`.github/workflows/ci.yml`（web job 门禁）、
`scripts/check_test_shards.py`（新测试文件同 commit 分片注册）、`docs/PROJECT_ONBOARDING.md` §9.2
（与 CI web job 同步）、`tests/test_validation_docs.py`（web 门禁镜像守卫同步）、`webui/`（如本地
发现前端测试缺口则补，不重写既有测试）。

**非目标**：不改任何研究语义（数据、公式、Reward、执行、统计、晋级语义零变化）；不新增 ESLint
（方案 B02 明示独立评估，本包仅保留 `lint = tsc --noEmit` 并如实记录）；不新增 CI 矩阵腿（Windows/
跨 Python 版本覆盖由本地测量 + 既有 ubuntu/3.12 CI 构成，方案要求"合理覆盖"）；不动 `reports/`
既有治理记录；不删除/弱化既有测试。

## 3. 行为契约（canary 测试断言的唯一来源）

### 3.1 canary 数据库（小型、确定性、无网络）

合成双库（primary + v2，复用 `tests/conftest.py` 既有发布/装载 helper，不新增第二套 DDL 或发布实现）：
- **多股票多会话**：12 只合成股票 × 40 个业务日会话（2024-01-01 起）；
- **PIT 成分**：全部股票为 `000300.SH` 自 2020-01-01 成分（`min_listed_sessions=1`）；
- **公告延迟**：v2 财务表携带 `ann_date`/`available_date` 晚于 `end_date` 的报表行（as-of 语义）；
- **复权**：至少一只非核心股票携带常数 `adj_factor ≠ 1`（adjusted 基数 ≠ raw，收益率不变）；
- **交易阻断**：一个停牌单元 + 一个涨停阻断买入口（`daily_suspension` / `daily_price_limit` 精确规则）；
- **确定性行情**：两条上升路径股票 + 十条常数路径股票，价格为手工构造的已知数值。

### 3.2 闭环（真实入口，全部形式路径）

C1 配置解析：`load_config()` 从真实 YAML 文件解析（非 Python dict 直构）。
C2 身份与资格：`confirm_input_snapshot()` 得到复合身份 → `resolve_runtime_runspec()` 冻结 RunSpec →
   LifecycleWriter 走完 `IDEA → SPEC_LOCKED → DATA_QUALIFIED → FACTOR_SET_QUALIFIED →
   SEARCH_PLAN_ADMITTED`（真实 writer + 证据校验）。
C3 最小搜索：train 形式入口（`--spec-id/--run-id`，`--steps 1 --batch-size 4`，`searcher: random`）
   在 canary 数据上产出策略产物（RunStore 登记）。`reward` 下限按 canary YAML 显式声明
   （`make_reward_config` 全字段透传）；这是单次运行的实验配置，不是生产默认值变化。
C4 回测：backtest 形式入口消费 C3 的注册策略（registered-artifact 校验 + RunContext + dataset 一致性），
   产出 backtest 产物并登记进同一 run。
C5 API 读取：`webapi.service.list_runs/get_run_detail` 以配置派生的 RunStore 根读到该 run 的状态、
   事件链与策略/backtest 产物（真实入口写 → API 读同根，F05/A01 闭环面）。

### 3.3 新增关键行为覆盖（与既有测试的映射）

| 行为 | 本包动作 | 既有覆盖（不重复实现） |
|---|---|---|
| 数据身份失效阻断 | canary：篡改 v2 内容后形式入口拒绝（`DatasetIdMismatch` 族），且不产生新业务产物 | A02 p26 契约测试（协议层） |
| run 根一致 | canary C5（真实入口写 → API 读） | A01 C8/C9（fixture run + API） |
| 非法生命周期无业务写入 | 映射既有测试，canary 不复制 | A03 R12（p27 契约测试） |
| 未来扰动不影响过去 | canary：篡改未来 bar 后，过去会话的 VM 特征输出逐元素不变 | test_price_semantics（除权维度）、test_universe（成员维度） |
| 信号 t → 执行 t+1 | 映射既有测试 | test_backtest::test_engine_signal_date_and_entry_date_alignment、test_golden_parity（t+2） |
| 连续日期轴调仓节奏 | 映射既有测试 | test_rebalance_policy（含周末缺口轴） |
| 现金、费用、整手、T+1 与账户对账 | canary：对真实 loader 缓存跑引擎，用**手工推导**的首期净收益、权益曲线、持仓权重与日期对账 | test_portfolio_constructor / test_execution / test_trading / test_core（T+1）/ test_golden_parity |
| 重复同日执行不能重复成交 | 映射既有测试 | test_run_sim（重放拒绝 + resume 水位）、test_jobmanager |

手算样例独立性：费用公式取自执行契约（`ashare_execution.ExecutionCostModel` 文档字符串：买单
`commission=max(rate×notional, min)+transfer+slippage`，卖单另加印花税；`fraction = total/capital`），
费率取自 canary 自己写入的 YAML；价格/日期取自 canary 构造的行情与日历。预期值不调用被测引擎生成。

### 3.4 CI 与文档门禁

- CI web job 变为 `npm ci → npm ls --depth=0 → npm test → npm run lint → npm run build`
  （方案 B02：`npm ci → npm test → npm run lint → npm run build`；`npm ls` 为既有步骤保留）。
- `docs/PROJECT_ONBOARDING.md` §9.2 与 CI web job 同步（新增 `npm test`、`npm run lint`）；
  `tests/test_validation_docs.py::test_onboarding_web_gates_match_ci` 同 commit 扩展镜像断言。
- 新测试文件同 commit 注册进 `scripts/check_test_shards.py`（fail-closed）。

## 4. 不变量与 RED 策略

- **I1** 闭环全部走形式路径；任何一环拒绝时 run 不得新增业务产物（strategy/protocol/backtest/paper_state）。
- **I2** canary 断言不产生网络/时钟依赖：行情日期固定 2024 轴；G8 评估日经
  `production_gate_asof` 注入（与 A03 测试同机制）。
- **I3** 手算样例值在测试内从场景构造推导，不从被测实现输出复制。
- **RED**：(a) `test_validation_docs` 扩展断言在 onboarding/CI 更新前失败；(b) canary 为守卫型测试
  （对应 §2.2.3"无法产生有意义 RED 必须有守卫"）：其 RED 阶段以真实运行记录——若全绿则记录
  "守卫即交付"，若暴露真实缺陷则缺陷先以最小失败证据记录再修复。

## 5. 版本影响表

| 版本 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| protocol/reward/grammar/feature registry/research domain/search contract/execution/lifecycle 等全部 `*_VERSION` | 现值 | 不变 | 零研究语义变化（纯测试/CI/文档门禁）；`git grep` 复核无版本所有者被触碰 |

## 6. 验收

1. canary 测试文件全绿且耗时远小于全量测试（目标 < 120 s，含 DB 构建与最小搜索），重复运行结果一致；
2. `python scripts/check_test_shards.py --check` 通过（新文件已注册）；
3. `test_ci_sharding.py`、`test_validation_docs.py` 全绿（CI/文档镜像同步）；
4. webui `npm test`、`npm run lint` 本地通过（干净 `npm ci` 后）；
5. 相邻契约测试（p26/p27/p30 相关文件）无回归；
6. 测量日志（evidence-only，引用精确被测 commit）记录环境、命令、耗时、警告与未运行项。

## 7. 停止条件

- 发现真实缺陷时：先以最小 RED 证据记录，再决定本包内修复（行为修复需符合 §2.2）或另立包；
  禁止为绿灯放宽断言。
- train 最小搜索在 canary 数据上无法产生被选候选（系统性门槛不满足）时：降级为"固定候选经真实
  资格链 + 评价入口"的等价闭环（方案 B02 明示允许"最小搜索**或**固定候选"），并在测量日志记录原因。
- 触碰数据库移动/删除/旧工作区产物处置时停止并列明清单（方案 A01 停止条款同款）。

## 8. 修订记录

### 修订 1（RED 阶段发现的真实缺陷，依 §7 授权本包内修复）

**发现**：canary 闭环（C3→C4 边界）在 RED 探针中暴露真实缺陷——形式 train 入口在"首个唯一语义
评价不合格"时无法落盘策略产物：search 契约（`search_contract.SearchResult`）允许 best-so-far 曲线
以 `-inf` 起步（运行最优在尚无合格候选时为 `-inf`；`tests/test_evaluation.py` 已构造过该合法状态），
而类型化产物经 fail-closed canonical JSON（`identity.canonical_json_strict`）拒绝一切非有限浮点。
两个权威矛盾：任何首评价不合格的正式搜索（GP/random/TPE；RL 的 `best_val_reward` 同理）都会在
`CanonicalJSONError` 上崩溃。RED 证据：canary 闭环探针输出
`CanonicalJSONError: value.history[0].avg_reward: non-finite float -inf …`
及修复一层后的 `value.search_result.best_so_far[0][1]` 同类失败。

**修复（工程遥测表示，非研究语义）**：
1. `train_loop.history_reward_value()`：历史遥测中"尚无最优"表示为 `None`（JSON null；
   `StrategyArtifact.history: list[dict]` 本就接受），RL 步历史的 `best_val_reward` 同规则；
2. `train_artifacts._json_telemetry_value()`：产物载荷内嵌遥测（search_result 曲线、score、
   imitation 等）的非有限浮点叶子统一表示为 `null`——JSON 无 ±inf，null 是无损表示；
   身份字段（字符串/整数/布尔）不受影响，canonical writer 保持 fail-closed 兜底。

**版本影响**：全部 `*_VERSION` 不变（`history: list[dict]` 与 `search_result: dict | None`
均为宽松类型，载荷内容表示变化不构成 schema 版本变化；不新增/修改任何研究语义常量）。
**新增 RED**：`test_search_history_maps_no_best_yet_to_none`（D1，先于实现失败）+ 上述端到端
失败输出。**GREEN**：canary 全文件 + `test_train_module_split`（导入面）+ `test_artifact_schemas`
+ `test_artifact_writer` 全绿。
