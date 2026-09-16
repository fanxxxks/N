# P33 B04 装载与缓存性能预注册契约（草案）

状态：**APPROVED**，用户 2026-09-16 会话直接指令「完成B04」（授权记录：本会话用户消息，
对齐 p30–p32 先例「用户直接指令 + 执行模型实施 + 双审」）。执行模型起草，分支
`codex/b03-resumable-execution-budget` @ `64b0887`（勘察基线 commit；B04 依赖 A02、A04、
B02——分别在 p26/p28/p31 交付并已在本分支线性链上；B03 已在该分支交付，从该 HEAD 而非
落后于 B03 的 `main` 创建任务分支 `codex/b04-load-cache-performance`）。

本文依据《AlphaGPT 整改与提升执行方案》B04 工作包与 F07 观察起草，是工程行为契约：
不改变任何研究语义（见 §6 版本影响表），先固定行为契约与验收，再 RED、再实现、再测量。

## 1 问题陈述与证据

F07（方案 §2.3）：loader 在缓存查询前已构建 PIT、capital、industry 帧——缓存命中仍
可能支付主要装载成本。方案 B04 要求：优先消除重复工作；将足以判定命中的身份查询前置；
缓存完整需要的帧/张量，使热加载不再先重建昂贵 PIT 帧，仍必须执行资格/身份检查；分别
记录缓存 disabled/miss/hit/invalidated 及原因，仅相关输入变化使缓存失效。

勘察基线事实（2026-09-16，分支 `codex/b03-resumable-execution-budget` @ `64b0887`，
只读源码核对）：

* `AshareDataLoader.load_data`（`ashare_model/data_loader.py` L504–754）现行次序：
  v2 bars 读取 → normalize/adjusted scale → universe mask → raw/adjusted tensor maps →
  `build_pit_frames`（L646）→ `build_capital_frames` + 截面 z（L647–651）→
  `build_industry_member_frame`（L652）→ `industry_index_freshness`（L658）→
  `confirm_input_snapshot`（L687）→ `loader_factor_cache_identity`（L689）→
  `cache.get`（L695）→ miss 分支 `compute_factor_tensor` + `put`。**PIT、capital 与
  因子计算全部先于缓存命中判定执行**——命中路径白白支付这三段（F07 实锤）。
* `confirm_input_snapshot`（`runspec.confirm_composite_snapshot`）每次调用对两库做
  **全量未缓存重哈希**（A02 I2 语义）——它是每次装载的固定身份成本，热/冷路径同样
  支付；本包不得弱化该检查（热加载仍必须执行身份/资格检查，方案 B04 step 2）。
* `FactorCache.get`（`ashare_model/factor_cache.py` L210–256）对「无条目 / 条目损坏 /
  存储身份与请求不一致」三种情形一律返回 `None` 且只区分计数器（`n_misses`/
  `n_rejected`）；loader 侧 `factor_cache_outcome` 只有 `{None, disabled, miss, hit}`
  四值——**invalidated 与其原因不可观测**（方案 B04 step 4 缺口）。
* 查询下推现状：`_load_v2_bars` 已把 ts_code/trade_date 下推进 SQL；
  fundamentals family-⑤ as-of 已按股分组（t74）；`build_capital_frames`
  （`ashare_data/capital_flow.py` L483–617）的 `margin_detail_pit`/`daily_basic_pit`/
  legacy sw-index 查询**只下推 ts_code、不下推窗口日期**——窗口外行被读出后在
  `reindex(columns=dates)` 处丢弃（读取带宽浪费，输出不受影响）。
* 既有权威与先例：身份八元组维度覆盖由 `tests/test_load_cache.py::
  test_identity_binds_every_store_dimension` 锚定；`tests/conftest.py` 的
  `make_bars/_populate_db/populate_v2_*` 是合成双库 fixture 的单一实现，可被脚本复用。

## 2 范围与非目标

**允许修改/新增**：`ashare_model/data_loader.py`（load_data 重排 + 观测字段）、
`ashare_model/load_cache.py`（reason 化身份推导）、`ashare_model/factor_cache.py`
（get 读取侧 last_outcome 观测）、`ashare_data/capital_flow.py`（仅
`build_capital_frames` 的日期下推）、`scripts/bench_load_stages.py`（新增阶段基准，
契约要求的测量工具）、`tests/test_b04_load_cache_performance.py`（新增）、
`scripts/check_test_shards.py`（同 commit 注册新测试）、`docs/CONTRACTS.md`（同
commit 登记本契约）。

**非目标（明确不做）**：

* 不改任何研究语义：protocol/reward/grammar/feature registry/factor compute/data
  tier/搜索语义全部不变；特征数值、可见性、mask、执行语义逐位不变（等价性验收 §4）。
* 不下推 `fundamental_pit`（legacy 主库基本面）查询的日期过滤：TTM 机器消费完整报告
  历史，「可见窗口等价」的下推需要另行预注册的语义分析；主库体量小于 v2，收益次要。
  记为后续小任务，不在本包顺手做。
* 不做「只增量计算受变更影响分区」的增量重算：需要 A02 发布协议提供分区粒度的变更
  记录，属独立工作包。
* 不做数值精度/dtype 修改（方案 B04 step 5 明示另作语义评估）。
* 不做内存映射/进程间共享数组（方案 step 5「必要时」——由 §5 阶段基准先判断；本包
  的重复工作消除不依赖它，若实测热路径仍不达标，在测量日志披露构成并另包处理）。
* 不缓存 `industry_codes`（industry 帧两路径都需构建以产出 VM 的行业分组张量；缓存
  它将改变缓存条目格式，是否值得由 §5 基准的阶段占比决定，本包不做）。
* 不改 VM/execution 下游消费方（StackVM、组合执行不在 B04 文件范围；其耗时属于 B02
  canary 与 R01 工程校准）。
* 不做数据同步/回填/发布，不写任何业务数据库；真实库基准只读 + 写本地 factor cache
  产物（`data/r1/medium_a/factor_tensor_cache/`，A01 语义下的允许输出）。
* 不修改 AGENTS.md；不 push/建 PR/合并；不触碰其他工作区（AlphaGPT-factor/t23）。

## 3 行为契约

### 3.1 I1 身份与命中判定前置（F07 修复）

`load_data` 重排为（全部既有失败语义、公开产物与错误类型保持）：

1. reset → contract → stock meta → codes/dates 校验（原样）；
2. v2 bars 读取、normalize、adjusted scale、universe mask、raw/adjusted tensor
   maps、price_degraded（原样——两条路径都需要）；
3. `industry_index_freshness`（原样，两路径都执行）；
4. **`confirm_input_snapshot` → 身份推导 → `cache.get` 前移**到 PIT/capital/industry
   帧构建之前（confirm 保持在 bars 读取之后：missing-v2 场景维持先抛
   `V2PriceSourceError` 的既有错误类型次序；confirm 的全量重哈希照旧执行，热路径
   不豁免——A02 语义不变）；
5. miss/disabled 分支：`close_wide` → `build_pit_frames` → `build_capital_frames` +
   截面 z（与 hit 共享的 `build_industry_member_frame` 移到分支之后、单次构建）→
   `compute_factor_tensor` → `cache.put`（identity 非 None 时）；
6. **hit 分支：跳过 `close_wide`/`build_pit_frames`/`build_capital_frames`/截面 z/
   `compute_factor_tensor`**；仍执行 `build_industry_member_frame` 与
   `encode_industry_frame`；
7. `industry_codes`、`target_ret` 与返回值组装不变；`factor_cache_outcome`、
   `dataset_identity_status` 等既有字段语义不变。

### 3.2 I2 缓存结果与失效原因观测（方案 step 4）

* `FactorCache.get` 每次读取在实例上记录 `last_outcome ∈ {hit, no_entry, unreadable,
  identity_mismatch}`（返回类型 `np.ndarray | None`、既有计数器与日志事件不变）。
* `load_cache` 新增 `loader_factor_cache_identity_with_reason(...)`：返回
  `(identity | None, reason | None)`，reason ∈ `{identity_inputs_missing,
  ingest_source_missing, ingest_source_empty, marker_tables_missing,
  identity_derivation_failed, None}`；既有 `loader_factor_cache_identity(...)` 成为
  同一实现的兼容投影（返回 `[0]`），不维护第二套推导。
* `AshareDataLoader` 新增实例字段 `factor_cache_reason`（每次 load 重置）：
  `disabled:<reason>`（含 `identity_derivation_failed`）/ `miss:no_entry` /
  `invalidated:unreadable` / `invalidated:identity_mismatch` / `hit`；
  `factor_cache_outcome` 保持既有词汇 `{None, disabled, miss, hit}` 并**新增
  `invalidated`**（请求键下存在存储条目但被拒绝——unreadable 或身份错配——一律重算，
  与 miss 的「无条目」区分）。结果经既有 loguru 记录，不新增第二遥测路径。
* 失效判定的唯一权威仍是身份八元组（维度覆盖由既有
  `test_identity_binds_every_store_dimension` 锚定）：身份变化 → 新键 → miss；
  同键身份载荷错配/文件损坏 → invalidated。任何实际依赖变化不得误命中（§4 R4）。

### 3.3 I3 capital 帧日期下推（方案 step 3，行为不变优化）

* `build_capital_frames` 的 `margin_detail_pit`（available_date）、
  `daily_basic_pit`（trade_date）、legacy sw-index（trade_date）查询把窗口
  `[min(dates), max(dates)]` 下推进 SQL WHERE。等价性依据：三处透视后均
  `reindex(columns=dates)`，窗口外行本就被丢弃——下推只减少读取带宽，不改变任何
  输出值（含 `_MARGIN_WINDOW` 位移与 rolling：它们作用于 reindex 后的窗口列）。
* 既有签名与降级行为（表缺失 → 中性帧）不变；`fundamental_pit` 查询不动（§2 非目标）。

### 3.4 I4 阶段基准与冷/热协议（方案 step 1，engineering 测量）

* `scripts/bench_load_stages.py`：复用 `tests/conftest` 的 fixture 助手构建确定性
  合成双库（默认 120 码 × 300 会话、固定种子、含 financial PIT/margin/basic/industry
  表与真实复合发布），经**真实发布路径**（`publish_composite_snapshot`）取得身份；
  对 `load_data` 的阶段打点：confirm、identity、read、normalize+scale、universe、
  tensor_maps、pit、capital、industry、factor、copy、target_ret、总墙钟；psutil
  采样峰值 RSS；`--json` 输出原始测量。
* 协议四相：cold（空缓存，miss）→ hot（同身份，hit）→ invalidate（v2 追加一条
  ingest 行 + 重新发布 → 必须 miss）→ hot2；各相阶段耗时与结果（outcome/reason）
  全部落 JSON。
* `--real` 模式：真实库固定窗口（`configs/r1_medium_a.yaml`；universe 排序后前
  `--max-codes`（默认 300）个码 + 数据截止前 `--last-sessions`（默认 260）个开放
  会话），只读装载 + 本地 cache 产物；cold/hot 的 `factor_tensor` 做 bitwise 等价
  断言（`np.array_equal(..., equal_nan=True)`），不等即基准失败退出非零。
* 测量只支持工程边界结论（AGENTS §4.1）；「热加载 ≤ 冷加载 30%」是方案的**待标定
  建议**目标——本契约预先固定测量协议与如实披露义务：实测比值、阶段构成、未达标
  时热路径剩余成本的解释全部入测量日志；禁止以减少数据窗、删除因子或跳过门禁冒充
  加速。

## 4 失败测试（RED → GREEN）与等价性验收

新测试文件 `tests/test_b04_load_cache_performance.py`（同 commit 注册进
`scripts/check_test_shards.py` shard-data，fail-closed 守卫生效）：

| # | 场景 | 断言要点 |
|---|---|---|
| R1 | 热加载不再重建昂贵帧 | 冷载（miss）后同身份热载（hit）：`build_pit_frames`/`build_capital_frames`/`compute_factor_tensor` 调用次数 = 0；`factor_tensor` 与冷载 bitwise 一致 |
| R2 | hit 路径全量产物等价 | hit 与冷载的 raw/adjusted tensor maps、target_ret、universe mask、industry_codes、price_degraded、dates/ts_codes 逐键 bitwise/逐项一致（NaN 以 equal_nan 比较） |
| R3 | invalidated 两态 | (a) 篡改条目文件字节 → outcome=invalidated、reason=invalidated:unreadable、张量重算且等于冷载；(b) 键位置放身份载荷错配的条目 → outcome=invalidated、reason=invalidated:identity_mismatch |
| R4 | 依赖变化不误命中 | miss 存储后 v2 追加 ingest 行 + 重新发布（dataset_id 变）→ 重载 outcome=miss、reason=miss:no_entry、张量重算等于冷载、旧条目保留不删除 |
| R5 | 身份查询前置 | 调用次序记录：身份推导先于 `build_pit_frames`（现状相反 → RED） |
| R6 | reason 词汇完备 | identity_inputs_missing / ingest_source_missing / ingest_source_empty / marker_tables_missing / identity_derivation_failed 五类 disabled 原因逐一触发并断言 `factor_cache_reason` |
| R7 | capital 下推 golden parity | 含窗口外 margin/basic/sw 行的 fixture，帧输出与手工推导期望逐值一致（行为不变优化的等价守卫——按 AGENTS §2.2-3 预注册：无法产生有意义 RED，以 golden/parity 守卫替代并在此说明） |
| R8 | disabled 路径不变 | 既有 `test_loader_cache_disabled_without_identity_sources` 保持绿（outcome=disabled、无 cache 目录、无 put） |

等价性验收（方案 B04「等价性验收」条目）：同一 fixture 与固定真实窗口，冷/热输出在
**bitwise（float32，equal_nan=True）** 一致——NaN 掩码、日期轴、股票顺序、因子顺序
一致；任何实际依赖（八元组任一元素）变化不得误命中（R4 + 既有身份维度测试）。

## 5 资源与运行上限

* 合成基准默认 120 码 × 300 会话（脚本参数可调，默认值冻结于此）；真实库窗口
  `--max-codes 300`、`--last-sessions 260`。16GB 主机串行执行，基准不并发。
* confirm 全量重哈希（两库）是每次装载的固定成本，热/冷同付；实测值入测量日志。
* 全量验证按 AGENTS §10.2 顺序：聚焦 → 相邻契约 → 全量 → compileall →
  `git diff --check`。

## 6 版本影响表

| 版本 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| `FACTOR_CACHE_VERSION` | 2 | 2（不变） | 身份八元组语义与条目格式不变；`last_outcome`/reason 是读取侧观测，不改变「什么能被复用」 |
| protocol/reward/grammar/feature registry/factor compute/data tier/search contract | 现值 | 不变 | 无研究语义变化：次序重排、观测字段与行为不变下推；hit/miss 重算产出的张量逐位一致由 R1/R2 锚定 |
| 新 artifact-facing 版本 | — | 无 | `factor_cache_reason` 是 loader 实例内存态观测字段，不入任何 artifact schema |

## 7 验收对照（方案 B04「完成时必须交付」）

* 冷/热运行分阶段耗时与等价性证据：I4 基准 JSON + R1/R2 + 真实窗口 bitwise 等价
  （测量日志记录实测）。
* 修改任何实际依赖都不能误命中：R4 + 既有八元组维度测试。
* 性能不以减少数据窗、删除因子或跳过门禁冒充加速：I1 保持全部读取/资格/确认步骤，
  I4 协议固定窗口与列。
* 「热 ≤ 30% 冷」建议目标：按 §5 协议实测并如实报告；未达标时披露热路径剩余构成。

## 8 停止条件

* 重排或下推引发任何语义疑点（输出与基线不一致、错误类型次序变化）→ 停止，先修订
  本契约再继续，不静默扩大范围。
* 真实库 confirm 失败（发布代际漂移）→ 如实记录为环境阻断；**不**发布新代际
  （A02 发布边界非本包授权），合成基准证据照常交付并在测量日志注明真实窗口证据缺失。
* 实施中发现 §3 与代码事实不符（如 industry 帧存在未知的 hit 路径消费者）→ 先修订
  契约，不绕过。

## 9 修订记录

（暂无）
