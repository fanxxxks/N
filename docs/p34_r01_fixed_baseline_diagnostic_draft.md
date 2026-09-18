# P34 R01 固定基线与分域诊断（第一轮：medium Tier-A 基线诊断）预注册契约

- 状态：**APPROVED**，用户 2026-09-16 会话直接指令「阅读AlphaGPT_整改与提升执行方案_2026-09-09.md，完成R01」（p30/p31/p32/p33 先例：用户直接指令批准 + 执行模型起草；批准记录回写于本文件头）。
- 起草：执行模型（DeepSeek Harness 会话），分支 `codex/b04-load-cache-performance`。
- 勘察基线：`codex/b04-load-cache-performance @ faad2f4`（含 A01→B04 全链；A01/A02 已并入 main，A03–A05、B01–B04 为已双审叠层分支）。`git status` 干净。
- 上位依据：`reports/AlphaGPT_整改与提升执行方案_2026-09-09.md` §四 R01（新一轮固定基线与分域诊断）与 §五 第三阶段；AGENTS §3.2/§4.1/§4.4；p8 生命周期契约（唯一状态机权威）；p27 动作模式表；p21 R1 域研究预注册（fold/baseline_signals/`tier_set` 语义来源）。
- 契约 hash（本文件正文 sha256，实现 commit 与 RunSpec IdeaRecord 引用）：见 `docs/p34_r01_baseline_measurement_log.md` 的证据节（后置测量引用）。

## 1 问题陈述与目标

整改方案 R01 要求在搜索之前先完成**固定基线**并给出「是否值得搜索」的裁决。
本契约冻结第一轮（round 1）的全部研究计划：仅 medium 域 Tier-A 配置的小规模
基线诊断；基线先于任何搜索完成并入账；不同时铺开大规模搜索。

**目标**：

1. 在真实数据身份（A02 复合确认）与正式生命周期身份（p8 状态机）下，对
   `configs/r1_medium_a.yaml`（sha256 `3DC7F6CAA2285205856DD476144D753BFFE75D339B95D9D126A7AC7C88924D4D`）
   的 5 个预注册 fold（测试年 2021–2025）运行固定基线臂。
2. 产出可复核基线报告（0x/1x/2x 三档成本、年度分解、活动与费用漏斗、全部臂
   而非最高一行）。
3. 按本契约 §7 冻结规则作出「是否值得搜索」的裁决并落账。

**非目标**：

- 不运行任何搜索臂（GP/TPE/Random/RL/训练候选）；不消费唯一语义评价预算。
- 不运行 slow AB / short A / unified 对照域（待 medium 轮裁决与数据条件满足后，
  另立指令或修订本契约）。
- 不修改生产默认配置（`config/ashare_config.yaml`）、不改默认策略语义。
- 不做晋级、paper、受保护 OOS 声明。2021–2025 为反复研究过的开发验证窗口，
  本轮一切结论都不得称为 holdout/已验证 alpha（AGENTS §4.4）。
- 不做市场状态（牛/熊）分类分解：仓库现无该分类权威，新造分类器属 R05 范围；
  本轮以**年度分解**（5 fold）+ 月度稳定性描述代替，并在报告中声明该缺口。

## 2 数据与身份

1. 数据身份：`AshareDataLoader` confirm 模式（A02 `confirm_composite_snapshot`）
   产生复合 id；RunSpec 冻结用 `frozen_data_cutoff()`；装载后执行漂移守卫
   （`dataset_id` 不匹配即拒绝，A02 I2/p27 I6）。p33 实测同一数据身份
   `b369c49b…` 于 2026-09-16 确认复合通过；本轮运行前重新实时确认，不沿用旧值。
2. G1–G8：`ProductionGateRunner(data_config).run(mode="formal")` 全量执行；
   结果作为 §5.1 DataQualificationReport 的唯一门禁来源（零复制）。
3. **loader 帧降级类型化（p8 §5.1 分期缺口闭合）**：t16 基线时数据层无类型化
   降级源，`build_data_qualification_report` 因此恒发
   `loader_frames="untyped_in_current_data_layer"` 并被验证器拒绝（真实验证
   永远 fail-closed）。p19/p17/D5 落地后数据层已有权威降级源
   （`price_degraded`、`industry_source_degraded`、universe contract status、
   dataset binding 状态、dev fallback 开关）。本包为报告构建器接入该类型化
   状态：仅当全部为 clean 时发 `"clean"`，否则发结构化降级原因并被拒绝转换。
   验证器的「无未披露降级」裁决规则**不变**；p8 §5.1 本就要求类型化字段。
4. 生命周期：RunStore 根 = `data_config.data_dir`（r1_medium_a 即
   `data/r1/medium_a`，与 evaluation 入口同一解析，无第二套根）。状态链：
   `IDEA → SPEC_LOCKED → DATA_QUALIFIED →（§5.2 基线证据生产）→
   FACTOR_SET_QUALIFIED → 裁决落账`。本轮**不进入** `SEARCH_PLAN_ADMITTED`
   （无搜索计划准入）；裁决为正时 run 停留于 `FACTOR_SET_QUALIFIED`（非终态，
   等待后续显式指令准入搜索），裁决为负或被阻断时按 §7 进入 `RETIRED`/
   `FAILED`。所有状态转换走 `LifecycleWriter` 单一守卫路径，无第二套写法。

## 3 冻结的研究计划（不可在见数后修改）

### 3.1 域、窗口与股票池

| 项 | 冻结值 | 来源 |
|---|---|---|
| 研究域 | `medium_cross_section`，`tier_set: A`（17 特征） | r1_medium_a.yaml + p21 §8 |
| 频率/视界 | `every_10_days` / horizon 10 | r1_medium_a.yaml |
| folds | 5 个：train_end 2020-12-31…2024-12-31，test_end 2021-12-31…2025-12-31 | r1_medium_a.yaml（p21 预注册） |
| 股票池 | calendar/constituents 主库 + v2 PIT store，`min_listed_sessions: 60` | r1_medium_a.yaml |
| 资金 | `initial_capital: 100000.0` | r1_medium_a.yaml |
| 调仓与组合 | top_n 20，equal_weight，buy_rank 20，sell_rank 30，min_trade 5000，target_weight_change_threshold 0.01，single_weight_cap 0.05 | r1_medium_a.yaml |
| seeds | RunSpec 记录 (42, 7, 2024)（p21 预注册，供未来搜索轮）；本轮无随机臂，**seed 对基线结果无作用**，报告如实披露 | r1_medium_a.yaml |

### 3.2 基线臂（全部经 `evaluate_signal`/同一引擎路径）

| 臂 | 构造（冻结） | 方向规则（冻结） |
|---|---|---|
| `benchmark:equal_weight` | 既有 `benchmark_row`（同池等权、零成本参考，不计 trial） | — |
| `baseline:MOMENTUM_20` / `baseline:RSQ_60` / `baseline:REVERSAL_60` | 既有 `baseline_candidates`（裸因子直接作为信号） | 训练窗 masked rank IC 符号（既有协议语义，仅用训练窗，无 OOS 访问） |
| `combo:all3` | 三臂成员逐日截面 rank 归一（`reward._rankdata`，仅 universe 可格；NaN 不参与），方向同上，等权平均；单日某成员缺失时取可得成员均值，全缺失置 NaN | 成员方向 = 上述同一训练窗规则 |
| `combo:momentum_reversal` | 同上，成员 = MOMENTUM_20 + REVERSAL_60（跨动量/反转子族的两成员组合） | 同上 |
| `control:cash` | 全 NaN 信号（引擎无可选持仓 → 全程空仓） | — |

`combo:all3` 与 `combo:momentum_reversal` 即「少量方向已冻结的简单组合」：
冻结的是**构造规则与方向来源**（训练窗规则 + p21 预注册因子集），不是看数后
挑选的方向；其数值型方向在执行时产生并连同证据 hash 入账。

### 3.3 成本档（每臂 × 每折 × 三档）

| 档 | 定义（冻结） |
|---|---|
| 0x（毛） | `commission_rate=min_commission=stamp_tax_rate=transfer_fee_rate=slippage_rate=0` |
| 1x（净） | r1_medium_a.yaml 原值 |
| 2x（压力） | `commission_rate/stamp_tax_rate/transfer_fee_rate/slippage_rate` ×2；`min_commission` 不变（保持费率下限语义） |

`benchmark_row` 为零成本参考，不参与成本分档；现金控制臂三档成本均为 0
（这正是它要暴露的「低成本假象」对照）。

### 3.4 报告必含（每臂 × 每折 × 每档）

净/毛超额（对 `benchmark:equal_weight`）、active IR（`oos_active_ir`）、
回撤、Sharpe、有效 IC 日数（`n_ic_dates`）、IC/ICIR（universe 与 tradable 两
口径）、调仓应发生数（`rebalance_due_count`）、实际调仓数（`rebalance_count`）、
订单数（`order_count`）、抑制数（`suppressed_trade_count`）、平均换手、持仓数
（均值/最大）、投入比例均值（现金比例 = 1 − 投入比例）、实现成本占比合计
（`cost_fraction_sum`）、费用结构（配置参数原文）。汇总层给出逐 fold 年度分解
（2021–2025）与跨折中位数/最小/最大。**全部臂全部折全部档入报告，不裁剪、
不突出最高一行。**JSON 序列化 NaN→null（json-safe 断言进测试）。

### 3.5 统计口径（R05 全量机制前置的冻结子集）

- 第一轮为**描述性诊断**：不做显著性检验、不做 DSR/max-t（R05 范围）、不做
  多重比较校正声明。
- 三 seed 仅在未来搜索轮作初筛；本轮无随机成分，报告披露「seed 不适用」。
- 失败/缺折规则：任何臂在任何折上引擎失败或整折 NaN，该 (臂,折) 记
  `failed` 及原因入报告，不虚构零收益补位；≥2 折失败的臂在其汇总中标
  `unreliable`。
- 试错台账：本轮所有臂行写入 R01 报告 artifact（run store 登记，hash 链），
  且测量日志记录运行命令、commit、dataset id、配置 hash、墙钟、内存。

### 3.6 预算（冻结）

- 墙钟上限：**6 小时**（依据：p33 实测真实库全历史冷载 2892s/热载 332s +
  confirm 216s，本轮 warm 因子缓存命中预期 ≤10 min；90 次引擎折运行按
  B02 canary 与 p33 阶段基准外推 ≤1 h；上限含全部重试一次的机会）。超限即
  停止并记 `FAILED`（运行故障），不得缩窗冒充完成。
- 唯一语义评价预算：RunSpec `requested_budget=512`（p21 screening tier
  2×256，为未来搜索轮预注册）；**本轮基线不消费任何唯一语义评价**，报告
  `consumed=0`。
- 内存：单进程运行（16 GB 主机，p33 实测全历史装载峰值 ~7.9 GB）；禁止与本
  运行并发其他重载进程。

### 3.7 裁决规则（冻结；执行模型不得用 OOS 结果反改）

对每个成本承载臂（3 单因子 + 2 组合；现金控制与 benchmark 不参与裁决）：

- **活动健全性**：`rebalance_due_count>0` 而 `rebalance_count==0` 的
  (臂,折) 记「冻结账本」；整轮 `invested_ratio_mean < 0.5` 的臂记
  「低活动」，其低成本优势判定为构造假象，不进入价值判定。
- **净空间**：臂「显示净空间」⇔ 1x 档跨折平均净超额 > 0 且 1x active IR > 0
  且平均持仓数 ≥ 5。
- **单年依赖**：显示净空间的臂若 1x 净超额 > 0 的 fold 数 < 3/5，标
  「单年依赖」。
- **成本脆弱**：显示净空间的臂若 2x 净超额 ≤ 0，标「成本脆弱」。
- **worth_search（继续信号）**：⇔ 存在至少一个显示净空间、非低活动、
  非单年依赖的臂。
- 裁决输出：
  1. `worth_search=True` → 结论「简单信号显示成本后净空间，值得进入小规模
     搜索轮」；run 停留 `FACTOR_SET_QUALIFIED`；搜索轮须另立显式指令/修订，
     本契约不授权搜索。
  2. `worth_search=False` 且证据完整 → 结论「简单基线未证明成本后净空间，
     暂停算法扩张」，附瓶颈分解（数据资格/覆盖/成本/活动/单年依赖）；
     run → `RETIRED`（原因引用报告 hash）。**负结果是合法终态。**
  3. 证据不完整（门禁拒绝/装载失败/超时）→ 对应 `REJECTED`/`FAILED` 语义，
     不产生研究裁决。

## 4 变更分类与版本影响表

**Rev 1（实现期修订，先于实现 commit）**：§5.2 因子资格报告构建器的
check-4 存在潜伏缺陷——`in_domain_execution` 用
`int(spec.horizon) in domain.horizon_range` 判成员，而 `horizon_range`
是闭区间 `(min, max)`；p8 §5.2 check 4 点名的权威是
`ResearchDomain.is_legal_execution`（区间语义 + P3 非重叠标签校验）。
此前所有消费者均为 `unified` 域（跳过该检查），缺陷未暴露；R01 是第一个
真实命中者（mini canary RED：`VOL_20` 在 horizon 10 ∈ [5,20] 被拒）。
本包将该检查改为直接调用权威 `is_legal_execution`，属缺陷修复
（裁决语义与契约原文一致），连同本节修订一并进入实现 commit。

**Rev 2（双审修复轮，先于修复 commit）**：双审（合规 PASS；技术
NEEDS_REVISION）裁定的修复全部在本契约冻结语义内，无研究语义变化：

- **T1** `loader_frame_status` 降级契约分支改用 `UniverseContractStatus.warnings`
  （原引用的 `reason_codes` 字段不存在——恰在必须披露降级时崩溃；R1b RED）。
- **T2** run 打开后的一切失败路径收敛到诚实终态：预算超限、漂移、运行时异常
  一律 `writer.fail`（FAILED），由幂等的 `_mark_failed` 守卫防双转移；兑现
  §3.6「超限即停止并记 FAILED」。
- **T4** 机器判别器归位：**已完成裁决但验证失败**的 §5.1/§5.2 门禁 →
  `writer.reject`（REJECTED，携带已登记证据 artifact）；**未完成**的评估 →
  FAILED。REJECTED 边已全图激活（lifecycle.py ACTIVATED_EDGES = 全 §4 图）。
- **T3** 引擎调用按 (臂,折,档) 捕获失败并记 `failed` 行 + 原因（§3.5 原文
  要求的逐 (臂,折) 失败记账；`summarize_arms` 消费）。
- **T5** `unreliable` = ≥2 失败折（§3.5 原文）；单折失败记 `fold_failure`
  信息标记，不取消资格（修复代码偏离冻结规则之处）。
- **T6** 报告字段更名 `net_space_ex_holdings`（§3.7 的持仓下限在 `thin_book`
  旗标中执行，字段名不再夸大语义）。
- **T7** benchmark 臂 `cost_level="ref"`（§3.3：零成本参考不参与阶梯）。
- **T8** e2e 经显式 `adjudication_rules` 测试缝驱动**真实**裁决器
  （canary 规模放宽阈值），不再 mock 裁决函数；负路径用冻结规则的真实负结果。
- **T9** 消融子集为空（单族因子集）→ fail-closed 拒绝，不对空列表出证据哈希。
- **C5**（合规审建议采纳）驱动对 `--contract-sha256` 与已提交契约文件自校验，
  不匹配先于昂贵计算拒绝——provenance 从自证变为自验。

**Rev 3（WP3 报告补齐，先于实现 commit）**：用户 2026-09-18 直接指令「完成
工作包 3：补齐 R01 报告，并按修正后的执行版本重跑固定基线。范围：
r01_baseline.py、eval_metrics.py、报告与相关测试」批准。**全部为报告与
可观测性扩展，研究语义零变化**：

- §3.2 臂、方向规则、§3.3 成本档、§3.7 裁决规则与阈值**逐字冻结**；
  `summarize_arms` 既有字段（裁决消费的 fold 均值统计）保持原名原义。
  组合构造保持**原始截面秩**（`reward._rankdata`）：不同成员覆盖数产生
  不同秩尺度是已披露的既有形态；百分位标准化属新定义，另立包独立验证，
  本包禁止实施。
- 报告必含缺口补齐（§3.4 原文已要求、实现缺失的部分）：逐 fold 年度分解
  与跨折中位数/最小/最大；**跨折拼接指标**——每 (臂,档) 按时间序拼接
  daily 序列后计算 total/annual/Sharpe/maxDD/超额/**active IR**（拼接
  IR 不得用平均年度 IR 代替；两种统计并列呈现、分别命名，§3.7 裁决继续
  消费冻结的 fold 均值统计）。拼接复用 `eval_metrics.stitch_oos_series`
  唯一权威（按档过滤行集，无第二套拼接实现）。
- 行级附加可观测字段（`evaluate_signal` additive，既有字段逐位不变）：
  `oos_active_ir`（测试窗 active IR，区别于 scorer 的训练窗 `active_ir`）
  与执行漏斗计数——应调仓中信号退化/目标稳定/全额抑制/部分抑制会话数、
  计划订单数 vs 引擎成交订单数（含买卖拆分）、计划 vs 成交换手、
  抑制原因拆分（threshold vs min_trade）、持仓状态（最小持仓、零持仓
  会话数、最小投入比例）。目的：**正常低换手**（目标≈当前，无需订单）
  与**订单被抑制**（需要交易但被阈值/最小交易额/现金上限/涨跌停/停牌
  挡下）可区分。数据源为既有构造器诊断与引擎结果，零引擎改动。
- 报告级新增块：`data_completeness`（loader 帧类型化状态、缺失交易日、
  完整/不完整状态与原因）；每臂**描述性四向诊断**
  （`measurement_incomplete` / `weak_signal`（0x 拼接超额≤0 或 0x 拼接
  active IR≤0）/ `execution_blocked`（漏斗证据：全额抑制占比>10% 或
  订单未成交率>10% 或 有毛超额而投入比例<0.5）/ `cost_swallowed`
  （0x 为正而 1x 拼接超额≤0）），阈值冻结于报告 `diagnostic_rules` 内
  自描述；该分类是**描述性归因**，不是 §3.7 裁决，二者并列不互替。
- `engine_version_attribution` 块：记录 EXECUTION_SPEC_VERSION（=4，
  p35 股数基记账）与构造器版本；如实记录「不存在旧执行版本基线数字」
  （WP0 收口 §6.3：p35 前引擎数字不可作参数选择依据且从未产出），因此
  无旧版-新版数值归因可计算——记录该事实而非伪造对比。
- `R01_REPORT_SCHEMA_VERSION` 1→2（测量产物内部版本，非 artifact_schemas
  消费者；v1 报告从未产出，无迁移面）。基准行失败时该折标记
  `benchmark_missing` 并计入 measurement_incomplete。
- 重跑：以本 Rev 3 契约实时 sha 印签，按当前数据身份与 exec v4 重跑
  固定基线；若 §5.1 数据资格仍因类型化降级 fail-closed 拒绝，按契约
  §7 落 REJECTED 诚实终态——「测量不完整」即四向诊断之一的真实落点。

报告 schema v1 的字段修正发生在无任何真实消费者阶段（与 §4 数据报告同理由）。
版本影响表其余行不变。

本包工程改动（不改任何既有研究语义默认值）：

1. `ashare_model/data_loader.py`：新增 `loader_frame_status()` 类型化降级状态
   输出（权威降级源已在，本包只做单一出口）。
2. `ashare_model/lifecycle.py`：`build_data_qualification_report` 接入类型化
   loader 帧状态（§2.3）；`_validate_data_report` 校验结构化 clean/降级原因。
   payload 的 `degraded_sources.loader_frames` 由恒失败字符串升级为类型化
   对象——v1 报告从未有真实消费者（真实验证恒拒），`report_schema_version`
   保持 1 并在实现 commit 说明。
3. `ashare_model/eval_metrics.py`：`evaluate_signal` 返回值**附加**可观测字段
   （P10 活动门同型 additive：`rebalance_due_count`、`holdings_mean`、
   `holdings_max`、`invested_ratio_mean`、`cost_fraction_sum`）；既有字段
   数值零变化。
4. 新增 `ashare_model/r01_baseline.py`（本轮唯一驱动）+ `scripts/r01_baseline.py`
   CLI shim + 测试。

| 版本 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| `PROTOCOL_VERSION` | 27 | 不变 | 不运行评价协议；`evaluate_signal` 只加附加字段，既有字段语义零变化 |
| `REWARD_VERSION` | 15 | 不变 | 评分器零改动 |
| `RUNSPEC_SCHEMA_VERSION` | 2 | 不变 | 无新 spec 字段 |
| `LIFECYCLE_CONTRACT_VERSION` | 2 | 不变 | 状态集/边/裁决规则零变化；§5.1 类型化字段是契约原文要求的补全，验证器裁决规则不变 |
| 数据报告 `report_schema_version` | 1 | 1（payload 字段升级） | v1 从无真实通过者；实现 commit 举证 |
| R01 报告 artifact | — | `r01_report_schema_version=1`（新测量产物，非 artifact_schemas 消费者） | 属测量目录产物（tier_report/p3_measurement 同型）；不进入晋级证据链，兼容性权威不适用，特此说明。**Rev 3（WP3）：1→2**——报告补齐（拼接指标/年度分布/漏斗/状态/四向诊断/版本归因）；v1 从未产出，无迁移面 |

## 5 预期 RED 测试（实现前失败）

`tests/test_p34_r01_baseline.py`（同 commit 注册分片）：

- R1 `loader_frame_status`：clean 场景返回 clean；注入 `price_degraded` 后返回
  degraded + 原因（测方法逻辑，不 mock 方法本身）。
- R2 类型化数据报告：clean 通过 `_validate_data_report`；degraded 被拒且带原因。
- R3 `evaluate_signal` 附加字段：真实迷你引擎运行断言字段值与引擎诊断一致；
  现金控制臂持仓 0、投入比例 0、成本 0、收益 0。
- R4 组合构造：rank 归一、方向、缺失成员规则逐项断言（手算期望）。
- R5 json-safe：R01 行含 NaN 时序列化为 null。
- R6 成本档：0x 成本 0；2x 成本 > 1x 成本（同信号）。
- R7 裁决规则：worth_search 三条路径 + 低活动/单年依赖/成本脆弱标记（纯函数）。
- R8 端到端（迷你双库）：完整生命周期链（genesis→lock→data→factor）真实报告
  通过、报告 artifact 落 run store、负裁决路径产生 RETIRED 事件。

## 6 验证矩阵

1. `python -m pytest -q tests/test_p34_r01_baseline.py`
2. 相邻契约：`tests/test_p27_a03_lifecycle_runcontext.py tests/test_p31_b02_entry_canary.py tests/test_run_store.py tests/test_evaluation.py tests/test_data_loader.py tests/test_baseline_harness.py`
3. `python -m compileall -j 0 -q ashare_data ashare_model ashare_portfolio ashare_trading scripts webapi`
4. `python scripts/check_test_shards.py --check`
5. `git diff --check`
6. 全量：`python -m pytest -q tests -n auto`（既有 parallel-vs-serial 对账基线沿用；结果入测量日志）
7. 正式运行后：测量日志（命令/commit/dataset id/配置 hash/墙钟/内存/原始产物路径/sha256）。

## 7 停止条件

- 门禁拒绝（G1–G8 任一 fail、降级未披露、复合身份失效）→ 报告阻断并按
  `REJECTED`/`FAILED` 落账，不绕行（AGENTS §4.3 fail closed）。
- 装载/引擎异常或 6 h 墙钟超限 → `FAILED`，保留失败证据，禁止缩窗重跑冒充。
- 契约与实现冲突 → 停下修订契约，不得改实现凑数。
- 发现 2021–2025 之外的新窗口需求、其他域对照需求或搜索需求 → 另立指令，
  不扩权。

## 8 与生命周期动作模式的对应（p27 §3）

- 本轮**没有**正式 `train`/`evaluation`/`backtest`/`paper_sim` 入口调用；
  基线臂是 p8 §5.2 check 6 要求的资格证据生产，发生在 `DATA_QUALIFIED` 状态、
  由 `LifecycleWriter` 单一守卫路径记录，产物全部登记进 run store。
- 裁决不改变任何门禁裁决权：`qualify_data`/`qualify_factor_set` 的机器裁决
  仍由既有验证器独立作出；本包的裁决规则只决定 run 在
  `FACTOR_SET_QUALIFIED` 之后的**行政走向**（驻留/退役），不产生晋级证据。
