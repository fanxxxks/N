# P39 R02 第二轮：合法调仓周期独立比较（every_5_days/5 vs every_10_days/10）预注册契约

- 状态：**APPROVED**，用户 2026-09-19 会话直接指令「全部授权」（覆盖
  2026-09-19 工作包 4 交付报告所列「第二轮再独立比较合法周期」；第一轮
  p37 与数据解除 p38 已按同一授权先行交付）。
- 起草：执行模型（GLM Harness 会话），分支 `codex/b04-load-cache-performance`。
- 勘察基线：`codex/b04-load-cache-performance @ 1019362`（p38 实现提交；
  数据修复已执行、复合身份已重发布为 `d36024cd…`/publication 6）。
- 上位依据：用户工作包 4 指令原文「第二轮再独立比较合法周期，避免一次
  改变过多条件」；整改方案 §四 R02（首轮 12 组合中的「两组该域已有且非
  重叠的合法调仓周期」）；p37 契约（第一轮的冻结与裁决先例）；
  `ashare_domain` 权威（medium_cross_section 合法执行点 = every_5_days/5
  与 every_10_days/10，非重叠规则下合法视界限于 5–10——every_20_days/
  monthly 不属于该域，不纳入）。

## 1 问题陈述与目标

第一轮（p37）把周期冻结在 every_10_days/10，只回答订单门槛与持仓结构。
本轮独立回答：**在同一冻结组合配置下，合法周期之间的选择是否改变净空间
结论**。一次只变周期这一个轴（用户指令：避免一次改变过多条件）。

**目标**：对两个合法周期点各运行 p37 同款冻结臂面板（参考格组合配置），
产出与 p37 同构的四路裁决（周期面板级），与第一轮结论并列呈现。

**非目标**：不改组合/订单参数（见 §3 固定项——参考格选择**预先**冻结于
见任何第一轮结果之前）；不跨周期直接比较 reward/IC（不同 horizon 的
标签语义不同，§5 口径约束）；不合并两轮结论为单一「最优配置」；不做
晋级/holdout 声明；不修改生产默认。

## 2 与第一轮的关系

- 第一轮（p37）在 every_10_days/10 上回答持仓结构；本轮在同域另一合法点
  every_5_days/5 上重复同一测量，并把 every_10_days/10 作为本轮面板的
  内置对照臂（同轮、同数据、同代码——不与第一轮的运行拼接，AGENTS §4.3）。
- 两轮各自独立运行、独立 run 身份、独立裁决；本轮报告如实引用第一轮
  verdict 作对照，但不改写第一轮结论。

## 3 冻结面板（不可在见数后修改）

| 项 | 冻结值 | 来源 |
|---|---|---|
| 周期点 | `every_5_days`/horizon 5 与 `every_10_days`/horizon 10（该域全部合法点） | ashare_domain 权威 |
| 组合配置 | **参考格 `n20_m5000` 逐字**：top_n 20 / buy_rank 20 / sell_rank 30 / cap 0.05 / min_trade 5000 / threshold 0.01 | p37 §3.2 参考格（= r1_medium_a 原值）；选择先于本轮见数冻结 |
| 其余 | 资本 10 万、费率 0x/1x、股票池、folds 定义（train_end/test_end 原值）、lot 100 | p37 §3.1 |
| 候选 | p34 冻结臂（3 单因子 + 2 组合）+ 基准（每折每周期一次）+ 现金对照（参考格 1x，每周期一次，披露其周期无关性仅指空仓构造） | p37 §4 |
| 方向 | 每 **(周期, 折)** 由同一训练窗规则（rank IC 符号）经 `baseline_candidates` 在该周期 fold 上学习——horizon 改变标签，方向按周期独立学习是构造使然，如实披露；**同一周期内跨一切臂冻结复用** | p37 §4 方向规则的周期推广 |

## 4 报告必含

p37 §6 全部行级与汇总字段（含 lotgap/首建仓漏斗），按 (周期, 臂, 折, 档)
全量呈现；两周期并列对照块（同一臂在两周期下的净/毛超额、active IR、
换手、费用、漏斗与 lotgap）。禁止把不同周期的 IC/reward 直接排序比较
（标签视界不同）；并列呈现、各自命名。

## 5 裁决规则（冻结；与 p37 §7 同构，作用于周期轴）

- 臂在某周期「成立」：同 p37 判据（1x 跨折平均净超额>0 且平均
  `oos_active_ir`>0 且 invested ≥0.5 且 holdings ≥ ⌈top_n/2⌉=10 且
  失败折<2）。
- 周期「成立」⇔ 该周期存在 ≥1 成立臂。
- **相邻**：周期轴上两个合法点互为相邻（该域合法集仅此两点）。
- verdict（四路）：
  1. `stable_period_region` ⇔ 两个周期均成立。
  2. `isolated_period_winner` ⇔ 恰一个周期成立 → 记录为不稳定结果，
     不据此选择周期。
  3. 无成立 且 ∃(周期,臂) 0x 拼接超额>0 且 0x 拼接 active IR>0 →
     `gross_info_execution_blocked_period_panel` → 继续执行诊断。
  4. 无成立 且无毛信息 → `period_panel_no_reliable_info` → 暂停周期轴
     扩展。
- 行政走向：verdict 1/2 → run 停驻 `DATA_QUALIFIED`；3/4 → `RETIRED`。
  不产生晋级证据；两轮 verdict 均不构成默认周期变更依据。

## 6 预算（冻结）

墙钟 **4 小时**（2 周期 × 5 臂 × 5 折 × 2 档=100 次引擎折运行 + 方向路径
10 次 + 基准/对照；因子缓存若因新 dataset id 未命中则含一次冷装载
~48 min，p33 实测）；单进程；`consumed=0`。超限即 FAILED，禁止缩窗。

## 7 实现（最小改动）

- `ashare_model/r02_matrix.py`：`run_r02_panel` 增加 `cells` 参数
  （默认 `R02_GRID`，既有调用零变化）；新增 `run_r02_periods` 驱动
  （复用 `run_r02_panel`（cells=参考格）、`summarize_arms`、
  `assemble_cell_summaries`、`cell_variant`）与周期级裁决纯函数；
  每 (周期, 折) 的 folds 经 `resolve_folds_for` 按该周期 frequency/
  horizon 解析（同一权威）。
- `scripts/r02_periods.py` CLI shim；测试同 commit 注册分片。
- 版本影响：零 bump（PROTOCOL/EXECUTION/CONSTRUCTOR/RUNSPEC 不变；
  `R02_PERIODS_REPORT_SCHEMA_VERSION=1` 新测量产物，p37 报告同型先例）。

## 8 预期 RED 测试（实现前失败）

`tests/test_p39_r02_periods.py`：P1 合法周期集合冻结（恰两点，与
`ashare_domain` 权威一致）；P2 周期变体构造（frequency/horizon 恰两字段
变化）；P3 周期级四路裁决纯函数（含 isolated_period_winner 不稳定注记与
毛信息分支）；P4 端到端迷你链（真实裁决器；verdict→状态映射一致；两周期
行齐全、基准每折每周期一次、现金对照每周期一次）。

## 9 验证矩阵与停止条件

同 p37 §13（聚焦/相邻/compileall/分片/全量/正式运行入测量日志）。
停止条件：门禁拒绝 → REJECTED 诚实落账；契约与实现冲突 → 停下修订；
发现新阻断集合 → 如实报告，不扩权处置。
