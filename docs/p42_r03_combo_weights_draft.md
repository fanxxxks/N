# P42 工作包 6 裁决与 R03 组合权重面板预注册契约

- 状态：**APPROVED**，用户 2026-09-19 直接指令「完成工作包 6：依据结果选择
  R03 或 R04」（附冻结决策表；p30–p41 先例；批准记录回写于本文件头）。
- 起草：执行模型，分支 `codex/b04-load-cache-performance`。
- 勘察基线：`27d3dc6`（p41 收口后），工作树干净（除未跟踪 `.mimosa/`）。
- 依据文档：`D:\minequant\reports\AlphaGPT_整改与提升执行方案_2026-09-09.md`
  §R03/§五（第三阶段「先冻结 R05 规则并实施 R01/R02，再由结果决定是否需要
  R03/R04」——R05 已由 p40/p41 落地，R01/R02 由 p34/p37/p39 落地）。

## 1 分支裁决（对照用户冻结决策表，逐行）

已有证据（全部来自已收口运行的报告 artifact，非重算）：

- **p37 R02 第一轮**（dataset `d36024cd`，run `d4ff2b3e`，verdict
  `stable_region_found`，报告 `data/r1/r02_matrix/r02_matrix_report.json`）：
  - n10 稳定区（m1000/m2500/m5000 三格全成立），**唯一成立臂 =
    `combo:all3`**（1x 跨折平均净超额 +2.24%~+2.71%，`oos_active_ir`
    +0.083~+0.111，投入 0.68–0.69，持仓 8.0–8.6）；
  - **三个单因子臂全部净负**（MOMENTUM_20 −1.2%~−1.3%、RSQ_60
    −2.6%~−3.0%、REVERSAL_60 −7.2%~−7.6%，1/5~2/5 折为正）；
    `combo:momentum_reversal` ≈ 0（+0.06%~+0.10%）；
  - n20 行 0x **毛**超额 −5.1%~−43.7%（毛信号本身为负，非费用吞噬）；
    n10 毛超额 +9%~+12%（费用大幅折减但未吞噬）；
  - 单年依赖披露：combo:all3 仅 2023（+14.8%）、2024（+14.6%）折为正。
- **p39 R02 第二轮**：参考格 n20_m5000 上两合法周期均
  `period_panel_no_reliable_info` → RETIRED（周期轴非瓶颈亦非拯救；
  n10×周期交互未预注册未测）。
- R02 面板行**不携带 val_reward**（固定信号直接经引擎评估，未经 scorer）。

| 决策表行 | 证据匹配 | 判定 |
|---|---|---|
| 1 简单信号有净空间，组合增量不清楚 → **R03** | 前半句被证据**反转**（单因子全部净负），后半句被 p37 **部分回答**（跨族等权组合增量明确为正——组合是唯一净正臂）。R03 的第一形态（跨信息族等权 rank 组合）已测且成立；第二形态（**训练窗拟合的收缩线性组合**）未测——这正是开放问题 | **选中（动作）** |
| 2 毛表现尚可，费用/成交吞噬 → 继续 R02 | n10 毛 +9%~+12%→净 +2.2%~+2.7%（未吞噬）；n20 毛本身为负（不是费用问题） | 不匹配 |
| 3 reward 排名与净表现明显错位 → R04 | R01/R02 均为固定臂，面板无 val_reward，**无任何排序证据**（无证据 ≠ 错位被否定，但「明显错位」不成立） | 不匹配 |
| 4 medium 域没有空间 → slow 域基线 | n10 区有净空间（combo +2.2%~+2.7%） | 不匹配 |
| 5 简单基线稳定 → 小预算搜索对照 | 基线稳定性存疑（2/5 折为正、2023–24 集中）；当前开放问题是组合结构而非搜索算法；顺序上 R03 先行 | 不匹配 |
| 6 多轮无增量 → 暂停扩算法 | 有正增量证据（组合臂） | 不匹配 |

**裁决：执行 R03**（整改方案 §R03 的第二形态比较）。R04 不触发（无排序
错位证据）；若 R03 之后需要奖励对齐实验，另立指令。

## 2 问题陈述

跨族等权组合是 n10 稳定区唯一净正臂，但等权是**未学习的选择**：三成员的
训练窗证据（验证窗 ICIR）并不对等。开放问题：**训练窗内拟合的受约束/
收缩权重组合能否在整条 n10 稳定区上稳定超过等权组合**（增量 ≥ 预注册
边际，全格成立），以及 p37 的等权净空间在新运行身份下是否复现
（同期同数据集的期内复测）。

## 3 冻结实验设计

**成员与方向**（p34/p37 逐字沿用）：MOMENTUM_20 / RSQ_60 / REVERSAL_60；
方向 = 每折训练窗 masked rank IC 符号（`baseline_candidates` 既有语义，
参考格 `n10_m2500` 上学习一次跨格冻结复用——p37 §4 同款规则，参考格取
稳定区中位格，先于任何 R03 结果选择）。

**组合基座**（p34 §3.2 逐字沿用 + 唯一扩展）：成员信号按方向翻转后逐日
截面 rank 归一（`reward._rankdata`，仅 universe 可格，NaN 不参与），
**加权平均**取代等权平均；单日缺成员按可得成员权重重归一，全缺失置 NaN。
等权重量的加权平均 ≡ 既有 `combo_signal`（RED 以精确等式钉住）。

**拟合权重规则（冻结，超参数不做任何搜索）**：每折，成员得分
s_i = 该折 `baseline_candidates` 返回的 `val_icir`（训练窗内验证窗
统计量——与方向学习同一次调用、同一权威，零新增拟合机制、零 OOS 访问）。
权重 w_i = (1−λ)/K + λ·max(s_i, 0)/Σ_j max(s_j, 0)；若 Σ max(s_j,0) = 0
则回退等权（逐折记录回退事件）。λ = 0.5 为主臂（裁决对象）；
λ ∈ {0.25, 0.75} 为描述性臂（**不参与裁决**，仅披露敏感性）。

**面板**：格 = n10_m1000 / n10_m2500 / n10_m5000（p37 稳定区，coupled
参数与 `cell_variant` 权威逐字复用）；周期 = every_10_days/horizon 10
（p37 冻结；p39 已证周期轴在参考格非瓶颈，本包不重开）；成本档 0x/1x
（0x 供毛信息判定，裁决在 1x）。臂 = benchmark:equal_weight（每折一次）、
combo:all3（等权，期内复测参照）、combo:fitted_lam050（主臂）、
combo:fitted_lam025 / combo:fitted_lam075（描述性）；三单因子仅经参考格
方向路径实测一次（1x，参考格行照常入报告，不参与裁决——p37 同款披露）。
现金控制不设（活动门由 invested/holdings 承载）。

**裁决规则（纯函数，逐字复用 p37 §7 的臂级规则与阈值）**：
`R02_ADJUDICATION_RULES`（min_invested_ratio 0.5 / min_positive_folds 3 /
max_failed_folds 1）、`_arm_verdict`（net_space = 净超额>0 ∧ IR>0；
unreliable/low_activity/thin_book/min_positive 旗标同款；holdings 下限 =
格 top_n 的一半，向下取整）、`_has_gross_info`（0x 拼接超额>0 ∧ 0x 拼接
active IR>0）——同一实现直接调用，不建第二套。

**四路裁决（冻结）**：
1. `fitted_increment_region`：主臂在三格全部 viable **且** 每格
   mean_net_excess_1x(主臂) ≥ mean_net_excess_1x(combo:all3) + **0.005**
   （50bp，预注册边际，先于任何结果固定）；
2. `equal_weight_stands`：非 1 且 combo:all3 三格全部 viable（p37 净空间
   期内复现，等权仍是最好组合）；
3. `combo_no_net_space`：combo:all3 在 ≥2 格不 viable（p37 净空间未复现）；
4. `panel_no_reliable_info`：其余混合情形。
`execution_blocked`：面板执行前的数据/身份/门禁拒绝（R03Refusal，
p37 §2 同款前置层，不产生四路 verdict）。

**披露（不否决）**：逐臂年度分解与逐折正负；逐折拟合权重、val_icir 输入
与回退事件；`single_year_dependence` 旗标（p37 同款语义：正折 confined
于 ≤2 个日历年）。

**生命周期与行政走向**：IDEA→SPEC_LOCKED→DATA_QUALIFIED 全真实证据链
（`run_r02_matrix` 同款走查：契约 sha256 自校验 → 身份确认 → 冻结
RunSpec（purpose=historical_research）→ 正式门禁 → 数据资格报告）；
verdict 1/2 驻留 **DATA_QUALIFIED**（非终态）；3/4 → **RETIRED**。
墙钟预算 4h（面板 ≈140 次评估 ≈ p37 规模之半）；预算熔断 = R03Refusal
诚实终态。

## 4 版本影响表

| 版本 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| 全部研究语义版本 | PROTOCOL 29 / reward 15 / grammar 9 / EXECUTION 4 / 构造器现值 | **不变** | 固定候选 + 既有引擎/评估路径零改动；新增组合形态是研究臂不是语义变更（p37/p39 先例：驱动 + 纯裁决器零 bump） |
| 报告 schema | — | `R03_REPORT_SCHEMA_VERSION = 1`（模块内新常量，additive） | 新报告产物自有版本 |
| 旧产物 | — | 保留 | p37/p39 报告原样不动，本包报告独立落 `data/r1/r03_combos/` |

## 5 预期 RED（独立 oracle——禁止从实现反推）

`tests/test_p42_r03_combos.py`（同 commit 注册分片）：

- **N1** 加权组合基座：等权重量的加权平均与 `combo_signal` **逐位相等**；
  非等权为成员秩的加权平均（手算小例）；缺成员日按可得成员权重重归一；
  权重非负且和一（断言）。
- **N2** 拟合规则：给定 val_icir 输入的手算权重（含 λ 混合式、负得分截零、
  全负回退等权 + 回退旗标）；输出记录权重与输入。
- **N3** 裁决器：四路 verdict 各由手造 summaries 触发（1：三格增量 ≥50bp
  且 viable；2：viable 无边际；3：等权两格死；4：混合）；断言复用
  `R02_ADJUDICATION_RULES` 同一常量；描述性 λ 臂不进裁决输入。
- **N4** 面板驱动（迷你端到端）：行携带 `<cell>::<arm>` 复合键；方向与
  权重每折在参考格学习一次、跨格冻结（断言同折跨格权重恒等）；单因子仅
  参考格 1x 落行；描述性臂在报告中标注 `adjudicated=False`。
- **N5** 驱动拒绝路径：契约 sha 不匹配 → R03Refusal（p37 C5 同款）。

## 6 验证矩阵

1. 聚焦：`python -m pytest -q tests/test_p42_r03_combos.py`
2. 相邻：p37/p39/p34 回归（`test_p37_r02_matrix.py`、
   `test_p39_r02_periods.py`、`test_p34_r01_baseline.py`、
   `test_p34_r01_report_wp3.py`）+ r01/baseline 批次。
3. `python -m compileall -j 0 -q …`、`python scripts/check_test_shards.py
   --check`、`git diff --check`、全量 `python -m pytest -q tests -n auto`
   （里程碑一次）。
4. 正式运行：`scripts/r03_combos.py --contract-sha256 <sha> --out-root
   data/r1/r03_combos`；测量日志 evidence-only 后置 commit。

## 7 停止条件

- 门禁/身份拒绝 → 诚实终态（REJECTED/FAILED），不重跑、不放宽。
- 发现拟合权重规则存在 OOS 泄漏路径（val_icir 非 training-only）→ 停止
  回契约，不夹带修复。
- 全量回归出现无法归因的新红/新 warning → 停止诊断。
- 裁决后不得因结果调整 λ、边际或格集合；敏感性披露不改变 verdict。

## 8 授权边界登记（用户尾注，非本包执行范围）

- 未来数据观察（C03）在候选与规则冻结后开展——本包 verdict 1/2 仅驻留
  DATA_QUALIFIED，**不解锁**任何 paper/观察。
- 界面（C01）优先修复同一次运行的结果关联；缓存/同步/页面扩展（B04 等）
  按实际阻塞程度安排——各自另立包，本包零 Web/缓存改动。
