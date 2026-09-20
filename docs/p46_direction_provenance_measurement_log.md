# P46 方向学习与组合定义明确化 — 测量日志（工作包 10）

- 契约：`docs/p46_direction_provenance_draft.md`（APPROVED，用户
  2026-09-20 直接指令工作包 10）。
- 被测实现：本包四提交（契约 `f99a21f` → RED `c870ce5` → GREEN
  `dc80c12` → 本日志）。全量验证跑在精确候选树 `dc80c12` 上。
- 环境：Windows 10 22621 x64，Python 3.13.12；真实数据
  `data/ashare.duckdb`（截至 20260916）。基线：p45 完成后 `01ef617`。

## 1 交付（软件正确性）

- `ashare_model/direction.py`（DIRECTION_RULE_VERSION=1，additive）：规则 A
  `reference_return`（现行规则形式化，决策**委托** CandidateScorer 单一
  实现、语义零变化，优化对象显式为"参考组合配置下训练窗验证净收益"，
  provenance 记 `portfolio_config_provenance`）；规则 B `label_ic`
  （纯信号对照：训练窗固定标签逐日秩 IC 均值符号，接口**不含任何组合
  配置**，IC 复用 `rank_ic_stats` 单一权威）；`DirectionDecision` 五要素
  （candidate/direction/rule/train_cutoff/label/missing_handling/
  reference_config(A)/evidence）；`decide_direction` 分派、未知规则拒绝。
- `baseline_candidates` 每行新增 `direction_provenance`（方向数值不变，
  镜像补跑一次 score_many 提供对向 evidence）；r01/r02/r03 行自动继承。
- `r01_baseline.percentile_normalize_signal`（avg_rank/N_eff∈(0,1]）——
  新候选族 `combo:all3_pct` 的定义，**未接入** `R01_COMBO_ARMS`，原始秩
  组合语义逐位不变。
- 验收 oracle 全部落地（test_p46 8P）：规则 B 接口无配置参数（签名
  性质钉住）+ IC 符号决策 + 确定性；规则 A 同配置复现、异配置记录
  可区分（`reference_config` 归因）；两规则毒化测试期（≥训练截止列
  信号/标签反号极值）决策与 evidence 逐位不变；行级五要素完整；镜像
  稳定（−signal→−direction，两规则）；百分位助手手算用例。
- 相邻批次：candidates+r01 30P、r02/r03/p45 38P；compileall、分片
  121/121、`git diff --check` 干净。

## 2 真实数据测量（`reports/p46/`，r1 medium 五折、参考格 n10_m2500）

### 2.1 成员有效截面（B5 前置测量）

四成员逐日有效截面 N_eff 在该面板上**逐日相同**（各折 mean 796–800，
逐成员无差异——tier-A medium 面板统一覆盖）。因此原始秩的尺度差异
**在本面板不成立**：`combo:all3`（原始秩）与 `combo:all3_pct`（百分位）
逐日截面 Spearman=1.0000、top-10 选股重叠=1.00、参考格引擎总收益
**逐位一致**（五折全部，如 2021 折 −7.2613% vs −7.2613%）。

**B5 条件分支裁决：不触发**。百分位归一在成员截面一致时是秩的公共
单调变换，组合序不变。`percentile_normalize_signal` 保留为新候选定义
（供未来覆盖不齐的面板/宇宙），`combo:all3` 维持原语义——测量先行的
意义正在于此：避免了无差异的"改进"。

### 2.2 两规则方向对照（A=参考格收益择向，B=固定标签 IC 择向）

| 折 | MOMENTUM_20 | RSQ_60 | REVERSAL_60 | IND_REL_RET_20 | 一致 |
|---|---|---|---|---|---|
| 2021 | A+1 / B−1 | +1 / +1 | A−1 / B+1 | A+1 / B−1 | 1/4 |
| 2022 | −1 / −1 | +1 / +1 | +1 / +1 | A+1 / B−1 | 3/4 |
| 2023 | A+1 / B−1 | +1 / +1 | +1 / +1 | A+1 / B−1 | 2/4 |
| 2024 | −1 / −1 | +1 / +1 | +1 / +1 | −1 / −1 | 4/4 |
| 2025 | A+1 / B−1 | A−1 / B+1 | +1 / +1 | −1 / −1 | 2/4 |

（"A+x / B+y"仅在与 B 不同时标注 A 值；训练窗 IC 均值：MOMENTUM_20
五折全部为负 −0.052..−0.061；RSQ_60 全正 0.037..0.050；REVERSAL_60
全正 0.042..0.057；IND_REL_RET_20 全负 −0.040..−0.046。）

- 总一致率 **12/20（60%）**：两条事前固定的规则是真对照，方向选择
  对规则选择实质敏感。
- **MOMENTUM_20 最刺眼**：训练 IC 五折持续显著为负，规则 B 恒选 −1；
  规则 A 却在 3/5 折选 +1——参考组合收益择向在验证窗内压过了稳定的
  负 IC。这与 p45 的"代理 vs 引擎方向翻转"同族：方向决定对打分通道
  敏感。IND_REL_RET_20 同型（IC 全负，A 早期三折选 +1）。
- RSQ_60 最稳（B 恒 +1，A 仅 2025 折相反）——与 p45 中其唯一全面在界
  一致。

### 2.3 结论边界（按用户验收第 4 条）

- 以上只评价两条择向规则在所测面板的行为差异，**不裁决哪条规则
  "正确"**（那是需要预注册 OOS 检验的研究决策）。
- p42 的 ICIR 加权负结果**不外推**为"学习权重无价值"——本包未测任何
  加权方案；百分位对照的恒等结果也不构成对原始秩组合的褒贬（同面板
  无差异是覆盖性质，不是组合性质）。

## 3 全量回归

`python -m pytest -q tests -n auto`（精确候选树 `dc80c12`，24:00）：
**2131 passed / 2 failed / 1 skipped**（`reports/p46_full_suite.log`）。
总数对账 2126（p45 树）+ 8（p46）= 2134 ✓。两个失败（b03 R9 用户
中止恢复、task_executor 心跳超时）均属自 p43 起归因的进程树/墙钟
时序波动家族：单测复跑分别 114s / 24s 通过，不导入本包任何被改模块，
非 p46 回归（该家族已登记为测试基建后续项）。并行-串行对账 push 前
补记（p40–p45 同口径）。

## 4 裁决与后续（登记，不实施）

- 工程验收达成：两规则可解释（五要素 provenance 随行）、可复现
  （同配置复现、毒化不变）、规则 B 与资金/最低订单**概念隔离**
  （接口无此参数）。
- 研究决策项：(1) 搜索层方向学习是否改用规则 B 或双规则并行——
  需预注册 OOS 检验两规则的折外表现（本包只测了一致性，未测优劣）；
  (2) 方向冻结纪律维持 p37/p42 既有实现（本包未改）；
  (3) `combo:all3_pct` 候选族已定义未启用（本面板无差异，无启用依据）。
