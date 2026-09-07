# P23 草案：INDUSTRY_MOMENTUM 方向重预注册（authored +1 修复后证伪 → −1）+ F 批次相关性核对照表

- 状态：**REJECTED — 重假设预注册驳回（captain 裁决，t24 路由）：冗余预检不通过**——ΔIC −0.0167 低于 IND_REL_* 族内成员增量（+0.0079~+0.0283）且实证形态（行业反转）与族方向同源，纳入即违反 ΔIC 增量判据。处置 = 负结果 + 冗余预检失败完整留档（INDUSTRY_MOMENTUM promotion_allowed=False 已实施，authored +1 保留为证伪证据）；未来族结构变化可重新提案（新预注册）。本文件其余部分保留为驳回依据记录。
- 证据基线：t22 修复后重算 `docs/factor_inventory_audit_v2_fix2_20260906/metrics.json`（dataset `157e0cf5…`，FACTOR_COMPUTE 4）；t18 轮 `docs/factor_inventory_audit_v2_20260905/metrics.json`（dataset `43ef1e4c…`）。勘察基线 commit：`327f102`。

## 1. 完整历史披露（captain 裁定第 1 条要求的三段式）

| 轮 | dataset | 覆盖 | IC10（ICIR） | ΔIC_OOS | 残差 OOS | 效力 |
|---|---|---|---|---|---|---|
| round-1（t18） | 43ef1e4c（修复前） | **0.000** | None | +0.0357* | None | **无效——数据伪影**（消费路径缺陷 + D3 缺行：行业帧近全 NaN，IC 无从计算；ΔIC 为基线翻号伪影，p9 附录 B 实证） |
| round-2（t22 after + 附录 B 修正重算，同值） | 157e0cf5（修复后） | **0.838** | −0.0096（−0.059）/ h20 −0.0176（−0.114） | −0.0167 | −0.0170 | **有效证伪**：authored +1 在全 horizon 为负（h1 −0.0063）、年度同向 4/13 |

\* round-1 的 ΔIC 为 p9 附录 B 缺陷值（零信号 × w_b<0 → 基线翻号），双重无效。

- **authored 方向（+1）保留为负结果证据**：`feature_metadata.py` 的 +1 假设（"行业动量溢出：热行业延续"）与四轮 h 域负 IC 一并被本草案记录为已检验且被证伪的历史，不得删除或改写。

## 2. 新方向假设（预注册）

- **新 `expected_direction = −1`**：申万行业指数 20 日动量在 A 股呈**反转**（行业热度过热 → 成分股后续跑输），与修复后实测的 IND_REL_* 残差族（authored −1，ICIR −0.19…−0.43，六成员全部符号一致）的经济学一致——但**信息层不同**：INDUSTRY_MOMENTUM 是行业指数层面的方向信号，IND_REL_* 是个股对行业的残差，两者构造正交（实测 |ρ|max = 0.379 vs MOMENTUM_20，详 §4）。
- 按 P9 §5.4"方向按测量裁决"惯例落盘；批准后禁止按未来测量再单方改向（再改须新预注册）。

## 3. 范围、版本与守卫

- **范围**：`feature_metadata.py` 的 INDUSTRY_MOMENTUM `expected_direction`（+1 → −1）；`FEATURE_REGISTRY_VERSION` 在当时现值上 **+1**（起草时点现值 7 → 8；若 F1 批次先行批准落地则顺延为 8 → 9，两案取先后序）；`docs/feature_registry.md` 同 commit 再生。
- **不变**：公式/数据源/可用时间（t22 修复后语义）；FACTOR_COMPUTE（方向元数据不改变因子值）；GRAMMAR（无词表变化）；tier（C，政策阻断不因方向变化解除）。
- **迁移/拒绝**：旧 artifact 携带自身 direction 载荷，零迁移；元数据变更只影响未来治理/搜索解读。
- **RED**：元数据断言（== −1）+ registry 文档 drift guard（预期方向列）。

## 4. 冗余预检（实测，修复后数据）

| 邻居（fix2 全词表 top-|ρ|） | ρ | 结论 |
|---|---|---|
| MOMENTUM_20 | +0.379 | 低于 0.9；且信息层不同（个股横截面动量 vs 行业指数动量） |
| RET_10 / MOMENTUM_60 / RET_5 / REVERSAL_5 / MAX_20 | 0.175-0.259 | 同上 |
| **IND_REL_*（六成员）** | — | 构造正交（残差 vs 指数层），无直接配对；翻转后与 IND_REL_* 的增量关系按下次重测裁决 |

- 结论：方向翻转**不产生任何 ≥0.9 新对**，无 Part A 触发。

## 5. F 批次相关性核对照表（captain 裁定第 3 条要求）

| 配对 | 经济重叠预期 | 现测可行 | 实测值 | 后续测量点 |
|---|---|---|---|---|
| MARGIN_BALANCE_CHG × F1-1 NET_MARGIN_TREND_4Q | 低（融资流量 vs 盈利能力趋势） | 否（F1 未实现） | — | F1 实现后重测 |
| MARGIN_BALANCE_CHG × F1-2 LEVERAGE_TREND_4Q | 低-中（负债侧经济学相邻） | 否 | — | 同上 |
| MARGIN_BALANCE_CHG × F1-3 CAPEX_INTENSITY | 低 | 否 | — | 同上 |
| MARGIN_BALANCE_CHG × F1-4 CFO_TO_DEBT | **中**（现金流/负债经济学相邻，需实测排除冗余） | 否 | — | 同上（若 ≥0.9 → Part A 裁决） |
| INDUSTRY_MOMENTUM × F1-1..4 | 极低（行业指数动量 vs 个股基本面） | 否 | — | 同上 |
| MARGIN_BALANCE_CHG × 既有 77 | 已测：max = MARGIN_CROWD_60 **+0.794**（<0.9，同源不同变换，无 Part A 触发） | ✓ | 见左 | — |
| INDUSTRY_MOMENTUM × 既有 77 | 已测：max = MOMENTUM_20 **+0.379** | ✓ | 见左 | — |

## 6. 批准与实施边界

- 批准流程：governance-auditor 登记 + captain 代行批准 → factor-engineer 实施（单 commit：元数据方向 + 版本 +1 + 文档再生 + RED→GREEN）→ 下一次重测执行全链条（PIT/覆盖/ΔIC/成本后空间——captain 裁定第 2 条：无跳步）。
- 本草案不授权 push/PR/merge。
