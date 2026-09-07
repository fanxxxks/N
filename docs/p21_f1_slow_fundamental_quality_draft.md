# P21 草案：F1 批次预注册——慢基本面质量因子（≤4，第⑤族之外的补位）

- 状态：**DRAFT — 待 governance-auditor 登记与 captain 代行批准**（起草：factor-engineer，2026-09-07；captain t22 验收通知 item 4 "F1 批次解锁：预注册起草开始"）。批准后本文保持前瞻性，禁止以实现/测量输出反向修改。
- 证据基线：t22 修复后重测 `docs/factor_inventory_audit_v2_fix2_20260906/metrics.json`（dataset `157e0cf5…`，FACTOR_COMPUTE 4 / FEATURE_REGISTRY 7 / GRAMMAR 6；全部 77 名有效横截面 0.995）。勘察基线 commit：codex/factor-semantic-fixes @ `327f102`。
- 本文使用 AGENTS.md 的"必须/禁止/应当"语义。任务路由：**研究语义变更**（AGENTS §3.2：新增特征）。

## 1. 问题陈述与批次依据

词表 77 名中慢基本面（slow_fundamental 域，16 名）的质量经济学存在三个未覆盖维度：**方向**（利率/杠杆的 4 季度趋势——存量水平已在词表，方向缺位）、**投资强度**（流量面资本开支——ASSET_GROWTH 只覆盖存量变化）、**偿债质量**（经营现金流对负债的覆盖——CASHFLOW_QUALITY 只覆盖对盈利的确认）。本批次按 p13 §8 模式补位 ≤4 个新特征，全部只用 p17 §5.3 已落库的 v2 三表字段（零 schema 变更），复用 t22 已修复的单一路径（`asof_winner_intervals` + `_ttm`/单期转换）。

## 2. 候选因子（预注册公式；全部 slow_fundamental 域；Tier B）

记号：`TTM(x)_t` = 截至 t 报告期的滚动四季和（`_ttm` 单一路径）；`Δ4q(x)_t` = x_t − x_{t−4 报告期}；可见性一律经 `asof_winner_intervals`（p17 §6.2，修订追踪），分母/分子各取 winner-resolved 值、较晚可见。

| # | 特征 | 预注册公式 | 经济假设 | 方向（预注册） | NaN 守卫 |
|---|---|---|---|---|---|
| F1-1 | NET_MARGIN_TREND_4Q | `TTM(n_income_attr_p)/TTM(revenue)` 的 Δ4q | 净利率改善（定价权/成本控制）延续为正收益；恶化惩罚 | **+1** | 任一 TTM 断档/分母 ≤0 → NaN |
| F1-2 | LEVERAGE_TREND_4Q | `total_liab/total_assets` 的 Δ4q | 杠杆 4 季度恶化（被动加杠杆）→ 财务困境风险上升 | **−1** | 任一端缺失或 total_assets ≤0 → NaN |
| F1-3 | CAPEX_INTENSITY | `TTM(−n_cashflow_inv_act)/total_assets` | 投资/资本开支强度高 → 过度投资与代理成本 → 后续收益转负（投资因子流量面，与 ASSET_GROWTH 的存量面互补） | **−1** | total_assets ≤0 或投资现金流符号缺失 → NaN（n_cashflow_inv_act 通常 ≤0，取负） |
| F1-4 | CFO_TO_DEBT | `TTM(n_cashflow_act)/total_liab` | 经营现金流对总负债覆盖高 → 偿债质量与财务弹性 | **+1** | total_liab ≤0 或 TTM 断档 → NaN |

- **与既有词表的非重复性论证**：ROE/DEBT_RATIO/GROSS_MARGIN/NET_MARGIN（legacy 读，静态水平）不在本批——它们的 v2 as-of 切换是 db76700 已登记的后置任务，不是新特征；family-⑤ 覆盖盈利-现金确认（CASHFLOW_QUALITY）与存量增长（ASSET_GROWTH），本批补**趋势/流量/偿债**三维度。预注册的冗余预检：实施后重测若与 ROE/DEBT_RATIO/NET_MARGIN/family-⑤ 任何成员 |ρ|≥0.9 → 按族内精简裁决（同 p9 §7.2 机制）。
- **族原子性**：4 特征同 commit 进词表、同批裁决、同进退（p13 §8 同款）。

## 3. 范围与非目标

- **范围**：`ashare_data/fundamentals.py` 新增 4 帧（复用 `_family5_asof_frames` 同构路径与 `_ttm`/单期转换单一路径）；`ashare_model/{vocab,feature_metadata,feature_registry,research_domain}.py` 追加登记；`docs/feature_registry.md` 再生。
- **非目标**：不改既有特征公式/源；不动 `financial_indicator_audit`（p17 §5.4 禁止进入因子正式路径）；不改族-⑤/11 legacy 特征；不做 paper/lifecycle 转换。

## 4. 版本影响（预注册）

| 版本 | 旧 | 新 | 理由 |
|---|---|---|---|
| GRAMMAR（vocab.py） | 7 | 8 | 词表 77→81（追加，token id 不移位）。（sequencing 注：草案起草时基线为 6；t70 方向重注册与 p9 t69 Part A 弃用先行落地后，基线已是 7——实施时按当时现值顺延） |
| FEATURE_REGISTRY | 9 | 10 | 4 特征元数据 + depends_on。（sequencing 注：t70 方向重注册已消费 8→9，实施时按当时现值顺延） |
| RESEARCH_DOMAIN | 3 | 4 | slow_fundamental 16→20 |
| FACTOR_COMPUTE | 4 | 不变 | 纯特征追加，既有行计算语义零改动（p13 先例） |
| DATA_TIER | 2 | 不变 | PIT_FUNDAMENTAL→B 既有映射 |
| dataset_id | 157e0cf5 | 随内容重算 | 无 DB 写入——三表已含全部所需字段，**dataset 不变** |

- 旧 artifact 拒绝：GRAMMAR 6 公式仍可按名解码（分派机制既有），新特征仅进未来搜索空间。

## 5. 预期 RED 测试（实现前先红）

1. 合成三表上四公式逐值断言（含 TTM 断档 NaN、分母守卫、Δ4q 对齐）。
2. as-of 可见性：公告日前不可见、修订按 winner 时间线反映。
3. 词表 77→81 追加不移位（既有 token 往返 parity）；GRAMMAR 7 钉值。
4. 域归属 slow_fundamental ×4；RESEARCH_DOMAIN 4 钉值。
5. 未回填态 fail-closed：新特征帧 NaN → 质量门拒绝。

## 6. 测量方案与裁决（预注册）

- 实现后按 t18/t22 同口径全量重测（81 名；h∈{1,2,3,5,10,15,20}、非重叠 IC、min_stocks=10、IS/OOS 2022 分界、|w_b| 修正基线 per p9 附录 B、同成本模型、五 folds）。
- **族门（4 特征整族原子）**：以 7 信号等权复合为基线（|w_b| 修正语义），族内 ≥1 成员 ΔIC_OOS ≥ +0.005 且残差符号与预注册方向一致 → 族通过（成员维持 promotion_allowed=True）；否则负结果：保留计算与采样、promotion_allowed=False（p13 §8 先例）。
- 相关性：与既有特征或族内 |ρ|≥0.9 → Part A 机制裁决；coverage=0 因子的 ΔIC 腿按附录 B 记 not-evaluable。
- 方向按测量裁决（P9 §5.4 惯例）；证伪即如实降级，禁止单方改向。

## 7. 资源上限与停止条件

- 重测单次 ≤30 分钟（参考口径；同 p13 §9.1）。
- 停止：任一 RED 无法稳定失败 → 回到本文修订；覆盖率/门禁红 → 带证据上报；governance/captain 否决候选集 → 修订再批。

## 8. 批准与实施边界

- 批准流程：governance-auditor 登记（CONTRACTS.md 同 commit）+ captain 代行批准 → 实施由 factor-engineer 在 codex/factor-semantic-fixes 上执行（RED → 最小实现 → GREEN → 全量 → 重测 → 裁决 → 测量日志）。
- 本草案不授权 push/PR/merge。
