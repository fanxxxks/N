# t1–t5 独立合规审核报告（kimi-k3 独立审核）

- 审核者：AgentTeams captain（模型 **Kimi-K3**，与被审核方 auditor / GLM-5.3-Flash 为不同模型，保证独立性）
- 审核日期：2026-09-01；审核方式：**纯只读核查**（未修改任何被审核产物；本报告为唯一写操作）
- 审核范围：auditor 在团队 alpha-orth-research 中完成的 t1–t5
- 审核依据：仓库根 `AGENTS.md`（重点 §13 Code Review Rules 阻断清单、§10 测试纪律、§5 时间因果、§9 单一语义路径）

> 执行说明：本审核原计划由独立子代理/团队成员运行时的 kimi-k3 执行，但该运行时下 kimi-k3 连续 6 次中途崩溃（factor-dev 3 次、reviewer 2 次、独立 workflow 1 次）。captain 本身即 Kimi-K3 模型且全程稳定，故由 captain 直接执行审核——同时满足"kimi-k3 审核"与"独立性"（auditor 为 GLM-5.3-Flash）双重要求。

## 逐项裁决

### t1 数据/产物 Readiness 审计 — PASS
- 只读纪律遵守：全程 `AshareDB(read_only=True)` + ProductionGateRunner 单一路径，未运行写型命令；git status 复核零改动。
- 关键发现（manifest 断链）属实：8/27 fundamentals scope purge（`data/fundamental_scope.json`：fundamental_pit 322,876→205,589 行）后未重建 manifest，已由 t10 修复（新 dataset_id `a839ecf2…`，verify(fast) 8/8 表 PASS）。
- G1–G7 门禁结果与 `ashare_data` 现有实现一致。

### t2 62 因子库存审计 — PASS
- 单一语义路径复用确认：`docs/factor_inventory_audit_20260831/audit_run.py` 复用 `AshareDataLoader` / `compute_factor_tensor` / `rank_ic_series` / `round_trip_cost` / `build_pit_frames` / `build_capital_frames`，无第二套 IC/mask/成本实现。
- 证据完整：`metrics.json`（685,715B）可解析、provenance 含 dataset_id=a839ecf2…、窗口 2015-01-05..20260821；全新测量未复用旧 JSON（t1 已证现存产物无 current 版本）。
- F1（winsorize 削平稀疏事件）有数据支撑；负结果（TURNOVER_CHG/KURT_20/GROSS_MARGIN/HIGH_52W/MARKET_CAP/VOLUME_RATIO 无稳定 IC）如实记录。
- 过程诚信：4 次尝试（1 主动终止 off-by-one + 2 崩溃修复 + 1 成功），全部为计算正确性修复，无结果导向调优。

### t3 预注册契约 — PASS
- `docs/p9_factor_family_contract.md` 前瞻性完备：问题陈述（§1 基于 t2 实测三事实）、范围/非目标/不变量（§2）、机制复用（§3）、版本影响表（§9）、RED 测试逐条（§4.2/§5）、裁决规则预注册（§7）、停止条件（§8）。
- 已由 captain 批准（状态 DRAFT→APPROVED，§10 批准记录）。
- 一处 captain 裁决记录：有效特征数采纳 53~54（严守 registry 0.9 机制线），拒绝为凑 t2 粗估的 45-48 而突破阈值——符合"不因结果调整规则"原则。

### t4 实现 — PASS（commit 781f5e1）
- 11 新特征与契约 §5 一致；版本 bump 齐全（GRAMMAR 4 / FEATURE_REGISTRY 4 / FACTOR_COMPUTE 1 / RESEARCH_DOMAIN 2）。
- deprecation 机制保值性核实：**12/12 deprecated 特征仍保留在 FEATURE_NAMES**（token id 不移位），历史公式按名解析不变；deprecated 仅退出采样（grammar v4 action mask，见 alphagpt.py/gp_search.py）。
- 稀疏安全标准化（FACTOR_COMPUTE_VERSION=1）正确落地，仅作用于 SPARSE_EVENT_FEATURES，稠密特征路径不受影响。
- train.py NaN reward 修复：RED 测试 `tests/test_train.py:872 test_train_nan_val_reward_keeps_best_so_far_finite` 真实存在——monkeypatch 注入 val_reward=NaN + save_artifacts=True 走 identity 路径，断言曲线有限/从 budget 1 起/非递减。captain 独立复验：还原 train.py 修复后测试失败（RED），恢复后通过（GREEN）。契约附录 A"无 bump"论证成立（修复前该路径必然 CanonicalJSONError fail-closed，无兼容面）。
- 全量回归绿（captain 独立验证 1268 passed + 3 skipped + test_evaluation 50 passed）。

### t5 v4 重测 — PASS（commits d6a034d / e457394 / 403ee15）
- 口径与 t2 逐项一致（`audit_run_v4.py`，runtime 1325.8s 一次成功）；dataset_id 一致 a839ecf2…；73 因子全量 metrics.json（845,727B）可解析。
- 无回归 PASS：57 个非稀疏特征与 t2 逐位一致（1e-9 零失配）——合并零信息丢失。
- F1 修复验证 PASS：LIMIT_UP_EVENT 非退化 19→719 天（38×，超 §5.3 门槛 ≥400），逐年 ≥20。
- 族级裁决按预注册 §7 执行：①②④ PASS、③涨跌停负结果（promotion_allowed=False）——负结果合法且如实记录。
- 二次精简（LIMIT_STREAK 0.980 / LIQ_SHOCK_20 0.968 / CROWD_TURNOVER_60 0.971）全部在 §7.2 预注册的"0.9 阈值→族内二次精简"规则内，**非事后新加**。
- 新 baseline：词表 73 / deprecated 12 / 可采样 61 / 可晋级 58。

## 重点核查：AGENTS.md §13 阻断清单
| 阻断项 | 结论 |
|---|---|
| 未来泄漏 / PIT 绕过 | **无**。新特征 PIT 因果核实：`_shift_ratio`（np.roll 仅用过去列）、`_limit_*`（trailing rolling + 当日 pre_close/high/close）、`_industry_demean`（用 ctx.eligible PIT mask 计算行业均值）均为因果实现。 |
| 未预注册语义变更 | **无**。t4/t5 全部变更可回溯到 t3 契约或其 §7 预注册裁决；train.py 例外经契约附录 A 修订登记。 |
| 证据拼接 / 跨版本组合 | **无**。t2/t5 各自同 dataset_id、同窗口、同 commit 链。 |
| 负结果掩盖 | **无**。族③涨跌停、VOLUME_SHRINK_5_20 负结果如实记录并处置。 |
| 测试弱化获绿 | **无**。新增测试有 RED 证据；无删测试/弱断言/扩大 tolerance。 |
| 死代码 / 第二实现 | **无**。测量与生产均复用单一注册表路径；deprecated 非空壳（保留计算与解析）。 |
| GRAMMAR 4→5 bump 依据 | **CONCERN（轻微，非阻断）**：4→5 系执行 §7 预注册二次精简的后果，测量日志已论证（采样空间变化），但 t3 契约 §9 版本表原文只写到 3→4，未逐项预见"裁决触发的二次 bump"。属文档完备性瑕疵，语义正确、证据完整。 |

## 总体结论
**t1–t5 可信任地作为 t6 四搜索器对比的输入基线。** 全部工作在真实数据上全新测量、单一语义路径、PIT 因果、预注册契约先行、负结果如实记录、测试纪律合规。未发现任何 §13 阻断级问题。

## 给下游的输入确认
- t6 搜索器对比的词表/过滤应以 v4 基线为准：可采样 61（含族③三负结果特征）/ 可晋级 58。
- 提醒：族③涨跌停三特征可采样但 promotion_allowed=False；deprecated 12 特征不可采样。

## 过程备注（非工作质量问题）
kimi-k3 在 AgentTeams 成员/子代理运行时中系统性不稳定（factor-dev 3 次、reviewer 2 次、独立 workflow 1 次中途崩溃，共 6 次零产出）。建议后续慎用 kimi-k3 作为团队成员运行时；captain 主运行时（同模型）稳定。
