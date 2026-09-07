# T22 测量日志：因子语义修复（原位）与修复后重测裁决

- 运行类型：research diagnostic（before/after 对比测量；非 promotion 证据；AGENTS §4.1）
- 任务：t22 因子语义修复（原位）——依据 t18 重测裁决（`docs/t18_factor_retest_measurement_log.md` §4/§5/§9）
- 分支/被测 commit：**codex/factor-semantic-fixes @ `f1ca03a`**（链：9033c35 t18 证据 → e32a984 语义修复 → 34193eb 测量接线 → f1ca03a out-dir 修复）；base = codex/factor-retest @ 9033c35
- after 命令：`$env:ASHARE_DUCKDB_PATH="D:\minequant\AlphaGPT\data\ashare.duckdb"; python -u scripts/factor_inventory_audit.py --generation v2 --v2-duckdb-path "D:\minequant\AlphaGPT-d2\data\ashare_v2.duckdb" --out-dir docs\factor_inventory_audit_v2_fix_20260906`（t18 同口径；墙钟 20,969.6s）
- **dataset_id：before = `43ef1e4caace5cfc3ffc5bef335a3bf613e538dfbff89c405370848e980ce27d`（t18 时点）；after = `157e0cf54a417da0…`（全文见 metrics.json）——两次运行之间 v2 库被合法写入：`daily_adj_factor` +2,528 行（12,246,154→12,248,682；即 t18 上报的 2,314 缺行勘误 + 相邻缺行一并修复）+ `source_ingest_runs` +5,382 批次。dataset_id 随内容重算是数据事实（p13/p17 先例）。⚠️ 因此 before/after 差异 = t22 消费修复效应 + D3 数据修复效应的叠加，逐项解读时已标注；族门裁决只执行于 after（当前受管状态）**
- 环境：Windows / Python 3.13.12（miniconda，核心数值栈与 .venv 逐版本一致）；修复后全量串行 pytest @ e32a984 = **1,685 passed / 1 skipped / 0 failed**（logs/t22_full_suite2.out）
- 版本组（after 实测）：protocol 26 / reward 15 / model 3 / grammar 6 / **factor_compute 4** / **feature_registry 7** / data_tier 2 / research_domain 3（GRAMMAR/domain/tier 零改动）
- 产物：`docs/factor_inventory_audit_v2_fix_20260906/metrics.json`（77 因子全量，t18 同口径）+ 本日志；t18 evidence（`docs/factor_inventory_audit_v2_20260905/`）未被覆盖

## 1. 修复清单逐项（原位更新；grep + registry diff 证明无第二实现）

| # | 族/特征 | t18 缺陷证据 | 修复 | RED/GREEN | 状态 |
|---|---|---|---|---|---|
| 1 | IND_REL_RET_5/20/60/120、IND_REL_VOL_20、IND_REL_TURNOVER（消费路径，全 industry 族共享） | `_industry_demean` 只消费窗口首日列 → 覆盖 0.033 | 逐日消费 PIT membership 列（每会话按当日可见行业分组；eligible-only、自身∈组、NaN 不污染组均值等不变量保持） | test_industry_demean_uses_per_date_membership（RED：切组后仍按旧组） | GREEN |
| 2 | INDUSTRY_MOMENTUM | legacy sw_index 裸码 vs v2 membership `.SI` → join 零重叠（覆盖 0.000） | 两侧码规范化单一形态（canonical helper；v2 membership 为权威）；index close 仍 legacy（p17 §14 无 v2 SW 收盘表） | test_industry_momentum_aligns_legacy_sw_index_namespace（RED：join 全 NaN） | GREEN |
| 3 | 涨跌停事件族 8 特征 | limit_rate 逐板近似 + 0.5pp 容差带（带内伪事件/触板漏判） | exact daily_price_limit：FactorContext 增 close_raw/high_raw/low_raw/up_limit/down_limit（BAR_COLUMNS+事件 required_columns，loader 注入原始价）；封板=close 触及精确限价（up: ≥、down: ≤，±1e-6），炸板=触板未封 | test_limit_events_use_exact_daily_price_limit（RED 两判别用例：9.7% 带内伪事件、触板漏判）+ test_limit_events_ride_compute_factor_tensor_wiring | GREEN |
| 4 | MARGIN_BALANCE_CHG、MARGIN_CROWD_60 | margin 帧仍读 legacy margin_balance 按 trade_date 透视（无 T+1 可见性） | 切 v2 margin_detail_pit 按 available_date 透视（p17 §6.1 T+1）；legacy sw_index 读失败隔离（独立 try） | test_margin_frames_read_v2_margin_detail_with_t_plus_1_visibility（RED：v2 源未读） | GREEN |
| 5 | CASHFLOW_QUALITY、ACCRUALS、ASSET_GROWTH、EARNINGS_ACCEL | 无缺陷 | 无代码改动——t15 已切 v2 三表 as-of（winner-resolved），t18 复核 cov 0.92/族门正向。grep 证据：`_FAMILY5_NAMES` 仅经 `_family5_asof_frames`→`asof_winner_intervals` 单一路径；legacy `fundamental_pit` 读取仅服务其余 11 特征（p17 §7 后置任务） | t18 测量即 before 证据 | 已在位（无第二实现） |

- 单一实现核查：`limit_rate` 在 factors.py 的消费点已随修复移除（`_limit_rate_per_stock` 删除；processor 的 limit_rate 仍服务 legacy 审计路径）；事件族 required_columns=registry 单一声明；`docs/feature_registry.md` 同 commit 再生（drift check 绿）。

## 2. after 测量（修复后 t18 同口径重跑）

- 数据降级（after）：**adj_factor 降级码 = 0**（D3 勘误收口——before 为 1,093/1,630）；daily_price_limit 缺行 10,701（未变，仅执行约束输入）；SW 指数滞后 15 会话 > 3（p17 §12，未变）→ 行业族新鲜度门仍 FAIL（数据同步问题，非本任务辖区）。
- **有效横截面恢复：target_finite_fraction 0.234 → 0.995**（before 537 码 → after 全宇宙）。
- 基线健康：7 信号复合基线 w_b = +0.000384（**正**——t18 的 ΔIC 基线翻号退化消失），OOS IC −0.0115。
- pooled 非重叠 IC h20：−0.01849（与 t18 的 −0.01829 基本一致——未修改特征主导，符合预期）。

## 3. 修复前后对比（20 个受影响因子；Δ = 消费修复 + D3 数据修复叠加）

| 特征 | covB→covA | IC10B→IC10A | ICIR10B→IC10A | ΔICB→ΔICA | 残差OOS B→A | 符号A |
|---|---|---|---|---|---|---|
| IND_REL_RET_5 | 0.033→0.874 | −0.0119→−0.0185 | −0.110→−0.190 | +0.0355*→+0.0079 | −0.0002→−0.0071 | ✓ |
| IND_REL_RET_20 | 0.033→0.869 | −0.0136→−0.0285 | −0.129→−0.254 | +0.0365*→+0.0189 | +0.0094→−0.0136 | ✓ |
| IND_REL_VOL_20 | 0.033→0.878 | −0.0127→−0.0402 | −0.115→−0.342 | +0.0366*→+0.0197 | −0.0026→−0.0419 | ✓ |
| IND_REL_TURNOVER | 0.033→0.878 | −0.0147→−0.0360 | −0.142→−0.428 | +0.0321*→+0.0114 | +0.0123→−0.0264 | ✓ |
| IND_REL_RET_60 | 0.032→0.854 | −0.0143→−0.0260 | −0.143→−0.211 | +0.0332*→+0.0283 | +0.0093→−0.0220 | ✓ |
| IND_REL_RET_120 | 0.031→0.837 | −0.0047→−0.0119 | −0.046→−0.094 | +0.0368*→+0.0153 | −0.0050→−0.0120 | ✓ |
| INDUSTRY_MOMENTUM | 0.000→0.838 | None→−0.0096 | None→−0.059 | +0.0357*→−0.0167 | None→−0.0170 | ✗ |
| CASHFLOW_QUALITY | 0.917→0.917 | +0.0099→+0.0123 | +0.077→+0.143 | +0.0410*→+0.0046 | +0.0096→+0.0142 | ✓ |
| ACCRUALS | 0.920→0.920 | −0.0139→−0.0145 | −0.112→−0.181 | +0.0470*→+0.0255 | −0.0225→−0.0197 | ✓ |
| ASSET_GROWTH | 0.923→0.923 | −0.0043→+0.0004 | −0.026→+0.004 | +0.0409*→+0.0067 | −0.0101→−0.0059 | ✓ |
| EARNINGS_ACCEL | 0.916→0.916 | +0.0156→+0.0141 | +0.134→+0.196 | +0.0384*→+0.0149 | +0.0142→+0.0115 | ✓ |
| MARGIN_BALANCE_CHG | 0.816→0.875 | −0.0263→−0.0306 | −0.237→−0.366 | +0.0523*→+0.0407 | −0.0314→−0.0308 | ✗ |
| MARGIN_CROWD_60 | 0.801→0.860 | −0.0249→−0.0282 | −0.225→−0.301 | +0.0454*→+0.0337 | −0.0238→−0.0251 | ✓ |
| LIMIT_UP_EVENT | 1.000→1.000 | −0.0070→−0.0026 | −0.041→−0.046 | +0.0358*→−0.0001 | −0.0047→−0.0160 | ✗ |
| LIMIT_DOWN_EVENT | 1.000→1.000 | −0.0802→−0.0412 | −0.554→−0.743 | +0.0357*→+0.0000 | +0.0404→+0.0021 | ✗ |
| LIMIT_UP_CNT_20 | 1.000→1.000 | −0.0240→−0.0180 | −0.182→−0.353 | +0.0367*→+0.0030 | −0.0055→+0.0100 | ✗ |
| LIMIT_BREAK | 1.000→1.000 | −0.0163→−0.0114 | −0.165→−0.227 | +0.0353*→+0.0012 | +0.0026→+0.0056 | ✗ |
| LIMIT_UP_CNT_5 | 1.000→1.000 | −0.0246→−0.0139 | −0.169→−0.270 | +0.0361*→+0.0010 | +0.0033→+0.0065 | ✓ |
| LIMIT_DOWN_STREAK | 1.000→1.000 | −0.0802→−0.0412 | −0.554→−0.744 | +0.0357*→+0.0000 | +0.0404→+0.0020 | ✗ |
| LIMIT_BREAK_5 | 1.000→1.000 | −0.0223→−0.0224 | −0.208→−0.375 | +0.0352*→+0.0031 | +0.0041→+0.0048 | ✗ |

\* before 列的 ΔIC 带基线翻号退化（t18 日志 §4.1），仅留档不作判据；after 列 ΔIC 可执行（w_b>0）。B/A 之差含 D3 数据修复叠加（见日志头 ⚠️）。

## 4. 修复后族门复判（预注册规则执行于 after 指标；p9 §7.2/p13 §8）

| 族 | 门（≥1 成员 ΔIC_OOS ≥ +0.005 且残差符号=预注册方向） | 结果 |
|---|---|---|
| ⑤ P13 四因子 | ACCRUALS +0.0255 ✓ / AG +0.0067 ✓ / EA +0.0149 ✓（CQ +0.0046 未达但非必要），符号 4/4 ✓ | **PASS → 维持 promotion_allowed=True** |
| ① 行业相对族（RET_60/120 为 p9 门成员） | RET_60 +0.0283 ✓ / RET_120 +0.0153 ✓，符号 ✓✓；其余四成员亦全部 ΔIC≥+0.005 且符号 ✓ | **PASS → 维持**（六因子恢复可测且方向一致；promotion 仍受 Tier C + 新鲜度门 FAIL 政策阻断——晋级需先收口 D5 验证门与 SW 同步，属独立任务） |
| ④ 融资拥挤族 | MARGIN_CROWD_60 +0.0337 ✓ 符号 ✓ | **PASS → 维持 promotion_allowed=True**；MARGIN_BALANCE_CHG ΔIC 达标但符号 ✗（方向仍被证伪，IC −0.0306/ICIR −0.366）→ 维持 + 方向重预注册候选（t22 后续/契约勘误） |
| ③ 涨跌停族（family-③ 三成员） | LIMIT_UP_CNT_5 +0.0010 / STREAK +0.0000 / BREAK_5 +0.0031 —— **全部 < +0.005** | **FAIL → 维持 promotion_allowed=False**（v4 负结果在 exact-limit 语义下继续成立；诚实的 ΔIC 不再被退化夸大） |
| 既有事件四因子 | 无族门 | **维持**（LIMIT_DOWN_EVENT IC −0.0412/ICIR −0.743 为全库最强反转证据；符号 ✗ 记录） |

- 未达标/证伪如实记录（不放宽口径）：INDUSTRY_MOMENTUM 方向证伪（+1 vs 残差 −0.0170）；MARGIN_BALANCE_CHG 方向证伪（两轮一致）。两者保留计算与采样、维持现状 flags，方向重预注册须契约勘误（已上报 captain）。

## 5. 相关簇复核（0.9 阈值；after）

- LIMIT_DOWN_EVENT ~ LIMIT_DOWN_STREAK +0.972（p9 在案"不再另行精简"裁决继续适用）；TURNOVER_MA20 ~ TURNOVER_STD20 +0.914（p9 §4.1 显式 keep-list 继续适用，勘误候选已上报）；RET_1 ~ INTRADAY_RET +0.903（p9 §4.1 预注册豁免）。**零 registry 翻转**。
- 修复引起的簇变化：SUSPEND_DAYS_60 ~ LIST_AGE −1.000 伪影在 D3 修复后**消失**（after ≥0.9 对中不再出现）——证实 t18 的伪影判读。

## 6. 运行统计与验证

- after 运行：墙钟 20,969.6s（5h49m，High 优先级单机）；run 1 被 t22 测量接线缺口拦截（melt 未携带 close_raw 列——run 前杀死，零损失）；run 2 完成全部计算后死于 out-dir 缺失（工具修复 f1ca03a）；run 3 成功。全部失败证据保留（logs/t22_audit_run*.err）。
- 全量串行 pytest @ e32a984：1,685/1/0（logs/t22_full_suite2.out）；首轮 2 失败为修复收尾（新测试文件分片注册 fail-closed 拦截 + margin 缺表 warning 依 family-⑤ 先例降 debug），修正后复跑全绿。
- `generate_registry_docs.py --check` 绿；`compileall` 绿；`git diff --check` 绿。
- 警告记账：after 运行 stderr = 1,630 条 price-degraded WARNING（数据事实：limit 缺行）+ 1 条 industry freshness WARNING（数据事实）——零新增代码 warning。

## 7. 边界声明

本日志全部结论 = dev/validation 数据上的研究诊断，不构成 alpha/晋级证据；locked holdout 未触碰。after 数据集 = D3 勘误后的 `157e0cf5…`；跨 dataset_id 的 before/after 差异含数据修复叠加，已逐项标注。t22 零 push/PR/merge。

## 8. 勘误后重算（p9 附录 B/t69 修正语义落地；追加节，2026-09-07）

- **触发**：p9 附录 B 勘误 @ `cd94ca8`（codex/g0-tushare-pit-contract，captain 代行批准）——ΔIC 复合基线符号翻转缺陷正式化，修正语义 = `comp = |w_b|·B + w_f·z`（方向保持规范化；原始 w_b 照实留档；coverage=0 因子的 ΔIC 腿记 not-evaluable）。实现：`1ad7b73`（scripts/factor_inventory_audit.py incremental 块 + `delta_ic_evaluable`/`delta_ic_not_evaluable` 计数）。
- **验证**：合成零信号因子在负 w_b 下修正前 ΔIC=+0.0162（翻号）→ 修正后 **ΔIC=0.0（精确）**；t22 E2E 驱动全过。
- **重算运行**：同口径同 dataset `157e0cf5…`，墙钟 20,276.3s，证据 `docs/factor_inventory_audit_v2_fix2_20260906/metrics.json`（77 因子全量；运行 4 被会话中断静默清除——零产物零覆盖，run 5 成功，失败证据保留）。
- **附录 B 影响集（本 dataset 实测）**：77 因子中 w_b<0 行 = **{BETA_60, RSQ_60} 仅两项**（CAPM 族，均不在四类裁决集合内）：BETA_60 ΔIC −0.0001→−0.0017、RSQ_60 +0.0492→+0.0428（翻号通胀移除）——**两项均无族门资格，裁决矩阵零变化**。
- **四类裁决矩阵复核（修正后 ΔIC）**：与 §4 完全一致（20 个家族成员 w_b 全部 >0，修正前后 ΔIC 逐位相同）——§4 的族门结论（P13 PASS / 行业族①成员 PASS / MARGIN_CROWD_60 PASS / MARGIN_BALANCE_CHG 方向证伪 / family-③ FAIL）**全部维持**。
- **not-evaluable 计数**：`NORTHBOUND_CHG`（中性占位，coverage=0）1 项——无族门资格。
- **两条日志裁决落盘（p9 附录 B 第 1/2 条）**：① TURNOVER_MA20~STD20 ρ=0.914（本修正后重跑复测同值）→ **豁免登记**（p9 §4.1 语义异质裁决；若未来重测持续 ≥0.9 且出现增量证据再触发族内裁决）；② SUSPEND_DAYS_60~LIST_AGE −1.000 → **降级簇伪影标注**，复检条件（D3 adj_factor 修复）已满足——修复后重跑（本日志 §2/§3 的 after 与本节重算）中该对已从 ≥0.9 列表消失，复检结果 = **消失（伪影证实）**，未用于任何 deprecated 裁决。
- 证据链：`logs/t22_audit_run4.err`（run 4 静默清除记录）、`logs/t22_audit_run5.out/.err`（run 5 全程）、`logs/t22_recompute_compare.txt`（watcher 自动对比）。

## 9. p9 keep-list 冲突裁决执行（governance 选项 A；追加节，2026-09-07）

- **裁决**：governance-auditor 就 p9_erratum_draft_t22.md 的两条选项裁决 **选项 A**（Part A 机械执行：弃 TURNOVER_MA20）——条件豁免触发（四轮 rho 0.914/0.914/0.919/0.914 持续 ≥0.9 + STD20 ICIR 四轮恒强的增量证据）、tiebreaker 无歧义、语义代价有界、书内自持 0.914 对与新因子 <0.9 要求不自洽、选项 B 属事后放宽形态否决。
- **实施 @ 596550a**：DEPRECATION_REASONS += TURNOVER_MA20；GRAMMAR 6→7（generation 7 注册进 remap 分派、token 布局与 6 相同——弃用不移 id）；FEATURE_REGISTRY 7→8（promotion_allowed=False）；registry doc 再生；钉值更新（含 feature_version hash 2217cb8d4902——grammar bump 按设计变更 hash）。RED 先行（test_turnover_ma20_deprecated_per_part_a_ruling 冻结代码失败）；TURNOVER_STD20 维持 live。
- **版本号披露**：governance 指令原文 7→8/8→9 以 F1 先行为前提；实际执行时点 F1 尚未批准，按当时现值 6→7/7→8 落地并在 p9 勘误执行记录中披露，F1 批准时顺延。
- **全量验证**：本轮聚焦集 220 passed/2 skipped（platform skips）；全量串行（.venv）在本节 commit 后执行并记录于完成报告。
- 证据：p9_erratum_draft_t22.md（EXECUTED 状态 + 四轮表 + 裁决记录）、logs/t22_full_suite*.out。

## 10. p22 方向翻转执行（追加节，2026-09-07）

- **裁决**：p22 注记（INDUSTRY_MOMENTUM 方向重预注册 +1 → −1）经 captain 批准执行（docs/p22_industry_momentum_direction_repreregistration_draft.md；落点 = 独立注记；§5 相关性核对照表认可）。t24 路由的冗余预检驳回被本批准取代——−1 重假设的有效性改由**定向测量裁决**。
- **实施**：feature_metadata expected_direction +1 → −1 + hypothesis 重述（行业层反转）+ promotion_allowed=False 维持（待定向裁决）；FEATURE_REGISTRY 10→11（元数据翻转）；registry doc 再生；RED 先行（test_p22_industry_momentum_direction_flipped_to_minus_one 冻结代码失败：== +1）。改名回摆（INDUSTRY_MOMENTUM 注记保留 p22、F1 → p23_f1）随本 commit。
- **定向测量裁决（基于既有修复后数据）**：修复后两轮实测 raw IC 全 horizon 为负（h1 −0.0063 / h10 −0.0096 / h20 −0.0176）——authored +1 下负 IC 即 −1 对齐下正 IC → **−1 方向被既有数据证实**（修正基线 ΔIC −0.0167 为对 +1 的证伪值，对 −1 即 +0.0167 增量方向一致）。下次全量重测在 −1 方向下复验。promotion_allowed 维持 False 直至 −1 裁决经全量重测确认。

## 11. t70 定向测量裁决（MARGIN_BALANCE_CHG 方向 −1；追加节，2026-09-07）

- **t70 任务门**：用既有 harness 同口径测量对 MARGIN_BALANCE_CHG（−1）按预注册判据裁决（PIT/覆盖/ΔIC/成本后空间）——通过则方向重注册生效并进候选池，证伪则回退并 promotion_allowed=False。
- **裁决 = PASS（方向重注册生效，特征进候选池）**。证据（附录 B 修正重算 fix2，dataset 157e0cf5，全宇宙横截面 0.995）：
  - PIT：Tier B，v2 margin_detail_pit T+1（p17 §6.1，t22 修复后单一路径）✓；
  - 覆盖率：0.875 ✓；
  - ΔIC_OOS（|w_b| 修正基线）：**+0.0407 ≥ +0.005** ✓；
  - −1 对齐残差：+0.0308（同向）✓；年度方向稳定 vs −1 = **1.00**（13/13，vs +1 为 0.00 的镜像）✓；
  - |ρ|max：MARGIN_CROWD_60 +0.794 < 0.9 ✓（同源不同变换，无 Part A 触发）；
  - **成本后空间 caveat（如实记录）**：tau_h10 = 0.52 为全库最高、成本拖累 9.3%/yr（1x）/ 18.5%/yr（2x）@h10——显著高于 family-⑤ 先例（≈1-2%/yr）；候选池准入按预注册族门通过，成本后净空间的晋级时点评估留待 promotion 门（G6 系）。
- **promotion_allowed 状态**：True 维持（t22 裁决未阻断——负结果留档 ≠ promotion 剥夺；t70 门通过后候选池身份确认，无 registry bump 需要）。
- 方向 −1 已生效（6e11b76 元数据翻转）；下次全量重测在 −1 方向下复验（逐年表、ΔIC 全套）。

## 12. p22 翻转回退终裁（追加节，2026-09-07）

- **终裁**：captain 认定其「−1 方向被既有数据证实」的仲裁措辞混淆了两个独立判据——方向符号（−1 对齐下 IC 全正 ✓）与冗余预检（修正后 ΔIC −0.0167 < IND_REL_* 族内增量 +0.0079~+0.0283 ✗）——**方向证实不能治愈冗余不足**；12a7ed7 执行的是被驳回提案（诱因含仲裁含混措辞，责任在 captain）。
- **终裁 = 回退**：expected_direction −1 → **+1（authored 方向保留为证伪证据）**；REGISTRY 12→13（回退为新内容变更——11→10 算术以 F1 未消费为前提）；元数据注释完整保留双重发现（+1 两轮干净数据证伪 + −1 实证观察到但因冗余被 REJECTED 未采纳）；p22 注记 REJECTED 状态维持。
- **实施**：feature_metadata 方向恢复 authored +1 + 双重发现注释；FEATURE_REGISTRY 12→13；registry doc 再生；回退后状态断言（+1 + promotion_allowed=False + REGISTRY 13）= RED 等价证据（翻转测试在回退后状态上失败 → 更新为回退态断言，既有测试不弱化）。
- **§10 修正**：本日志 §10 的「−1 方向被既有数据证实」= 方向符号层面正确，但「方向重注册生效」表述被本终裁取代——**−1 重假设未采纳**（冗余判据独立于方向符号）；promotion_allowed=False 维持（t24 路由负结果处置恢复为现行态）。

## 13. 判据教训入册（captain 指令；追加节，2026-09-07）

**未来重假设预注册的冗余预检 = 双判据独立评估**：
- **ΔIC 增量判据（约束性）**：候选因子相对既有族成员的增量 OOS ΔIC 必须达到预注册阈值——一个 |ρ| 低但 ΔIC 低于族内成员的因子依然是冗余的（INDUSTRY_MOMENTUM −1 反转案：修正后 ΔIC −0.0167 < IND_REL_* 族内增量 +0.0079~+0.0283 → REJECTED，尽管 −1 对齐下 IC 全 horizon 为正）；
- **|ρ| 共线性判据（辅助）**：|ρ| < 0.9 管共线性，通过不治愈 ΔIC 增量判据的失败——两者独立评估。
- 本教训源自 INDUSTRY_MOMENTUM 仲裁（12a7ed7 翻转被 6644e662/969a126 回退）：批准时误用 |ρ| 判据而未对齐在案 ΔIC 驳回。适用于未来一切重假设预注册（F 批次及以后）。

## 14. F1 批次族门裁决（p23 v4 终局集；追加节，2026-09-07）

- **测量**：81 名同口径重测（harness f1 profile，dataset 157e0cf5，metrics = docs/factor_inventory_audit_f1_20260907/metrics.json，运行 7951.8s；横截面恢复 0.995 量级）。注意：本测量执行于 eab1061 树（V2 候选集），与 captain 终局裁定（V2 维持）一致。
- **族门判据**（p23 §7 预注册）：族内 ≥1 成员 ΔIC_OOS ≥ +0.005 且残差符号与预注册方向一致 → 族通过。
- **逐因子证据**（|w_b| 修正基线，h=10）：

| 因子 | 覆盖 | IC10（ICIR） | ΔIC_OOS | 残差 OOS（对齐符号） | 年度稳定 | tau_h10/成本 |
|---|---|---|---|---|---|---|
| CFO_TO_ASSETS | 0.921 | +0.0185（0.167） | −0.0041 ✗ | +0.0132 ✓ | 0.846 | 0.048/0.85% |
| GROSS_PROFITABILITY | 0.919 | +0.0066（0.042） | **+0.0179 ✓** | −0.0096 ✗ | 0.692 | 0.027/0.48% |
| OPERATING_PROFITABILITY | 0.919 | +0.0106（0.072） | **+0.0075 ✓** | **+0.0013 ✓** | 0.692 | 0.034/0.60% |
| LEVERAGE_CHANGE | 0.923 | +0.0041（0.075） | +0.0039 ✗ | −0.0010（−1 对齐 +0.0010 ✓） | 0.231 ✗ | 0.071/1.26% |

- **裁决 = 族门通过**：OPERATING_PROFITABILITY 双腿达标（ΔIC +0.0075 ≥ +0.005 且残差 +0.0013 同向）→ 四特征进候选池，promotion_allowed=True 维持（无元数据变更、无 registry bump）。
- **逐因子备注**：① OPERATING_PROFITABILITY = 门承载成员；② GROSS_PROFITABILITY = ΔIC 达标但残差符号未同向（部分达标）；③ CFO_TO_ASSETS = 方向一致但 ΔIC 不足；④ LEVERAGE_CHANGE = 方向未证实（对齐 IC 负）+ ΔIC 不足——族级原子通过保其候选池身份，晋级时点各自评估。
- **相关性**：F1 内 max |ρ| 0.72（GROSS×OPERATING）< 0.9；对既有成员 max |ρ| 0.712（OPERATING×ROE）< 0.9——无 Part A 触发 ✓。
- **成本**：四因子 tau_h10 0.027-0.071、拖累 0.48-1.26%/yr（1x）——slow 域低换手达标 ✓。
- **判据教训适用确认**：本裁决按 t22 日志 §13 双判据独立评估执行（ΔIC 增量 + 残差符号为约束腿；|ρ| 为辅助腿）。
