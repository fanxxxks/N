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
