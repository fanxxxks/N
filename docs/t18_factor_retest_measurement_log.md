# T18 测量日志：77 因子全量重测与重新裁决（ashare_v2 新 dataset + D4/D5/D6 语义）

- 运行类型：research diagnostic（非 promotion 证据；AGENTS §4.1）
- 被测 commit：**codex/factor-retest @ `53a675b`**（commit 链：base `db76700`［captain 裁决的 D4+D5+D6 超集线］→ `0a8565e` 测量工具 v2 profile → `c8d0043` D6 勘误#4 修复 cherry-pick［自 4b065c0］→ `97d6b1d` test_runspec FACTOR_COMPUTE 钉值承接［t58 评审件 aefb71d 的 hunk，captain 指令］→ `53a675b` 测量工具 UnboundLocalError 修复）。被测实现语义面 = db76700 + 勘误#4；其余为测量基础设施与测试钉值，零研究语义改动。
- 命令：`$env:ASHARE_DUCKDB_PATH="D:\minequant\AlphaGPT\data\ashare.duckdb"; python -u scripts/factor_inventory_audit.py --generation v2 --v2-duckdb-path "D:\minequant\AlphaGPT-d2\data\ashare_v2.duckdb"`（单一 harness `scripts/factor_inventory_audit.py`，p13 §8"新脚本"要求由其 v2 profile 满足：77 名断言含 P13 族尾）
- 数据：v2 = `D:\minequant\AlphaGPT-d2\data\ashare_v2.duckdb`，**dataset_id = `43ef1e4caace5cfc3ffc5bef335a3bf613e538dfbff89c405370848e980ce27d`**（只读经 `build_v2_dataset_manifest` 重算，与 captain 简报/t53 一致；57,131,659 行；daily_bar_raw 11,886,384 行 20140102..20260904）。legacy = `D:\minequant\AlphaGPT\data\ashare.duckdb`（只读零写入，AGENTS §5.6）。
- 环境：Windows / Python 3.13.12（miniconda；与项目 .venv 的核心数值栈逐版本一致：numpy 2.5.2 / pandas 3.0.5 / duckdb 1.5.5）/ PYTHONDONTWRITEBYTECODE=1；墙钟 **19,580.4s（5h26m，单机串行；与队友 P14 bench 部分并发；资源上限 p13 §9.1 的 30 分钟为参考环境口径，本机 v2 6.3GB 库 + 团队负载下如实记录，未缩窗口）**。
- 口径（预注册逐项）：h∈{1,2,3,5,10,15,20} 非重叠 IC（`reward.rank_ic_series` 单一路径）；min_stocks=10；IS<20220101 / OOS≥20220101；成本 = `cost_matrix.round_trip_cost`@10万/20仓，1x=共享费率、2x=费率×2；标签 = adjusted open-to-open（p19 §2.3，NaN-preserving adjusted cache）；因子输入 = adjusted 基座（p19 §2.3），volume/amount/turnover_rate raw；相关簇 = 注册表口径 |ρ|≥0.9 union-find（pooled tensor z-score Pearson）+ 注册表实现日期子样交叉核对；**五个既有 walk-forward folds** = ProtocolConfig 预注册 (2020-12-31→2021-12-31)…(2024-12-31→2025-12-31)（逐 fold 测试窗 IC 已入库）；**seeds [42,7,2024]** = 预注册协议种子（本审计确定性：rank IC 无 RNG、无搜索 → **唯一语义预算消耗 = 0**）。
- 版本组（被测树实测）：protocol 26 / reward 15 / model 3 / grammar 6 / **factor_compute 3** / feature_registry 6 / **data_tier 2** / research_domain 3 / search_contract 4 / execution_spec 3 / portfolio_constructor 1 / rebalance_policy 2。
- 产物：`docs/factor_inventory_audit_v2_20260905/metrics.json`（77 因子 × 7 h 全量 + nonoverlap + 逐年(h1/h10/h20) + 逐 fold + 方向/层级审计块 + 77×77 相关矩阵 + 簇 + 换手 1x/2x + 增量 OOS）+ 本日志。运行一次成功，无失败产物混入。

## 1. 窗口披露（与预注册口径的差异，如实记录）

加载窗 = v2 数据全集 **20140102..20260904（3,083 会话 × 1,630 码）**。既有 p9/p13 口径的"窗口 2015-01-05 起"反映的是 legacy 库数据范围；p17 §11.3 预注册"研究评价起点 2015-01-01 不变"——评价侧（OOS=2022 分界、五 folds 锚点 2021-2025）全部日期锚定且不受影响；2014 年增量仅进入 IS 侧统计（IS 起点未在任何契约中钉死）。裁决阈值零改动。跨窗口与 v4 数值的对比一律标注语义版本差异（captain 简报注意事项①），不做值漂移误判。

## 2. 数据降级审计（run-3 实测，全部留档）

1. **adj_factor 缺失/非正（重大）**：2,314 个 (code,date) bar 行无有效 adj_factor（占 11.89M bar 行 ≈0.02%，分散于 2014-2026）→ 按 p19 §2.1 **整股 fail-closed：1,093/1,630 码全调整序列 NaN**（标签全 NaN、价格类因子中性化）→ **有效测量横截面 ≈ 537 码**（target finite fraction ≈0.234 与之一致；min_stocks=10 满足）。p19 fail-closed 语义的真实后果，非测量缺陷。**已报 captain 建议 D3 数据勘误补齐 2,314 行后重测复核**；本日志全部结论以此为条件。
2. daily_price_limit 缺 10,701 个 (code,date) 行（2025+ 为零）→ 537 码"≥1 会话 limit 缺失"降级标记；limit 价仅执行约束输入、不进因子公式 → 因子测量不受影响。
3. **申万行业指数收盘滞后 15 个开市会话 > 容差 3**（p17 §12）→ loader `industry_source_degraded`（legacy sw_index 源未同步至 v2 数据尾）。行业族新鲜度门 FAIL，如实进入 Tier 审计。
4. 原始缺失度口径限制（测量工具面，如实披露）：raw coverage/exposure 在 1,093 只 adj 降级码上低估纯 raw 输入特征（turnover/volume 族实测 cov 0.235，其 as-consumed 真值 ≈1.0）——共享 `normalize_daily_bars` 对 NaN open/close 行的丢弃所致；IC/相关/换手/增量全部骑 loader 张量，不受影响；受影响数值仅 coverage/exposure 两块，裁决不依赖它们。

## 3. 全量测量摘要

- pooled 非重叠 IC（全库）：h1 −0.0127 / h2 −0.0127 / h3 −0.0133 / h5 −0.0144 / h10 −0.0154 / h15 −0.0170 / h20 −0.0183。
- 7 信号等权基准（IS 拟合复合基线，h=10）：IS IC +0.0108 / **OOS IC −0.0179**（IS 系数 w_b = −0.000391 <0，见 §5 退化性发现）。
- 覆盖率分层：价格类因子 ≈0.235（受 §2.1 降级约束）；财务 legacy 0.74-0.87；family-⑤ v2 as-of **0.92**；margin 0.80-0.82；LIMIT_* 事件 1.00；行业族 **0.031-0.033**（见 §5.2）；NORTHBOUND_CHG 0（中性占位）。

## 4. 契约级发现（裁决前置缺陷，逐项留证）

1. **ΔIC_OOS 判别腿退化（p9 §7.2 / p13 §8 族门）**：IS 拟合给基线系数 w_b = −0.000391 <0 → 复合 = |w_b|·(−B) 把 OOS 基线 IC 翻号（−0.0179 → +0.0179）。**零信号证明**：INDUSTRY_MOMENTUM（cov=0.000，w_f=0.0）ΔIC_OOS = +0.0357 ≠ 0 —— 纯基线翻号伪影。全体特征 ΔIC ≈ +0.035±0.005（无判别力）。族门的 ΔIC≥+0.005 腿本轮**不可执行**；判别腿仅剩"残差 IC 符号 = 预注册方向"。**处置：族门整体不执行翻转（见 §5），ΔIC 退化性作为 p9 §7.2 契约勘误候选上报**（t18 任务书预注册语 Sentence"相对**方向对齐**基准"与冻结工具的未对齐复合基线不符——修复属契约勘误，非本任务可擅改）。
2. **D5 消费路径缺口（t22 辖区）**：`factors.py:663 _industry_demean` 仅消费 `ctx.industry` **第 0 列（窗口首日 20140102 的行业）**——legacy 快照源下 ≈"当前行业"近似成立；D5 PIT 区间源下，in_date>2014 的股票（含 2021 SW 换代 935 行）在首列为 NaN → 行业族 coverage 崩塌至 0.031-0.033、INDUSTRY_MOMENTUM 0.000。**本测量有效反映现行为**；修复（逐日列消费）= t22 原位修复，修复后须重测再裁决。
3. **降级簇伪影警示**：SUSPEND_DAYS_60 ~ LIST_AGE pooled ρ = **−1.000**——由 1,093 只降级码的共同聚类（LIST_AGE→中性 0、SUSPEND_DAYS_60→缺失计数极大值）主导，**非真实语义镜像**；Part A 弃用机制在其前提（两真信号冗余）不成立时不得执行。已作为 D3 勘误依赖项记录，修复数据后须复检。

## 5. 四类重新裁决（预注册规则执行；每因子结论 + 证据）

裁决总规则：只执行预注册裁决（p9 §7 / p13 §8）；发现 §4.1 的 ΔIC 腿退化后，族门不产生任何 promotion 翻转（维持现状 = 唯一可执行结论）；负结果/现状维持的全部证据保留于 metrics.json。

### 5.1 P13 四个基本面因子（family-⑤，v2 三表 as-of，Tier B）——**全部维持**

| 因子 | cov | IC10 | ICIR10 | ΔIC_OOS | 残差OOS | 预注册方向 | 符号一致 | stab10 | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| CASHFLOW_QUALITY | 0.917 | +0.0099 | +0.077 | +0.0410* | +0.0096 | +1 | ✓ | 0.54 | **维持** promotion_allowed=True |
| ACCRUALS | 0.920 | −0.0139 | −0.112 | +0.0470* | −0.0225 | −1 | ✓ | 0.46 | **维持** |
| ASSET_GROWTH | 0.923 | −0.0043 | −0.026 | +0.0409* | −0.0101 | −1 | ✓ | 0.46 | **维持** |
| EARNINGS_ACCEL | 0.916 | +0.0156 | +0.134 | +0.0384* | +0.0142 | +1 | ✓ | 0.85 | **维持** |

\* ΔIC 腿退化（§4.1），数值仅留档不作为门判据；判别证据 = 四成员残差符号全部与预注册方向一致 + h20 ICIR（CQ +0.232 / ACCRUALS −0.273 / EA +0.249）+ 覆盖 0.92。族门若按字面执行 = PASS（与维持结论一致）。**正向结果，无 registry 翻转**。

### 5.2 行业相对族（IND_REL_RET_5/20/60/120、IND_REL_VOL_20、IND_REL_TURNOVER、INDUSTRY_MOMENTUM）——**降级：本轮裁决不成立，不得进入正式 A+B 候选**

- 证据：coverage 0.031-0.033（INDUSTRY_MOMENTUM 0.000、其 IC 全 None）；Tier C（p17 §7，t14 诚实锁定）；新鲜度门 FAIL（§2.3）；消费路径缺陷（§4.2）。测得的 IC/ΔIC 为 3% 覆盖噪声，**不可作为族门判据**（对 3% 覆盖信号执行 ΔIC 门 = 测量表演）。
- 结论：七因子全部**降级**（从可裁决集合移出；registry flags 不翻转——翻转建立在无效测量上同样违规）。修复路径：t22 原位修复消费路径 + 新鲜度门收口 + 数据勘误后重测再裁决。

### 5.3 融资拥挤族——**MARGIN_CROWD_60 维持；MARGIN_BALANCE_CHG 维持（方向证伪证据留档）**

| 因子 | cov | IC10 | ICIR10 | ΔIC_OOS | 残差OOS | 方向 | 符号一致 | stab10 | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| MARGIN_CROWD_60 | 0.801 | −0.0249 | −0.225 | +0.0454* | −0.0238 | −1 | ✓ | 1.00 | **维持** promotion_allowed=True（残差符号+13/13 年方向一致+ICIR 联合支持；Tier B 晋级仍受 Tier-A-only 政策约束） |
| MARGIN_BALANCE_CHG | 0.816 | −0.0263 | −0.237 | +0.0523* | −0.0314 | +1 | ✗ | 0.08 | **维持**（无预注册翻转规则）；** authored +1 方向被证伪**（IC 为负、仅 1/13 年同向）→ t22 语义修复候选（方向重预注册），证据留档 |

### 5.4 exact-limit 涨跌停事件族——**全部维持现状（family-③ 负结果判定继续成立；既有四因子维持）**

| 因子 | cov | IC10 | ICIR10 | 残差OOS | 方向 | 符号一致 | 现状 | 结论 |
|---|---|---|---|---|---|---|---|---|
| LIMIT_UP_EVENT | 1.000 | −0.0070 | −0.041 | −0.0047 | +1 | ✗ | promo=True | **维持**（覆盖 1.0 = F1 修复红利保持；方向证据弱） |
| LIMIT_DOWN_EVENT | 1.000 | −0.0802 | −0.554 | +0.0404 | −1 | ✗ | promo=True | **维持**（原始 IC 强烈支持反转语义但残差符号不符——基线翻号污染下不动 flags） |
| LIMIT_UP_CNT_20 | 1.000 | −0.0240 | −0.182 | −0.0055 | −1 | ✓ | promo=True | **维持** |
| LIMIT_BREAK | 1.000 | −0.0163 | −0.165 | +0.0026 | −1 | ✗ | promo=True | **维持** |
| LIMIT_UP_CNT_5 | 1.000 | −0.0246 | −0.169 | +0.0033 | +1 | ✓ | promo=**False** | **维持 False**（族③负结果判定继续成立；残差 +0.0033 为边际弱证据，不足以在 ΔIC 腿退化下恢复晋级——恢复须待契约勘误后重测） |
| LIMIT_DOWN_STREAK | 1.000 | −0.0802 | −0.554 | +0.0404 | −1 | ✗ | promo=**False** | **维持 False**（同族③） |
| LIMIT_BREAK_5 | 1.000 | −0.0223 | −0.208 | +0.0041 | −1 | ✗ | promo=**False** | **维持 False**（同族③） |

### 5.5 相关簇与二次精简（0.9 阈值；活特征对全部复核，零 registry 翻转）

| 对 | ρ | 处置 | 依据 |
|---|---|---|---|
| SUSPEND_DAYS_60 ~ LIST_AGE | −1.000 | **不弃用** | §4.3 降级簇伪影（1,093 码聚类主导），Part A 前提不成立；D3 勘误后复检 |
| LIMIT_DOWN_EVENT ~ LIMIT_DOWN_STREAK | +0.996 | **不弃用** | p9 测量日志 §2.3 对同一对已有"保留计算、不再另行精简"的显式在案裁决（0.985→0.996 为同关系强化）；翻案须勘误 |
| TURNOVER_MA20 ~ TURNOVER_STD20 | +0.919 | **不弃用** | p9 §4.1 显式 keep-list（MA20=慢代表、STD20=异质语义）；推翻显式 keep 须先修契约；0.919 作为勘误候选留档 |
| RET_1 ~ INTRADAY_RET | +0.908 | 不弃用 | p9 §4.1 预注册豁免（隔夜/日内构造性原子对），0.902→0.908 关系稳定 |
| 其余 ≥0.9 对（13 对） | — | 不适用 | 全部涉及已 deprecated 成员（Part A 无行动），清单见 metrics.json `high_corr_pairs_0p7` |

## 6. 逐因子方向稳定性、换手与成本（摘要；全量见 metrics.json）

- 方向稳定性（年度同向占比，h=10）：HIGH 稳定组 = MARGIN_CROWD_60 1.00、SUSPEND_DAYS_60 1.00、TURNOVER_MA20/STD20/TURNOVER 0.92、LIMIT_UP_CNT_20 0.92；LOW 组 = MARGIN_BALANCE_CHG 0.08、LIST_AGE 0.00（方向证伪）。
- 换手/成本：direction-implied 五分位篮子 tau_h1/h5/h10 与 1x/2x 年化成本拖累逐因子入库（metrics.json `per_factor[*].turnover`；2x = 1x 拖累×2）。慢速族（财务/行业）低换手（tau_h10 ≈0.02-0.09），事件族高换手高退化篮子占比（quintile_degenerate_share 如实入库）。
- 逐 fold IC（五个预注册 walk-forward 测试窗，243/242/242/242/243 会话）：逐因子 × 7 h 全量入库（`per_factor[*].fold_ic`）。

## 7. 运行统计与偏差

- 三次运行记录：run-1（db76700+0a8565e）在 loader 崩溃（D6 缺陷，fail-closed，见 §8）；run-2（+D6 修复+钉值）于 loader 完成后死于测量工具 UnboundLocalError（本地 import 遮蔽，commit 53a675b 修复留证；logs/t18_audit_run2.log）；run-3 成功（本日志被测对象）。全部失败证据保留，零覆盖。
- 控制台 stderr：1,630 条 price-degraded WARNING（数据事实）+ 1 条 industry freshness WARNING（数据事实）；无新增代码 warning。
- 验证：全量串行 pytest @ 53a675b = **1,680 passed / 1 skipped / 0 failed（24:12，项目 .venv；logs/t18_full_suite_venv.out）**；miniconda 环境跑出 2 个 test_lock_files 失败为环境口径差异（该环境次要依赖 pin 与 lock 不一致；核心数值栈逐版本一致），如实记录不以 --dev 规避。`compileall`、`git diff --check` 在提交时执行（见提交记录）。

## 8. 首跑阻塞与 D6 勘误 #4（背景，证据链）

首跑（db76700）在 loader `build_pit_frames → _family5_asof_frames → _shift_report_date` 崩溃：真实 `income_statement_pit` 含 10 行非季末 end_date（20170131/20160430/20190831/20170531/20200430/20151130/20171031/20190731；码 003040.SZ/300959.SZ/301001.SZ/301038.SZ/301200.SZ/301217.SZ/301256.SZ/688063.SH/688772.SH/688779.SH）。fail-closed 拦截（零错误数据）。captain 派勘误 #4（非季末行不参与季度 TTM 算术、NaN+审计计数）+ D6 修复 4b065c0；本日志被测树含其 cherry-pick c8d0043，端到端验证通过（family-⑤ cov 0.92）。

## 9. 上报与后续（不属本任务执行项）

1. **D3 数据勘误候选**：2,314 个 adj_factor 缺行 → 1,093 码整股 fail-closed（p19）；补齐后建议复测（尤其 §5.5 的 SUSPEND~LIST_AGE 伪影与有效横截面恢复）。
2. **p17/D5 消费缺口（t22 辖区）**：`_industry_demean` 窗口首列消费 → 行业族须原位修复后重测。
3. **p9 §7.2 勘误候选**：ΔIC 基线翻号退化（方向对齐基线，t18 任务书语句子以预注册支持）。
4. **p9 §4.1/测量日志勘误候选**：TURNOVER_MA20~STD20 0.919（keep-list 与新共线的冲突）。
5. **MARGIN_BALANCE_CHG 方向重预注册候选（t22）**：+1 方向被证伪（0.08 稳定度）。

本日志只记实际发生之事；预注册内容见 p9/p13/p17/p19 契约与 t18 任务书。测量结果未反向改写任何契约或阈值；registry flags 零翻转（所有候选翻转均被在案裁决/伪影/退化性阻断，证据全部保留）。
