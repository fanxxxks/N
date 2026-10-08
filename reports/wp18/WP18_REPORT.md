# WP18 工程报告（WP18_REPORT.md）——固定计数分阶段研究复测：交付与停止表裁决

- 生成：2026-09-27（报告撰写员）。契约 = `reports/wp18/FROZEN_18_PLAN.md`（运行前冻结、本包只读）；五问与全部数字的逐行依据 = `reports/wp18/ANALYSIS.md`（本报告引用其章节与行键，不重复推导）。
- **全部 2021–2025 结果为开发验证（development validation）**：不作样本外结论、不作统计显著性结论发布（契约 §11；ANALYSIS §10：本包未生成任何 ClaimRecord、未运行 max-t 资格审定，全部结论为开发验证下的描述性比较）。
- 方法修订守卫未触发：`ashare_model/method_revision_registry.json` 读取核实 `{"episodes": []}`（本报告会话内亲读）。
- 本报告撰写会话执行的核查命令清单见附录 A；凡标注"本会话亲验"的数字均为撰写时重新运行/重读所得。

---

## 1. 执行概览

### 1.1 各阶段行数对账表（冻结预算 预期 vs 实测）

冻结预算表 = FROZEN_18_PLAN §4；实测来源 = `reports/wp18/stage12_result.json`、`reports/wp18/{medium,slow}/run_meta.json` 与各产物 JSON。本会话亲验（重读各 JSON 的 row_count/failed/entries）：

| 阶段 | 内容 | 预期（冻结） | 实测 | 失败行 | 亲验来源 |
|---|---|---|---|---|---|
| M1 | medium 方向证据（3 单因子×2 规则×5 折） | 30 | **30** | 0 | `stage12_result.json` per_stage_rows.M1 |
| M2 | medium 纯信号摘要（4×2×5） | 40 | **40** | 0 | 同上；`medium/signal_summary.json` row_count=40 |
| M3 | medium 账户评价（80+控制 10，1x） | 90 | **90** | 0 | `medium/account_report.json` row_count=90、failed_rows=0（本会话亲读） |
| S1 | slow 方向证据（5×2×5） | 50 | **50** | 0 | `stage12_result.json` per_stage_rows.S1 |
| S2 | slow 纯信号摘要（6×2×5） | 60 | **60** | 0 | 同上；`slow/signal_summary.json` row_count=60 |
| S3 | slow 账户评价（120+控制 10，1x） | 130 | **130** | 0 | `slow/account_report.json` row_count=130、failed_rows=0（本会话亲读） |
| C1 | medium 条件性成本压力（≤40） | ≤40 | **20**（仅规则 B 过初筛） | 0 | `medium/cost_stress.json` row_count=20（本会话亲读） |
| C2 | slow 条件性成本压力（≤40） | ≤40 | **0**（A/B 均未过初筛，跳过理由在案） | — | `slow/cost_stress.json` row_count=0；run_meta stress_screen 两条 skip_reason |

会计口径核对（契约 §4.6 写死）：

- 测量行：方向证据 80 = 30+50 ✓；纯信号摘要 100 = 40+60 ✓（不占 300 预算）。
- **正式评价行两域合计 = 90 + 130 + 20 + 0 = 240 ≤ 300** ✓（`{medium,slow}/run_meta.json` formal_row_total=240，两文件本会话亲读）。
- 算法搜索预算 = 0：无 GP/搜索器调用、无权重拟合、无新因子（`combo:slow_all5` 为唯一预注册新 ID，等权、只用现有五成员，契约 §1.3）；run_meta `algorithm_search_budget: 0`。
- 方向红线：全部带方向的账户行（medium 80、slow 120）与全部 20 条压力行 `direction_source="frozen_book(no_relearn)"`（本会话对 `medium/account_report.json` 90 行、`slow/account_report.json` 130 行、`medium/cost_stress.json` 20 行逐一计数核实；10+10 条控制行按设计无方向字段），消费各自 17:49/对应时刻冻结的方向书（`direction_frozen.json` frozen=true，entries 30/50，本会话亲读）。

完整性对账（WP14 experiment_integrity，ANALYSIS §0.2）：

- 阶段 1+2 子集：medium 70/70、slow 110/110，0 缺失/重复/意外/失败（`integrity/reconcile_stages12.json`）。
- 阶段 1+2+3：medium 160/160、slow 240/240 全零（`reconcile_stages123.json`）；含压力行（stages123c）180/180、240/240 全零。
- 全量计划 reconcile：medium missing=20、slow missing=40、重复/意外/失败均 0（`reconcile_full.json` 顶层标量，本会话亲读）——缺失恰为未获初筛授权的 C1:A 行与 C2 行；`status_full.json` label=`measurement_incomplete`（两域，本会话亲读）是**初筛拒绝授权的设计内状态**，不是测量失败，不构成经济负结果的证据。
- 预检：`reports/wp18/precheck.json` ok=true（两域）。

### 1.2 运行时（记录值，来源 run_meta / stage12_result）

| 腿 | 运行秒数（记录） | 上限（冻结 §10.1/§10.2） |
|---|---|---|
| medium 阶段 1+2 | 279.3 s | 3600 s |
| medium M3 | 455.6 s（`medium/run_meta.json` m3_elapsed_sec） | 计入域门 |
| medium 累计（阶段 1+2+M3） | 279.3+455.6 = **734.9 s ≪ 3600 s** | 3600 s |
| slow 阶段 1+2 | 536.1 s（`stage12_result.json` runtime_sec.slow） | 7200 s |
| slow S3 | 470.1 s（`slow/run_meta.json` s3_elapsed_sec） | 7200 s |
| C1 | 384.7 s；C2 0.1 s（0 行运行） | 计入所在域 |

因子缓存：medium hit（data/r1/medium_a）；slow 经 §9(b) 只读复制后 hit（首次误用默认 data_dir 的 miss 事件见 §1.4 契约记录与 errata E3）。

### 1.3 域门结果（medium 门通过——实际执行形态的如实披露）

- **工作流级判定：medium 域门通过，slow 域获准执行**（判定输入=medium 执行员 StageResult：90/90 行、0 缺失/重复/失败、状态非 measurement_incomplete、运行时 734.9 s 累计口径 ≤ 3600 s 上限；slow 阶段确实执行，S3/C2 产物在盘）。
- **errata E4 披露（运行后审计，append-only 在案）**：冻结顺序（medium 全部阶段 → 门 → 任何 slow 阶段）**未逐字执行**。真实时序（文件 mtime + 账本时间戳，本地 UTC+8）：medium 阶段 1+2 16:23–16:38 → **slow 阶段 1+2 于 16:57–17:49 先行运行**（slow 方向书 17:49:05 冻结）→ medium M3 18:13 → medium stages123 对账 18:14:34 → S3 18:30:54–18:32:07 → C1/C2。即：冻结形态的门（medium 全域后再进 slow）从未执行；slow 启动前实际可用的判据只有 medium 阶段 1+2 子集门（279.3 s ≪ 3600 s，`stage12_result.json` runtime_sec.domain_gate 17:54:39 落盘，晚于 slow 阶段 1+2 完成）；对 S3 腿存在事实上的第二道检查点（S3 于 medium stages123 全量对账落盘之后才运行）。缓解已核实：任何口径下运行时都远低于上限、已执行阶段对账全零（E4 mitigations_verified）。
- 处置：FROZEN_18_PLAN §8 要求的域门账本登记以 **post-hoc 补记条目**补上（`domain_gate:backfill`，明标 recorded_post_hoc=true，经账本 API 追加）；ANALYSIS §0.2/§11 已改写为真实时序（E4 fix_actions）。用户须知：任务简报"先跑一个域再决定"仅对 slow 阶段 1/2 腿以部分（子集门）形式满足；S3 腿确实位于 medium 全域对账之后（E4 user_spec_note 逐字保留于 errata）。

### 1.4 契约裁决记录（醒目留痕，可被用户推翻回退）

`stage12_result.json` contract_adjudication_record + `reports/wp16/FROZEN_A.md` §8.3 + errata E1/E2/E9（ANALYSIS §0.3 汇总）：

- **裁决（2026-09-27）**：slow Rule A 评分被 WP16-A (C1) 守卫的浮点 ε 区间（缺口 1–2e-12）误拒 19/25 键；最小物理费用比例 5e-5，ε 违规量低其 7 个数量级 → 认定规格缺陷，授权窄幅勘误（仅 19 个失败键重学）。
- **落地（窄于授权字面并经追认）**：`_P52_C1_TOLERANCE` 保持 1e-12（成交记账精度逐字节不变）；新增 `_P52_C1_REFUSAL_TOLERANCE=1e-9` 仅用于最终拒绝比较；31 个已成功键与 medium 全部冻结件不动（防结果选购）。
- **追认（E9，运行所有者级决定，用户仍可推翻）**：含落地形态偏差一并追认，审计关闭、现状保留、不回退。依据三条：20 条 attempt-1 失败行全部为 fail-closed 拒绝、无方向产出可选购，attempt-2 属测量缺陷修复后的补全；双常量拆分比授权字面更符合 WP16-A 本意；程序完整（FROZEN_A §8.3 只追加、红证据 2026-09-27 节、回归全绿）。**若推翻，回退动作清单**见 errata E9 standing（恢复 slow 方向书 31 键 partial、slow 账户/压力行按停止表行 1 停止经济解释、E2/§8.3 勘误作废、重走裁决）。
- **E1（对账前修复）**：WP18 双实验 ID 曾共享报告 schema 键导致自拒，改为每域独立键后任何 reconcile 运行前修复（severity 注明 fixed_before_any_reconcile_ran）。
- **E3（缓存纪律披露）**：首次 slow 运行在驱动拒绝前，`load_data` 内部冷重建（~52 min）向既有 `data/r1/slow_ab/factor_tensor_cache/` 追加一个字节相同的新条目（sha256 81db3d19…）；无既有文件被改；账本 `budget:factor_cache_miss_correction` 更正在案。

---

## 2. 五问答案（引 ANALYSIS 行键证据；全部为开发验证-描述性）

**Q1 固定标签上的 IC、分位排序与有效覆盖是否显示信息？**（ANALYSIS §1，行键 `M2:{A|B}:None:{候选} × fold_test_end` / `S2:…`）
medium 域信号轨显示可检出信息且集中在 B 择向：B 方向单因子 14/15 格-折 IC>0（量级 .006–.079）、组合 5/5 折为正、分位差与 IC 同号 18/20 行-折、覆盖 ~0.99–1.00、N_eff≈800；A 择向单因子仅 6/15 为正。slow 域信号阶段即偏弱（B 择向单因子 11/25 格-折为正；组合 B 3/5、均值 −.006）。**测量限制**：DIVIDEND_YIELD 有效覆盖仅 0.291–0.361，其"无信息"读数受覆盖不足限制，不得与其它因子同权解读（ANALYSIS §1 答案 2）。0x 压力行不用于本问（两轨纪律）。

**Q2 A/B 差异来自评分规则，还是取证窗口、参考配置变化？**（ANALYSIS §2）
见 §3 归因结论：差异由评分规则本身产生；窗口与参考配置逐格冻结相同（derived.json `all_same_train_cutoff=true`，medium 15/15、slow 25/25，本会话亲读）。

**Q3 动态交易相对首次持有是否有净增量？**（ANALYSIS §3，行键 `M3:{A|B}:{dynamic|hold}:{候选} × fold_test_end`）
相对增量普遍存在但都是"少亏"：20 个候选-规则格中 18 个 dyn−hold>0（且全部 ≥+0.90pp，超过 50bp 门槛；例外 slow ROE/B −0.25pp、slow combo/B −0.95pp）；但两腿年化净超额 20 格中 19 格为负（唯一正值 slow PE_TTM/B +1.94%）。主组合几何累计净超额（折级权威口径）：medium A −9.80%/B −3.81%、slow A −26.54%/B −21.49%（与 `cost_stress.json` screen_adjudication 逐字一致，本会话亲读 medium −0.098043/−0.038089）。机制注记：slow 2021 折 A 的 dynamic 与 hold 逐字节同路径（初始建仓只成交 5/20，再平衡全被抑制，invested_ratio_mean=0.149）——预算约束使动态腿在该折退化为持有。

**Q4 超额中有多少来自实际现金暴露而非选股？**（ANALYSIS §4；重放产物 `reports/wp18/_analysis/replay_{medium,slow}.json`）
medium：负超额以现金暴露为主，B 择向选股实际为正——组合 B 选股效应 +21.8pp（算术累计）但暴露效应 −29.5pp、费用 −1.6pp → 净 −9.4pp，即超过 100% 的负净超额来自未投资现金（实际投资比 0.40–0.71；min_trade/整手/预算抑制每折 137–578 单）。slow：负超额以选股为主（组合 A 选股 −84.2pp、暴露 +49.5pp——低配一个下跌持仓簿的"少亏"）。分解基于**实际成交后持仓**重建（重放 daily_returns 与存储行逐字节相等，毛收益恒等式最大误差 6.3e-16/8.7e-16；ANALYSIS §0.4）。

**Q5 成本变化是直接费用，还是现金/整手/预算的后续路径变化？**（ANALYSIS §5，两口径分表）
在本测量的低换手（到期会话均值 0.5–1.4%）下，成本变化的作用**几乎全部是直接费用**：固定路径核算的直接费用效应 0.12–0.32pp/年（量级=费用/本金）；重新执行 vs 固定路径的**路径效应 ≤0.10pp/年，2x 档≈0**（换手基本不动、invested_ratio_mean 变动 ≤0.015）。两口径分表报告、未合并；0x 行不用于信号轨结论。

---

## 3. A/B 归因结论（开发验证-描述性）

**结论：A/B 差异来自评分规则本身（验证 reward 比较秩 vs 训练切片标签 IC 符号），不是取证窗口或参考配置的"变化"。**证据（ANALYSIS §2）：

1. 同折同标签：每格 A 与 B 的 train_cutoff 完全一致（medium 15/15、slow 25/25，`_analysis/derived.json` q2_direction_books，all_same_train_cutoff=true 本会话亲读）；取证窗口是两规则各自身份的固有部分（A=训练窗尾部验证子窗；B=验证段之前的训练切片），并非运行中"变化"。
2. 方向分歧率高：medium 10/15（66.7%）、slow 16/25（64.0%）（derived.json n_disagree/n_cells，本会话亲读）；分歧格中两规则引用不同统计量——窗口与配置相同，差异由评分规则产生。
3. **配置敏感性分离（p46 交叉验证，独立运行、不同参考格 n10_m2500）**：三共享因子×5 折，B 方向 15/15 完全一致（构造上配置不变量，经验证实）；A 方向仅 8/15 一致（7 格随配置翻转）。参考配置若变，受影响的是 A，不是 B。
4. 事后对齐（描述性）：以测试窗带向 IC 符号为参照，A 方向 medium 单因子 6/15、slow 7/25 与之一致；B 为 14/15、11/25——A 的"验证 reward 择向"在本测量期更经常选到测试窗 IC 的反侧。
5. 账户后果：B 优于 A 的候选 medium 3/4、slow 3/6（年化净超额，dynamic，1x）；§5 初筛唯一通过的是 medium/combo:all3/B。
6. 处置：按停止表行 4，保留此负结果，**不试探新的方向规则**；`B-fulltrain` 已在契约 §2.3 预注册否决（全仓 grep 零命中，非既有物）。

---

## 4. 机械筛选判定与去最大年份结果

**标注（逐字，每处判定重复）：机械筛选门槛是研发资源分配门槛，非统计显著性证明也非市场规律。**判定单位=（域, 主组合, 规则），依据=1x 账户行拼接 2021–2025，主策略腿=dynamic；来源 `reports/wp18/{medium,slow}/cost_stress.json` screen_adjudication（本会话亲读 medium A/B 判定与累计值）。

| 判定单位 | (i) ≥3/5 年正 | (ii) 中位正 | (iii) dyn−hold ≥50bp | 裁决 | 后续 |
|---|---|---|---|---|---|
| medium combo:all3 A | ✗（2/5） | ✗（−0.64%） | ✓（+3.19pp） | **不过** | 0 行压力（skip_reason 在案） |
| medium combo:all3 B | ✓（3/5） | ✓（+0.23%） | ✓（+1.27pp） | **通过** | C1 20 行（0x/2x×两模式×5 折） |
| slow combo:slow_all5 A | ✗（1/5） | ✗（−5.72%） | ✓（+2.66pp） | **不过** | 0 行 |
| slow combo:slow_all5 B | ✗（1/5） | ✗（−6.19%） | ✗（−0.95pp） | **不过** | 0 行 |

主组合逐折净超额（dynamic）：medium B [+1.09, +9.22, +0.23, −7.50, −6.03]%；medium A [−5.46, +0.34, +2.42, −0.64, −6.56]%；slow A [−17.11, +5.06, −4.11, −5.72, −6.69]%；slow B [−7.50, +1.31, −4.10, −6.19, −6.88]%（ANALYSIS §6）。

**去掉贡献最大年份后（避免只看一行平均数；ANALYSIS §7，来源 screen_adjudication.*.screen.ex_minus_max_year）**：

| 判定单位 | 剔除年（其净超额） | 剩余 4 年正年数 | 剩余中位 |
|---|---|---|---|
| medium combo:all3 A | 2023（+2.42%） | 1/4 | −3.05% |
| medium combo:all3 B | 2022（+9.22%） | 2/4 | −2.90% |
| slow combo:slow_all5 A | 2022（+5.06%） | 0/4 | −6.20% |
| slow combo:slow_all5 B | 2022（+1.31%） | 0/4 | −6.54% |

四个判定单位去掉最大贡献年后**无一满足 (i)/(ii)**——无单年依赖的幸存者；medium B 的 3/5 正年中 2022（+9.22%）贡献了全部正读数的大半。

**描述性参考（非判定单位）**：全部 20 个候选-规则格过三条件者仅 3 格——medium MOMENTUM_20/A、medium combo:all3/B（判定单位）、slow PE_TTM/B（`_analysis/derived.json` screen_descriptive.passed，本会话亲读逐一核对）。单因子格不是 §5 判定单位，仅作资源分配参考，不得表述为"通过筛选"。

---

## 5. 停止/继续条件表逐行裁决（FROZEN_18_PLAN §6.1 逐字条件；ANALYSIS §8）

| # | 条件（逐字） | 裁决 | 证据（来源） | 下一步动作（按表，不越权） |
|---|---|---|---|---|
| 1 | 主要臂缺失/程序失败/日期或价格口径不一致 → 停止经济解释，修测量完整性 | **未命中** | 全部主臂物化且 0 失败（§1.1）；reconcile stages12/123/123c 全零；所有行同一 `evaluate_signal` 引擎路径与同一 PIT/价格口径；`status_full=measurement_incomplete` 仅因初筛拒绝授权的压力行未运行（设计内） | —（测量完整性成立，经济解释允许进行） |
| 2 | 纯信号有信息但账户执行后消失 → 只处理已定位的执行问题，不扩搜索 | **命中（medium）** | medium 信号轨 B 择向单因子 14/15 格-折 IC>0（ANALYSIS §1），账户年化净超额全负（§3 表）；已定位的执行问题=现金暴露（实际投资 0.40–0.71）+ min_trade/整手/预算抑制；slow 信号阶段本身弱，该行在 slow 不完全成立 | 只处理已定位的执行问题（现金暴露/抑制结构属后续包的测量议题）；**不扩搜索**——本包零搜索、零新因子、零权重拟合，已兑现 |
| 3 | 持有价值、动态没有增量 → 转向静态选股/持有假设，停止宣称动态选股有效 | **未命中（反向）** | hold 腿普遍更差：18/20 格 dyn−hold>0；两腿净超额同为负——"持有有价值"不成立（ANALYSIS §3 答案 3） | 本包同样**不**宣称动态选股有效（无绝对增量）；不转向持有假设（前提不成立） |
| 4 | A/B 对结果影响很大但都无稳健增量 → 保留负结果，停止继续试探方向规则 | **命中** | 分歧率 66.7%/64.0%（§3）；账户后果分化（medium B 过初筛 A 不过，年化差最大 7.4pp）；20 格年化净超额 19/20 为负；去最大年后无幸存（§4） | **保留负结果**（本报告+全部产物在盘）；**停止试探方向规则**（B-fulltrain 已预注册否决，未试探任何第三规则） |
| 5 | 可执行、拼接净超额与 active IR 为正、年度较一致、动态优于持有、成本压力下仍有空间 → 才考虑有限 GP/Random 对照 | **未命中** | 四判定单位拼接净超额几何全负（折级权威口径 −9.80%/−3.81%/−26.54%/−21.49%；errata E5 更正后值）；active IR 多折为负（medium B 组合 [+0.08,+0.64,−0.03,−0.66,−0.62]，3/5 折负，derived.json 本会话亲读）；年度不一致（§4 逐折） | **不启动 GP/Random 对照**（搜索预算保持 0；本报告不越权启动任何对照） |
| 6 | 两域在有效、可比的测量下均无增量 → 暂停这批假设，新经济假设或新信息源才值得重启 | **命中** | 两域判定单位 (i)(ii) 全不过（§4）；测量有效性由行 1 背书（完整、可比、同引擎同口径） | **暂停这批假设**（medium 3 因子 + slow 5 因子及其等权组合、A/B 两择向、参考格 n20_m5000/配置块）；重启条件=新经济假设或新信息源 |

**总裁决：停止/暂停**（行 4 + 行 6 命中；行 5 未命中故无对照搜索）。负结果按契约 §11 作为本包产出保留。

---

## 6. 试错登记汇总（不只登记最终幸存者）

账本 = `<data_dir>/runs/<spec_id>/<run_id>/trials.jsonl`（追加式哈希链 JSONL）。本会话逐一计数 7 个账本（行数含 record+complete 成对行）：

| 账本 | 行数（本会话亲数） | 内容（ANALYSIS §11） |
|---|---|---|
| `data/r1/wp18_medium/runs/wp18_preregistration/.../trials.jsonl` | 18（=9 试验） | 9 条预注册决策：参考格 n20_m5000（含对简报 n10_m1000 的否决理由）、零搜索否决、Rule B 正式命名铸造、B-fulltrain 不保留、A/B 主比较纪律、combo:slow_all5+分散化假设、combo:all3 复用、hold 实现选择、正式行预算（`reports/p56_red_evidence.txt` GREEN 节） |
| `data/r1/wp18_medium/runs/wp18_stage12/.../trials.jsonl` | 64 | 30 次方向学习（A/B 各 15，含 ±signal 双侧证据摘要）+ 阶段条目 |
| `data/r1/wp18_slow/runs/wp18_stage12/.../trials.jsonl` | 156 | 50 次方向学习（含 19 键 §8.3 重学 attempt-2 + attempt-1 拒绝原文、4 条 budget:factor_cache_miss、2 条 correction）+ **2 条 post-hoc 补记**（`domain_gate:backfill` 与 `ledger_note:intermediate_attempt_attribution`，均明标 recorded_post_hoc=true；本会话亲数恰为 4 行 = 2 试验×2 行） |
| `data/r1/wp18_medium/runs/wp18_m3/.../trials.jsonl` | 182 | 90 账户行 record+complete + 阶段条目（含 hold 模式语义登记） |
| `data/r1/wp18_slow/runs/wp18_s3/.../trials.jsonl` | 262 | 130 账户行 + 阶段条目 |
| `data/r1/wp18_medium/runs/wp18_c1/.../trials.jsonl` | 46 | 2 条初筛判定（A 拒绝+B 通过，含失败条件）+ 20 压力行 + 阶段条目 |
| `data/r1/wp18_slow/runs/wp18_c2/.../trials.jsonl` | 6 | 2 条初筛判定（A、B 均拒绝，理由在案）+ 阶段条目（0 行运行） |

覆盖面核对：方向学习（80/80 含被拒失败行原文保留）、规则（A/B+B-fulltrain 否决）、格点（参考格选择+扫描否决）、组合（combo:all3 复用 + combo:slow_all5 预注册）、模式（hold 实现路径）、初筛（含被拒者）全部在案——满足"不能只登记最后留下的候选"。勘误链 E1–E9 追加于 `reports/wp18/errata/errata.jsonl`（只追加，本会话亲读 9 条）。

---

## 7. 审核汇总（三路审核）

- **三路章程**（工作流脚本 dwfrun-b6ac411e ask#12）：审核员A-预算与试错（行数与冻结表一致、零搜索证明、方向一次冻结消费、试错完整）；审核员B-五问证据（行键级证据抽查≥8 处、实际成交后持仓输入、三分开、去最大年份、A/B 隔离证据）；审核员C-完整性与声明（对账在盘、缺行如实升级 measurement_incomplete、域门执行记录、开发验证标注、新产物只写新目录、冻结计划未被运行后修改）。
- **运行级结果（由本报告任务书从运行日志内插传入，本会话无法直接读运行日志——见附录 A 说明）：三路审核 17 条发现，must-fix 已交修复员处置，复核员复核后未解决条目为空（全部解决）。**分项 finding 清单与 must-fix 计数存于工作流运行日志（本会话 GetWorkflowRun 不可用，无法逐条转录；不作臆造）。
- **在盘的修复记录（可逐条复核）**——must-fix 处置落地为 errata E4–E9（`reports/wp18/errata/errata.jsonl`，append-only，修复脚本 `reports/wp18/_analysis/append_errata_fixer.py`）：
  - **E4** 域门执行弱于冻结（时序披露 + 账本 post-hoc 补记 + ANALYSIS §0.2/§11 改写为真实时序）；
  - **E5** ANALYSIS §3/§8 慢域 B 几何累计净超额曾误用会话级口径 −29.10% 并附虚假一致括注 → 更正为折级权威口径 −21.49%（四个判定单位在两种口径下均为负，无裁决翻转）；
  - **E6** ANALYSIS §5 表A"2x 重估费用"5/8 格与 replay 工件不符（原值无法从现有工件复现，披露疑点）→ 按 `fixed_path_2x.fees_total_sum` 实值逐格更正；1x 费用列与 0x/2x 年化列勘误前即 8/8 正确，Q5 结论不受影响；
  - **E7** 账本 09:11:17Z 无标记中间态失败尝试的归属说明（attempt-1 31 成功行逐字保留、20 拒绝行 fail-closed 无产出）；
  - **E8** 已知问题登记（reward_baskets 拒绝消息文案与在役门槛 1e-9 不符——仅文案，行为由 :563 比较治理，推迟到未来包，见 §9 遗留）；
  - **E9** E2 裁决的运行所有者追认（依据三条+回退清单，见 §1.4）。
- 修复后质量门由工作流重跑（regate），结果见 §8；复核员防弱化断言/防改计划正文/防掩盖完整性问题，未解决条目为空。

---

## 8. 质量门（通过）

- **本会话重跑（2026-09-27，报告撰写会话）**：`cd D:\minequant\AlphaGPT && python -m pytest tests/test_p56_wp18_round.py tests/test_p50_wp14_acceptance.py tests/test_p50_wp14_errata.py tests/test_p50_wp14_plan.py tests/test_p50_wp14_precheck.py tests/test_p50_wp14_reconcile.py tests/test_p50_wp14_status.py tests/test_p51_wp15_funding.py tests/test_p51_wp15_integration.py tests/test_p51_wp15_planner.py tests/test_p52_wp16a_cash_identity.py tests/test_p53_wp16b_p45_parity.py tests/test_p54_wp16c_combo_invariance.py tests/test_p54_wp16c_direction_evidence.py tests/test_p55_wp17_claim_qualification.py -q --tb=short` → **323 passed in 13.19s（exit 0）**。文件集 = 工作流质量门的同一 glob 范围（p56 + p50–p55 全部基线）。
- 子套件复核（同会话）：`tests/test_p56_wp18_round.py` → **60 passed in 3.84s**；`tests/test_p52_wp16a_cash_identity.py` → **12 passed in 3.23s**。
- 在案记录（`reports/p56_red_evidence.txt`，append-only）：RED（not_due 早退缺陷逐字复现 + p56 无法收集）→ GREEN（60 passed）+ 新模块清单 + 回归记录；E2 红证据节记录家族 323 passed in 22.34s、FROZEN_A §7 锚定套件 90 passed in 23.60s（命令与退出码在案）。
- 既有失败披露（红证据 GREEN 节）：三个 p34/p52 既有失败与本包无关（bisect 证明：p56 constructor 上提暂还原后失败原样），命中 WP16-A p52 费用融资刃边，路径本包不触碰；不在质量门文件集内。

---

## 9. 遗留

1. **E8 已知问题（推迟到未来包）**：`ashare_model/reward_baskets.py:583` 拒绝消息文案打印 1e-12 形式而在役拒绝门槛为 1e-9（比较在 :563 治理，无行为影响）；为保持 p52 家族红证据状态不动而明确推迟。
2. **契约裁决可推翻**：E2/E9 追认为所有者级决定，用户仍可推翻——回退动作清单见 errata E9 standing（§1.4）。
3. **域门形态偏差已披露未回滚**：冻结顺序的门未逐字执行（E4），以子集门+第二道检查点+post-hoc 账本补记收尾；未来包若复制本流水线，应在编排层把"信号诊断阶段"按域拆开或将门定义与阶段划分对齐。
4. **status_full=measurement_incomplete（两域，设计内）**：全量计划缺行恰为未获授权的 C1:A（20）与 C2（40）压力行；若未来初筛判据变更需补跑，行集已在计划中物化。
5. **工作流运行日志中的三路审核分项清单**（17 条逐条 finding 与 must-fix 计数）本会话不可达（GetWorkflowRun 能力缺口）；在盘可查的是处置结果（errata E4–E9、ANALYSIS 更正、账本补记）与 §8 质量门复跑。
6. **E3 缓存目录追加条目**：`data/r1/slow_ab/factor_tensor_cache/` 多出一个字节相同的内容寻址条目（选择披露而非删除——删除是对既有目录的进一步改写）。
7. **重启条件（停止表行 6）**：这批假设（medium 3 因子 + slow 5 因子及等权组合、A/B 择向、参考格）暂停；新经济假设或新信息源才值得重启。B-fulltrain 如需保留须以独立政策身份另行预注册（契约 §2.3）。
8. **测量限制重申**：DIVIDEND_YIELD 覆盖 0.291–0.361 的"无信息"读数受覆盖不足限制（measurement_incomplete 语义，不作经济负结果表述）；slow 2021 折预算约束使 dynamic 退化为 hold（Q3 机制注记）。

---

## 附录 A：本报告撰写会话执行的核查（命令与结果）

| # | 核查 | 命令/方式 | 结果 |
|---|---|---|---|
| 1 | 质量门（全文件集） | §8 所列 pytest 命令 | 323 passed in 13.19s，exit 0 |
| 2 | p56 / p52 子套件 | pytest（§8） | 60 passed / 12 passed |
| 3 | 方法修订守卫 | 读 `ashare_model/method_revision_registry.json` | `{"episodes": []}` |
| 4 | 行数对账 | 读 `{medium,slow}/{account_report,cost_stress,signal_summary,direction_frozen,run_meta}.json`、`stage12_result.json` | 90/20/40/30、130/0/60/50；failed 全 0；formal_row_total=240；frozen=true |
| 5 | 方向红线 | 计数 account/stress 行 `direction_source` | medium 80+slow 120+stress 20 全为 `frozen_book(no_relearn)`；控制行 10+10 无该字段（按设计） |
| 6 | 完整性 | 读 `integrity/reconcile_full.json`、`status_full.json` | medium missing=20、slow missing=40（=未授权压力行），dup/unexpected/failed=0；label=measurement_incomplete |
| 7 | 初筛判定 | 读 `medium/cost_stress.json` screen_adjudication | A fail（−0.098043）/ B pass（−0.038089） |
| 8 | 账本计数 | 逐行数 7 个 `trials.jsonl` | 18/64/156/182/262/46/6；slow stage12 含 2 条 post-hoc（4 行） |
| 9 | Q2/Q4 数据 | 读 `_analysis/derived.json` | 分歧 10/15、16/25；all_same_train_cutoff=true；screen_descriptive 通过恰 3 格；combo:all3/B passed=true（3/5、+0.23%、+1.27pp）；active_ir_by_fold 与 §5 引用一致 |
| 10 | errata | 读 `errata/errata.jsonl`、`_analysis/append_errata_fixer.py` | E1–E9 在案（9 条，append-only） |
| 11 | 交叉源 | 读 `reports/p56_red_evidence.txt`、`FROZEN_18_PLAN.md`、`ANALYSIS.md`、`_recon/map.md`、工作流脚本（dwfrun-b6ac411e） | 章程/预算表/停止表/门规则与本报告引用一致 |

**未能执行的核查**：工作流运行日志（三路审核 17 条 finding 的逐条清单与 must-fix 计数）——本会话 GetWorkflowRun 返回能力不可用（workflow_introspection_unavailable）；该部分以任务书内插的运行级结论（17 条、must-fix 已处置、全部解决）+ 在盘处置记录（errata E4–E9）表述，未臆造分项。

---

*本报告全部结论为开发验证（2021–2025 测试窗）下的描述性结果：未生成 ClaimRecord、未运行 WP17 max-t 资格审定，任何语句不可作为名义 5% 显著性或样本外推断发布；机械筛选门槛为研发资源分配门槛，非统计显著性证明也非市场规律。*
