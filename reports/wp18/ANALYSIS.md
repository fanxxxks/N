# WP18 分析报告（ANALYSIS.md）——信号诊断与账户执行两轨

- 生成：2026-09-27（报告撰写者：分析员；契约 = `reports/wp18/FROZEN_18_PLAN.md`，运行前冻结、对本包只读）
- **全部结果为开发验证（development validation，2021–2025 测试窗）**：不作样本外结论、不作统计显著性结论发布（契约 FROZEN_18_PLAN §11；见本文 §10 WP17 资格状态）。
- 本报告回答冻结计划 §7 的五问 + §6.2 机械筛选门槛 + §6.1 停止/继续条件表逐行裁决。所有数字给出**来源行键/文件**；分析侧新产物在 `reports/wp18/_analysis/`（新目录，见 §0.4）。

---

## 0. 范围、预算与测量完整性

### 0.1 行数与预算对账（冻结预算表逐项核对）

| 冻结预算（§4） | 物化 | 来源 |
|---|---|---|
| 方向证据 8 单因子×2 规则×5 折 = 80（测量行） | M1=30 + S1=50 = **80**，0 失败 | `reports/wp18/stage12_result.json` per_stage_rows / totals（180/180 物化，0 缺失/重复/意外/失败） |
| 纯信号摘要 10 候选×2 规则×5 折 = 100（测量行） | M2=40 + S2=60 = **100** | 同上；`reports/wp18/{medium,slow}/signal_summary.json` row_count |
| medium 账户评价 90（正式行，1x） | M3 = **90**（80+控制 10），0 失败 | `reports/wp18/medium/account_report.json` row_count=90, failed_rows=0 |
| slow 账户评价 130（正式行，1x） | S3 = **130**（120+控制 10），0 失败 | `reports/wp18/slow/account_report.json` row_count=130, failed_rows=0 |
| 条件性成本压力 ≤40/域（正式行） | C1 = **20**（仅规则 B 过 §5 初筛）；C2 = **0**（A/B 均未过） | `reports/wp18/{medium,slow}/cost_stress.json` row_count；screen_adjudication |
| **正式评价行两域合计 ≤300** | 90+130+20+0 = **240 ≤ 300** | `data/r1/wp18_{medium,slow}/wp18_*_run_meta.json` formal_row_total=240 |
| 算法搜索预算 = 0 | 无 GP/搜索器/权重拟合/新因子（`combo:slow_all5` 为唯一预注册新 ID，等权） | run_meta `algorithm_search_budget: 0`；驱动脚本 `scripts/wp18_{stage12,m3_account,s3_account,cost_stress}.py` |

方向红线（§2.4）：全部 90+130 账户行与 20 压力行 `direction_source = "frozen_book(no_relearn)"`（account_report/cost_stress 逐行字段），消费 `reports/wp18/{medium,slow}/direction_frozen.json`（frozen=true；slow 含 31 保留 + 19 按 FROZEN_A §8.3 重学条目，见 §0.3）。

### 0.2 完整性对账（WP14 experiment_integrity）

- 阶段 1+2 子集：medium 70/70、slow 110/110，0 缺失/重复/意外/失败（`stage12_result.json` integrity 块；`reports/wp18/{medium,slow}/integrity/reconcile_stages12.json`）。
- 阶段 1+2+3：medium 160/160、slow 240/240 全零（`reconcile_stages123.json`）。
- 含压力行（stages123c）：medium 180/180、slow 240/240 全零（`reconcile_stages123c.json`）。
- 全量计划：`reconcile_full.json` medium 缺 20 = 恰为未获授权的 C1:A 行；slow 缺 40 = 未获授权的 C2 行——`stage12_result.json` `full_plan_missing_rows_are_exactly_unrun_stages` 明确此构成，故 `status_full.json` label=`measurement_incomplete` 是**初筛拒绝授权的设计内状态**，不是测量失败。
- 域门（§10.2）——**实际执行弱于冻结口径，见 errata E4**：冻结门要求 medium 全部阶段（含 M3）完整性通过且运行时 ≤3600 s 后才进 slow。真实时序（文件 mtime + 账本时间戳，本地 UTC+8）：medium 阶段 1+2 16:23–16:38；**slow 阶段 1+2 于 16:57–17:49 先于 medium M3（18:13）运行**——冻结形态的门从未执行；slow 启动前唯一已存在的运行时读数是 medium 阶段 1+2 的 279.3 s（≪3600 s），`stage12_result.json` runtime_sec.domain_gate（17:54:39 落盘，晚于 slow 阶段 1+2 完成）记录的即是这个**阶段 1+2 子集门**；medium 阶段 1+2 子集对账 17:49:41 才落盘（晚于 slow 方向书冻结 17:49:05 约 36 s）。对 S3 腿存在事实上的第二道检查点：medium 阶段 1+2+3 完整性对账 18:14:34 落盘后，S3 才于 18:30:54–18:32:07 运行。运行时判据在任何口径下都远低于上限（子集 279.3 s；含 M3 累计 734.9 s ≪ 3600 s）；M3 自身 455.6 s（`wp18_medium_run_meta.json` m3_elapsed_sec）不是门前判据。S3 470.1 s；C1 384.7 s；C2 0.1 s（无行运行）。§8 所要求的域门账本登记以 post-hoc 补记条目补上（`domain_gate:backfill`，明标 post-hoc，见 §11 与 errata E4）。precheck 全部通过（`reports/wp18/precheck.json` ok=true）。

### 0.3 契约裁决记录（醒目留痕，可被用户推翻回退）

`reports/wp18/stage12_result.json` `contract_adjudication_record` + `reports/wp16/FROZEN_A.md §8.3` + `reports/wp18/errata/errata.jsonl` E1/E2/E3（运行后审计另追加 E4–E9：域门执行时序、两处报告数字更正、账本中间态归属说明、已知问题登记、E2 追认记录）：

- **追认（2026-09-27，运行所有者——所有者级决定，用户仍可推翻）**：E2 裁决**含落地形态**（双常量拆分偏差，见下条）经运行所有者追认，**审计关闭、现状保留、不回退**（errata E9）。完整两步链：**授权**（原裁决：窄幅勘误 + 仅 19 个失败键重学）→ **落地实现偏差**（双常量拆分，窄于授权字面，E2 landed_as 披露）→ **追认**（E9，含偏差一并追认）。追认依据三条（全文见 E9）：(1) 20 条 attempt-1 失败行全部为 (C1) fail-closed 拒绝、无方向产出——不存在可选购对象，attempt-2 属测量缺陷修复后的补全而非对已有决定的重学；下游 200 账户行 + 20 压力行全部 `direction_source="frozen_book(no_relearn)"`、消费同一份 17:49 冻结书；(2) 双常量拆分（1e-12 恒等式口径不动、1e-9 仅拒绝门）比授权字面更符合 WP16-A 本意，naive 变体无 slow 账本留痕；(3) 程序完整（§8.3 只追加、红证据 2026-09-27 节、回归全绿：家族 323/p56 60/p52 12/锚 90）。**若日后推翻，回退动作清单**：恢复 slow `direction_frozen.json` 至 31 键 partial 状态；slow 账户/压力行按停止表行 1 停止经济解释；errata E2 与 FROZEN_A §8.3 勘误作废标注；然后重走裁决。

- **裁决（2026-09-27）**：slow Rule A 评分被 WP16-A (C1) 守卫的浮点 ε 区间（缺口 1–2e-12）误拒 19/25 键；最小物理费用比例 5e-5（=min_commission/资金），ε 违规量比其低 7 个数量级 → 认定规格缺陷（容差钉在 round-12 网格自身噪声地板上），授权窄幅勘误。
- **落地**：`_P52_C1_TOLERANCE` 保持 1e-12（成交记账精度逐字节不变）；新增 `_P52_C1_REFUSAL_TOLERANCE=1e-9` 仅用于最终拒绝比较；只重学 19 个 slow Rule A 键，31 个已成功键与 medium 全部冻结件不动（防结果选购）。GREEN 证据：p52 12 通过、p50–p56 家族 323 通过、FROZEN_A §7 锚定套件 90 通过（`reports/p52_red_evidence.txt` 2026-09-27 节，命令与退出码在案）。
- **E3 缓存纪律披露**：首次 slow 运行在驱动拒绝前，`load_data` 内部冷重建（~52 min）向既有 `data/r1/slow_ab/factor_tensor_cache/` 写入一个字节相同的新条目（sha256 81db3d19…，与 medium_a 同身份张量）；无既有文件被改；后续 slow 运行改用 `data/r1/wp18_slow`（§9(b)），账本 `budget:factor_cache_miss_correction` 更正记录。

### 0.4 分析侧重放（本报告 Q4/Q5 的取证方式）

账户行只带汇总（行内 `invested_ratio_mean` 是**目标权重**口径，eval_metrics.py:402-407 取自 `result.target_weights`），而 §7 冻结口径要求**实际成交后持仓**。故分析员以 `scripts/wp18_analysis_replay.py`（新文件）确定性重放同一批 1x 账户行（同冻结方向、同参考配置、同折、同引擎调用），仅观测引擎内部：

- 逐行校验：重放 `daily_returns` 与存储行**逐字节相等**（medium 80/80、slow 120/120，replay_mismatches=[]）；
- 实际权重重建与引擎簿记逐步一致（到期/非到期会话的份额再推导 + 漂移，backtest.py:286-446）：毛收益恒等式 `dot(w,R)=gross` 最大误差 medium 6.3e-16、slow 8.7e-16（`reports/wp18/_analysis/replay_{medium,slow}.json` max_gross_identity_abs_err）；
- 固定路径费用重估的 1x 自检：用实际费用重演权益路径，复现行 total_return 至 ≤2.7e-15（fixed_path_1x_sanity.max_abs_total_return_gap）；
- 分解用仓库权威助手 `cash_exposure_decomposition` / `reprice_fixed_path_fees` / `split_cost_q5_effects`（`ashare_model/wp18_five_questions.py:126/:320/:351`）；会话恒等式 净超额=暴露+选股+费用 逐会话成立（脚本内 1e-10 校验通过）。
- 纪律：重放**不新增正式评价行**、不写账本、不写 `data/r1`；缓存命中才运行（两域 outcome=hit）；产物只写 `reports/wp18/_analysis/`。静态派生（Q1/Q2/Q3/门槛/去最大年）在 `reports/wp18/_analysis/derived.json`（生成脚本 `reports/wp18/_analysis/derive_static.py`，只用冻结助手函数）。

**口径注意（三列分列）**：净超额的**几何累计/年化**一律用 §5 权威口径（折级 ratio 形式：各折 `(1+Tot_p)/(1+Tot_b)−1` 的 Π 与 `^(1/5)`，`geometric_cumulative_excess`/`annualized_from_cumulative`）；**算术累计贡献**按会话级求和（分解分量的唯一可加口径，Σexp+Σsel+Σfee=Σ净超额 逐 session 恒等）。两个算术（5 折和 vs ~1156 会话和）是不同量，表中分列标注，不混用。

---

## 1. Q1：固定标签上的 IC、分位排序与有效覆盖是否显示信息？

**两轨纪律（§0 逐字）**：不得用受预算限制的 0x 回测亏损推导信号没有信息。本节只看信号轨（M2/S2 摘要行：固定因果标签 `causal_target_returns`/`adjusted_open_o2o`，horizon medium=10、slow=20，`direction_frozen.json` provenance.label）。IC 每折在再平衡到期日网格上计算（n_ic_dates：medium 21–24、slow 11）。

### medium（来源：`reports/wp18/medium/signal_summary.json`，行键 `M2:{A|B}:None:{候选} × fold_test_end`）

| 候选/规则 | 各折 IC（2021→2025） | IC>0 折数 | IC 均值 | 分位差符号与 IC 同号 | 覆盖（min–max） | N_eff |
|---|---|---|---|---|---|---|
| MOMENTUM_20 A | −.023 −.014 −.030 −.006 +.051 | 1/5 | −.004 | 4/5 | .994–.999 | ≈796–800 |
| MOMENTUM_20 B | +.023 +.014 +.030 +.006 +.051 | **5/5** | +.025 | 4/5 | 同上 | 同上 |
| RSQ_60 A | +.019 +.066 −.026 +.017 −.055 | 3/5 | +.004 | 5/5 | .999–1.000 | 同上 |
| RSQ_60 B | +.019 +.066 +.026 +.017 +.055 | **5/5** | +.037 | 5/5 | 同上 | 同上 |
| REVERSAL_60 A | −.079 −.046 −.001 +.005 +.052 | 2/5 | −.014 | 4/5 | .995–.999 | 同上 |
| REVERSAL_60 B | +.079 +.046 +.001 −.005 +.052 | 4/5 | +.035 | 4/5 | 同上 | 同上 |
| combo:all3 A | −.045 +.002 −.023 +.005 +.031 | 3/5 | −.006 | 4/5 | 1.000 | 同上 |
| combo:all3 B | +.054 +.055 +.023 +.008 +.070 | **5/5** | +.042 | 5/5 | 1.000 | 同上 |

### slow（来源：`reports/wp18/slow/signal_summary.json`，行键 `S2:{A|B}:None:{候选}`）

| 候选/规则 | 各折 IC | IC>0 | 均值 | 分位同号 | 覆盖 | N_eff |
|---|---|---|---|---|---|---|
| ROE A / B | −.055 +.031 −.092 +.027 −.008 / 镜像负 | 2/5 / 1/5 | −.020 / −.039 | 4/5 / 4/5 | .970–.996 | ≈796–800 |
| PE_TTM A / B | −.036 −.048 −.053 −.032 +.011 / 全正镜像 | 1/5 / **5/5** | −.032 / +.036 | 2/5 / 3/5 | .889–.917 | 同上 |
| REVENUE_YOY A / B | −.033 −.035 −.070 +.009 −.049 / −…+0.049 | 1/5 / 1/5 | −.035 / −.020 | 5/5 / 5/5 | .972–.998 | 同上 |
| DIVIDEND_YIELD A / B | −.008 −.019 +.004 −.013 +.023 / 镜像 | 2/5 / 2/5 | −.003 / −.001 | 4/5 / 4/5 | **.291–.361** | 同上 |
| MARKET_CAP A / B | −.072 −.022 −.040 −.046 +.039 / 镜像 | 1/5 / 2/5 | −.028 / +.003 | 5/5 / 5/5 | .999–1.000 | 同上 |
| combo:slow_all5 A / B | −.082 −.043 −.104 −.022 +.007 / +.005 −.008 −.054 +.022 +.004 | 1/5 / 3/5 | −.049 / −.006 | 3/5 / 3/5 | 1.000 | 同上 |

**回答（开发验证，描述性）**：

1. **medium 域信号轨显示可检出信息，且集中在 B 择向**：B 方向的测试窗截面秩 IC 在单因子 14/15 格-折、组合 5/5 折为正（量级 .006–.079）；分位 Q 高低差符号与 IC 同号 18/20 行-折；覆盖 ~0.99–1.00、N_eff≈800（有效截面接近满配）。A 择向的单因子 IC 仅 6/15 为正——A 的方向常指向测试窗 IC 的反侧（详见 Q2）。
2. **slow 域在信号阶段即偏弱**：即使取事后较优的 B 择向，单因子也只 11/25 格-折为正；组合 slow_all5 B 仅 3/5、均值 −.006。**DIVIDEND_YIELD 有效覆盖仅 0.291–0.361**（约三分之二会员缺失股息定义），其"无信息"结论受覆盖不足限制，不能与其它因子同权解读。
3. 0x 压力行（仅 medium B 有）不用于本问；账户轨的亏损见 Q3/Q4，属执行与暴露问题，不推翻信号轨的上述描述性读数。

---

## 2. Q2：A/B 差异来自评分规则，还是取证窗口、参考配置变化？

**设计分离事实（冻结面）**：

- 同折同标签：A 与 B 的每格 provenance `train_cutoff` 完全一致（medium 15/15、slow 25/25 格 `same_train_cutoff=True`，`_analysis/derived.json` q2_direction_books）；标签、universe mask、数据相同。
- 取证窗口是**规则身份的一部分、未发生"变化"**：A = 训练窗尾部验证子窗（`validation_windows`，scripts/wp18_stage12.py:369-373）；B = 验证段之前的训练切片 `(train_signal_start, val_start)`（scripts/wp18_stage12.py:374-377；p46/wp16c 钉死形态）。两规则的取证日期都不越过该折 train_end（§2.4）。
- 参考配置冻结未变：medium 两规则同用参考格 `n20_m5000`（= r1_medium_a.yaml backtest 块逐字）；slow 同用 r1_slow_ab backtest 块逐字；B 的决策接口本身无配置参数（direction.py:116 `label_ic_direction` 签名），B 行 provenance 中的 reference_config 仅为登记。

**经验证据**：

1. **方向分歧率高**：medium 10/15（66.7%）、slow 16/25（64.0%）（`_analysis/derived.json`；来源 `direction_frozen.json` 每格 A_dir/B_dir）。典型稳定模式：MOMENTUM_20 A=+1 而 B=−1（前 4 折；B 的训练切片 ic_mean≈−.052~−.060）；REVERSAL_60 A=−1 而 B=+1（前 4 折；ic≈+.042~+.057）；slow PE_TTM A=+1 而 B=−1（前 4 折）。分歧格中两规则引用的是**不同的统计量**（A 的 `val_reward_plus/minus` vs B 的 `ic_mean/n_dates`），窗口与配置均相同——差异由评分规则本身产生。
2. **事后对齐（描述性）**：以测试窗带向 IC 符号为参照，A 方向 6/15（medium 单因子）、7/25（slow）；B 方向 14/15、11/25（`_analysis/derived.json` q1_signal ic_pos_folds）。即 A 的"验证 reward 择向"在本测量期更经常选到测试窗 IC 的反侧。
3. **p46 交叉验证（配置敏感性分离）**：与 `D:/minequant/reports/p46/direction_measurement.json`（独立运行、**不同参考格 n10_m2500**）对比三共享因子 × 5 折：**B 方向 15/15 完全一致**（B 无配置依赖，构造上配置不变量，经验证实）；**A 方向仅 8/15 一致**（7 格随配置翻转，如 MOMENTUM_20 2022/2024/2025）。结论：**A/B 的组内差异 = 评分规则**；参考配置若变，受影响的是 A（配置敏感），不是 B。首折模式与 p46 完全一致（MOM+1/RSQ+1/REV−1 vs MOM−1/RSQ+1/REV+1）。
4. **账户后果**：年化净超额（dynamic，1x）B 优于 A 的候选：medium 3/4（MOM −2.16% vs −1.05%；RSQ −0.58% vs −5.28%；REV −1.84% vs −9.24%；combo −0.77% vs −2.04%）；slow 3/6（PE +1.94% vs −4.23%；MCAP −4.48% vs −5.93%；combo −4.72% vs −5.98%）。§5 初筛唯一通过的就是 medium/combo:all3/B。

**回答（开发验证，描述性）**：A/B 差异**来自评分规则本身**（验证 reward 比较秩 vs 训练切片标签 IC 符号），不是取证窗口或参考配置的"变化"——两者在本包逐格冻结相同；窗口差异是两规则各自身份的固有部分。附加发现：Rule A 的方向对参考配置敏感（跨 p46 配置 7/15 翻转），Rule B 不敏感（0/15）；且 A 的选择在本测量期与测试窗 IC 符号的一致率明显低于 B。按停止表行 4：保留此负结果，不再试探新的方向规则（`B-fulltrain` 已在 §2.3 预注册否决）。

---

## 3. Q3：动态交易相对首次持有是否有净增量？

来源：`reports/wp18/{medium,slow}/account_report.json`（行键 `M3:{A|B}:{dynamic|hold}:{候选} × fold_test_end`），三列口径见 §0.4；dyn−hold 用 §5(iii) 权威次序（各腿先年化再作差）。hold = `HoldModeConstructor`（constructor.py:674；非到期强制退出腿 :364-377、到期 hold 腿 :383-391），同引擎同会话网格同冻结方向；hold 行 `funnel_hold_forced_exit_sessions`：combo A/B medium 5、slow A 4/B 3（dynamic 行恒 0）——强制风险退出腿只在 hold 模式出现，语义落地。

| 域/候选 | dynamic 年化净超额 | hold 年化净超额 | dyn−hold（年化 pp） | (i) 正年数 | (ii) 中位 |
|---|---|---|---|---|---|
| medium MOMENTUM_20 A / B | −1.05% / −2.16% | −4.60% / −4.15% | **+3.55 / +1.99** | 3/5 / 2/5 | +0.44% / −1.16% |
| medium RSQ_60 A / B | −5.28% / −0.58% | −7.51% / −3.72% | **+2.23 / +3.14** | 2/5 / 2/5 | −2.33% / −1.56% |
| medium REVERSAL_60 A / B | −9.24% / −1.84% | −10.56% / −4.35% | **+1.31 / +2.51** | 1/5 / 2/5 | −11.05% / −4.50% |
| medium combo:all3 A / B | −2.04% / −0.77% | −5.23% / −2.04% | **+3.19 / +1.27** | 2/5 / 3/5 | −0.64% / +0.23% |
| slow ROE A / B | −8.95% / −12.29% | −10.80% / −12.05% | **+1.85 / −0.25** | 0/5 / 1/5 | −9.61% / −11.81% |
| slow PE_TTM A / B | −4.23% / **+1.94%** | −7.64% / −1.11% | **+3.41 / +3.05** | 1/5 / 3/5 | −1.06% / +0.98% |
| slow REVENUE_YOY A / B | −5.80% / −6.11% | −6.92% / −7.01% | **+1.11 / +0.90** | 1/5 / 1/5 | −7.95% / −7.54% |
| slow DIVIDEND_YIELD A / B | −3.49% / −5.51% | −5.72% / −7.83% | **+2.22 / +2.32** | 2/5 / 1/5 | −4.06% / −5.73% |
| slow MARKET_CAP A / B | −5.93% / −4.48% | −6.98% / −6.28% | **+1.05 / +1.80** | 1/5 / 1/5 | −4.74% / −3.55% |
| slow combo:slow_all5 A / B | −5.98% / −4.72% | −8.64% / −3.77% | **+2.66 / −0.95** | 1/5 / 1/5 | −5.72% / −6.19% |

主组合逐折净超额差（dynamic−hold）：medium A [+3.3, +5.1, −9.4, +12.7, +2.8]pp、B [+1.1, −4.2, −8.4, +5.5, +9.8]pp；slow A [0.0, +0.4, −0.1, +10.6, +2.0]pp、B [−3.0, 0.0, +0.5, +4.1, −6.5]pp。

**回答（开发验证，描述性）**：

1. **相对增量普遍存在但都是"少亏"**：20 个候选-规则格中 18 个 dyn−hold>0（且 18 个全部 ≥ +0.90pp，超过 50bp 门槛；仅 slow ROE/B −0.25pp、slow combo/B −0.95pp 为负）。但**两腿的年化净超额几乎全为负**（20 格中 19 格；唯一正值 slow PE_TTM/B +1.94%，非 §5 判定单位）。主组合几何累计净超额（§0.4 折级权威口径）：medium A −9.80%、B −3.81%；slow A −26.54%、B −21.49%（`_analysis/derived.json` q3_account.*.table[…].dynamic.geometric；与 `{medium,slow}/cost_stress.json` screen_adjudication 的 dynamic_cumulative_net_excess 逐字一致：−0.098043/−0.038089/−0.265384/−0.214913）。**口径警示（errata E5 更正）**：本行此前曾误写 slow B −29.10% 并附"与 cost_stress.json 一致"的虚假括注——−29.10% 是**会话级逐日几何**（`_analysis/replay_slow.json` per_candidate 三列 geometric_cumulative_net_excess，~1156 会话），属 §0.4 明令"不同量、不混用"的非权威口径；四判定单位的会话级几何为 medium A −18.11%/B −12.44%、slow A −32.19%/B −29.10%，仅作分解语境参考，不参与任何判定。
2. **机制注记**：slow 2021 折 A 的 dynamic 与 hold 逐字节同路径（dyn−hold=0.0）：初始建仓只成交 5/20 单（现金上限），此后全部再平衡订单被抑制（`funnel_suppressed_min_trade_count`=230、invested_ratio_mean=0.149，行键 `S3:A:dynamic:combo:slow_all5 × 2021-12-31`）——**预算约束使动态腿在该折退化为持有**。medium 的抑制主要来自 min_trade（组合 B 每折 137–578 单被弃）。
3. 停止表行 3 的前提（"持有有价值、动态没有增量"）**不成立**——是反向（持有更差、动态相对少亏）；但两者都无绝对增量，落入行 4/6（§9）。

---

## 4. Q4：超额中有多少来自实际现金暴露而非选股？（实际成交后持仓分解）

方法与验证见 §0.4。恒等式（逐会话精确）：净超额 = 暴露/现金效应 `(E−1)·Σ(w/E)·r` + 选股效应 `Σ(w/E)·r − 基准` + 费用腿，E=Σ实际成交后权重（现金收益 0）。下表：三列（净超额：算术=会话和 / 几何=折级 ratio / 年化=几何^(1/5)）+ 三分量算术累计贡献（5 测试年合计；+22pp 应读作 +0.22）。来源 `reports/wp18/_analysis/replay_{medium,slow}.json` per_candidate.three_columns。

### medium（dynamic 腿 + 组合 hold 腿；行键 `M3:{规则}:{模式}:{候选}`）

| 候选/规则 | 净超额 算术(会话)/几何(折级)/年化 | 暴露效应 | 选股效应 | 费用腿 | 实际投资比均值 |
|---|---|---|---|---|---|
| MOMENTUM_20 A | −.107 / −.051 / −1.05% | −.250 | +.156 | −.014 | 0.399 |
| MOMENTUM_20 B | −.170 / −.103 / −2.16% | −.108 | −.048 | −.015 | 0.397 |
| RSQ_60 A | −.305 / −.238 / −5.28% | −.117 | −.174 | −.014 | 0.674 |
| RSQ_60 B | −.047 / −.029 / −0.58% | −.155 | +.126 | −.017 | 0.709 |
| REVERSAL_60 A | −.539 / −.384 / −9.24% | +.043 | **−.571** | −.012 | 0.455 |
| REVERSAL_60 B | −.162 / −.089 / −1.84% | −.222 | +.075 | −.015 | 0.400 |
| combo:all3 A | −.160 / −.098 / −2.04% | −.210 | +.065 | −.016 | 0.470 |
| combo:all3 B | −.094 / −.038 / −0.77% | **−.295** | **+.218** | −.016 | 0.474 |
| combo:all3 A hold | −.324 / −.236 / −5.23% | −.207 | −.110 | −.008 | 0.452 |
| combo:all3 B hold | −.147 / −.098 / −2.04% | −.252 | +.114 | −.009 | 0.589 |

### slow（dynamic 腿 + 组合 hold 腿；行键 `S3:{规则}:{模式}:{候选}`）

| 候选/规则（dynamic，除注明 hold） | 净超额 算术(会话)/几何(折级)/年化 | 暴露效应 | 选股效应 | 费用腿 | 实际投资比均值 |
|---|---|---|---|---|---|
| ROE A / B | −.506 / −.374 / −8.95%；−.743 / −.481 / −12.29% | +.284；+.679 | **−.783；−1.418** | −.007；−.003 | 0.546；0.272 |
| PE_TTM A / B | −.253 / −.194 / −4.23%；+.052 / +.101 / +1.94% | −.043；−.081 | −.203；+.142 | −.008；−.010 | 0.642；0.826 |
| REVENUE_YOY A / B | −.359 / −.258 / −5.80%；−.370 / −.270 / −6.11% | +.003；+.092 | −.355；−.456 | −.007；−.006 | 0.473；0.492 |
| DIVIDEND_YIELD A / B | −.201 / −.163 / −3.49%；−.342 / −.247 / −5.51% | −.086；−.021 | −.108；−.315 | −.007；−.006 | 0.690；0.597 |
| MARKET_CAP A / B | −.360 / −.263 / −5.93%；−.292 / −.205 / −4.48% | −.052；−.038 | −.302；−.247 | −.006；−.006 | 0.525；0.545 |
| combo:slow_all5 A | −.353 / −.265 / −5.98% | **+.495** | **−.842** | −.006 | 0.416 |
| combo:slow_all5 B | −.304 / −.215 / −4.72% | −.093 | −.204 | −.007 | 0.585 |
| combo A hold / B hold | −.527 / −.363 / −8.64%；−.260 / −.175 / −3.77% | +.422；−.263 | −.942；+.010 | −.007；−.007 | 0.304；0.527 |

口径注：每行第一列净超额的**算术**=会话级求和（分量恒等式锚），**几何/年化**=折级权威口径（§0.4）；两个算术（会话和 vs 5 折和）是不同量，分列不混用。

**回答（开发验证，描述性）**：

1. **medium：负超额以现金暴露为主，选股（B 择向）实际为正**。实际投资比例仅 0.40–0.71（组合 ~0.47）：top_n=20 × cap 5% 的名义满配被 min_trade=5000 与整手/预算抑制（组合 B 每折 137–578 单被 min_trade 弃、lotgap_l1_mean_due 0.007–0.028）。组合 B：选股效应 +21.8pp（算术累计）但暴露效应 −29.5pp、费用 −1.6pp → 净 −9.4pp。即**超过 100% 的负净超额来自未投资现金**（选股贡献为正）。
2. **slow：负超额以选股为主**。A 择向的选股效应大负（ROE A −78pp、组合 A −84pp），部分被低暴露抵消（组合 A 暴露 +50pp——低配一个下跌的持仓簿"少亏"）；B 组合选股 −20pp、暴露 −9pp。slow 2021 折实际投资低至 ~0.15（初始建仓只成交 5/20）。
3. 目标权重口径（行字段 invested_ratio_mean，如 medium 组合 B 0.475）与实际成交后均值（0.474）在均值上接近，但 §7 冻结口径的分解只能用实际持仓（本节全部数字来自实际成交后权重重建）。
4. 推断性表述（"现金暴露吞没了正选股"）为开发验证描述性比较，未走 WP17 资格审定（§12）。

---

## 5. Q5：成本变化是直接费用，还是现金/整手/预算引起的后续路径变化？（两口径分表）

### 表A：固定成交路径费用核算（固定 1x 路径，仅改费率重估；`reprice_fixed_path_fees`，同名义本金）

来源：`_analysis/replay_{medium,slow}.json` per_candidate.fixed_path_{0x,2x}（各腿年化 = 折级几何 ^(1/5)，与 §3 同口径）。

| 腿（主组合） | 1x（=实际行，折级年化） | 固定路径 0x | 固定路径 2x | 1x 费用总额（5 折，元） | 2x 重估费用（元） |
|---|---|---|---|---|---|
| medium A dynamic | −2.04% | −1.74% | −2.16% | 1596 | 2227 |
| medium A hold | −5.23% | −5.08% | −5.29% | 782 | 1049.4 |
| medium B dynamic | −0.77% | **−0.45%** | **−0.90%** | 1704 | 2397 |
| medium B hold | −2.04% | −1.87% | −2.10% | 838 | 1130 |
| slow A dynamic | −5.98% | −5.86% | −6.02% | 644 | 873.8 |
| slow A hold | −8.64% | −8.51% | −8.68% | 686 | 932.1 |
| slow B dynamic | −4.72% | −4.59% | −4.77% | 731 | 972.7 |
| slow B hold | −3.77% | −3.64% | −3.82% | 711 | 941.7 |

2x 重估费用 < 2×1x 费用（如 medium B dyn 2397 < 2×1704=3408）：min_commission=5 元地板不随档位放大（§3.3），小额单费用非线性；八腿 2x/1x 比值一致落在 1.33–1.41（1049.4/782、873.8/644、932.1/686、972.7/731、941.7/711 等）。**errata E6 更正**：本列原 slow 四格与 medium A hold 一格（1032/1211/1288/1378/1336）与所引 replay 工件不符（原值无法从现有工件复现，披露见 E6），已按 `fixed_path_{0x,2x}.fees_total_sum` 实值逐格更正；1x 费用列与 0x/2x 年化列勘误前即 8/8 正确。**直接费用效应**（1x−固定0x，年化）：medium A/B dyn 0.30/0.32pp、hold 0.15/0.17pp；slow 四腿 0.12–0.13pp——量级即"费用/本金"（1704/100000≈1.7% 五年复合 ≈0.33pp/年）。固定路径 0x/2x 的年化与 §3/表B 同为折级几何口径，可与 1x 列直接相减。

### 表B：重新执行成本压力（C1 行，全路径重跑；仅 medium B 获授权）

来源：`reports/wp18/medium/cost_stress.json` 行键 `C1:B:{dynamic|hold}:combo:all3 × {0x|2x} × fold_test_end`。slow 无 C2 行（A/B 初筛均未过，`reports/wp18/slow/cost_stress.json` row_count=0 + skip_reason）。

| 腿 | 1x 年化 | 重执行 0x 年化 | 重执行 2x 年化 | 路径效应 0x（重执行−固定路径） | 路径效应 2x |
|---|---|---|---|---|---|
| medium B dynamic | −0.77% | −0.55% | −0.90% | **−0.10pp** | **+0.01pp** |
| medium B hold | −2.04% | −1.88% | −2.10% | −0.01pp | +0.00pp |

（几何累计：重执行 0x dyn −2.73%、2x −4.40%；hold 0x −9.05%、2x −10.05%。）

**路径变量的直接读数（重执行 vs 1x，行内字段）**：换手率基本不动（2021 折 0x/1x/2x = 0.0127/0.0128/0.0128；2023 折 0x=0.0107 略高于 1x/2x=0.0094——唯一可见的费率→可执行单差异）；invested_ratio_mean 变动 ≤0.015；holdings_mean ≤0.6 名；suppressed_min_trade 除 2022（578→597）与 2023（551→509）外同值。

**回答（开发验证，描述性）**：在本测量的低换手（到期会话均值 0.5–1.4%）下，成本变化的作用**几乎全部是直接费用**；重执行与固定路径的差（路径效应：现金、整手、预算联动）≤0.10pp/年，且 2x 档路径效应≈0。固定路径核算与重执行压力回答的是不同问题，本节两表分列、未合并。0x 行不用于信号轨结论（§0 两轨纪律）。

---

## 6. 机械筛选门槛判定（预注册；**研发资源分配门槛，非统计显著性证明也非市场规律**）

判定单位 =（域, 主组合, 规则），依据 = 1x 账户行拼接 2021–2025，(i)(ii) 主策略腿=dynamic（`CONDITIONS_I_II_LEG`，理由登记于 `cost_stress.json`）。来源：`reports/wp18/{medium,slow}/cost_stress.json` screen_adjudication + 账本 `screen:1x_initial` 条目（含未通过者）。

| 判定单位 | (i) ≥3/5 年正 | (ii) 中位正 | (iii) dyn−hold ≥50bp | 裁决 | 后续 |
|---|---|---|---|---|---|
| medium combo:all3 A | ✗（2/5） | ✗（−0.64%） | ✓（+3.19pp） | **不过** | 0 行压力（skip_reason 在案） |
| medium combo:all3 B | ✓（3/5） | ✓（+0.23%） | ✓（+1.27pp） | **通过** | C1 20 行（0x/2x × 两模式 × 5 折） |
| slow combo:slow_all5 A | ✗（1/5） | ✗（−5.72%） | ✓（+2.66pp） | **不过** | 0 行 |
| slow combo:slow_all5 B | ✗（1/5） | ✗（−6.19%） | ✗（−0.95pp） | **不过** | 0 行 |

逐折净超额（dynamic）：medium B [+1.09, +9.22, +0.23, −7.50, −6.03]%；medium A [−5.46, +0.34, +2.42, −0.64, −6.56]%；slow A [−17.11, +5.06, −4.11, −5.72, −6.69]%；slow B [−7.50, +1.31, −4.10, −6.19, −6.88]%。

**描述性参考（非判定单位）**：全部 20 个候选-规则格的同一三条件（`_analysis/derived.json` q3_account.*.screen_descriptive，同样标注研发门槛）：通过三条件的仅 3 格——medium MOMENTUM_20/A（3/5、+0.44%、+3.55pp）、medium combo:all3/B（判定单位）、slow PE_TTM/B（3/5、+0.98%、+3.05pp）。单因子格不是 §5 判定单位，仅作资源分配参考，不得表述为"通过筛选"。

---

## 7. 去掉贡献最大年份后的结果（§6.2，避免只看一行平均数）

| 判定单位 | 剔除年（其净超额） | 剩余 4 年正年数 | 剩余中位 |
|---|---|---|---|
| medium combo:all3 A | 2023（+2.42%） | 1/4 | −3.05% |
| medium combo:all3 B | 2022（+9.22%） | 2/4 | −2.90% |
| slow combo:slow_all5 A | 2022（+5.06%） | 0/4 | −6.20% |
| slow combo:slow_all5 B | 2022（+1.31%） | 0/4 | −6.54% |

（来源：`cost_stress.json` screen_adjudication.*.screen.ex_minus_max_year。）四个判定单位去掉最大贡献年后无一满足 (i)/(ii)——**无单年依赖的幸存者**；medium B 的 3/5 正年中有 1 年（2022 +9.22%）贡献了全部正读数的大半。

---

## 8. 停止/继续条件表逐行裁决（§6.1 逐字条件 → 裁决与证据）

| # | 条件（逐字） | 裁决 | 证据 |
|---|---|---|---|
| 1 | 主要臂缺失/程序失败/日期或价格口径不一致 → 停止经济解释，修测量完整性 | **未命中** | 全部主臂物化且 0 失败（§0.1）；reconcile stages12/123/123c 全零缺失/重复/意外/失败（§0.2）；所有行同一 `evaluate_signal` 引擎路径（eval_metrics.py:147）与同一 PIT/价格口径；`status_full=measurement_incomplete` 仅因初筛拒绝授权的压力行未运行（设计内）。测量完整性成立，经济解释允许进行。 |
| 2 | 纯信号有信息但账户执行后消失 → 只处理已定位的执行问题，不扩搜索 | **命中（medium）** | medium 信号轨 B 择向 IC 14/15 单因子格-折为正（§1），账户年化净超额全负（§3）；已定位的执行问题=现金暴露（实际投资 0.40–0.71，§4）+ min_trade/整手/预算抑制（每折 137–578 单被弃；slow 2021 初始建仓 5/20）。处置符合动作：本包零搜索、零新因子、零权重拟合；slow 域信号阶段本身弱（A 7/25、B 11/25 为正），该行在 slow 不完全成立。 |
| 3 | 持有价值、动态没有增量 → 转向静态选股/持有假设，停止宣称动态选股有效 | **未命中（反向）** | hold 腿普遍更差：18/20 格 dyn−hold>0（§3）；两腿净超额同为负——"持有有价值"不成立。本包同样**不**宣称动态选股有效（无绝对增量）。 |
| 4 | A/B 对结果影响很大但都无稳健增量 → 保留负结果，停止继续试探方向规则 | **命中** | 分歧率 66.7%/64.0%（§2）；账户后果分化（medium B 过初筛 A 不过；年化差最大 7.4pp）；20 格年化净超额 19/20 为负；去最大年后无幸存（§7）。动作已执行：负结果保留于本报告与全部产物；未试探任何第三规则（B-fulltrain 预注册否决，FROZEN_18_PLAN §2.3）。 |
| 5 | 可执行、拼接净超额与 active IR 为正、年度较一致、动态优于持有、成本压力下仍有空间 → 才考虑有限 GP/Random 对照 | **未命中** | 四判定单位拼接净超额几何全负（折级权威口径 −9.80%/−3.81%/−26.54%/−21.49%，errata E5 更正；会话级几何亦全负，但为非 §5 判定口径）；active IR 多折为负（如 medium B 组合 [+0.08,+0.64,−0.03,−0.66,−0.62]，3/5 折负）；年度不一致（§6 逐折）。**不启动 GP/Random 对照**（搜索预算保持 0）。 |
| 6 | 两域在有效、可比的测量下均无增量 → 暂停这批假设，新经济假设或新信息源才值得重启 | **命中** | 两域判定单位 (i)(ii) 全不过（§6）；测量有效性由行 1 背书（完整、可比、同引擎同口径）。裁决：**暂停这批假设**（medium 3 因子 + slow 5 因子及其等权组合、A/B 两择向、参考格 n20_m5000/配置块）。 |

**总裁决：停止**（行 4 + 行 6 命中；行 5 未命中故无对照搜索）。负结果按契约 §11 作为本包产出保留。

---

## 9. 结论汇总（全部为开发验证）

1. 信号轨（medium，B 择向）有可检出信息；账户轨净超额全负的主因是执行结构——现金暴露吞没正选股（Q1/Q4）。
2. A/B 差异由评分规则产生，非窗口/配置变化；A 配置敏感（p46 交叉 7/15 翻转）、B 不敏感（15/15）；A 的事后对齐率低（Q2）。
3. 动态相对持有普遍"少亏"（18/20 格 ≥ +0.90pp），但无绝对增量；预算约束使 slow 2021 折动态退化为持有（Q3）。
4. 负超额：medium=现金暴露主导（组合 B 暴露 −29.5pp vs 选股 +21.8pp，算术累计）；slow=选股主导（组合 A 选股 −84.2pp、暴露 +49.5pp）（Q4）。
5. 成本变化在本测量中几乎全是直接费用（medium B dyn 1x→0x +0.32pp/年）；重执行路径效应 ≤0.10pp/年、2x 档≈0（Q5）。
6. 机械门槛：仅 medium combo:all3/B 过三条件（研发资源分配门槛）；去最大年后无幸存（§6/§7）。
7. 停止/继续：行 4、行 6 命中 → 保留负结果、暂停这批假设；不启动 GP/Random（§8）。

---

## 10. WP17 资格状态（推断性措辞纪律）

- 本报告**未生成任何 ClaimRecord**（WP17 `claim_record.py` 五组字段：statistics/multiplicity/operating-point/structure/qualification），**未运行 max-t 资格审定**——因此**没有任何语句可作为名义 5% 显著性或样本外推断发布**。§1–§9 全部结论的地位：开发验证下的描述性比较（§11"描述性结果照常保存（不受资格限制）"）。
- 涉及推断性措辞的语句及其状态：§2"A 方向更常指向测试窗 IC 反侧"、§4"现金暴露吞没正选股"、§8 行 2/4/6 裁决——均标注：**开发验证-描述性，未走 WP17 资格审定，不可外推**。
- 机械筛选门槛的冻结标注在每处判定重复：**研发资源分配门槛，非统计显著性证明也非市场规律**。
- 方法修订守卫未触发：`ashare_model/method_revision_registry.json` 读取核实 `episodes=[]`（2026-09-27 本分析会话内检查）。

## 11. 试错登记汇总（§8：不只登记最终幸存者）

| 账本 | 条目 | 内容 |
|---|---|---|
| `data/r1/wp18_medium/runs/wp18_stage12/.../trials.jsonl` | 64 行（32 试验） | 30 方向学习（A/B 各 15，含双侧证据摘要）+ 阶段条目 |
| `data/r1/wp18_slow/runs/wp18_stage12/.../trials.jsonl` | 156 行（78 试验；20 条 failed 行保留） | 50 方向学习（含 19 键 §8.3 重学 attempt-2 + attempt-1 拒绝原文）、4 条 `budget:factor_cache_miss`、2 条 `budget:factor_cache_miss_correction`（E3）、2 条 post-hoc 补记（`domain_gate:backfill` 域门判定补记——E4；`ledger_note:intermediate_attempt_attribution` 09:11:17Z 中间态失败尝试归属说明——E7；均明标 recorded_post_hoc，经账本 API 追加、哈希链复验通过） |
| `data/r1/wp18_medium/runs/wp18_m3/...`、`data/r1/wp18_slow/runs/wp18_s3/...` | 182 / 262 行 | 每账户行 record+complete（90/130）+ 阶段条目（含 hold 模式语义登记） |
| `data/r1/wp18_medium/runs/wp18_c1/...` | 46 行 | 2 条初筛判定（A 拒绝+B 通过，含失败条件）+ 20 压力行 + 阶段条目 |
| `data/r1/wp18_slow/runs/wp18_c2/...` | 6 行 | 2 条初筛判定（A、B 均拒绝，理由在案）+ 阶段条目（0 行运行） |

预注册决策登记（FROZEN_18_PLAN §8 清单）：参考格选择 n20_m5000（含对简报 n10_m1000 的否决理由，§2.5）；Rule B 正式命名铸造（direction.py:58）；B-fulltrain 不保留（§2.3）；combo:slow_all5 预注册与分散化假设（§1.3）；hold 模式实现路径（§3.2）；域门判定（**运行时未登记，post-hoc 补记**：实际执行的是阶段 1+2 子集门 279.3 s ≪ 3600 s、先于 M3 运行，弱于冻结口径——errata E4 + 账本 `domain_gate:backfill` 补记条目）；每次方向学习与每条初筛判定（含被拒者）——分别位于 stage12/c1/c2 账本与 `stage12_result.json`。勘误 E1–E9 追加于 `reports/wp18/errata/errata.jsonl`（只追加；E4 域门时序、E5/E6 两处报告数字更正、E7 账本中间态归属说明、E8 已知问题登记、E9 E2 追认记录为运行后审计追加）。

## 12. 来源文件清单（本报告引用）

阶段产物：`reports/wp18/stage12_result.json`；`reports/wp18/{medium,slow}/{direction_frozen,signal_summary,account_report,run_meta,cost_stress}.json`；`reports/wp18/{medium,slow}/integrity/{plan,plan_stages123,plan_stages123c,reconcile_stages12,reconcile_stages123,reconcile_stages123c,reconcile_full,status_*}.json`；`reports/wp18/precheck.json`；`reports/wp18/integrity_stage12_summary.json`；`data/r1/wp18_{medium,slow}/wp18_{medium,slow}_report.json`、`wp18_*_run_meta.json`；六个 `trials.jsonl`；四份运行日志；`reports/wp18/errata/errata.jsonl`；`reports/wp16/FROZEN_A.md`（§8.3）；`reports/p52_red_evidence.txt`（2026-09-27 节）；`D:/minequant/reports/p46/direction_measurement.json`（交叉验证）。
驱动/权威代码：`scripts/wp18_stage12.py`、`scripts/wp18_m3_account.py`、`scripts/wp18_s3_account.py`、`scripts/wp18_cost_stress.py`、`scripts/wp18_reconcile_{stage12,m3,s3,stress}.py`；`ashare_model/wp18_five_questions.py`；`ashare_model/wp18_signal_vs_account.py`；`ashare_model/direction.py`；`ashare_model/eval_metrics.py`；`ashare_model/backtest.py`；`ashare_portfolio/constructor.py`；`ashare_model/method_revision_registry.json`。
分析侧新产物（本报告）：`reports/wp18/_analysis/derived.json`（生成脚本 `reports/wp18/_analysis/derive_static.py`）；`reports/wp18/_analysis/replay_medium.json`、`replay_slow.json`（重放脚本 `scripts/wp18_analysis_replay.py`；Q4/Q5 全部数字来源）。
