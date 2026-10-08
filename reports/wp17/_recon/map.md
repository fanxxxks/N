# WP17 侦察图（只读侦察，2026-09-23）

工作目录约定：测试/脚本以 `D:/minequant/AlphaGPT` 为 cwd。本包新产物一律写绝对路径
`D:/minequant/AlphaGPT/reports/wp17/`（上轮教训：子智能体以 AlphaGPT 为 cwd 时相对路径
`reports/...` 会落到 AlphaGPT/reports/ 下——本包恰需落在那里，但仍用绝对路径消除歧义）。

---

## 1) promotion.py G2 全函数与调用链

### G2 本体
- `ashare_model/promotion.py:840-947` `_gate_g2_significance(artifact, thresholds)`。
  签名只有 `(artifact, thresholds)` ——**没有任何校准输入**；promotion.py 全文件不
  import stat_calibration（grep 证实零引用）。这就是 WP17 要接的缺口：p=0.001 但方法
  未校准，今天照样过 G2。
- 消费顺序（三层，全部 fail-closed）：
  1. **p41 B4 身份绑定** `:865-892`：dsr/max_t 两块必须携带五组身份字段
     （`adjudicated_candidate/run/window/benchmark/trial_set` + 按键存在检查
     `adjudicated_seed`），且 adjudicated candidate/seed 必须等于 artifact 的
     `top_trial`；缺身份或指向他人 → 拒收（continue）。
  2. **p43 B5-B9 内容核验** `:893-895` → `_identity_content_reasons`
     （`promotion.py:471-646`）：run.kind 必须 `current_run`（:489-496）；seed class
     由候选家族决定（`candidate_seed_class` :160-176，确定性家族 seed=None、
     seeded 家族必须整数 seed，:501-517）；dsr window 必须等于 top trial 自身日期轴、
     max_t window 必须落在轴内且 ≥ `P40_MIN_WINDOW_DATES`（:519-575）；benchmark
     definition 必须 `BENCHMARK_DEFINITION` 且 span 与自身 window 一致（:576-595）；
     trial_set 计数闭合且与 artifact 的 `stitched.n_trials`/`dsr_extra_trials`
     对账（:596-645）。
  3. **p43 N5 数值裁决** `:896-947`：`_finite_metric`（:664-701，唯一数值入口，
     missing/NaN/Inf/bool 全部转拒收理由）读 `dsr`（有限、∈[0,1]、≥0.95）与
     `max_t.p_value`（有限、∈(0,1]、`significant_95` 必须布尔且与 p 阈值一致、
     p ≤ 0.05）。

### 组装与调用链
- `evaluate_challenger` `promotion.py:1163-1253`：只组装七门（gates dict
  :1200-1218，`"significance"` 键 :1208），`promoted = all(passed)` :1221；
  verdict 记录 `promotion_rule_version`（`PROMOTION_RULE_VERSION = "4"`，:128，
  v1→v4 演革见 :116-127 docstring）、thresholds 全量（:1239-1252）。
- `evaluate_challenger` 的调用方（grep 全仓）：仅 `promotion.py` 自身 CLI `main()`
  两处（:1439、:1449）+ `tests/test_promotion.py:135`（`_verdict` 助手）。
  **无其他生产调用方。**
- CLI `main()` `:1295-1470`：`read_boundary_artifact(formal=True)` 读镜像
  (:1356-1361) → `assert_consumable_protocol_artifact` (:1256-1292，lifecycle
  `assert_qualified_source_run` + artifact_writer `assert_registered_boundary_artifact`)
  → `RegimeRegistry(root/data/holdout_registry.json)` (:1400，默认 :1302) →
  evaluate_challenger → verdict 落盘 `--output` 默认 `data/promotion_verdict.json`
  (:1301，写 :1457-1463) → stdout 逐门 PASS/FAIL + `promoted=` (:1464-1469)，退出码
  0/1。
- **verdict 怎么进用户可见结论**：
  1. `data/promotion_verdict.json`（CLI 默认输出）；
  2. CLI stdout（每门理由逐行打印）；
  3. lifecycle `PROMOTED` 证据 `build_promotion_evidence`
     （`ashare_model/lifecycle.py:1141-1160`）携带 `promotion_verdict_hash`
     （校验 :1169-1188，"promotion.py G1–G7 remains the gate authority"）——
     该函数生产调用方为零，只有 `tests/test_run_store.py:990,999` 调用（接线尚未发生）。
- **:738-809 的 RegimeRegistry 消费**：该区间是 `_gate_g1_data_formula_p0`
  （:735-837）；regime 检查在 **:808-815**：`regime is None or regime.regime is
  None` → 拒收理由 "no data regime declared; a final evaluation must use future or
  strictly locked data"；否则 `regime.assert_final_evaluation(artifact["folds"])`，
  `HoldoutViolation` 文本入理由。G5 另用 `regime.classify_window` 判 paper window
  是否 dev（:1068-1076）。RegimeRegistry 本体 `ashare_model/regime.py:128-256`
  （classify_window :186-203，assert_final_evaluation :244-255）；声明 CLI
  `python -m ashare_model.regime`（regime.py:259-300）。注意：G1 消费的是 registry
  **对象**（持久化 JSON），不是调用方字符串——与"regime 字符串不得授予资格"的裁定
  一致的做法现已在 G1 存在，G2 的资格判定应沿用同一形态（证据工件，非字符串）。

### stat_calibration 的 pre-claim 查询
- `significance_readiness(calibrations)` `stat_calibration.py:185-213`：输入
  `{场景名: verdict字典}`，输出 `{场景名: {verdict, claim_5pct: bool, reason}}` +
  固定条目 `unlisted_regime = {verdict:"uncalibrated", claim_5pct:False,
  reason:"No reliable inference: the regime has no calibration record on a held-out
  seed block (p48 B5)."}`（:205-212）。calibrated/conservative → claim_5pct=True；
  其它任何 verdict（含 power_reported）→ False（passthrough，:197-199）。
- **现有调用方**：仅 `tests/test_p48_stat_calibration.py:91-102,186-190` 与仓外
  runner `D:/minequant/reports/p48/run_calibration.py:47,68-73`。生产代码零消费
  （契约 draft :72-75 明言"接线到声明流程属后续语义变更"——WP17 即该变更）。

---

## 2) 校准证据工件与"方法版本"

### 工件位置（关键：在仓外）
- **权威工件：`D:/minequant/reports/p48/calibration_report.json`**（repo 外——
  draft §3 :87 明文"重校准运行（repo 外 `reports/p48/`）"。仓内
  `D:/minequant/AlphaGPT/reports/` 只有 p52-p54 红证据与 wp15/wp16*/wp17 目录，
  无 p48）。runner：`D:/minequant/reports/p48/run_calibration.py`
  （SIZE_REPS=1000 / POWER_REPS=400 / n_perms=999 / block=10，逐场景调
  `calibrate_scenario`，写 JSON）。
- 仓内 `data/` 下唯一的 calib 命中是 `data/p10_calibration_engineering.json`（无关）。

### 身份字段（JSON 实测，:1-173）
- 每场景行：`scenario, kind, n_perms=999, block_size=10, seed_start`
  （900000..960000，每场景 10k 切片），`reps`（1000 或 400）, `rejections, alpha,
  rate, ci_low/high/width, verdict, claim, seconds, ms_per_call`。
- 顶层：`scenarios, readiness（significance_readiness 全量输出，含 unlisted_regime）,
  total_seconds=376.16, size_reps=1000, power_reps=400, n_perms=999, block_size=10`。
- **缺失的身份字段**：无方法版本、无 `CALIBRATION_VERSION`、无代码/commit sha、
  无显式 `k_trials/n_dates`（仅由场景名→当时 SCENARIO_SPECS 隐含）。W2 的身份匹配
  只能以 JSON 现有字段原样登记（硬性红线：既有校准证据按身份原样消费，不得回溯
  发明版本号）。

### "方法版本"现状
- `stat_calibration.py:25 CALIBRATION_VERSION = 1`——是**校准 harness 版本**且
  **未写进 JSON**；test_p48:193 钉 `== 1`。
- **max-t/dsr 实现模块 `eval_corrections.py` 没有任何 VERSION 常量**
  （grep "VERSION" 零命中）。`versions.py` 只有 `PROTOCOL_VERSION = "29"`（:50）等
  协议族版本。
- **无参数对象**：方法参数是函数 kwargs——`max_t_from_matrix(matrix, *,
  n_perms=5000, seed=0, block_size=10, observed_index=None)`
  （eval_corrections.py:362-368）；生产侧 `eval_artifacts.py:301-303` 传
  `max_t_perms`。即"方法版本与参数"这一身份组在当前系统中**不存在**，WP17 W1/W2
  需新建登记（从 JSON 的 n_perms/block_size/seed_start 等既有字段读取）。
- **标签分歧（重要）**：JSON 的 power 行 verdict 为 `power_reported`、claim 为
  "Detection rate 0.930, 95% CI [...] - reported as measured; power scenarios carry
  no acceptance band (p48 B2)."——**当前 `calibration_verdict`
  （stat_calibration.py:120-182）产生不了这两个串**（无 power 分支；只有
  calibrated/conservative/unreliable_at_nominal/insufficient_samples）。commit
  `84184f4` 只改了 docs+tests（`git show --stat` 证实：仅
  docs/p48_stat_calibration_measurement_log.md + tests/test_p48_stat_calibration.py），
  power 行标签是按存储计数在仓外 JSON 里重算的；模块从未加 power 分支。
  test_p48:180-190 只钉 readiness 对 `power_reported` 的 passthrough。

---

## 3) "实际共同日期数 / 候选池大小 / 缺折政策"三量

### 可得环节（全部已在生产路径上产出）
- **实际共同日期数**：`max_t.window.n_dates` —— 联合裁决窗 W = 当前 run 各 stitched
  trial 日期轴的**交集**（`eval_corrections._adjudication_window`
  :478-496；记入 `_max_t_from_pool` :563-575）。`dsr.window.n_dates` 则是被裁决
  trial 自身 stitched 轴长。生产者拒绝 n<30 的窗（`P40_MIN_WINDOW_DATES=30`
  :302，:426-433 raise）。
- **候选池大小**：`max_t.trial_set = {n_current, n_extra, n_trials,
  matrix_excluded}`（`_identity_blocks` :197-203；`_max_t_from_pool` :571-574）；
  artifact 级另有 `n_candidates=len(rows)`（eval_artifacts.py:292）、
  `stitched.n_trials`（:295-298）、`dsr_extra_trials`（:304）、`dsr.n_trials`。
- **缺折政策**：是**结构性事实而非开关**——
  1) stitched trial 记录 `failed_folds` 列表（`eval_metrics.py:713-719`），
     全折失败则整个 trial 落选（:597-598）；
  2) 绝不零垫（p41 B1：benchmark 缺失/失配 raise，:624-638）；
  3) 不覆盖窗 W 的 trial 进 `matrix_excluded`（reason="window_mismatch",
     n_missing_dates，eval_corrections.py:534-543）——**剔除出矩阵但保留在
     多重计数里**。
  p43 是其背后的完整性层（N1 非有限 stitch 拒收 eval_metrics.py:640-660、N2 有限
  矩阵 eval_corrections.py:338-359、N3 参数合法性 :305-335），但 p43 契约文本
  并未把三量作为一组命名——"三量"是 WP17 词汇（用户规范），W2 的覆盖域判定要
  自行把这三量从上述字段聚出来。
- 对应校准证据侧的覆盖域参数：JSON 每场景 `reps/n_perms/block_size` +
  SCENARIO_SPECS 的 `k_trials/n_dates`（k=4..9、n∈{30,250}，操作点边界见测量日志
  §7 :74-75——"其他操作点属未校准域"）。

### 正式报告/产物写出点
- **protocol artifact**：`eval_artifacts.build_result`（:155-311；dsr :300、
  max_t :301-303、三量相关字段 :292-304）→ `ProtocolResultArtifact.
  model_validate`（:310）→ `write_boundary_artifact(artifact_type="protocol",
  ...)`（`evaluation.py:1004-1011`，RunStore 登记 + 便利镜像；writer 本体
  `artifact_writer.py:114-160`，fail-closed）。
  schema（`artifact_schemas.py:317-357`，`ARTIFACT_SCHEMA_VERSION=3` :51）把
  `dsr/max_t` 类型为裸 `dict|None`——**W1 在块内加字段是 additive**，不动内层
  schema。
- **promotion verdict**：`data/promotion_verdict.json`（promotion.py:1301 默认，
  :1457-1463 写）。
- **evaluation CLI stdout**（用户可见统计结论面）：`evaluation.py:1026-1037` 打印
  `dsr: ...` 与 `max_t: observed=... p=... significant_95=...`，:1038-1043
  selfcheck 判定——**当前无任何资格限定语**，属 W3"一切用户可见正式结论"的改造面。
- r01/r02/r03、tier_reports、research_doctor、admission 均不读 max_t/dsr/
  significant（grep 零命中）——统计结论的用户可见面就是上面三处。

---

## 4) power_sr025/sr05 消费面 + 标签章节位置

### 消费面（穷尽）
- 定义：`SCENARIO_SPECS`（stat_calibration.py:76-83，sr=0.5/0.25，k=8,n=250，
  scale=0.01）；生成器 :249-253（仅 m[0] 加 `sr*scale` 植入）。
- 读方：仓外 JSON 的 `scenarios.power_sr025/power_sr05` 行 + `readiness` 条目
  （claim_5pct=False）+ `run_calibration.py`；仓内仅 test_p48:180-190（passthrough
  钉扎）与 test_p48:175-177（生成器植入检查）。
- 措辞落点：测量日志 §3（:31-36，"检测率 100% [99.05%,100%] / 93.0% [90.1%,95.1%]"，
  报告事实、无通过带）+ JSON claim 串。**没有生产代码把高检出率写进结论**；唯一
  的"detection rate"文本就在 JSON claim 与日志里。
- p40 时代的功效守卫（区别于 p48 功效点）：`tests/test_eval_resampling_calibration.
  py:116-129 test_c5_strong_signal_is_detected`（0.5 日 SR、band ≥0.90、**调参种子
  50000+**）与 draft `docs/p40_statistical_resampling_draft.md:95`。功效尺度换算
  （0.25/0.5 → 年化 3.97/7.94，×√252）是 WP17 用户给定冻结引用，仓内无此换算文本。
- W5 须标注"sr025/sr05 不可外推到任务尺度（<1% 年化相对超额）"——落点即上述
  SCENARIO_SPECS 注释/新场景登记处 + 契约文档。

### W4 标签修正目标与契约章节
- **过强措辞代码位置：`stat_calibration.py:150-155`**——
  `claim = f"Nominal {alpha:.0%} size supported: rate={rate:.3f}, 95% CI
  [{low:.3f}, {high:.3f}] contains {alpha}."`（"size supported ... contains"）。
  保守分支 :156-162、不可靠分支 :163-171、不足分支 :143-149。
- **契约标签章节：`docs/p48_stat_calibration_draft.md` §2 B4（:55-70）**——判定
  规则表（`CI 包含 0.05` → `calibrated` 等）+ 勘误 1（:62-66，先例：勘误以附注
  追加）。B2 表 :40-48 列功效两点；B5 :72-75 声明 readiness 接线属后续语义变更
  （WP17 正是该变更，按仓例应新立 WP17 契约 draft）。
- 同步面：测量日志 §2 表（:21-27）与仓外 JSON 的 claim 串**保留原样**（冻结/只读，
  勘误只追加）；`[4.6%,7.5%]` 例 = JSON `independent_noise` 行
  （ci_low=0.04601, ci_high=0.07536, rate=0.059, reps=1000）——W4 测试锚点。
- 钉扎检查：**无测试钉正向 claim 的精确串**（test_p48 只钉
  `"no reliable inference" in claim`（:77,102）与 `"0.05" in claim`（:88））——
  W4 换串对现有测试是低破坏面，但新测试须钉新冻结串 + [4.6,7.5] 反例语义。

---

## 5) 既有测试清单（p55 前钉扎面）

| 文件 | 数量 | 与 WP17 相关的钉扎 |
|---|---|---|
| `tests/test_p48_stat_calibration.py` | 11 | N2 Wilson 手算 :41；N3 判定表 :63、no-widening :80、readiness 拒未列域 :91；N4 perms floor :108/:113；N5 种子块不相交 :130、注册表覆盖 :139（**钉住 7 场景名集合**）、生成器依赖结构 :152、power passthrough :180、`CALIBRATION_VERSION==1` :193。正向 claim 串无精确钉扎。 |
| `tests/test_promotion.py` | 51 | G2 值域测试 :262-291（拒低 dsr / 拒不显著 max_t）；**promoted=True 四处**：:138-151（`test_all_gates_pass_promotes`，`_strong_artifact()` :71-90 走真实 build_result）、:678、:695、:1069（G6/G7 正例）——全部依赖 G2 现语义，**收紧即破，须按冻结文档逐个重锚**；`test_promotion_rule_version_pinned` :1125（钉 "4"，W2/W3 语义变更 → bump 决策 + 重锚）；G1 regime 拒收 :217-243；consumption 检查 :1354-1420。 |
| `tests/test_p41_candidate_evidence_binding.py` | 8 | J5 G2 拒绝裁决他人候选 :202、缺身份块 :230；J6 版本钉 "4" :255。 |
| `tests/test_p43_statistical_input_integrity.py` | 26 | N5 G2 数值拒收族 :191-251 + **正控制 :253-257（G2 通过一致有限统计——收紧即破）**；B5-B9 内容核验族 :271-477（run kind/window/benchmark/trial_set/seed class，含精确理由子串）；N1-N3 stitch/matrix/参数合法性 :85-189。 |
| `tests/test_eval_resampling_calibration.py` | 5 | C1-C5 反退化守卫（**调参种子 10000-50000**，带宽 [0.010,0.115] 头注已声明"披露容差、非验收带"角色 :16-27）；C5 功效守卫 :116-129。 |
| `tests/test_p40_statistical_resampling.py` | 9 | stitch 语义/窗交集/历史剔除但计数 :145-204；最小窗拒收 :223。 |
| `tests/test_evaluation.py`（相邻） | — | `max_t_from_rows` 噪声不显著 :1227-1232、强信号显著 :1234-1239；artifact dsr/max_t 缺失 :734。未钉 CLI stdout 的 max_t 行。 |
| `tests/test_gates.py`(29)/`test_completion_gates.py` | — | **与 promotion G2 无关**（ashare_data ProductionGateRunner / 完成门，编号撞名而已）。 |

- 分片登记：`scripts/check_test_shards.py`——**新 p55 测试文件必须同 commit 登记进
  某个 shard 元组**（p48 在 :179，p53/p54 在 :173-178 邻近）。
- `pytest.ini`：`pythonpath = .`（须以 AlphaGPT 为 cwd：`cd D:/minequant/AlphaGPT
  && python -m pytest tests/<文件> -q`）。

---

## 附：本次侦察实测验证过的命令
- `python -c "from ashare_model.stat_calibration import ..."`（AlphaGPT cwd）：确认
  现注册表 7 场景 seed 切片 900000..960000，且**插入排序在 short_window_30d 之前的
  新场景名会把 short_window_30d 的 seed_start 从 960000 推到 970000**（用
  `power_task_scale` 实测）——W5 新增场景若命名不当会移动既有场景的种子切片身份，
  与仓外 JSON 记录的 seed_start 不再对应。
- `git log/show --stat`（AlphaGPT 是 git 仓库，HEAD dbaf729）：确认
  stat_calibration.py 只有 59ba7ac 一个提交；84184f4 只改 docs+tests。
- JSON 读取：`python -c "import json; ..."` 确认 calibration_report.json 顶层键。
