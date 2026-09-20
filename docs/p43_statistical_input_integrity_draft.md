# P43 统计输入、数值与证据绑定收口预注册契约（工作包 7）

- 状态：**APPROVED**，用户 2026-09-20 直接指令「以『在可执行的订单规则和可信
  的统计边界下，判断现有组合是否提供了超过被动持有、现金暴露和简单基线的
  增量』为目标，完成工作包 7：统计输入、数值与证据绑定收口。P0，先做。
  范围：eval_metrics.py、eval_corrections.py、promotion.py 及相关测试。
  ……先写本次两个反例的失败测试。再加『候选名相同但窗口/基准/运行不同』的
  拒绝测试。……数值校验与身份绑定可分成两个小提交」（批准记录回写于本文件头）。
- 起草：执行模型，分支 `codex/b04-load-cache-performance`（承接 p40–p42 同一
  集成线）。
- 勘察基线：p42 完成后 HEAD `27a5285`，工作树干净（仅既有未跟踪 `.mimosa/`）。
- 两个已复现反例（2026-09-20 探针，`_tmp_wp7_probe.py`，跑在基线树上）：
  1. 四条噪声序列 max-t `p=0.9581`；仅把一个元素改为 NaN 后 `p=0.0020`
     （= 1/(500+1)，最小可达 p 值）。机制：NaN 进入 bootstrap 后每次
     `boot_t.max() >= observed` 比较为假，计数恒零。行接口（`max_t_from_rows`）
     同样复现（0.2555 → 0.0020）；基准序列含一个 NaN 同样复现；
     `dsr_from_rows` 对 NaN 输入返回 `NaN`，而 `NaN < 0.95` 为假 → G2 通过。
  2. 构造 dsr/max_t 块：`p_value=0.90` 与 `significant_95=True` 互相矛盾，
     且 `run.kind="engineering_run"`、window 为 1999 年 3 天、benchmark 定义
     `csi300_index`、trial_set 与 artifact 不符——G2 仍 `passed=True`。

## 1 问题陈述（已实证）

1. **非有限数值可制造最小 p 值**：`max_t_from_matrix` 缺有限性检查；
   `stitch_oos_series` 对 daily/benchmark 序列中的 NaN/Inf 静默拼接。一个
   NaN 即把噪声池变成「显著」。NaN/Inf 进入统计计算的所有路径均未关闭。
2. **重采样参数无合法性验证**：`n_perms`/`block_size` 无类型与下界检查
   （浮点会被 `int()` 静默截断；`0`/负值产生无意义结果）。
3. **G2 信任布尔值而非数值**：`significant_95` 与 `p_value` 可以互相矛盾而
   门不察；`dsr=NaN` 时 `NaN < dsr_min` 为假 → 通过；门未校验取值域。
4. **G2 只验证「字段存在」不验证内容**：p41 B4 建立了身份五元组与候选/seed
   绑定，但运行类型、窗口、基准、试错集合的内容不与 artifact 自身证据核对
   ——同候选名 + 指向别处窗口/基准/运行/试错集的证据仍然通过。
5. **seed=None 无候选类型语义**：确定性基线（`benchmark:`/`baseline:`/
   `selfcheck:`/`control:`/`combo:` 族，含 r02/r03 的 `cell::` 前缀形态）
   按构造 seed=None，当前检查靠 `None != None` 为假而偶然通过；伪造整数
   seed 或把 seeded 候选的 seed 抹成 None 同样可满足相等比较。

## 2 冻结语义（实现必须逐条可对照）

**N1 stitch 数值完整性**：`stitch_oos_series` 中，成功行的 `daily_returns`
或 `benchmark_daily_returns` 含任何非有限值（NaN/±Inf）→ `ValueError`
（与基准缺失/长度错配同级的完整性错误；错误信息含 candidate/seed/fold 与
非有限计数）。有限值路径逐位不变。

**N2 联合矩阵有限性**：`max_t_from_matrix` 对输入矩阵做有限性检查，含任何
非有限格子 → `ValueError`（fail closed：不产出 p 值，绝不产出
1/(n_perms+1) 垃圾）。防御纵深：即使调用方绕过 stitch 直接喂矩阵也被拦截。

**N3 重采样参数合法性**：`max_t_from_matrix` 的 `n_perms`/`block_size`/
`seed` 必须是非 bool 整数（`int`/`np.integer`），且 `n_perms ≥ 1`、
`block_size ≥ 1`；否则 `ValueError`。浮点/bool/字符串不再被静默截断。

**N4 产出取值域**（既有行为，本包钉住）：`p_value = (count+1)/(n_perms+1)`
∈ (0,1]；`significant_95` 由 p 值对阈值计算得出，永不接受注入值；
`dsr = psr(...)` ∈ [0,1]。

**N5 promotion G2 数值判据**：
1. `dsr["dsr"]` 必须是有限数值且 ∈ [0,1]，且 ≥ `dsr_min`；None/NaN/Inf/
   越界 → 门 FAIL（各自明确的 reason）；
2. `max_t["p_value"]` 必须是有限数值且 ∈ (0,1]；None/NaN/Inf/越界 → FAIL；
3. `significant_95` 必须是 bool，且必须等于 `p_value ≤ max_t_p_max`；
   矛盾 → FAIL（「p=0.90 且 significant_95=True」类证据作废）；
4. 门按数值阈值判定（p_value ≤ max_t_p_max），布尔只是被校验的一致性
   声明，不是判据本身。

**B5 运行类型绑定**（G2，p41 B4 延伸）：`dsr`/`max_t` 块的
`run.kind` 必须为 `"current_run"`；工程/自检/其他运行类型的统计证据不能
支持晋级。

**B6 窗口绑定**（G2）：artifact `top_trial` 必须携带拼接日期轴 `dates`
（缺 → FAIL）。
- dsr 块：`window.first_date/last_date/n_dates` 必须逐一等于 top 试验自身
  日期轴的首/末/长度（DSR 用被裁决试验全序列）；
- max_t 块：`window` 必须落在 top 试验日期轴内
  （`first_date ≥ dates[0]`、`last_date ≤ dates[-1]`、`n_dates ≤ len(dates)`，
  ISO 日期字典序即时间序），且 `n_dates ≥ P40_MIN_WINDOW_DATES`（生产者
  低于该下限拒绝运行，声称更低窗口的块必为伪造）。

**B7 基准绑定**（G2）：`benchmark.definition` 必须等于
`engine:equal_weight_universe`（`BENCHMARK_DEFINITION` 单一权威）；
`benchmark.n_days/first_date/last_date` 必须等于其自身 `window` 的
`n_dates/first_date/last_date`。

**B8 试错集合绑定**（G2）：`trial_set` 的 `n_current`/`n_extra`/`n_trials`
必须是非负整数、`n_current ≥ 1`、`n_current + n_extra == n_trials`；块的
`n_trials == trial_set.n_trials`；且与 artifact 自身证据核对：
`n_current == artifact["stitched"]["n_trials"]`（本次运行拼接试验数）、
`n_extra == artifact["dsr_extra_trials"]`（历史 extras 计数）。artifact 缺
任一载体 → FAIL（fail closed）。

**B9 seed 类别按候选类型**（G2）：候选名按「剥去 `cell::` 前缀后取 `:`
前的族名」分类。确定性族 = {`benchmark`, `baseline`, `selfcheck`,
`control`, `combo`}：top 试验与统计块的 seed 必须为 `None`（伪造整数 seed
→ FAIL）。其余族（`trained`/`random_search`/`gp_search`/`tpe_search`/
`rl:` 等）为 seeded：seed 必须为非 bool 整数（None/浮点 → FAIL）。族名
不在确定性集合的未知候选默认按 seeded 处理（更严方向）。p41 B4 的
「块 seed == top seed」相等比较保留，类别检查叠加其上。

## 3 范围与非目标

范围：`ashare_model/eval_metrics.py`（N1）、`ashare_model/eval_corrections.py`
（N2/N3）、`ashare_model/promotion.py`（N5/B5–B9）、新测试文件
`tests/test_p43_statistical_input_integrity.py`（同 commit 注册分片）、
`PROMOTION_RULE_VERSION` 与其 pin 的同步更新。

非目标：DSR/max-t 数学与 p40 重采样机制、阈值数值、G1/G3–G7 门语义、
`top_trial` 选择规则、G3 的 NaN 阈值比较（top 指标来自 stitched_metrics，
N1 关闭其输入侧后残余风险另行诊断）、artifact schema spine、r01/r02/r03
面板语义、对历史 artifact 的重写、§四 中误报率校准的方法学改造（拒绝率
置信区间与留出验收属下一轮测量方案，不在本包代码范围）。

## 4 版本影响表

| 版本 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| `PROMOTION_RULE_VERSION` | "3" | **"4"** | G2 数值判据收紧（N5）+ 身份五元组全量内容校验（B5–B9）；v4 于本包两个实现提交内分两步落地（数值在前、身份在后），两步均在本分支本包内闭合 |
| `PROTOCOL_VERSION` | 29 | **不变** | 所有有限输入的统计输出逐位不变；stitch 新增拒绝规则只影响非有限输入——该类行从未产出过合法测量（其产物是 NaN 统计，属缺陷输出而非既有语义），与 p41 bump（合法旧输入行为改变 + 产物结构新增字段）不同类；说明理由记于此 |
| `ARTIFACT_SCHEMA_VERSION` | 3 | 不变 | spine 字段集合不变，本包不新增块键 |
| 旧 artifact | — | 保留原版本不重写 | 携带 NaN/矛盾统计块的旧产物在 G2/v4 下被拒绝（fail closed，即本包目的）；无日期/无身份者已由 p40/p41 拒绝 |

## 5 预期 RED（独立 oracle——禁止从实现反推）

`tests/test_p43_statistical_input_integrity.py`（同 commit 注册分片）：

- **N1a** 一个 NaN 的 daily_return → `max_t_from_rows` 抛 `ValueError`
  （行接口反例）；干净同型输入照常出数且 p ∈ (0,1]。
- **N1b** 一个 NaN 的 benchmark → `ValueError`；Inf 同样拒绝。
- **N2** 直接喂 `max_t_from_matrix` 的矩阵单格 NaN → `ValueError`（矩阵接口
  反例）。
- **N3** `n_perms=0`/`n_perms=500.5`/`block_size=0`/`block_size=True`/
  `seed=None` → `ValueError`；合法整型参数照常运行。
- **N5** G2：`dsr=NaN` → FAIL 且 reason 指名非有限；`p_value=0.90` +
  `significant_95=True` → FAIL 指名矛盾；`p_value=1.5` → FAIL 越界；
  `dsr=1.2` → FAIL 越界；`significant_95=False` + `p_value=0.01` → FAIL
  矛盾；有限一致且达标 → 门 PASS（既有强 artifact 正向对照）。
- **B5** `run.kind="engineering_run"` → FAIL 指名运行类型。
- **B6** dsr 块 window 与 top 试验日期轴不符（首/末/长度任一）→ FAIL；
  max_t 块 window 落在 top 轴外或 `n_dates < 30` → FAIL。
- **B7** `benchmark.definition` 非等权引擎基准，或 span 与自身 window 不符
  → FAIL。
- **B8** trial_set 与 artifact `stitched.n_trials`/`dsr_extra_trials` 不符，
  或内部加法不闭合 → FAIL。
- **B9** 确定性候选（`baseline:` 族）seed=None 正常通过绑定；同候选伪造
  `seed=7` → FAIL；seeded 候选（`trained`）seed=None → FAIL。
- **正向对照**：`build_result` 产出的真实 artifact（有限输入）在 v4 G2 下
  绑定与数值全部通过（既有 `test_all_gates_pass_promotes` 亦为回归钉）。

既有 pin 的更新随 GREEN（白名单规则 2）：
`test_p41_candidate_evidence_binding.test_j6_*` 与
`test_promotion.test_g7_passes_registry_promotable_formula` 的
`promotion_rule_version == "3"` pin → `"4"`（本契约 §4 为依据；断言强度
不变，仅版本字面更新）。

## 6 验证矩阵

1. 聚焦：`python -m pytest -q tests/test_p43_statistical_input_integrity.py`。
2. 相邻：`tests/test_promotion.py`、`tests/test_p41_candidate_evidence_binding.py`、
   `tests/test_p40_statistical_resampling.py`、`tests/test_eval_resampling_calibration.py`、
   `tests/test_stitched_oos.py`、`tests/test_evaluation.py`、
   `tests/test_eval_module_split.py`、`tests/test_artifact_schemas.py`。
3. `python -m compileall -j 0 -q ashare_data ashare_model ashare_portfolio ashare_trading scripts webapi`、
   `python scripts/check_test_shards.py --check`、`git diff --check`、
   全量 `python -m pytest -q tests -n auto`（与串行对账按 §10.2）。

## 7 停止条件

- N1 的 stitch 拒绝使既有合法路径（引擎有限输出）出现新红 → 停止诊断，
  不放宽为过滤/替换（过滤即静默改数据，违反 fail-closed 方向）。
- B6–B8 的核对字段在真实 `build_result` 产物上不闭合（如 `dsr_extra_trials`
  语义与 pool 计数不一致）→ 停止回契约修订，不放松校验。
- 全量回归出现无法归因于本包的新红/新 warning → 停止诊断。
- 发现需要改动 G1/G3–G7 或 corrections 数学才能闭合 → 停止回契约。
