# P41 候选与统计证据绑定预注册契约（工作包 5B）

- 状态：**APPROVED**，用户 2026-09-19 直接指令「工作包 5B：绑定候选与统计
  证据。范围：候选选择、统计结果结构、promotion.py。统计结果必须明确标识
  候选、运行、窗口、基准和试错集合。历史试验可以进入多重检验范围，但不能
  把它的显著性授予另一个候选。基准缺失或长度错配应明确失败，不能悄悄补零
  后继续解释为超额收益」（p30–p40 先例；批准记录回写于本文件头）。
- 起草：执行模型，分支 `codex/b04-load-cache-performance`。
- 勘察基线：p40 实现完成后（GREEN commit），工作树仅含 p40 变更。
- 时序边界（用户原文）：与 5A（p40）同批，必须在正式统计结论之前完成。

## 1 问题陈述（已实证）

1. **基准静默补零**（`eval_metrics.stitch_oos_series`）：成功行的
   `benchmark_daily_returns` 缺失或与 `daily_returns` 长度错配时，拼接以
   `[0.0] * len(d)` 补零继续——此后 excess_returns 实为毛收益，却仍被 DSR/
   max-t/G3 解释为超额收益。已由 `test_stitch_missing_benchmark_uses_zeros`
   钉住（该行为即缺陷本身）。
2. **统计证据不标识归属**：`dsr`/`max_t` 结果块只带 `best_candidate`/
   `best_seed`（DSR）与计数，不标识被裁决候选、窗口、基准与试错集合构成；
   promotion G2 只读 `dsr["dsr"]` 与 `max_t["significant_95"]` 数值，不校验
   统计证据与 artifact `top_trial` 的候选一致性。
3. **历史显著性可被授予他人**：`_dsr_from_pool` 在「本次运行 + 历史试验」
   合并池上取**全局**最大 SR 作为被裁决试验——当历史试验 SR 更高时，本
   artifact 记录的 dsr/best_candidate 属于历史候选；G2 数值通过而 G3/G5/G6/G7
   作用在本次 `top_trial` 上——统计证据与被晋级候选不是同一对象。

## 2 冻结语义（实现必须逐条可对照）

**B1 基准缺失/错配显式失败**：`stitch_oos_series` 中，成功行的
`benchmark_daily_returns` 缺失或 `len != len(daily_returns)` →
`ValueError`（与重复折同级的完整性错误；错误信息含 candidate/seed/fold）。
补零路径删除。失败行不受影响（本就不拼接）。

**B2 统计结果身份结构**：`dsr_from_rows`/`max_t_from_rows` 的结果块必须
记录身份五元组：
- **候选**：`adjudicated_candidate`/`adjudicated_seed`（被裁决试验，仅出自
  本次运行的行；见 B3）；
- **运行**：`run`（`{"kind": "current_run"}` 标识统计量来自本次行集；运行
  级 spec_id/run_id 由 artifact 层承载——`build_result` 的身份块为既有权威，
  G2 校验两者一致，不在 corrections 内复制第二套运行身份）；
- **窗口**：`window`（`first_date`/`last_date`/`n_dates`；DSR 用被裁决试验
  全拼接序列的日期轴，max_t 用联合矩阵窗口 W——p40 已建）；
- **基准**：`benchmark`（`definition`、`n_days`、`first_date`/`last_date`，
  取自被裁决试验/max_t 矩阵首列试验的基准序列；`definition` 记录引擎等权
  基准标签 `engine:equal_weight_universe`）；
- **试错集合**：`trial_set`（`n_current`/`n_extra`/`n_trials`/
  `matrix_excluded`——max_t 复用 p40 字段，DSR 同构记录）。

**B3 历史试验进入多重检验范围、不授予显著性**：
- DSR：被裁决试验 = **本次运行试验中 SR 最高者**；n_trials（多重计数）=
  全池（本次 + 历史）。若历史试验 SR 高于本次最高，记录
  `extra_outranks_current=true`（诚实披露，不改变被裁决对象）；
  `best_candidate` 字段保留为被裁决候选（与 `adjudicated_candidate` 一致）。
- max-t：**观察统计量 = 被裁决试验在 W 上的 t 统计量**（被裁决试验与
  DSR/top_trial 同一选择规则——拼接 Sharpe；Hansen-SPA 式"被选模型的统计量
  对 max 零分布"）；bootstrap 零分布的 max 取**联合矩阵内全部试验**（本次 +
  覆盖 W 的历史）——历史试验计入零分布（多重检验范围）但永远不可能成为被
  裁决统计量的来源。（勘误 1，2026-09-19 实现期：起草句「观察统计量只取
  本次运行试验的 max-t」在"按 W 上 t 取 max"字面读法下会选出与 top_trial
  不同规则的候选，使 B4-2 对合法 artifact 误报；澄清为被裁决试验自身的
  t 统计量，方向不变——历史永不供应观察值。校准驱动的是矩阵级对称口径
  （observed_index=None），不受影响。）
- 本次运行无试验 → `dsr`/`max_t` 返回 `None`（既有行为）。

**B4 promotion G2 绑定**（`evaluate_challenger` significance 门）：
1. `dsr`/`max_t` 缺身份块（adjudicated_candidate/seed、window、benchmark、
   trial_set）→ 门 FAIL（"statistical evidence not bound"——pre-p41
   artifact 一律拒绝，fail closed）；
2. `dsr.adjudicated_candidate/seed` 或 `max_t.adjudicated_candidate/seed`
   ≠ artifact `top_trial.candidate/seed` → 门 FAIL（统计证据属于另一个
   候选——含历史试验授予场景的边界执法）；
3. 数值判据不变（dsr ≥ 0.95、max-t p ≤ 0.05）；
4. `PROMOTION_RULE_VERSION` "2" → "3"，verdict 记录新版本。

## 3 范围与非目标

范围：`ashare_model/eval_metrics.py`（stitch 基准硬失败）、
`ashare_model/eval_corrections.py`（身份结构 + 被裁决对象重定界）、
`ashare_model/promotion.py`（G2 绑定 + 版本）、`evaluation.load_trial_rows`
（无日期拒绝已在 p40；本包不重复）、测试与分片注册、
`test_stitch_missing_benchmark_uses_zeros` 的契约化替换（§10.1 白名单
规则 2：本契约 B1 即依据；新断言钉住显式失败，覆盖不弱于旧钉住）。

非目标：G1/G3–G7 门语义、阈值数值、DSR/max-t 数学与 p40 重采样机制、
`top_trial` 的选择规则（按拼接 Sharpe 选候选不变——本包绑定证据，不改选法）、
artifact schema spine、对历史 artifact 的重写。

## 4 版本影响表

| 版本 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| `PROTOCOL_VERSION` | 28 | **29** | 统计结果结构（身份块）+ 拼接基准硬失败语义变更（§3.2 统计检验类；p40 已 27→28，本包在其上） |
| `PROMOTION_RULE_VERSION` | "2" | **"3"** | G2 从数值消费升级为绑定+数值消费（B4） |
| `ARTIFACT_SCHEMA_VERSION` | 3 | 不变 | spine 字段集合不变（dsr/max_t 仍 dict，行为新增键） |
| 旧 artifact | — | 保留原版本不重写 | v≤28 产物经 G1 版本不匹配拒绝（既有权威）；无日期 extras 经 p40 拒绝 |

## 5 预期 RED（独立 oracle——禁止从实现反推）

`tests/test_p41_candidate_evidence_binding.py`（同 commit 注册分片）：

- **J1** stitch 基准硬失败：缺失 → `ValueError`；长度错配 → `ValueError`；
  等长基准照常拼接（超额 = daily − bench 手算对照）。
- **J2** DSR 被裁决对象=本次最优：本次 3 噪声试验 + 1 历史强试验
  （extra rows，同日期轴）→ `adjudicated_candidate` ∈ 本次候选、
  `dsr` 显著低于无历史时（多重计数含历史）、`extra_outranks_current`
  == true、`n_trials` == 4；旧实现把 `best_candidate` 判给历史候选（RED）。
- **J3** max-t 不授予历史显著性：本次弱噪声 + 历史强试验（同日期轴，
  覆盖 W）→ `adjudicated_candidate` ∈ 本次候选且 max-t 不显著（本次噪声）；
  旧实现观察统计量取历史试验的 max-t → 显著（RED 方向：断言新语义下
  `significant_95 is False` 且 adjudicated 指向本次候选）。
- **J4** 身份五元组在场：`dsr`/`max_t` 结果含 B2 全部字段，且
  benchmark.n_days == window.n_dates（被裁决试验自身）。
- **J5** G2 绑定执法：构造 dsr/max_t 身份块指向非 top_trial 候选的
  artifact → significance 门 FAIL 且原因含绑定字样；身份块齐全且指向
  top_trial → 门按数值判定（构造 dsr=0.96/p=0.03 通过、dsr=0.9 不通过）。
- **J6** verdict 记录 `promotion_rule_version == "3"`。

`tests/test_promotion.py`/`test_stitched_oos.py` 既有用例的夹具更新随
GREEN commit（白名单规则 2/3：契约 B1/B4 生效后，旧钉住「补零」的断言由
契约替换；通过路径断言强度不变）。

## 6 验证矩阵

1. 聚焦：`python -m pytest -q tests/test_p41_candidate_evidence_binding.py`
2. 相邻：`tests/test_evaluation.py`、`tests/test_stitched_oos.py`、
   `tests/test_promotion.py`、`tests/test_eval_module_split.py`、
   `tests/test_artifact_versions.py`、p27 生命周期、r01/r02 批次。
3. `python -m compileall -j 0 -q …`、`python scripts/check_test_shards.py
   --check`、`git diff --check`、全量 `python -m pytest -q tests -n auto`。

## 7 停止条件

- B3 的被裁决对象重定界迫使 top_trial 选择规则或 G3–G7 语义变化 → 停止
  修订契约，不夹带。
- 全量回归出现无法归因于本包的新红/新 warning → 停止诊断。
- 发现 promotions/webapi 消费面需要超出 §3 的改动 → 停止回契约。
