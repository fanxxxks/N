# WP17 工程报告——校准资格声明闭环（W1-W6）

- 日期：2026-09-24。工作区 `D:/minequant/AlphaGPT`（测试 cwd；`pytest.ini` pythonpath=.）。
- **期望真源**：`reports/wp17/FROZEN_17.md`（2026-09-23 实现前冻结，638 行，本次全文实读；实现取舍冲突时以它为准）。侦察图 `reports/wp17/_recon/map.md`（235 行，实读）。
- 本报告只陈述可复核事实：代码给 `文件:行号`（本次会话实读当前树），命令给逐字命令与输出（本次会话亲跑，2026-09-24），规格事实注明出自 FROZEN_17.md 条款，运行内事实（三路审核摘要）注明出处为工作流主控的确认返回。
- 红线遵从状态：本包**未启动任何校准/重采样运行**（守卫与零运行证明见 §5.4）；`data/**` 与既有 reports 产物只读（WP17 新产物全部在 `ashare_model/`、`tests/`、`reports/p55_red_evidence.txt`、`reports/wp17/`、`docs/p48_stat_calibration_draft.md` 追加节）；冻结文档只追加（p48 draft 勘误 2，`git diff --stat` = **31 insertions, 0 deletions**，本次实测）。

---

## 1 声明闭环：五组字段落盘实证与资格判定算法（W1/W2）

### 1.1 正式报告的三个载体与五组字段的写出点

FROZEN_17.md §1（:36-45）定义"正式报告"为三个载体；本次实读核实三处全部落盘：

| 载体 | 写出点（实读行号） | 内容 |
|---|---|---|
| ① protocol artifact（`build_result` 产物） | `ashare_model/eval_artifacts.py:308-320` | `result["claim_record"] = build_claim_record(...)` 在构造 dsr/max_t 的同一处**无条件**构造（:302-304 之后）；注释明示"描述性保存路径永不因资格被阻断"。schema 面：`ashare_model/artifact_schemas.py:364-368` 新增 `claim_record: dict \| None = None`（additive 带默认，pre-wp17 载荷仍可校验）；`ARTIFACT_SCHEMA_VERSION` 3→4（:51-55），generation 3 并入 `LEGACY_SCHEMA_VERSIONS = (1, 2, 3)`（:57-61，只增不减规则） |
| ② promotion verdict（`evaluate_challenger` 返回 / CLI 默认落盘 `data/promotion_verdict.json`，promotion.py:1404） | `ashare_model/promotion.py:1328-1336` | verdict 携带 `"claim_record"`（artifact 记录的深拷贝，:1331-1335）与 `"qualification"`（G2 重算摘要，:1336；摘要构造 `_qualification_summary` :1230-1248：qualified/reason_codes/covering/blocking，**重算而非读 artifact 自带标志**） |
| ③ CLI stdout | `ashare_model/evaluation.py:1039-1043` | max_t 行之后逐行打印 `claim_record_stdout_lines`（`ashare_model/claim_record.py:513-529`：`claim_record: qualified=... reason_codes=[...]`，qualified=False 时追加 `no reliable inference: {reason}`）；promotion CLI 另有 `--calibration-evidence` 旗标（promotion.py:1407-1415，默认取注册路径，:1520-1531，不可读 → fail-closed 置 None） |

五组字段本体在 `ashare_model/claim_record.py:308-343`（`build_claim_record`）逐组构造：

1. **第一组 原始统计量与 p 值**（:310-315）：`statistics.{dsr, max_t_observed, max_t_p_value, max_t_significant_95}`，取自 artifact 的 dsr/max_t 块；
2. **第二组 方法版本与参数**（:317-323）：`max_t_method_version`（= `eval_corrections.MAX_T_METHOD_VERSION`，eval_corrections.py:61 实读 =1，向前钉扎不回溯）、`calibration_version`（stat_calibration.py:25 =1）、`n_perms`/`block_size`（实际所用）、`dsr_method="analytic_bldp_uncalibrated_v1"`（:44；p48 draft §2 B1：DSR 解析式副统计量，声明未校准，资格只作用于 max-t）；
3. **第三组 实际共同日期数/候选池/缺折政策**（:325-333）：`n_common_dates`（max_t.window.n_dates）、`n_pool_trials`/`n_current_trials`/`n_extra_trials`（trial_set 计数，多重计数全域）、`matrix_excluded`（window_mismatch 明细）、`n_failed_folds_total`（Σ stitched failed_folds，:301-304）、`missing_fold_policy="structural_no_imputation_no_zero_padding_v1"`（:39，唯一注册值）；
4. **第四组 校准证据身份**（:335-340）：`source`/`sha256`（注册登记 `CALIBRATION_EVIDENCE_REGISTRATION` :58-66：`D:/minequant/reports/p48/calibration_report.json`，sha256 `b85a6527…6d0e`）、`rows_consulted`（仅身份字段 scenario/kind/n_perms/block_size/seed_start/reps/verdict，IDENTITY_ROW_FIELDS :71-74——**不搬运历史 claim 串**）、`method_version_attestation`（:46-52 逐字）；
5. **第五组 资格**（:342，由 `qualify_claim_record` 填充）：`qualified/status/reason_codes/reason/covering_scenarios/blocking_scenarios`。

字段级证据：`tests/test_p55_wp17_claim_qualification.py:266-327`（`test_w1_five_groups_structure` 逐组钉扎键集合与值）、:366-384（`test_w1_w3_descriptive_artifact_never_blocked_by_qualification`：qualified=False 时五组照常全量填写，且因生产默认 n_perms=5000≠999 给出 `EVIDENCE_IDENTITY_MISMATCH`——如实状态）。

### 1.2 资格判定算法（W2，fail-closed 固定次序）

实现于 `claim_record.py:346-432`（`qualify_claim_record(record, evidence)`），次序与 FROZEN_17.md §2（:160-226）逐条对应：

1. **第一步 证据可消费性**（:357-361）：evidence 为 None 或未指向注册条目 → `EVIDENCE_NOT_REGISTERED`；实测 sha256 ≠ 注册 sha256 → `EVIDENCE_HASH_MISMATCH`。任一失败**直接终止**（后续码不产生）。
2. **第二步 方法身份匹配**（:363-379）：任务侧 `(max_t_method_version, n_perms, block_size)` 逐项对证据行集合与注册条目钉扎版本；不相等 → `EVIDENCE_IDENTITY_MISMATCH`，终止。**如实状态**：证据 n_perms=999，生产默认 `max_t_perms=5000`（FROZEN §2.2 :180-183 记录；eval_artifacts.py:164 / eval_corrections.py:588 实读复核），故默认产物在此步即不合格——修复=以 999 重测或经 W6。
3. **第三步 覆盖域**（:381-393）：`n_common_dates ∈ {30, 250}`（**点集合成员**，CALIBRATED_N_DATES :91；180/400/31/249 全部落选，测试 p55:452-458 参数化钉扎）；`n_pool_trials ∈ [4, 9]`（K_BAND :92）；`missing_fold_policy` == 注册值。
4. **第四/五步 覆盖场景集与合取**（:394-417）：覆盖集由 n_common_dates **唯一决定**（COVERING_SCENARIOS_BY_N :96-102：250→{independent_noise, common_factor_095, ar1_phi04, duplicate_candidates}；30→{short_window_30d}），**调用方无权指定**；逐场景取证据行 verdict：缺行→`SCENARIO_UNCALIBRATED`；insufficient_samples→`SCENARIO_INSUFFICIENT_SAMPLES`；unreliable_at_nominal→`SCENARIO_UNRELIABLE_AT_NOMINAL`（入 blocking_scenarios）；calibrated/conservative 通过（B5 语义）；`power_reported` 永不进入覆盖集（:412-417 passthrough）。
5. `qualified = (reason_codes == [])`（:419）；reason 由冻结模板渲染（`no_reliable_inference_reason` :132-141 / `qualified_claim_reason` :144-159，逐字对应 FROZEN §3.4 :299-319）。

**regime 字符串永不授予资格**（用户裁定）：算法输入只有前四组与证据对象，无任何 regime 参数；`test_w2_caller_regime_string_never_grants_qualification`（p55:515-523）实测注入 `record["task_actuals"]["regime"]="iid"` 不改变 `N_DATES_OUT_OF_CALIBRATED_DOMAIN` 拒绝，且正常构造的记录 JSON 中根本不含 regime 键。

**当前冻结证据下的必然结果**（FROZEN §2.6 :228-234 的机器可执行形式，本次亲跑证实）：`tests/test_p55_wp17_claim_qualification.py:530-557` 的 `TestFrozenEvidenceBlocksEveryOperatingPoint`（skipif 保护、本机证据存在故执行，含于本次 61 passed）：n=250 任务被 ar1_phi04（unreliable_at_nominal，rate=0.095）阻断；n=30 被 short_window_30d（unreliable，rate=0.108）阻断——两行 verdict 本次直接读自证据 JSON 亲核（见 §7 命令 5）。即**在方法修订并经 W6 产生新合格证据之前，一切任务 qualified=False**。

### 1.3 G2 的 verify-don't-trust 消费（与 ①的闭环对接）

`_gate_g2_significance` 第四层（promotion.py:975-1013，详 §2）：步骤 2 用 `claim_record_consistency_reasons`（claim_record.py:435-510）把记录的第一/二/三组与 artifact 自身 dsr/max_t/stitched/matrix_excluded **逐字段**对账（12 项 checks :454-503）；步骤 3 用证据对象**重算**资格，artifact 自带的 `qualification.qualified` 布尔不被信任（篡改测试 p55:652-666）。

---

## 2 G2 消费语义变化：收紧前后对照与适用边界（W3）

### 2.1 收紧前后对照

| 面 | 收紧前（侦察图实测记录，map.md:12-33） | 收紧后（本次实读当前代码） |
|---|---|---|
| G2 输入 | `_gate_g2_significance(artifact, thresholds)`，**无任何校准输入**；promotion.py 全文件不 import stat_calibration（recon §1 grep 证实） | `_gate_g2_significance(artifact, thresholds, calibration_evidence=None)`（promotion.py:854-857）；`evaluate_challenger` 新关键字参数 `calibration_evidence: CalibrationEvidenceRegistry \| None`（:1262，对象形态，同 G1 RegimeRegistry 先例），七门组装处下传（:1300-1302） |
| 判定层数 | 三层：p41 B4 身份绑定（:893-923）→ p43 B5-B9 内容核验（:921-923 经 `_identity_content_reasons`）→ p43 N5 数值裁决（:924-974） | 三层**原样保留**（:893-974 本次实读确认逐层未动）+ **第四层资格层**（:975-1013）：① 五组齐全否则串 A（:980-988，`STRING_A_NO_CLAIM_RECORD`，claim_record.py:109-114 逐字）；② 一致性对账否则串 B（:990-994）；③ 重算不合格 → 串 C（:995-1013，`STRING_C_UNQUALIFIED_TEMPLATE`，claim_record.py:122-129 逐字，格式化 n/k/np/bs/policy/codes） |
| p=0.001 的地位 | 三层全过即过 G2——方法未校准照样支持晋级（本包要封的缺口） | 数值层**先评估但不再充分**：p=0.001 + 证据未注册 → G2 FAIL（测试 `test_w3_g2_refuses_p_0001_when_no_evidence_registered`，p55:573-589，断言 reasons == 串 C(EVIDENCE_NOT_REGISTERED) 且 promoted=False）；p=0.001 + 覆盖场景 unreliable → 同样 FAIL（p55:592-604，blocking=["ar1_phi04"]）。反之资格层不可能挽救数值层失败（docstring promotion.py:884-887） |
| 版本 | `PROMOTION_RULE_VERSION = "4"` | `"5"`（promotion.py:142；docstring :136-141 记 v5 = WP17 语义） |
| 描述性路径 | — | **不受阻断**：`build_result` 无条件构造五组（§1.1①）；evaluation CLI selfcheck 判定与退出语义不变（evaluation.py:1044-1049 原样）；保存永不因 qualified=False 被拒（p55:366-384 字段级证明） |

### 2.2 对既往晋级判定的适用边界（历史产物不动、不回改，新口径只管今后）

- **历史产物物理不动**：`data/**` 只读红线（任务书【硬性红线】；FROZEN_17.md :27）；本次 `git status` 核实 WP17 改动面不含 data/ 下任何文件（改动清单见 §7 末）。
- **历史 artifact 的未来消费语义**：pre-wp17 产物不含 `claim_record` → 若**今后**拿去走 promotion，G2 以串 A 拒收（"pre-wp17 artifacts cannot support promotion"）；同时 schema 面把 generation 3 归入 legacy（audit-only、never rewritten，artifact_schemas.py:51-61）——即旧产物只作审计读取，**不重写、不重算、不追溯翻案**。
- **既往已作出的晋级判定不回改**：新口径只作用于今后的 `evaluate_challenger` 调用（其生产调用方仅 promotion CLI `main()` 两处与测试，recon §1 :39-41 本次复核无新增——grep `evaluate_challenger` 全仓命中 promotion.py 自身 + tests/test_promotion.py:135 一带）。
- **未来接线约束（登记未实现）**：lifecycle `build_promotion_evidence`（lifecycle.py:1141，当前生产调用方为零）若接线，PROMOTED 证据只能引用 qualified=True 的 verdict——FROZEN §3.4 :322-324 作为冻结约束登记，本包不实现；本次核实 lifecycle.py 的 build_promotion_evidence 函数体无 WP17 改动（working tree 中该文件仅有的 diff 是无关的 P1 fsync 持久化修改，`git diff ashare_model/lifecycle.py` 实读）。

---

## 3 P48 正向标签修正：新旧串逐字对照与 [4.6%, 7.5%] 例（W4）

### 3.1 新旧串逐字对照表

| 项 | 逐字内容 | 出处 |
|---|---|---|
| **旧串（渲染例）** | `Nominal 5% size supported: rate=0.059, 95% CI [0.046, 0.075] contains 0.05.` | 仓外证据 JSON `independent_noise` 行的 claim 字段（**只读冻结历史记录**；本次亲读命令见 §7 命令 5 之后的补充读取） |
| 旧串（代码形态） | `f"Nominal {alpha:.0%} size supported: rate={rate:.3f}, 95% CI [{low:.3f}, {high:.3f}] contains {alpha}."` | FROZEN_17.md §4.1 表 :334（对修改前代码的冻结引文） |
| **新串（代码）** | 见 `ashare_model/stat_calibration.py:185-193`：`f"No significant departure from the nominal {alpha:.0%} level detected: rate={rate:.3f}, 95% CI [{low:.3f}, {high:.3f}] contains {alpha:.0%} — supports non-rejection only; not a verification that the false-positive rate is at most {alpha:.0%} (an interval containing {alpha:.0%} also admits rates above {alpha:.0%})."` | 本次实读；与 FROZEN §4.2 :341-349 冻结逐字一致 |
| **新串（渲染例）** | `No significant departure from the nominal 5% level detected: rate=0.059, 95% CI [0.046, 0.075] contains 5% — supports non-rejection only; not a verification that the false-positive rate is at most 5% (an interval containing 5% also admits rates above 5%).` | 测试精确钉扎 `test_w4_positive_label_numeric_example`（p55:716-729） |
| 其余三分支 | **保留逐字不变**：conservative（stat_calibration.py:196-200）、unreliable_at_nominal（:202-209）、insufficient_samples（:172-176）；测试 `test_w4_other_three_branches_unchanged_verbatim`（p55:741-760）逐串钉扎 | 本次实读 |

**裁定依据**（用户裁定，经 FROZEN_17.md §4.1 :334 与任务书原文）：旧串 "size supported" 措辞过强——CI 包含 5% 只表示未检出与 5% 的显著偏离，支持"未拒绝"，不证明误报率 ≤ 5%；例 [4.6%, 7.5%] 同时容许高于 5%；CI 与检验的对应只支持拒绝/未拒绝，不自动验证上限。

### 3.2 [4.6%, 7.5%] 数值例（测试锚点，本次随 61 passed 亲跑通过）

`tests/test_p55_wp17_claim_qualification.py:716-729`：`calibration_verdict(rejections=59, reps=1000, alpha=0.05)` →

- `verdict == "calibrated"`；`ci_low ≈ 0.0460143`、`ci_high ≈ 0.0753609`（JSON 原值 0.04601429821280047 / 0.07536090277145915，FROZEN §4.3 :370-371）；
- `ci_low < 0.05 < ci_high`（CI 含 5%）**且** `ci_high > 0.05`（上界容许高于 5%）——两行断言把"包含≠上限验证"钉进测试；
- `claim` 与 §3.1 新渲染例**精确等值**；
- 语义断言 `test_w4_no_overclaiming_wording_anywhere`（p55:732-738）：旧措辞 "size supported" 不再出现；"at most 5%" 仅出现在否定从句 "not a verification that …" 之内（`claim.count("at most") == 1`）。

如实记录一处规格内勘误：任务书括注曾写作"Wilson 下界>5% 但 CI 含 5%"，二者逻辑上不能同时成立；冻结文档按用户 W4 正文与实测数值取"**上界** > 5% 且 CI 含 5%"（FROZEN §4.3 :376-382）。

### 3.3 同步面与勘误纪律

- 仓外 JSON 的 claim 串与测量日志 §2 表**原样保留**（只读；`rows_consulted` 只登记身份字段，不搬运 claim 串——claim_record.py:68-74 注释与 p55:316-322 断言双保险）。
- `docs/p48_stat_calibration_draft.md` **追加**"附录：勘误 2 与 W5 功效尺度标注"（:131-158）：勘误 2 声明正向标签语义改为"未检出与名义水平的显著偏离（CI 包含 5%）——支持未拒绝，不构成误报率≤5% 的验证"，判定表四分支不变；`git diff --stat` 实测 **+31/-0**（只追加，原文未删改）。
- 现有 test_p48 11 项不破坏（:139 超集判断容纳第 8 场景；:77/:88/:102 子串在新标签下继续成立；:193 钉 CALIBRATION_VERSION==1 不变）——本次随组合套件 305 passed 亲跑证实（含 test_p48_stat_calibration.py）。

---

## 4 功效尺度：推导、新场景登记与不可外推标注（W5）

### 4.1 冻结推导（FROZEN_17.md §5.1 :403-419，全算术，本包未运行任何新校准）

任务边界取 1% 年化相对超额（任务严格低于此，取上界使场景"与任务相称"）；全部量按 252 个交易日年化（与 ×√252 同口径）：

1. 日均超额植入 μ = 0.01/252 = 3.9683e-5；
2. 日噪声 σ = scale = 0.01（沿用全部既有场景的 scale=0.01，同族连续性）；
3. 日 Sharpe sr = μ/σ = **1/252 = 0.0039683**（精确有理数）；
4. 年化 Sharpe = 1/√252 = **0.062994 ≈ 0.063**；
5. 既有两点等价折算（同 scale）：sr05 → 年超额 0.5×0.01×252 = **126%**、年化 Sharpe 0.5×√252 = **7.9373 ≈ 7.94**；sr025 → **63%**、**3.9686 ≈ 3.97**（与用户给定冻结引用 3.97/7.94 一致）；
6. 差距（写死）：**126 倍 / 63 倍**（闭式 0.5×252=126、0.25×252=63）。

闭式与数值全部被 `test_w5_scale_gap_arithmetic_frozen`（p55:815-823）钉扎（含 `(0.01/252)/0.01 == 1/252`）。

### 4.2 新场景登记（注册不运行）

`ashare_model/stat_calibration.py:94-110`：`SCENARIO_SPECS["task_scale_excess_1pct"]` = `{kind: "power", k_trials: 8, n_dates: 250, sr: 1.0/252.0, scale: 0.01, task_scale: True, note: <§4.3 英文串>}`（字段逐一被 p55:778-786 钉扎）。生成器并入 power 分支（:287-291：`if name in ("power_sr05", "power_sr025", "task_scale_excess_1pct")`，植入式不变 `m[0] += sr*scale`，等价日增 3.9683e-5；确定性植入被 p55:789-798 用同种子对照矩阵验证）。

**命名硬约束（实测依据）**：场景名排序必须在 `short_window_30d` 之后，否则会把后者的 seed_start 从 960000 推走、破坏与仓外 JSON 的对应（recon §9.3 用反例名 `power_task_scale` 实测证实）。落位后 `task_scale_excess_1pct` 取 slice 970000，既有 7 场景 900000..960000 不动——`test_w5_frozen_seed_slices_unchanged`（p55:801-812）逐场景钉扎，本次随 61 passed 亲跑通过。

k=8/n=250 与既有 power 点对齐；相关性形态覆盖沿用 size 家族四形态声明（将来依 W6 运行时须同形态集，FROZEN §5.2 :441-444）。**该场景无证据行**（红线 1 不运行）→ readiness 永远 unlisted/uncalibrated → "无法推断"；将来运行须走 W6（episode purpose 可为 scenario_extension）。

### 4.3 不可外推标注（逐字，两处）

- **代码/注册表英文串**（stat_calibration.py:98-109，与 FROZEN §5.3 :454-463 逐字一致，被 p55:766-786 精确钉扎）：
  > power_sr05/power_sr025 are NOT extrapolable to the task scale: daily Sharpe 0.5/0.25 annualize to 7.94/3.97 (x sqrt(252)), i.e. 126%/63% annualized relative excess at scale=0.01 - 126x/63x the task-scale edge (<1% annualized relative excess; task_scale_excess_1pct plants sr=1/252 ~= 0.00397, annualized Sharpe 1/sqrt(252) ~= 0.063). Detection rates measured at 0.5/0.25 say nothing about detection at the task scale; no power claim is made at the task scale until task_scale_excess_1pct is calibrated under the wp17 W6 protocol.
- **文档中文串**（docs/p48_stat_calibration_draft.md:146-151 追加节内，逐字对应 FROZEN §5.3 中文串 :467-473）：
  > power_sr05/power_sr025 不可外推到任务尺度：日 Sharpe 0.5/0.25 简单年化 ≈ 7.94/3.97（×√252），按 scale=0.01 折算即 126%/63% 年化相对超额，为任务尺度（<1% 年化相对超额；task_scale_excess_1pct 植入 sr=1/252≈0.00397，年化 Sharpe 1/√252≈0.063）的 126 倍/63 倍。在这两点测得的检测率（100%/93%）对任务尺度的检测能力不作任何声明；在 task_scale_excess_1pct 依 WP17 W6 协议校准之前，任务尺度不作任何功效声明。

消费面核实（recon §5 / FROZEN §5.3 :475-480）：无生产代码把高检出率写进结论；仓外 JSON claim 与测量日志 §3 只读保留；p40 时代 C5 功效守卫（tests/test_eval_resampling_calibration.py，反退化回归守卫）与本标注互不改写（本次该文件随组合套件 305 passed 通过）。

---

## 5 方法修订条件触发协议全文与守卫状态（W6：只文档+守卫，零执行）

### 5.1 协议全文（冻结于 FROZEN_17.md §6 :484-564；实现于 `ashare_model/method_revision.py` + 登记文件 `ashare_model/method_revision_registry.json`，均在包内以避开 data/** 只读红线）

**触发判定**（"出现值得正式检验的候选"，四条同时满足才可开 episode；FROZEN §6.1）：

- (a) 候选是某注册 protocol artifact 的 current-run stitched top trial（p41 身份五元组完整）；
- (b) 现行冻结方法下名义统计达标：max_t p ≤ 0.05 且 dsr ≥ 0.95；
- (c) 其 G2 失败**归因于资格码**（W2 原因码表任一——横在候选与晋级之间的是统计资格，而非数据/身份/内容/数值失败）；
- (d) episode 的 target operating points 覆盖该任务实际操作点（n_dates、k_trials、n_perms、block_size、相关性形态集 = §2.4 覆盖集的形态族；n=250、K=8 不外推）。

触发记录一次写全不可改：`episode_id/purpose/opened_utc/artifact(run_id/spec_id/candidate/seed/artifact_sha256)/blocking_reason_codes/target_operating_points/versions(≤2)/frozen_budget_seconds/pilot_seeds(声明为参与调参、永不复用于验收)/acceptance/status ∈ {open, accepted, rejected, budget_exhausted}`。**没有触发记录就开始方法版本比较 → 守卫拒绝（G1）**。

**两版本上限**：一个 episode 最多登记 2 个方法版本（第三个 → G2）；新 episode 须新触发记录且旧 episode 已终态。

**计时与冻结总预算**：比较开始前对每个 target operating point 用合成矩阵计时 20 次 `max_t_from_matrix` 取均值 ms/call（pilot 种子记入 pilot_seeds）；`frozen_budget_seconds = ceil(1.5 × Σ reps_planned × mean_ms / 1000)` 写入触发记录后不可改（1.5 冻结安全系数）；墙钟超限 → G4 中止，状态 budget_exhausted，保留"无法推断"。

**一次独立种子验收**：只选一个版本进入一次新的独立种子验收；fresh 种子块起点 ≥ 1,000,000，与 p40 调参块（10000-50000）及 p48 验证块（900000+100000）显式不相交（重叠 → G5）；判定 = 现行冻结 B4 规则原样（不放宽不改判据），尺寸场景 reps ≥ R_MIN=500；验收后产生新证据注册条目（新 sha256 + 新 max_t_method_version），W2 按新证据重算；**失败保留"无法推断"**：阻断场景仍 unreliable/insufficient → episode rejected，资格维持 not_qualified，**不增加种子直到碰巧通过**（第二次验收请求 → G3；也不得以提高 reps/改判据重试）。

**守卫挂点**（三个可执行面，FROZEN §6.5）：
1. `stat_calibration.calibrate_scenario` 新增 keyword-only `episode_token: str | None = None`（stat_calibration.py:310），函数体第一件事 `require_open_episode(episode_token)`（:325-327）——token 为 None 或非 open/属主匹配 → ValueError(G1)，**默认拒绝 = fail-closed**；
2. 钉扎测试：`MAX_T_METHOD_VERSION != 1` 且注册表无 accepted 终态 episode → 失败（`assert_method_version_endorsed`，method_revision.py:467-482；测试 p55:1038-1048）；
3. 注册表 schema 校验：episodes 追加式、已写字段不可变、种子块不相交**每次读取重验**（`_load` + `_validate_episode` + `_verify_mutual_disjointness`，method_revision.py:288-309、:131-231、:265-285）。

**拒绝串 G1-G5 逐字**（method_revision.py:43-74，与 FROZEN §6.5 :551-557 逐字一致；p55:885-1000 逐串精确钉扎）：G1 无 open 触发记录拒比较/验收；G2 第三版本；G3 第二次验收；G4 超冻结预算；G5 种子块重叠（tuning 10000-50000 / validation 900000+ / 既往 episode）。

**豁免面 = 零豁免**（FROZEN §6.5 :559-564）：无白名单、无 skip 旗标、无环境变量旁路——`test_w6_zero_exemptions_no_bypass_surface`（p55:1051-1060）扫描模块源码断言不含 `skip/exempt/whitelist/getenv/environ`。episode purpose 允许 `method_revision`/`scenario_extension` 两类（后者跳过两版本上限但仍须触发记录+预算+新种子块）——同一纪律的两个登记类别，非豁免（method_revision.py:92-96）。

### 5.2 W6 全量测试面（本次随 61 passed 亲跑通过）

G1 逐字+默认拒绝（p55:885-895）、`calibrate_scenario` 默认守卫（:898-901）、open token 放行/终态或未知 token 拒绝（:904-918）、G2 第三版本（:921-931）、G3 第二次验收（:934-952）、G4 预算边界 3600 过/3601 拒（:955-970）、G5 五组重叠参数化（p40 两块/p48 两块/既往 episode 块，:973-1000）、fresh 不相交块放行（:1003-1016）、open_episode 校验与追加式（:1019-1028）、注册表 schema 读取重验（:1031-1035）、方法常量翻版需 accepted episode 背书（:1038-1048）、零豁免面（:1051-1060）。全部 W6 测试经 `tmp_path` 夹具运行，**不触碰出厂注册表**。

### 5.3 登记文件与零运行证明（本次亲测）

- 出厂注册表 `ashare_model/method_revision_registry.json` 内容 = `{"episodes": []}`（原始文件读取 + `_load()` 调用双途径核实，命令见 §7 命令 6）——**零 episode = 协议从未触发**；
- 默认拒绝实测：`calibrate_scenario("independent_noise", reps=1)`（不传 token）→ `ValueError: method revision refused: no open trigger record exists ...`（§7 命令 6 输出）——**任何校准运行无法在无触发记录下启动**；
- 本包未启动任何校准/重采样运行（红线 1）：`task_scale_excess_1pct` 只注册不运行（§4.2）；W6 全部为守卫与文档；证据 JSON 只读且 sha 钉扎（§7 命令 5）。

---

## 6 审核汇总（三路）

**三路审核在工作流运行内真实执行**（本节运行内事实的出处：工作流主控对本报告撰写员的 escalate 确认返回，2026-09-24）：

- 三名独立审核子智能体及章程：**审核员 A——闭环与绕行**（声明闭环完整性与绕行面）、**审核员 B——统计措辞与换算**（W4 标签语义与 W5 功效尺度算术）、**审核员 C——触发协议与范围**（W6 协议与红线范围）；
- 运行日志权威摘要（逐字，经主控确认）：**"三路审核：3/3 通过，8 条发现，must-fix 0"** 与 **"审核结论：无需必改项"**——三份章程全部通过（无 blocker、无 major），8 条发现**全部 minor**，故 must-fix=0、修复/复核轮未派发（该阶段 0 步）；
- **留痕缺口（如实声明）**：审核发现的明细（各条标题与内容）仅存在于运行内部的结构化返回值，**没有写出任何磁盘记录文件**——本报告撰写员以 `find`/`grep` 查遍 `reports/wp17/`、`docs/`、两处 `.mimosa/reports/` 与全部红证据文件均无命中（最新 task-review 磁盘记录停在 2026-09-23 16:09，早于 WP17 实现）后向主控确认属实。**8 条 minor 的具体条目不可从磁盘恢复，本报告只报计数、不虚构任何一条内容**；该缺口已登记为流程改进项（§8 第 5 条）。

可核实的磁盘处置证据（撰写员本次亲核）：红证据 `reports/p55_red_evidence.txt`（RED 逐字 + GREEN 四命令 + 11 条旧钉扎重锚记录，其中 7-11 号是冻结表 §7.2 未预见、由执行/核查发现并按冻结语义处置的条目——见 §7 引用）；docs 两处追加（勘误 2 + W5 标注，+31/-0）；质量门实测（§7）。

---

## 7 质量门记录（通过）与红证据位置

### 7.1 本次会话亲跑（2026-09-24，cwd=D:/minequant/AlphaGPT）

| # | 命令（逐字） | 输出（逐字尾部） |
|---|---|---|
| 1 | `python -m pytest tests/test_p55_wp17_claim_qualification.py -q` | `61 passed in 4.72s` |
| 2 | `python -m pytest tests/test_p55_wp17_claim_qualification.py tests/test_promotion.py tests/test_p41_candidate_evidence_binding.py tests/test_p43_statistical_input_integrity.py tests/test_p48_stat_calibration.py tests/test_artifact_writer.py tests/test_artifact_schemas.py tests/test_eval_resampling_calibration.py tests/test_p40_statistical_resampling.py tests/test_run_store.py tests/test_eval_module_split.py -q` | `305 passed in 58.06s` |
| 3 | `python -m pytest tests/test_evaluation.py tests/test_stitched_oos.py -q` | `72 passed in 513.19s (0:08:33)`（收集数独立核实：`--collect-only` 51+21=72） |
| 4 | `python scripts/check_test_shards.py` | `shard union == full set: 138 files across 4 shards cover all 138 test files`（p55 登记于 scripts/check_test_shards.py:179-182，p48 条目邻近，实读） |
| 5 | `python -c "...hashlib.sha256(open('D:/minequant/reports/p48/calibration_report.json','rb').read())..."` | sha256 = `b85a652705f8768fe5629473de5aac7841b575d2ff46617388e6a49c1b296d0e`，6467 字节，与 `CALIBRATION_EVIDENCE_REGISTRATION[0]["sha256"]` **match: True**；同命令顺带读出 7 场景 verdict/seed_start（ar1=unreliable 0.095@900000 … short=unreliable 0.108@960000，与 §1.2/§2 引用一致）及旧 claim 串（§3.1 引用） |
| 6 | `python -c "..._load(); calibrate_scenario('independent_noise', reps=1)..."` | episodes = `[]`；默认守卫触发 `ValueError: method revision refused: no open trigger record exists ...`（§5.3 零运行证明） |

命令 1-4 全部 exit code 0。**计数出入如实报告**：红证据 GREEN 命令 3（同一命令）记录 `128 passed in 473.83s`，当前树 `--collect-only` 为 72；两次均通过（exit 0），差异无法从磁盘复核（WP17 改动未提交、无中间快照），如实登记不改写（§8 第 6 条）。

### 7.2 红证据位置与内容

`D:/minequant/AlphaGPT/reports/p55_red_evidence.txt`（只追加）：RED 逐字（2026-09-23，ImportError: cannot import name 'MAX_T_METHOD_VERSION'——测试先于实现）；GREEN 实现摘要（新 claim_record.py/method_revision.py/注册表/p55 测试/MAX_T_METHOD_VERSION，改 stat_calibration/promotion/eval_artifacts/evaluation/artifact_schemas/check_test_shards/docs）；**11 条旧钉扎重锚记录**（每条注明 FROZEN_17 依据）：1-6 对应 FROZEN §7.2 表行 1-6（test_promotion 四处 promoted=True 正例的 fixture 扩为 8 候选×250 共同日期×max_t_perms=999 + 内存证据注册表；版本钉 "4"→"5" ×2；test_p43 N5 正控制注入证据），7-11 为冻结表未预见、按冻结语义处置的新发现（test_p43 B9 正控制 :459-468 因 k=1/n=500/n_perms=5000 落入未校准域而破——按 §3.1 正路径需资格的语义重锚；test_artifact_writer 两处与 test_artifact_schemas 一处为 §7.1 schema 3→4 的连带钉扎推进；test_promotion CLI 参数化 id "unknown-v3" 更名 "legacy-v3"，断言零改动）；"Not re-anchored" 节记录 FROZEN §7.2 行 7-9 不动项的执行核实；GREEN 四命令输出尾部；证据 sha256 复核。

重锚的磁盘实证（本次抽查实读）：`tests/test_promotion.py:43-101`（8 试次×2 折×125 会话=250 日期轴 fixture，docstring 注明依据 FROZEN_17 §7.2 row 1）、:104-132（`_calibration_evidence` 内存注册表，期望值手推声明）、:1144 与 :1209（版本钉 "5"）、`tests/test_p41_candidate_evidence_binding.py:255`（`test_j6_promotion_rule_version_is_5`）。重锚后的全部套件随 §7.1 命令 2 通过。

### 7.3 改动清单（本次 `git status`/`git diff --stat` 核实；基线 dbaf729，WP17 未提交）

新增（untracked）：`ashare_model/claim_record.py`、`ashare_model/method_revision.py`、`ashare_model/method_revision_registry.json`、`tests/test_p55_wp17_claim_qualification.py`、`reports/p55_red_evidence.txt`、`reports/wp17/`（FROZEN_17.md、_recon/map.md、本报告）。修改：`docs/p48_stat_calibration_draft.md`（+31/-0）、`ashare_model/stat_calibration.py`、`promotion.py`、`eval_artifacts.py`、`evaluation.py`、`artifact_schemas.py`、`eval_corrections.py`（+MAX_T_METHOD_VERSION）、`scripts/check_test_shards.py` 及重锚测试文件。注：working tree 同时含其他工作包的未提交改动（如 lifecycle.py 的 P1 fsync 修改，与 WP17 无关，已甄别）。

---

## 8 遗留与 deferred

1. **资格为否的历史统计结论如何重述（边界说明）**：历史产物（pre-wp17 artifact、既有 verdict/promotion 记录）**不回改、不重算、不追溯翻案**（data/** 与既有 reports 只读；schema 把 generation 3 归入 legacy 仅审计读取，artifact_schemas.py:51-61）。新口径只管今后：历史 artifact 若再被送去晋级，将以串 A 拒收（不携带五组声明记录，无法支持晋级）；对历史结论的**今后引用**应按 §3.4 冻结模板重述——描述性统计量照常引用，名义 5% 显著性与晋级声明一律以"No reliable inference: … (reason codes)"形式收回（模板 claim_record.py:132-141；含 p 极小值场景，FROZEN §3.4 :313-319）。既有仓外 JSON/测量日志中的正向标签属冻结历史记录，原样保留（§3.3）。
2. **默认操作点全数不合格的常态**：生产默认 n_perms=5000 ≠ 证据 999（EVIDENCE_IDENTITY_MISMATCH），且即便对齐，n=250 被 ar1_phi04（9.5% [7.8,11.5]）、n=30 被 short_window_30d（10.8% [9.0,12.9]）以 SCENARIO_UNRELIABLE_AT_NOMINAL 阻断（§1.2，测试类 TestFrozenEvidenceBlocksEveryOperatingPoint 亲跑通过）。这是测量日志 §4.2"正式统计结论与晋级声明冻结"的机器可执行形式，**不是缺陷**；出路只有 W6（修方法 + 新不相交种子块重校准）或以 999 重测并扩证据注册表。
3. **lifecycle PROMOTED 证据接线约束**：`build_promotion_evidence` 只能引用 qualified=True 的 verdict——冻结约束已登记（FROZEN §3.4 :322-324），本包不实现（该函数当前生产调用方为零，lifecycle.py:1141 实读无 WP17 改动）。
4. **任务尺度功效未知**：`task_scale_excess_1pct` 已注册未运行 → 任务尺度（<1% 年化相对超额）检出能力"无法推断"；sr025/sr05 检出率不可外推（§4.3 标注写死）。运行须走 W6（scenario_extension）。
5. **审核留痕流程缺口**（本次三路审核暴露，主控确认）：审核明细只存在于运行内部返回值，无磁盘记录。流程改进：后续工作包应要求审核员将 findings 逐条写盘（如 `reports/wpNN/audit_findings.md`），避免审计留痕只存在于运行内存。
6. **红证据计数出入**：GREEN 命令 3 记录 128 passed，当前树同命令收集 72（§7.1）——两次均通过；因改动未提交无法复核当时树状态，如实登记。
7. **提交纪律**：WP17 改动（含 WP14-WP16 遗留的未提交改动混合在同一 working tree，基线 dbaf729）尚未形成 commit；分片登记要求"与实现同 commit"（FROZEN 红线 4），提交动作留给集成方，提交前建议重跑 §7.1 四命令。

---

*报告撰写员核实方法备注：本报告全部代码行号引用自 2026-09-24 对当前工作树的实读；全部命令输出为当日亲跑；无法亲跑或磁盘不存在的事实（三路审核明细、红证据记录时点的树状态）已如实标注出处或列为不可复核项，未以推测填补。*
