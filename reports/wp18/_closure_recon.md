# WP18 收口侦察（p57 前置，只读）

侦察日期：2026-09-27。本文件为 p57 收口包的现场核验记录，**零改动**：未跑任何评价行/回测/评分，未触碰 data/** 既有内容，未触发方法修订。所有引用均为本会话亲读/亲跑（命令随条目附注）。

---

## 1. slow 成本压力 skip 记录与未运行行键保全（R1）

### 1.1 skip 记录在盘（两处独立载体，本会话亲读）

- **`reports/wp18/slow/cost_stress.json`**（`python -c "import json; ..."` 解析）：
  - 顶层 `row_count=0`、`failed_rows=0`、`rows=[]`、`generated="wp18-c2-20260927"`、`development_validation="开发验证（development validation, 2021-2025 测试窗）"`；
  - `screen_adjudication.A.skip_reason`（逐字）= `"FROZEN_18_PLAN §5: conditions i_3_of_5_positive+ii_median_positive not met — no 0x/2x stress rows for this rule (研发资源分配门槛)"`；A 的 screen：`n_positive_years=1`（门槛 3）、`median_net_excess=-0.057167090715069646`、`dynamic_minus_hold_annualized=0.026558626302869315`（条件 iii 单过）；
  - `screen_adjudication.B.skip_reason`（逐字）= `"FROZEN_18_PLAN §5: conditions i_3_of_5_positive+ii_median_positive+iii_dynamic_minus_hold_50bp not met — no 0x/2x stress rows for this rule (研发资源分配门槛)"`；B：`n_positive_years=1`、`median=-0.06187716155980438`、`dynamic_minus_hold_annualized=-0.009492098360277224`。
- **账本** `data/r1/wp18_slow/runs/wp18_c2/wp18-c2-20260927/trials.jsonl`（共 6 行，本会话 `wc -l` + `grep -n "screen:1x_initial"`）：
  - 第 1–2 行 `algorithm="screen:1x_initial"`、`candidate="slow:combo:slow_all5:A"`（running+succeeded 一对，trial_id `wp18-c2-20260927-00001`），`payload.skip_reason` 与 cost_stress.json 逐字一致，`payload.stress_rows_authorized=0`，`metrics.stress_rows_authorized=0`；
  - 第 3–4 行同构，`candidate="slow:combo:slow_all5:B"`（trial_id `...-00003`）。
- 旁证：`reports/wp18/slow/run_meta.json` 的 `stress_screen.{A,B}.skip_reason` 同串、`stress_rows=0`、`factor_cache_outcome="not_applicable(no stress row executed)"`、`c2_elapsed_sec=0.1`。

### 1.2 未运行行键在计划中保全——**数量口径澄清（重要）**

任务书写"60 个已物化未运行的行键"；盘面事实是 **60 = medium C1:A 的 20 + slow C2 的 40**（与 WP18_REPORT.md:194 表述一致），slow 单域是 40：

- **slow 侧（40）**：`reports/wp18/slow/integrity/plan.json`——56 臂、`expected_row_count=280`；其中 **8 条 C2 臂**（`C2:{A,B}:{dynamic,hold}:combo:slow_all5` × cost_levels `["0x"]`/`["2x"]`，成员=slow 五因子）已**物化在计划中**。`reconcile_full.json`：`n_expected_keys=280, n_actual_keys=240, n_missing=40, n_duplicates=0, n_unexpected=0, n_failed_rows=0`，`missing_by_stage={"C2": 40}`，40 个 missing 行键逐一在 `missing` 列表（如 `C2:A:dynamic:combo:slow_all5|0x|2021-12-31`）。`missing_adjudication`（逐字要点）：`"the missing rows are EXACTLY the §5-screened-out stress arms (planned at the 40-row upper bound, §4.5; the skip decisions with full evidence live in cost_stress.json screen_adjudication)"`。`status_full.json`：56 条 arm_status 中 48 citable + **8 missing（即全部 C2 臂，n_expected_rows=5×8=40, n_present_rows=0, failure_reasons=[]）**——保全供未来轮次，非失败。
- **medium 侧（20）**：`reports/wp18/medium/integrity/reconcile_full.json` `n_missing=20, missing_by_stage={"C1": 20}`（= 仅 B 过初筛后 C1:A 的 0x/2x 行未获授权）；`status_full.json` 36 citable + 4 missing，`label=measurement_incomplete`。
- 冻结计划授权链：`FROZEN_18_PLAN.md:181`（C1/C2 条件性行集：仅 A 通过⇒20 行、A/B 全过⇒40 行、全不过⇒0 行）与 `:201`（"通过 ⇒ 该 (域, 主组合, 规则) 补跑 0x 与 2x（§4.5 行集内）；不通过 ⇒ 不补"）。

### 1.3 measurement_incomplete 语义

两域 `integrity/status_full.json` 均为 `label="measurement_incomplete"`、`verdict_valid=false`、`adjudication_as_recorded=null`。语义锚点（WP18_REPORT.md:38，本会话核对）：这是**初筛拒绝授权的设计内状态，不是测量失败，不构成经济负结果的证据**；缺行恰为未获授权的压力行。收口报告照此表述，不得写成经济负结果。

---

## 2. E8 现场（reward_baskets 拒绝消息文案，R5）

- 常量：`ashare_model/reward_baskets.py:35` `_P52_C1_TOLERANCE = 1e-12`（booking 精度）；`:51` `_P52_C1_REFUSANCE_TOLERANCE = 1e-9`（E2 后的拒绝门，注释 :36-50 详述双常量拆分）。
- **在役比较**：`:562-565`
  ```python
  if (
      shortfall > _P52_C1_REFUSANCE_TOLERANCE
      and ledger_physical[row]
  ):
  ```
  即 :563 使用 **`_P52_C1_REFUSANCE_TOLERANCE`**（1e-9）。**收口修复不得动这四行。**
- **消息现场**：`:577-585` 的 `raise ValueError(...)`；其中 `:582` 为
  ```python
  f"{1.0 + _P52_C1_TOLERANCE!r} (WP16-A FROZEN_A §1 (C1)/"
  ```
  本会话实测渲染（`python -c`，双常量代入）：当前消息打印 **`1.000000000001`**（1e-12 形），而对齐在役门槛应打印 **`1.000000001`**（1e-9 形）——差三个数量级，与 errata E8 的 finding 一致。
- 追认挂点：`reports/wp16/FROZEN_A.md` 现末节为 `§8.3`（:325，文件共 336 行）——**§8.4 为下一个空闲附注编号**，E8 收口的"消息披露对齐、无行为变化"附注追加于此（只追加）。
- 在案登记：`reports/wp18/errata/errata.jsonl` 第 8 条 `E8-known-issue-reward-baskets-refusal-message-wording`（severity `known_issue_deferred_to_future_package`；finding 注明 20 条 slow 拒绝行账本文本均带 1e-12 形文案；disposition 注明推迟理由=p52 家族红证据状态不动）。p57 的 RED 测试断言当前消息含 `1.000000001` → 先失败，再改 :582（或其等价消息行）→ GREEN。

---

## 3. claim_record.py 公开 API 与 WP17 声明措辞契约（R2）

### 3.1 公开 API（`ashare_model/claim_record.py`，529 行，本会话通读）

| 符号 | 行号 | 签名/要点 |
|---|---|---|
| `build_claim_record` | :281 | `(artifact_max_t: dict|None, artifact_dsr: dict|None, stitched_trials: list, max_t_perms: int, *, evidence: Any = _AUTO) -> dict`——组装五组字段并内嵌资格结果；`evidence` 缺省 `_AUTO`（自动加载登记文件），**不可得 ⇒ qualified=False + EVIDENCE_NOT_REGISTERED，从不是写阻断** |
| `qualify_claim_record` | :346 | `(record: dict, evidence) -> dict`——W2 算法，返回 `{qualified, status, reason_codes, reason, covering_scenarios, blocking_scenarios}`；`qualified = (reason_codes == [])` fail-closed |
| `claim_record_consistency_reasons` | :435 | `(record, artifact) -> list[str]`——G2 一致性校验（组 1–3 逐字段） |
| `claim_record_stdout_lines` | :513 | `(record) -> list[str]`——CLI 两行输出（qualified=False 追加 no-reliable-inference 行） |
| `CalibrationEvidenceRegistry` | :198 | `from_file(path)` :220（实测 sha256）/ `from_payload(payload, *, registration=..., measured_sha256=...)` :237（`registration=None` ⇒ 无登记 ⇒ EVIDENCE_NOT_REGISTERED） |
| 原因码 | :78-86 | 9 枚冻结枚举；**`EVIDENCE_NOT_REGISTERED = "EVIDENCE_NOT_REGISTERED"`（:78）** |
| G2 拒绝串 | :109-129 | STRING_A（无记录）/ STRING_B（不一致模板）/ STRING_C（资格失败模板） |
| 结论渲染 | :132/:144 | `no_reliable_inference_reason(codes)` / `qualified_claim_reason(record)` |

**p57 演练注意**：要稳定得到 `qualified=False + EVIDENCE_NOT_REGISTERED`，须**显式传 `evidence=None`**（如 `build_claim_record(..., evidence=None)`）——缺省 `_AUTO` 会去读登记文件 `D:/minequant/reports/p48/calibration_report.json`（:58-66 登记，sha256 `b85a6527…`），该文件在盘时走的是后续步骤的码（如 N_DATES/K 越域），而非 EVIDENCE_NOT_REGISTERED。保存不阻断由 :289-295 docstring 与 `_auto_evidence` :266-275（`except Exception: return None`）共同保证。

### 3.2 措辞契约位置（供审计词表引用，`reports/wp17/FROZEN_17.md`，638 行）

- **§3.3（:274-296）**：G2 冻结逐字串 A/B/C。
- **§3.4（:297-327）**：正式结论模板——合格分支（仅 qualified=True 可用：`max-t p={p:.4f} is qualified for a nominal-5% significance claim …`）；**无法推断分支**（qualified=False 的一切正式结论：`No reliable inference: … descriptive statistics are retained, and no nominal-5% significance or promotion claim is made.`）；CLI stdout 两行。
- **§4.1-§4.4（:330-399）**：P48 正向标签措辞——`calibrated` 新串（`No significant departure from the nominal 5% level detected … supports non-rejection only; not a verification that the false-positive rate is at most 5% …`）；`conservative`/`unreliable_at_nominal`/`insufficient_samples` 三串保留逐字；**"at most 5%"只许出现在否定从句内**；仓外 JSON 旧串只读保留、不搬运进新报告。
- 辅助：§5.3（:450）不可外推标注串；§6.5（:540-566）方法修订拒绝串 G1-G5。
- 推断性模式词表（审计用，源自上述契约）：禁用/需资格背书的形态——`nominal-5% significance`、`qualified for … significance claim`、`size supported`、`误报率不超过/at most`（肯定句）、`promotion/PROMOTED`、样本外推断措辞；允许的描述性形态——`No reliable inference`、`supports non-rejection only`、`not a verification`、`descriptive statistics are retained`、`development validation/开发验证`、`研发资源分配门槛，非统计显著性证明也非市场规律`。
- WP18 自报状态（ANALYSIS.md:275-281 §10）：未生成任何 ClaimRecord、未运行 max-t 资格审定、§1–§9 全部为开发验证描述性比较——**这是 p57 机械扫描要验证的对象面声明**。

---

## 4. 方法修订守卫（R3）

- 注册表：`ashare_model/method_revision_registry.json` = `{"episodes": []}`（3 行，本会话 cat 亲读）——**零触发记录**。
- 守卫入口：`ashare_model/method_revision.py:323` `require_open_episode(episode_token, *, path=None)`——token 非 str/为空/不匹配任何 open episode ⇒ `raise ValueError(G1_REFUSED)`（:329-336）。
- G1 拒绝串（逐字，`:44-50`）：`method revision refused: no open trigger record exists for this work - register the candidate worth formal testing (artifact identity + blocking reason codes) before any method-version comparison or acceptance calibration starts (wp17 W6)`（G2-G5 见 :51-74 与 FROZEN_17 §6.5）。
- 挂点：`ashare_model/stat_calibration.py:310` `calibrate_scenario(..., episode_token: str|None = None)`，:325-327 import 并调用 `require_open_episode(episode_token)`；FROZEN_17 §6.5 载明**零豁免面**。
- **活性实测（本会话，只读）**：`python -c "from ashare_model.method_revision import require_open_episode; ..."`——`require_open_episode(None)` 与 `require_open_episode('nonexistent-episode')` 均抛 ValueError，消息以 G1 串开头。守卫处于在役拒绝态，可正式记录关闭。

---

## 5. E3 缓存条目与字节同一性（§9.6 处置）

- 现场路径：`data/r1/slow_ab/factor_tensor_cache/`（`ls -la` 亲读）——两条目：
  - `4ab7587af59df64c45a37f9ccf4fbc7b6d1a2166b8e755f1a6ce123ef9e4210d.npz`（1,684,391,058 B，mtime 2026-09-21——既有条目，未被改）；
  - `ed8dc37c37eecfe36af9f5bcf09effa1c8a7c3170a574091588e2b052cd851f0.npz`（同尺寸，mtime 2026-09-27 17:35——E3 披露的新增条目）。
- **字节同一性核验方法（内容寻址键对比）**：同一内容寻址键 `ed8dc37c…npz` 在三处目录各有一份（`data/r1/slow_ab/`、`data/r1/medium_a/`、`data/r1/wp18_slow/` 的 `factor_tensor_cache/`，`ls` 亲证）；对三份文件逐块 sha256。
- **本会话已实测**（`python -c` + hashlib，1MiB 分块）：三处均为 `81db3d19f49424ddb51734519ff4e99284c5b694dce3116d00fd59ce9f4d8b4f`，与 errata E3（`reports/wp18/errata/errata.jsonl` 第 3 条）登记的 sha256 逐字一致。**E3 可转为"核实过的无操作"**：不删除（data/** 零改动红线），核验证据即收口物。
- 佐证链：`stage12_result.json:133`（E3 摘要）、账本 `data/r1/wp18_slow/runs/wp18_stage12/wp18-stage12-20260927/trials.jsonl:151-152`（`budget:factor_cache_miss_correction`，`cold_rebuild_seconds=3120, existing_files_modified=0`）、ANALYSIS.md:41、WP18_REPORT.md:68/:196。

---

## 6. WP18 产出完整清单（声明扫描对象面，R2）

人读文本面（扫描优先级最高）：
- `reports/wp18/FROZEN_18_PLAN.md`（333 行）、`WP18_REPORT.md`（222 行）、`ANALYSIS.md`（298 行，§0–§12，其中 §10 为 WP17 资格状态自报）、`_recon/map.md`（175 行）、`errata/errata.jsonl`（9 条 E1–E9）。
- `reports/wp18/_analysis/`：`derive_static.py`（255 行）、`derived.json`（~100 KB）、`append_errata_fixer.py`（298 行）、`replay_medium.json`（2.1 MB）、`replay_slow.json`（3.0 MB）——后两者为机器重放 JSON，扫描按注记字段机械化处理。

结构化产物面：
- `reports/wp18/precheck.json`、`stage12_result.json`、`integrity_stage12_summary.json`；
- `reports/wp18/medium/`：`direction_frozen.json`、`signal_summary.json`、`account_report.json`、`run_meta.json`、`cost_stress.json`（row_count=20，B 过初筛）+ `integrity/`（plan、plan_stages12/123/123c、precheck_medium、reconcile_×4、status_×4，共 12 文件）；
- `reports/wp18/slow/`：同名五件（cost_stress row_count=0）+ `integrity/` 12 文件；
- `data/r1/wp18_medium/`：`runs/wp18_preregistration|wp18_stage12|wp18_m3|wp18_c1`（trials.jsonl 行数 **18/64/182/46**，本会话 `wc -l` 亲数）、`wp18_medium_report.json`、`wp18_medium_run_meta.json`、两份 `.log`；
- `data/r1/wp18_slow/`：`runs/wp18_stage12|wp18_s3|wp18_c2`（**156/262/6** 行，同上亲数）、`factor_tensor_cache/`（5 个 npz）、`wp18_slow_report.json`、`wp18_slow_run_meta.json`、三份 `.log`。
  （七账本行数与 WP18_REPORT 附录 A #8 的 18/64/156/182/262/46/6 完全一致。）

代码/测试面（产出物，非扫描重点但属对象面）：
- `ashare_model/wp18_five_questions.py`、`ashare_model/wp18_signal_vs_account.py`；
- `scripts/wp18_{stage12,m3_account,s3_account,cost_stress,reconcile_stage12,reconcile_m3,reconcile_s3,reconcile_stress,register_trials,analysis_replay}.py`（10 个）；
- `tests/test_p56_wp18_round.py`；`reports/p56_red_evidence.txt`（2026-09-27 节）。

---

## 7. 17 条审核明细的盘面残留搜索（R4 前置）

**结论：无逐条明细残留。** 在盘只有（a）运行级摘要与（b）处置结果。搜索过程（本会话全部执行）：

1. `.mimosa/reports/` 2026-09-27 两份 task-review JSON——均为 hook 扫描器 `inconclusive` 运行（scanner ETIMEDOUT / baseline missing），`detailed_findings=0`，与三路审核无关。
2. `.mimosa/hook-state/` 全量 grep `must-fix|must_fix`——命中仅为 **WP14/WP15/WP16 工作流脚本的模板代码**（`must-fix ${mustFix.length}` 日志语句；三个唯一内容哈希 2438cf…/6d23e3…/8864ab…，head 亲验分别为 WP16/WP14/WP15 脚本），非 finding 明细。
3. WP18 运行身份确认：`.zcode/workflow-runs/dwfrun-b6ac411e-4886-413d-b4f0-f78255fe7b51.mjs` payload 即 WP18 脚本（三路审核=actor#10 ×3 并行，`allFindings.flatMap`/`mustFix` 逻辑在脚本内）。其审核员会话目录 `C:/Users/Administrator/.zcode/cli/exec/sess_dwf-dwfrun-b6ac411e-…-actor_10_{1,2,3}/`：仅 actor_10_1 有一份 139,641 B 工具 stdout 日志（账本罗列输出；`grep -c severity`=0，blocker/major/minor 零命中），10_2/10_3 为空。**明细只存在于工作流运行时日志，未落盘。**
4. `reports/`+`docs/` 文件名扫 `*finding*|*audit*`——无 WP18 相关件；`审核员A/B/C` 字样仅出现在 WP18_REPORT.md:168（汇总句）。
5. 在盘处置结果（非明细）：errata E4–E9、ANALYSIS 更正（E5/E6）、账本补记（`domain_gate:backfill`、`ledger_note:intermediate_attempt_attribution`、`budget:factor_cache_miss_correction`）。

与 WP18_REPORT §9 第 5 条及附录 A"未能执行的核查"自认一致。**p57 的 `docs/p57_audit_findings_persistence.md`（含模板）即对此缺口的流程固位；本包审核员按其执行（吃自己的狗粮）；收口报告须如实重申：17 条明细不可复原，只能登记处置面。**

---

## 附：与任务书的口径差异（须在收口报告如实表述）

1. **"60 个未运行行键"**：slow 单域为 40；60 = medium C1:A(20) + slow C2(40)（WP18_REPORT.md:194 原口径）。收口措辞建议写"两域合计 60（20+40），行集已在各自 integrity/plan.json 物化"。
2. **ANALYSIS.md §10 的五组字段别名**：该节把五组写作 `statistics/multiplicity/operating-point/structure/qualification`，而 FROZEN_17 §1.1 与 claim_record.py 的冻结键名为 `statistics/method/task_actuals/calibration_evidence/qualification`——描述性措辞漂移（非推断性声明），p57 声明扫描会命中；按红线以 **errata 追加**处置，不改写 ANALYSIS 正文。
