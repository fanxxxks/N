# WP17 冻结规格（FROZEN_17）——校准资格声明闭环

- 状态：**FROZEN，实现前冻结**（2026-09-23）。本文件是 WP17 实现期的**唯一期望真源**：
  实现者/修复器对 W1-W6 的任何取舍以本文为准；本文与实现输出冲突时，以本文为准并停下回报。
- 依据材料（本文件起草时全部实读）：
  - `ashare_model/stat_calibration.py`（全文实读；关键行：:25 CALIBRATION_VERSION=1、
    :36-37 R_MIN=500/CI_MAX_WIDTH=0.04、:42-52 种子块、:57-84 SCENARIO_SPECS、
    :120-182 calibration_verdict、:185-213 significance_readiness、:216-254 生成器、
    :257-262 scenario_seed_start）；
  - `ashare_model/promotion.py`（:128 PROMOTION_RULE_VERSION="4"、:735-837 G1、
    :808-815 regime 对象消费、:840-947 G2、:1163-1253 evaluate_challenger）；
  - `ashare_model/eval_corrections.py`（:146 BENCHMARK_DEFINITION、:168-203
    _identity_blocks、:362-368 max_t_from_matrix 签名、:478-496 _adjudication_window、
    :499-584 _max_t_from_pool、:588-601 max_t_from_rows 默认 n_perms=5000）；
  - `ashare_model/eval_artifacts.py`（:164 max_t_perms=5000 默认、:292-305 产物字段、
    :310 model_validate）；`ashare_model/eval_metrics.py`（:713-719 failed_folds）；
  - `ashare_model/evaluation.py`（:1004-1011 write_boundary_artifact、:1026-1037 统计量
    stdout、:1038-1043 selfcheck）；
  - `docs/p48_stat_calibration_draft.md`（§2 B1/B2/B4/B5）、
    `docs/p48_stat_calibration_measurement_log.md`（§2/§3/§4/§7）、
    `docs/p40_statistical_resampling_draft.md`（:95-96 C5 功效守卫）；
  - 仓外证据 `D:/minequant/reports/p48/calibration_report.json`（实测读取，见 §9）；
  - `reports/wp17/_recon/map.md`（侦察图，本包起点）。
- 红线重申（对实现者具有约束力）：
  1. **不启动任何新的校准/重采样运行**。声明闭环=接线与登记；既有校准证据按身份原样消费，
     不得回溯发明版本号。
  2. `data/**` 与既有 reports 产物只读；冻结文档（p48 draft/log）只读，勘误只追加；
     红证据只追加。
  3. 因 G2 收紧而变的旧钉扎测试，逐个按本文重锚并记录（依据=本文，非实现输出）。
  4. 本包新产物一律写绝对路径 `D:/minequant/AlphaGPT/reports/wp17/`；新测试前缀 p55，
     并与实现同 commit 登记进 `scripts/check_test_shards.py` 的分片元组
     （p48 在 :179，p53/p54 在 :173-178 邻近——实读核实）。

---

## 1 W1 —— ClaimRecord 五组字段（精确模式）

正式报告 = ① protocol artifact（`eval_artifacts.build_result` 产物）、② promotion
verdict（`data/promotion_verdict.json`）、③ evaluation CLI stdout。三者都必须保留
以下五组字段。载体：artifact 顶层新键 **`claim_record`**（dict），由 build_result 在
构造 dsr/max_t 的同一处（eval_artifacts.py:300-303 之后）一并构造；verdict 由
`evaluate_challenger` 携带 `"claim_record"`（artifact 记录的拷贝）；CLI stdout 追加
资格行（§3.4）。schema：`ProtocolResultArtifact` 增加
`claim_record: dict | None = None`（additive、带默认，旧产物仍可校验），
`ARTIFACT_SCHEMA_VERSION` 3→4。

新模块 **`ashare_model/claim_record.py`**，两个入口函数：

- `build_claim_record(artifact_max_t, artifact_dsr, stitched_trials, max_t_perms) -> dict`
  —— 只聚合、只登记，不做资格判定；
- `qualify_claim_record(record, evidence) -> dict` —— §2 的 W2 算法，产出第五组。

### 1.1 字段表（键名 / 类型 / 来源函数）

```
claim_record = {
  # ── 第一组：原始统计量与 p 值 ──────────────────────────────
  "statistics": {
    "dsr":                   float | None,   # artifact["dsr"]["dsr"]
                                           # （源 _dsr_from_pool，eval_corrections.py:234-）
    "max_t_observed":        float | None,   # artifact["max_t"]["observed_max_t"]
    "max_t_p_value":         float | None,   # artifact["max_t"]["p_value"]
    "max_t_significant_95":  bool | None,    # artifact["max_t"]["significant_95"]
  },
  # ── 第二组：方法版本与参数 ─────────────────────────────────
  "method": {
    "max_t_method_version":  int,     # 新常量 MAX_T_METHOD_VERSION（见 §1.3）
    "calibration_version":   int,     # stat_calibration.CALIBRATION_VERSION（=1）
    "n_perms":               int,     # artifact["max_t"]["n_perms"]（实际所用）
    "block_size":            int,     # artifact["max_t"]["block_size"]（实际所用）
    "dsr_method":   "analytic_bldp_uncalibrated_v1",
    # p48 draft §2 B1 + 测量日志 §7：DSR 是解析式副统计量，声明未校准；
    # 资格判定只作用于 max-t（晋级口径的绑定输入）。
  },
  # ── 第三组：实际共同日期数 / 候选池大小 / 缺折政策 ──────────
  "task_actuals": {
    "n_common_dates":  int | None,   # artifact["max_t"]["window"]["n_dates"]
                                     # （源 _adjudication_window，eval_corrections.py:478-496）
    "n_pool_trials":   int | None,   # artifact["max_t"]["trial_set"]["n_trials"]
                                     # = n_current + n_extra（多重计数全域）
    "n_current_trials": int | None,
    "n_extra_trials":  int | None,
    "matrix_excluded": list[dict],   # trial_set.matrix_excluded（window_mismatch 明细）
    "n_failed_folds_total": int,     # Σ stitched trials 的 failed_folds
                                     # （源 eval_metrics.py:713-719）
    "missing_fold_policy": "structural_no_imputation_no_zero_padding_v1",
    # 冻结的结构性事实描述符（唯一注册值，见 §1.2）：全折失败→整个 trial 落选
    # （eval_metrics.py:597-598）；基准缺失/失配 raise、绝不零垫（p41 B1）；
    # 不覆盖裁决窗 W 的 trial 剔出矩阵但保留在多重计数（eval_corrections.py:534-543）。
  },
  # ── 第四组：校准证据身份 ───────────────────────────────────
  "calibration_evidence": {
    "source":  "D:/minequant/reports/p48/calibration_report.json",
    "sha256":  "b85a652705f8768fe5629473de5aac7841b575d2ff46617388e6a49c1b296d0e",
    "rows_consulted": [               # 每行原样登记 JSON 现有字段，不增不改
      {"scenario": str, "kind": str, "n_perms": int, "block_size": int,
       "seed_start": int, "reps": int, "verdict": str}, ...
    ],
    "method_version_attestation":
      "max_t_method_version=1 is attested, not retro-invented: the evidence "
      "file predates versioning and carries no version field; the mapping is "
      "declared forward by docs/p48_stat_calibration_measurement_log.md "
      "(统计实现 eval_corrections 零改动) and the git history examined in "
      "reports/wp17/_recon/map.md §2.",
  },
  # ── 第五组：是否具备可靠推断资格及原因 ─────────────────────
  "qualification": {
    "qualified":      bool,           # 唯一资格布尔，W3 的一切消费者只读它
    "status":         "qualified" | "not_qualified",
    "reason_codes":   list[str],      # §1.2 枚举，空列表 ⇔ qualified=True
    "reason":         str,            # §3.3 冻结模板渲染
    "covering_scenarios": list[str],  # 本操作点的覆盖场景集（§2.4，判定者计算，
                                      # 非调用方指定）
    "blocking_scenarios": list[str],  # verdict ∉ {calibrated, conservative} 的
                                      # 覆盖场景
  },
}
```

`build_result` 无条件构造全部五组（描述性路径不阻断的.field 级证明见 §3.2）。
`qualification` 组在 build 时由 `qualify_claim_record` 用**当时可得的证据注册表**填好
（证据不可得 → qualified=False + 对应码，绝不禁写）；G2 在晋级时**重新计算**并核对
（verify-don't-trust，§3.1）。

### 1.2 资格枚举与原因码表（冻结）

`status ∈ {"qualified", "not_qualified"}`。原因码（`reason_codes` 元素，全大写字符串）：

| 码 | 触发条件（精确） | 修复路径 |
|---|---|---|
| `EVIDENCE_NOT_REGISTERED` | 证据注册表缺失/文件不可读/调用方未提供证据对象 | 登记证据或经 W6 产生新证据 |
| `EVIDENCE_HASH_MISMATCH` | 注册的 sha256 ≠ 实测文件 sha256 | 重新登记（新注册表条目） |
| `EVIDENCE_IDENTITY_MISMATCH` | 任务 method.(max_t_method_version, n_perms, block_size) ≠ 证据行的对应参数 | 以匹配参数重测产物，或经 W6 产生覆盖该参数点的新证据 |
| `N_DATES_OUT_OF_CALIBRATED_DOMAIN` | n_common_dates ∉ {30, 250} | 仅新证据（W6）可扩域 |
| `K_OUT_OF_CALIBRATED_DOMAIN` | n_pool_trials ∉ [4, 9] | 同上 |
| `MISSING_FOLD_POLICY_UNREGISTERED` | missing_fold_policy ≠ 注册值 `structural_no_imputation_no_zero_padding_v1` | 政策变化须另立契约 |
| `SCENARIO_UNCALIBRATED` | 覆盖集中存在无证据行的场景名 | 运行该场景（须 W6 触发记录） |
| `SCENARIO_INSUFFICIENT_SAMPLES` | 覆盖集中存在 verdict=insufficient_samples 的场景 | 按 W6 重跑（不得加种子钓鱼） |
| `SCENARIO_UNRELIABLE_AT_NOMINAL` | 覆盖集中存在 verdict=unreliable_at_nominal 的场景 | **唯一出路**：修方法 + 新不相交种子块重校准（W6 协议） |

合格（qualified=True）当且仅当 `reason_codes == []`。**没有**"regime 字符串"参与判定
（用户裁定：调用方字符串永不授予资格；与 G1 消费 RegimeRegistry 对象而非字符串的
现行形态一致，promotion.py:808-815 + recon §1 末段）。

### 1.3 新常量（方法身份登记，additive）

- `eval_corrections.py` 新增 `MAX_T_METHOD_VERSION = 1`：把"p40 实现 + p48 校准所测的
  同一 max_t_from_matrix 实现"**向前**钉为版本 1。该文件今天没有任何 VERSION 常量
  （recon §2 实测 grep 零命中），JSON 也没有版本字段——本常量是**新登记**，不是回溯
  改写证据；版本 1 与既有证据的对应由 `method_version_attestation`（§1.1 第四组）
  与注册表条目声明。
- `claim_record.py` 新增 `MISSING_FOLD_POLICY = "structural_no_imputation_no_zero_padding_v1"`
  与 `CALIBRATION_EVIDENCE_REGISTRATION`（source 路径 + sha256 + 所钉
  max_t_method_version；列表结构，允许未来追加新条目，旧条目不可改）。
- bootstrap `seed` **不是**身份比较键：校准扫过每场景 400-1000 个种子，方法覆盖是
  种子无关的；任务侧 seed 仅照常记录于 max_t 块（eval_corrections.py:473）。

---

## 2 W2 —— 资格判定规则（fail-closed 精确算法）

输入：`claim_record` 前四组 + 证据对象 `CalibrationEvidenceRegistry`（从注册登记的
JSON 文件加载；测试可用 `from_payload` 内存构造）。输出：第五组。
**次序固定，首个失败码即记录（全部评估，码可多个，但按下列次序追加）。**

### 2.1 第一步：证据可消费性

1. 证据对象存在且指向注册登记的文件，否则 `EVIDENCE_NOT_REGISTERED`。
2. 实测文件 sha256 == 注册 sha256（本条冻结值
   `b85a652705f8768fe5629473de5aac7841b575d2ff46617388e6a49c1b296d0e`，
   2026-09-23 实测，6467 字节），否则 `EVIDENCE_HASH_MISMATCH`。
   任一失败 → 直接终止判定（后续码不产生），qualified=False。

### 2.2 第二步：方法身份匹配（比较键）

任务侧比较键 = `(method.max_t_method_version, method.n_perms, method.block_size)`；
证据侧比较键 = 每行的 `(n_perms, block_size)` + 注册条目声明的
`max_t_method_version`。匹配要求**逐项相等**：

- `max_t_method_version` 相等（当前：1 == 1）；
- `n_perms` 相等（证据为 999；**生产默认 build_result max_t_perms=5000**
  （eval_artifacts.py:164），故今天一切默认产物在此步即
  `EVIDENCE_IDENTITY_MISMATCH`——这是如实状态，不是缺陷；修复=以 999 重测或经 W6）；
- `block_size` 相等（证据为 10，与生产默认一致）。

不相等 → `EVIDENCE_IDENTITY_MISMATCH`，终止。

### 2.3 第三步：覆盖域判定（"n=250、K=8 不外推"的精确判据）

冻结操作点 = 测量日志 §7 :74-75 的逐字声明：**k=4..9、n∈{30,250}、n_perms=999、
block=10；其他操作点属未校准域，默认"无法作可靠推断"**。据此：

- `n_common_dates` 必须**等于** 30 或 250。判据是**点集合成员**，不是区间：
  n=180（插值）、n=400（外延）、n=31..249 全部 → `N_DATES_OUT_OF_CALIBRATED_DOMAIN`。
  理由：校准只测过 n∈{30,250} 两个点，中间长度无证据，插值/外延都是外推。
- `n_pool_trials` ∈ [4, 9]（日志声明的带；场景实测点 k=4、8、9）。出带 →
  `K_OUT_OF_CALIBRATED_DOMAIN`。
- `missing_fold_policy` == 注册值，否则 `MISSING_FOLD_POLICY_UNREGISTERED`。

### 2.4 第四步：覆盖场景集（判定者计算；相关性形态归属）

覆盖集由 (n_common_dates) **唯一决定**，调用方无权指定、无权收窄：

| 任务 n_common_dates | 覆盖场景集 | 各场景覆盖的相关性形态 |
|---|---|---|
| 250 | {independent_noise, common_factor_095, ar1_phi04, duplicate_candidates} | iid 基线；截面共同因子（载荷 0.95）；**序列自相关 AR(1) φ=0.4**；重复候选/多重性（3×3, k=9） |
| 30 | {short_window_30d} | P40 最小窗操作点（n=30,k=4）的 iid 基线 |

理由（冻结）：任务真实收益的相依形态不可由调用方声明（同 §1.2 的裁定），而校准家族
在同一操作点测了四种形态；方法必须在**整族**上控制名义尺寸才可作声明，故资格 =
覆盖集上 verdict 的**合取**（最差者决定）。测量日志 §4.2 已指明生产拼接收益
（重叠视界+十日调仓）本质带自相关、属 ar1 域邻域——该场景正是当前的阻断者。

### 2.5 第五步：合取判定

对覆盖集中每场景取证据行 verdict：

- 行缺失 → `SCENARIO_UNCALIBRATED`；
- `insufficient_samples` → `SCENARIO_INSUFFICIENT_SAMPLES`；
- `unreliable_at_nominal` → `SCENARIO_UNRELIABLE_AT_NOMINAL`（记入
  blocking_scenarios）；
- `calibrated` / `conservative` → 通过（B5 语义，stat_calibration.py:197-199）；
- `power_reported` 行**不是**尺寸证据，永不进入覆盖集（test_p48:180-190 的
  passthrough 语义保持）。

`qualified = (reason_codes == [])`；`covering_scenarios`/`blocking_scenarios` 如实登记。

### 2.6 当前冻结证据下的必然结果（必须如实呈现，不是缺陷）

以既有证据运行本算法：n=250 任务被 `SCENARIO_UNRELIABLE_AT_NOMINAL`（ar1_phi04：
9.5% [7.8%,11.5%]）阻断；n=30 任务被 short_window_30d（10.8% [9.0%,12.9%]）阻断；
且默认产物先被 `EVIDENCE_IDENTITY_MISMATCH`（5000≠999）阻断。即**在方法修订并经
W6 流程产生新合格证据之前，一切任务 qualified=False**——这正是测量日志 §4.2
"正式统计结论与晋级声明冻结"的机器可执行形式。

---

## 3 W3 —— G2 接线语义与正式结论

### 3.1 G2 的第四层（资格层）

`evaluate_challenger` 增加关键字参数 `calibration_evidence: CalibrationEvidenceRegistry
| None = None`（对象形态，同 G1 的 RegimeRegistry 先例；promotion.py:1400 的加载
方式对应增加 CLI 旗标 `--calibration-evidence`，默认取
`CALIBRATION_EVIDENCE_REGISTRATION` 登记路径；`None`/不可得 → fail-closed）。
`_gate_g2_significance(artifact, thresholds, calibration_evidence)` 在既有三层
（p41 身份绑定 :865-892 → p43 内容核验 :893-895 → p43 N5 数值裁决 :896-947）**之后**
追加第四层，固定次序：

1. `artifact["claim_record"]` 存在且五组齐全，否则拒收（冻结串 A，§3.3）。
2. **verify-don't-trust**：第一/二/三组与 artifact 自身的 dsr/max_t/stitched/
   matrix_excluded 内容逐项一致（statistics.p == max_t.p_value；
   method.n_perms == max_t.n_perms；task_actuals.n_common_dates ==
   max_t.window.n_dates；……），不一致 → 拒收（冻结串 B）。artifact 自带的
   `qualification.qualified` 布尔**不被信任**，G2 用证据对象重跑 §2 算法。
3. 重算结果 `qualified == False` → 拒收（冻结串 C），reason_codes 原样入理由。

数值层（p ≤ 0.05 等）**先于**资格层评估但**不足以**通过：三层全过而第四层失败，
G2 仍 FAIL——"p=0.001 但方法未校准 → 不得支持晋级"由此成立。反之，资格层不可能
挽救数值层失败。

### 3.2 描述性路径不受阻断（字段级证明）

- 持久化路径：`evaluation.py:1004-1011` 的 `write_boundary_artifact` 在 build_result
  之后**无条件**调用，其间不执行任何 gate 代码；`claim_record` 五组在
  qualified=False 时**照常全量填写**（第一~四组与资格无关；第五组
  qualified=False + reason_codes + reason）。保存永不因资格被阻断。
- CLI：evaluation CLI 的 selfcheck 判定（:1038-1043）与退出语义不变；新增输出仅是
  §3.4 的资格行。promotion CLI（唯一消费晋级语义的入口，recon §1 证实无其他生产
  调用方）才是阻断面。
- 研究面：r01/r02/r03、tier_reports、research_doctor、admission 不读
  max_t/dsr/significant（recon §3 grep 零命中），零改动。

### 3.3 冻结逐字串（G2 拒绝理由 / 一致性）

- **串 A**（缺 claim_record，含全部 pre-wp17 旧产物）：
  ```
  artifact carries no claim record (wp17 W1): pre-wp17 artifacts cannot
  support promotion - the five field groups (statistics, method, task
  actuals, calibration evidence identity, qualification) are required
  ```
- **串 B**（记录与自身证据不一致）：
  ```
  claim record is inconsistent with the artifact's own statistical blocks
  (wp17 W1): {field} records {recorded!r} but the artifact carries {actual!r}
  ```
- **串 C**（资格失败；`{n}/{k}/{np}/{bs}/{policy}/{codes}` 为格式化值）：
  ```
  significance qualification fail-closed: no calibrated evidence covers
  this artifact's operating point (n_common_dates={n}, pool_trials={k},
  n_perms={np}, block_size={bs}, missing_fold_policy={policy}); reason
  codes: {codes}; a nominal-5% significance claim is unsupported
  regardless of the p-value (wp17 W2/W3)
  ```
- `calibration_evidence is None` 时串 C 以码 `EVIDENCE_NOT_REGISTERED` 呈现。

### 3.4 正式结论模板句式（用户可见面，冻结）

- evaluation CLI stdout（插在 evaluation.py:1037 的 max_t 行之后）：
  ```
  claim_record: qualified={qualified} reason_codes={codes}
  ```
  且当 `qualified=False` 追加一行：
  ```
  no reliable inference: {reason}
  ```
- **合格**分支（模板，仅 qualified=True 时可用）：
  ```
  max-t p={p:.4f} is qualified for a nominal-5% significance claim at the
  calibrated operating point (n={n} common dates, K={k} pooled trials,
  evidence {source} sha256 {sha256[:8]})
  ```
- **无法推断**分支（qualified=False 的一切正式结论，含 p 极小值场景）：
  ```
  No reliable inference: the significance method lacks calibrated-evidence
  qualification for this operating point (reason codes: {codes});
  descriptive statistics are retained, and no nominal-5% significance or
  promotion claim is made.
  ```
- verdict JSON：`evaluate_challenger` 返回值增加 `"claim_record"`（artifact 记录
  拷贝）与 `"qualification"`（G2 重算摘要：qualified/codes/covering/blocking）。
- lifecycle `build_promotion_evidence`（lifecycle.py:1141-1160，当前生产调用方为零）
  若未来接线，其 PROMOTED 证据只能引用 qualified=True 的 verdict——本条作为冻结
  约束登记，不在本包实现。

---

## 4 W4 —— P48 正向标签修正（三分支全审 + 数值例）

### 4.1 新旧对照表（`stat_calibration.py:143-171`，逐字）

| 分支 | 现串（冻结证据/日志中已存，**只读保留**） | 裁定 | 新串（实现落点 `calibration_verdict`） |
|---|---|---|---|
| `calibrated`（:150-155） | `f"Nominal {alpha:.0%} size supported: rate={rate:.3f}, 95% CI [{low:.3f}, {high:.3f}] contains {alpha}."` | **替换**（"size supported" 措辞过强：CI 包含 5% 只支持未拒绝，例 [4.6%,7.5%] 同时容许高于 5%；CI 与检验的对应只支持拒绝/未拒绝，不自动验证上限） | 见 §4.2 |
| `conservative`（:156-162） | `f"Size below nominal (CI [{low:.3f}, {high:.3f}] entirely under {alpha}): {alpha:.0%} claims are valid and conservative; power is reduced."` | **保留逐字**。审计：区间整体低于名义是单侧 95% 裁定，"名义声明有效且保守"方向安全（实测尺寸低于名义 ⇒ 声明保守），无上限验证式过强语义 | 不变 |
| `unreliable_at_nominal`（:163-171） | `f"NO {alpha:.0%} significance claim: measured size {rate:.3f}, 95% CI [{low:.3f}, {high:.3f}] lies entirely above {alpha}. The honest statement is the measured effective size; fixing the method requires a new contract and a fresh disjoint seed block."` | **保留逐字**。审计：只作否定性声明 + 诚实披露实测有效尺寸，无过强语义 | 不变 |
| `insufficient_samples`（:143-149，第四分支一并审） | `"No reliable inference: fewer than {r_min} repetitions (or CI width {width:.3f} > {ci_max_width}) — the measured rate cannot bound the size."` | **保留逐字**。审计："cannot bound" 是如实的不可判声明；test_p48:77 钉 `"no reliable inference" in claim.lower()` 继续成立 | 不变 |

### 4.2 新 `calibrated` 串（代码形态，冻结逐字）

```python
claim = (
    f"No significant departure from the nominal {alpha:.0%} level "
    f"detected: rate={rate:.3f}, 95% CI [{low:.3f}, {high:.3f}] "
    f"contains {alpha:.0%} — supports non-rejection only; not a "
    "verification that the false-positive rate is at most "
    f"{alpha:.0%} (an interval containing {alpha:.0%} also admits "
    f"rates above {alpha:.0%})."
)
```

对 (rejections=59, reps=1000)（= 仓外 JSON `independent_noise` 行）的**期望渲染串**
（逐字，测试锚点）：

```
No significant departure from the nominal 5% level detected: rate=0.059,
95% CI [0.046, 0.075] contains 5% — supports non-rejection only; not a
verification that the false-positive rate is at most 5% (an interval
containing 5% also admits rates above 5%).
```

（单行无换行；上表折行仅为排版。格式核对：f"{0.059:.3f}"="0.059"，
f"{0.046014:.3f}"="0.046"，f"{0.075361:.3f}"="0.075"，f"{0.05:.0%}"="5%"。）

### 4.3 [4.6%, 7.5%] 数值例的期望断言（新 p55 测试，冻结）

```python
v = calibration_verdict(rejections=59, reps=1000, alpha=0.05)
assert v["verdict"] == "calibrated"
assert v["ci_low"]  == pytest.approx(0.0460143, rel=1e-6)   # JSON: 0.04601429821280047
assert v["ci_high"] == pytest.approx(0.0753609, rel=1e-6)   # JSON: 0.07536090277145915
assert v["ci_low"] < 0.05 < v["ci_high"]                    # CI 含 5%……
assert v["ci_high"] > 0.05                                  # ……且上界容许高于 5%
assert v["claim"] == "<§4.2 期望渲染串>"                     # 精确钉扎
```

语义断言（用户裁定的核心，不得弱化）：当 Wilson 上界 > 5% 而 CI 含 5% 时，
claim **不得**输出"误报率不超过 5%"式语义——由精确等值断言落实（新串中
"at most 5%" 只出现在否定从句 "not a verification that …" 之内）。
如实记录一处原文措辞出入：任务书括注写作"Wilson 下界>5% 但 CI 含 5%"，二者
逻辑上不能同时成立；按用户 W4 正文（"例 [4.6%,7.5%] 同时容许高于 5%"）与实测
数值（下界 4.60% < 5% < 上界 7.54%），本冻结取"**上界** > 5% 且 CI 含 5%"。

### 4.4 同步面与勘误纪律

- 仓外 JSON 的 claim 串与测量日志 §2 表**原样保留**（冻结/只读；勘误只追加）。
  JSON 的 `independent_noise` 行仍存旧串 "Nominal 5% size supported: …"——它是历史
  记录，`rows_consulted` 只登记身份字段（§1.1），不搬运 claim 串，因此不会把旧措辞
  带进新报告。
- `docs/p48_stat_calibration_draft.md` §2 B4 追加**勘误 2**（先例：勘误 1 以附注追加，
  draft :62-66）：正向标签语义改为"未检出与名义水平的显著偏离——支持未拒绝，不构成
  误报率≤5% 的验证"；判定表四分支**不变**（本勘误只改措辞，不改判定）。
- power 行的 `power_reported` + "reported as measured" 串是仓外重算产物（recon §2
  标签分歧；模块本无 power 分支）——**不在本包范围内**，模块不加 power 分支，
  readiness 对其 passthrough（claim_5pct=False）语义原样。
- 现有测试破坏面：test_p48 无正向 claim 精确钉扎（recon §4 核实；:77/:88/:102 只钉
  子串），`CALIBRATION_VERSION==1`（:193）不受影响（本包不改该常量）。

---

## 5 W5 —— 功效尺度：任务相称场景与不可外推标注（不运行校准）

### 5.1 <1% 年化相对超额 → 日尺度植入参数（全算术，冻结推导）

用户给定冻结引用：日 Sharpe 0.25/0.5 简单年化 ≈ 3.97/7.94（×√252）；任务实际候选
为**不到 1% 年化相对超额**。推导（全部量按 252 个交易日年化，与 ×√252 同口径）：

1. 任务边界取 1% 年化相对超额（任务严格低于此，取上界使场景"与任务相称"）：
   年超额 = 0.01。
2. 日均超额植入 μ = 0.01 / 252 = 3.9683e-5。
3. 日噪声尺度 σ = scale = 0.01（沿用全部 7 个既有场景的 scale=0.01，
   stat_calibration.py:59-83——同族连续性，相关性形态覆盖声明因此可沿用）。
4. 日 Sharpe：sr = μ/σ = (0.01/252)/0.01 = **1/252 = 0.0039683**（精确有理数）。
5. 年化 Sharpe：sr×√252 = √252/252 = **1/√252 = 0.062994 ≈ 0.063**。
6. 既有两点的等价折算（同 scale）：sr05 → 年超额 0.5×0.01×252 = 1.26（**126%**），
   年化 Sharpe 0.5×√252 = **7.9373 ≈ 7.94**；sr025 → 年超额 0.25×0.01×252 = 0.63
   （**63%**），年化 Sharpe 0.25×√252 = **3.9686 ≈ 3.97**（与用户给定 3.97/7.94 一致）。
7. 差距（写死）：126% / 1% = **126 倍**；63% / 1% = **63 倍**（等价地
   7.9373/0.062994 = 126.0、3.9686/0.062994 = 63.0，闭式 0.5×252=126、0.25×252=63）。

### 5.2 新场景登记（SCENARIO_SPECS 追加；**不运行**）

```python
"task_scale_excess_1pct": {
    "kind": "power", "k_trials": 8, "n_dates": 250,
    "sr": 1.0 / 252.0, "scale": 0.01,
    "task_scale": True,
    "note": "<§5.3 英文标注串>",
},
```

- 生成器：并入现有 power 分支（stat_calibration.py:249-253 改为
  `if name in ("power_sr05", "power_sr025", "task_scale_excess_1pct")`），
  植入式不变 `m[0] += sr*scale` → 日均 3.9683e-5。
- **命名硬约束（实测依据 §9.3）**：名字必须排序在 `short_window_30d` 之后。
  实测：`task_scale_excess_1pct` → 排序索引 7 → seed_start=970000，
  short_window_30d 保持 960000，既有 7 场景种子切片身份不动。反例实测：
  `power_task_scale` 会把 short_window_30d 推到 970000（破坏与仓外 JSON 的
  seed_start 对应）——**禁用此类名字**。
- k=8/n=250 与既有 power 点对齐（同操作点，只有植入强度不同）；
  **相关性形态沿用既有 size 场景的覆盖声明**：本场景登记为 power 点，其覆盖声明
  = size 家族四形态（iid/共同因子/AR(1)/重复候选）在同一操作点的覆盖声明之上；
  将来依 W6 运行时，功效检查须与 size 家族同形态集（登记于 episode 的
  target operating points），不得只跑 iid 形态。
- 该场景**无证据行**（不运行校准，红线 1）→ 其 readiness 永远是 unlisted/uncalibrated
  → "无法推断"。将来运行须走 W6（episode purpose 可为 scenario_extension）。
- 既有钉扎兼容性：test_p48:139 用超集判断（`<= names`），追加不破坏；:146-149 的
  kind/n_dates/k_trials 通检通过。

### 5.3 不可外推标注串（逐字）

代码/注册表英文串（SCENARIO_SPECS note + wp17 契约引用）：

```
power_sr05/power_sr025 are NOT extrapolable to the task scale: daily
Sharpe 0.5/0.25 annualize to 7.94/3.97 (x sqrt(252)), i.e. 126%/63%
annualized relative excess at scale=0.01 - 126x/63x the task-scale edge
(<1% annualized relative excess; task_scale_excess_1pct plants
sr=1/252 ~= 0.00397, annualized Sharpe 1/sqrt(252) ~= 0.063). Detection
rates measured at 0.5/0.25 say nothing about detection at the task
scale; no power claim is made at the task scale until
task_scale_excess_1pct is calibrated under the wp17 W6 protocol.
```

文档中文串（wp17 契约 draft / 勘误追加面）：

```
power_sr05/power_sr025 不可外推到任务尺度：日 Sharpe 0.5/0.25 简单年化
≈ 7.94/3.97（×√252），按 scale=0.01 折算即 126%/63% 年化相对超额，为任务尺度
（<1% 年化相对超额；task_scale_excess_1pct 植入 sr=1/252≈0.00397，年化
Sharpe 1/√252≈0.063）的 126 倍/63 倍。在这两点测得的检测率（100%/93%）对
任务尺度的检测能力不作任何声明；在 task_scale_excess_1pct 依 WP17 W6 协议
校准之前，任务尺度不作任何功效声明。
```

消费面核实（recon §4）：无生产代码把高检出率写进结论；唯一 "detection rate" 文本
在仓外 JSON claim 与测量日志 §3——两者只读保留。p40 时代 C5 功效守卫
（tests/test_eval_resampling_calibration.py:116-129，日 SR 0.5、band≥0.90、调参种子
50000+；docs/p40_statistical_resampling_draft.md:95-96）是**反退化回归守卫**，
与本标注互不改写。

---

## 6 W6 —— 方法修订条件触发协议（只文档+守卫；本包不执行任何比较）

新模块 **`ashare_model/method_revision.py`** + 登记文件
**`ashare_model/method_revision_registry.json`**（包内新文件：与守卫代码同版本演进，
且不触碰 `data/**` 只读红线；episodes 追加式、已写字段不可变，守卫每次重读并校验
schema）。

### 6.1 触发判定（"出现值得正式检验的候选"，冻结）

同时满足四条才可开 episode：

- (a) 候选是某注册 protocol artifact 的 current-run stitched top trial（p41 身份
  五元组完整）；
- (b) 现行冻结方法下名义统计达标：max_t p ≤ 0.05 且 dsr ≥ 0.95；
- (c) 其 G2 失败**归因于资格码**（§1.2 表中任一；即横在该候选与晋级之间的屏障是
  统计资格，而非数据/身份/内容/数值失败）；
- (d) episode 的 target operating points 覆盖该任务实际操作点（n_dates、
  k_trials、n_perms、block_size、相关性形态集=§2.4 覆盖集的形态族——
  n=250、K=8 不外推：检验范围必须覆盖实际任务的相关性/长度/试错池）。

触发记录字段（开 episode 时一次写全，不可改）：`episode_id`、`opened_utc`、
`artifact`（run_id/spec_id/candidate/seed/artifact_sha256）、
`blocking_reason_codes`、`target_operating_points`（列表）、`versions`（≤2）、
`frozen_budget_seconds`、`pilot_seeds`（计时试探用种子，声明为参与调参、永不复用
于验收）、`acceptance`（null 或 {version_id, seed_block_start, reps, verdict,
closed_utc}）、`status ∈ {open, accepted, rejected, budget_exhausted}`。
**没有触发记录就开始方法版本比较 → 守卫拒绝**（串 G1）。

### 6.2 两版本上限

一个 episode 的 `versions` 最多登记 **2** 个方法版本（开发阶段比较上限）；登记
第三个 → 串 G2。新 episode 须新触发记录（且旧 episode 已终态）。

### 6.3 计时与冻结总预算（操作，冻结）

1. 比较开始**前**，对每个 target operating point 用合成矩阵（形状=该点的
   [k, n]，n_perms/block_size=该点参数）计时 20 次 `max_t_from_matrix` 调用，
   取均值 ms/call（pilot 种子记入 `pilot_seeds`）。
2. `frozen_budget_seconds = ceil(1.5 × Σ_points reps_planned × mean_ms / 1000)`
   写入触发记录，之后**不可改**（1.5 为冻结安全系数）。
3. 墙钟超过 `frozen_budget_seconds` → 中止（串 G4），状态 `budget_exhausted`，
   保留"无法推断"。

### 6.4 一次独立种子验收

- 只选 **一个** 版本进入 **一次** 新的独立种子验收：fresh 种子块起点 ≥ 1,000,000，
  与 p40 调参块（10000-50000）及 p48 验证块（900000+100000）显式不相交（重叠 → 串
  G5；不相交性沿用 test_p48:130 的钉扎逻辑扩展到 episode 块）。
- 判定 = 现行冻结 B4 规则原样（calibration_verdict 四分支 + 勘误 1，**不放宽、
  不改判据**）；尺寸场景 reps ≥ R_MIN=500。
- 验收后产生新证据注册条目（新 sha256 + 新 max_t_method_version），W2 资格按新
  证据重算。
- **失败保留"无法推断"**：任何阻断场景仍 unreliable/insufficient → episode
  `rejected`，资格维持 not_qualified；**不增加种子直到碰巧通过**（第二次验收
  请求 → 串 G3；也不得以提高 reps/改判据重试）。

### 6.5 守卫挂点与拒绝串（逐字）；豁免面 = **零豁免**

挂点（可执行面）：

1. `stat_calibration.calibrate_scenario` 增加 keyword-only `episode_token: str |
   None = None`：token 为 None 或非 open/属主匹配的 episode → ValueError（串 G1）。
   既有测试不调用该函数（recon §5 核实），零破坏；默认拒绝 = fail-closed。
2. 钉扎测试（p55）：`MAX_T_METHOD_VERSION != 1` 且注册表中无 `accepted` 终态
   episode → 失败（方法常量翻版必须有已验收 episode 背书）。
3. 注册表 schema 校验：episodes 追加式、已写字段不可变、种子块不相交重验。

拒绝串（逐字）：

- **G1** `method revision refused: no open trigger record exists for this work - register the candidate worth formal testing (artifact identity + blocking reason codes) before any method-version comparison or acceptance calibration starts (wp17 W6)`
- **G2** `method revision refused: episode {episode_id} already lists {n} method versions; at most two may be compared per episode (wp17 W6)`
- **G3** `method revision refused: episode {episode_id} has already run its single independent-seed acceptance calibration; a failed or insufficient acceptance keeps 'no reliable inference' - no re-runs and no additional seeds (wp17 W6)`
- **G4** `method revision refused: episode {episode_id} exceeded its frozen budget of {budget_seconds}s; the honest state remains 'no reliable inference' (wp17 W6)`
- **G5** `method revision refused: the requested seed block [{start},{span}) overlaps a used block (tuning 10000-50000, validation 900000+, or a prior episode); acceptance requires a fresh disjoint block (wp17 W6, p48 B3)`

**豁免面选择：零豁免**（无白名单、无 skip 旗标、无环境变量旁路）。理由：任何
豁免面都会重建 W2 刚封死的洞——调用方字符串/旗标授予资格，与"regime 字符串不得
授予资格"同一原则；本仓 fail-closed 先例（p43 N5 的 missing/NaN/bool 全拒、
G1 regime 对象制）一致。episode purpose 允许 `method_revision` /
`scenario_extension` 两类（后者跳过两版本上限但仍须触发记录+预算+新种子块），
这不是豁免而是同一纪律的两个登记类别。

---

## 7 版本影响与钉扎重锚清单（依据=本文，非实现输出）

### 7.1 版本

- `PROMOTION_RULE_VERSION` "4" → **"5"**（v5 = WP17：G2 消费校准资格，fail-closed；
  promotion.py:116-128 docstring 追加 v5 段）。
- `ARTIFACT_SCHEMA_VERSION` 3 → **4**（additive `claim_record`，带默认）。
- 新常量：`MAX_T_METHOD_VERSION=1`（eval_corrections）、`MISSING_FOLD_POLICY`、
  `CALIBRATION_EVIDENCE_REGISTRATION`（claim_record.py）。
  `CALIBRATION_VERSION` 保持 1；`PROTOCOL/REWARD` 等不动。

### 7.2 因收紧而须重锚的旧钉扎（逐个；实读核实行号）

| 测试（file:line，实读/recon） | 破因 | 重锚处方（依据本文） |
|---|---|---|
| test_promotion.py:138-151 `test_all_gates_pass_promotes`（`_strong_artifact` :71-90 走真实 build_result） | promoted=True 依赖 G2 旧语义；且现有 fixture k=1、n≠250、n_perms=5000，落入未校准域 | fixture 扩为 8 个 current 候选、拼接轴 250 共同日期（2 折×125 会话）、build_result 传 `max_t_perms=999`；`_verdict` 增 `calibration_evidence=` 内存构造注册表（覆盖集四场景 verdict="calibrated"，n_perms=999/block=10）。期望值由 §2 算法手推，非实现回读 |
| test_promotion.py:678、:695（data_tier 正例 promoted=True） | 同上 | 同上（复用同一 fixture 工厂） |
| test_promotion.py:1069（G6/G7 正例 promoted=True；:1068 钉版本 "4"） | 同上 + 版本钉 | 同上；:1068 改钉 "5"（§7.1） |
| test_promotion.py:1125 `test_promotion_rule_version_pinned` | 钉 "4" | 改钉 "5" |
| test_p41_candidate_evidence_binding.py:255（J6 钉 "4"） | 版本钉 | 改钉 "5" |
| test_p41:202/:230（J5 拒绝族） | 不破（断言的是拒绝+子串；收紧只增理由） | 不动；如断言精确理由全集则按串 C 格式补 |
| test_p43_statistical_input_integrity.py:253-257（N5 正控制：G2 通过） | G2 通过现需资格层 | `_significance_gate` 助手同样注入内存证据注册表 + 域内 fixture（n=250、k∈[4,9]、n_perms=999）；其"数值裁决正控制"目的保留 |
| test_p43:191-251（N5 拒收族） | 不破（拒绝仍拒绝） | 不动 |
| test_p48 全部 11 项 | W4 只换 calibrated 串；无精确钉扎（recon §4 核实；:77/:88/:102 子串继续成立）；:139 超集判断容纳新场景；:193 钉 CALIBRATION_VERSION==1 不变 | 不动；新增断言见 §4.3 |
| tests/test_eval_resampling_calibration.py、test_p40_*.py、test_evaluation.py | 与资格无消费关系（recon §5） | 不动（test_evaluation 若加 claim_record 断言另立新测） |

新测试文件 `tests/test_p55_wp17_claim_qualification.py`（前缀 p55；同 commit 登记
shard，置于 p48 条目 :179 邻近），最低覆盖：W1 五组结构、W2 每个原因码各一条、
串 A/B/C 逐字、§4.3 数值例、W5 场景登记+seed 切片不动（short_window_30d 仍
960000）、W6 守卫 G1-G5 逐字与零豁免（默认 token=None 拒绝）。

---

## 8 红线遵从（自查）

- 不启动新校准/重采样：本包代码只登记与接线；`task_scale_excess_1pct` 只注册不运行
  （§5.2）；W6 全部为守卫与文档（§6）。
- 不比较方法版本：本包零比较；守卫使无触发记录的比较不可能启动（§6.5）。
- data/** 与既有 reports 只读：新登记文件放包内
  （ashare_model/method_revision_registry.json）；证据 JSON 只读并 sha 钉扎。
- 冻结文档只追加：p48 draft 勘误 2（§4.4）、测量日志不动。
- 旧钉扎重锚：§7.2 逐条记录且依据本文。

## 9 本文件起草时的实测记录（命令与结果，2026-09-23）

1. JSON 实测：`python -c "import json; ..."` 读取
   `D:/minequant/reports/p48/calibration_report.json`——顶层键
   `{block_size, n_perms, power_reps, readiness, scenarios, size_reps,
   total_seconds}`；`scenarios` 为 dict（键=7 场景名）；每行字段
   `{alpha, block_size, ci_high, ci_low, ci_width, claim, kind, ms_per_call,
   n_perms, rate, rejections, reps, scenario, seconds, seed_start, verdict}`；
   seed_start 900000..960000（ar1/common/duplicate/independent/power_sr025/
   power_sr05/short 顺序）；independent_noise 行 rate=0.059、
   ci_low=0.04601429821280047、ci_high=0.07536090277145915、reps=1000、
   verdict=calibrated；ar1=unreliable(0.095)、short=unreliable(0.108)、
   power 两行 verdict=power_reported；readiness 含 unlisted_regime（claim_5pct=False）。
2. sha256：`hashlib.sha256(open(...,'rb').read())` →
   `b85a652705f8768fe5629473de5aac7841b575d2ff46617388e6a49c1b296d0e`（6467 字节）。
3. 种子排序模拟：`sorted(7 名 + 候选名)` → `task_scale_excess_1pct` 索引 7
   （seed_start 970000），short_window_30d 保持 960000；`power_task_scale` 会把
   short_window_30d 推到 970000（禁用）。
4. 零引用核实：`grep -n "stat_calibration|claim_record|qualif"
   ashare_model/promotion.py`——无 stat_calibration/claim_record 引用（仅无关的
   "qualify" 词）；`grep -rn "MAX_T_METHOD_VERSION|method_revision"
   ashare_model/` 零命中（均为本包新增面）。
5. 生产 n_perms 默认核实：eval_artifacts.py:164 `max_t_perms: int = 5000`、
   eval_corrections.py:590 `n_perms: int = 5000`（与证据 999 不匹配——§2.2 如实
   记录）。
6. 未运行项如实声明：未运行任何校准/重采样；未运行 pytest（本文件为实现前冻结，
  无实现可测）；test_p41:202/:230 与 test_p43:191-251 的行为预判基于实读的断言
  形态（子串断言）而非执行。
