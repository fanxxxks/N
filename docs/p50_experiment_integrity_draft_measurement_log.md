# P50 实验完整性与报告状态 — 测量日志（工作包 14）

- 契约：`docs/p50_experiment_integrity_draft.md`（本日志与之成对，p45–p49
  惯例）。设计权威：`reports/wp14/DESIGN.md`（Rev3 记录交付偏离）。
- 被测实现：`AlphaGPT/ashare_model/experiment_integrity/`（types/plan/
  precheck/reconcile/status 六模块）+ 六份 `tests/test_p50_wp14_*.py` +
  执行脚本 `reports/wp14/wp14_run_integrity.py`。
- 执行阶段与审计修复阶段的所有命令均 cwd=`D:/minequant/AlphaGPT`、只读
  `data/**`（运行前后 sha256 比对一致，见 §3）。

## 1 产物身份锚（只读实测，file digest）

| 文件 | sha256 | 字节 |
|---|---|---|
| data/r1/r01_slow_baseline/r01_baseline_report.json | 95fac73249c0ee93f86e0f385ce25ddf5ae7e7a92bc830db0706710bc05f550b | 2049495 |
| data/r1/r01_slow_baseline/r01_baseline_run_meta.json | 3d39a934e2d9d8b1d75b3fb4aebf5a9603a0afa64350651a47151aef8626462a | 2209 |
| data/r1/r02_matrix/r02_matrix_report.json | 859baefc1d65b331…（首 16 hex） | 5635434 |
| data/r1/r02_periods/r02_periods_report.json | 4f78220acce3c46f… | 2185861 |
| data/r1/r03_combos/r03_combo_report.json | 49376a722565e551… | 3273817 |

行数实测：125 / 310 / 120 / 140；r01 失败行 30（combo:all3 15 +
combo:momentum_reversal 15，3 档×5 折完整网格），唯一错误串
`KeyError: 'MOMENTUM_20'`；四报告 completed 均为 true。

## 2 执行阶段测量（2026-09-22）

- `python -m pytest tests/ -k wp14 -q` → **102 passed**（六个阶段文件，
  修复前基线）。
- `python reports/wp14/wp14_run_integrity.py` → 退出 0，产出四份
  `<exp>_status.json` + `runplan_materialized.json` +
  `integrity_findings.md`；落点标签实测：
  r01_slow_baseline=**measurement_incomplete**（M3：30 主臂程序失败行，
  裁决隔离）、r02_matrix=**economic_positive**（stable_region_found）、
  r02_periods=**economic_negative**（period_panel_no_reliable_info，真负
  结果判例）、r03_combos=**economic_positive**（equal_weight_stands）。
- precheck 基线计数：r01 = **15** issues {member_outside_domain:5,
  member_missing_direction_evidence:5, combo_member_not_in_plan_singles:5}；
  其余三实验 0。
- 勘误两文件（ERRATUM.md / errata.json）由
  `reports/wp14/r01_slow_errata.py` 一次性写入（拒绝覆盖守卫 + ER2 目录
  面快照=恰 4 文件）。

## 3 审计修复阶段测量（2026-09-22，本日志最新一轮）

仲裁五条必改全部落地；关键实测：

1. **kind 词表绕过复现（修复前，RED 证据）**：真实 r01 计划
   arms[candidate='combo:all3'].kind 改 'comboo' 后 plan_from_json 静默
   接受；run_precheck issue 计数 **15 → 6**，提及该臂的 issue **0**；
   benchmark:equal_weight 挂 members=['GHOST_MEMBER_X'] 同样**零告警**。
   修复后（GREEN）：plan_from_json 抛 PlanValidationError（消息含
   experiment/arm.candidate/kind/ARM_KINDS 四元定位）；直接构造的计划报
   可定位 `unknown_arm_kind`；幽灵成员报 `comparison_arm_with_members`。
2. **provenance 锚定（8/8 匹配实测）**：四报告 recorded
   `provenance.config_sha256`/`contract_sha256` 与当前 configs/r1_slow_ab.yaml、
   r1_medium_a.yaml 及四份契约文档实算逐一匹配（十六进制大小写不敏感：
   r0x 写盘方记录大写 config 摘要、`_sha256_file` 产小写）。篡改
   config_sha256 → IntegrityRefusal（消息含两侧值）；删除 provenance 块 →
   2 条 `provenance_anchor_absent` finding。
3. **四产物输出逐字段不变**：修复后重算 precheck/reconcile/status 与修复
   前 `reports/wp14/<exp>_status.json` 逐字段比对（overall、三态、
   precheck 计数与码集、reconcile 摘要、issue 数、citable 清单）——
   **全部一致**；`data/**` 十文件 sha256 运行前后不变（§1 表）。
4. 测试：`python -m pytest tests/ -k wp14 -q` → **106 passed**（102 基线
   + PC13/PC14/RC10/RC11 四个新测试；含 PC12 期望 import 面随 ARM_KINDS
   导入更新）。新测试先 RED（5 failed 实测存档于
   `reports/p50_red_evidence.txt`）后 GREEN。
5. 契约文档落地：`docs/p50_experiment_integrity_draft.md`（§1–§10；模块
   docstring / ERRATUM.md / 六测试 docstring 的全部 § 引用实测可解析，
   STATUS_LABELS 列序与 types.py 元组逐条一致）。

## 4 已知环境事实

- `scripts/check_test_shards.py` SHARDS 登记六个 `test_p50_wp14_*.py`
  （shard-model-2）。
- 测试运行会写 `AlphaGPT/logs/pytest.txt`（ashare_logging 自动导出），
  与 `data/**` 无关。
