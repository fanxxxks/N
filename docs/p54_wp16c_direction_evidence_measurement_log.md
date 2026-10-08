# P54 (WP16 16-C) 一次评分保存两个固定方向的原始证据 — 测量日志

- 契约真源：`reports/wp16/FROZEN_C.md`（2026-09-23 冻结，先于实现；对实现
  者只读，本日志遵守其 §0.3 红线与 §4 范围栅栏）。
- 逐字命令与退出码：`reports/p54_red_evidence.txt`（只追加）。本日志为其
  docs 侧汇总，只引用不复制推导。
- 环境：Windows 10 22621 x64，Python 3.13（miniconda），pytest 9.1.1；测试
  一律 `cd D:/minequant/AlphaGPT && python -m pytest ...`。
- 原始产物红线：`D:/minequant/reports/p45/**`、`D:/minequant/reports/p46/**`
  全程只读（跑后核实 mtime 仍为 09-20，未写入）；p46 §3 修复面板的全部新
  输出落 `AlphaGPT/reports/wp16c/`。

## 1 落地内容（对照 FROZEN_C）

| 冻结项 | 状态 | 落点 |
|---|---|---|
| §0.3 红线 1：禁止二次调用择优评分器取反方向 | **本包实现** | `candidates.py` `score_many` 新增 `oriented_evidence_out` 出参：一次 interleaved ±signal 批量（既有 469-491 结构原样）把两侧 `rewards / val_rewards / icir / val_icir / objectives[B,8]` 逐元素带出（plus=2k/minus=2k+1，与 `_choose_direction` 消费的同批同源）；被拒候选记 `rejected_reason`，数字留空不造数 |
| §1 双向证据字段模式 | **本包实现** | `direction.py` 新增 `reference_return_provenance`：双侧 reward/ICIR（train+val）、`objectives_plus/minus`（[B,8] 行整存，费用与活动由此出）、`train_signal_range`、`train_dates`/`val_dates`（实际日期轴，新增可选 `dates` 参数）、`val_windows`、`reward_clip=[-10,+10]` 披露、`direction`、`applied_signal_basis="direction * raw_signal"`、`tie_break_reason`、`canonical_orientation_locus`、`selection_rule`（§2 正式命名“参考配置下的验证 reward 择向（Rule A）”）。非有限浮点按仓库严格 JSON 惯例存 `None`（`rows_to_jsonable` 同款） |
| §1 两个真实数值锚 | 已达成（结构上） | 落选侧真值不再与胜者同值记两遍：p54 合成例胜者 +10（clip 高）落选 −10（clip 低，携 `reward_clip` 披露）；wp16c 实盘面板逐成员记录 `val_reward_plus/minus` 真值（`combo_pct_measurement.json`） |
| §2 选择规则逐字保持 | 未触碰 | `CandidateScorer._choose_direction` 一字未改；`reference_return_direction` 的 direction 仍来自同一次批量内的该比较（direction VALUES 不变）。新增 `classify_rule_a_tie_break` 只做分支归类（validation_reward / validation_icir / canonical_orientation，NaN→−inf 降级同款），p54 以 6⁴ 网格断言其与 `_choose_direction` 逐点一致 |
| §0.3 红线 2：证据修复≠策略变更 | 遵守 | 本包唯一行为性删除是 `direction.py` 旧二次 `scorer.score` 与 `eval_search.py` 旧整轮镜像 `score_many`（同为被替换对象，FROZEN_C §2）；无任何比较键/扫描序/默认 +1 改动 |
| §3 V1/V2/V3 | 钉测试 | `tests/test_p54_wp16c_direction_evidence.py` 13 测试：V1 双向差分例 + 独立批量逐元素对账（不经选择层）+ clip 披露；V2 取反交换（含 canonical 分支 locus 定位与施加暴露不变）；V3 毒化测试期（−10⁹ 与 NaN 变体）全字段不变 |
| eval_search 基线行 | **本包实现** | `baseline_candidates` 一批出证（`oriented_evidence_out`），删除 163-179 整轮重复评分；provenance 经同一 `reference_return_provenance` 组装（携带真实训练日期轴） |
| §4 p46 百分位修复 | **本包实现** | 新脚本 `reports/wp16c/run_measurement.py`（原 `D:/minequant/reports/p46/run_measurement.py` 只读）：combo_raw vs combo_pct 恒等以**结构性**呈现（逐列 max_abs_diff==0 验证 + §4.2.1 冻结原因串，中英文输出均含）；新增 no-re-rank 对照臂（脚本本地 helper，均值聚合已归一化成员，非生产组合定义）；§4.2.2 恒等式小测试 `tests/test_p54_wp16c_combo_invariance.py`（percentile + 两个任意仿射，单/多成员，容差 0）钉住 |
| §4.3 栅栏 | 遵守 | `r01_baseline.py`、`R01_COMBO_ARMS`、词表、组合定义零改动；combo 语义测试（p34/p42）与 p46 既有测试（N1/N4/N5/N6）全绿 |

## 2 关键机制说明

- **一批出证为何可行**：`score_many` 内部本就把每候选 ±signal 交错送入一次
  `reward_function`（candidates.py 469-491 既有结构），只是两侧数组在择向
  后被丢弃；`oriented_evidence_out` 只是把已在手的数组按候选对齐带出，
  选择行为、返回契约、monkeypatch 面（`score_many` 类属性补丁，含
  test_evaluation 两个 spy）均不变。
- **日期轴**：`reference_return_direction` / `reference_return_provenance`
  新增可选 `dates`（与信号列对齐）；不传时 `train_dates/val_dates=None`，
  其余不变——p46 既有调用（N1/N3/N4/N6）零改动仍绿。生产调用方
  （eval_search 基线行、wp16c 面板）传入真实日期轴。
- **legacy 四键**：`val_reward/val_icir/val_reward_opposite/val_icir_opposite`
  保留（§5 红线 2），语义修正为真实 chosen/opposite（同批同源、未扣复杂度
  罚——direction/bare-factor 规格 penalty=0，数值与旧实现一致）。

## 3 RED / GREEN（详见 reports/p54_red_evidence.txt）

| 命令 | 退出码 | 结果 |
|---|---|---|
| 基线：`python -m pytest tests/test_p46_direction_provenance.py tests/test_candidates.py tests/test_evaluation.py tests/test_p34_r01_baseline.py tests/test_p42_r03_combos.py -q` | 0 | 103 passed（717.56s） |
| RED `python -m pytest tests/test_p54_wp16c_direction_evidence.py tests/test_p54_wp16c_combo_invariance.py -q`（实现前） | 1 | 13 failed, 5 passed, 1 skipped |
| GREEN 同命令（实现后） | 0 | **19 passed**（面板产物落地后 skip 项激活实跑；单跑 combo 文件 6 passed） |

RED 13 失败三类：`dates` 关键字不存在（10）、`classify_rule_a_tie_break`
不存在（2）、`baseline_candidates` 评分批量计数 [1,1]≠[1]（1）。GREEN 前
置 5 通过为 §4.2.2 恒等式钉（绿到港，钉冻结事实 §4.1）；skip 1 为面板产
出 gated 测试（产物落地后实跑通过）。

## 4 回归（本会话实测）

| 命令 | 结果 |
|---|---|
| `python -m pytest tests/test_p52_wp16a_cash_identity.py tests/test_p53_wp16b_p45_parity.py tests/test_p54_wp16c_direction_evidence.py tests/test_p54_wp16c_combo_invariance.py tests/test_p46_direction_provenance.py tests/test_p45_scoring_parity.py tests/test_reward.py tests/test_p3_portfolio_parity.py tests/test_ci_sharding.py -q` | **151 passed**（16-A/16-B/16-C 合并门） |
| `python -m pytest tests/test_p46_direction_provenance.py tests/test_candidates.py -q` | **23 passed** |
| `python -m pytest tests/test_evaluation.py tests/test_train.py -q` | **100 passed, 2 skipped**（两个 score_many spy 原样通过） |
| `python -m pytest tests/test_p50_wp14_*.py tests/test_p51_wp15_*.py tests/test_execution.py tests/test_no_signal_semantics.py -q`（9 文件展开） | **173 passed** |
| `python -m pytest tests/test_universe.py tests/test_p34_r01_baseline.py tests/test_p42_r03_combos.py -q` | **62 passed**（201.65s） |
| `python scripts/check_test_shards.py` | shard union == full set: 137/137（p54 两文件已注册） |

`refuse_unwired_trade_groups` 守卫（p51）、p50 锚点机制、p3 parity、
test_reward（57）均未破坏；`ashare_model/backtest.py`、`ashare_portfolio/**`
本包零改动。

## 5 p46 §3 修复面板（reports/wp16c/，FROZEN_C §4.2.1）

- 脚本：`reports/wp16c/run_measurement.py`（configs/r1_medium_a.yaml 同
  五折、同成员 MOMENTUM_20/RSQ_60/REVERSAL_60、同 n10_m2500 参考格）；成员
  方向改由 16-C 单批 `reference_return_direction` 取得（同一 Rule A，顺带
  记录双侧证据与 tie-break）。
- 产物：`reports/wp16c/combo_pct_measurement.json`（+ 控制台文本，含冻结原
  因串逐字）。
- 结果（5 折一致）：combo_raw vs combo_pct 逐列 **max_abs_diff == 0**，以
  结构性恒等呈现（非经验发现）；引擎总收益 raw==pct 逐折相同（构造使然）。
  no-re-rank 对照臂（均值聚合已归一成员）：rho=1.0——JSON 内附说明：成员同
  列共享 N_eff 时均值(排名/N) 是 combo_raw 的列内仿射映射，秩不变属预期退
  化；成员有限集不同的单元格上排序才分离（3/5 折 top10_overlap_min=0.90，
  另 2 折 1.00）。
- 只读合规：`D:/minequant/reports/p45`、`D:/minequant/reports/p46` mtime
  跑后仍为 09-20，未写入；`FROZEN_C.md` 未改。

## 6 残留与边界（如实记录）

- `reference_return_direction` 的 `dates` 为可选：不传则 `train_dates/
  val_dates=None`（p46 既有调用面零改动的代价；生产路径均传入）。
- 被拒（近常数/无效公式）候选的 §1 数字字段为 `None`、`tie_break_reason
  =None`（无选择发生；FROZEN_C §1 未枚举该边缘，None 为诚实值，direction
  仍为旧实现的 +1）。
- 对照臂在本面板上 rho=1.0 的退化解释已写入产物；更换成员覆盖差异更大的
  面板时分离会显现（测量事实，非本包承诺）。
