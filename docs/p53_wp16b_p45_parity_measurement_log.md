# P53 (WP16 16-B) P45 可比化 — 测量日志

- 契约真源：`reports/wp16/FROZEN_B.md`（2026-09-23 冻结，先于实现；对实
  现者只读，本日志遵守其 §0.3 红线与 §4 命名契约）。
- 逐字命令与退出码：`reports/p53_red_evidence.txt`（只追加）。本日志为其
  docs 侧汇总，只引用不复制推导。
- 环境：Windows 10 22621 x64，Python 3.13（miniconda），pytest 9.1.1；测试
  一律 `cd D:/minequant/AlphaGPT && python -m pytest ...`。
- 原始产物红线：`D:/minequant/reports/p45/**` 全程只读（重跑后核实 mtime
  仍为 09-20 14:47/14:53，未写入）；全部重跑输出落
  `AlphaGPT/reports/wp16b/`。

## 1 落地内容（对照 FROZEN_B §1 缺口清单）

| 冻结项 | 状态 | 落点 |
|---|---|---|
| F1 收益基础=复权开盘价 | 已做→钉测试 | B1.1/B1.3 于 S1（含 −50% 负对照与"错误值不出现"断言）；B1.2 harness 侧缺键 fail-closed（`_REQUIRED_CACHE_KEYS`） |
| F2 成交基础=原始价格 | 引擎已做→钉测试 | S1 的 shares 翻倍/名义不变断言（B2.1）；basket 权重空间差异如实记录（未造价格层） |
| F3 local_signal_range 贯穿 | **本包实现** | `compare_candidate_parity`/`run_fixed_candidate_parity` 新增 keyword 参数，`_run_orientation` 同一元组传 sim 与 engine（B3.1 桩断言）；B3.3/B3.5 出口预校验（越界索引/n_dates/t+delay+1 三要素消息）；B3.4 引擎 contiguity 语义直钉 |
| F4 长度/日期不一致拒绝 | **本包实现（删裁剪）** | 删 `n = min(...)`/`[:n]` 三处截断；daily/fee/turnover 三族各自长度相等断言，四要素消息（`length mismatch`/`sim={n}`/`engine={m}`/候选+窗口标识）；日期轴长度/重复校验（B4.2）；2-D raw_cache 键逐键形状校验；一切拒绝先于任何误差计算（B4.4） |
| F5 调仓轴/PIT/阻断 | 已做+**显性计量** | 调仓轴/universe 同对象传递保留；新增 `CandidateParityRow.blocked_extra_buy_columns`（basket_blocked & ~engine_blocked 的列计数，harness 按 backtest.py 同式自算引擎矩阵）；B5.1 零不对称钉测试 |
| F6 同一固定方向+费用/资本 | 已做→钉测试 | B6.1 同 cfg 对象（桩断言 id）；B6.2 零费两路径费用序列恒 0（逐元素，精确）；B6.3 行级误差族用候选给定 direction（± 候选对照断言） |

引擎 `ashare_model/backtest.py` 未改（FROZEN_B §0.3 红线 3；引擎现状已满足
F1/F2/F6 引擎侧）。harness 侧参数为加法式：`local_signal_range=None` 保持
现行默认路径（FROZEN_B §5 红线 3），面板重跑显式传。

## 2 命名契约落地（FROZEN_B §4）

- `ashare_model/scoring_parity.py` 新增 `ORIENTATION_DISCLOSURE`（§4.2 冻结
  披露串逐字）；`proxy_reliability_verdict` 输出携带
  `orientation_disclosure` 键；flip 措辞改为 §4.4 允许形式（"direction-flip
  diagnostic — 两条路径在测试窗累计净收益口径上排序相反"，并明示"NOT the
  production Rule A validation-reward orientation"）。
- 两诊断正式命名：**测试窗累计净收益方向对照**（本面板
  `test_window_total_return_orientation (proxy|engine)`，诊断一）与
  **生产验证 reward 择向**（candidates.py Rule A，诊断二；本面板不测量、
  不复现）。重跑 JSON 顶部 `diagnostic_naming` 块两分列示。
- p53 测试对重跑输出断言披露串存在 + §4.3 禁用表述缺席
  （`test_p53_s42_rerun_output_text_honors_the_naming_contract`）。

## 3 RED / GREEN（详见 reports/p53_red_evidence.txt）

| 命令 | 退出码 | 结果 |
|---|---|---|
| RED `python -m pytest tests/test_p53_wp16b_p45_parity.py -q`（实现前） | 1 | 15 failed, 3 passed, 1 skipped |
| GREEN 同命令（实现后） | 0 | **19 passed**（面板产物落地后 skip 项激活实跑） |

## 4 回归（本会话实测）

| 命令 | 结果 |
|---|---|
| `python -m pytest tests/test_p45_scoring_parity.py tests/test_reward.py tests/test_p3_portfolio_parity.py tests/test_p46_direction_provenance.py tests/test_p52_wp16a_cash_identity.py tests/test_ci_sharding.py -q` | **113 passed** |
| `python -m pytest tests/test_p50_wp14_*.py tests/test_p51_wp15_*.py tests/test_execution.py tests/test_no_signal_semantics.py -q`（9 文件展开） | **173 passed** |
| `python scripts/check_test_shards.py` | shard union == full set: 135/135（p53 文件已注册 portfolio-trading 片） |

p50/p51 基线、`refuse_unwired_trade_groups` 守卫、p3 parity（12）、
test_reward（57）均未破坏。

## 5 P45 固定小面板重跑（FROZEN_B §3 两遍协议 + 对照）

脚本 `reports/wp16b/run_measurement.py`（机制复用
`D:/minequant/reports/p45/run_measurement.py`：同 r1_medium_a、同 folds[2]
2023、同 n10_m2500 参考格、同 5 候选方向全 +1；输出全部写 wp16b/）。三遍：
pass0 `local_signal_range=None`（连续性锚）、pass1 `(0, 240)` 显式默认等价
窗、pass2 `(0, 231)` 折真窗（`test_signal_end − test_signal_start`，窗口
242 列）。

### 5.1 连续性控制（FROZEN_B §3.2 第一遍期望的归因式核验）

- **pass1 vs pass0：0 处数值不一致** —— 16-B 的显式窗口传递对默认路径是
  零行为变化（这满足 §3.2 "16-B 修复未引入其他行为变化"的判据）。
- **engine 侧 vs 原始报告：0 处不一致**（5 候选 engine_total_net 与
  engine_direction 逐位相同）—— 引擎未被 p52/p53 触及的直接证据。
- **proxy 侧 vs 原始报告：5 处变化** —— 全部归因 **16-A（p52）**：原始
  parity_report.json 产出于 16-A 落地之前，basket 路径的可加现金费用恒等式
  + 费用预留是其已声明语义变化（proxy 累计：MOM +0.3629%→+0.5467%、
  RSQ +7.4622%→+7.6817%、REV −0.5700%→−0.4240%、IND −10.5158%→−10.2170%、
  combo −3.5410%→−1.7528%；方向全部不变）。归因后按 §3.2 进入第二遍。

### 5.2 差异分解：原始 → pass2（折真窗，FROZEN_B §3.3；界取各自报告自身）

族级三数：**resolved 0 / remains 3 / new-regression 4**（20 项中其余 13 项
双侧界内）。

| 候选 | 族 | old | new | 判定 |
|---|---|---|---|---|
| MOMENTUM_20 | daily_net | 8.690e-03 | 8.690e-03 | 仍在 |
| REVERSAL_60 | daily_net | 1.0354e-02 | 1.0362e-02 | 仍在（+7.6e-06，16-A） |
| IND_REL_RET_20 | daily_net | 1.526e-02 | 1.526e-02 | 仍在 |
| combo:all3 | daily_net | 1.078e-02 | 8.235e-03 | 仍在（数值降 24%，仍超界） |
| RSQ_60 | turnover | 1.5585827e-02 | 1.5585830e-02 | 新增（+3.6e-10，界内） |
| REVERSAL_60 | fee | 1.2517e-04 | 1.2529e-04 | 新增（+1.2e-07，界内） |
| REVERSAL_60 | turnover | 4.8922e-02 | 4.8968e-02 | 新增（+4.6e-05，界内） |

四项 new-regression 的 pass0 == pass1 == pass2 **逐位相同**，即全部源于
16-A（p52）在 16-B 之前已落地的 basket 语义（费预留使 basket 换手成为
"已付费换手"），**无一来自 16-B**（pass1 ≡ pass0 已排除）。四项绝对量
3.6e-10 … 4.6e-05，均远在各自界内。

- 方向翻转：before {MOMENTUM_20, REVERSAL_60} → after 相同（测试窗累计净
  收益口径的方向翻转诊断，§4.2 披露串随行）。
- ranking spearman：0.9 → 1.0（combo 名次变动）；material inversion
  (MOMENTUM_20, REVERSAL_60) 消失（pass2 无 material inversion）。
- verdict：unreliable → unreliable（7 → 6 条 reason）。
- **blocked_extra_buy_columns = 242/242 列**（每行）：basket 侧 loader
  阻断（额外叠加 ~universe_mask 与 adj_factor 堕名禁买）在真实面板上逐列
  都严于引擎自算矩阵——F5 不对称从隐性变成显性计量（FROZEN_B §1 F5 的
  要求即此）。

### 5.3 判读（§3.3 纪律之内）

统一口径后，原差异**族级消失 0、仍在 3（+1 项仍在但数值略升）**：原始
daily-net 超界并非窗口/掩码/基错位所致——两条路径现在共享同一窗口、同一
掩码、同一复权基与同一配置，剩余差异属于近似本身（权重空间 vs 整手执行
账本）与 F5 计量出的阻断不对称（242/242 列）。本分解只回答"原差异多少
消失/仍在/新增"，**不回答**生产择向（Rule A 验证 reward 择向）是否被复
现——本面板不测量该规则。

## 6 变更文件清单（16-B）

- `ashare_model/scoring_parity.py`（F3/F4/F5 计量/§4 命名；引擎式加法参数）
- `tests/test_p53_wp16b_p45_parity.py`（19 测试）
- `scripts/check_test_shards.py`（注册 p53 文件，2 行）
- `reports/wp16b/{run_measurement.py, parity_report_pass0_default.json,
  parity_report_pass1_explicit_default.json,
  parity_report_pass2_fold_window.json, comparison.json, run_console.txt}`
- `reports/p53_red_evidence.txt`（只追加）、本日志
- 未改：`ashare_model/backtest.py`、`data/**`、`reports/p45/**`、
  `reports/p46/**`、FROZEN_*.md
