# P42 工作包 6 裁决与 R03 组合权重面板测量日志（evidence-only）

契约：`docs/p42_r03_combo_weights_draft.md`（APPROVED，用户 2026-09-19
直接指令工作包 6；含勘误 1）。本文件只记录实际发生什么；计划在契约。

## 1 分支裁决（§1 冻结表的执行记录）

逐行对照已收口证据（详见契约 §1 表格）：

- 行 1（→R03）**选中**：p37 n10 稳定区唯一净正臂 = 跨族等权
  `combo:all3`（+2.24%~+2.71% 1x），三单因子全部净负（−1.2%~−7.6%）——
  等权组合的第一形态已测且成立，开放问题恰为 R03 第二形态（训练窗拟合
  收缩权重）。
- 行 2（继续 R02）不匹配：n10 毛超额 +9%~+12%（费用折减但未吞噬）；
  n20 毛信号本身为负（−5.1%~−43.7%，非费用问题）。
- 行 3（→R04）不匹配：R02 面板行不携带 val_reward，无任何
  reward-排序-vs-净表现证据。
- 行 4/5/6 不匹配：medium 有净空间；基线稳定性存疑（单年依赖）而开放
  问题是组合结构；存在正增量证据。

## 2 软件交付

| 项 | 值 |
|---|---|
| 分支 / base | `codex/b04-load-cache-performance`，base `27d3dc6`（p41 收口）；未 push |
| 提交链 | 契约 `cbcd94a` → RED `915637e`（13 失败，证据 `D:\\minequant\\reports\\p42_red_evidence.txt`）→ GREEN `63bce15` → CLI 引导修复 `141afbd`（首次正式启动在 ModuleNotFoundError 处死亡，未消耗任何 run 身份/落盘） |
| 实现 | `ashare_model/r03_combos.py`（weighted_combo_signal / fit_combo_weights / decide_r03_adjudication / run_r03_panel / run_r03_matrix）+ `scripts/r03_combos.py` CLI；复用 r01_baseline 与 r02_matrix 的全部单一权威（combo 基座、cost_variant、cell_variant、_arm_verdict、_has_gross_info、R02_ADJUDICATION_RULES 同一对象、summarize_arms、生命周期走查） |
| 版本 | 零 bump（PROTOCOL 29 / reward 15 / grammar 9 / EXECUTION 4；`R03_REPORT_SCHEMA_VERSION=1` 新增） |
| 验证 | 聚焦 `tests/test_p42_r03_combos.py` **13P**（含迷你端到端全生命周期）；相邻 p37/p39/p34 回归 **47P**；compileall 干净；分片 117/117；`git diff --check` 干净 |

测试修改白名单（§10.1 规则 3，相对 RED commit）：N1 oracle 修正为真实
NaN 范围（个股-日格而非整日列，修正后钉住回退格与加权均值邻格——覆盖更
强）；N3 补进程内 import；N2 的 set-vs-approx 断言形态改为逐值 approx
（同一性质）。无断言弱化。

## 3 正式运行

命令（2026-09-19 23:45–23:56，工作树干净 @ `141afbd`）：

```
python scripts/r03_combos.py --contract-sha256 <p42 契约 sha256> \
  --out-root data/r1/r03_combos
```

| 项 | 值 |
|---|---|
| run | spec/run 见 `data/r1/r03_combos/r03_combo_run_meta.json`；report artifact 经 RunStore 登记 + 测量目录副本 |
| 身份 | dataset `d36024cd1b19…`（与 p37 同一复合，实时重确认）；cutoff `20260916`；purpose `historical_research`；git `141afbd` |
| 规模/墙钟 | 140 行；**605.1s（≈10 min）**（因子缓存命中，冷计费已在 p37 支付） |
| 数据资格 | G1–G8 全过；loader frames clean；benchmark 五折全 ok |
| **verdict** | **`equal_weight_stands`** → run 驻留 **DATA_QUALIFIED**（非终态） |

### 3.1 面板结果（n10 稳定区，1x，跨折平均）

| 格 | 臂 | 净超额 | IR | 正折 | 投入 | 持仓 | viable |
|---|---|---|---|---|---|---|---|
| n10_m1000 | combo:all3（等权） | **+1.21%** | +0.055 | 2/5 | 0.69 | 8.9 | ✓ |
| n10_m1000 | fitted λ0.50 | −0.56% | −0.095 | 2/5 | 0.76 | 9.5 | ✗ |
| n10_m2500 | combo:all3 | **+1.17%** | +0.058 | 2/5 | 0.70 | 8.9 | ✓ |
| n10_m2500 | fitted λ0.50 | −0.67% | −0.113 | 2/5 | 0.76 | 8.9 | ✗ |
| n10_m5000 | combo:all3 | **+0.74%** | +0.029 | 2/5 | 0.69 | 8.3 | ✓ |
| n10_m5000 | fitted λ0.50 | −0.67% | −0.113 | 2/5 | 0.76 | 8.9 | ✗ |

λ 敏感性（描述性，不参裁）：λ0.25 ≈ −0.4%、λ0.75 ≈ −2.4%（三格一致，
拟合越重越差——单调）。逐折拟合权重与 val_icir 输入全量在报告
`fold_evidence`（2023 折全负 → 等权回退事件如实记录）。

**裁决依据**：等权组合三格全部 viable（p37 净空间在**新运行身份**下复现，
量级减半，见 §3.2）；拟合主臂三格全部净负——训练窗验证 ICIR 不含可用的
相对权重信息，方向上甚至为反向（权重最大者 RSQ_60 恰是拖累）。增量边际
（50bp）远未触及；`equal_weight_stands` 不是险胜，是明确负结果。

### 3.2 与 p37 的数值差异归因（重要披露）

R03 等权臂净超额（+0.74%~+1.21%）低于 p37（+2.24%~+2.71%）。逐折对账：
2021/2022/2023/2025 四折**逐位一致**，唯一差异折 = 2024
（14.57% → 7.07%）。根因：**fold 2024 的 RSQ_60 方向在两个参考格之间
翻转**（p37 参考格 n20_m5000 学得 −1；R03 参考格 n10_m2500 学得 +1）。
`CandidateScorer._choose_direction` 按 val_reward（词典型）择向，而
val_reward 依赖给定 bt_cfg 的验证窗小回测——**方向学习是组合配置的
函数**，p37「与格无关」的披露错误（已作为勘误 1 追记于 p37 日志）。
R03 的方向/权重以其自身预注册参考格 n10_m2500 学习（先于结果冻结），
裁决有效；机制修复（方向学习与配置解耦）属语义变更，须另立契约。

### 3.3 披露（不否决）

- `single_year_dependence`：等权组合正折仍集中于 2023（+14.8%）/2024
  （+7.1%）；其余三折为负——与 p37 旗标同向，量级因 §3.2 减半。
- 单因子仅参考格 1x 落行（预注册设计）；等权臂 = p34 构造的加权形式
  （等权时逐位等于 `combo_signal`，N1 钉住）。

## 4 结论与下一步

- **工作包 6 裁决的问题已回答**：在 n10 稳定区，训练窗拟合的收缩权重
  组合**不**稳定优于跨族等权组合（明确负结果）；等权净空间在新运行身份
  下复现但量级减半（方向-参考格依赖披露在案）。run 驻留
  DATA_QUALIFIED；**无晋级/holdout 声明；默认配置未变**。
- 对照冻结决策表的后续走向（各自另立指令）：等权组合的 2023–24 集中
  归因与年度稳健性；方向学习机制与组合配置解耦（语义变更，需契约）；
  小预算 GP/Random 搜索对照（行 5 的问题在组合问题已答后可问）。
- 尾注政策（用户 2026-09-19）：未来数据观察（C03）在候选与规则冻结后
  开展——本次 verdict 不解锁任何 paper/观察；界面（C01）优先修复同一
  次运行的结果关联；缓存/同步/页面扩展按实际阻塞程度安排。均在本包
  范围外。

## 5 未运行项与声明边界

- 全量 `python -m pytest -q tests -n auto`（精确候选树 `141afbd` + 本
  日志文档）→ **2054 passed / 0 failed / 6 skipped / 3 xfailed**（18:19）；
  p41 收口基线 2041P + 本包 13P；skip/xfail 无变化。
- 未运行：生产数据同步、search/RL 入口、promotion、web 构建——非目标。
- 本日志全部数字为历史研究（purpose=historical_research，开发窗
  2021–2025）；不构成 alpha/晋级/纸面观察资格的任何声明。
- Mimosa 钩子多次报告 scanner_enobufs（未获完整扫描结论）；不做安全
  声明，建议择机完整审计。
