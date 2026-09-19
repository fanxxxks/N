# P37 R02 六格组合/成本矩阵测量日志（evidence-only）

本日志只记录实际发生了什么；计划见
`docs/p37_r02_portfolio_cost_matrix_draft.md`（APPROVED，用户 2026-09-19
直接指令工作包 4「把 R02 首轮缩成六格，先固定调仓周期」）。

## 1 软件交付（受控变更闭环）

| 项 | 值 |
|---|---|
| 分支 | `codex/b04-load-cache-performance`（WP1–WP3 同链叠层） |
| 提交链 | 契约 `d2ba655`（+ CONTRACTS.md 登记）→ RED `df804dc`（17 失败 + 分片注册 111 files/4 shards + 契约 Rev 1）→ 实现 `ad84e07` |
| RED 证据（实现前树） | 17 failed 全部落在契约 §12 能力缺口：W1/W2/W4–W7 `ModuleNotFoundError: r02_matrix`；W3 `ImportError: lot_gap_fields`。证据存档 `D:\minequant\reports\p37_red_evidence.txt` |
| 实现 | ① `ashare_model/eval_metrics.py`：`lot_gap_fields(result, ts_codes)` 纯读取函数（应调仓会话上连续目标权重 vs 整手实现账本的逐名绝对偏差之和（完整 L1）、投入差、首建仓会话/计划/成交订单数；数据源=既有 `target_weights`/`positions`/`buy_weights`/`sell_weights`/`construction_diagnostics`，引擎与构造器零改动）；`evaluate_signal` 返回值 `**lot_gap_fields(...)` 附加，既有字段逐位不变。② `ashare_model/r02_matrix.py`：冻结六格表 `R02_GRID`（耦合参数 buy_rank=top_n / cap=1/top_n / sell_rank=⌈1.5·top_n⌉）、`cell_variant`（恰替换五字段）、`cell_adjacent`（单轴单步）、`run_r02_panel`（候选=p34 冻结臂；方向每折在参考格 `n20_m5000` 经 `baseline_candidates` 学习一次跨格冻结复用；0x/1x 两档；基准每折一次、现金对照仅参考格并披露格点无关性）、`assemble_cell_summaries` + `decide_r02_adjudication`（§7 四路裁决纯函数：stable_region_found / isolated_single_winner（不稳定结果）/ gross_info_execution_blocked / panel_no_reliable_info）、`run_r02_matrix`（IDEA→SPEC_LOCKED→DATA_QUALIFIED 生命周期链 + C5 契约自校验 + 预算守卫 + 诚实终态守卫；verdict 1/2 驻留 DATA_QUALIFIED、3/4 RETIRED）。复用 r01_baseline 的 `cost_variant`/`combo_signal`/`cash_signal`/`summarize_arms`/`rows_to_jsonable`/`resolve_folds_for`/`_expected_and_missing_sessions` 单一权威，零第二套实现。③ `scripts/r02_matrix.py` CLI shim（退出码 0/1/2 约定） |
| 裁决/方向/成本档 | 与契约逐字一致；未用任何 OOS 结果反改 |

### 1.1 验证矩阵（聚焦/相邻在候选树内容 = `ad84e07`）

| 验证 | 结果 |
|---|---|
| 聚焦 `python -m pytest -q tests/test_p37_r02_matrix.py` | **17 passed**（58.08s） |
| 相邻批次 1（p34×2 + p35 + p36 + baseline_harness + stitched_oos） | **76 passed**（229.56s） |
| 相邻批次 2（evaluation + run_store + backtest + portfolio_constructor + golden_parity + execution） | **190 passed**（565.55s） |
| `python -m compileall -j 0 -q ashare_data ashare_model ashare_portfolio ashare_trading scripts webapi` | exit 0 |
| `python scripts/check_test_shards.py --check` | 111 files across 4 shards（并集=全集） |
| `git diff --check` | 干净（仅 autocrlf 提示） |

测试说明：`tests/test_p37_r02_matrix.py` 为新文件（无既有测试修改）；RED
阶段一次接口澄清——`lot_gap_fields` 以引擎运行的 `ts_codes` 列表为参数
（W3 两处调用随接口补充该参数），手算期望值未变。canary 端到端 verdict
确定性钉住 `panel_no_reliable_info` → `RETIRED`（合成收盘价无毛空间、
测试内注明机理）。

### 1.2 全量验证

`python -m pytest -q tests -n auto` 两轮一致：

| 轮次 | 结果 | 墙钟 |
|---|---|---|
| 第 1 轮 | **2006 passed, 0 failed, 6 skipped, 3 xfailed** | 1222.68s |
| 第 2 轮（`-rs` 复验 + skip 归因） | **2006 passed, 0 failed, 6 skipped, 3 xfailed** | 1209.20s |

6 项 skip 逐项归因（全部为既有环境条件基线 skip，无本包新增；本包 17
个测试零 skip）：`test_lock_files.py:302`（IP-04/03-F-01 dev-machine 锁
审计守卫，登记基线）+ 5 × `requires CUDA`（test_train×2、test_vm×2、
test_ops×1）。宿主当前解释器为 `torch 2.11.0+cpu`（`cuda_available:
False`）；WP3 记录的 1 skip 对应当时可用的 CUDA 环境——该差异为宿主
环境变化（torch 由 GPU 版变为 CPU 版，发生在本包之外），非本包引入。

## 2 正式运行（EXECUTION_SPEC_VERSION=4；2026-09-19 15:13–15:20 本地）

命令（工作树干净 @ `ad84e07`，仅 `.mimosa/` 未跟踪，与 WP3 先例同型；
无并发重载进程）：

```
python scripts/r02_matrix.py \
  --contract-sha256 d9f64a10e96004946619f03ef61a3eaabf71b4063b9eb4530de7df4b3f1e971c \
  --out-root data/r1/r02_matrix
```

（sha 为含 Rev 1 的已提交契约文件实时 sha256；C5 自校验通过。）

| 项 | 值 |
|---|---|
| run | spec `42bac23cafcb9632b9c3e727003e9425254bb9ffe50bf3b77e433476b903936e` / run `2e0a864e253442efa60eaa09c703c418` |
| 身份 | dataset `767d955b0fd99507fed95b4275a2bdde5bdab177e1e22c8dc156a45c107e21b2`（**重新实时确认**，与 09-17/09-18 一致；双库自 09-16 23:47 未变更）；cutoff `20260916`；purpose `historical_research`；`execution_spec_version=4`；git_commit `ad84e07`；seeds (42,7,2024)；requested_budget 512（未消费，`consumed=0`） |
| 生命周期 | IDEA → SPEC_LOCKED（15:13:53 本地）→ **REJECTED**（15:19:43 本地，诚实终态）；artifacts：idea `41e4e862…` / spec_lock `855207b5…` / data_qualification_report `7298522c…`（append-only 登记） |
| 门禁 | G1–G8 **8/8 全过**（p36 历史基准）；universe contract clean（constituents_pit_intervals，8797 open sessions）；日历对账 expected 2846 / loaded 3091 / **missing 0**（a6689a1 修复后形态）；coverage 0.4756 |
| 因子缓存 | hit key `fe96090e…`（dataset `767d955b…`，窗口 20140102..20260916:3091），未重算 |
| 拒绝点 | §5.1 loader_frames **degraded**：`price_semantics_degraded`（**恰好 1,630 码**，含 001914.SZ——全格网 pre-listing 限额披露语义，loader 既有形态，p19 §2.4）+ `industry_source_degraded`（申万指数收盘滞后 7 会话，源止于 20260907，本次只读探查再次确认）。R02Refusal → CLI exit 2，无部分状态、无面板行 |
| 墙钟 | ≈5m50s（confirm + 门禁 + 装载 + 拒绝；SPEC_LOCKED→REJECTED） |

**判读**：与契约 §9 预声明、与 WP3 重跑（run `93b17c2d`）**逐项一致**的
fail-closed 诚实终态——这是四路裁决之外的前置数据资格层拒绝：面板从未
获得执行机会，四路 verdict（stable/isolated/blocked/no_info）**均未产生**，
R02 第一轮无任何臂数值。阻断集合自 WP0 收口以来零变化：① 申万指数收盘
源新鲜度（p17-D5 换源或依赖升级）；② 1,630 码 pre-listing 限额格网披露
语义（p19 §2.4 契约修订）；③ 001914 前身历史限额（数据修复）。三项均为
契约治理决策，超出本包授权，未越权处置。数据解锁后以同一命令（印签当时
实时契约 sha）重跑即可，p37 契约未过期。

## 3 数据解除后的正式运行（2026-09-19 17:02–17:58，**成功出数**）

用户当日「全部授权」覆盖三个数据阻断的处置与重跑；解除过程见
`docs/p38_data_repair_measurement_log.md`（p19 §2.4 必需格网收窄 +
三笔限额修复 + 申万追新；复合身份重发布为 `d36024cd1b19…`/publication 6）。

命令（工作树干净 @ `013a11a`，p37 契约实时 sha 不变 `d9f64a10…`）：

```
python scripts/r02_matrix.py \
  --contract-sha256 d9f64a10e96004946619f03ef61a3eaabf71b4063b9eb4530de7df4b3f1e971c \
  --out-root data/r1/r02_matrix
```

| 项 | 值 |
|---|---|
| run | spec `1b058a49ebf68ae8…` / run `d4ff2b3e0c09…`；report artifact `58047b0c…` |
| 身份 | dataset `d36024cd1b19…`（p38 修复后复合，**实时重确认**）；cutoff `20260916`；purpose `historical_research`；`execution_spec_version=4`；git `013a11a`；`consumed=0` |
| 数据资格 | G1–G8 8/8；loader frames **clean**；completeness **complete**（2846/3091/missing 0）；基准行五折全部 ok |
| 因子缓存 | miss→重算→put（key `4ab7587a…`/dataset `d36024cd…`）——新数据身份的预期冷计费 |
| 规模/墙钟 | 310 行；3369.1s（≈56 min，含冷因子计算） |
| **verdict** | **`stable_region_found`** → run 停驻 **DATA_QUALIFIED**（非终态；后续另立指令） |

### 3.1 六格结果摘要（全部格×臂×折×档在报告 artifact 内，不裁剪）

**成立格**：`n10_m1000` / `n10_m2500` / `n10_m5000`（整个 n10 行；相邻对
m1000–m2500、m2500–m5000 → 稳定区域）。**成立臂**：三格均为
`combo:all3`（1x 跨折平均净超额 +2.24%~+2.71%，`oos_active_ir`
+0.083~+0.111，投入 0.68–0.69，持仓 8.0–8.6 ≥ 下限 5）。**不成立**：
n20 行全部三格六臂（净超额 −1.0%~−10.8%，且 0x 毛超额 −5.1%~−43.7%——
毛信号本身为负）；`n20_m5000` 叠加 low_activity（投入 0.47–0.48）。

**订单门槛轴**：n10 行内三档净超额差异 ≤0.47pp——门槛在 n10 行内不
敏感；**持仓结构轴是决定性的**：同一 combo 信号的 0x 毛超额在 n10 为
+9%~+12%、在 n20 为 −19%~−23%（前十名集中，11–20 名稀释并反转）。

**特格诊断（20×5000=100000 恰覆盖本金，实测非断言）**：投入均值仅
0.472–0.479；min_trade 抑制 3,143–3,373 次/臂；未成交订单 6–31；
lotgap_l1_mean_due 逐格 0.010–0.016（n20 略高于 n10；投入差≈L1 全部
来自现金短差）。名义乘法成立的格点在真实费用/整手/现金下只部署约一半。

**单年依赖旗标**（披露，非否决）：combo:all3 五折仅 2023（+14.8%）、
2024（+14.6%）为正——净空间集中于 2023–2024。

## 4 当前状态与下一步

- **第一轮问题已回答**：稳定区域 = n10 全行（最低订单 1000/2500/5000
  均成立）；n20 在 10 万资本下结构性不可部署（≈半仓 + 毛信号转负）。
  run 停驻 DATA_QUALIFIED；无晋级/holdout 声明；默认配置未变。
- 第二轮（周期轴）见 `docs/p39_r02_periods_measurement_log.md`：参考格
  上两合法周期均无净空间与毛信息 → 周期轴非瓶颈。n10×周期交互未
  预注册未测。
- 后续候选（各自另立指令）：n10 行内 2x 成本与资本缩放对照；
  combo:all3 的 2023–2024 集中归因；若进入搜索轮，应在 n10 结构上
  进行。
- 历史节 §2（数据阻断下的 REJECTED）保留为第一手证据，未被本节改写。


## 勘误 1（2026-09-19，p42/R03 运行发现并量化）：方向学习并非与格无关

本日志 §3 与报告 `reporting_notes.direction_reuse` 曾声明「方向学习
（train-window validation rank-IC sign）与组合配置无关、跨格不变」。
p42 R03 运行（同一 dataset `d36024cd`、同一 cutoff `20260916`）在参考格
`n10_m2500` 学习方向时，**fold 2024-12-31 的 RSQ_60 方向与 p37 参考格
`n20_m5000` 相反**（p37 = −1，R03 = +1）：`CandidateScorer._choose_direction`
按 **val_reward**（词典型，val_icir 次之）比较 ±信号，而 val_reward 来自
给定 `bt_cfg` 下的验证窗小回测——**方向是组合配置的函数**，"by
construction 与格无关"的声明错误。

量化归因：两运行 n10_m2500 `combo:all3` 1x 逐折超额中，2021/2022/2023/
2025 四折逐位一致，仅 2024 折 14.57% → 7.07%（方向翻转所致），区域均值
+2.67% → +1.17%。

影响边界：p37 的全部数值仍是「以其预注册参考格 n20_m5000 学习方向」下的
如实测量，verdict（stable_region_found）不变；但方向学习机制披露错误，
且任何换参考格的复测应预期 fold-2024 级别的方向漂移。p42 R03 以其自身
预注册参考格 n10_m2500 学习方向（先于结果冻结），其裁决不受本勘误影响。
机制修复（如改为纯 train 窗 rank-IC 符号、与配置解耦）属语义变更，须
另立契约，不在勘误内夹带。
