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

## 3 当前状态与下一步

- 软件与治理：**完成**（六格冻结面板 + 连续-整手差异可观测 + 四路裁决 +
  生命周期链 + 17 项测试；全量 2006P/0F 两轮一致）。
- R02 第一轮正式测量：**被数据资格阻断**（§2 的 REJECTED 即当前最新
  证据）；无面板数值、无 verdict——诚实状态。
- 面板软件的 canary 级行为已在受控夹具上验证（17 项测试，含四路裁决
  纯函数与端到端链）；canary 附带观察：13 码小池上 n20 格因 buy_rank ≥
  池内可格数而使所有臂选出同一账本（机械正确，生产 ~5,000 码池不触发），
  合成数据无毛空间 → canary verdict 确定性为 `panel_no_reliable_info`。
- 下一步属于各自独立授权的包：三个数据阻断的契约治理决策（见 §2）；
  数据解锁后重跑六格面板 → 产出四路 verdict 之一（负结果合法）；
  第二轮周期比较（every_5/20_days、monthly 等合法周期）待第一轮有
  verdict 后另立指令。
