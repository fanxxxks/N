# P34 R01 固定基线诊断测量日志（evidence-only）

契约：`docs/p34_r01_fixed_baseline_diagnostic_draft.md`（rev 1，APPROVED，用户
2026-09-16 直接指令「完成R01」）。本文件只记录实际发生什么；计划在契约，二者
不混写（AGENTS §11.1）。

## 1 软件交付（受控变更闭环）

| 项 | 值 |
|---|---|
| 分支 / base | `codex/b04-load-cache-performance`（A01→B04 叠层链 tip = `faad2f4`）；未 push、未合并（无授权） |
| 提交链 | 契约 `96c901e`（+ CONTRACTS.md 登记）→ RED `4c9b317`（tests/test_p34_r01_baseline.py + shard-model-2 注册，10 用例）→ 实现 `2805847` |
| 环境 | Windows 16 GB 主机；Python 3.13（`D:\minequant\.venv\Scripts\python.exe`）；torch 2.11.0+cpu；DuckDB（锁文件版本） |
| 实现内容 | `AshareDataLoader.loader_frame_status()`（p8 §5.1 类型化降级源单一出口）；`build_data_qualification_report` 接入类型化 loader_frames（验证器拒净语义不变；v1 payload 升级，从未有真实通过消费者）；`evaluate_signal` 附加可观测字段（rebalance_due_count / holdings_mean / holdings_max / invested_ratio_mean / cost_fraction_sum；既有字段零变化）；`ashare_model/r01_baseline.py` 驱动 + `scripts/r01_baseline.py` CLI；§5.2 check-4 潜伏缺陷修复（改用 `ResearchDomain.is_legal_execution` 区间权威——旧 `in horizon_range` 成员测试拒绝一切合法内部视界，因既有消费者全为 unified 而未暴露；契约 rev 1 已记录） |
| 版本 | 无研究语义版本变更（PROTOCOL 27 / reward 15 / grammar 9 / registry 14 / LIFECYCLE_CONTRACT 2 均未动；数据报告 payload 字段升级在契约 §4 预注册） |

### 1.1 验证结果（均在精确候选 commit `2805847` 上）

| 验证 | 结果 |
|---|---|
| RED（commit `4c9b317` 树上） | `tests/test_p34_r01_baseline.py` 首证 `AttributeError: loader_frame_status`（契约钉 API 缺位）；R8 端到端经 mini 双库逐项逼出 G6 门禁形态、D7 manifest 依赖、§5.2 check-4 区间缺陷、tensor 首轴布局四类真实边界 |
| 聚焦 | `python -m pytest -q tests/test_p34_r01_baseline.py` → **10 passed**（88.84s） |
| 相邻契约 | `test_p27_a03_lifecycle_runcontext test_p31_b02_entry_canary test_run_store test_evaluation test_data_loader test_baseline_harness test_p29_b01_data_quality` → **207 passed**（515.53s） |
| compileall | exit 0 |
| 分片权威 | `python scripts/check_test_shards.py --check` → 107 files across 4 shards cover all 107 test files |
| 全量 | `python -m pytest -q tests -n auto` → **1964 passed, 1 skipped, 0 failed**，908.82s（1 skip 为既有环境 skip。**勘误 1**：撤回「无净新增 warning/skip」断言——warning 计数因 `-q` 尾部截断未单独留档。**勘误 3**：本行初稿的分解「p33 基线 1953P + 新增 11P」有误——1953P 实录于 `ffe3361`（当时 b04 测试文件 10 项），`553f95b` 又新增 1 项（p33 rev 1 记 11P 新套件、未重跑全量），故 faad2f4 树推算 1954P；1954 + 本包新增 10P = 1964P 实测，自洽） |
| `git diff --check` | 干净 |

## 2 正式运行：被 G8 数据新鲜度门禁 fail-closed 阻断

命令（2026-09-16 19:08:08，工作树干净 @ `2805847`，无并发重载进程）：

```
D:\minequant\.venv\Scripts\python.exe scripts\r01_baseline.py \
  --contract-sha256 177953908aecf9b0060e9ad59f73327a340aa2322e48e44115f1dd79ad94a8b9 \
  --out-root data/r1/r01_baseline
```

结果：**refused（exit 非零，R01Refusal），无任何 run/run-store/报告产物落盘**
（门禁在 run 创建之前拒绝——与 evaluation 入口的 `require_production` 同一边界
次序；`data/r1/r01_baseline` 与 `data/r1/medium_a/runs` 均不存在）。

门禁原文（**勘误 3：证据出处更正**——拒绝文本捕获自进程 **stderr/控制台输出**
（后台作业输出留档），并非下述 `.log` 文件；该文件仅含 143 字节的
"Run logging configured" DEBUG 头，R01Refusal 走 print-to-stderr 路径，
未经过 loguru 文件 sink。后续包应将拒绝路径接入日志体系或归档控制台原始输出）：

```
r01_baseline: refused: production gate check failed (p34 §7): G8 daily bars
fresh within the open-session tolerance (daily 20260907 vs reference 20260915,
lag=6 > N=3, today=20260916)
```

判读：主库 `daily_bar` 最新交易日 2026-09-07；G8 参考日（评估日前最后一个开市
会话）2026-09-15；其后开市会话数 lag=6 > 容差 3（`FRESHNESS_TOLERANCE_SESSIONS`，
p16 预注册常量，非可调旋钮）。数据落后 6 个开市会话（09-08…09-15），主库文件
mtime 2026-09-11 与该事实一致。

### 2.1 勘察过程中的已纠正误判（记录以免误导后续会话）

1. **勘误 2（日历"覆盖缺口"撤回）**：初稿称「主库 trade_calendar 在
   2026-08-25..2026-09-16 区间无任何行、2026 开市会话零行」——这是探针以
   `YYYY-MM-DD` 破折号日期与库内紧凑 `YYYYMMDD` 日期做字典序比较造成的第二个
   假象。紧凑格式复核：2026-09-01 之后日历有 82 行，G8 门禁自身取到的参考日
   20260915 也证明日历开市会话覆盖正常。**日历覆盖无缺口**，该披露撤回。
2. **探针 lag=0 误判**（初稿已记，维持）：同根因的第一次假象；正式门禁按库内
   格式取参考日，lag=6，阻断。以正式门禁输出为准。

以上两个假象同根因：只读探针的日期字面量格式与库内紧凑格式不一致。教训已记：
对该库的一切日期比较必须使用 `YYYYMMDD` 紧凑字面量。

### 2.1b 数据一致性复核（只读，2026-09-16）

- 主库 `daily_bar`：5,648 个不同 ts_code，最新交易日 20260907；
- v2 `daily_bar_raw`：最新交易日 20260907；`source_ingest_runs` 共 58,865 条
  全部 succeeded，最新 finished_at 2026-09-08 10:34。
- 判读：双库一致地停在 2026-09-07 快照（最后同步 2026-09-08 上午），是
  **一致但过期的干净快照**，不是部分同步的中间态。

### 2.2 被评估过的解锁路径与不执行理由

1. **放宽 G8 容差 / 注入评估日 / dev 模式**：均为 fail-closed 硬门禁绕行
   （p16「never a bypass」；AGENTS §4.3），禁止。
2. **双库增量同步**：主库每日 bar 走既有 AkShare 同步管线为逐股全历史重拉
   （约 5,648 个 ts_code × 全窗口，小时级墙钟），v2 侧需 backfill 步骤重跑
   （约 6 个会话：09-08…09-15）并再发布 composite 代际——是对生产研究库的
   有状态写入（AGENTS §5.6），超出本契约冻结范围，且方案文本对数据同步一贯
   要求显式具体授权。未执行。
3. **手工写 6 个会话进主库**：构成第二套同步实现，且会再次制造主库 manifest
   与内容漂移（F02 家族），禁止。

**解除条件**（满足后同 spec 流程可直接重跑；本契约未过期）：
主库 `daily_bar` 更新至 ≥2026-09-11（lag ≤ 3）、v2 侧同步并重新发布 composite
代际、G1–G8 复跑通过后，以同一命令重新运行 `scripts/r01_baseline.py`。基线
窗口（2021–2025 fold）与数据新鲜度无关，但正式入口的新鲜度门禁适用于一切
正式运行，无窗口豁免。

## 3 当前 R01 第一轮状态

- 软件与治理交付：**完成**（契约/RED/实现/全量验证，3 commits，树干净）。
- 正式基线测量：**未运行——被数据新鲜度门禁阻断**。没有任何基线数值、没有
  「是否值得搜索」的研究裁决；本日志不伪造任何中间结果。
- 按整改方案 R01 的裁决语义，本轮当前结论为：**瓶颈 = 数据新鲜度（G8），
  阻断所有正式研究入口**；是否值得搜索的裁决保持「无证据」状态，等待数据
  解锁后重跑。负结果/阻断均是合法交付形态（AGENTS §1 原则 1）。

## 4 双审与修复轮（2026-09-16）

两个独立审核子 agent（用户指令恰两个，只读审查）：

| 审 | 结论 | 要点 |
|---|---|---|
| 合规审 | **PASS** | 受控闭环/RED 证据/G8 fail-closed 无绕行/版本表/白名单/契约登记/不同步决定全部核过；3 项文档精度缺陷（C1 拒绝文本出处误标 .log（实为 stderr 捕获）、C2 日历缺口断言（勘误 2 已撤回）、C3 全量分解算术（勘误 3 已修））+ 2 项 low 观察（C4 契约 rev 1 随实现 commit 可接受、C5 建议驱动自校验契约哈希） |
| 技术审 | **NEEDS_REVISION** | T1(HIGH) loader_frame_status 引用不存在的 `reason_codes` 字段（降级契约披露路径必崩，R1b RED）；T2(HIGH) run 打开后失败路径不完整（预算/异常可留非终态悬挂）；T3(MED) 缺 §3.5 逐 (臂,折) 失败记账；T4(MED) 已完成裁决的门禁拒绝误记 FAILED（应为 REJECTED）；T5-T9(LOW) 规则/字段/标签/测试缝/空消融守卫 |

处置（修复 commit，契约 rev 2 已先记）：

- **已修复**：T1、T2、T3、T4、T5、T6、T7、T8、T9、C5 —— 详见契约 §4 Rev 2
  逐条；RED（commit `0edbb12`，6 处失败恰为修复点）→ 修复 →
  **聚焦 14 passed**（122.03s）+ 相邻 **207 passed**（482.56s）+
  compileall 0 错 + 分片 107/107 + diff-check 干净 + 全量结果见下。
- **C1/C3**：勘误 3（commit `5537ade`）。
- **C2**：勘误 2（commit `2a3459a`）已覆盖。
- **C4**：无需修复（合规审已裁可接受；未来实现期缺陷修复优先独立契约修订
  commit）。
- 技术审 NOT COVERED 项（ProtocolConfig dataclass 形态、conftest helpers、
  RunStore 严格 JSON、formal entry 门禁次序原文）不影响本包结论：其中
  ProtocolConfig 已被实现实际消费并经全量验证，其余与修复无关。
