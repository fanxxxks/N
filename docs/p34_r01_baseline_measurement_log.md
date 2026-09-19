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

### 4.1 修复后正式入口复核（2026-09-16 20:21）

修复 commit `4f25abf` 后以当前命令重跑正式入口两次：

1. 旧哈希 `1779539…` → **被 C5 自校验拒绝**（"contract sha256 mismatch"）——
   契约经 rev 2 演进后旧印签不再被接受，provenance 自验按设计生效。
2. 当前契约哈希 `8264362761698befd447b44c5b4008a366870f7e3aa7529d2c22b0d678fd5d37`
   → 自校验通过 → G1–G7 通过 → **G8 同样拒绝**（daily 20260907 vs reference
   20260915, lag=6 > N=3, today=20260916），exit 2，`data/r1/medium_a/runs`
   不存在（无部分状态）。

结论维持 §3：阻断 = G8 数据新鲜度；修复未改变边界行为，仅完成诚实终态与
判别器归位。**注意：后续重跑必须印签当时已提交契约文件的实时 sha256**
（`Get-FileHash docs/p34_...md`），契约每次修订都会使旧印签失效（设计如此）。

## 5 用户授权后的双库同步与正式运行（2026-09-16 深夜 – 09-17 凌晨）

用户指示「全部授权」（授权本节全部有状态数据操作）+「完成后自动关机」。
按 AGENTS §5.6 顺序执行：只读审计 → 备份 → 执行 → 复核。

### 5.1 预同步审计与备份（只读留档）

- 审计脚本/输出：`D:\minequant\reports\r01_presync_audit.py` /
  `r01_presync_audit_20260916.txt`。主库 daily_bar 4,919,070 行
  （20150105..20260907）、calendar 8,797、constituents 2,574、stocks 5,542；
  v2 各价格表 max 20260907（suspension 0904、margin 0903）；
  `snapshot_publication` 仅一组完整 composite（publication 1）。
- 主库备份：`data/backup_20260916_r01/ashare.duckdb`（1,015,296,000 字节）。

### 5.2 v2 增量回填（scripts/backfill_v2.py --steps 2,3,4,8）

2.8 分钟完成：44 批次、182,604 行（逐日端点账本去重，仅拉 09-07..09-16 新
会话；margin 至 09-15 T+1）；新 v2 manifest `67ee0b32…`（19 表 57,339,133 行，
range 20140102..20260916）。**范围收缩记录**：跳过 steps 5–7（行业/财报/
事件族）——R01 Tier-A 价格+margin 因子集不消费其新鲜度；此决定的后果见
§5.4(b)。

### 5.3 主库 AkShare 同步（python -m ashare_data.sync）

- 第一遍（`--no-fundamentals --no-capital-flow`）：约 55 分钟，universe
  2,118 码，failures 0，管线自清理 14,102 行界外旧行；daily_bar →
  **20260916**；主库 manifest generation 2（`281cbb24…`）。
- 第二遍（`--no-fundamentals`，capital-flow 愈合）：margin 2,846 日
  （5,719,773 行）、申万行业 141,378 行入主库；主库 manifest generation 4
  （`8c700f61…`）。**申万指数收盘源未前进**（仍 20260907，见 §5.4(b)）；
  另观察到本地 akshare 包解析错误（"Length mismatch … 13 elements"，
  margin sse 20260916）——疑似本地包版本与上游 API 不兼容（依赖治理范围）。

### 5.4 诊断出的残余降级集合（类型化，全部如实入证据）

(a) **价格语义（001914.SZ 1,370 会话——结构性）**：其 2014 起的 bar 历史
    挂在改名前代码下，tushare stk_limit 按现码只能取回 1,627 行，前身时段
    限额不可寻址。另有 000022.SZ（2）/688033.SH（1）零星缺口：按码补拉已
    入库（TushareStore 权威，账本 `daily_price_limit_20260916T154352Z_*`）
    但单元格仍缺（疑似停牌会话源无行，保留披露）。
(b) **行业源滞后 7 会话**：主库/ v2 申万指数收盘均止于 20260907——AkShare
    源本身未更新（capital-flow 重跑无前进）+ 本地包解析错误。**系统性阻断**：
    依赖升级或换源属 p17/D5 与依赖治理决策，不在本次授权内。
(c) **全格网 missing_limit 披露语义**：loader 对全 (码×会话) 格网判定，
    post-2014 上市的 pre-listing 空格也计入 → **1,630 码** price_degraded
    （同步前后两次运行计数一致 =1,630，与本次同步无关，为 loader 既有形态；
    我构建的类型化接线首次使其可见）。p19 §2.4 的最严读法支持现状；
    收窄属 p19 契约修订决策。

### 5.5 正式运行记录（三次尝试，全部诚实终态）

1. 22:49（同步前复核后）：进程在冷载中途被操作者终止（发现类型化降级，
   先修数据）——run `e0d7da37/35e41920` 留于 SPEC_LOCKED；事后经
   `clear_stale_lock`（存活校验通过）+ `writer.fail` 诚实收口为 **FAILED**。
2. 23:49：冷载完成（factor cache put `fe96090e`）→ 数据报告被拒——暴露
   驱动 bug：日历对账把未来预挂会话 20260917..20261231 计为缺失 →
   **REJECTED**（run `2983d61d/ac445541`）。修复：预期会话按确认数据截止日
   封口（RED `test_r1c`，commit `a6689a1`）。
3. 01:16（cutoff 修复后，factor cache hit）：G1–G8 全过 → composite
   `767d955b…` 确认 → 装载 → 数据报告被 §5.1 拒绝（1,630 码价格语义 +
   行业滞后 7，类型化全披露）→ **REJECTED**
   （run `f80e042e/31d16c7c`；证据 artifact：idea_record / spec_lock /
   data_qualification_report 均已登记）。

### 5.6 当前结论

- 数据底座已恢复新鲜（双库至 09-16/09-15，composite `767d955b…` 确认通过），
  G1–G8 门禁全过；G8 新鲜度阻断解除。
- **R01 正式基线测量仍未能执行**：DATA_QUALIFIED 因 §5.4 的三类未消除降级
  fail-closed 拒绝（p8 §5.1「任何降级未消除 ⇒ 拒绝转换」按契约执行）。
  这是门禁栈对「当前数据底座不可达标」的正确回答，不是故障。
- 解除条件（均属契约治理决策，超出本次授权）：
  1. 申万指数收盘源新鲜度恢复（本地 akshare 升级或 D5 换源，p17 范围）；
  2. pre-listing 限额格网的披露语义裁决（p19 契约修订）或 001914 前身历史
     限额的数据修复。
- 两个 REJECTED 运行与一个 FAILED 运行的全部证据保留于
  `data/r1/medium_a/runs/`；未删除、未改写。

## 6 R01 收口（WP0，2026-09-17）——「实际跑了什么、缺了什么、结果在哪」

本节是 R01 第一轮的收口记录，回答验收三问并声明记账限制。原始规则
（p34 契约 rev 2）、全部输出与失败记录保持原样，未被本节改写。

### 6.1 实际跑了什么

- **代码**：分支 `codex/b04-load-cache-performance`，提交链
  `96c901e`（契约）→ `4c9b317`（RED）→ `2805847`（实现）→ 双审修复
  `0edbb12`/`4f25abf`（契约 rev 2）→ 日历对账修复 `a6689a1`；
  驱动 = `scripts/r01_baseline.py` + `ashare_model/r01_baseline.py`。
- **数据身份**：composite `767d955b0fd99507fed95b4275a2bdde5bdab177e1e22c8dc156a45c107e21b2`
  （primary 成员 `8c700f61…`，v2 成员 `c1ed1821…`，publication 5，全量重哈希
  确认通过）；数据截止 20260916；配置 `configs/r1_medium_a.yaml`
  （sha256 `3DC7F6CA…24D4D`）。
- **生命周期运行**：三次正式尝试，G1–G8 生产门禁在新鲜数据上全过，
  复合身份确认通过，RunSpec 冻结（SPEC_LOCKED）各次成立。
- **基线臂与折**：**一个都没有执行**。全部臂（等权基准、三单因子、两组合、
  现金控制）与全部折（train_end 2020-12-31…2024-12-31，test_end
  2021-12-31…2025-12-31）的冻结计划在 p34 契约 §3，其执行在
  DATA_QUALIFIED 之后，从未到达。

### 6.2 缺了什么

1. 基线臂执行、0x/1x/2x 成本档、年度分解、报告 artifact、worth_search
   裁决——全部未发生（三次运行均在 DATA_QUALIFIED 门前 fail-closed）。
2. 阻断原因（类型化、已登记于被拒数据报告）：
   - 申万指数收盘源滞后 7 会话（AkShare 源停止更新 + 本地包解析错误，
     依赖治理范围）；
   - 1,630 码价格语义降级（全格网 pre-listing 限额空格，loader 既有形态，
     p19 §2.4 最严读法）+ 001914.SZ 结构性 1,370 会话与两处零星缺口；
3. 依赖的前置工程缺口（本轮发现、另行处理）：回测引擎非调仓期收益记账
   使用冻结权重（等效每日零成本再平衡），见 §6.3 与 WP1。

### 6.3 权重记账限制（强制注记）

回测引擎当前在非调仓会话以**冻结目标权重**计算收益并驱动构造器
（`constructor.construct` 的 `not_due` 分支原样返回 prev；引擎按
`dot(weights, o2o)` 计利），等效于对持仓**每日零成本再平衡回目标权重**，
而非真实持仓的状态演化。WP1 将以股数基状态演化修正该语义并升级执行版本。
**在 WP1 修复落地并升版本之前，本引擎产出的任何收益数字都不存在正确的
持仓状态语义；R01 本轮（以及任何使用当前引擎的历史数字）不得作为参数
选择、搜索准入或收益排名的依据**——本轮亦无此类数字产出。

### 6.4 结果在哪里

- 生命周期运行与证据：`data/r1/medium_a/runs/` 下
  `e0d7da37…/35e41920…`（FAILED，进程终止收口）、
  `2983d61d…/ac445541…`（REJECTED，未来会话误计，驱动 bug 已修）、
  `f80e042e…/31d16c7c…`（REJECTED，残余降级全披露，最终收口运行）；
  各 run 的 artifact index 登记其证据（idea_record / spec_lock /
  data_qualification_report），append-only，未改写。
- 本测量日志 §1–§5 与审计/探针脚本：`D:\minequant\reports\r01_*.py|*.txt`。
- 数据备份：`data/backup_20260916_r01/ashare.duckdb`。
- 提交链：`96c901e…3ea1c50`（本节随收口 commit 追加）。

## 7 WP3 报告补齐 + 修正执行版本下的正式重跑（2026-09-18，用户直接指令工作包 3）

契约 Rev 3（commit `27f22a8`，用户指令「完成工作包 3：补齐 R01 报告，并按
修正后的执行版本重跑固定基线」批准）。本节 evidence-only：只记录实际发生
什么；计划见契约 §4 Rev 3。

### 7.1 软件交付

| 项 | 值 |
|---|---|
| 提交链 | 契约 Rev 3 `27f22a8` → RED `0c19f51`（tests/test_p34_r01_report_wp3.py 7 用例 + 分片注册 110 文件）→ 实现 `dcd9291` |
| RED 证据（实现前树） | 7 failed 全部落在 Rev 3 能力缺口：W1 `KeyError: 'oos_active_ir'`、W2 `KeyError: 'stitched'`、W3 `ImportError: classify_arm_diagnostics`、W5 `assert 2 == 1`（schema 钉 1→2） |
| 实现 | `evaluate_signal` 附加字段：`oos_active_ir`（行自身日序列上的权威公式，命名避开 scorer 训练窗 `active_ir`）+ 执行漏斗（due 会话五桶精确分割：no_signal/target_stable/fully_suppressed/partial_suppressed/executed_clean；no_signal 用构造器同款 `has_cross_sectional_dispersion` + 同一 mask 矩阵重判，不新增第二语义；计划 vs 成交订单/换手；threshold 与 min_trade 抑制原因拆分；持仓最小值/零持仓会话）——引擎与构造器零改动，既有字段逐位不变。`r01_baseline` 报告 schema 2：`annual`（逐折×逐档 excess/oos_active_ir/total_return + ok/failed/benchmark_missing 状态）、`fold_stats_1x`（中位/最小/最大）、`stitched`（经 `stitch_oos_series` 唯一权威按档拼接，active IR 在拼接序列上重算，失败折计入 failed_folds）、`benchmark_status`、`data_completeness`（类型化 loader 帧 + 会话对账）、`arm_summaries`、`arm_diagnostics`（描述性四向：measurement_incomplete/weak_signal/execution_blocked/cost_swallowed，阈值冻结于报告 `diagnostic_rules` 自描述；与 §3.7 冻结裁决并列、永不互替）、`engine_version_attribution`（EXECUTION_SPEC_VERSION=4 实时常量；`prior_version_numbers_exist=false` 附 WP0 §6.3 理由——exec v3 基线数字从未存在，无可归因差异，记录而非伪造）、`reporting_notes`（组合原始秩尺度披露 + 拼接 vs 折均值的区分声明） |
| 裁决/方向/成本档 | **逐字未动**（§3.2/§3.3/§3.7 冻结；组合保持原始截面秩，百分位标准化另立包） |

### 7.2 验证矩阵（均在候选树内容 = `dcd9291`）

| 验证 | 结果 |
|---|---|
| 聚焦 | `test_p34_r01_report_wp3.py` → **7 passed**（58.38s）；`test_p34_r01_baseline.py` → **15 passed**（155.17s） |
| 相邻契约 | p27/p35/p36/stitched_oos/baseline_harness → **81 passed**（53.90s）；evaluation/data_loader/run_store/p31 → **133 passed**（539.75s） |
| compileall | exit 0 |
| 分片权威 | 110 files across 4 shards（新文件同 commit 注册） |
| `git diff --check` | 干净（仅 autocrlf 提示） |
| 全量首轮（`-n auto`，p34 §6 沿用既有 parallel-vs-serial 对账基线） | **1993 passed, 1 failed, 1 skipped, 3 xfailed**（1061.90s）。唯一失败 `test_task_executor.py::test_rows_run_in_independent_subprocesses_with_heartbeat_and_metrics`：与本包零导入交集（task_executor 不触 r01_baseline/eval_metrics，反之亦然）、单测复跑 passed（13.01s）、整文件复跑 9 passed——`-n auto` 负载下子进程收集竞态，非本包回归 |
| 全量复轮（同树 `-n auto`） | **1994 passed, 0 failed, 1 skipped, 3 xfailed**（1158.98s）——上述失败未复现，确认偶发 |

既有测试修改白名单（AGENTS §10.1）：`tests/test_p34_r01_baseline.py:942` R8
报告 schema 钉 `1→2`——第 2 类（需求变更，契约 Rev 3 §4 版本表为批准依据），
断言强度不变（`==` 精确匹配，仅值随 bump 更新）。

### 7.3 正式重跑（EXECUTION_SPEC_VERSION=4；2026-09-18 11:45–11:50 本地）

命令（工作树干净 @ `dcd9291`，无并发重载进程）：

```
python scripts/r01_baseline.py \
  --contract-sha256 d5460955ab15dc026592047734bcb419358e303fb9db75374f8da23bf8857f0f \
  --out-root data/r1/r01_baseline
```

（sha 为含 Rev 3 的已提交契约文件实时 sha256；C5 自校验通过。）

| 项 | 值 |
|---|---|
| run | spec `42d8801fae1044b48d…` / run `93b17c2d8f774ba7bc91677b15501ba1` |
| 身份 | dataset `767d955b0fd99507fed95b4275a2bdde5bdab177e1e22c8dc156a45c107e21b2`（**重新实时确认**，与 09-17 一致；双库自 09-16 23:47 未变更）；cutoff `20260916`；purpose `historical_research`；`execution_spec_version=4`（RunSpec 内）；git_commit `dcd9291` |
| 生命周期 | IDEA → SPEC_LOCKED（11:45:12）→ **REJECTED**（11:50:07，诚实终态）；artifacts：idea `0b07fe09…` / spec_lock `c92c9ee8…` / data_qualification_report `a3775451…`（append-only 登记） |
| 门禁 | G1–G8 全过；G8 `basis=historical_snapshot, daily 20260916 vs reference 20260916, lag=0`（p36 历史基准首次在真实 R01 运行上生效）；日历对账 expected 2846 / loaded 3091 / **missing 0**（a6689a1 修复后形态） |
| 拒绝点 | §5.1 loader_frames **degraded**：`price_semantics_degraded`（**恰好 1,630 码**，与 §5.4(c) 同步前后两次普查一致——结构性、与同步无关）+ `industry_source_degraded`（申万指数收盘滞后 7 会话，源止于 20260907）。R01Refusal → exit 2，无部分状态、无基线行 |
| 因子缓存 | hit key `fe96090e…`（与 09-17 相同 dataset 窗口 20140102..20260916:3091），未重算 |
| 墙钟 | ≈4m55s（confirm+门禁+装载+拒绝） |

**判读**：重跑按契约 §7 走到 fail-closed 诚实终态——这正是 Rev 3 四向诊断中
`measurement_incomplete` 的真实落点（测试已证明另外三向的判定机能在受控
fixture 上成立；正式路径因数据资格未过而无法产生臂数值）。阻断集合与
WP0 收口时**逐项一致**（申万源新鲜度 / 1,630 码 pre-listing 限额格网
披露语义 / 001914 前身历史），解除条件仍是三个契约治理决策（p17-D5 换源
或依赖升级、p19 §2.4 语义裁决、001914 数据修复），均超出本包授权，
未越权处置。基线方向、候选、成本档与裁决规则零变化；worth_search 维持
「无证据」。

引擎版本归因：exec v3 基线数字从未存在（WP0 §6.3 不可采纳且未产出），
故无旧版-新版数值差异可归因——该事实已固化为报告字段
`engine_version_attribution.prior_version_numbers_exist=false`，报告 schema
v2 的机器验证见 W5。

### 7.4 勘误 4：§5.5/§6.4 两个 REJECTED run 的 ID 归属对调

核对 run store 原始证据（runspec `git_commit` + lifecycle 时间戳 + 数据
报告内容）发现 §5.5 第 2/3 条与 §6.4 把两个 REJECTED run 的 ID 写反了：

| run | runspec git_commit | SPEC_LOCKED→REJECTED（本地） | 数据报告特征 | 即 §5.5 的 |
|---|---|---|---|---|
| `f80e042e…/31d16c7c…` | `7279rc1`（修复**前**） | 09-16 23:53 → 09-17 00:57 | `missing_trading_days` 含 20260917..20261231（未来会话误计） | 第 2 条（23:49 运行，驱动 bug 暴露） |
| `2983d61d…/ac445541…` | `a6689a1`（修复**后**） | 09-17 01:15 → 01:21 | `missing=[]`，仅 1,630 码价格语义 + 行业滞后 7 | 第 3 条（01:16 收口运行，残余降级全披露） |

修复 commit `a6689a1` 时间戳 09-17 01:11 本地——23:49 的运行不可能在其上。
原叙事结论（bug 是什么、修复是什么、最终拒绝原因、四类降级清单）全部
不变，仅两处 run ID 对调；§6.4 的目录对照随之按上表理解。原始产物与
事件未做任何改写。

### 7.5 当前 R01 第一轮状态（WP3 后）

- 软件与治理：**完成**（报告 schema v2 + 四向诊断 + 执行漏斗 + 版本归因；
  全量 1994P/0F）。
- 正式基线测量：**仍被数据资格阻断**（本节 §7.3 的 REJECTED 即当前最新
  证据）；无基线数值、无 worth_search 裁决——诚实状态。
- 下一步属于各自独立授权的包：三个数据阻断的契约治理决策（见 §7.3）；
  数据解锁后以同一命令（印签当时实时契约 sha）重跑即可，本契约未过期。
