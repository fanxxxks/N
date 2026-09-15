# P32 B03 可恢复执行与预算控制测量日志

本文只记录「实际发生什么」（AGENTS §11.1）；计划以
[docs/p32_b03_resumable_execution_budget_draft.md](p32_b03_resumable_execution_budget_draft.md)
为准。

## 1 实施证据

| 项 | 值 |
|---|---|
| 分支 | `codex/b03-resumable-execution-budget`（自 `codex/b02-entry-canary-ci` @ `2ccbbc4` 创建） |
| 契约 commit | `b14d30e` |
| RED commit | `a3bfb16`（tests/test_b03_campaign_executor.py 18 项 + 分片注册） |
| 实现 commit | `fc3243a`（被测实现） |
| 工作树 | 实现提交时仅 `reports/` 未跟踪（用户既有产物，未触碰） |
| 首次 RED 证据 | `ImportError: cannot import name 'campaign_executor'`（模块缺失，2026-09-15 19:43）；随后 R1–R9/守卫逐项红转绿 |
| 版本影响 | 无 `*_VERSION` bump（加性可选字段，p32 §6）；`CAMPAIGN_STATE_SCHEMA_VERSION = 1` 新增 |

## 2 验证命令与结果

环境：Windows（本机），Python `D:\minequant\.venv\Scripts\python.exe`（venv 解释器），
4 CPU 逻辑核量级、`pytest -n auto`（xdist），无网络依赖。

| # | 命令 | 结果 | 备注 |
|---|---|---|---|
| V1 | `python -m pytest tests/test_b03_campaign_executor.py -q -p no:cacheprovider` | **18 passed**（多次迭代后稳定；运行约 6–7 min，R4/R9 为真实子进程演练） | R1–R9 九场景 + G1–G6 守卫全部通过 |
| V2 | `python -m pytest tests/test_task_executor.py tests/test_ledger.py tests/test_run_store.py tests/test_eval_module_split.py scripts/check_test_shards.py -q` | **89 passed** | 相邻契约（E3 执行器/账本/RunStore/facade/分片权威） |
| V3 | `python -m pytest tests/test_evaluation.py tests/test_research_domain.py -q` | **73 passed** | evaluation 入口与 run_protocol 消费方 |
| V4 | `python scripts/check_test_shards.py` | exit 0；`105 files across 4 shards cover all 105 test files` | 新测试文件同 commit 注册 |
| V5 | `python -m compileall -j 0 -q ashare_data ashare_model ashare_portfolio ashare_trading scripts webapi` | exit 0 | — |
| V6 | `git diff --check` | exit 0 | — |
| V7 | `python -m pytest -q tests -n auto` | **1938 passed, 1 skipped, 0 failed**，825.09s（13:45） | 基线对齐：p31 记录 1920P/1S @ 6b9963b；本包新增 18 项（1938 = 1920 + 18），1 skip 为既有环境 skip |
| V8 | `powershell` 解析 `scripts/r1_campaign.ps1`（Parser::ParseFile） | 0 errors | PS1 薄委托入口语法；**端到端执行未运行**（默认会启动真实 calibration leg，属 R01/正式运行授权范围） |
| V9 | `python scripts/r1_campaign.py --help` | usage 输出正常 | CLI 冒烟；未启动真实 leg |

## 3 RED→GREEN 迭代中确认的关键行为事实

以下事实在调试中实测确认，并已写入实现/测试（不是推测）：

1. **venv launcher shim**：本机 `D:\minequant\.venv\Scripts\python.exe` 是启动器，
   真实解释器是其子进程。行命令的 RSS/预算必须按**整棵命令子树**采样
   （`task_executor._subtree_rss_mb`），预算终止必须 `_kill_tree`
   （只 terminate 直接子进程会孤儿化真实工作进程）。实测：直接子进程 RSS 3.5MB
   vs 子树 213MB。
2. **子进程死亡竞态**：子进程在采样瞬间退出时子树读数返回 None，
   `max(数值, None)` 会让 worker 崩溃（实测 TypeError，行结果 wall≈90s 处）——
   已加 None 守卫，保留前值。
3. **心跳超时语义**：worker 心跳独立于其子命令（子命令 sleep 期间心跳照常）；
   「子进程失联」的可用口径是心跳间隔 > 超时（仅首拍落地），
   R5 按此构造（与既有 test_task_executor 超时测试同型）。
4. **驱动树被杀后的恢复链**：campaign 陈旧锁 + leg run 陈旧写锁同时存在；
   resume 侧先经 campaign 锁持有人 pid 死亡核验（显式 `--clear-stale-lock`），
   同一已验证事实授权清理该 leg run 的写锁后 attach 同 run
   （interrupted attempt 关闭为 failed + 补执行，账本证据保留）。
   实测账本序列 `[running, failed, running, succeeded]`。
5. **用户中止**：`_thread.interrupt_main()` 将 KeyboardInterrupt 送入主线程；
   TaskExecutor 杀活动 worker 树（cmdline 归属核验）后重抛，驱动记录
   leg=aborted、exit 130；中断若在 leg 启动前到达则无 in-flight leg
   （测试已改为等 stub 宣告存活后中断，保证演练确定性）。

## 4 未运行项与剩余风险

* **真实 R1 campaign 端到端演练未运行**：驱动默认（无参）会对真实双库做一次
  全量复合确认并启动 calibration leg（小时级、需要正式授权）。本包验证覆盖
  到「驱动 × stub 命令」与「evaluation × 真实 canary 库（G6）」两层，
  「驱动 × 真实 evaluation leg」的首次连接留给 R01 的正式演练（p32 §8 停止条件）。
* **PS1 端到端未运行**（V8 仅语法/解析）。
* 预算数值未冻结到 R1 manifest（p32 §3.3：数值属 campaign 计划，非代码）；
  现有 manifest 字段为 `None`（显式不设限）。
* 复用判定的数据身份是**全量确认**（每次驱动启动一次重哈希）；真实库上的
  耗时未测量（首次真实运行时记录）。

## 5 裁决

工程验收按 p32 §7 对照成立：九个失败演练通过、终态三义与退出码 0/1/130/2
成立、部分失败不返回整体成功、恢复语义显式（attach 新 attempt 或显式重跑）、
杀树前归属核验、账本一致（hash 链由 ExperimentLedger 读取时强制验证）。
本包不产生任何研究结论，不解除 STOP_SIGNAL，不构成任何晋级/运行证据。
