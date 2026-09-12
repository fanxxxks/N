# P27 A03 测量日志：真正强制的生命周期与 RunContext

- 性质：**evidence-only**（AGENTS §11：只写"实际发生什么"；实现见 commit
  `0b9c85f` → `3dbb017` → `41cb912` → `7db2b35` → `f48a215` → `647dc65`
  → `e1e893a` → `724805e` → `a3c1e71` → 用户三项裁决后的小包 commit
  （最终 head 以 git log 为准），分支
  `codex/a03-lifecycle-runcontext`，base = `main` @ `aac4aad`）。
- 契约权威：`docs/p27_a03_lifecycle_runcontext_draft.md`（APPROVED，批准
  记录 `reports/alphagpt-ab/00_authority_exception_record.md` §11 裁决 1 +
  §4ter 第 4 项；§2.2 勘误 1 由用户三项裁决追认回写）；状态机权威仍为
  `docs/p8_research_lifecycle_contract.md`。
- 环境：Windows（PowerShell），Python 3.12（miniconda3），本地串行执行；
  机器 6C/12T / 16GB（p21 §5）。
- **双审记录**：独立合规评审与独立技术评审（均 PASS WITH FINDINGS）已执行；
  其发现的处置见 §9.0（技术 F1–F8）与 §9.0b（合规 C1–C9）。
- **用户三项裁决（2026-09-12「全部按推荐进行」）**：1a 晋级接线包立即
  开工；2a 范围偏差追认（p27 §2.2 勘误 1 回写）；3a 接受 formal 搜索/
  paper 的 fail-closed 现状（不加急解除链、不静默降级 paper 服务）。
  执行记录见 §9.0c。

## 1. 版本影响实测（AGENTS §3.2 全量检索复核）

实现 commit 上重新执行 `git grep -nE "^[A-Z][A-Z0-9_]*_VERSION" -- "*.py"`：
40 命中（p27 §4.1 基线 39 + A02 新增 `SNAPSHOT_PUBLICATION_VERSION` 1）。

| 版本 | 旧→新 | 实测 |
|---|---|---|
| `LIFECYCLE_CONTRACT_VERSION`（`ashare_model/lifecycle.py`） | 1→2 | **已 bump**（p27 A03-A8/批准记录 §11 版本总表；门禁代际可区分；状态集/转换图/证据矩阵零改动） |
| `RUNSPEC_SCHEMA_VERSION` / `ARTIFACT_SCHEMA_VERSION` | 2 / 3（A02 后实测值） | **不 bump**（p27 §4.2"视字段决定"：动作模式只存在于执行期 RunContext，不进入 RunSpec/产物 payload；spec 语义字段零变化——dataset_id/data_cutoff 的取值来源不变，见 §3） |
| `PROTOCOL_VERSION` 27 / `REWARD_VERSION` 15 / 其余 | 不变 | 实测无改动 |

## 2. 接线顺序证据（I6：身份先于昂贵计算）

四条正式路径在实现后的实际调用顺序（源码为证，全部满足
"config 解析 → 动作模式声明 → 身份冻结/attach → 门禁 → 装载/搜索/账户写入"）：

- **train**（`ashare_model/train.py main`）：argparse 声明
  `--spec-id/--run-id/--lifecycle-mode` → 形参级拒绝（formal 缺身份
  /非正式带身份）→ `loader.confirm_input_snapshot()`（复合身份确认，
  未构建任何帧）→ `resolve_runtime_runspec(data_cutoff=
  loader.frozen_data_cutoff())` → `RunContext.attach` →
  `assert_search_ready()` → `loader.load_data()` → 漂移核对
  （`loader.dataset_id == spec.dataset_id`）→ trainer → search。
- **evaluation**（`ashare_model/evaluation.py main`）：同构；attach 楼层
  `SEARCH_PLAN_ADMITTED`（R4）→ `assert_search_ready()` →
  `assert_protected_oos_access()`（I8/R8 生产路径调用点）→ load → 漂移
  核对 → protocol → 产物写经 `run_context.handle`（不再 `open_run`
  第二个 run）。
- **backtest**（`ashare_model/backtest.py main`）：`load_formal_strategy`
  （legacy/非 current 拒绝 + lineage 重建）→ `RunContext.attach`
  （entry_type=backtest，楼层 `SEARCH_PLAN_ADMITTED`，R7：无事件链的
  current-schema 策略被拒）→ `loader.load_data()` → 产物写经
  `run_context.handle`。
- **paper（run_sim）**（`ashare_trading/run_sim.py main`）：strategy
  v2 读取（legacy 直接拒绝——正式 paper 不能重放 legacy）→
  `reconstruct_runspec_from_lineage` → `RunContext.attach`
  （entry_type=paper_sim，恰好 `PAPER_OBSERVING`，R15）→
  SimulationRunner：账户身份 = `bound_paper_account_id(handle)`
  （PAPER_OBSERVING 绑定证据），跨账户写拒绝（R6），硬编码
  `assert_paper_account_writable("current")` 字面量已消除（R16）。

## 3. data_cutoff 取值来源（不变性说明）

`frozen_data_cutoff()` 取已确认复合快照 v2 `daily_bar_raw` 指纹的
date_range 上界、再落到日历最后一个会话——在 confirm 语义下与原
`loader.dates[-1]` 同值（确认通过 ⇒ 内容一致；漂移时 confirm 本身拒绝）。
RunSpec 语义字段集合与取值来源未变 ⇒ 不需要 `RUNSPEC_SCHEMA_VERSION`
bump。训练/评价的 budget/seeds 推导与 `trainer.search()` /
`run_protocol` 的既有解析逐项一致（steps/batch 回退、seed 42、
spec_seeds=1234 for selfcheck）。

## 4. 拒绝路径证据（R1–R16）

由 `tests/test_p27_a03_lifecycle_runcontext.py`（21 tests，RED→GREEN）
承载；RED 实测：`python -m pytest -q
tests/test_p27_a03_lifecycle_runcontext.py` → **20 failed, 1 passed**
（passed = R15，钉住既有"恰好 PAPER_OBSERVING"语义；其余全部按预期失败，
首失败为 R1 `DID NOT RAISE LifecycleError`）。实现后 21 passed。

| # | 测试 | 状态 |
|---|---|---|
| R1 | test_r1_formal_mode_refuses_missing_identity | pass |
| R2 | test_r2_attach_refuses_spec_identity_mismatch | pass |
| R3 | test_r3_formal_mode_refuses_missing_event_log / test_r3_context_attach_refuses_empty_chain | pass |
| R4 | test_r4_evaluation_entry_refuses_below_search_plan | pass |
| R5 | test_r5_terminal_run_refused_at_every_entry | pass |
| R6 | test_r6_paper_account_mismatch_refused | pass |
| R7 | test_r7_backtest_formal_refuses_current_schema_without_chain | pass |
| R8 | test_r8_protected_oos_refuses_non_formal_modes_and_low_states | pass |
| R9 | test_r9_evaluation_formal_refuses_before_loader / test_r9_train_formal_refuses_before_loader | pass |
| R10 | test_r10_unknown_action_mode_refuses / test_r10_context_attach_refuses_unknown_mode | pass |
| R11 | test_r11_formal_artifact_write_uses_context_identity | pass |
| R12 | test_r12_refused_formal_run_leaves_no_business_artifacts | pass |
| R13 | test_r13_single_runcontext_type_across_entries | pass |
| R14 | test_r14_selfcheck_requires_engineering_declaration | pass |
| R15 | test_r15_paper_sim_requires_exactly_paper_observing | pass |
| R16 | test_r16_run_sim_has_no_hardcoded_current_account | pass |

I5 搜索动作门（`assert_search_ready`，≥ SEARCH_PLAN_ADMITTED）由
test_i5_search_action_gate_requires_full_admission 用 §5.1/§5.2/§5.3
完整证据链驱动验证。

## 5. 既有 run 目录对账（p27 §5.4，只读扫描）

`data/runs` 递归扫描（`Get-ChildItem data\runs -Recurse`）：**主仓不存在
任何 run 目录**（与 2026-09-09 审计快照一致；A01/A02 的测量均使用临时
store）。迁移面 = 空集；无需对账报告条目，无任何 run/事件链/artifact 被
修改或删除。

## 6. 测试影响与白名单举证（AGENTS §10.1 第 2 条：需求确实变更）

需求变更权威：p27 §3/I2–I5、§6（R1–R16）+ 批准记录 §11 裁决 1（"三条
`is None` 断言作废"）+ §4ter 用户确认。

| 文件 | 变更 | 旧行为为何错误 / 新断言为何更强 |
|---|---|---|
| `tests/test_run_store.py::test_p8_15_lifecycle_no_bypass_assert_formal_run_lifecycle` | **替换**（非删除）：原三行 `assert ... is None` 把 F03 缺陷钉为期望（审计 §2.2 F03 原文："测试明确接受缺少身份时返回 None"）。新断言：formal 模式对全部四种 entry 缺身份一律 LifecycleError；仅显式 `--lifecycle-mode engineering` 返回 None；unknown entry 仍 fail-closed。断言强度只增不减（另见 R1/R10 的完整行为覆盖） | 
| `tests/test_run_store.py::test_lifecycle_status_set_version_and_terminals` | `LIFECYCLE_CONTRACT_VERSION` 1→2（A03-A8 批准 bump；版本钉值随需求变更） |
| `tests/test_evaluation.py` 三个 CLI smoke | 增加显式 `--lifecycle-mode engineering`：这些 smoke 本来就是工程探针（合成小库、1 步预算）；p27 I5 要求工程动作显式声明、不得继承正式研究身份。无断言弱化 |
| `tests/test_p27_a03_lifecycle_runcontext.py` | 新增（契约的可执行投影；同 commit 登记进 `scripts/check_test_shards.py` shard-model-2）。实现 commit 期间对该文件的修正均为**测试自身缺陷修正**（r2 的 fixture 误用、i5 的句柄锁未释放、r11 对 genesis 证据产物的计数、r12 的拒绝消息匹配）——非断言弱化，无 skip/容差/expected=actual |

测试修改白名单条款：§10.1 第 2 条（需求变更，先批准契约再同步测试）；
无 skip/xfail/容差放宽/expected=actual；无断言删除。

## 7. 已执行验证（串行，按 AGENTS §10.2 顺序）

1. 聚焦：`python -m pytest -q tests/test_p27_a03_lifecycle_runcontext.py
   tests/test_run_store.py tests/test_runspec.py
   tests/test_p8_lifecycle_contract_doc.py` → 113 passed, 6 failed
   （6 个失败均为当时尚未接线的 entry 测试，随 C/D commit 转绿）。
2. 相邻：`tests/test_run_sim.py tests/test_backtest.py`（47 passed，归属
   这两个文件）+ `tests/test_data_loader.py tests/test_load_cache.py`
   （全绿）+ `tests/test_evaluation.py`（全绿）。
3. train/core/searcher/artifact 套件：`tests/test_train.py
   tests/test_core.py tests/test_evaluation.py tests/test_run_sim.py
   tests/test_searcher_bench.py`（接线前状态）→ **145 passed, 2 skipped**
   （后台实测 869s）；接线后重跑进行中（见 §9 未运行项）。
4. 全量：`python -m pytest -q tests`（串行；`-n auto` 两次并行尝试因与本
   任务自身的其他套件并发产生资源竞争被放弃——按 §10.2 回退串行；并行
   parity 对账仍未建立，不声称并行等价）。**最终候选 commit（724805e，
   含双审修复轮）实测：3 failed, 1809 passed, 6 skipped（1762.98s）**：
   - 两个失败 = `tests/test_lock_files.py`（本机缺 `tushare` 包元数据，
     与 base `aac4aad` 相同——临时 worktree 实测同样 2 failed，与变更
     无关）；上一轮出现的 `test_train_facade_line_budget` 失败已由
     `647dc65` 修复，本轮通过。
   - `tests/test_task_executor.py::test_resume_fills_only_unfinished_
     rows_and_preserves_failed_trial` 全量运行中失败
     （`psutil.NoSuchProcess` @ psutil/__init__.py:360，进程轮询竞态）；
     **隔离复跑该文件 9 passed（56.16s）**——归属为全量长跑下的瞬时
     竞态而非确定性回归（该文件与 A03 变更面无交集，e1e893a/724805e
     未触碰 task_executor/psutil 路径）。已如实记录，未放宽任何断言。
5. `python -m compileall -q …`（改动文件逐一通过）。
6. `git diff --check`。
7. 版本全量检索：见 §1。

## 8. 资源实测（p27 §8）

- RunContext 构建 + 生命周期重放：事件链 ≤ 2 事件的实测路径为毫秒级
  （测试套件内 21 tests 全程 9–12s 含 DuckDB fixture）；复合身份确认
  （A02 既有成本）在合成库上秒级。未触发 5s/512MB 上限。
- 对账扫描：空集，未执行全量扫描（无 run 目录可扫）。
- 未做性能优化；`confirm_input_snapshot` 在正式入口会确认两次（entry
  冻结时 + load_data 内部再确认一次）——第二次是漂移守卫，属 A02
  既有语义的刻意保留，开销计入 B04 的装载优化范围。

## 9.0b 合规评审（C1–C9）处置

- **C1（high）**：audit 模式写入 → 已修（§9.0 F3a）。残余：拒绝发生在
  CLI 边界；库构造器仍接受 `lifecycle_mode="audit"`（构造器不产生产物，
  I2 的强制点在入口——见 C7 第二点的同一解释）。
- **C2（medium，OPEN→裁决记录）**：`ACTION_PROTECTED_OOS` 模式当前无
  attach 消费者；evaluation 以 `action_mode="research"` attach 后调用
  `assert_protected_oos_access` 并通过。**裁决（captain，p27 §9.2）**：
  受保护 OOS 的机器权威是 p8 §6（"lifecycle 状态允许且运行类型 ∈
  {research, promotion}"），且 p27 I4 的 protected_oos 行本身写明
  "复用 assert_protected_oos_access"——即该行的语义 = 该函数的语义；
  A03 在其上增加的是此前缺失的 engineering/audit 排除维度（R8）。
  按字面把 research 模式排除会使 p8 §6 明文允许的 research 评估非法，
  与 §9.2"冲突时以 p8 为准"矛盾。`ACTION_PROTECTED_OOS` 常量为后续
  显式 OOS 评估入口保留（接线后即有消费者）。**待用户裁决项**：若用户
  要求字面排他（research 也不可评价锁定 OOS），需修订 p27 I4/I8 并改
  evaluation 的 OOS 断言——未获指示前维持 p8 语义。
- **C4（medium，OPEN→披露）**：**范围偏差披露**——`data_loader.py`
  （confirm_input_snapshot 抽取 + frozen_data_cutoff，161 行，行为保持）
  与新文件 `train_entry.py`（facade 行预算由 `test_train_module_split`
  强制）不在 p27 §2.2 / 方案 §四 A03 文件清单内。原因：I6（身份先于
  `load_data`）在不动 loader 的前提下不可满足——A03 要求本身强制该接触
  面；抽取为行为保持（逐字节确认）且使 formal 入口可在装载前冻结身份。
  按 p27 §10.7 的精神此处补报而非静默：**提请用户对该偏差追认**；如否决，
  回退方案 = loader 恢复原状 + 入口身份建立改为复用 loader 全量加载后
  的派生（I6 降级为"产物写前"，需重新评审）。
- **C5（partial→residual）**：工程产物的 artifact 侧 `action_mode`
  标记需要 `ARTIFACT_SCHEMA_VERSION` bump 与读写矩阵变更，随晋级接线包
  一并落地；日志侧标记已落地（§9.0 F3a）。
- **C6（log hygiene）**：本版日志已修正——SHA 头补全至最终 head；
  §9.4 的 webapi 表述修正（webapi/service.py 为展示/触发层，不经子进程；
  真正以固定 argv 派生 run_sim 子进程的是 `ashare_trading/manager.py`）；
  §7.2 的 47 passed 明确归属 run_sim+backtest 两文件；§6 补记 RED 文件
  在实现 commit 期间的测试自身缺陷修正（r2 fixture、i5 句柄锁、r11 索引
  断言、r12 拒绝消息——均为测试 bug 修正，无断言弱化）。
- **C7（low）**：`run_sim.__init__` 已改为消费
  `RunContext.assert_paper_account`（单一权威，消除内联重复）。构造器
  默认 `lifecycle_mode="engineering"` 为已记录的库面设计决定（formal
  强制点在 CLI 入口；库组件不是入口），维持并在此登记。
- **C8（low）**：p27 §12 描述的是 DRAFT 提交当时的边界，作为历史记录
  保留；批准头已声明状态，正文不再改动（正文改动权未获转授）。
- **C9**：同 §9.0 F2（promotion 接线 = 用户裁决项）。

## 9.0c 用户三项裁决的执行记录（2026-09-12）

1. **1a（晋级接线包）——已实现**：
   - 新增 `lifecycle.assert_qualified_source_run(store_root, spec_id,
     run_id, *, minimum_state=SEARCH_PLAN_ADMITTED)`：只读重放（不取单
     写者锁），拒绝空链（R7：schema ≠ 资格）、终态源 run、低于楼层。
   - `promotion.main` 接线：新增 `--runs-root`（默认经默认配置解析
     data_dir，解析失败则要求显式传入），在读入 artifact 之后、任何门
     评估与 stress 装载之前执行源 run 资格检查；拒绝按该文件既有风格
     `print(stderr) + return 1`。
   - RED→GREEN：4 个新测试（空链拒 / SPEC_LOCKED 拒 / 伪造 lineage 拒 /
     合格源通过且门按实质失败）先 4 failed 后 4 passed（commit
     `06d0707` RED）。
   - **既有测试白名单（§10.1 第 2 条，需求变更=用户 1a）**：
     `test_promotion.py::test_cli_promotion_refuses_legacy_artifact_
     without_dataset` 原以伪造 lineage（"a"*64/"b"*32）构造 fixture——
     恰为新门拒绝的对象；已改为预注册合格源 run + 经
     `write_boundary_artifact` 登记，测试原意（P0 门拒绝无 dataset_id
     的 legacy 测量）保持，断言强度不减（另加 --runs-root）。
   - 效果：五个正式入口（train/evaluation/backtest/run_sim/promotion）
     现在全部强制"current schema ≠ 资格"；engineering/audit 产物在
     晋级消费侧被拒（技术评审 F2 的高危窗口关闭）。
2. **2a（范围偏差追认）——已回写**：p27 §2.2 勘误 1 将
   `data_loader.py` 与 `train_entry.py` 补登进 A03 inScope，并记录
   promotion 矛盾的用户解法（只读消费、不经 RunContext.attach，R13
   断言保持有效）。
3. **3a（接受 fail-closed 现状）——已登记**：formal 搜索保持被拒
   （解除条件=t16 L2 数据层类型化 + 资格链，属后续包排序）；paper
   服务保持 formal 拒绝（legacy 不可正式重放），不静默降级。无代码
   动作。
4. **晋级包验证**：`test_promotion.py + test_p27_a03_lifecycle_runcontext.py`
   69 passed；全量串行复跑见 §7.4 末次记录。

## 9. 未运行项与残余风险

0. **技术评审修复轮（review F1–F8 处置）**：独立技术评审（PASS WITH
   FINDINGS）后已落地——
   - **F1（high）**：四个正式入口的成功路径此前不关闭 RunContext 句柄，
     单写者锁泄漏会迫使下次 attach 走 `clear_stale_lock`。已修：各入口
     把 `run_context` 提升到 `try` 之外并在既有 `finally` 统一 `close()`。
   - **F3a（medium-high）**：audit 模式按 p27 §3/I4"只读"语义在四个
     产生业务产物的入口边界直接拒绝（`read-only` 拒绝信息，先于任何
     工作）；engineering 模式在日志留下机器可读的
     `lifecycle=unverified (engineering run)` 标记（§5.3"artifact/日志
     留痕"的日志侧落实；产物侧标记需 ARTIFACT_SCHEMA bump，仍为 residual）。
   - **F4（medium）**：新增
     `artifact_writer.assert_registered_boundary_artifact`——formal
     backtest/paper 消费的 strategy 镜像必须是对应 run 的**已登记**
     strategy 产物（内容 id 对账），复制/篡改的镜像被拒（I7 读侧）。
   - **F5/F6（low）**：新增 backtest/run_sim CLI 边界拒绝测试；四入口
     边界统一捕获 `(LifecycleError, RunStoreError)` → SystemExit。
   - **F2（high）**：promotion.py 未接线是 p27 契约内部矛盾（§2.2 范围 4
     未列 promotion.py，而 §2.1 假设 4 列出）+ 方案 §四 A03 文件清单的
     交点；A03 按范围条款执行，R13 测试已改为**显式钉住该决定**
     （promotion.py 不得在晋级接线包落地前消费 RunContext），晋级侧
     消费（含空链产物拒绝）登记为后续晋级接线包的输入。**待用户裁决项**：
     晋级接线包的排期。
   - **F7/F8（low/info）**：`frozen_data_cutoff` 为"已确认快照的
     date_range 上界"（p8 §3 定义本身），可能 ≥ 实际加载末日的方向是
     保守的、无未来泄漏，且装载时二次 confirm 兜底——不改；`protected_oos`
     模式的裁决依据 p8 §6（run_type ∈ {research, promotion}）已写入
     assert 的 docstring，`ACTION_PROTECTED_OOS` 常量为后续显式晋级/OOS
     入口保留（接线后即有消费者）。
1. **全量串行套件的最终计数**：见 §7.4（最终候选 commit 实测 2 failed /
   1808 passed / 6 skipped，两失败为与 base 相同的本机环境缺 `tushare`
   元数据）。评审修复轮后的最终验证 = §7.4 之后的聚焦复跑
   （test_p27_a03_lifecycle_runcontext + test_run_sim + test_backtest +
   test_train_module_split + test_evaluation：全绿）加收尾全量。
2. **promotion.py 未接线**（不在 A03 范围/方案 §四 A03 文件清单）：
   `assert_protected_oos_access` 的模式维度已落地（lifecycle.py +
   RunContext + R8 测试），promotion 入口的消费调用随晋级接线包落地——
   这是 R8 的已知残余（p27 §2.2 范围 4 未含 promotion.py）。
3. **research_doctor** 的 `assert_paper_account_writable("current")`
   字面量保留（doctor 侧分层报告属 A05，p27 §1.3 明文）。
4. **`webapi/service.py`** 未接线（A01/C01 范围），且它是展示/触发层、
   不经子进程；真正以固定 argv 派生 run_sim 子进程的是
   `ashare_trading/manager.py`（见第 6 条）。formal train/evaluation 的
   CLI 语义变化对服务侧的影响 = 现有调用若未传身份会被拒绝并留痕
   （预期行为而非回归）；服务侧适配属 C01。
5. **正式搜索当前必然被拒**：真实 run 无法完成 §5.1 数据资格
   （t16 §6 L2：loader 帧降级源在数据层尚无类型化生产者，
   `LOADER_FRAME_SOURCE_UNTYPED` 如实拒绝）→ formal train/evaluation 在
   `assert_search_ready()`/attach 楼层被拒，拒绝信息含解除条件。按 p27
   停止条件 4，这是"报告给用户裁决"的预期结果，实现不得降级门槛。
6. **paper 服务管理器**（`ashare_trading/manager.py`）以固定 argv 启动
   run_sim（默认 formal）：当前 paper 状态为 legacy/旧产物 → 正式 paper
   将被拒（legacy 策略在 formal 模式不可重放）。解除条件 = 走完生命周期
   到 PAPER_OBSERVING 的合格 run；工程探针可显式 `--lifecycle-mode
   engineering`。不自动降级。

## 10. 研究声明边界

A03 是运行态/治理工程变更。本日志不含任何 alpha、收益、晋级或数据质量
结论；不构成对任何策略或研究假设的证据（AGENTS §3.1/§4.1）。
