# P31 B02 测量日志（真实入口 canary 与 CI 质量）

Evidence-only（AGENTS §11.1：本文只记录实际发生的事；计划见
`docs/p31_b02_entry_canary_ci_draft.md`）。所有测量均引用精确被测 commit。

## 1. 环境

- 主机：Windows（本机），分支 `codex/b02-entry-canary-ci`。
- **解释器归属更正（合规审核 F-2，2026-09-15）**：本机有两个 Python 3.13.12（Anaconda）解释器。
  **交付审核前的全部测量**（RED 探针、聚焦/相邻/全量首跑，提交 `898dec3..d6b19f6` 所引用者）
  实际运行在 PATH 首位的 **`C:\ProgramData\miniconda3\python.exe`**（torch 元数据 2.11.0+cpu、
  **缺 `tushare` 发行元数据**；全量失败栈中的 miniconda3 路径可证）。而 onboarding §9.2 声明的
  本地运行时是 **`D:\minequant\.venv`**（torch 2.11.0+cu128、`tushare==1.4.29` 自 2026-09-04 起在位、
  `tests/test_lock_files.py` 在其中 14 passed/1 skipped）。初版日志把 miniconda3 的失败误归因为
  "本机 venv"，且环境头混合了两个解释器的报告（torch import cu128 / 元数据 cpu）——已更正；
  实质结论（2 失败为既有环境属性、与本包 diff 无关）经双审独立复核成立，且 B01 测量日志当时
  已有正确归因先例。**双审处置后的复验全部改用 venv 解释器执行**（§3 标注）。
- Node：v22 系（系统 Node.js），webui 依赖经干净 `npm ci` 按 `package-lock.json` 安装。
- 契约与提交链：
  - 契约：`1f4f724`（p31 预注册 + CONTRACTS.md 登记，APPROVED 依用户 2026-09-15 直接指令）；
  - RED：`898dec3`（canary 测试文件 + web 门禁镜像守卫扩展 + 分片注册）；
  - 缺陷修复：`dfacfe5`（D1：历史遥测 no-best-yet → null）；
  - CI/文档：`f64ffb2`（web job 门禁 + onboarding 镜像 + 契约修订 1）；
  - 测量日志（初版）：`d6b19f6`；
  - 双审处置：`dd00ffc`（C8 断言窗口加强）+ `6b9963b`（契约修订 2 + onboarding 2.3 事实更正）。
- 被测精确候选 commit：
  - 初版验证矩阵：`f64ffb2`；
  - 双审后复验（venv canary 36P、venv 全量）：`6b9963b`（运行启动时的工作树内容与该 commit
    完全一致，提交仅落盘未改内容）。

## 2. RED 证据（按 §2.2.3 记录）

1. **web 门禁守卫 RED**（`898dec3` 之前的工作树）：
   `python -m pytest -q tests/test_validation_docs.py` →
   `FAILED test_onboarding_web_gates_match_ci - AssertionError`（ci.yml web job 与 onboarding §9.2
   均缺 `npm test` / `npm run lint`）。修改 ci.yml/onboarding 后 GREEN。
2. **canary 闭环 RED**（D1 缺陷）：`test_canary_loop_config_spec_train_backtest_api` 在 train 形式
   入口的产物落盘处失败：
   - 第一次：`CanonicalJSONError: value.history[0].avg_reward: non-finite float -inf is not
     allowed in content identity`；
   - 修一层后：`CanonicalJSONError: value.search_result.best_so_far[0][1]: non-finite float -inf …`。
   根因与修复见契约修订 1（§8）。修复提交 `dfacfe5` 后 GREEN。
3. **探针过程中的场景修正**（非缺陷，canary 自身构造问题）：
   - 合成股票代码 `100000.SZ` 被 `is_valid_a_share_code` 拒绝 → 成员集为空 → 改用主板形代码
     `000001.SZ…000012.SZ`；
   - 成分表 in_date 全同触发宇宙契约"当前快照横跨历史"启发式 → in_date 交错（PIT 区间形）；
   - canary YAML 缺 `start_date` → 入口侧 DataConfig 落回默认 2015-01-01 → spec 身份不匹配
     （RunContext.attach 按设计拒绝，p27 R2）→ YAML 补 `start_date: 2024-01-01`。
   这些是测试夹具向契约对齐，未放宽任何断言、未改动任何生产代码。

## 3. 验证结果（被测 commit `f64ffb2`）

| 命令 | 结果 | 耗时 |
|---|---|---|
| `python -m pytest -q tests/test_p31_b02_entry_canary.py tests/test_validation_docs.py tests/test_ci_sharding.py`（miniconda3，`f64ffb2`） | **36 passed** | 52.1 s |
| 同上，**venv 解释器，`6b9963b`**（双审后复验，含 C8 窗口加强） | **36 passed** | 23.1 s |
| `python -m pytest -q tests/test_p31_b02_entry_canary.py tests/test_train_module_split.py tests/test_artifact_schemas.py tests/test_artifact_writer.py`（修复后首轮，miniconda3） | **66 passed** | 24.6 s |
| 相邻批次（miniconda3，`test_train test_evaluation test_baseline_harness test_searcher_bench test_gp_search test_artifact_versions test_run_store test_webapi test_config test_runspec test_experiment_tracking`） | **301 passed, 2 skipped** | 639.4 s |
| 全量门禁 `python -m pytest -q tests -n auto`（**miniconda3**，初版记录） | 1913 passed, 2 skipped, **2 failed**（`test_lock_files` 两例，归因见下与 §1 更正） | 866.0 s |
| 全量门禁 `D:\minequant\.venv\Scripts\python.exe -m pytest -q tests -n auto`（**venv，`6b9963b`**，双审处置复跑） | **1920 passed, 1 skipped, 0 failed**（1920+1=1921 与可收集总数精确对账） | 706.3 s；原始输出：`logs/p31_b02_full_suite_venv.txt` |
| `python -m pip check`（**venv**，合规 F-6） | "No broken requirements found"，exit 0 | — |
| `python scripts/check_test_shards.py --check` | 104 files across 4 shards == 全集 | <1 s |
| `python -m compileall -j 0 -q ashare_data ashare_model ashare_portfolio ashare_trading scripts webapi` | exit 0 | — |
| `git diff --check`（每个提交前） | exit 0 | — |
| webui `npm ci`（干净锁安装） | 成功（恢复 F11 的过期 node_modules） | — |
| webui `npm test` / `npm run lint` / `npm run build`（CI-equivalent web job，`f64ffb2`） | **5 passed** / 通过 / 构建成功 | 25.2 s / <5 s / 26.0 s |

### 初版全量（miniconda3）的 2 个失败：解释器环境属性，非本包引入

`tests/test_lock_files.py::test_pins_match_installed_environment` 与
`test_freeze_lock_check_passes` 在 miniconda3 解释器上因 `PackageNotFoundError: No package
metadata was found for tushare` 失败（该解释器缺 tushare 元数据；venv 中同测试 14 passed/
1 skipped）。本包 diff 不触碰任何 requirements/lock 文件；失败模式与 A03/A04 测量日志记录的
"2 pre-existing env failures"同源（CI 从 requirements 安装，不受影响）。**venv 复跑（见上表）
全绿，该归因由运行证据直接闭合。**

## 4. canary 特性核对（验收对照）

- **快**：canary 三文件集（含 3 次建库 + 闭环 + 手算 + 扰动 + 全部守卫）miniconda3 52.1 s /
  venv 23.1 s 完成（含 torch 导入），远小于全量 11.8–14.4 min。目标 <120 s 达成。
- **可重复**：闭环断言全部确定性（固定 seed 42、`--device cpu`、合成库内容固定、G8 评估日
  经 `production_gate_asof` 注入为场景末日）；同 commit 重跑结果一致（本日志两次运行均绿）。
- **无网络**：canary 路径无任何网络调用（合成双库本地文件；webapi 仅进程内读 RunStore）。
- **契约级错误在昂贵研究前暴露**：C6 证明身份漂移在 loader 建帧前拒绝且无业务产物；
  D1 证明产物边界在落盘时 fail-closed（并修复了其中的权威矛盾）。
- **手算独立性**：C7 的日期/权重/金额全部从场景构造与文档化费用公式推导
  （两笔 50000 元买单 → 每笔 12.5+0.5+25=38 元、合计 76 元、占比 0.00076；第二期零调仓
  零费用）；未调用被测引擎生成任何预期值。
- **跨平台覆盖的如实记录**：canary 在 Windows/py3.13 本机全绿；CI ubuntu/py3.12 覆盖待
  push 后由 CI 证明（本包未获 push 授权，仅本地交付）。未声明 Windows CI 腿（契约 §2 非目标）。

## 5. 关键行为覆盖映射（§3.3 契约表 → 实际）

| 行为 | 覆盖 |
|---|---|
| 数据身份失效阻断 | canary C6（入口级，新增）；A02 协议层既有 |
| run 根一致 | canary C5（真实入口写→API 读，新增）；A01 C8/C9 既有 |
| 非法生命周期无业务写入 | A03 R12 既有（canary 不复制）；C6 亦断言无新增产物 |
| 未来扰动不影响过去 | canary C8（真实 loader 因子张量逐元素不变，新增）；test_price_semantics/universe 既有维度 |
| 信号 t → 执行 t+1 | test_backtest 对齐测试 + golden parity 既有；C7 日期断言复核 |
| 连续日期轴调仓节奏 | test_rebalance_policy 既有（含周末缺口）；C7 跨周末日期断言复核 |
| 现金、费用、整手、T+1 与账户对账 | portfolio/execution/trading/golden 既有；C7 金额手算对账新增（权重级） |
| 重复同日执行不能重复成交 | test_run_sim 重放拒绝 + resume 水位既有 |

## 6. 残余风险与后续

1. **D1 修复面**：`_json_telemetry_value` 只作用于策略产物载荷；其他类型化产物（protocol/
   backtest/paper state）如未来嵌入遥测出现非有限浮点，会在 canonical 边界 fail-closed
   （当前无此类路径；技术审核逐面复核未发现遗漏的活跃 ±inf 源）；如出现，按同一表示规则
   另立修复。
2. **数据身份拒绝的入口边界形态**：train 入口在 `confirm_input_snapshot` 处抛出
   `DatasetIdMismatch`（未被 p27 I11 的 SystemExit 包装捕获）。p27 未规定该类型在 CLI 边界
   的包装语义，canary 如实钉住现状；如需统一为 SystemExit，属 p27 范畴的后续小包。
3. **双解释器环境**：本机 PATH 首位 miniconda3 缺 tushare 元数据（影响 lock↔env 审计类测试
   在该解释器的可运行性）；onboarding 声明的 venv 运行时全绿。后续会话应以 venv 为准运行
   全量门禁（与 onboarding §9.2 的激活步骤一致）。
4. **`eval_search.py:323` 的隐式不变量**（技术审核 #4）：对 `history[-1]["avg_reward"]` 直接
   `float()` 依赖"有选中则末位有限"的不可达性论证；D1 后历史末位仍可能为 None 的路径不存在，
   但下次触碰该文件时应加 None 守卫或复用 `history_reward_value`。本包不触碰该文件（范围纪律）。
5. **web build 的 chunk>900kB 提示**：vite 信息级提示，先前构建同样存在，非本包引入。

## 6a. 双审裁决与处置（2026-09-15）

- **技术审核**（独立子审核人 1）：**PASS**。4 项发现全部处置：
  - #1（medium，C8 断言窗口 [0:25] vs 扰动边界 dates[30]）→ 已修复：`past = index(dates[30])`，
    扰动边界前全部会话纳入位等同断言（`dd00ffc`）；venv 复验 36 passed（23.1 s）。
  - #2/#3（low，契约 C3 参数与费率措辞）→ 契约修订 2 更正并登记偏差（`6b9963b`）。
  - #4（low/信息，eval_search.py:323）→ 记入 §6.4 后续项，本包不触碰该文件。
- **合规审核**（独立子审核人 2）：**NEEDS_REVISION**，6 项发现全部处置：
  - F-1（medium，onboarding 2.3 "前端没有测试和 lint" 与同流程门禁矛盾）→ 已更正（`6b9963b`）。
  - F-2（medium，tushare 失败误归因 venv + "现场复现"声明不实 + 环境头混合两解释器）→
    §1/§3 已更正（实际解释器 miniconda3；venv 有 tushare 且通过）；并以 venv 全量复跑
    （1920P/1S/0F）使归因由运行证据闭合。
  - F-3（low-medium，全量行不可对账 + 缺原始产物路径）→ venv 复跑 1920+1=1921 精确对账，
    原始输出存档 `logs/p31_b02_full_suite_venv.txt`（路径已记录）。
  - F-4（low，契约范围未列修复文件、修订 1 未点名 train_search_run、修订后置）→ 修订 1 补记
    + 修订 2 范围补列（`6b9963b`）；D1 RED 先于实现的门禁顺序经审核确认未倒置。
  - F-5（low，C3 参数偏差未记录）→ 修订 2 登记。
  - F-6（low，本地矩阵未含 pip check）→ 已在 venv 运行并记录（上表）。

## 7. 交付物

- 分支 `codex/b02-entry-canary-ci`：`1f4f724` → `898dec3` → `dfacfe5` → `f64ffb2` → `d6b19f6`
  → `dd00ffc` → `6b9963b` → 本日志提交。
- 新测试文件：`tests/test_p31_b02_entry_canary.py`（分片 shard-model-1）。
- 缺陷修复：`ashare_model/train_loop.py`、`ashare_model/train_search_run.py`、
  `ashare_model/train_artifacts.py`。
- CI/文档：`.github/workflows/ci.yml`、`docs/PROJECT_ONBOARDING.md`、
  `tests/test_validation_docs.py`、`scripts/check_test_shards.py`。
- 双审：技术 PASS（4 发现全处置）、合规 NEEDS_REVISION（6 发现全处置，见 §6a）。
- 未 push、未建 PR、未合并（未获授权）。
