# P36 历史研究与在线观察资格分离 · 测量日志

- 性质：evidence-only（AGENTS §11.1/§11.4）。只记录实际发生什么；预注册内容见
  `docs/p36_purpose_eligibility_separation_draft.md`（APPROVED，用户 2026-09-17
  直接指令工作包 2）。
- 被测实现：commit `68f7493`（分支 `codex/b04-load-cache-performance`；契约
  提交 `acafe54`；父基线 `588f378` = p35 证据链头）。
- 环境：Windows 10 22621 x64，Git Bash；`C:\ProgramData\miniconda3\python.exe`
  Python 3.13.12；本机 CPU；无 GPU 参与。工作树在被测 commit 上干净
  （仅无关未跟踪 `.mimosa/`）。

## 1. RED（实现前，2026-09-17 19:52 本地时间）

命令：`python -m pytest -q tests/test_p36_purpose_eligibility_separation.py`
（实现前工作树 = 基线 588f378 + 本契约文档）。

结果：**13 failed, 2 passed（17.38s）**。全部 13 个失败均可归因于缺失的 p36
行为（抽查确认失败原因正确，非夹具错误）：

- `test_r1_*` / `test_r7_*`：`build_runspec() got an unexpected keyword
  argument 'purpose'`、payload 无 `purpose` 键（KeyError）；
- `test_r2_*`：`ashare_data.gates` 无 `_shanghai_today` 属性（历史基准的
  免墙钟证明目标尚不存在）；
- `test_r3_*` / `test_r9_*`：`run() got an unexpected keyword argument
  'purpose'` / 快照基准缺失；
- `test_r5_*`：spec 构造 TypeError（D7 用途推导不存在）；
- `test_r6_*`：入口源码钉住失败（`campaign_executor.py` 1 处调用 0 处
  purpose 等）。

2 个通过项为**设计内的保持性钉住**（钉住本包不得破坏的现状）：
`test_r4_online_stale_still_rejects`（p16 在线语义原样）、
`test_r4b_run_sim_pins_the_online_basis`（run_sim 现状即在线基准）。

## 2. GREEN 与聚焦验证（被测实现后）

| 命令 | 结果 |
|---|---|
| `python -m pytest -q tests/test_p36_purpose_eligibility_separation.py` | **15 passed**（多轮复跑一致；最后一轮 27.98s 在最终测试文件状态） |
| `python -m pytest -q tests/test_gates.py tests/test_runspec.py tests/test_run_store.py tests/test_artifact_writer.py tests/test_task_executor.py` | 136 passed（110.86s） |
| `python -m pytest -q tests/test_p34_r01_baseline.py` | 15 passed（174.87s） |
| `python -m pytest -q tests/test_p31_b02_entry_canary.py tests/test_p34_r01_baseline.py tests/test_p36_purpose_eligibility_separation.py` | 35 passed（144.21s，行数预算修复后的复跑） |
| `python -m pytest -q tests/test_p27_a03_lifecycle_runcontext.py tests/test_b03_campaign_executor.py` | 50 passed（315.94s） |
| `python -m pytest -q tests/test_train.py` | 45 passed, 2 skipped（110.64s） |
| `python -m pytest -q tests/test_train_module_split.py` | 5 passed（3.81s） |
| `python -m compileall -j 0 -q ashare_data ashare_model ashare_portfolio ashare_trading scripts webapi` | 通过（无输出） |
| `git diff --check` | 通过（仅 autocrlf 换行提示，无空白错误） |
| `python scripts/check_test_shards.py` | `shard union == full set: 109 files across 4 shards` |

## 3. 全量回归（`python -m pytest -q tests -n auto`）

- **首轮**（2026-09-17 20:29 启动，实现在候选前一状态）：**3 failed, 1979
  passed, 6 skipped, 3 xfailed**，墙钟 47787.69s（≈13.3h——夜间机器竞争/休眠
  所致，非常态耗时）。3 个失败：
  1. `test_train_module_split.py::test_train_facade_line_budget`——
     **本包引入的回归**：`train_artifacts.py:_persist_strategy_artifact`
     103 行 > 100 预算（p36 加 4 行越界）。已修：非正式 spec 构造提取为
     `_nonformal_strategy_spec`；随后暴露 `train_entry.py:main` 104 行 >
     100，同批修复：A03 模式声明双拒绝块提取为
     `_refuse_ambiguous_mode_declaration`。两处提取零语义变化，IP-07b 预算
     测试未放宽（见 §5 白名单）。
  2/3. `test_lock_files.py::test_pins_match_installed_environment` /
     `test_freeze_lock_check_passes`——**既有环境漂移，与本包无关**：
     `importlib.metadata.PackageNotFoundError: No package metadata was found
     for tushare`；取证：`python -c "import tushare"` → `ModuleNotFoundError`
     （环境中 tushare 整体缺失），`requirements.txt:14` pin `tushare==1.4.29`
     未被本包触碰（`git status` 零依赖文件改动）。p35 测量日志（2026-09-17
     晚）记录全量仅 1 失败——环境在两日之间丢失 tushare，非本 diff 所致。
     未代修（安装/改环境超出本包授权），如实移交用户。
- **复轮**（行数预算修复后的候选状态，2026-09-18）：**2 failed, 1980 passed,
  6 skipped, 3 xfailed，935.06s**。唯一失败 = 上述两个 lock 环境测试
  （同一 tushare 元数据缺失）；行数预算回归已消除。

**候选 commit `68f7493` 与全量复轮被测树的确切差异**：新测试文件中移除一个
未使用的 `DataConfig` import（lint 级、惰性变更，发生在复轮收集之后）；该
文件在最终状态单独复跑 15 passed。除此之外逐字节一致。

## 4. 版本影响（实测确认）

- `RUNSPEC_SCHEMA_VERSION` 2→3：`test_r1`/`test_r7` 实测 v2 形状 payload
  （无 purpose）过 `RunSpec.model_validate` 即 ValidationError；同参数仅
  purpose 不同 ⇒ `spec_id` 不同。
- 其余 `*_VERSION`（ARTIFACT_SCHEMA=3、LEGACY=(1,2)、LIFECYCLE_CONTRACT=2、
  DATA_QUALITY_REPORT=2、p19/p17 语义、组合/执行/晋级）零变化
  （`git grep -nE "^[A-Z][A-Z0-9_]*_VERSION" -- "*.py"` 全量核对后未改动）。

## 5. 既有测试修改白名单（AGENTS §10.1 举证）

| 文件:位置 | 白名单 | 依据与说明 |
|---|---|---|
| `tests/test_runspec.py:93,349`（原 93/345） | 第 2 类 | 契约 §7：schema 2→3；断言 `== 3` 强度不变，注释补 P36 来源 |
| `tests/conftest.py::make_test_spec` | 第 3 类 | 研究夹具默认 `purpose="historical_research"`（契约 §11.2）；断言集合不变 |
| `tests/conftest.py::production_gate_asof`、`tests/test_p34_r01_baseline.py::gate_run_asof` | 第 3 类 | 签名机械转发新 gate kwargs；**历史基准调用不注入时代日**（注入即基准混用，真实 gate 会拒）——语义由 R2 免墙钟测试另行钉住；原 era 注入语义对在线基准逐字保留 |
| `tests/test_runspec.py BASE_KWARGS`、`test_run_store.py::_spec`、`test_task_executor.py::_spec`、`test_artifact_writer.py::_spec`（2 处）、`test_b03_campaign_executor.py:851`、`test_p31_b02_entry_canary.py::_freeze_spec`、`test_p34_r01_baseline.py::_mini_spec` | 第 3 类 | 补必填 `purpose="historical_research"`，零断言变化 |
| `ashare_model/train_artifacts.py`、`ashare_model/train_entry.py`（行预算提取） | 工程重构（非测试） | IP-07b 100 行预算回归的本包修复；`test_train_module_split` 断言未放宽，提取函数单一职责、零语义变化 |
| 新测试内 R3-1 断言 `lag=8`→`lag=9` | —（本包新测试的自纠） | 实现交叉检查发现起草笔误：历史基准参考会话=cutoff 会话本身（评估日=cutoff+1），9 个缺失会话全部计入；契约 §2.3 冻结的是「lag > N ⇒ 拒绝」，非特定数值 |

## 6. 裁决与遗留

- 四条验收语义全部由 R1–R9 机器钉住并通过：用途进身份（R1/R7）、同快照同
  资格且历史基准不读墙钟（R2/R5）、窗口末端缺失仍拒（R3-1，另 G1–G7/A02
  确认/漂移检查/D7 阈值原样未动）、在线过期仍拒（R4/R4b/R9）。
- 遗留（非本包）：① 两个 lock 环境测试红——环境缺 tushare，需环境侧处置；
  ② 重复全库确认的性能收敛——用户明示后置的独立工作包；
  ③ Mimosa commit 钩子于两次提交时报两个既有 low 发现
  （`ashare_model/tpe_search.py:142`、`ashare_model/train_windows.py:206`
  「不安全的随机数」），均不在本包改动文件内，未处置，移交用户；
  ④ run_sim/portfolio 中既有的 "schema-v2" 错误文案（指策略产物 lineage，
  非 RunSpec schema）为 p36 之前的既有措辞，未顺手改（§9.1）。
- 本地提交未 push、未建 PR、未合并（未获相应授权）。
