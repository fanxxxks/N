# P30 A05 测量日志（分层 doctor 资格报告）

- 被测实现：分支 `codex/a05-doctor-eligibility` @ `33f70ac8640d…`
  （`33f70ac`；链：`a18cac8` A03 集成 → `160eea1` 契约 → `5c5b2b3` RED →
  `7a6d924` 实现 → `006c8e4` 修复 1 → `df08c94` 双审修复 → `33f70ac` 元组
  bug 修复。树状态：跟踪文件干净，仅既有未跟踪 `reports/` 用户工作与本任务
  未跟踪 docs 证据，未触碰前者）。
- 契约：`docs/p30_a05_doctor_eligibility_draft.md`（APPROVED——批准依据 =
  用户 2026-09-14 会话直接指令「完成A05」；转授记录 §3 不覆盖 p30，见契约
  批准行与双审处置记录）。
- 环境：Windows（PowerShell 7 / pwsh），Python 3.13.12（`D:\minequant\.venv`），
  依赖以 doctor 报告 dependencies 节为准（torch 2.11.0+cu128、deap 1.4.4、
  pandas 3.0.5、duckdb 1.5.5 等，全部非 None）。
- 时点：最终验收 doctor 运行 2026-09-14（UTC 时间戳见 §4 绑定块）。

## 1. RED（先于实现，commit `5c5b2b3`）

```
python -m pytest -q tests/test_p30_a05_doctor_eligibility.py -p no:cacheprovider
→ ERROR collecting tests/test_p30_a05_doctor_eligibility.py
  ImportError: cannot import name 'readiness' from 'ashare_model'
  （实现缺失的预期失败；1 error，未能收集）
```

同 commit 登记 `scripts/check_test_shards.py`（shard-model-1），
`check_test_shards.py --check` 通过：
`shard union == full set: 102 files across 4 shards cover all 102 test files`。

## 2. GREEN 与双审修复轮（全部在精确 commit 上实跑）

| commit / 阶段 | 命令 | 结果 |
|---|---|---|
| `7a6d924` | p30 文件聚焦 | 19 passed（R1–R12；R4 拆 legacy-N/A 与非 legacy-fail，R5 拆观察存在/不存在） |
| `006c8e4` | 相邻契约五文件 | 154 passed |
| `006c8e4` | 全量串行 `python -m pytest -q tests` | **1872 passed, 1 skipped，2049.80s（0:34:09）**（双审修复前的全量记录，保留作链路证据） |
| `df08c94` | p30 + research_doctor + webapi + run_store + p27 | 145 passed（含新增 9 项回归：锁保护/象限/乱码守卫/逐 G 明细/解除条件/404 provenance/R3 共存） |
| `33f70ac` | p30 + research_doctor + webapi | 67 passed（含 release_condition 字符串守卫） |
| `33f70ac` | p30 文件 | **28 passed** |
| `33f70ac` | `python scripts/check_test_shards.py --check` | 通过（union == full set） |
| `33f70ac` | `python -m compileall -j 0 -q ashare_data ashare_model ashare_portfolio ashare_trading scripts webapi` | exit 0 |
| `33f70ac` | `git diff --check` | exit 0 |
| `33f70ac` | 全量串行 `python -m pytest -q tests` | 见 §4.1（最终数字，后置补记） |

R3 说明（合规审 C2）：R3 拆为两例——`gates=None`（config-error 真实路径）
+ `test_r3_ok_gates_cannot_whitewash_a_failing_runtime`（ok gates 共存 ⇒
runtime 仍 fail、stale pass 留在 data 维度）。

## 3. 既有测试适配（AGENTS §10.1 白名单第 2 类：需求变更）

- `tests/test_run_store.py::test_p8_15_research_doctor_lifecycle_section`：
  旧断言 `paper_account_writable in check_names` 固化了
  `assert_paper_account_writable("current")` 恒真检查（该函数仅比较字面量，
  不读任何账户分类/绑定——`reports/alphagpt-ab/00_baseline_audit.md` §223
  实证；p30 §0/§3.4/R9 据此将其移除）。替换断言
  `paper_account_writable not in check_names` + `lifecycle_edges_activated`
  保留，覆盖强度不减。依据：契约 §3.4、§4 R9、批准行。
- `tests/test_p27_a03_lifecycle_runcontext.py` 的"doctor 源码不含
  `assert_paper_account_writable("current")` 字面量"守卫继续通过（零改动）。

## 4. 验收证据：真实环境 doctor 分层报告（最终 HEAD `33f70ac`）

命令：

```
python -m ashare_model.research_doctor --json --output D:\minequant\reports\AlphaGPT-p30-a05-evidence\doctor_readiness.json
```

原始产物：`D:\minequant\reports\AlphaGPT-p30-a05-evidence\doctor_readiness.json`。
退出码 1 = 既有工程冲突规则（G8 error → NOT HEALTHY），**退出码语义未变**
（p30 §3.4）。

绑定块（bindings）：

| 键 | 值 |
|---|---|
| readiness_version / action_mode | "1" / "audit" |
| evaluated_at / evaluation_day | 2026-09-14T15:36:xx+00:00（JSON 自述）/ 20260914 |
| config_path / config_sha256 | `D:\minequant\AlphaGPT\config\ashare_config.yaml` / `9273ad43…94773e` |
| git_commit | `33f70ac8640d…` |
| store_root / source | `D:\minequant\AlphaGPT\data` / config（A01 resolver，audit 模式） |
| snapshot | **composite_id `b369c49b…874cf`，publication 1**，primary `7c6a6f62…fbf17c`（manifest v2），v2 `edf27650…252aa`（双库全量未缓存重哈希确认） |
| bound_run | null（未绑定 run） |

维度结论（9 pass / 3 fail / 0 unknown / 5 not_applicable；每个非 pass 检查
均带解除条件，production_gates 检查内嵌 8 项逐 G 明细）：

| 维度 | 状态 | 明细 |
|---|---|---|
| runtime | **pass** | 配置加载、双输入库存在、依赖齐全、store 根可解析 |
| snapshot_integrity | **pass** | A02 复合身份确认（publication 1；全量重哈希通过） |
| data_eligibility | **fail** | G1–G8 = 7/8：G8 新鲜度 `daily 20260907 vs reference 20260911, lag=4 > N=3`（数据截至 2026-09-07，今日 2026-09-14）——真实阻断 |
| research_ready | **fail** | carried gates 失败（新的正式研究运行当前会被 G8 拒绝） |
| promotion_ready | **not_applicable** | 仅存 legacy 产物（strategy/protocol 均已盖 legacy 章）——可审计、永不晋级；非 legacy 冲突为空 |
| paper_ready | **fail** | `data\sim_portfolio_state.json` 存在但 RunStore 无任何 PAPER_OBSERVING 绑定账户——「无账户却声称可续跑」的假阳性已关闭，解除条件 = 经审计迁移或退役 |

对照方案 A05 验收原文：「本次环境应准确报告『运行时可用，但指定研究配置/
身份有阻断；旧产物可审计』」——runtime pass + data/research fail（具体 G8
滞后数字）+ promotion N/A（legacy 可审计）+ paper fail（无绑定不可续跑），
逐项成立；无统一 HEALTHY 替代（doctor 顶部的 `healthy` 仅反映工程冲突规则，
退出码语义未变）。

### 4.1 最终全量回归（精确 HEAD `33f70ac`，串行）

```
python -m pytest -q tests
→ 1881 passed, 1 skipped in 1904.28s (0:31:44)
```

（较 `006c8e4` 的 1872 净增 9 项 = 双审修复批次新增的回归测试；唯一 skip
为既有平台能力基线 skip，与本包无关；无新增 warning。）

## 5. 未运行项与边界

- 未运行 `--spec-id/--run-id` 绑定真实 run 的端到端：主仓 RunStore 当前无
  真实预注册 run（data/runs 在 2026-09-09 审计时不存在，本任务不创建任何
  run——创建运行身份不在 A05 授权内）；该路径由 R8/tmp RunStore 真实
  lifecycle 链测试与 R11 的 CLI 配对校验覆盖。
- webapi `/api/readiness` 以 `service.get_readiness` 直调覆盖（R12 两例 +
  finding 6 断言）；未做浏览器端 UI 验证（A05 范围不含前端改造，C01 才涉及）。
- 全量套件未做并行-串行 parity 对账，按 AGENTS §10.2 回退串行全量。
- 本包为工程行为变更：零研究语义变化（版本影响表见契约 §5，全部
  `*_VERSION` 不变；双审 T 项核实 `_VERSION_IMPORTS` 未触）；不产生、
  不修改任何研究结论。

## 6. 裁决

软件正确性：p30 R1–R12 + 双审新增回归全绿；相邻契约绿；`006c8e4` 全量
1872 passed / 1 skipped（2049.80s）；`33f70ac` 最终全量见 §4.1（完成后如实
补记）。工程测量：doctor 端到端 ≈2.5 分钟（含双库全量身份确认）；readiness
为只读组合层，无新增写入路径；双审 HIGH-1（乱码）与 HIGH-2（锁误删）均已
修复并有守卫测试。研究证据：无（本包不声明任何 alpha/晋级结论）。
生产/实盘就绪度：不适用——paper_ready 的如实 fail 正是本包要消除的
假阳性的反义词面。
