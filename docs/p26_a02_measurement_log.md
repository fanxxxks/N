# P26 A02 测量日志（docs/p26_a02_measurement_log.md，p26 §7 指定）

格式（AGENTS §11.2）：每条记录命令、commit/tree 状态、环境、关键数字、
原始产物路径与裁决。本文只写「实际发生什么」；计划在契约，不在此。
本日志由 A02 完成会话（2026-09-10/11，分支 `codex/a02-snapshot-identity`）
建立；证据-only 追加条目按时间顺序续写，不改写历史条目。

- 机器：6C/12T、16 GB RAM、Windows（D: 盘仓库）；Python 3.x +
  duckdb 1.5.5（run identity 行实测）；串行执行。
- 真实库：主库 `data/ashare.duckdb` 1,015,296,000 B；v2
  `data/ashare_v2.duckdb` 7,957,393,408 B。所有真实库操作均为
  read_only 连接（`write_free` 逐条证明）。

## 1. 两库差异报告（只读，p26 §5.4；2026-09-10/11 实测）

- 主库最新已发布 manifest：`b7b4dd4b03fe…`（v1，2026-09-01T21:29:56Z）。
  逐表 vs 当前内容：`daily_bar` 4,896,877 → 4,919,070（**+22,193**）、
  `sw_industry_index` 141,378 → 141,874（**+496**）、`fundamental_pit`
  行数与日期范围不变但 **merkle 漂移**（值级变更；R2 场景的现场证据，
  `fast` 不可见）；其余 5 表一致。
- v2 最新已发布 manifest：`af825c2e4a20…`（2026-09-08T02:34:29Z）。
  **v2 全库 `full` 内容校验未执行**（not_run；B01/B04 范围），仅行数级
  与内容身份级探针。
- 原始产物：`reports/AlphaGPT-audit-2026-09-09/boundary_probes.json`、
  `reports/alphagpt-ab/T47_manifest_publication.md` §2、
  `D:\minequant\_tmp_a02\drift_investigation.json`。

## 2. 身份漂移取证（t46 前置条件 (a) 的闭合证据；2026-09-11）

§4quinquies 登记的「标的漂移」（`e5ca3513…` → `7c6a6f62…`）经取证
**不是数据写入**，是 `dataset_id` 载荷内嵌 `MANIFEST_VERSION` 的代码
状态差（772ce14 于 2026-09-10 20:10 把它 1→2）：

| 库状态 | MANIFEST_VERSION="1" 哈希 | "2" 哈希 |
|---|---|---|
| 修复前内容（备份 `_db_backup_pre_t46`） | `13b0cd70e416…`（审计 §0.1 ✓） | `42fbe369…` |
| 修复后内容（当前） | `e5ca351334f5…`（T46 §7 ✓） | `7c6a6f629ad2…`（T47 ✓） |
| v2 当前内容（未变） | `af825c2e4a20…`（== 已发布 ✓） | `edf27650…` |

- 备份 vs 当前的全表 `EXCEPT ALL` 差异（8 表全覆盖，只读）：
  **仅 `daily_bar` 118 行双向差异**（= 授权的 ISO 规范化），其余 7 表
  **零差异**（含 `fundamental_pit`）。
- 18:59:18 的文件写入 = captain 修复后验证的无缓存构建对
  `dataset_manifest_cache` 的刷新（cache `computed_at` 全体
  2026-09-10T10:58:56–10:59:10Z 实测佐证）；无任何数据表写入。
- 结论：**当前内容 = 修复前备份 + 授权的 118 行 ISO 修复**；其当前
  协议身份 `7c6a6f629ad2…`。§4quater(2) 授权标的（修复后真实内容）
  未被任何未审写入改变。
- 原始产物：`D:\minequant\_tmp_a02\{drift_investigation,cache_contamination,
  e5ca_reconstruction,version_theory_proof}.py/.json`、
  `v2_version_check.py`（四向身份复现脚本）。

## 3. 实现验证矩阵（分支 `codex/a02-snapshot-identity`）

逐提交（db18812 → 931eddf → 79da667 → dd5caa1 → 8a64d0f → B1/B2 修复）：
- RED 取证：detached worktree @父提交实跑（composite API ImportError ×4；
  sync 无发布 ×1；loader 复合测试 ×5）；技术审查探针 P1/P2/P3
  （`_tmp_a02/review_probe.py`）。
- GREEN（修复 B1/B2 后，最终状态）：`tests/test_manifest.py` 38 passed；
  `test_data_loader.py + test_load_cache.py + test_industry_pit.py` 48
  passed；`test_universe.py + test_core.py + test_run_sim.py` 65 passed；
  `test_searcher_bench.py + test_train.py` 62 passed, 2 skipped
  （既有平台基线 skip）；`test_evaluation.py` 51 passed（547.64s）；
  `test_sync.py + test_runspec.py + test_industry_pit.py` 66 passed。
- 门禁：`compileall` exit 0；`git diff --check` exit 0；
  `check_test_shards.py --check` 99 files / 4 shards 并集==全集 exit 0
  （本包未新增测试文件）。
- 双审：技术半（t14 等效）NEEDS_REVISION → 修复 dd5caa1 → 复审 PASS；
  合规半（t13 等效）NEEDS_REVISION（B1 红树两例 + B2 声明矛盾）→
  本批修复（B1 两测试、B2 勘误+声明修正）→ 复审 PASS（两项非阻断
  文档跟进：docstring 落地 `6f9728b`；§4 数字见下）。
- 全量串行（最终候选 `b5e4242`，2026-09-11）：**1787 passed / 6
  skipped / 2 failed**（1986.16s）。两处失败均为 `tests/test_lock_files`
  的 **commit 无关环境漂移**：`requirements.in` 自 2026-09-04（`172ebcb`）
  起 pin 了 `tushare`，但本机所有解释器（miniconda3 base/EPSA/d2l-zh/
  skill-env、Python314、python-build）均未安装该包，且 site-packages
  自 2026-09-04 后无任何安装/卸载痕迹；同一失败在 `9aa78ec`（交接
  文档的「绿」头）detached worktree **同样复现**（2 failed, 12 passed,
  1 skipped）→ 与 A02 变更无因果路径（A02 提交未触碰任何依赖/spec/pin
  文件，合规审查已核范围）。6 skipped 含既有基线 skip
  （`test_lock_files.py:302`）与平台级 skip；A02 提交未新增任何
  skip（合规审查收集数审计）。task_executor 既有 flake 本次未触发。

## 4. 资源上限（p26 §8；裁决 4 的「报告实测」路径）

- §8.2 热路径目标 ≤60s：**实测超限**。2026-09-11 真实库只读实测：主库
  无缓存构建 15.9s，v2 无缓存构建 105.8s，**合计 121.7s**；复合确认
  全程（两库构建+校验）wall 154.9s。按裁决 4 处置：不自行调高上限、
  不降级为 fast 冒充；数字如上报告。设计建议（移交 B04 + captain
  裁决）：§8.2 设想的「发布清单 + 冻结分区哈希」热路径——确认时按
  (count,max) 快速比对 + 发布期冻结哈希 + 可信变更记录（v2 采集
  ledger），将热路径成本降到聚合查询量级。
- §8.3 发布操作目标：发布相位 wall 128.6s、峰值 RSS 4314 MB
  （**未超 60min/8GB**，在目标内）。
- 确认路径在 `load_data` 中的位置（昂贵帧构建之后）已记录（A03 排序
  范围；合规审查 m3）。
- 测试环境墙钟：`test_manifest.py` ≤120s 目标实测达标（38 passed 约
  52–57s）。

## 5. 已知声明/文档项（合规审查 M2/m1/m2/m3 的处置记录）

- `scripts/archive_run.py` 的 `dataset_identity()` 仍为单库查询侧
  （best-effort 标注）：**推迟**（不在 A02 inScope 文件清单内；I11 的
  标注升级与 A03 归属一起处理）。
- `confirm_dataset_identity` 的 `verify` 参数已校验但恒为无缓存全量
  重算：docstring 已注明（无廉价快路；能力声明归
  `verification_capability`/`verify_dataset`）——补注落地于 `6f9728b`
  （合规复审跟进 1）。
- `docs/PROJECT_ONBOARDING.md` 的 dataset_id 当前值引用：**推迟同步**
  （不在 A02 inScope 文件清单内；t46 发布后该行本就要再变，归入
  C02/H-6 文档批）。
- `reports/` 整目录未纳版本控制：**待用户裁决**（H-7，不自行 git add）。

## 6. t46 单次有界发布（用户授权 §4quater(2)；执行完成，2026-09-11）

- 前置 (a) 漂移源：见 §2（已闭合，无未审写入）。前置 (b) H-2：已合入
  `main`（merge `c868134`，§4bis 授权的「先合并再推进」路径；发布从
  合并后的 main 工作树执行，满足 §4quater(2) 的「合入之后」条件）。
- 冻结检查（audit 相位，只读，`write_free=true`）：主库当前内容身份 =
  `7c6a6f629ad22e54…`（= 修复后真实内容），v2 = `edf276501926…`
  （v2 内容与 09-08 发布逐字节一致，仅协议身份不同）。
- **发布（publish 相位，对主库的唯一一次写入；v2 全程零写入
  `v2_write_free_all_phases=true`）**：
  - `publication_id = 1`（首个复合代际），成员 `primary` +
    `tushare_v2` 均就位；
  - `composite_id =
    b369c49b20f18a0adc4b5f57b4050a1bd735fdd788de4d9076f13be9e86874cf`；
  - primary 成员 = `7c6a6f629ad2…`（10,941,392 行 / 8 表）；
    v2 成员 = `edf276501926…`（57,156,485 行 / 19 表）；
  - 主库写集合（= §4quater(2) 授权的「新 manifest 代际」的精确范围）：
    `dataset_manifest` 追加 1 行（`7c6a6f62…`，manifest v2）+
    `snapshot_publication` 追加 2 行（group 1 的两个成员）+
    `dataset_manifest_cache` 分区哈希刷新（无缓存构建的固有副作用，
    语义中性）+ 事务性建表（首次 `snapshot_publication` DDL）。历史
    4 行 v1 manifest **原样保留**（append-only，I9）。
- **发布后复核（verify 相位）**：
  - `confirm_composite_snapshot` **通过**，返回同一 `composite_id`；
  - `latest_manifest` = `7c6a6f629ad2…`（不再指向陈旧的 b7b4dd4b）；
  - `daily_bar` 4,919,070 行、20150105–20260907、ISO 行 0；
  - `python scripts/check_production_gates.py` exit 0，**ALL GATES
    PASS**（G1–G8 全过，G8 lag=3 ≤ N=3）；
  - `python -m ashare_model.research_doctor` exit 0，status HEALTHY，
    `data: dataset_id=7c6a6f629ad2… manifest v2`（F02 的 doctor 症状
    关闭）。
- 资源：audit wall 122.7s；publish wall 128.6s / 峰值 RSS 4314 MB；
  verify wall 154.9s / 峰值 RSS 4352 MB。
- 回滚方式（未使用）：行级（精确）——删除 `snapshot_publication`
  group 1 的两行与 `dataset_manifest` 的 `7c6a6f62…` 行即回到发布前
  状态（缓存刷新语义中性）；文件级——以
  `D:\minequant\_db_backup_pre_t46\ashare.duckdb.20260910_185715.bak`
  为基线（该备份为修复前状态）+ 重放已授权的 ISO 修复。发布**可逆**。
- **授权消费状态**：§4quater(2) 的单次发布授权**已消费**（本条即
  消费记录）；如内容再变，需重新请批。
- 原始产物：`D:\minequant\_tmp_a02\t46_{audit,publish,verify}.json`、
  `t46_phase_metrics.txt`、`t46_publish.py`（执行脚本，仓库外）。
