# P1-5 数据时效测量日志：daily_bar 补同步（evidence-only）

> 团队任务 t12（成员 impl-c-data，独占 DB 写入窗口）。
> 性质：§11.1/§11.4 后置 evidence-only 记录，只写"实际发生什么"；被测树为
> `fde9f8b72f7b274024c15045cc4be538a3b8c897`（`codex/agents-md-slim` 与 `main`
> 同指同树，阶段 0 验收 t4 已确认全量测试证据适用于该树）。本日志自身未提交，
> 是否按 §11.4 以独立 evidence-only commit 落库由 captain/集成决策。
> 运行类型：engineering（数据运维同步）；不构成任何研究/收益结论。

## 1. 环境

| 项 | 值 |
| --- | --- |
| 日期 | 2026-09-02（Asia/Shanghai），执行区间 01:19–02:36 |
| 主机 | Windows，本地单用户环境 |
| Python | 3.13.12，`D:\minequant\.venv\Scripts\python.exe` |
| 工作目录 | `D:\minequant\AlphaGPT`（HEAD = fde9f8b，branch `codex/agents-md-slim`） |
| 数据库 | `D:\minequant\AlphaGPT\data\ashare.duckdb`（gitignored） |
| 配置 provenance | `config/ashare_config.yaml` @ fde9f8b；无 `config/runtime_overrides.yaml`、无 `config/.env`、无 `ASHARE_*` 环境变量（执行前逐一核实） |
| 依赖 | akshare 1.18.91 / duckdb 1.5.5（requirements.txt @ fde9f8b pin） |
| 工作树 | 跟踪文件零改动；用户 3 个未跟踪项（`.agent-teams/`、`docs/p5_implementation_plan.md`、`papers/`）全程未触碰 |

## 2. 同步前只读审计（§5.6 第一步，01:19–01:24）

命令：临时只读脚本（read_only=True 连接）+ `python -m ashare_model.research_doctor --json`。

| 项 | 实测值 |
| --- | --- |
| daily_bar | 4,874,595 行，2015-01-05 … **20260821** |
| trade_calendar | 8,797 行（19901219…20261231）；≤ 今日(20260902) 的最后一个开市日 = 20260902 |
| 落后 | 日历口径 **8 个开市日**：20260824/25/26/27/28、20260831、20260901、20260902；其中 7 个为已完成交易日（0824–0901），20260902 在执行时点（01:19）尚未开盘 |
| dataset_id（现值） | **a839ecf2284b354a5ab6ed3228d13fc5d7f3d93a2fadba0b08d8c909edf194fd**（created_at 2026-08-31T15:23:51+00:00）；manifest 1 行 + cache 129 行；total 10,886,056 行 / 8 表 |
| 门禁基线 | research_doctor 01:24：healthy=true，**formal 模式 G1–G7 全 PASS**（G6 min eligible 2015:473，2026:795；G7 2,426 区间 0 零 bar） |

**与任务预期的偏差披露**：任务背景预期 dataset_id 现值为 `b927074a`，实测已是
`a839ecf2`。manifest 持久化历史显示 `b927074a`（2026-08-27T10:31:21+00:00）→
`a839ecf2`（2026-08-31T15:23:51+00:00）的更替发生在本任务之前；行数差
11,003,350 − 10,886,056 = 117,294 = 117,287（P2-01 fundamental_pit purge）+ 7
（B 股脏行清除），与 PROJECT_ONBOARDING §0.1 记载的 P2-01 治理一致，即 a839ecf2
是 purge 后的正确现值。本任务实际发生的 dataset_id 更替为 **a839ecf2 → e15b4fc4**。

## 3. 执行（§5.6 第二步，独占窗口）

- 窗口开启前核实：无 python/uvicorn/streamlit 进程；无 sim 运行（根目录 `STOP_SIGNAL` 在位、无 `data/sim_run.json`）；D 盘空闲 76.2 GB；Eastmoney 可达（HTTP 200）。已向 captain 报备窗口开启（01:25）。
- 命令（仅 daily_bar 范围；fundamental 回填属 t17，明确不在本窗口）：

```powershell
& D:\minequant\.venv\Scripts\python.exe -m ashare_data.sync --no-fundamentals --no-capital-flow
```

- 起止：2026-09-02 01:24:50 → 02:15:24（墙钟 **50m34s**；峰值内存未采样，进程结束后无法回采，如实记为未运行项）。
- 数据源行为：Eastmoney 日线端点连续失败后，客户端按既有兜底逻辑自动切换 **Sina** 完成其余抓取；全程 **287 条 AkShare 重试 WARNING，0 个硬失败**（`Daily bar fetch failed` 计数 = 0）。
- 同步结果（sync_all 返回值，日志原文）：

```text
{'calendar_days': 8797, 'stocks': 5205, 'universe': 2118,
 'constituent_snapshot_symbols': 716, 'pit_constituent_rows_written': 0,
 'daily_rows': 4896759, 'failures': [], 'purged_rows': 0, 'purged_parquet': 0,
 'dataset_id': 'e15b4fc47c4dd9322294f3f7d65d4b17c8fb3a40a878db2c2c95342d84e07547',
 'fundamental_quarters': 0, 'fundamental_rows': 0, 'fundamental_supplements': 0,
 'fundamental_failures': 0, 'margin_rows': 0, 'margin_dates': 0, 'industries': 0,
 'industry_rows': 0, 'capital_failures': 0}
```

`universe=2118` = 快照校验基础集(716) ∪ PIT 历史成员 ∪ 本地缓存码（§12 生存感知并集，即 CSI300+CSI500 生产域），与门禁管辖域一致；`purged_rows=0`、`failures=[]`。`fundamental_*`/`margin_*`/`industries_*` 全 0 证明 --no-* 标志生效，范围未越界。

## 4. 复核（§5.6 第三步，02:31–02:36，全部只读或幂等）

### 4.1 行数 / 日期范围 / 覆盖

| 项 | 同步前 | 同步后 |
| --- | ---: | ---: |
| daily_bar 行数 | 4,874,595 | **4,896,877**（净 +22,282） |
| daily_bar 日期范围 | 2015-01-05…20260821 | 2015-01-05…**20260901** |
| distinct ts_code | （未采样） | 2,118 |
| 新增 7 个交易日的逐日覆盖 | — | 20260824:2024 / 25:2023 / 26:2024 / 27:2024 / 28:2025 / 31:2023 / 0901:2023 行（每日 2,023–2,025 码，缺口为停牌/未上市码，属正常） |
| 剩余缺口 | 8 个日历开市日 | **仅 20260902**（执行时点尚未开盘的当日节；至最近已完成开市日 20260901 覆盖完整） |

净增量 22,282 = 新增交易日行 14,164 + 既有日期上的数据方重述/补齐行（如停牌复牌、前复权重算），upsert 语义下属正常内容更替，manifest 内容哈希已如实反映。

### 4.2 manifest 重建与 dataset_id 对齐

- sync 尾部已自动 build+save：`Dataset manifest recorded: e15b4fc4…e07547 (10908341 rows across 8 tables)`（02:15:24）。
- 显式重建（§5.6 复核命令）：`python -m ashare_data.manifest` → 输出同一 `dataset_id e15b4fc47c4dd9322294f3f7d65d4b17c8fb3a40a878db2c2c95342d84e07547`，exit 0 —— 内容寻址幂等成立。
- `dataset_manifest` 持久化历史（审计线索）：`b927074a`(08-27) → `a839ecf2`(08-31) → **`e15b4fc4`(09-02)**。
- manifest 逐表：daily_bar 4,896,877（…20260901）；trade_calendar 8,797；**fundamental_pit 205,589（max report 20260630，未触碰，留待 t17）**；margin_balance 5,642,388（…20260814，未触碰）；sw_industry_index 141,378 / sw_industry_member 5,196（未触碰）；stocks 5,542（5,539→5,542，upsert 新上市 3 行）；constituents 2,574（不变）。
- 旧 dataset_id 绑定的既有产物按 `check_dataset_id` 既定 fail-closed 迁移策略拒绝混用（预期行为，A 线 t21 正式测量应绑定 `e15b4fc4…`）；本任务不修改任何研究产物。

### 4.3 G1–G7 数据资格门禁（formal 模式，无 `--dev`）

命令：`& D:\minequant\.venv\Scripts\python.exe scripts\check_production_gates.py`（默认 formal；本任务全程未使用 `--dev`）→ **exit 0，ALL GATES PASS**：

| 门禁 | 结果 | detail |
| --- | --- | --- |
| G1 历史成员区间 | PASS | 000300.SH:817, 000905.SH:1757；snapshot-shaped=False |
| G2 上市日完整 | PASS | missing stock rows: 0, null list_date: 0 |
| G3 日历覆盖日线窗口 | PASS | calendar 19901219..20261231 vs daily 2015-01-05..**20260901** |
| G4 区间无重复/重叠 | PASS | clean |
| G5 strict PIT 契约 | PASS | mode=strict, constituent_rows=2574, stock_rows=5542, open_sessions=8797 |
| G6 年度最少 eligible | PASS | min 473(2015)；2026:795 |
| G7 零 bar 区间审计 | PASS | 2,426 observed intervals, **0 zero-bar**, median coverage 100.00% |

### 4.4 research_doctor 重跑

`python -m ashare_model.research_doctor --output data\research_doctor.json --json` → exit 0，
**healthy: true / status: HEALTHY**（02:35:55 +08）；dataset_id `e15b4fc4…`；gates formal 全 PASS。
产物已刷新：`data/research_doctor.json`（gitignored 运行时产物）。同步前基线（01:24，
healthy=true，dataset_id `a839ecf2…`）留档于任务记录。

## 5. 事件与运行纪律记录

- 窗口期间其他成员的 `pytest -q tests -n auto`（miniconda 解释器，02:20/02:31 起）与生产 DB 无交集（测试用 tmp fixtures，§10.1），不构成正式运行占用。
- sync 在 02:15:24 完成（manifest 落库、日志导出后）解释器退出挂起；因 DuckDB 连接已在 sync_all 的 finally 关闭、结果与日志完整，对挂起包装进程执行了清理 kill（含后台 job pwsh-4），无数据影响。
- 未 push、未建远程 PR、未在 main 直改；跟踪文件零改动；本日志为唯一新增未跟踪文件。

## 6. 原始产物与命令索引

| 产物 | 路径 |
| --- | --- |
| 同步运行日志（loguru 导出） | `logs/sync_20260902_021524.txt`（gitignored） |
| 同步控制台捕获 | `%TEMP%\t12_sync_run.log`（会话临时） |
| research_doctor 报告（同步后） | `data/research_doctor.json`（gitignored） |
| 审计脚本（只读，临时） | `%TEMP%\t12_audit.py`（前）、`%TEMP%\t12_audit_after.py`（后） |
| doctor 基线/复核 JSON 捕获 | `%TEMP%\t12_doctor_before.json`、`%TEMP%\t12_doctor_after.json`（会话临时） |

## 7. 裁决

daily_bar 数据时效同步完成：覆盖至最近已完成开市日 **20260901**，G1–G7 formal 全绿，
dataset_id **e15b4fc4…e07547** 已对齐并记录。A 线 t21 正式测量可以本 dataset_id 起
步（数据截止 20260901；20260902 当日节尚未发生，属自然边界而非缺口）。
