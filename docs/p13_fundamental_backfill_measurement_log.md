# P13 fundamental 回填测量日志：net_operate_cash_flow + total_assets（evidence-only）

> 团队任务 t20（成员 impl-c-data，DB 独占窗口②，串行于 t12 窗口①之后）。
> 性质：§11.1/§11.4 后置 evidence-only 记录，只写"实际发生什么"；被测实现 =
> t14 分支 `codex/p0-3-fundamental-fields` @ **c2e1ec793c47d668afba4ec45758dd63ba480faa**
> （= 02c30e8 主实现 + c2e1ec7 窗口②缺陷修复，见 §3；代码先于回填落库，符合
> p13 §3.4"代码侧先交付"的次序）。运行类型：engineering（数据运维回填）；
> 不构成任何研究/收益结论；族级裁决（§8）另行执行。

## 1. 环境与执行身份

| 项 | 值 |
| --- | --- |
| 日期 | 2026-09-02（Asia/Shanghai），窗口②区间 04:03–05:31 |
| Python | 3.13.12，`D:\minequant\.venv\Scripts\python.exe` |
| 工作目录 | `D:\minequant\AlphaGPT-c`（分支 `codex/p0-3-fundamental-fields` @ c2e1ec7） |
| 数据库 | `D:\minequant\AlphaGPT\data\ashare.duckdb`（经 `ASHARE_DUCKDB_PATH` 指定；`ASHARE_PARQUET_DIR` 同指生产缓存） |
| 配置 provenance | `config/ashare_config.yaml` @ c2e1ec7；无 runtime overrides、无 config/.env |
| 数据源 | akshare 1.18.91（Eastmoney 业绩报表/现金流量表/资产负债表批量端点 + Sina 补充），在线模式 |
| 独占性 | 执行前无任何 python 进程；无 sim 运行；全队无其他 DB 写入者（captain 确认） |
| 备份 | 回填前快照备份 `data/ashare.duckdb.p13bak`（1,009,790,976 字节，迁移前状态） |

## 2. 执行前只读审计（§5.6 第一步；与 t14 验收#1 审计记录一致）

- 基线（04:0x）：fundamental_pit 205,589 行 / 5,202 码 / 94 报告期（19901231–20260630）；
  **两新字段 0 有限值**（迁移前列尚不存在）；announce master 98.51%；
  dataset_id `e15b4fc4…`（10,908,341 行）；scope = 5,542 码。
- 回填范围：in-scope universe（`persisted_scope_codes` = stocks ∪ PIT 成员 ∪ 缓存 bar 码 = 5,542）× 46 报告期（2015Q1–2026Q2）× 2 字段 —— 与 t14 审计记录逐字一致。

## 3. 执行过程（含三次缺陷的完整记录）

### 3.1 schema 迁移（§5.1）

幂等 `create_schema`（内含 `ALTER TABLE … ADD COLUMN IF NOT EXISTS` ×2）先行应用于生产库：
两列加入（DOUBLE 可空），既有 205,589 行原值不变、新列 NULL。回滚 = 代码 revert（契约 §5.1）。

### 3.2 主回填驱动（单一语义路径 sync_fundamentals，universe = 5,542）

04:03–04:10 三相完成，每相 46/46 季：

```text
Earnings reports synced: 46 quarters, … rows
cash_flow reports synced: 46 quarters, … rows
balance_sheet reports synced: 46 quarters, … rows（实际 0 行写入，见缺陷①）
```

### 3.3 缺陷①：total_assets 0 行写入（t14 代码缺陷）

`get_balance_sheet` 调用了 `ak.stock_zcfzb_em` —— akshare 1.18.91 无此函数（正确名
`stock_zcfz_em`，已对照本机 site-packages 源码 L20 核实）。46/46 季 AttributeError
被 sync 的既有失败护栏记录并跳过 → total_assets 0 行。**t14 的 RED 未抓到它，因为
测试网络层被 mock，真实 akshare 属性面从未被行使**（§2.2 第 5 步教训，修复 commit
说明如实写明）。修复 c2e1ec7：函数名改正 + 不触网契约测试
（monkeypatch akshare 模块，断言被调用属性名 == "stock_zcfz_em"，并删除错误属性名
确保调用即响），先红（AttributeError 实证）后绿。

### 3.4 缺陷②：域外行复活 +116,613 行（既有缓存路径缺陷）

首轮后表涨至 332,564 行 / 11,045 码。根因：`sync_fundamentals` 的**缓存读取路径**
不做 universe 过滤，而 `earnings_*.parquet` 缓存为 2026-08-15（P2-01 purge 之前）
的全市场内容（`earnings_20150630.parquet`：8/15 18:18 写入、6,255 行 = 6,255 码，
与 DB 该期行数精确吻合）。回填把 8/27 purge 掉的域外行复活（earnings 字段），
且复活行自带 announce master，进而放进 cash_flow 行（5,205 码的 cfo 数据不受影响：
全部 ⊆ in-scope）。修复 c2e1ec7：earnings 与 bulk 两循环的缓存读取后同样应用
universe 过滤（与 fetch 路径同纪律）；不触网 RED：全市场 earnings+cash_flow 缓存
fixture → 断言仅 in-scope 码入表，先红（600000.SH 复活实证）后绿。

### 3.5 缺陷③：每股 Sina 补充爬取超资源上限（按 §9.1 中断）

补充循环对 ~3,400 无缓存码逐码真实拉取，Sina 端点对多数码 3 次重试失败
（'NoneType' object ...）。在 164 次失败（~55 分钟）后按 §9.1 资源上限
（"回填：分钟级"）中断驱动。影响：roa/debt_ratio/dividend_yield 覆盖率维持
回填前状态（~10.1%/10.1%/1.4%），COALESCE 语义保证零清空；**不影响两目标字段**。
遗留事项：无缓存码的补充覆盖为既有局限（非本契约引入），族级裁决时按实际覆盖率
评估；是否补爬另行决策。

### 3.6 数据清理（B）：官方 purge 工具

`python scripts/check_fundamental_scope.py --purge`（P2-01 官方工具，幂等）：

```text
purged 116613 fundamental rows (332564 -> 215951) and 0 invalid stocks rows (5542 -> 5542)
```

purge 后 `--report`：`215951 rows / 0 out-of-scope codes; scope = 5542 codes; scope OK`。
207,021 个 cfo 值全部属于 ⊆scope 的码，purge 零损失（purge 前后分项审计见 §2/§4）。
审计文件：`data/fundamental_scope.json`（运行时产物）。

### 3.7 重跑（C）：balance_sheet 46 季（修复后代码）

驱动复用修复后 sync 区段组件（`get_balance_sheet` + universe 过滤 + master join +
COALESCE upsert），2026-09-02T05:29:04 → 05:29:22（**18 秒**）：
写入 201,251 行；751 行因无 master 匹配被丢弃（fail-closed 纪律）；46/46 季完成。

## 4. 复核（§5.6 第三步）

### 4.1 行数 / 日期范围 / 覆盖率门（§5.4，fail-closed）

| 项 | 回填前 | 回填后 |
| --- | ---: | ---: |
| fundamental_pit 行数 | 205,589 | **215,951** |
| distinct ts_code | 5,202 | 5,205 |
| net_operate_cash_flow 有限值 | 0 | **207,021** |
| total_assets 有限值 | 0 | **201,250** |
| 两字段报告期范围 | — | 均为 20150331–20260630 |

§5.4 覆盖率门（分母 = in-scope 码 × 已同步报告期 2015Q1–2026Q2 的既有行 207,343）：

| 字段 | 有限值 | 覆盖率 | 门（≥0.9） |
| --- | ---: | ---: | --- |
| net_operate_cash_flow | 207,021 | **0.9984** | PASS |
| total_assets | 201,250 | **0.9706** | PASS |

逐季填充示例（完整 46 季数据见任务记录）：20231231: cfo 5,205 / ta 5,205；
20260630: cfo 5,204 / ta 5,204；早年季（2015Q1: cfo/ta 2,759/2,759）随上市家数
自然增长，缺口为停牌/无数据码，属数据事实非管线缺陷。

### 4.2 manifest 重建与 dataset_id（§6.2 如实记录）

`python -m ashare_data.manifest`（05:29:56 +08 后）：
**dataset_id `e15b4fc4…` → `b7b4dd4b03fef19755814530dfbead040d6fb88137a3fc5f3dc4992526b64377`**
（10,918,703 行 / 8 表；manifest 持久化历史 b927074a → a839ecf2 → e15b4fc4 → b7b4dd4b）。
manifest 版本常量不 bump（契约 §6.1：dataset_id 随内容重算是数据事实）。

### 4.3 G1–G7 数据资格门禁（formal 模式，无 `--dev`）

`python scripts/check_production_gates.py` → **exit 0，ALL GATES PASS**：
G1 817/1757 区间；G2 零缺失上市日；G3 calendar 19901219..20261231 vs daily
2015-01-05..20260901；G4 clean；G5 strict 契约（constituents 2574 / stocks 5542 /
sessions 8797）；G6 min eligible 473（2026:795）；G7 2,426 区间 0 零 bar。
（同步前基线同全绿，见 t12 日志。）

### 4.4 research_doctor 重跑

`python -m ashare_model.research_doctor --output data\research_doctor.json --json` →
exit 0，**healthy: true / status: HEALTHY**（05:30:51 +08）；code.commit = c2e1ec7；
dataset_id `b7b4dd4b…`；gates formal 全 PASS。

## 5. 原始产物与命令索引

| 产物 | 路径 |
| --- | --- |
| 回填驱动日志（首跑含缺陷证据） | `%TEMP%\t20_backfill_run.log` |
| 修复后 balance_sheet 重跑驱动 | `%TEMP%\t20_bs_rerun.py`（会话临时，逻辑镜像已提交的 sync 区段） |
| 审计脚本（只读） | `%TEMP%\t20_pre_snapshot.py`、`t20_post_audit.py`、`t20_diagnose.py` |
| 范围审计文件 | `data/fundamental_scope.json`（--purge 运行时产物） |
| doctor 报告 | `data/research_doctor.json`（刷新至 b7b4dd4b…） |
| 备份 | `data/ashare.duckdb.p13bak`（迁移前，1.01GB） |

## 6. 裁决与后续

1. **覆盖率门两字段 PASS**（0.9984 / 0.9706 ≥ 0.9）：p13 §6 预注册解除条件的数据
   侧验收达成；族⑤代码已解锁（t14），族级裁决重测（§8，77 名全量、新审计脚本、
   引用 post-backfill dataset_id `b7b4dd4b…`）另行执行。
2. 后续测量（t32/t34）应绑定 **dataset_id `b7b4dd4b…`**（数据截止：daily 20260901 /
   fundamental 20260630）；旧 id 绑定产物按 fail-closed 拒绝混用。
3. 未运行项：全量 pytest（t31）；补充爬取补全（~3,400 码，既有局限，另行决策）。
4. 全程未 push、未合并；跟踪文件零改动（本日志为唯一新增未跟踪文件，t14 修复已
   以 c2e1ec7 提交于任务分支）。
