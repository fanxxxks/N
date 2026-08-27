# Phase 6 measurement log (P2-01 .. P2-05: 免费数据可信度分层)

> 规则：改动前先 commit 当前状态 → 分支开发 → 测试 → 验证 → 合并回 main。
> 测试计数为仓库根目录 `pytest tests`（排除 `tests/test_webapi.py`：既有环境阻塞，
> starlette 1.3.1 的 TestClient 需要不可离线安装的 `httpx2`）。
> 契约见 [docs/p2_data_tier_contract.md](p2_data_tier_contract.md)。
> 本阶段为**数据可信度治理与测量**：产物是分层诊断/消融报告、基本面表范围审计、
> 晋级门禁与公式追溯；**不宣称发现 Alpha**。

| Stage | Commit | Tests passed | Δ | Notes |
|---|---|---|---|---|
| Baseline (main, pre-P2) | `600ba0e` | **941** | — | `logs/pytest_p2_baseline.log`（本阶段基线复跑，webapi 排除） |
| docs: P2 contract | `bf2454b` | — | — | 契约先行（Tier A/B/C 定义、时间规则、门禁、迁移策略） |
| test: data-tier contract (red) | `c59b3d8` | — | — | `tests/test_data_tier.py` |
| feat: data-tier core | `a659ff5` | — | — | `ashare_model/data_tier.py`（DATA_TIER_VERSION 1）、`ir.feature_names`、`feature_registry.pit_level_of` |
| feat: P2-02 artifact recording | `20fc14f` | — | — | registry v2、PROTOCOL_VERSION 21、factor report、candidate scores、strategy artifact |
| feat: P2-03/04 promotion gate | `f0433c5` | — | — | 第六道 data_tier 门：默认 A-only；B 单独对照；C 永不晋级 |
| feat: P2-01 fundamentals scope | `5819447` + `ff08204` | — | — | sync 按 universe 过滤、upsert_stocks 校验、审计/purge 脚本 |
| feat: P2-05 tier reports | `67990ac` | — | — | `ashare_model/tier_reports.py`（TIER_REPORT_VERSION 1）+ CLI |
| docs: README/CATREADME P2 | `98405a7` | — | — | 文档与代码同 commit |
| 数据治理测量（purge） | 运行（无代码） | — | — | `data/fundamental_scope.json`：322,876→205,589 行；stocks 5,546→5,539 |
| 分层诊断（真实数据） | 运行（无代码） | — | — | `data/tier_report_diagnostics.json`：A=44 / AB=56 / all=62 特征 |
| 分层消融（真实数据，confined 语义） | `84b7221`、`a9630cc`、`6a80a13` | — | — | 每次消融 run 报告 token 级限定在层级集合内的公式（selected-first） |
| docs: R-21 已落实 | `7199ba8` | — | — | 既有风险清单勾销 |
| 最终验证 | `cc27073`（合并） | **981** | — | 全量复跑 981 passed（+40 新增，0 回归），合并回 main |

## 1. 提交前后不变量

| 不变量 | 改动前（main @ 600ba0e） | 改动后（main @ cc27073） | 验证方式 |
|---|---|---|---|
| 全量 Python 测试 | 941 passed | **981 passed, 0 failed**（+40 新增，0 回归） | `pytest -q tests`（webapi 排除） |
| 语义版本 | PROTOCOL_VERSION 20 / REWARD_VERSION 13 / FEATURE_REGISTRY_VERSION 1 | PROTOCOL_VERSION **21**（产物逐行记录 data_tier）/ REWARD_VERSION 13 不变 / FEATURE_REGISTRY_VERSION **2** / 新增 DATA_TIER_VERSION 1、TIER_REPORT_VERSION 1 | `grep` 各版本常量 |
| 依赖 pin | requirements 不变 | 不变（零新依赖） | `freeze_lock.py --check`（通过） |
| 既有产物 | legacy 盖章产物不变 | v20 协议产物保持可读归档；晋级门按版本拒绝（G1 既有行为）；registry v1 产物仍为有效历史 | — |
| 数据表范围 | fundamental_pit 322,876 行（含 117,287 行范围外） | **205,589 行，全部在持久化股票范围内**（0 行范围外）；stocks 5,546→5,539（清 7 个 B 股行） | `scripts/check_fundamental_scope.py --report` |

## 2. 契约（先于实现，`docs/p2_data_tier_contract.md`）

### P2 数据等级（DATA_TIER_VERSION 1）

- **Tier A**：价格/成交量/换手/上市日/可靠历史成员 → `PitLevel.PIT_DAILY`
  （44 个特征，全部为本地日线计算，required columns ⊆ BAR_COLUMNS）。
- **Tier B**：融资融券/保守发布日期基本面 → `PIT_FUNDAMENTAL` + `PIT_CAPITAL`
  （12 个：11 个 PIT 财报 + `MARGIN_BALANCE_CHG`）。
- **Tier C**：当前行业快照/历史 ST 近似/占位数据 → `SNAPSHOT` + `NEUTRAL`
  （6 个：`INDUSTRY_MOMENTUM` + 4 个 `IND_REL_*` + `NORTHBOUND_CHG`）。
- 每档携带**可用时间规则**（`TIER_TIME_RULES`）：A 收盘即定、次日可用；B 按
  披露季末日起可见（Q1→04-30、H1→08-31、Q3→10-31、年报→次年 04-30）、两融自
  feed 日起；C 仅快照当日事实/仅研究展示。
- 公式追溯 API：`formula_data_tier_report(tokens | bare_feature_name) →
  {max_tier, tiers_used, per_feature}`；协议产物 v21 起逐行记录。

### P2-01 基本面表范围

- `fundamental_pit` 只含合法 A 股代码且属于持久化股票范围（stocks ∪ PIT
  constituents ∪ 日线缓存）的行；批量财报在 `sync_fundamentals` 落库/缓存前按
  `universe` 过滤；`upsert_stocks` 拒绝非 A 股代码；`scripts/check_fundamental_scope.py`
  提供 `--report` / `--purge`（前后计数写入 `data/fundamental_scope.json`）。

### P2-02 产物记录

- factor_report：逐特征 `data_tier` + `tier_summary` + `data_tier_rules`；
- feature registry v2：逐特征 `data_tier` + `data_tier_version`；
- CandidateScore / 策略产物：`data_tier` 块；
- 协议产物 v21：逐行/stitched/top_trial `data_tier` + 顶层 `data_tier_version`。

### P2-03 / P2-04 晋级门禁

- 第六道门 `data_tier`：默认 `allowed_data_tiers=("A",)`；Tier B 仅显式
  `("A","B")` 单独对照（裁决记录 `data_tier_policy`）；Tier C 与未知档在
  策略层抛 `ValueError`，永不进入晋级结果。

### P2-05 分层报告（TIER_REPORT_VERSION 1）

- 三个层级集合 A / A+B / all 各自输出诊断（coverage/rank-IC/ICIR/相关矩阵）
  与消融（同 seed/steps/batch 训练；all=基线、AB=剔除 C、A=剔除 B+C）；
  每个消融公式携带 `formula_data_tier` 追溯。

## 3. 关键测量（运行结果）

数据：`dataset_id b927074a45…`（同 Phase-0/P1，11,003,350 行 / 8 表；purge 后
fundamental 表 205,589 行）；机器：Windows / Python 3.13 / torch 2.11.0+cu128 /
venv `D:\minequant\.venv`。

### 3.1 P2-01 范围治理（`data/fundamental_scope.json`）

| 表 | 改动前 | 改动后 | 移除 |
|---|---|---|---|
| fundamental_pit 行数 | 322,876 | **205,589** | 117,287（5,843 个范围外代码：5,832 个 .BJ/新三板 + 11 个无 bar 沪深代码） |
| stocks 行数 | 5,546 | **5,539** | 7（900xxx B 股，历史脏数据） |

purge 后验证：`fundamental_pit` 中**0 行**代码同时不在 stocks 与 constituents；
抽查移除代码（002720.SZ / 300728.SZ / 301688.SZ / 601123.SH）均无成分记录、无
日线 bar——管线不可消费，移除无损。全市场批量财报在后续同步中按 universe
过滤，不会复发（测试钉死）。

### 3.2 P2-05 分层诊断（`data/tier_report_diagnostics.json`，训练窗 2015-01-05..2023-12-27）

1630 只股票（全周期代码并集），日均 eligible ≈ 777：

| 层级集合 | 特征数 | 档位构成 | 最强 |IC| | 最强 ICIR | 备注 |
|---|---|---|---|---|---|---|
| **A** | 44 | 全部 A | LIMIT_UP_EVENT 0.0858 | LIMIT_UP_EVENT 0.466 | 无 fundamental/external/industry 家族（验收：A 档独立） |
| **A+B** | 56 | A 44 + B 12 | LIMIT_UP_EVENT 0.0858 | LIMIT_UP_EVENT 0.466 | 新增 fundamental 家族（within-family \|corr\| 0.187） |
| **all** | 62 | A 44 + B 12 + C 6 | LIMIT_UP_EVENT 0.0858 | LIMIT_UP_EVENT 0.466 | 新增 industry/external 家族 |

要点：A 档 top 特征与全量一致（LIMIT_UP_EVENT），B/C 档加入不改变训练窗内
top-IC 结论；A 档 44 特征内部 liquidity 家族相关最高（0.640）、event 最低
（0.002）。**仅测量，不宣称 Alpha**。

### 3.3 P2-05 分层消融（`data/tier_report.json`，steps=8 × batch=64，seed=42）

同一预算下三个层级集合各训练一次；报告的公式为**管线选中候选**（与家族消融同一
选择路径），当选中公式的 token 级特征越出层级集合时改报集合内最佳 eligible 候选
（`confined` 标记）。剔除 = 信号置中性（家族消融同法）。每次训练 ~33 分钟。

| 层级集合 | 剔除 | confined | best_reward | best_formula | formula_data_tier | delta vs all |
|---|---|---|---|---|---|---|
| all（基线） | 无 | True | **0.894** | `DELAY1(KURT_20) SUB MA20(ABS(STD20(DECAY(ILLIQ_20))) DIV DELTA5(RET_1))` | A（全部 A 档） | — |
| A+B | C 档 6 特征 | True | **0.887** | `MA60(CS_NEUTRALIZE(STD10(RET_5))) DIV (TURNOVER CORR5 DOWNVOL20(DOWNVOL5(STD10(PROFIT_YOY))))` | B（PROFIT_YOY + A 档） | −0.007 |
| A | B+C 档 18 特征 | True | **0.756** | `DOWNVOL60(GATE(REVERSAL_5, CS_RANK(MACD_DIF), NEG(MAX_20) DIV (MARKET_CAP CORR20 CLOSE_POSITION)))` | **A（全部 A 档）** | −0.137 |

要点：全部三档 `confined=true`——报告的每个公式的 token 级特征都属于其层级集合
（验收：任意公式可追溯到所依赖的数据等级；A 档 run 的公式确实只依赖 A 档数据）。
剔除 C 档（行业快照/占位）几乎无损失（−0.007）；进一步剔除 B 档（基本面/两融）
损失 0.137——该训练窗内 B/C 档信号确有贡献。**仅测量，不宣称 Alpha**；预算为
smoke 级（8×64=512 唯一评价/run），结论不用于晋级。

### 3.4 晋级门禁与追溯（测试证据）

- `tests/test_promotion.py`：Tier B 公式默认被拒（reason 指名特征与档位）、
  `("A","B")` 显式对照放行并记录 policy、Tier C 即使显式对照也被拒、
  未知档/`"C"` 直接 `ValueError`。
- `tests/test_stitched_oos.py`：协议产物 v21 逐行/stitched/top_trial 记录
  `data_tier`，无公式行记录 `null`。
- 任意公式追溯：`formula_data_tier_report`（token 或裸因子名）在
  `tests/test_data_tier.py` 全覆盖。

## 4. 版本变更（语义变更才 bump）

| Module | Before | After |
|---|---|---|
| ashare_model.data_tier | — (new) | DATA_TIER_VERSION 1 |
| ashare_model.tier_reports | — (new) | TIER_REPORT_VERSION 1 |
| ashare_model.evaluation | PROTOCOL_VERSION "20" | **"21"**（产物逐行/stitched/top_trial 记录 data_tier；公式语义不变） |
| feature_registry | FEATURE_REGISTRY_VERSION 1 | **2**（逐特征新增 data_tier、payload 记录 data_tier_version） |
| ashare_model.promotion | 五道门 | **六道门**（+data_tier；默认 A-only；--allow-tier-b 单独对照） |
| REWARD_VERSION / 其余版本 | 不变 | 不变 |

## 5. 迁移 / 拒绝策略

- **fundamental_pit 范围外行（117,287）与 stocks B 股行（7）**：`--purge` 已清理，
  前后计数入 `data/fundamental_scope.json`；同步路径按 universe 过滤 + upsert 校验
  防止复发；数据库改动前已备份（`data/ashare.duckdb.p2bak`）。
- **v20 协议产物**：晋级门 G1 按 `protocol_version` 拒绝（既有行为），保持可读
  归档；需要 v21 重测才能晋级。
- **registry v1 / 旧 factor_report 产物**：加性字段，仍为有效历史，不重写。
- **错误 .BJ 归属**：北交所/新三板 4/8 开头代码与真实 A 股前缀无法区分，契约
  裁决为**范围交集**——不入同步范围（无日线）即不落库；未来若有真实成员资格则
  自然放行。

## 6. 验收证据

1. **任意公式可追溯到所依赖的数据等级**：`formula_data_tier_report` +
   协议 v21 逐行 `data_tier` + 消融报告每个公式的 `formula_data_tier`（§3.3）。
2. **Tier A 实验不依赖基本面、当前行业或历史 ST 近似**：A 档 = 44 个本地日线
   特征（`required_columns ⊆ BAR_COLUMNS`，测试断言）；A 档诊断报告无
   fundamental/external/industry 家族（§3.2）；历史回放本就不使用 ST 快照。
3. **不需要购买付费数据**：全部端点仍为免费 AkShare；requirements 零变化。
4. **Champion 默认 A-only、B 单独对照、C 永不晋级**：`tests/test_promotion.py`
   六道门全绿（§3.4）。
5. **基本面表范围**：purge 后 0 行范围外（§3.1），同步过滤与 upsert 校验有测试钉死。
