# P2 免费数据可信度分层 — 契约 (contract)

> 改动规则：改动前先 commit 当前状态 → 分支开发 → 测试 → 验证 → 合并回 main。
> 本文是 P2 全部改动的契约与仲裁依据：测试断言唯一合法来源是本契约与既有接口契约，
> 实现不得超出本契约范围；实现与测试冲突时回到本契约判断谁错。

## 1. 数据可信等级定义

| 等级 | 内容 | 说明 |
|---|---|---|
| **Tier A** | 价格、成交量、换手、上市日、可靠历史成员 | 交易所日线（开高低收/量/额/换手）、经回填校验的 `stocks.list_date`、PIT 历史成员区间（`constituents` 表 + 上市日，BaoStock 月末快照压缩） |
| **Tier B** | 融资融券、保守发布日期基本面 | 两融余额（交易所日度 feed）、基本面（按法定披露季末日对齐：Q1→04-30、H1→08-31、Q3→10-31、年报→次年 04-30，从不提前可见、不追踪重述） |
| **Tier C** | 当前行业快照、历史 ST 近似、占位数据 | 申万一级行业成分**当前快照**（投射到历史）、`stocks.is_st` 当前快照（仅真实当日撮合，历史回放一律按板块价幅判定）、无数据源占位（`NORTHBOUND_CHG` 2024-08 起停披露，恒中性） |

**可用时间规则（P2-02 记录在产物中）**：
- A：交易日收盘即知，次日可用；`LIST_AGE` 自校验后的上市日可用；PIT 成员资格自 `in_date` 起、按数据窗口封顶。
- B：基本面自披露季末日起可见并前向填充（`fundamental_pit` 的 `announce_date`）；两融自 feed 交易日起（保守取次日）。
- C：行业快照仅自其同步日（作为当日事实）有效，**禁止投射历史**；ST 快照仅当日撮合；占位因子无任何可用信号。

## 2. PitLevel → DataTier 映射（唯一事实来源）

`ashare_model.feature_registry.PitLevel`（既有的最弱数据源分级）映射到 P2 三档：

| PitLevel | DataTier |
|---|---|
| `PIT_DAILY`（本地日线计算） | **A** |
| `PIT_FUNDAMENTAL`（PIT 财报） | **B** |
| `PIT_CAPITAL`（两融/资金流 feed） | **B** |
| `SNAPSHOT`（当前行业快照外推） | **C** |
| `NEUTRAL`（无数据源占位） | **C** |

即：A = 全部日线族 + `LIST_AGE`（上市日属 A）；B = `FUNDAMENTAL_PIT_NAMES` + `MARGIN_BALANCE_CHG`；
C = `INDUSTRY_MOMENTUM` + `IND_REL_*` + `NORTHBOUND_CHG`。

**Tier A 独立性保证（验收）**：Tier A 因子集合必须排除全部基本面（`FUNDAMENTAL_PIT_NAMES`）、
当前行业（`INDUSTRY_MOMENTUM`、`IND_REL_*`）与占位（`NORTHBOUND_CHG`）；Tier A 因子的计算路径
只消费日线列 + 上市日 + PIT 成员掩码。历史回放无日期化 ST（既有事实），ST 快照永不进入历史因子路径。

## 3. P2-01 基本面表范围契约

- `fundamental_pit` 只允许包含**合法 A 股代码**（`is_valid_a_share_code`）且**属于持久化股票范围**
  （`stocks` 表 ∪ PIT `constituents` 历史成员 ∪ 日线 parquet 缓存代码）的行。
- `stocks` 表只允许合法 A 股代码（现存的 900xxx B 股行是历史脏数据，须清理，且 `upsert_stocks` 增加校验防止再入）。
- **错误 .BJ 归属**：东财业绩报表的裸代码经 `_ts_code_from_symbol` 映射，新三板/退市 4/8 开头代码可能被
  误标 `.BJ`（前缀法与真实北交所 A 股无法区分）。契约裁决：**入库行必须同时命中股票范围**——北交所/新三板
  代码不在同步范围（无日线）时一律不落库；若未来某代码成为真实宇宙成员（有日线、有成分记录），则允许入库。
  该规则同时满足"非股票范围"与"错误 .BJ 归属"两项过滤。
- 既有脏数据迁移：`scripts/check_fundamental_scope.py --purge` 删除范围外行（记录前后行数/代码数到
  `data/fundamental_scope.json`）；同步路径在 `sync_fundamentals` 落库前按 `universe` 过滤，防止复发。

## 4. P2-02 因子产物记录契约

所有携带因子/公式的产物记录：`data_tier_version`、逐特征 `data_tier`、公式 `tiers_used`（含最大档）：

| 产物 | 记录内容 | 版本 |
|---|---|---|
| `data/factor_report.json`（diagnostics） | 逐特征 `data_tier` + 汇总 + 时间规则 | `DATA_TIER_VERSION` |
| feature registry 输出 | 逐特征 `data_tier`（新增字段） | `FEATURE_REGISTRY_VERSION` 1→2 |
| 候选评分（`CandidateScore.to_dict`） | `data_tier` / `data_tiers_used` | `DATA_TIER_VERSION` |
| `best_ashare_strategy.json` | `data_tier` 块（公式级） | `DATA_TIER_VERSION` |
| 协议产物（protocol artifact） | 逐行 `data_tier` + top_trial | `PROTOCOL_VERSION` 20→21 |

- 公式级追溯 API：`formula_data_tier_report(tokens | bare_feature_name) -> {max_tier, tiers_used, per_feature}`，
  通过 IR 提取特征名后逐特征定档；`formula=None` 的裸基线行按 `formula_text` 定档。
- 无公式行（equal_weight/noise 等）记录 `data_tier: null`（无因子可追溯）。

## 5. P2-03 / P2-04 晋级门禁契约

- `evaluate_challenger` 新增 **`data_tier` 门**（第六道门）：默认 `allowed_data_tiers=("A",)`——
  Champion 候选公式的每个特征都必须属于 Tier A；任一 B/C 特征即失败并列出违规特征。
- Tier B **单独对照**：显式 `allowed_data_tiers=("A","B")` 时 B 可放行，裁决记录 `data_tier_policy`
  （allowed 集合 + 公式逐特征档位），两条裁决路径互不可混淆。
- Tier C **永不进入晋级结果**：`allowed_data_tiers` 含 `"C"` 或未知档位时抛 `ValueError`（契约拒绝，
  而非静默降级）；C 档因子只出现在研究展示产物（registry / factor_report / tier 报告）。
- CLI：`python -m ashare_model.promotion --allow-tier-b` 显式开启 Tier B 对照，默认保持 A-only。

## 6. P2-05 分层诊断与消融报告契约

- 新增 `ashare_model/tier_reports.py`（`TIER_REPORT_VERSION 1`）+ `scripts/tier_reports.py`。
- 三个层级集合各输出**诊断**与**消融**：`A`（仅 A）、`AB`（A+B）、`all`（A+B+C）。
- 诊断：复用 `factor_report` 全链路（coverage / rank-IC / ICIR / 相关矩阵），per_feature 只含该集合特征。
- 消融：基线 = all；`ablate_C`（剔除 C → A+B）、`ablate_BC`（剔除 B+C → A）；同 seed/steps/batch，
  记录 best_reward、best_formula，且**每个 best_formula 携带其数据等级追溯**（验收：任意公式可追溯到数据等级）。
- 剔除语义：与家族消融同法（`ablate_factors` 把被剔除特征信号置中性，特征仍在采样词表内）。
  每个 run 报告的公式为**管线选中候选**（与家族消融同一选择路径）；当选中的 token 级特征
  越出层级集合时，改报该集合内**预算内最佳 eligible 候选**（两者都记 `confined=true`）；
  仅当集合内不存在任何合格候选时才退回全局选择并如实追溯（`confined=false`）。
- 产物：`data/tier_report.json`（版本、三集合诊断、消融结果、公式追溯、confined 标记）。

## 7. 迁移 / 拒绝策略

- `fundamental_pit` 范围外行、`stocks` 非法行：purge 脚本清理，前后计数入 `data/fundamental_scope.json`。
- v20 协议产物：晋级门 G1 按版本拒绝（既有行为），保持可读归档；需重新测量 v21 产物才能晋级。
- `FEATURE_REGISTRY_VERSION` 1→2：新增字段为加性变更；旧 registry 产物仍为有效历史，不重写。
- `PROTOCOL_VERSION` 20→21：行/stitched/top_trial 记录数据等级；无公式语义变化，仅产物 schema 加字段。

## 8. 验收映射

| 验收项 | 证据 |
|---|---|
| 任意公式可追溯到所依赖的数据等级 | `formula_data_tier_report` + 各产物记录 + 消融报告中公式追溯块 |
| Tier A 实验不依赖基本面、当前行业或历史 ST 近似 | Tier A 集合断言测试 + factor_report(A) 只含 A 档特征 + Tier A 因子计算路径只用日线/上市日/成员掩码 |
| 不需要购买付费数据 | 全部数据源为既有免费 AkShare 端点，零新依赖（requirements 不变） |
