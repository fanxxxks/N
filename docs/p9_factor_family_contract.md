# P9 因子族精简与正交因子族预注册契约（已批准）

- 状态：**APPROVED — 2026-09-01 由 AgentTeams captain 批准**（用户已授权 captain 自主推进，豁免 AGENTS.md 1.7 质问义务）。批准后可按 §8 顺序实施。批准后本契约保持前瞻性，禁止用实现输出反向修改本契约。
- 起草：auditor（AgentTeams alpha-orth-research t3），2026-09-01
- 证据基线：t1 数据/产物 Readiness 审计；t2 62 因子库存审计（`docs/factor_inventory_audit_20260831/audit_report.md` + `metrics.json`，commit 7958fa5，dataset_id=a839ecf2…，窗口 2015-01-05..20260821）
- 本文使用 AGENTS.md 的"必须/禁止/应当"语义。

## 1. 问题陈述与动机

t2 库存审计在真实数据上测得三个结构性事实，构成本契约的全部动机：

1. **窗口变体强共线**：62 特征在 0.9 相关阈值下聚为 50 簇；其中 3 对精确镜像（|ρ|=1.000：RET_5~REVERSAL_5、MOMENTUM_60~REVERSAL_60、RET_120~REVERSAL_120）与 9 对 ≥0.9 强冗余对（VOLUME_RATIO~VOLUME_IMPACT 0.977、LIMIT_UP_EVENT~LIMIT_STREAK 0.933、TURNOVER_MA5~MA20 0.928、TURNOVER~MA5 0.926、MACD_DIF~DEA 0.926、TURNOVER_MA20~STD20 0.914、VOL_60~IVOL_60 0.907、RET_10~BIAS_20 0.902、RET_1~INTRADAY_RET 0.902）。TURNOVER 大簇（TURNOVER/MA5/MA20/STD20，彼此 0.86–0.93）内部冗余最重。
2. **稀疏事件特征被标准表示削平（t2 发现 F1，HIGH）**：`winsorize_cross_section`（1%/99% 截面分位）使当日发生率 <1% 的 0/1 事件特征整列退化为 0：LIMIT_UP_EVENT/LIMIT_STREAK 仅 19/2826 天、LIMIT_DOWN_EVENT 8 天、LIMIT_BREAK 122 天、LIMIT_UP_CNT_20 537 天具有非退化截面。事件族在 as-consumed 管道中休眠，搜索有效词表 < 名义词表。
3. **正交性缺口明确**：40 个单例簇之外，规模中性化后的流动性/波动残差、事件条件化、拥挤度均无词汇表达；而第 ⑤ 类（PIT 现金流质量/应计/资产增长/盈利加速度）所需的现金流、总资产字段在 `fundamental_pit` schema 中不存在（t2 §3.1：DIVIDEND_YIELD/ROA/DEBT_RATIO 覆盖仅 ~0.25），无法立即实现。

## 2. 范围与非目标

**范围**：(A) 窗口变体因子族精简方案（去重不删数：deprecation 机制）；(B) 4 类立即实现的正交因子族预注册契约；(C) 第 ⑤ 类 pending_data 占位契约；(D) 版本影响表、RED 测试、测量方案、裁决规则与停止条件。

**非目标**：
- 不新增任何算子（4 族全部可用现有 OPS_CONFIG 表达语义；新增算子另行预注册）；
- 不改动股票池、PIT mask、Reward、组合构造、执行与费用语义；
- 不以"原始 IC 弱"为由淘汰任何特征（TURNOVER_CHG/KURT_20/GROSS_MARGIN/HIGH_52W/MARKET_CAP 等负结果特征**保留在词表**——搜索可能经组合后使用它们；本契约只按共线性淘汰）；
- 不实现市场宽度（breadth）作为词表特征（见 §5.4——逐日截面常数经 CS 标准化必然退化为 0，与 F1 同机制；breadth 作为协议层条件变量需独立预注册）；
- 不在本契约内实现任何特征（实现于批准后按 §8 顺序进行）。

**不变量**（实施全程必须保持）：PIT 因果（特征在 t 日只用 ≤t 数据）；单一语义路径（新特征一律经 FACTOR_REGISTRY + 既有 builder/帧，不复制第二实现）；fail-closed（元数据缺失即拒绝构建 registry）；每次小批量（一次一批 11 个新特征，不扩库）；负结果为合法结果；所有正式测量引用 dataset_id 与 data_end=20260821。

## 3. 机制选择（复用既有机制，不发明新机制）

| 需求 | 采用机制 | 先例 |
|---|---|---|
| 语义完全相同的重复 | `FEATURE_ALIASES` 名称映射（token 不移位） | RET_20→MOMENTUM_20 |
| 强共线但语义不同 | `_DEPRECATION_REASONS` deprecated + `promotion_allowed=False`：**继续计算、继续可解析，但不得晋级且（随 grammar v4）不再进入采样词表** | NORTHBOUND_CHG（deprecated，中性占位） |
| 新特征 | `_FEATURE_NAMES_V4` 追加 + `FACTOR_REGISTRY` 注册项 + `feature_metadata` 权威元数据（缺失即 fail-closed） | v2/v3 代际追加 |
| 稀疏事件表示修复 | `compute_factor_tensor` 对声明式稀疏事件特征集改用稀疏安全标准化（§5.3） | 无（本契约新增语义，需版本化） |

## 4. Part A：窗口变体精简方案（去重不删数）

### 4.1 保留/淘汰清单（t2 数据裁决）

淘汰 = deprecated（promotion_allowed=False + 退出采样词表）；保留名单即"代表性窗口"。逐项证据（t2）：

| 淘汰 | 保留（代表） | 共线证据 | 保留理由 |
|---|---|---|---|
| RET_5 | REVERSAL_5 | ρ=-1.000（NEG 可互表） | 方向显式、入 short 域基线；h=1 IC +0.031 全库短周期最强 |
| MOMENTUM_60 | REVERSAL_60 | ρ=-1.000 | REVERSAL_60 ICIR 0.166@10d、10/12 年一致 |
| RET_120 | REVERSAL_120 | ρ=-1.000 | 同镜像；RET_120 自身 8/12、ICIR -0.053 更弱 |
| RET_10 | RET_1（short 域原子）+ BIAS_20 | ρ=0.902（与 BIAS_20） | RET_10 信息被两端代表覆盖 |
| VOLUME_RATIO | VOLUME_IMPACT | ρ=0.977 | log 形态更稳健，ICIR -0.065 vs -0.027@10d |
| TURNOVER_MA5 | TURNOVER + TURNOVER_MA20 | ρ=0.926 / 0.928 | 保留日频原子 + 慢代表（换手 0.143、成本 2.5%/yr）+ STD20（异质语义） |
| MACD_DEA | MACD_DIF | ρ=0.926 | DIF 领先 DEA；DEA 无独立增量（ΔIC +0.043 与 DIF +0.035 同源） |
| VOL_60 | VOL_20 + IVOL_60 | ρ=0.907（与 IVOL_60） | IVOL_60 增量 OOS 第一（+0.068）、ICIR -0.335；VOL_20 保留 short 域覆盖 |
| LIMIT_STREAK（条件性） | LIMIT_UP_EVENT（原子） | ρ=0.933 | **条件裁决**：F1 修复后重测，若 |ρ| 仍 ≥0.9 则生效；否则撤销（见 §7 裁决规则） |

净效果：词表 62（含 1 deprecated）→ 62-9 deprecated（8 立即 + 1 条件）→ **有效特征 53→54**。RET_1~INTRADAY_RET（0.902）保留双方（隔夜/日内分解是构造性原子，不是窗口变体）。

### 4.2 版本影响与迁移/拒绝策略

- `GRAMMAR_VERSION` 3→4（v4：采样 mask 排除 deprecated 特征）；`FEATURE_REGISTRY_VERSION` 3→4（新 deprecation 原因 + 新特征元数据）。
- 迁移：deprecated 名称**保留在 FEATURE_NAMES 与 token 空间**（id 不移位），历史公式按名解析不变、历史 artifact 语义不变；仅"新搜索不再采样、新晋级不再接受"。拒绝策略：任何尝试把 deprecated 特征写进新晋级候选的门禁必须拒绝（fail-closed）。
- RED 测试（实现前先红）：(a) deprecated 特征 `promotion_allowed=False` 且带 deprecation_reason；(b) grammar v4 采样分布对 deprecated 名称零概率（属性测试：大样本采样不含 deprecated token）；(c) 含 deprecated 名称的旧公式 token 解析与执行结果与 v3 逐位一致（parity 守卫）；(d) 镜像对可由 NEG+保留名逐位复现淘汰名（数学等价守卫）。

## 5. Part B：4 类立即实现的正交因子族契约

通用条款（适用于全部 4 族）：新特征登记进入 `FACTOR_REGISTRY`（family/required_columns/warmup/description）+ `feature_metadata` 权威元数据（availability_rule、hypothesis、expected_direction、semantic_type、compute_cost、depends_on）；PIT 因果由既有 builder/帧保证；全部 Tier A（daily bars）或既有 PIT capital 帧数据，无新数据接口；**无新增算子**；实现后按 §8 重测裁决。

### 5.1 族①：市场与行业残差化动量/反转（medium_cross_section 域）

- **经济假设**：A 股截面的动量/反转结构主要来自行业内相对定价；剔除行业共同运动后的残差动量/反转更纯净（t2 证据：IND_REL_TURNOVER 12/12 年一致且 ICIR -0.430 优于未残差的 TURNOVER -0.297；IND_REL_VOL_20 12/12、-0.318）。市场层面残差化**不新增特征**——逐日截面 z-score 已实现市场中性（所有特征均值 0），在契约中显式声明以免重复建设。
- **数据源/可用时间**：daily bars（close）+ 申万一级成员帧（**当前快照、非 PIT**——与既有 IND_REL_* 同 caveat，t1/t2 已披露）；行业均值只在 eligible 单元上计算。
- **新特征（2 个）**：
  - `IND_REL_RET_60` = 60 日收益 − 行业均值（同 `_industry_demean` 路径）；
  - `IND_REL_RET_120` = 120 日收益 − 行业均值。
- **预期方向**：负（行业内反转；依 IND_REL_RET_5 外推），方向按测量裁决、预注册 expected_direction=-1。
- **预测周期/执行点**：medium 域（every_10_days, horizon 10）；warmup=61/121。
- **覆盖率预期**：≥0.95（同 IND_REL_RET_20 的 0.95）。
- **RED 测试**：PIT 因果（t 日行业均值只用 ≤t 的行业成员帧与 ≤t 价格）；缺失行业成员 → NaN（不得伪造分组）；与 `_industry_demean` 既有实现逐位 parity；与 RET_60/RET_120（若无 RET_60 则与 shift_ratio 基元）的差值等于行业均值（数学守卫）。

### 5.2 族②：流动性冲击、成交萎缩、量价背离（short/medium 域）

- **经济假设**：单日流动性突增（冲击）后短期收益反转（注意力/流动性溢价文献）；成交持续萎缩伴随趋势衰竭；量价背离（价涨量缩/价跌量增）预示趋势不可持续。
- **数据源/可用时间**：daily bars（volume、amount、close）；无新数据。
- **新特征（3 个）**：
  - `LIQ_SHOCK_20` = AMOUNT_SHARE_t / MA20(AMOUNT_SHARE) − 1（当日成交份额相对 20 日基线的冲击比）；
  - `VOLUME_SHRINK_5_20` = MA5(volume) / MA20(volume) − 1（成交萎缩比，<0 为萎缩）；
  - `PV_DIV_20` = 20 日滚动相关（日收益, Δlog volume）（量价背离；NaN 感知、仅用 ≤t）。
- **预期方向**：LIQ_SHOCK_20 = -1（冲击后反转）；VOLUME_SHRINK_5_20 = -1（缩量后反转回升 → 比值与未来收益负相关，A 股短周期缩量反转假设，按测量裁决）；PV_DIV_20 = -1。
- **预测周期/执行点**：short 域（every_5_days/horizon≤5）与 medium 域双重归属——**每个特征只能归属一个域**：LIQ_SHOCK_20/VOLUME_SHRINK_5_20 归 short_price_volume；PV_DIV_20 归 medium_cross_section。
- **覆盖率预期**：≥0.99（纯本地 bar 计算）；注意与 VOLUME_RATIO/VOLUME_IMPACT 的相关性**必须在实现后重测**，≥0.9 即触发本族内二次精简裁决。
- **RED 测试**：PIT 因果（滚动窗只用 ≤t）；MA 基线用 expanding 规则处理前导（与 `_ts_window` 语义一致）；量价相关的 NaN 传播与停牌日剔除守卫；全部在合成 bars 上的 golden 值测试。

### 5.3 族③：涨跌停事件条件延续/反转（short_price_volume 域）——**先修表示，再加特征**

- **前置语义变更（本契约核心）**：`compute_factor_tensor` 对**声明式稀疏事件特征集**（SPARSE_EVENT_FEATURES = 既有 6 个 LIMIT_*/事件特征 + 本族新特征）改用**稀疏安全标准化**：跳过 1%/99% 分位裁剪，直接以 eligible 截面的均值/标准差（std 下限 1e-9）标准化。效果：无事件日截面为常数 → IC 合法地 NaN（当日无信号是事实）；有事件日信号**不再被裁剪成 0**。此为研究语义变更，必须版本化：**新增 `FACTOR_COMPUTE_VERSION = 1`（ashare_model/factors.py，接入 artifact versions 记录，additive）**。
- **新特征（3 个）**：
  - `LIMIT_UP_CNT_5` = 近 5 日一字涨停计数（LIMIT_UP_CNT_20 的短窗互补）；
  - `LIMIT_DOWN_STREAK` = 连续一字跌停天数（LIMIT_STREAK 的镜像）；
  - `LIMIT_BREAK_5` = 近 5 日炸板（触板未封）计数。
- **经济假设**：涨停条件延续（封板强度/连板高度）与炸板回落（反转）；方向：LIMIT_UP_CNT_5=+1（延续）、LIMIT_DOWN_STREAK=-1（延续下跌）、LIMIT_BREAK_5=-1（炸板回落），均按测量裁决。
- **数据源/可用时间**：既有 daily bars 的 close/high/low/pre_close + `limit_rate` 规则（无 ST 历史标志的既有限制，t2 已披露）；无新数据。
- **覆盖率验收门槛（fail-closed 裁决）**：实现后重测，LIMIT_UP_EVENT 非退化天数须从 19 **提升 ≥20 倍（≥400/2826）**且 2015–2026 每个日历年 ≥20 天；LIMIT_* 各特征同测。**不达门槛 → 族③整体记为负结果（表示修复失败/事件过于稀疏），新特征不进入词表 v4**，已有特征维持原状——禁止为通过门槛放宽任何其他语义。
- **RED 测试**：稀疏安全标准化单元测试（合成截面：99.5% 零 + 0.5% 一 → 输出非退化）；非稀疏特征经该路径**必须与现行 winsorize 路径逐位一致**（防误伤守卫）；事件帧因果性（既有 `_limit_events` 复用）；FACTOR_COMPUTE_VERSION 进入 artifact versions 的 schema 测试。
- **风险披露**：事件特征 2015-2026 各年发生率不均匀（t2 的 19 个非退化日集中于风暴期）；修复后其 IC/换手结论必须整体重测，不得沿用 t2 的条件化数值。

### 5.4 族④：横截面拥挤度（medium_cross_section 域）——纯个股拥挤，breadth 出词表

- **经济假设**：个股层面的拥挤（换手/成交份额/两融余额相对自身历史的高位）预示未来收益转负（套利限制与踩踏风险）。
- **数据源/可用时间**：turnover_rate、amount（daily bars）；rzye（`margin_balance`，2015-01-05 起，PIT capital 帧既有来源）；无新数据。
- **新特征（3 个）**：
  - `CROWD_TURNOVER_60` = turnover_t / MA60(turnover)_t；
  - `CROWD_AMOUNT_60` = amount_share_t / MA60(amount_share)_t；
  - `MARGIN_CROWD_60` = rzye_t / MA60(rzye)_t（经 `build_capital_frames` 既有单一路径扩展原始帧，不复制第二套 margin 读取）。
- **预期方向**：均为 -1（拥挤 → 未来收益负）；按测量裁决。
- **市场宽度（breadth）明确出词表**：breadth 是逐日截面常数，经 CS 标准化必退化为全 0（与 F1 同机制，t2 已证）。其作为**协议层条件变量/择时叠加**的任何使用必须另行预注册契约，本契约禁止借道实现。
- **覆盖率预期**：CROWD_TURNOVER_60/CROWD_AMOUNT_60 ≥0.95；MARGIN_CROWD_60 ≥0.85（margin 起点 20150105，t2 实测 MARGIN_BALANCE_CHG 覆盖 0.88）。
- **RED 测试**：自比值因果性（MA60 只用 ≤t）；分母退化（MA60=0 或 NaN）→ NaN（禁 DIV 伪造）；rzye 帧经 `build_capital_frames` 单一路径扩展的 parity 测试；与 TURNOVER_MA20/TURNOVER_STD20/MARGIN_BALANCE_CHG 的相关性重测（预期 0.5-0.8，≥0.9 触发族内裁决）。

## 6. Part C：第 ⑤ 类 pending_data 占位（不实现、不进词表、不耗预算）

- **族**：PIT 现金流质量 / 应计 / 资产增长 / 盈利加速度。**状态：pending_data，promotion_allowed=False，不消耗搜索预算，不进入 FEATURE_NAMES v4。**
- **数据库缺口（t1/t2 实证）**：`fundamental_pit` schema（db.py create_schema）无现金流量表字段（经营现金流/净利润组件 → 应计不可构造）、无总资产（资产增长不可构造）；股利/资产类覆盖缺口（DIVIDEND_YIELD 0.25、ROA/DEBT_RATIO 0.26）。盈利加速度虽可由既有 profit_cum TTM 二阶差分部分逼近，但为保持族原子性**整族 pending**，禁止零散拼凑。
- **占位登记**：`feature_metadata` 中以 `pending_data` 状态登记 4 个占位名（CASHFLOW_QUALITY、ACCRUALS、ASSET_GROWTH、EARNINGS_ACCEL），带 availability_rule="pending_data：等待现金流/总资产字段同步"，进 registry 摘要但不进词表/grammar。
- **解除条件**：数据侧完成现金流量表与总资产字段的 PIT 同步（含来源、announce_date、覆盖率 ≥0.9 的验收门）后，另行预注册实现契约。

## 7. 测量方案与裁决规则

1. **实现后必须重测**：以与 t2 完全一致的口径（同一 `audit_run.py` 口径：h∈{1,2,3,5,10,15,20}、min_stocks=10、IS/OOS=2022 分界、窗口 2015-01-05..20260821、同一成本模型）对 v4 词表全量重测，写入新的 measurement log（引用实现 commit 与 dataset_id）。
2. **族级裁决（每族独立、预注册如下）**：
   - 新特征 OOS（2022+）增量：以 t2 的 7 信号基准复合为基线，族内至少 1 个特征 ΔIC_OOS ≥ +0.005 且残差 IC 符号与预注册方向一致 → 族通过；否则该族记**负结果**（合法结论），其特征保留计算但 promotion_allowed=False。
   - 相关性：新特征与既有特征 |ρ|≥0.9 → 触发族内二次精简（沿用 Part A 机制），记录于测量日志。
   - 族③另有 §5.3 覆盖率硬门槛，先行裁决。
3. **条件性淘汰的裁决**：LIMIT_STREAK 的 deprecated 在 F1 修复重测后按 |ρ|≥0.9 判定；裁决结果与依据写入测量日志。
4. **测量日志**单独成文（`docs/p9_measurement_log.md`），只记"实际发生什么"，与本契约分离；正式测量必须引用精确被测 commit。

## 8. 资源上限、实施顺序与停止条件

- **小批量纪律**：本契约一次批量 = 11 个新特征 + 9 个 deprecation + 1 个标准化语义变更，不追加；任何追加需要新契约。
- **实施顺序（批准后）**：契约 PR（本文件定稿）→ RED 测试（§4.2/§5 各 RED 全部先红）→ 实现（factors.py/vocab.py/feature_metadata.py/feature_registry.py/research_domain.py/capital_flow.py 最小改动）→ GREEN + 全量回归 → `python scripts/generate_registry_docs.py` 同步生成物 → v4 词表全量重测（§7.1）→ 裁决写日志。
- **巨型模块例外（修订附录 A，captain 批准 2026-09-01）**：§5.3 的稀疏安全标准化按设计产生 NaN 验证奖励（无事件日/退化窗口的 IC 合法地为 NaN），而 `train.py` 的 best-so-far 记录 `max(-inf, nan)` 会保持 -inf 并进入 identity 载荷——canonical identity 层对该路径 fail-closed 拒绝（`CanonicalJSONError`），使任何含 NaN 奖励的运行**无法写出 run artifact**。这是 §5.3 的必要后果，故 `train.py` 的 best-so-far 记录必须随之做非有限值防御（RED/GREEN 证据见附录 A）；除此之外 `evaluation.py`/`train.py` 零语义改动。
- **停止条件**：(a) 族③覆盖门槛不过（§5.3）；(b) 任一 RED 无法稳定失败（说明契约不可测，回到本契约修订）；(c) 重测显示新特征与既有的相关结构使有效词表不增反减（诚实记录，族记负结果）；(d) 资源：重测单次 ≤30 分钟（t2 实测 1186s），超限须先优化测量而非缩小窗口。
- **回滚**：全部变更为 additive（deprecation 不删名、token 不移位、FACTOR_COMPUTE_VERSION 新增），revert 单一 commit 即恢复 v3 语义；无数据迁移。

## 9. 版本影响总表

| 版本 | 旧值 | 新值 | 变更内容 | 理由 |
|---|---|---|---|---|
| GRAMMAR_VERSION（vocab.py） | 3 | 4 | v4：采样 mask 排除 deprecated 特征；新增 11 特征 token（追加，id 不移位） | 词表/采样语义变更 |
| FEATURE_REGISTRY_VERSION | 3 | 4 | 新特征元数据 + 9 项 deprecation + pending_data 占位 | registry 语义变更 |
| FACTOR_COMPUTE_VERSION（factors.py，新增） | — | 1 | 稀疏事件特征稀疏安全标准化 | 因子张量语义变更必须版本化（接入 artifact versions 记录，additive） |
| RESEARCH_DOMAIN_VERSION | 1 | 2 | 11 个新特征归属域（①medium、②short×2+medium×1、③short、④medium） | 域特征清单变更 |
| FEATURE_NAMES / FEATURE_ALIASES | 62 名 | 62+11 名（aliases 不变） | v4 代际追加；无别名新增（淘汰走 deprecation 而非 alias——镜像对语义差一负号，alias 机制不适用） | 保留机制一致性 |
| DATA_TIER_VERSION / SEARCH_CONTRACT_VERSION / PROTOCOL_VERSION / REWARD_VERSION / 执行与组合版本 | 不变 | 不变 | 无数据 tier、搜索协议、Reward、执行语义变化 | — |
| ARTIFACT_SCHEMA_VERSION | 2 | 不变（versions dict 新增键为 additive） | artifact 记录 FACTOR_COMPUTE_VERSION | additive 兼容 |
| （无版本项）train.py best-so-far 非有限值防御 | — | 缺陷修复 | NaN 奖励不再以 -inf 进入 identity 载荷；曲线以 reward 下限（bad_reward）起底并保持非递减 | 该路径此前必然 fail-closed 失败（CanonicalJSONError，无任何合法 artifact 存在），无兼容面；不 bump 版本，依据见修订附录 A |

## 10. 批准与边界

- **批准记录**：2026-09-01 由 AgentTeams alpha-orth-research captain 审阅并批准（审阅依据：t1/t2 证据链完整、版本影响表 §9 齐备、RED 测试逐条可测、裁决规则 §7 预注册、回滚 single revert 可行）。批准后实施由 t4 执行；实施前 RED 必须先红并留证。
- 本契约批准后，实施由后续任务执行；实施前 RED 必须先红并留证。
- 本契约全部测量结论的边界：dev/validation 数据（无已声明 regime/锁定 holdout），不构成任何 alpha/晋级结论。
- 本契约基于 t2 的测量证据（2015-01-05..20260821，dataset_id=a839ecf2…）；数据窗口变更需重测后再裁决。

## 附录 A：修订记录（前瞻性，禁止以实现输出反推契约）

- **修订 A1（2026-09-01，captain 批准）**：§5.3 稀疏安全标准化的必要后果——`train.py` best-so-far 非有限值防御。§5.3 按设计使部分验证奖励为合法 NaN（无事件日 IC 为 NaN），而 best-so-far 记录 `max(-inf, nan)` 保持 -inf，canonical identity 层（`CanonicalJSONError`）fail-closed 拒绝该路径的 artifact 写出。修复：NaN/±inf 奖励不得更新 best（`math.isfinite` 守卫）；曲线必须仍从 consumed budget 1 开始（搜索结果不变量）且非递减，故非有限 best 以 reward 下限 `bad_reward`（有限）起底。无版本 bump：该路径在修复前必然失败，不存在任何含 -inf best-so-far 的合法 artifact，无兼容面。
- **A1 的 RED/GREEN 证据**：RED = `tests/test_train.py::test_train_nan_val_reward_keeps_best_so_far_finite` 在还原 train.py 修复后失败（monkeypatch candidate_scorer 注入 val_reward=NaN，`save_artifacts=True` 触发 identity 路径）；GREEN = 修复在位后同测试通过（曲线有限、从 budget 1 开始、非递减），`test_train.py` 与 `test_core.py` 全量保持绿。
