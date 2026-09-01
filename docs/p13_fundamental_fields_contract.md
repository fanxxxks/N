# P13 fundamental PIT 字段补齐与第⑤族解锁预注册契约（C 线）

- 状态：**DRAFT — 提交契约门禁评审（t10）**。批准后本文保持前瞻性，禁止用
  实现或测量输出反向修改；测量结果只能执行预注册裁决。
- 起草：contract-a（AgentTeams alphagpt-p0-p1 t7），2026-09-02
- 证据基线：main @ `fde9f8b`；`docs/p9_factor_family_contract.md` §6（第⑤族
  pending 占位与**预注册解除条件**——本文即该条款所指"另行预注册实现契约"）；
  `docs/p9_factor_family_contract.md` §7/§8（族级裁决模式与小批量纪律）；
  `docs/p9_measurement_log.md`（裁决执行先例）；`ashare_data/fundamentals.py`
  （PIT 管线与 announce 语义）、`ashare_data/db.py` L111-131/L372-413
  （`fundamental_pit` schema 与 COALESCE upsert）、`ashare_data/gates.py`
  （数据门 G1–G7）、`ashare_model/feature_metadata.py` L679-717
  （PENDING_DATA_FEATURES 4 项）、`ashare_model/data_tier.py` L76-81、
  `ashare_model/research_domain.py` L243-254。纯文档任务，计划/证据分离
  （AGENTS §11.1）。
- 任务路由：**研究语义变更**（AGENTS §3.2：特征定义、可用时间、数据 tier）
  + **数据/股票池门禁**（§5）。仲裁顺序：本文/批准 → 契约测试 → 实现 →
  测量日志。

## 1. 问题陈述

1. **第⑤族被数据缺口阻塞（p9 预注册事实）**：P9 契约 §6 将 PIT 现金流质量/
   应计/资产增长/盈利加速度 4 个特征登记为 `pending_data` 占位
   （`feature_metadata.py` L696-717：CASHFLOW_QUALITY、ACCRUALS、
   ASSET_GROWTH、EARNINGS_ACCEL，均 `promotion_allowed=False`、不进词表、
   不耗预算），原因：`fundamental_pit` schema（db.py L111-131）无现金流量表
   字段（经营现金流）与总资产字段；并**预注册了解除条件**——"数据侧完成
   现金流量表与总资产字段的 PIT 同步（含来源、announce_date、覆盖率 ≥0.9
   的验收门）后，另行预注册实现契约"。当前词表 73 名中该族经济信号
   （现金流确认、应计、资产增长、盈利加速度）零表达，且为保持族原子性
   整族 pending（p9 §6 明文"禁止零散拼凑"——EARNINGS_ACCEL 虽可由既有
   `profit_cum` TTM 逼近，亦不得单独先行解锁）。
2. **数据可达性已由既有路径证明**：`fundamental_pit` 的主来源
   （Eastmoney 业绩报表，`get_earnings_report`）是**全市场按报告期批量**
   端点，覆盖远高于逐股补充端点（t1/t2 实证 DIVIDEND_YIELD/ROA/DEBT_RATIO
   仅 ~0.25-0.26，即逐股补充路径覆盖不足的对照证据）。现金流量表与资产
   负债表存在同构的全市场按报告期批量来源（东财现金流量表/资产负债表按
   报告期），两字段（`net_operate_cash_flow`、`total_assets`）可经镜像
   earnings 路径补齐。
3. **范围拆分（A/B 线零交集）**：本线 = 纯数据侧（schema/同步/特征帧/
   注册表解锁）+ 数据回填；与 A 线（`reward.py`/config/candidates）和
   B 线（`promotion.py`/feature_registry 消费侧新增）**零语义交集**
   （B 线对 `feature_registry.py` 仅新增晋级消费函数，C 线仅追加特征数据
   与元数据——两线对该文件的改动为不同函数区域，集成时按 §12 串行合入
   由 integrator 保证不冲突）。

## 2. 假设

- **H1（批量覆盖假设）**：全市场按报告期批量端点对 `net_operate_cash_flow`
  与 `total_assets` 的历史覆盖 ≥ 0.9（p9 §6 预注册验收门），显著高于逐股
  补充路径（§1.2 对照证据）。
- **H2（增量信息假设）**：第⑤族 4 信号在既有 73 名词表之外携带增量 OOS
  信息（各自 pending 声明的经济假设：现金流确认盈利 → 持续超额；应计
  驱动盈利 → 反转；激进资产扩张 → 跑输；盈利加速度 → 动量延续）。若
  裁决否定，负结果合法（§8）。
- **H3（PIT 机制复用假设）**：既有的法定披露季末 announce 锚定 +
  `_ffill_from_announcements` + `_ttm` 单一路径足以承载新字段的
  point-in-time 语义，无需新 PIT 机制。

## 3. 范围与非目标

### 3.1 代码侧（t14，impl-c-data，任务分支，零 DB 写入）

1. `ashare_data/db.py`：`fundamental_pit` schema 追加两列
   `net_operate_cash_flow DOUBLE`、`total_assets DOUBLE`（additive 可空，
   幂等迁移，§5.1）；`upsert_fundamentals` 列清单同步扩展（COALESCE 合并
   语义原样，L396-412）。
2. `ashare_data/akshare_client.py`：新增两个批量方法（现金流量表、资产
   负债表，按报告期、全市场），镜像 `get_earnings_report` 的形状与
   universe 过滤约定（§5.2）。
3. `ashare_data/fundamentals.py`：同步管线扩展（新字段入 `_EARNINGS_COLUMNS`
   同级的批量写入路径 + announce master join）；`build_pit_frames` 扩展
   4 个新特征帧（全部经 `_single_periods`/`_ttm`/`_ffill_from_announcements`
   既有单一路径，§5.3）。
4. `ashare_model/feature_metadata.py`：4 项 PENDING_DATA_FEATURES 占位移除，
   转正为权威元数据（availability_rule 更新为真实规则；`promotion_allowed`
   authored=True，§5.3）。
5. `ashare_model/vocab.py` / `feature_registry.py` / `research_domain.py` /
   `factors.py`（如注册表需登记 depends_on）：词表 73→77 追加（token id
   不移位）、registry 记录、slow_fundamental 域归属（§5.5）。
6. `docs/feature_registry.md` 同 commit 再生成（drift guard）；测试 §7。

### 3.2 数据侧（t20，impl-c-data，独占 DB 窗口，串行于 P1-5 之后）

同步/回填两字段（含 availability timestamp）→ 复核 → 门禁 → evidence-only
测量记录 `docs/p13_fundamental_backfill_measurement_log.md`（§3.4、§5.6）。
本任务零代码改动（代码已由 t14 交付）。

### 3.3 非目标

- 不新增算子；不改 daily-bar 管线、universe/PIT mask、`ashare_data/gates.py`
  （G1–G7 数据门权威零改动——覆盖率验收门是**本契约的验收步骤**，不进
  gates.py）；不改 reward/promotion/搜索器/预算/seed（A/B 线辖区）；
- 不逐股猜测披露日期（禁止 per-stock announce 猜测路径，§5.6）；
- 不以当前快照回填历史（AGENTS §5.2）；
- 不做 paper/sim、lifecycle 转换；不动 P1-5 的 daily_bar 写入与
  t12 产物；
- 不编辑历史审计脚本（`docs/factor_inventory_audit_*/audit_run*.py` 冻结，
  裁决重测用新脚本，§8）；
- 族⑤原子性：禁止部分解锁（EARNINGS_ACCEL 不得脱离整族先行，§1.1）。

### 3.4 硬性门禁三步总纲（AGENTS §5.4/§5.6；DB 写入单点串行）

1. **先只读审计**：精确回填范围 = 股票（in-scope universe）× 报告期 × 字段；
   审计现表 schema、既有列覆盖、两字段缺口、来源端点抽样核验（含累计/
   单期口径确认）；产出只读审计记录（t14 交付）。
2. **再执行**：DB 写入只在**独占窗口**进行——全队唯一 DB 写入者
   impl-c-data 单点串行，且**必须排在 P1-5 daily_bar 同步（t12）完成之后**
   （本团队计划批准即视为用户对该写入窗口的授权）；执行前确认无其他正式
   运行占用数据库。
3. **后复核**：row count（按字段/按报告期）、日期范围、dataset_id（如实
   记录变更，§6.2）、`G1–G7` 数据资格门禁全记录
   （`ashare_data/gates.py`，CLI `python scripts/check_production_gates.py`；
   **禁止 --dev 结果冒充正式**）+ `python -m ashare_model.research_doctor`
   重跑并记录。

## 4. 不变量

1. **PIT 单一路径**：新字段可见性只经 announce master（法定披露季末锚定）
   + `_ffill_from_announcements`；TTM 只经 `_ttm`；禁止第二套 PIT 填充
   （AGENTS §1.3）。
2. **缺数即 NaN，绝不伪造中性值**（AGENTS §5.3）：字段缺失/报告期断档 →
   特征 NaN → 既有质量门（min_activity/min_valid_ic_days 等）拒绝；t14 与
   t20 之间的过渡态（特征在词表但数据未回填）必须保持该 fail-closed 行为。
3. **upsert 合并不清空**：COALESCE(EXCLUDED.x, existing) 语义原样——回填
   只能新增/更新两新列，不得清空或改写任何既有字段的历史值。
4. **词表追加不移位**：新 token 追加到词表尾部，历史公式按名/按 id 解析
   逐位不变（p9 Part A 先例）；legacy artifact（grammar v5 时代）保持
   可解析、可执行、只读有效。
5. **族⑤原子性**：4 特征同 commit 进词表、同批裁决、同进退。
6. **schema 迁移幂等**：重复执行无二次效果；既有行存活且新列为 NULL 直到
   回填。

## 5. 方案（预注册设计）

### 5.1 schema 迁移

`fundamental_pit` 追加 `net_operate_cash_flow DOUBLE`、`total_assets DOUBLE`
（可空、additive）。`CREATE TABLE IF NOT EXISTS` 对既有表不加列，故迁移经
幂等 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`（DuckDB 支持）挂接
create_schema/迁移路径；旧读取方零影响；不设新 DB schema 版本常量（additive
可空列，MANIFEST 表清单不变；manifest 版本所有者 DATA_TIER/MANIFEST 均不
bump，§6.1）。回滚：代码 revert 后列为惰性残留（无读取方，无害），无需
DB 回滚。

### 5.2 同步/回填机制（镜像 earnings 批量路径）

- 批量端点按报告期取全市场行 → `ts_code ∈ universe` 过滤（P2-01 scope
  契约原样）→ 与 earnings 表的 announce master 按 `(ts_code, report_date)`
  join → **无 master 匹配的行丢弃，绝不猜测披露日期**（Sina 补充先例）。
- 每报告期一行 upsert（COALESCE 合并）；缓存目录约定与 `_write_cached`
  原样；offline 模式整体跳过（既有语义）。
- **口径确认前置**：首次真实抓取时在只读审计中确认来源为**年初至今累计值**
  （与 `profit_cum` 同约定，经 `_single_periods` 转单期）；若来源实为单期
  值，在实现 commit 内以显式算术转换并记录——两种情况的 TTM 均走 `_ttm`
  单一路径。口径含糊即停（§9.2），禁止猜测。

### 5.3 四特征定义（预注册公式；全部 slow_fundamental 域）

TTM 记号：`TTM(x)_t` = 截至 t 报告期的滚动四季和（`_ttm` 路径，报告期断档
→ NaN）。可见性一律 = 分母/分子各自 announce 日期的**较晚者**（同报告期
对齐 join，禁止错期拼接）。

| 特征 | 预注册公式 | NaN 守卫 | 方向（预注册，按测量裁决） |
|---|---|---|---|
| CASHFLOW_QUALITY | `TTM(net_operate_cash_flow)_t / TTM(profit_cum)_t` | TTM 不完整或分母 ≤ 0 → NaN | +1 |
| ACCRUALS | `(TTM(profit_cum)_t − TTM(net_operate_cash_flow)_t) / total_assets_t` | TTM 不完整、total_assets 缺失或 ≤ 0 → NaN | −1 |
| ASSET_GROWTH | `total_assets_t / total_assets_{t−4 报告期} − 1` | 任一端缺失或 ≤ 0 → NaN | −1 |
| EARNINGS_ACCEL | `g_t − g_{t−1}`，`g_t = TTM(profit_cum)_t / TTM(profit_cum)_{t−4} − 1` | 分母 ≤ 0 或 TTM 断档 → NaN | +1 |

- 元数据转正：`promotion_allowed` authored=True（族级裁决后按 §8 结果
  维护，与 P9 Part B 家族同模式）；availability_rule 由 pending 占位文本
  更新为真实规则（"季度报告，法定披露季末后可见，TTM 需四季完整"）；
  `depends_on` 登记两新字段。
- PENDING_DATA_FEATURES：4 项移除；`PendingDataFeature` 机制保留（其
  "文档化占位"模式被 `tests/test_feature_metadata.py` 既有测试消费，非
  死代码，§9.1 依据记录于实现报告）。

### 5.4 覆盖率验收门（p9 §6 预注册条件的落地，fail-closed）

回填复核时逐字段测量：in-scope universe × 已同步报告期中该字段有限值占比
≥ **0.9**（p9 §6 预注册值，不得放宽）。任一字段不达 → 停止、保留证据、
上报 captain；**该族不解锁、词表不动**（t14 代码可先行合入但特征帧为
NaN，由不变量 2 保证无效果）。覆盖率门是本契约验收步骤，不修改
`ashare_data/gates.py`。

### 5.5 域归属与数据层级后果

- 4 特征归 `slow_fundamental`（`research_domain.py` L243-254 既有域，
  every_20_days/monthly 执行点）；RESEARCH_DOMAIN_VERSION 2→3。
- 新字段 PIT 等级 = PIT_FUNDAMENTAL → **Tier B**（`data_tier.py` L77）：
  默认 Tier-A-only 晋级政策下 G6 不放行，仅可经 P2-03 明文的
  `--allow-tier-b` 独立对比路径——这是数据事实而非缺陷，如实预注册；
  DATA_TIER_VERSION 不变（映射既有）。

### 5.6 PIT/availability 纪律与 DB 写入纪律

- **availability 时间戳**：新字段一律使用 earnings announce master 的法定
  披露季末锚定（Q1→04-30、H1→08-31、Q3→10-31、年报→次年 04-30，
  `fundamentals.py` 模块 docstring 既有约定）；endpoint 自带的公告列
  与任何逐股披露源**不得**用作可见日期（保守、永不提前可见，既有设计
  原样延伸）。
- **当前快照不得回填历史**（AGENTS §5.2）：历史值只能来自历史报告行；
  禁止以当前报表快照外推历史期间。
- **DB 写入纪律**：独占窗口（§3.4）；写前只读审计 + 窗口确认；写后复核
  row count/日期范围/dataset_id/G1–G7 全记录 + research_doctor 重跑；
  全部证据进 `docs/p13_fundamental_backfill_measurement_log.md`
  （evidence-only，引用命令、环境、被测代码 SHA = t14 commit）。

## 6. 版本影响总表与迁移/拒绝策略

### 6.1 版本影响表（`git grep -nE "^[A-Z][A-Z0-9_]*_VERSION" -- "*.py"` 全量检索，main @ fde9f8b 基线 33 命中 = 31 赋值所有者 + 2 docstring；下表列相关项，其余以 p11 契约 §6.1 全量清单为准，此处不复制第二份）

| 版本所有者 | 旧值 | 新值 | bump 理由 |
|---|---|---|---|
| ashare_model/feature_registry.py:57 FEATURE_REGISTRY_VERSION | 5 | **6** | 4 特征元数据转正 + pending 占位移除（p9 先例：3→4 同类） |
| ashare_model/vocab.py:226 GRAMMAR_VERSION | 5 | **6** | 词表 73→77，4 token 进入采样空间（追加、id 不移位；p9 先例 3→4） |
| ashare_model/research_domain.py:39 RESEARCH_DOMAIN_VERSION | 2 | **3** | 4 特征归属 slow_fundamental，域特征清单变更（p9 先例 1→2） |
| ashare_model/factors.py:46 FACTOR_COMPUTE_VERSION | 1 | 不变 | bar 张量计算零改动（新特征走 PIT 帧管线；p9 中仅张量标准化语义才 bump） |
| ashare_model/data_tier.py:37 DATA_TIER_VERSION | 1 | 不变 | PIT_FUNDAMENTAL→B 为既有映射，零改动 |
| ashare_data/manifest.py:54 MANIFEST_VERSION | 1 | 不变 | manifest 表清单不变；dataset_id 随数据内容变化**如实重算**（数据事实，非版本事件，§6.2） |
| ashare_model/evaluation.py:287 PROTOCOL_VERSION | "25" | 不变 | 评价协议零改动 |
| ashare_model/reward.py:141 REWARD_VERSION | "14" | 不变（本契约范围） | A 线 p11 独立负责 14→15；C 线实现分支集成最新 main 后随其生效 |
| ashare_model/artifact_schemas.py:44 ARTIFACT_SCHEMA_VERSION | 2 | 不变 | versions dict 新键（grammar 6/registry 6/domain 3）为 additive（p9 §9 先例） |
| SEARCH_CONTRACT / SEARCHER_BENCH / 其余 22 项 | — | 不变 | 与特征数据补齐无语义交集 |

### 6.2 迁移/拒绝策略

- **dataset_id 演进**：回填改变数据库内容 → manifest 重建 → 新 dataset_id
  （与 P1-5 的变化叠加，C 线审计基线 = post-P1-5 状态）。新旧 dataset_id
  如实记录于测量日志；跨 dataset_id 的测量禁止拼接结论（AGENTS §4.3）；
  族裁决测量（§8）只引用 post-backfill dataset_id。
- **legacy 兼容**：词表追加不移位 → v5 时代 artifact 公式解析/执行逐位
  不变（不变量 4）；无任何产物失效，无需迁移。
- **DB 迁移**：幂等 additive；回滚 = 代码单 revert（§5.1），无数据删除、
  无历史改写。
- **拒绝（fail-closed）**：覆盖率门不过 → 族不解锁（§5.4）；来源口径含糊
  → 停止修订契约（§9.2）；gates 红 → 不放行（§3.4）。

## 7. 预期 RED 测试清单（实现前先红；t14 落地，同 commit 注册分片）

1. **RED-1 schema 迁移**：迁移后两新列存在且类型 DOUBLE；重复执行幂等；
   既有行存活、新列 NULL（additive）。
2. **RED-2 批量同步路径**：universe 过滤（scope 外代码不入表）；无 announce
   master 匹配的行被丢弃（负例：注入无匹配行断言零写入——fail-closed，
   禁止猜测披露日期）。
3. **RED-3 PIT 因果属性测试**：合成数据上，t 日特征值只依赖 announce ≤ t
   的报告；未来报告对历史日期不可见（泄漏捕获）。
4. **RED-4 特征公式 golden**：合成 fundamentals 上四公式逐值断言（§5.3 表），
   含 NaN 传播（报告期断档）、分母守卫（profit TTM ≤ 0 / total_assets ≤ 0）、
   ACCRUALS 与 EARNINGS_ACCEL 的报告期对齐。
5. **RED-5 注册表解锁**：4 名进 FEATURE_NAMES（追加、既有 token id 零移位
   ——历史公式往返 parity 守卫）；PENDING_DATA_FEATURES 4 项移除；元数据
   完整且 availability_rule 非 pending；`promotion_allowed=True`；与
   deprecated 一致性守卫不触发。
6. **RED-6 域归属**：4 特征 resolve 为 slow_fundamental；RESEARCH_DOMAIN 3。
7. **RED-7 过渡态 fail-closed**：数据未回填状态下新特征帧全 NaN → 质量门
   拒绝（断言无中性值伪造、无 artifact 可能产出）。
8. **RED-8 版本 pin 与 drift**：FEATURE_REGISTRY 6 / GRAMMAR 6 /
   RESEARCH_DOMAIN 3 pin；`docs/feature_registry.md` 与注册表同 commit
   再生成（drift guard 绿）。
9. **既有测试更新（§10.1 白名单情形 2，引用本契约 §5/§6）**：词表计数类
   断言（73/61/58 → 77/65/62）、`tests/test_feature_metadata.py` 的
   pending 断言、`tests/test_vocab.py`/`tests/test_research_domain.py`
   相应期望；断言强度不得降低。历史审计脚本冻结不改（§3.3）。

## 8. 族级裁决（P9 §7 模式原样，预注册）

- **重测口径**：与 t2/P9 完全一致（h∈{1,2,3,5,10,15,20}、min_stocks=10、
  IS/OOS = 2022 分界、同窗口、同成本模型），对 v6 词表全量（77 名）重测；
  使用**新审计脚本**（audit_run_v5 模式，77 名断言），历史脚本冻结；
  引用被测 commit 与 post-backfill dataset_id。
- **族门（第⑤族整族原子裁决）**：以 t2 的 7 信号等权基准复合为基线 +
  族内消融，族内至少 1 个特征 ΔIC_OOS ≥ **+0.005** 且残差 IC 符号与
  §5.3 预注册方向一致 → 族通过（成员维持 promotion_allowed=True）；
  否则**负结果**（合法结论）：4 特征保留计算与采样、
  promotion_allowed=False（族③同模式，p9 测量日志 §3 先例）。
- **相关性二次精简**：新特征与既有特征或族内 |ρ| ≥ 0.9 → 触发 Part A
  机制的条件弃用裁决，记录于测量日志。
- **小批量纪律**：本契约一次批量 = 2 个 DB 字段 + 4 个特征，不追加；
  任何追加需新契约（p9 §8 同款）。
- **裁决记录**：evidence-only 写入
  `docs/p13_fundamental_backfill_measurement_log.md`（与回填证据同文件、
  独立章节），只记实际发生；禁止预写。

## 9. 资源上限与停止条件

### 9.1 资源上限

- 回填：批量按报告期拉取（约 45+ 季度 × 2 端点，分钟级）；DB 窗口独占、
  不与任何正式运行并发；
- 族裁决重测：单次 ≤ 30 分钟（p9 §8 先例 73 名实测 1186s；77 名同量级），
  超限先优化测量而非缩窗口；
- 不消耗任何搜索预算（特征进词表前后的搜索预算口径不变）。

### 9.2 停止条件（任一命中即停，保留证据，上报 captain）

1. 覆盖率门不过（< 0.9，§5.4）→ 族不解锁，数据缺口如实记录；
2. 来源口径含糊（累计/单期无法确认）→ 禁止猜测，停下修订本契约；
3. 任一 RED 无法稳定失败 → 契约不可测，回到本契约修订；
4. 回填后 G1–G7 或 research_doctor 红 → 不放行，带证据上报；
5. 族裁决负结果 → 合法结论（§8），不是停止条件而是裁决结果；
6. 与 P1-5 窗口调度冲突 → 串行等待，禁止并发写入。

## 10. 批准与实施边界

- 批准流程：经契约门禁评审（t10）通过后生效；批准记录补记于本节。
- 实施顺序（§2.2/§12）：契约 PR（本文）先行合入 main → t14（代码侧，
  分支 codex/p13-fundamental-fields，RED→最小实现→GREEN→全量回归→
  原子提交）→ 独立审核（t23 代码面）→ t20（数据回填，独占窗口，串行于
  t12/P1-5 之后）→ 族裁决重测（§8）→ t23 整合审核 → 集成合入（t28，
  串行小 PR）。
- 文件所有权：C 线代码侧唯一改动面 = §3.1 所列文件；与 A 线
  （reward/config/candidates）、B 线（promotion.py）零交集；
  `feature_registry.py`/`feature_metadata.py` 与 B 线的改动为不同函数
  区域，集成顺序由 integrator 串行保证。
- 本契约不授权 push/远程 PR；全程仅本地、禁止 push；DB 写入授权 =
  团队计划批准时的预授权（§3.4），窗口纪律如 §5.6。
- 未运行/未验证项如实列出；本文不预写任何测量结果（§11.1）。
