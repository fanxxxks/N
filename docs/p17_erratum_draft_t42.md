# P17 勘误草案（t42：repurchase proc 映射 / industry_taxonomy.is_new / stk_limit 行上限核对）——待用户批准，未生效

- 性质：**勘误草案**。本文件只起草、不修改 p17 正文；批准后由获授权任务按
  p13 t46/t57 追认式勘误模式回写 p17（一次原子 commit）。
- 起草：governance-auditor（AgentTeams tushare-pit-migration t42），2026-09-04。
- 触发链：t6 评审（D1 验收，verdict=pass）D2 交接②——p17 DDL 与探针源列名
  不符，按 p17 §15.6（"来源字段/单位无法确认 → 停下修订本文，禁止猜测"）
  走勘误，禁止 D2 静默改名。
- 证据权威：p17 §5.0（探针留档输出为字段映射核对权威）；探针报告
  `D:\minequant\tushare_probe\probe_report_d1.txt`（2026-09-04T10:15:39，
  23/23 OK，repo 外留档，token 零出现——t6 已核）。本草案所有"探针实证"
  逐字引自该报告对应接口条目。
- 勘察基线：分支 codex/g0-tushare-pit-contract @ `78d4dac`（p17 文本 =
  00e276b 版本 + 6653cf7 仅头部批准行，正文与 00e276b 逐字一致）。

## 勘误项 1：repurchase_event_pit 的 proc_date（源列实为 proc）

- **p17 原文**（§5.4，表 16 DDL 行）：
  `end_date VARCHAR, proc_date VARCHAR, exp_date VARCHAR,`
- **探针实证**：`[repurchase] columns (9): ts_code, ann_date, end_date, proc,
  exp_date, vol, amount, high_limit, low_limit`——源列名为 **proc**，无
  `proc_date`。
- **建议改法（双轨）**：
  - (A) 机制（建议永久采纳）：p17 增设附录 B「源列→契约列显式映射表」，并
    在 §5.0 补一句常设规则——今后任何源列名与契约列名不一致，必须经勘误在
    映射表落盘，D2 按映射实施，**禁止静默改名/跳列**（§15.6 的机制化）。
  - (B) 本批处置：契约列 `proc_date` **改名为源名 `proc`**。理由：§5.0 L1
    纯源字段原则（L1 逐字存源列名）；`proc_date` 名不副实——官方文档口径
    `proc` 为实施进度（文本），非日期；其取值语义在 D3 首批真实拉取时以
    样本值落档测量日志核验。D2 影响：DDL 一列名；客户端 INTERFACE_FIELDS
    已含 `proc`（零改动）；写入器断言与 RED 用例同步更名。
- **备选（不推荐）**：仅落 (A) 映射表并登记 `proc → proc_date`，契约列名
  不变——L1 列名与源名永久偏离，违背 §5.0 精神。
- 复核：`proc_date` 在 p17 全文仅此一处（grep 证实），改名无其他连带位置。

## 勘误项 2：industry_taxonomy 的 is_new NOT NULL（index_classify 源无此列）

- **p17 原文**（三处）：
  - §5.2 注释：「-- 7. 申万行业分类（index_classify；is_new=Y 为现行代，
    历史代并存可查）」；
  - §5.2 DDL 列行：`industry_code VARCHAR NOT NULL, is_new VARCHAR NOT
    NULL, src VARCHAR,`；
  - §14.3：「industry_taxonomy 保留各代分类（is_new 标志 + index_code
    代际），2021 版分类切换显式记录 taxonomy 代际」。
- **探针实证**：`[index_classify] columns (7): index_code, industry_name,
  level, industry_code, is_pub, parent_code, src`——**无 is_new**；
  `is_new` 仅存在于 `[index_member_all] columns (11): … in_date, out_date,
  is_new`。
- **建议改法：移除 industry_taxonomy.is_new 列**，连带两处 conforming
  修正——§5.2 注释改为「（index_classify；源列含 is_pub/parent_code，
  代际判定见 §14.3）」；§14.3 改为「industry_taxonomy 保留各代分类
  （index_code 代际 + industry_membership_pit.is_new）」。
- **理由**：§5.0 L1 = 源层、逐字存源字段；is_new 在 index_classify 源中
  不存在，入库必须靠派生/回填 = 把 L2 派生语义塞进 L1，制造第二套代际
  判定路径（AGENTS §1 原则 3）；而成员表 is_new 是真实源字段，本就是
  代际判定的既有单一权威（p17 §14.5、D5 验证门均锚定它）。
- **备选（不推荐）**：改可空 + 由 industry_membership_pit.is_new 回填——
  需定义派生规则（如"该 index 存在任一 is_new=Y 成员"），属 L2 语义进
  L1，且回填耦合拉取顺序（成员表须先入库），D2/D3 复杂度净增。
- **D2 影响**：industry_taxonomy 源映射列 6→5（index_code, industry_name,
  level, industry_code, src；+provenance 两列不变），is_new NOT NULL 断言
  删除，无回填逻辑；§5.0 允许的 additive 源列 is_pub/parent_code 由 D2
  deliberate extension 自行决定是否追加；**D5 的 taxonomy 代际追溯语义
  零变化**（其权威本就在成员表 is_new + index_code）。
- **连带核对**：t6 handoff ① 载"index_classify 缺 src"——探针证实源侧
  **有** src（7 列含 src），属客户端 INTERFACE_FIELDS 子集扩展项（t6 已
  路由给 D2 deliberate reviewed edit），**不在本勘误范围**。

## 顺带核对项 3：stk_limit 行上限数值（5800?）——结论：无需勘误

- **核对方法与结果**：grep p17 全文 `5800|7800|行上限|单次.*上限`——
  **无任何 5800/7800 命中**；§13 分页规则以"单次返回行数 == 接口上限"
  为触发条件（`hit_row_limit` 信号），未钉死任何数值型行上限；附录 A
  stk_limit 行记载的是权限档位（2000 积分）与官方更新时刻（交易日 9 点），
  非行上限数值。
- **事实记录**：探针 `[stk_limit] rows: 7781`（单日全市场截面
  20260904，needs_pagination: False）→ 真实单次上限 ≥ 7781；客户端常量
  7800（t6 论证：低估仅提前分页、方向性安全；高估才可能漏截断）。
- **结论**：p17 未钉死过时数值，**无需勘误**。可选增补（非勘误，captain
  决定是否并入批准批次）：§13 或附录 A 补一句"行上限以探针实测为准
  （首测 7781，客户端常量 7800，低估方向性安全）"。

## 范围外新增发现（不进本勘误表决，交 captain 分流；均给出 probe 证据）

起草核对中发现以下 p17 DDL vs 探针源列差异，与两勘误项同类但 **t6 未列入
②清单**；是否并入同一勘误表决由 captain 决定（一次收敛 vs 分项处理）：

1. **daily_price_limit.pre_close**：p17 §5.2 DDL 含 `pre_close DOUBLE`；
   探针 `[stk_limit] columns (4): trade_date, ts_code, up_limit,
   down_limit`——源无 pre_close。建议：删除该列（同项 2 的 L1 纯源原则）。
2. **margin_detail_pit.rqche**：p17 §5.2 DDL `rqche DOUBLE`；探针
   `[margin_detail] columns (10): trade_date, ts_code, rzye, rqye, rzmre,
   rqyl, rzche, rqchl, rqmcl, rzrqye`——源列为 **rqchl**。建议：改名
   `rqche` → `rqchl`（或经附录 B 映射表登记）。
3. **daily_suspension 结构**：p17 §5.2 DDL 主键 (ts_code, suspend_date,
   suspend_type) 并含 resume_date/ann_date；探针 `[suspend_d] columns (4):
   ts_code, trade_date, suspend_timing, suspend_type`——源无 resume_date/
   ann_date（t6 handoff ① 已列）、suspend_date vs 源 trade_date 命名不一、
   suspend_timing 为 §5.0 可追加的 additive 列。此项涉及 DDL 结构性修正
   （主键与 available_date 推导），重于改名，建议单独勘误项或待 D2 提出
   deliberate extension 后补勘误，由 captain 分流。
- 客户端 INTERFACE_FIELDS 子集扩展项（margin_detail 五列、income.oper_cost、
  cashflow.c_cash_equ_end_period、index_classify.src；suspend_d 同）按 t6
  handoff ① 由 D2 deliberate reviewed edit 处理，不在本草案（探针证实源侧
  均提供这些列）。

## 批准与执行预案（批准后由获授权任务执行，本草案不预写结果）

- 回写模式：p13 t46/t57 追认式——p17 文末新增「勘误记录」节（引用本草案 +
  用户批准原文逐字 + 日期 + 表决范围）+ 原位修正（§5.2/§5.4/§14.3 的 DDL
  行与措辞），一次原子 commit；CONTRACTS.md 索引不动（主题行已涵盖）。
- **表决范围由 captain 定**：最小 = 项 1 + 项 2；可扩展 = 并入范围外
  1–3 项一次收敛。本草案对每一项均给出独立建议改法与理由，支持任意组合
  表决；未获批准的项目保持 p17 原文并由 D2 按 §15.6 停止条件处理。
- **未生效声明**：本草案未修改 p17 任何正文；批准前 p17 现行文本继续有效，
  D2 遇及上述列位时按 §15.6 停下等待本勘误结果，禁止静默改名。
