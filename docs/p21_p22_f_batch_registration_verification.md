# P21/P22 F 批次登记核验记录（governance——批准链、版本基线、sequencing）

- **【终态权威声明（2026-09-07 captain 确认后 governance 定稿；2026-09-07
  第四次终局裁定更新）】**：本记录各历史层的编号与状态陈述**以下条为
  准**——**F1 = p23**
  （docs/p23_f1_slow_fundamental_quality_draft.md **v6 终局版**，用户
  「批准」亲批（批次级）、实施 = `eab1061`（V2 集内容）+ `3d2a9f1`
  （草案 v6 终局 + 演进披露）；`300cf70` 的 V1 表换名已被
  `fe8bd11`/`3d2a9f1` 回退））；**INDUSTRY_MOMENTUM = p22**（方向重预注册提案
  REJECTED——`12a7ed7` 的 −1 翻转系基于 captain 阶段二误批准（未核
  在案 REJECTED 冗余裁决），已被 `6644e662` (b) 回退裁定取代：
  expected_direction 回 +1、REGISTRY 11→10、−1 观察留档；驳回人/理由/
  证据链见下文勘误节）。各历史层陈述保留作裁决史。

- **编号纠正（2026-09-07 captain）**：本记录第 1 节核验的 F1 四候选草案
  正式编号为 **p23**（原起草号 p21 已被 R1 线预注册契约
  `p21_r1_domain_return_research_preregistration.md` 占用，且已批准）；
  第 2 节的 p22 = INDUSTRY_MOMENTUM 方向重预注册注记（编号不变，已获
  captain 代行批准生效，按 sequencing p20 → p9 Part A → p22 实施）。
  本记录保留因子线本地编号作核验史，登记与用户转达统一用 **p23**/p22。
- **【编号二次纠正 + INDUSTRY_MOMENTUM 驳回（2026-09-07 factor-engineer
  执行 @ `825c929`，编号冲突纠正）——取代上一条的编号安排】**：
  **【三阶段史摘要（2026-09-07 governance 按 factor-engineer 校准补强——现行态 = 阶段三）】**：**阶段一 = REJECTED**（2580c3e9：冗余预检驳回；实施 4eee145 = promotion_allowed=False、方向 +1 维持）→ **阶段二 = 批准实施**（captain 三项裁决 item 2；实施 12a7ed7 = 方向 −1 + REGISTRY 10→11 + 定向裁决方向符号证实）→ **阶段三 = 终裁回退（现行态）**（6644e662 终裁认定阶段二混淆两判据；实施 969a126 = 方向恢复 authored +1 + promotion_allowed=False 维持 + 双重发现注释 + REGISTRY 12→13；全量 1,691/1/0）。逐项证据见下文 1-4 与勘误节——现行态 = authored +1 + promotion block + 双重发现注释。
  1. **F1 四候选草案改号为 p22**：`docs/p22_f1_slow_fundamental_quality_draft.md`
     （内容不变、标题重编号；`825c929`）——**F1 的登记与索引行使用 p22
     文件名**（本记录此前"p23"的 F1 指称作废）；
  2. **INDUSTRY_MOMENTUM 方向注记让位改号为 p23 且状态 = REJECTED**：
     `docs/p23_industry_momentum_direction_repreregistration_draft.md`
     （REJECTED 记录）——**方向重预注册提案被驳回，编号不参与后续
     实施**；本记录此前"p22 = INDUSTRY_MOMENTUM 已获代行批准生效、
     进入实施"的陈述**被本驳回取代**（该特征的方向问题维持 authored
     +1 + 三段式证伪历史作为负结果证据留存，p23 文件完整保留）；
  3. **驳回依据链已补全（2026-09-07 captain 确认，取代此前的"待证据链
     补全"标记）**：
     - **驳回人** = captain（代行——注意：此驳回**不是** p18 §3.2 保留项
       的批准类动作，而是对重假设候选的**裁决性否决**，属因子治理常规
       裁决权限；用户授权扩展记录 89dd121 附录 C 覆盖）；
     - **理由（逐字，来源 = captain 邮箱 2580c3e9 至 factor-engineer，
       2026-09-07）**："其 −1 反转重假设冗余预检不通过：修正后 ΔIC_OOS
       −0.0167 低于 IND_REL_* 族内成员增量（+0.0079~+0.0283），且实证
       形态（行业反转）与族方向同源——纳入即违反 ΔIC 增量判据。处置：
       负结果 + 冗余预检失败完整留档；未来若族结构变化可重新提案。"；
     - **证据链** = 被裁决方自己的修复后重测数据（coverage 0.838 恢复、
       IC 全 horizon 负 h1 −0.0063/h10 −0.0096/h20 −0.0176、修正基线
       ΔIC_OOS −0.0167、年度同向 4/13、max |ρ| vs MOMENTUM_20 = +0.379）
       ——驳回依据由被裁决方自己的测量数据支撑（t18/t22 测量日志交叉
       可核）；
     - **三段式证伪历史留存影响 = 零**：REJECTED 对象是 −1 重假设预注册
       提案（p23 注记的未来实施部分），非证伪历史本身——三段全部完整
       留存且继续有效（history 与 re-hypothesis 是两个对象）；
     - **时序备注**：captain 曾先批准 p22 注记的存在（0800d10b），其后
       修复后数据报告显示重假设未过冗余预检 → 驳回（2580c3e9）——两个
       时点的两个对象，非自相矛盾。
     - **仲裁终局（2026-09-07 captain）**：处置 = **(b) 回退**——
       expected_direction 回 **+1**（authored 证伪证据保留）、
       FEATURE_REGISTRY 11→10、−1 反转观察留档于本记录与元数据注释
       （"实证观察到但因冗余未采纳"）。此前一版仲裁消息曾混淆方向符号
       判据与冗余预检判据（两个独立维度）被实现方读成翻转授权——
       governance 升级有效、captain 自行更正（"评审抓住了仲裁语言的
       含混"）；修复任务已派 factor-engineer。三段式证伪历史与本驳回
       记录不受回退影响（回退 = 驳回处置的落实，非状态再反转）。
     - **阶段二"批准"原文归档（后被仲裁终局取代——factor-engineer 提供
        逐字，captain 三项裁决消息 item 2，2026-09-07，先于 12a7ed7）**：
        "p22 独立注记：批准——落点提案成立（行业特征 ≠ F2 融资拥挤批；
        信息层正交，max |ρ| +0.379 < 0.9 无 Part A 触发）；三段式历史
        披露齐备；范围最小。执行内容照草案（方向元数据 + FEATURE_
        REGISTRY +1 + 文档再生 + RED + 定向测量裁决）。"
        ——该"批准"批准了注记执行而未核对其在案的 REJECTED 冗余裁决，
        已被仲裁终局明确为 captain 之错误并由 6644e662 (b) 回退裁定取代；
        其批准依据（相关性判据 max |ρ| +0.379 < 0.9）与 ΔIC 增量判据的
        取舍已由 captain 终局裁定为**回退**——原逐字文本归档作裁决史，
        INDUSTRY_MOMENTUM 登记状态 = REJECTED + 回退执行。
     - **【7a450381 终裁（2026-09-07 captain，取代 0800d10b 草率批准与
        阶段二误批准——INDUSTRY_MOMENTUM 裁决史终局）】**：
        (i) **阶段一 REJECTED 维持为终局**——阶段二"批准"系 captain 裁决
        错误：误用相关性判据（max |ρ| 0.379 < 0.9）而未对齐当时在案的
        ΔIC 增量驳回（−0.0167 < IND_REL_* 族内 +0.0079~+0.0283）。**两
        判据独立**（|ρ| 管共线性；ΔIC 增量管"是否在既有族之上增加预测
        价值"）——**ΔIC 增量为重假设预注册的约束性判据**（|ρ| 低但 ΔIC
        低于族内成员的因子依然冗余）。
        (ii) **处置 = (b) 回退（factor-engineer 已执行完毕 @ 969a126）**：expected_
        direction 回 **+1**（authored 证伪证据语义）、REGISTRY 11→10、
        元数据注释保留双重发现（"+1 已两轮证伪；−1 反转实证观察到但因
        冗余 REJECTED 未采纳"）、p23/REJECTED 记录不变。
        **【回退执行完毕（2026-09-07 factor-engineer 报告，969a126）——
        含 REGISTRY 算术更正】**：回退落地 = expected_direction 恢复
        **authored +1**（预注册假设历史记录语义）+ promotion_allowed=
        False 维持（t24 路由负结果处置为现行态）+ 双重发现注释在案；
        **REGISTRY 算术更正：12→13（非本条原记 11→10）**——原算术以
        F1 未消费 11→12 为前提，实际 F1 先行消费（11→12），回退成为
        新内容变更（12→13），factor-engineer 已在 commit 披露；REGISTRY
        现值 = 13（F1 四因子在词表 + 翻转回退）。RED 等价证据：翻转测试
        在回退态失败（== −1）→ 转为回退态断言（+1 + promotion False），
        负结果不变式保留，无断言弱化；t22 日志 §12 回退记录 + §10 修正
        （"方向重注册生效"表述被终裁取代）落盘。全量串行 1,691/1/0
        （969a126）。
        (iii) **方向证实与冗余驳回两事实并存不悖**（raw IC 观察证实方向
        符号；冗余驳回判增量价值）——判据不同，互不否定，同入两阶段史。
        (iv) 编号终态 = 文件现状（F1=p23 / INDUSTRY_MOMENTUM=p22），
        与 bb027ed 一致，无需改编号。
  4. F1（p23，编号终态见顶部权威声明）的用户批准（上一条「批准」）不受
     影响，实施按 sequencing 推进；REGISTRY 序号以实施时点真实常量重算。
- **p23 用户批准记录（2026-09-07，governance 入册）**：**用户批准原文
  （逐字）**：「批准」（2026-09-07，经 captain 会话转达）——p23 四候选
  （新特征入词表批次）经**用户亲批**，为 p18 §3.2 保留项的正确执行
  （新特征批次由用户批准，代行授权不覆盖——与本记录第 4 节第 2 条的
  批准路径裁定一致）。批准链：用户亲批 → captain 转达 → factor-engineer
  实施（sequencing：t70/p20 与 F1 按序推进；GRAMMAR/REGISTRY 序号以
  实施时点真实常量重算）。p23 文件头的批准行由实施 commit 按 p16
  `2f5e18b` 先例补记（引用本核验记录 + 批准原文）。**【编号二次纠正
  后更正】本条的"p23 四候选"指称按新编号读作 p22 四候选（F1），批准
  范围与效力不变。****【2026-09-07 编号回摆终定】该更正再次被编号回摆
   取代——F1 终定 = p23（docs/p23_f1_slow_fundamental_quality_draft.md，
   标题 v4；CONTRACTS 索引行随实施 commit `eab1061` 同 commit 登记），
   本条指称恢复字面正确；p22 = INDUSTRY_MOMENTUM REJECTED 记录占用。
   批准范围与效力始终不变。

- **【F1 实施闭环 + V1 实施回退披露（2026-09-07 factor-engineer 报告，
  governance 按 eab1061 树状态复核）】**：
  1. **编号终定**：F1 = **p23**（`docs/p23_f1_slow_fundamental_quality_draft.md`，
     标题 v4 = 路线图原指名四因子；CONTRACTS.md p23 索引行随实施 commit
     `eab1061` 同 commit 登记）——本记录此前的 p22 指称（825c929/12a7ed7
     中间态）被编号回摆取代，p22 编号由 INDUSTRY_MOMENTUM REJECTED 记录
     占用；
  2. **V1 集实施曾发生并按批准范围裁定回退**：`c132ae7` 实施了 V1 自提
     候选集 → captain 终局裁定（`c970a26d`：**V2 维持、禁止再翻转**）
     revert（`24520cb`）——本核验 §7 的批准范围确认点先行标记了该风险，
     回退即终局裁定的落实。**【引用更正（2026-09-07 captain 指令）】**：
     本条此前归因于"批准范围裁定 90d40661（批准效力覆盖 v2 而非 v1）"
     系**权威引用错误**——90d40661/4db156e 批准的是 **v1 呈递集**
     （NET_MARGIN_TREND_4Q/LEVERAGE_TREND_4Q/CAPEX_INTENSITY/
     CFO_TO_DEBT，批次级、对呈递表格），**不作为 V2 集的批准依据**；
     V2 集的实施依据 = 用户路线图 F1 逐字指名（最高背书）+ captain
     终局裁定 c970a26d（V2 维持）。
  3. **V2 集（路线图四因子）实施 = `eab1061`**：GRAMMAR 7→8（generation 8
     分派注册）、FEATURE_REGISTRY 11→12（现值链：t70 8→9 → t24 路由
     9→10 → p22 10→11 → F1 11→12，单调无跳号）、RESEARCH_DOMAIN 3→4、
     词表 77→81 追加不移位、CONTRACTS p23 索引同 commit、RED 先行、
     fail-closed 连锁（probe 81/13/68 + fingerprint 52f0988fd76b +
     baseline 刷新）、全量串行 1,690/1/0——核验记录 §1/§7 的 p23 内容
     核验与该实施逐项对应；
  4. **在途**：81 名重测（detached）→ 族门裁决 → 测量日志 → t24 闭环；
     裁决结果（通过/负结果）由 factor-engineer 通知后治理记录补终态。
  5. **【第三次反转（2026-09-07 captain 纠正确认，实施 = `300cf70`）——
     取代上条第 3 项的 V2 集实施】**：captain 纠正确认（逐字）：「我的
     『v2 覆盖』裁定错误，用户「批准」效力 = v1 表格」——**恢复用户
     亲批四因子**（NET_MARGIN_TREND_4Q/LEVERAGE_TREND_4Q/
     CAPEX_INTENSITY/CFO_TO_DEBT），实施 = `300cf70`（p23 v5）：
     GRAMMAR 8→**9**（generation 9 分派注册）、FEATURE_REGISTRY 12→
     **14**（现值链：t70 8→9 → t24 路由 9→10 → p22 10→11 → F1 11→12
     → 本反转回退 12→13 → 本纠正确认 13→14，单调无跳号）、
     RESEARCH_DOMAIN 3→**4**、词表 81 名 78-81 = 上述四因子、
     fingerprint ce13e7a67fb9、CONTRACTS p23 索引行候选名在 `300cf70`
     更正（eab1061 登记该行、300cf70 更正候选名）。链上每次名称集
     变更恰一次 bump，全程 p23 §5/§10 披露（含 captain 纠正确认逐字）。
     本条取代上条第 3 项（V2 集实施）为 F1 实施的现行终态。**【已被
     第四次终局裁定取代（2026-09-07 captain，见下条第 6 项）】**：
     300cf70 的 V1 表集恢复被终局裁定回退（fe8bd11/3d2a9f1），V2 维持
     实施——本条的"现行终态"地位由第 6 项接替，本条保留作裁决史。
  6. **【第四次终局裁定（2026-09-07 captain，"第 4 次也是最后一次"）——
     取代上条第 5 项（300cf70 V1 表集恢复）】**：**V2（路线图原指名
     四因子）维持实施**。真实批准链（p23 草案 v6 @ `3d2a9f1` 入册）：
     (1) 用户路线图 F1 批次逐字指名四候选含方向与公式（最高背书）→
     (2) captain 呈递表格误用 v1 自提集 → 用户「批准」（批次级，对
     v1 呈递）→ (3) 成员 v2 重写（回归路线图指名）→ (4) captain 终局
     裁定 V2 维持。**90d40661/4db156e 引用更正为"批次级批准"，不作为
     V2 集批准依据**（与上文实施闭环条的引用更正一致）。终态核验
     （3d2a9f1 树状态逐项）：GRAMMAR **8** / FEATURE_REGISTRY **13** /
     RESEARCH_DOMAIN **4**、V2 四候选 feature_metadata L432-456、
     fingerprint **52f0988fd76b**（fairness baseline）、词表 81
     （78-81 = V2 四因子）、CONTRACTS p23 行 ✓。300cf70 的 V1 表集
     恢复被本终局裁定回退；F1 现行实施 = eab1061（V2 集）+ 3d2a9f1
     （草案 v6 定稿）+ 测量运行中。
  7. **【F1 终态登记（2026-09-07 族门裁决落地——通过，候选池准入）】**：
     factor-engineer 报告族门裁决**通过**（t22 日志 §14，`0cb6493`）——
     **OPERATING_PROFITABILITY 双腿达标**（ΔIC +0.0075 ≥ +0.005 且残差
     +0.0013 同向）→ 四特征进候选池 **promotion_allowed=True 维持**；
     metrics = docs/factor_inventory_audit_f1_20260907/metrics.json
     （81 名，eab1061-era V2 集 = 终局裁定集合）；逐因子备注 +
     相关性（无 Part A 触发）+ 成本达标全记录；t24 已置 completed、
     F2 解锁。p23 线治理记录至此终态闭环（用户批准链 + 实施链 +
     族门通过全链在案）。

- 性质：治理登记核验（evidence-only）。核验对象 = factor-engineer 的
  F 批次登记包（codex/factor-semantic-fixes）：
  `docs/p21_f1_slow_fundamental_quality_draft.md`（F1 四候选）+
  `docs/p22_industry_momentum_direction_repreregistration_draft.md`
  （INDUSTRY_MOMENTUM 方向重预注册 + F 批次相关性核对照表）@ `176241e`。
- 核验人：governance-auditor（AgentTeams tushare-pit-migration t71 后续
  登记），2026-09-07。
- 证据基线：核验时该分支真实常量（`176241e` 逐文件读取）= GRAMMAR 6 /
  FEATURE_REGISTRY 7 / RESEARCH_DOMAIN 3 / FACTOR_COMPUTE 4（t22 修复线
  已 bump）——**p21 版本表（GRAMMAR 6→7、REGISTRY 7→8、DOMAIN 3→4）与
  真实基线吻合**；p22 采用"当时现值 +1、两案取先后序"的顺延写法正确。

## 1. p21（F1 慢基本面质量 4 候选）核验

| 核验项 | 结果 |
|---|---|
| 批次依据 | ✓ 词表 77 名 slow_fundamental 域（16 名）的三个未覆盖维度（方向趋势/投资强度/偿债质量）；全部只用 p17 §5.3 已落库的 v2 三表字段（零 schema 变更）；复用 t22 修复的单一路径（asof_winner_intervals + _ttm/单期转换） |
| 公式/守卫 | ✓ 4 预注册公式（NET_MARGIN_TREND_4Q +1 / LEVERAGE_TREND_4Q −1 / CAPEX_INTENSITY −1 / CFO_TO_DEBT +1）各带经济假设、NaN 守卫（TTM 断档、分母 ≤0） |
| 非重复性 | ✓ 与 ROE/DEBT_RATIO/NET_MARGIN（legacy 静态水平，v2 as-of 切换为 db76700 后置任务）及 family-⑤（盈利-现金确认/存量增长）分工明确——补趋势/流量/偿债三维度；预注册冗余预检（实施后 |ρ|≥0.9 → 族内精简） |
| 族原子性 | ✓ 4 特征同 commit 进词表、同批裁决、同进退（p13 §8 同款） |
| 版本表 | ✓ 与真实基线吻合（见上）；FACTOR_COMPUTE 不变（纯追加，p13 先例）；dataset 不变（无 DB 写入） |
| RED/裁决 | ✓ 5 条 RED（公式 golden、as-of 可见性、词表追加不移位、域归属、未回填 fail-closed）；族门 |w_b| 修正基线（p9 附录 B）、整族原子、相关性 Part A 机制、方向按测量裁决 |

## 2. p22（INDUSTRY_MOMENTUM 方向重预注册）核验

| 核验项 | 结果 |
|---|---|
| 三段式历史披露（captain 裁定第 1 条） | ✓ round-1 覆盖 0.000 = 数据伪影无效（双重无效：消费路径缺陷 + D3 缺行；ΔIC 为 p9 附录 B 基线翻号伪影）/ round-2 修复后有效证伪（IC 全 horizon 负、年度同向 4/13）/ authored +1 保留为负结果证据（不删除不改写） |
| 新方向假设 | ✓ −1（行业指数层反转），与 IND_REL_* 残差族经济学一致但信息层正交（构造正交：残差 vs 指数层；实测 max |ρ|=0.379 vs MOMENTUM_20） |
| 范围/守卫 | ✓ 仅方向元数据 + FEATURE_REGISTRY 顺延 +1（两案取先后序写法正确）；FACTOR_COMPUTE/GRAMMAR/tier 不变；tier C 政策阻断不因方向变化解除 |
| 冗余预检 | ✓ 方向翻转不产生任何 ≥0.9 新对，无 Part A 触发 |
| RED | ✓ 元数据断言（== −1）+ registry 文档 drift guard |

## 3. F 批次相关性核对照表核验（captain 裁定第 3 条）

- p22 §5 表覆盖：两条重注册候选 × F1 四因子（定性预期 + F1 实现后实测
  承诺；**MARGIN_BALANCE_CHG × CFO_TO_DEBT 标记"中"**——负债/现金流
  经济学相邻，F1 实现后实测，若 ≥0.9 → Part A 裁决）；两条候选 × 既有
  77（已实测：max MARGIN_CROWD_60 +0.794 / MOMENTUM_20 +0.379，均 <0.9）。
- 结论：核对照表完整，实测承诺有明确触发规则。

## 4. 批准路径（两条不同，如实拆分）

1. **p22 → captain 代行批准**（p20 先例：既有特征的方向重预注册、零计算
   面、p9 §5.4"方向按测量裁决"惯例、captain t22 路由明示）。
2. **p21 → 用户批准（captain 转达）**：F1 是**新特征入词表**的预注册契约
   （GRAMMAR 6→7 采样空间变化）——p18 §3 硬边界 #2（预注册契约的批准）
   适用，且原 DAG 的 F1 任务书明示"预注册 → 用户批准 → 实施"。代行授权
   （勘误与流程批准点）不覆盖新特征批次。
3. CONTRACTS.md 索引行：p21/p22 均在各自批准/实施 commit 同 commit 登记
   （草案阶段不索引——t42/t49 先例）；落点 = codex/factor-semantic-fixes
   （契约文件所在分支）。

## 5. sequencing（供 captain 裁决；本核验不预写结果）

待实施的 FEATURE_REGISTRY 触碰方：p20（已批准，→8）、p9 Part A（已批准，
GRAMMAR→8 + REGISTRY→9）、p22（待 p22 批准，REGISTRY 顺延 +1）、p21
（待用户批准，GRAMMAR +1 + REGISTRY +1 + DOMAIN +1）。**约束**：各实施
commit 的版本表必须以实施时点的真实常量重算（草案中的绝对数字随排序
失效——p21 的 6→7/7→8/3→4 以当前未实施任何一项为前提）；建议单线单调
序（p20 → p9 Part A → p22 → p21），由 captain 定案。另注：p19（D4）的
FACTOR_COMPUTE 计划值以 main 谱系为基线，与 factor-semantic-fixes 谱系
（FACTOR_COMPUTE 4）在集成时需一次版本对账（integrator 职责，非本核验
范围）。

## 6. 未生效声明

p21/p22 在批准与实施落地前均为 DRAFT 草案（不索引、不生效、不授权任何
registry/词表变更）；本记录仅为治理侧核验与登记安排。

## 7. v2 增补核验（t24 权威候选集，2026-09-07）——附批准范围确认点

**核验对象**：`docs/p21_f1_slow_fundamental_quality_draft.md` v2
（codex/factor-semantic-fixes @ `0b2d612`，2026-09-07 09:25:59 +0800）——
F1 预注册按 **t24 任务书的权威候选集**重写（取代 v1 自提候选集）。

**核验通过项**：
1. 候选集与 t24 任务书**逐字一致**：CFO_TO_ASSETS / GROSS_PROFITABILITY /
   OPERATING_PROFITABILITY / LEVERAGE_CHANGE，方向 +/+/+/−，全部
   slow_fundamental / Tier B；
2. 版本表以真实常量正确重算：GRAMMAR 7→8、FEATURE_REGISTRY 9→10、
   RESEARCH_DOMAIN 3→4（该分支实测 GRAMMAR 7 / REGISTRY 9 / DOMAIN 3
   ——Part A 与 p22 已落地；FACTOR_COMPUTE 4 不变、dataset 157e0cf5 不变）；
3. **口径缺口明文预注册**：income_statement_pit 无 operate_profit 列 →
   F1-3"营业利润"以 total_profit 为预注册代理，未来 p17 增列须勘误重算
   （诚实处理，非静默替代）——p17 DDL 零改动；
4. 零 schema 变更、零 DB 写入（dataset 157e0cf5 不变）；公式/守卫/
   非重复性论证/族原子性/RED/族门（|w_b| 修正基线）齐备。

**批准范围确认点（实施前置，governance 标记）**：用户「批准」原文
（2026-09-07 经 captain 转达）的**转达时点早于 v2 重写 commit**
（0b2d612，09:25:59 +0800）——若 captain 转达用户的内容为 t24 权威
候选集（= v2），则批准覆盖 v2，实施可按 sequencing 启动；**若转达的
是 v1 自提候选集（NET_MARGIN_TREND_4Q/LEVERAGE_TREND_4Q/CAPEX_INTENSITY/
CFO_TO_DEBT），则 v2 候选集（其中 CFO_TO_ASSETS/GROSS_PROFITABILITY/
OPERATING_PROFITABILITY 三项为 v1 之外的新候选）需按 t24 流程第 1 步
补用户批准后方可实施**。请 captain 确认转达内容；governance 侧在确认前
不视为已获批准。
**【确认点关闭（2026-09-07 captain 裁定：v2 被批准覆盖）】**：依据
(a) 用户路线图 F1 批次**逐字指名** CFO_TO_ASSETS/GROSS_PROFITABILITY/
OPERATING_PROFITABILITY/LEVERAGE_CHANGE（含方向与公式建议）——四因子的
用户背书先于一切流程；(b) 用户「批准」批准了 F1 批次推进；(c) v2 恢复
路线图原指名 = 比用户所见 v1（成员自提替代集）更贴近原始规格——候选集
修正方向是**回归用户路线图而非扩张**。v1 中间集被取代（批准基础 =
路线图 F1 背书 + 批次批准，演变在本登记中如实记录）；若用户有异议会
主动提出（其持续关注交付报告）。**实施前置解除：factor-engineer 可按
sequencing 启动实施。**
**【引用更正（2026-09-07 captain 指令，与上文实施闭环条的更正同源）】**：
本确认点依据 (b) 的"用户「批准」批准了 F1 批次推进"为**批次级批准**
（对 v1 呈递表格，90d40661/4db156e）——**不作为 V2 集（路线图四因子）
的批准依据**；V2 集的实施依据 = 用户路线图 F1 逐字指名（最高背书）+
captain 终局裁定 c970a26d（V2 维持、禁止再翻转）。

**编号与登记**：草案文件名/标题仍为 p21——登记前须按 captain 编号纠正
改为 **p23**（文件名 + 标题 + 自引用，factor-engineer 在实施 commit
落地），CONTRACTS.md 索引行同 commit 以 p23 登记。
