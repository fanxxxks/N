# P21/P22 F 批次登记核验记录（governance——批准链、版本基线、sequencing）

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
