# P9 勘误（t22 已执行）：§4.1 keep-list 与 Part A 共线规则在 TURNOVER_MA20 ~ TURNOVER_STD20 上的 standing 冲突——证据整理与裁决执行

- 状态：**EXECUTED — governance-auditor 裁决（选项 A）+ factor-engineer 实施 @ 596550a**（2026-09-07；证据整理稿起草：factor-engineer，2026-09-06；captain t22 路由指令第 3 条）。证据基线补第五轮：附录 B 修正重算 docs/factor_inventory_audit_v2_fix2_20260906/（rho 复测同值 0.914）。
- 证据基线：四轮审计的冻结/在档 metrics（t2 `docs/factor_inventory_audit_20260831/`、v4 `docs/factor_inventory_audit_v4_20260901/`、t18 `docs/factor_inventory_audit_v2_20260905/`、t22 `docs/factor_inventory_audit_v2_fix_20260906/`）。本文不含新裁决，只整理证据与可选项。

## 1. 冲突陈述

P9 契约 §4.1 的淘汰表以 TURNOVER 族共线证据淘汰了 TURNOVER_MA5（ρ=0.926/0.928 vs TURNOVER/MA20），**显式保留** TURNOVER（日频原子）+ TURNOVER_MA20（"月度拥挤水平"慢代表）+ TURNOVER_STD20（"异质语义"），保留理由明文入契约。同一 §4.1/§7 机制（Part A：|ρ|≥0.9 → 淘汰较弱代表）在四轮审计中测得**两个被显式保留的代表彼此 ρ = 0.914**：

| 轮 | dataset/语义 | ρ(MA20, STD20) | MA20 IC10（ICIR） | STD20 IC10（ICIR） |
|---|---|---|---|---|
| t2 | a839ecf2（legacy qfq） | **0.914** | −0.0594（−0.285） | −0.0651（−0.343） |
| v4 | a839ecf2（legacy qfq） | **0.914** | −0.0594（−0.285） | −0.0651（−0.343） |
| t18 | 43ef1e4c（v2 修复前） | 0.919 | −0.0532（−0.269） | −0.0579（−0.317） |
| t22 | 157e0cf5（v2 修复后） | **0.914** | −0.0593（−0.287） | −0.0654（−0.348） |

冲突：Part A 的机械规则（≥0.9 → 弃弱者）与 §4.1 的显式 keep-list（两代表各有语义 rationale）同场并存——规则触发不构成自动弃用（p9 §4.1 的 keep 是更具体的预注册），但冲突长期在案未决。

## 2. 证据要点（governance 裁决用）

1. **共线稳定**：0.914 跨四轮（两代数据底座、修复前后）不变——不是数据/语义事故，是结构共线（MA20 与 STD20 同源于 turnover_rate 的 20 日窗统计）。
2. **强弱一致**：STD20 的 ICIR 在四轮中恒强于 MA20（−0.343/−0.348 vs −0.285/−0.287；h20 同向）。若走 Part A 淘汰，被弃者为 **TURNOVER_MA20**（弱代表），保留 TURNOVER（日频）+ TURNOVER_STD20（波动/异质）。
3. **语义代价**：弃 MA20 = 词表失去"月度拥挤水平"慢代表（TURNOVER 短窗 + STD20 波动仍覆盖 crowding 维度；P9 §1 的正交性缺口评估当时以 MA20 补位）。
4. **t22 裁决先例**：t18/t22 两轮均判"不自动弃用"（显式 keep-list 优先），冲突以勘误候选留档——即本草案。
5. **影响面**（若弃 MA20）：DEPRECATION_REASONS +1、GRAMMAR 7→8（采样空间变化）、FEATURE_REGISTRY 8→9、旧公式按名保留可解析（P9 Part A 先例：deprecated 继续计算、退出采样、禁止晋级）、`docs/feature_registry.md` 再生、相关钉值更新。t22 分支已有该机制先例代码路径（P9 d6a034d 同构）。

## 3. 可选项（governance 裁决）

- **选项 A（Part A 机械执行）**：弃 TURNOVER_MA20（弱代表），按 P9 Part A 全套机制落地。
- **选项 B（显式例外入契）**：在 p9 §4.1 增补勘误文本，明文承认 MA20~STD20 0.914 为**已裁决的 standing exception**（两代表语义异质、四轮稳定），Part A 对该对不再触发——把今天的隐式状态显式化。
- 两选项均消解 standing 冲突；差别在词表采样空间是否收缩。证据完整性：无论选项，本节四轮 ρ/ICIR 表应随勘误入档。

## 4. 边界

- 本草案不实施任何 registry 变更；p9 §7.2 的 ΔIC 退化性勘误属另一条线（captain 路由 item ②，governance+semantics 辖区），不在本文。
- t22 相关轮次引用：`docs/t18_factor_retest_measurement_log.md` §5.5、`docs/t22_semantic_fix_measurement_log.md` §5。

## 5. 裁决执行记录（2026-09-07）

- **governance-auditor 裁决：选项 A**（Part A 机械执行）——理由要点：(1) t69 豁免是有条件豁免，条件已触发（四轮 rho 0.914/0.914/0.919/0.914 持续 >=0.9 跨两代底座与修复边界 + 增量证据 STD20 ICIR 四轮恒强）；再次豁免 = 把条件豁免变成无条件豁免；(2) Part A tiebreaker（弃弱者）无歧义（MA20 每轮更弱）；(3) 语义代价有界（水平维度由 TURNOVER 覆盖、波动维度由 STD20 覆盖）；(4) 内在一致性（新因子须过 |corr|<0.9 而书内自持 0.914 对不自洽）；(5) 选项 B（看完四轮数据追认例外）属 AGENTS §1 防的事后放宽形态，否决。
- **实施 @ 596550a**：DEPRECATION_REASONS += TURNOVER_MA20；GRAMMAR 6→7（采样空间变更；generation 7 注册进 remap 分派，token 布局与 6 相同——弃用不移 id）；FEATURE_REGISTRY 7→8（TURNOVER_MA20 promotion_allowed=False）；docs/feature_registry.md 再生；钉值更新（GRAMMAR 7、REGISTRY 8、feature_version hash 2217cb8d4902——按设计变更）；RED 先行（test_turnover_ma20_deprecated_per_part_a_ruling 在冻结代码上失败）。
- **版本号披露**：governance 指令原文 GRAMMAR 7→8 / FEATURE_REGISTRY 8→9 以 F1 先行为前提；实际执行时点 F1 尚未批准，按当时现值 6→7 / 7→8 落地，F1 批准时顺延（其草案版本表将随实施更新）。
- p23（F1 四因子草案，原编号 p21→p22）不受本弃用影响（四候选与 TURNOVER_MA20 无共线关系）；p22（INDUSTRY_MOMENTUM 方向注记，编号经 825c929 往返后维持 p22）同。
