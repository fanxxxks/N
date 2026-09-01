# 最终综合报告：AlphaGPT 正交信息补充与四搜索器对比（t9 合规审查 + 裁决综合）

- 执行：AgentTeams captain（Kimi-K3；reviewer 成员运行时 kimi-k3 连续崩溃不可用，captain 接管亲自执行）
- 日期：2026-09-01；团队 alpha-orth-research
- 性质：**研究诊断与搜索器行为测量**——不构成任何 alpha / 晋级 / admission / 生产变更结论
- 分支：`codex/p9-factor-families`（t1-t5）+ `codex/p10-searcher-comparison`（t6-t8）；**均未 push、未合并 main**

---

## 0. 一句话结论

**在当前预算（B=2000）、窗口（fold-0, 2021 测试年）、全成本日调仓协议下，四种搜索器（gp/tpe/random/rl）均未产出可通过臂级回测硬门槛的公式族——合法负结果。改进方向按预注册映射指向评价器/Reward 与成本-换手结构，而非某一搜索器。**

---

## 1. 任务完成矩阵

| 任务 | 执行者 | 状态 | 关键交付 |
|---|---|---|---|
| t1 数据/产物 Readiness | auditor | ✅ | G1-G7 PASS；发现 manifest 断链（BLOCKER-HIGH） |
| t10 manifest 重建 | captain | ✅ | 新 dataset_id `a839ecf2…`，verify 8/8 表 PASS |
| t2 62 因子库存审计 | auditor | ✅ | `docs/factor_inventory_audit_20260831/`（七维测量） |
| t3 因子族契约 | auditor | ✅ | `docs/p9_factor_family_contract.md`（APPROVED） |
| t4 注册表精简+新特征实现 | auditor | ✅ | commit `781f5e1`（11 新特征 + deprecation + 稀疏安全标准化） |
| t5 v4 全集复审计 | auditor | ✅ | `docs/factor_inventory_audit_v4_20260901/`（无回归 PASS） |
| t11 t1-t5 独立审核 | captain(Kimi-K3) | ✅ | `docs/p9_t1_t5_review.md`（逐项 PASS，无阻断） |
| t6 搜索器公平契约 | search-runner | ✅ | `docs/p10_searcher_fairness_contract.md`（APPROVED） |
| t7 四搜索器全量搜索 | search-runner | ✅ | 12/12 行 completed，3.91h（7h 上限 56%） |
| t8 回测对比与达标检验 | search-runner | ✅ | headline = `negative_no_admissible_formula` |
| t9 合规审查与综合报告 | captain | ✅ | 本文 |

---

## 2. 合规审查（AGENTS.md §13 阻断清单逐项核查）

| 阻断项 | 结论 | 证据 |
|---|---|---|
| 未来泄漏 / PIT 绕过 | ✅ 无 | t1 G1-G7 PASS；t2/t5/t7 全程 dataset_id=a839ecf2… 一致；新特征因果实现经 t11 独立核实 |
| 未预注册语义变更 | ✅ 无 | 全部语义变更（P9 因子族、P10 GP 长度对齐）均有批准的预注册契约；两处偏差（train.py NaN 修复、GRAMMAR 4→5）均经契约附录/测量日志合规登记 |
| 跨版本产物拼接 | ✅ 无 | t2 基线、t5 v4 基线、t7 搜索、t8 回测各自同 commit 链、同 dataset_id |
| 工程/小预算冒充研究结论 | ✅ 无 | 校准产物（p10_calibration_engineering.json）明确标 engineering，仅用于预算推导 |
| 搜索器公平性 | ✅ 通过 | 同一词表 v5（61 可采样）、同一 max_len=12（GP 长度缺口已修复并实测验证：12 行 max canonical len=12）、同一 B=2000、paired seeds [42,7,2024]、同一评分器/配置 |
| 第二套语义实现 | ✅ 无 | 审计/搜索/回测全复用 registry/loader/reward/evaluate_formula 单一路径（t11 核实） |
| 测试弱化获绿 | ✅ 无 | 新增 RED 测试先行留证；无删测试/弱断言/扩大 tolerance；全量回归 1374 passed/5 skipped/0 failed |
| 正式路径静默降级 | ✅ 无 | ledger unbound v0、config_hash=None 缺口均如实披露（非掩盖） |
| 负结果掩盖 | ✅ 无 | 族③涨跌停负结果、VOLUME_SHRINK_5_20 负结果、四搜索器全臂未过门槛——全部如实记录 |
| 死代码/兼容层 | ✅ 无 | deprecated 特征保留计算与解析（非空壳）；pending_data 占位不进词表 |

**合规结论：无 §13 阻断级问题。** 唯一 CONCERN（轻微非阻断）：GRAMMAR 4→5 系执行预注册裁决的后果，测量日志已论证，但 t3 契约 §9 版本表原文只写到 3→4（文档完备性瑕疵，语义正确）。

---

## 3. 研究结论综合

### 3.1 因子侧（t1-t5）：正交信息补充成功落地

- **库存审计（t2）**：62 因子七维测量完成。最强最稳 = 负向流动性/波动族（TURNOVER_STD20 ICIR -0.343@10d、IVOL_60、MARGIN_BALANCE_CHG 等，6 特征 12/12 年度方向一致）；正向慢变量随 horizon 增强（LIST_AGE ICIR 0.14→0.40）。
- **因子族精简（t3/t4）**：62→73 词表（+11 新特征），deprecation 去重不删数（token 保值，历史公式零破坏）。
- **关键修复（F1）**：稀疏事件特征被 winsorize 削平的缺陷已修复（LIMIT_UP_EVENT 非退化 19→719 天，38×）。
- **族级裁决（t5，预注册 ΔIC_OOS≥+0.005 门槛）**：
  - ① 行业残差化动量 **PASS**（IND_REL_RET_60 +0.0271）
  - ② 流动性/量价背离 **PASS**（PV_DIV_20 +0.0160）
  - ③ 涨跌停事件条件 **负结果**（三成员 ΔIC<0.005，延续方向被证伪 → promotion_allowed=False）
  - ④ 拥挤度 **PASS**（MARGIN_CROWD_60 +0.0395）
- **新 baseline**：词表 73 / deprecated 12 / 可采样 61 / 可晋级 58。
- **第⑤类（现金流质量/应计/资产增长/盈利加速度）**：pending_data 占位——DB 缺现金流/总资产字段，待数据侧 PIT 同步后另立契约。

### 3.2 搜索器侧（t6-t8）：四搜索器公平对比 = 合法负结果

- **公平性关键修正**：发现并修复 GP 有效长度缺口（7 vs 12 token），否则 matched comparison 不合法（AGENTS §7）。
- **Stage A 搜索行为**：GP 提前停滞（消耗 5-14% 预算）、TPE/Random 满额、RL ~64%；四臂 best-so-far 终点一致 = 0.98 平台。
- **Stage B/C 回测硬门槛**（年化>10%/回撤<15%/Sharpe≥1.0/Calmar≥1.0，fold-0 2021 测试窗）：
  - 臂级 admissible：gp 0/3、tpe 0/3、random 1/3、rl 0/3 → **无臂通过**
  - 唯一全过四门槛公式 = random:seed7（年化 22.77%/回撤 7.52%/Sharpe 1.268/Calmar 3.027），但组合活动门披露**近静态持仓**（5 次调仓/年、8640 抑制交易）——按 AGENTS §8.5 不得单独解读为有效，单 seed 不构成臂级通过
  - 8/12 行跑输等权基准（年化 13.77%）

### 3.3 两个结构性发现（改进方向的证据基础）

1. **评价器/Reward 区分度不足**：best-so-far 在 0.98 精确值饱和（12/12 行一致），area 四臂趋同 <0.4%——奖励函数无法区分搜索器质量，是首要结构性瓶颈。
2. **成本-换手结构硬约束**：12 条选中公式全部近静态（2-6 次调仓/年、0.44-0.68% 日换手、5940-9064 抑制交易）——印证 t2 成本结论：现费率下快价格信号（~14%/yr 成本）无经济空间，medium/slow 才是主战场。

---

## 4. 下一步改进方向（预注册 §8 映射执行 + captain 综合）

裁决映射原文执行：**转向评价器/Reward 与成本-换手结构，不归咎某一搜索器。** 具体立项建议（供后续新契约，非本次承诺）：

1. **评价器/Reward 重构**（最高优先级）：0.98 平台饱和表明 reward 缺乏区分度。建议预注册契约研究：更精细的 reward 构造（如 IC decay 加权、风险调整后 reward、或对近静态持仓施加惩罚项）。
2. **成本-换手结构**：利用 t8 新增的组合活动门字段，将换手/活动水平纳入搜索目标或晋级门槛，避免搜索收敛到近静态书。
3. **random:seed7 公式的独立追查**（可选）：该公式唯一全过四门槛但有近静态持仓 caveat。若追查，应另立预算放大研究（多 seed 验证是否稳健），不在本契约内追加裁决。
4. **第⑤类基本面因子**：待数据侧补齐现金流/总资产 PIT 字段后另立契约。
5. **数据同步**：daily_bar 落后 6 个开市日（data_end=20260821），进入任何 paper/生产前需补同步。

### 明确不做（边界）
- 不切换生产默认搜索器（gp 保持不变）；
- 不做 admission/lifecycle 状态转换（P8 另辖）；
- 不做 paper/sim 模拟盘操作（需另行预注册 ≥40 交易日观察窗契约）；
- 不 push/不合并 main（两个工作分支保持本地）。

---

## 5. 产物与证据索引

| 类别 | 路径 |
|---|---|
| t2 审计 | `docs/factor_inventory_audit_20260831/{audit_report.md, metrics.json, audit_run.py}` |
| t5 v4 审计 | `docs/factor_inventory_audit_v4_20260901/{audit_report_v4.md, metrics.json, audit_run_v4.py}` |
| t11 独立审核 | `docs/p9_t1_t5_review.md` |
| P9 契约 | `docs/p9_factor_family_contract.md`（APPROVED）+ `docs/p9_measurement_log.md` |
| P10 契约 | `docs/p10_searcher_fairness_contract.md`（APPROVED）+ `docs/p10_measurement_log.md` |
| t7 搜索产物 | `data/p10_searcher_comparison/{campaign.json, ledger.jsonl}` + `docs/p10_searcher_comparison_20260901/summary.json` |
| t8 回测产物 | `data/p10_searcher_comparison/adjudication.json` |
| 代码分支 | `codex/p9-factor-families`（781f5e1→403ee15）、`codex/p10-searcher-comparison`（ce98004→d5ce2aa） |

## 6. 过程质量与诚信记录

- 全部负结果（族③涨跌停、VOLUME_SHRINK_5_20、四搜索器无臂通过）如实记录，未删改任何失败证据。
- 两次缺陷修复（train.py NaN reward、config_hash 默认路径）均有 RED 测试 + 契约/日志登记。
- 成员运行时稳定性问题如实记录：kimi-k3 在 AgentTeams 成员/子代理运行时连续 7 次中途崩溃（factor-dev 3 + reviewer 3 + workflow 1），已通过改派 GLM-5.3-Flash（auditor/search-runner）与 captain（Kimi-K3）接管规避；建议后续慎用 kimi-k3 作为团队成员运行时。
- 未 push、未合并 main；全部工作保留在本地分支供用户审阅。
