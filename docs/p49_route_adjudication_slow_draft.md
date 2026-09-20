# P49 研究路线裁决与 slow 域简单基线轮预注册契约（工作包 13）

- 状态：**APPROVED**，用户 2026-09-20 直接指令工作包 13「按证据选择
  下一条研究路线」并冻结决策表（批准记录回写于本文件头）。
- 起草：执行模型，分支 `codex/b04-load-cache-performance`。
- 勘察基线：p48 完成后 HEAD（WP12 终提交），工作树干净（仅未跟踪
  `.mimosa/`）。
- 性质：路线裁决 + 一轮正式基线运行（p34 冻结机器，零仓库语义改动）。

## 1 冻策表逐行裁决（证据 → 路线）

| 行 | 触发条件 | 证据核对 | 裁决 |
|---|---|---|---|
| 1 | 动态信号有增量、执行与评分误差可控 | P47 第 2 步：动态劣于首建仓持有（4 臂中 3 臂；组合 −28.9pp/−1.7pp）；P45：评分 vs 引擎 unreliable（2 方向翻转 + material 逆序） | **未触发**——不开 GP/Random 对照 |
| 2 | 等权组合尚可、现有加权规则失败 | p42：ICIR 加权全负（已裁决）；P47：等权 hold 口径 ≈ 0（−0.67%） | **搭载生效**：等权保留为参照臂；无新假设不开组合方法比较（p42 结论维持） |
| 3 | medium 修正后仍无可靠净空间 | P47 裁决 1、3 双触发：hold≈0、动态为负、费用二阶；R01/R02/R03 全负 | **选定路线**：一轮数据合格的 slow 域简单基线 |
| 4 | 优势主要来自现金或初始持仓 | 无优势可归因（损失帧；暴露主导是 p44 规则病理，已登记修订项） | **未触发**——不转配置/持有问题研究 |
| 5 | 多轮有限实验均无增量 | medium 域确已多轮无增量，但 slow 域从未运行（p21 预注册配置在册） | **暂缓**——作为 slow 轮失败后的回退路线（条件见 §4） |

**结论：执行第 3 行。**

## 2 slow 域解除封堵证据（复测，非沿用）

`configs/r1_slow_ab.yaml` 头注（2026-09-06 BLOCKED：income_statement_pit
覆盖 85.60% < 90%）已过时。2026-09-20 以 D7 单一度量权威
（`data_eligibility._financial_coverage`）在当前 v2 库复测（宇宙=两指数
成分并集 1630 码、报告期 2014Q1–2023Q4、上市前排除）：

| 表 | 覆盖 | 阈值 |
|---|---|---|
| income_statement_pit（最差） | **94.00%** | ≥90% ✓ |
| balance_sheet_pit | 99.10% | ✓ |
| cashflow_statement_pit | 97.30% | ✓ |

探针 `reports/p49/probe_coverage.py`。正式运行内部仍走完整 G1–G8 +
数据资格链——探针不是替代，是解除"明显被封"的前置证据。

## 3 冻结运行

- 入口：`python scripts/r01_baseline.py --config configs/r1_slow_ab.yaml
  --contract-sha256 d5460955ab15dc026592047734bcb419358e303fb9db75374f8da23bf8857f0f
  --out-root data/r1/r01_slow_baseline`。
- 框架：p34 冻结契约（sha 自校验）、r01 驱动原样（臂=基准/现金控制/
  5 个 slow 单因子（ROE、PE_TTM、REVENUE_YOY、DIVIDEND_YIELD、
  MARKET_CAP）/2 个冻结跨族组合）、五折 2021–2025、0x/1x/2x、
  every_20_days/20、域默认预算与配置 verbatim、p46 方向 provenance
  随行、p44 priority 默认执行。
- 裁决规则：`R01_ADJUDICATION_RULES` 原样（worth_search/低活动/单年
  依赖/成本脆弱）。
- 墙钟预算 6h（p34 §3.6 原样）。

## 4 结果映射（冻结）

- **worth_search**（有折现净空间且活动/依赖/成本门通过）→ 登记 slow
  域小规模搜索假设（**不在本包开启**；开启需新契约 + WP12 解冻前置）。
- **无可靠净空间 / REJECTED / FAILED** → 决策表第 5 行生效：暂停扩大
  算法投入，保留研究平台与全部负结果；medium+slow 双域负结果在册。
- 数据资格任何门失败 → 如实 REJECTED 记录，不放松阈值。

## 5 非目标

新因子族、GP/TPE/RL 预算、medium 域重跑、组合方法比较（无新假设）、
方向规则更换、显著性/晋级声明（WP12 冻结维持：AR(1) 邻域与短窗域
不可作 5% 声明）、市场状态分类。

## 6 停止条件

- G1–G8/数据资格失败 → 运行自身 REJECTED，如实记录。
- 引擎/账本不变量失败 → 停止诊断。
- 墙钟超 6h → 预算熔断，记录已跑部分，不续跑。

## 7 版本影响

~~零版本 bump（路线裁决 + 既有机器的正式运行；无代码改动）。~~

**勘误 1（2026-09-21，前瞻性）**：首轮正式运行在因子资格阶段被
fail-closed 拒绝——check-6 的 leave-family-out 消融对单因子族集
（slow 五信号全属 fundamental 族）结构上无法产生子集，**运行零 OOS
行、未见过任何经济结果**，故按仓库勘误先例面向未来修订：单族集的
消融证据回退为**族内 leave-one-signal-out**（同一 baseline_candidates
机器、同一哈希纪律、标签 `family:signal`）；单信号集仍为空（拒绝
维持）。多族集行为逐位不变。`R01_REPORT_SCHEMA_VERSION` 2 → 3。
本轮 slow 结果按修订后机器产生；medium 历史报告不受影响。
