# P45 搜索评分与正式执行差异的测量与对齐决策预注册契约（工作包 9）

- 状态：**APPROVED**，用户 2026-09-20 直接指令「工作包 9：测清搜索评分与
  正式执行的差异，再决定如何对齐。范围：reward_baskets.py、正式回测
  路径、固定候选对照工具及相关测试。……先将差异分成：费用结算或现金
  守恒不一致应修复；整手和可买数量导致的预期近似应量化；近似导致方向
  选择、候选排序或资格变化应进入研究决策。固定一小组候选，在相同窗口、
  掩码、方向和配置上对照正式引擎，报告净收益误差、费用误差、方向翻转
  及排序变化。如果保留快速近似评分，入围候选应经过正式执行复核；误差
  超过预先设定的容限时，不能继续把代理分数作为可靠排序依据。本包完成
  前不扩大 GP/TPE/RL 搜索预算。不通过放宽断言消除 xfail。」（批准记录
  回写于本文件头）。
- 起草：执行模型，分支 `codex/b04-load-cache-performance`。
- 勘察基线：p44 完成后 HEAD `f615681`，工作树干净。

## 1 差异分类证据（2026-09-20 归因探针，基线树，三个 xfail 场景）

同窗同掩码同方向同配置下，篮子（权重空间，搜索评分路径
`CandidateScorer → simulate_basket_daily_returns`）vs 正式引擎
（股数空间，`AshareBacktestEngine`）：

| 场景 | 零费净差 max | 全配置净差 max | 费差 max（每期） | 换手差 max |
|---|---|---|---|---|
| A 掩码 universe（xfail-3） | 2.9e-5 | 1.9e-4 | 1.1e-4（总 5.26e-3 vs 4.80e-3） | 1.2e-2 |
| B 阻断涨跌停（xfail-1） | 3.3e-4 | 8.0e-4 | 1.1e-4 | 1.6e-2 |
| C 现金不足额建仓（xfail-2，零费） | — | — | 0 | ~1e-3 |

参考粒度：`g = lot_size × 价格 / 资金 = 100 × ~10 / 1e5 = 1%`。

**分类裁决**：
- **类一（费用结算/现金守恒不一致 → 应修复）：无**。两侧同一
  `ExecutionCostModel`（同费率、同逐单最低佣金公式）；引擎 p35 账本
  恒等式与篮子 `(1−cost)(1+gross)−1` 各自守恒（既有测试钉住）。实测
  费差（篮子系统性偏高 ~10%，源自连续权重名义 > 整手地板名义 + 引擎
  对被整手地板归零的腿不收费）与现金路径差（引擎以现金付费用并收缩
  买入、篮子以收益率拖拽与 1/(1−cost) 权重膨胀表达同一经济学）都是
  连续近似 vs 整手执行的**有界近似**，非公式错误。本包对
  reward_baskets.py **零代码改动**。
- **类二（整手/可买数量近似 → 量化）**：上表数值以 g 为单位记录
  （净差 max ≤ 0.08·g，费差 ≤ 0.011·g，换手差 ≤ 1.6·g），由 B2 容限
  与工具常态化量化。
- **类三（方向/排序/资格 → 研究决策）**：由固定候选对照工具实测
  （§3 实测），任何翻转进入研究决策并触发 B3 复核门。

## 2 冻结语义（实现必须逐条可对照）

**B1 对照工具**（新模块 `ashare_model/scoring_parity.py`，
`SCORING_PARITY_VERSION = 1`，additive）：
- `compare_candidate_parity(candidates, raw_cache, ts_codes, dates,
  universe_mask, backtest_config, *, blocked_buy, blocked_sell,
  rebalance_mask)`：每个固定候选 `(名字, 方向, 信号)` 在**同一窗口/
  掩码/方向/配置**上同时运行 (a) 篮子模拟与 (b) 正式引擎，逐候选报告
  `daily_net_error_max/mean`、`fee_fraction_error_max`、
  `turnover_error_max`、`total_net_return_error`、`total_fee_error`；
  报告级记录粒度 `g = lot_size × median(执行日开盘) / initial_capital`。
- 双向都跑（+信号与 −信号）：`proxy_direction`（篮子优选方向）vs
  `engine_direction`（引擎优选方向），不一致计为**方向翻转**。
- 排序对照：代理分数序（篮子优选方向的窗口总净收益）vs 引擎序，
  报告 Spearman ρ 与相邻逆序对，按分差是否 > 日净容限分为
  `material` 与 `tie`（披露不计败）。
- `run_fixed_candidate_parity(loader, fold, backtest_config, candidates)`：
  数据级接线（epoch_slice 同一窗口/掩码/阻断矩阵，候选=固定因子/
  组合信号），供固定候选实测与后续复核复用。

**B2 冻结容限**（机制推导：单腿整手地板误差 ≤ g；一次调仓 ≤ 2·top_n
腿；费用差源于连续 vs 整手名义与逐单最低佣金；实测锚点 §1）：

| 量 | 容限公式 | 实测锚点 |
|---|---|---|
| 日净收益误差 max | `0.2·g` | 0.08·g（B 场景） |
| 每期费用分数误差 max | `0.05·g` | 0.011·g |
| 每期换手误差 max | `10·g` | 1.6·g |
| 窗口总净收益误差 | `0.05·g·n_periods` | ~0.008·g·n |
| 方向翻转 | **0** | — |
| material 排序逆序 | **0** | — |

容限随 g 线性缩放（小资金/高价股 → 粒度变大 → 容限同步变大，与机制
一致）；g 由工具逐报告计算并记录，非全局常数。

**B3 复核门（政策判定函数）**：`proxy_reliability_verdict(report)` →
`reliable`（全部误差在容限内、零方向翻转、零 material 逆序；tie 逆序
与全部误差数值披露）或 `unreliable`（附逐项原因）。unreliable 时，
**该候选集的代理分数不得继续作为可靠排序依据**——入围候选必须以正式
引擎复核分数为准（工具即复核执行器）。搜索器接线（在搜索循环内强制
复核）属后续研究决策，本包不改搜索语义、不扩任何 GP/TPE/RL 预算。

**B4 三个 xfail 收敛（不放宽，换正确的契约形态）**：逐位相等
（abs=1e-12）要求连续权重 == 整手执行，本身是错误契约（用户明示不应
强行逐位相等）。替换为：机制推导界的有界 parity 断言（界=B2 公式，
锚点独立实测，非"跑过即可"）+ 结构性断言保持精确（阻断日零成交、
不补齐归一、掩码行永不入场、引擎现金守恒）。三个 xfail 标记移除。

**B5 版本影响**：新模块 `SCORING_PARITY_VERSION=1`（additive）；
`REWARD_VERSION`、`PROTOCOL_VERSION`、`EXECUTION_SPEC_VERSION`、
`PORTFOLIO_CONSTRUCTOR_VERSION` 全部不变（篮子与引擎零语义改动）。

## 3 固定候选实测（本包内完成，报告进测量日志）

- 候选（5 个，非 OOS 挑选，p34 冻结臂）：`MOMENTUM_20`、`RSQ_60`、
  `REVERSAL_60`、`IND_REL_RET_20`、`combo:all3`（三因子等权）。
- 配置：p37 参考格 n10_m2500（top_n=10、min_trade 2500、资金 1e5、
  medium 域默认预算与周期、v2 priority 分配）；窗口：2023 折（单折，
  与 R02 同口径 epoch_slice）。
- 报告：四族误差（日净/费用/换手/总净）、每候选方向（代理 vs 引擎）、
  5 候选排序对照（ρ + 逆序分类）、复核门裁决。

## 4 范围与非目标

范围：`ashare_model/scoring_parity.py`（新）、
`tests/test_p45_scoring_parity.py`（新，同 commit 注册分片）、
`tests/test_reward.py` 三个 xfail 的契约化替换（§10.1 规则 2）、
固定候选实测与测量日志。

非目标：reward_baskets.py / 引擎语义改动（类一为零的证据见 §1）、
搜索器循环内强制复核的接线（后续研究决策，且本包明示不扩搜索预算）、
优化器模式对照（等权路径先测清）、多折/多窗口面板（单折实测先行，
扩展属后续测量包）。

## 5 预期 RED（独立 oracle——禁止从实现反推）

`tests/test_p45_scoring_parity.py`：

- **N1** 工具在合成市场上对固定候选产出完整误差四族，且同向同配置下
  误差在 B2 容限内（正值对照）。
- **N2** 容限随 g 缩放：同一误差在粗粒度（小资金）下通过、细粒度下
  界限收紧（bounds 函数单位测试：g 的计算与各容限公式逐一对照手算）。
- **N3** 复核门：伪造误差报告 → reliable；超日净容限 / 超费用容限 /
  方向翻转 / material 逆序 → unreliable 且原因逐项点名；tie 逆序 →
  reliable 且披露。
- **N4** 排序对照：Spearman 计算与 material/tie 分类（伪造分数对照
  手算）；方向翻转检测（伪造双向报告）。
- **N5** 数据级接线：populated_db 上固定候选双向运行产出非空报告且
  窗口/掩码一致（两路径消费同一 epoch_slice）。

test_reward.py 的三个 xfail 替换为 B4 形态（规则 2 修改，非 RED）：
结构断言保持精确 + 数值断言换 B2 机制界。

## 6 验证矩阵

1. 聚焦：`python -m pytest -q tests/test_p45_scoring_parity.py`。
2. 相邻：`tests/test_reward.py`（全量，xfail 移除后必须无 xfail）、
   `test_p3_portfolio_parity.py`、`test_backtest.py`、
   `test_baseline_harness.py`、`test_p34_r01_baseline.py`。
3. `python -m compileall -j 0 -q …`、
   `python scripts/check_test_shards.py --check`、`git diff --check`、
   全量 `python -m pytest -q tests -n auto`（对账口径同 p43/p44）。

## 7 停止条件

- 实测发现类一（公式级不一致/守恒破坏）证据 → 停止，回契约补修复
  条目（那是修复包，不是测量包）。
- 固定候选实测出现方向翻转或 material 逆序 → 如实进入研究决策并给
  出 unreliable 裁决，不在本包内改评分语义。
- 全量回归出现无法归因于本包的新红/新 warning → 停止诊断。
- xfail 替换若只能靠"跑过即可"的数值（无机制推导）→ 停止，界必须
  先有公式与锚点。
