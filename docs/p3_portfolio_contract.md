# P3 组合、调仓与多周期标签契约

本文件是 P3-01～P3-07 的需求与测试断言来源。若实现、既有测试与本文件冲突，
先按本文件判断语义；不得把测试期望改成实现的当前输出。

## 1. 时间与调仓

所有索引均指严格递增的交易日轴，区间均为半开区间。信号、入场和持有期的
唯一合法关系是：

```text
signal[t] -> entry open[t + 1] -> exit open[t + 1 + horizon]
```

`horizon` 是持有的交易日数，必须为正整数。标签只读取入场和退出两个 open；
任一端点缺失时标签为 `NaN`，不得用 0 伪造有效研究观察。组合逐日收益仍由相邻
open 计算，不能把多日标签当作每日收益重复记账。

`RebalancePolicy` 只接受四个规范值：

| 值 | 可调仓信号日 |
|---|---|
| `daily` | 每个具有完整退出价格的信号日 |
| `weekly` | 完整日期轴上每个 ISO 周的最后一个交易日 |
| `every_5_days` | 从数据集首个交易日起按全局交易日序号 `0, 5, 10, ...` |
| `every_10_days` | 从数据集首个交易日起按全局交易日序号 `0, 10, 20, ...` |

非调仓日必须原样持有上一组合，目标变化、订单数、换手和成本均为 0。周频由完整
日期轴预先解析；在折内或验证子窗口重新从 0 计数属于协议错误。

为了不让同一段未来收益同时进入两个研究观察，允许的 `(frequency, horizon)` 必须
满足相邻调仓信号的交易日距离不小于 `horizon`。因此 `daily` 只允许 horizon=1；
`weekly` 只允许 horizon=1（节假日周的交易日距离可能小于 5）；`every_5_days`
允许 1～5；`every_10_days` 允许 1～10。无效组合必须在配置阶段拒绝，不能静默
抽稀、重叠或改写 frequency。多周期 target 的非调仓列为 `NaN`；学习、验证和
OOS 统计只消费调仓列。这个约束保证任意两个有效标签的开区间收益段不重叠，边界
open 可以相接但不共享收益。

训练/验证/测试时间契约的退出偏移统一为 `1 + horizon`。任何窗口最后一个可执行
信号都必须能在该窗口允许的价格上下文内退出；训练标签不得读取验证或 OOS 价格。

## 2. 排名缓冲

组合构造先按 `(signal 降序, stable_key 升序)` 得到确定性排名。新组合从现金出发
买入 Top-`buy_rank`。之后：

`buy_rank` 和 `sell_rank` 必须是正整数，且 `sell_rank >= buy_rank`；零或负排名
不是“空组合”快捷方式，必须在配置阶段拒绝。

1. 已持有且仍合格的股票，只要没有跌出 Top-`sell_rank` 就保留；
2. 只有进入 Top-`buy_rank` 的未持有股票可以补足空位；
3. 正常组合最多持有 `buy_rank` 只股票；旧状态超过上限时，按排名保留最优者并在
   diagnostics 中记录 `legacy_position_wind_down`；
4. 无分散度横截面不触发信号调仓，仍遵守强制退池和卖出受阻规则；
5. 买入受阻股票不能成为新持仓，卖出受阻的减仓必须延后并占用组合预算。

生产配置使用“买入 Top-20、跌出 Top-30 才卖”。为迁移旧调用，若只提供历史字段
`top_n` 而没有显式 `buy_rank`/`sell_rank`，则 `buy_rank=top_n` 且
`sell_rank=buy_rank`，保持旧的无缓冲语义；不得暗中给旧程序加 Top-30 缓冲。

## 3. 统一 PortfolioConstructor

`PortfolioConstructor` 是信号到目标权重的唯一生产实现，输出不可变的
`PortfolioOutput`：目标权重、买卖权重、选中股票、是否调仓、原因和 diagnostics。
支持两种方法：

- `equal_weight`：每个目标成员权重为
  `min(1 / buy_rank, single_weight_cap)`，不足部分留现金，不向上归一化；
- `optimizer`：复用 `PortfolioOptimizer`/CVXPY/OSQP，继承单票、行业、Beta、
  size、ADV、现金和换手约束；排名缓冲限定可优化成员，仍在缓冲内的持仓不得被
  优化器卖出。

两种方法共享同一后处理顺序：

1. PIT 合格性、排名缓冲和交易阻断；
2. 构造原始目标；
3. 强制持有卖出受阻仓位并只向下缩放新买入；
4. 对非强制变化应用 `target_weight_change_threshold`；
5. 对非强制变化应用 `min_trade_amount`；
6. 对已有组合的非强制变化应用 L1 `turnover_budget`，按原始变化同比例缩小；
7. 缩放后再次执行最小交易金额过滤并量化到 1e-12。

退池、非法超限和卖出阻断解除后的风险降低交易是强制变化，不得被最小金额、权重
阈值或换手预算阻止；它们单独计入 diagnostics。首次从现金建仓标记为
`initial_funding`，不受换手预算限制；换手预算衡量已有组合的更替，不是注资。

每次输出至少记录：`rebalance_due`、`rebalance_executed`、`order_count`、
`turnover`、`buffer_survivors`、`threshold_dropped`、`min_trade_dropped`、
`turnover_budget_scale`、`forced_exit_count` 和 `legacy_position_wind_down`。
回测汇总必须暴露调仓次数、订单数、被抑制交易数和平均换手，关键路径写结构化日志。

## 4. 四条消费路径的一致性

reward、完整 backtest、golden parity 和 simulation 不得私有实现选股或等权函数：
它们必须消费 `PortfolioConstructor` 的 `PortfolioOutput`。给定相同信号、日期轴、
PIT mask、阻断 mask、资本和前权重时：

- reward 与完整回测逐日 target weights、buy/sell weights、换手和成本必须逐元素一致；
- golden 的 free path 直接执行这些 target weights；lot path 只允许因整手、可用现金、
  T+1 和费用融资产生已归因差异；
- simulation 使用同一 constructor；非调仓日不得因价格漂移生成“对齐目标”的订单。

多周期标签仅用于 IC、方向、质量门和研究标签，不改变上述逐日资金曲线记账。

## 5. 生产默认与 10 万资金

`config/ashare_config.yaml` 的 P3 默认是：日频、horizon=1、等权、买入 Top-20、
跌出 Top-30、单笔最少 5,000 元、目标权重变化至少 1%、已有组合单次 L1 换手预算
20%。因此 10 万元首次最多生成 20 笔约 5,000 元订单，之后由排名缓冲、阈值和
换手预算抑制小额高频更替；不得回退为默认 30 笔微型日频订单。

固定裸因子比较必须提供同一信号、同一费用、同一 PIT universe 下的四个象限：
`daily/weekly x equal_weight/optimizer`。四行记录实际 frequency、method、horizon、
权重/成本/订单指标；这只是研究测量，不以“pytest 通过”代替收益或成本结论。

## 6. 版本和旧产物策略

P3 改变奖励组合、协议日历/标签和执行目标，版本同步提升：

- `REWARD_VERSION`: 13 -> 14；
- `PROTOCOL_VERSION`: 21 -> 22；
- `EXECUTION_SPEC_VERSION`: 1 -> 2。

新策略、训练选择、协议、裸因子和 parity 产物记录 reward/protocol/execution 版本及
组合配置。缺少 `execution_version`，或任一对应版本不匹配的旧策略/协议产物，仍可
读取和归档，但必须标记 legacy；promotion 拒绝，simulation 明示警告，禁止自动
转换成 v14/v22/v2 证据。旧 target weights 若没有逐日 constructor provenance，
不得参加 P3 golden parity 声明。旧 `top_n` 配置按第 2 节的兼容规则读取，不改写
原文件。

## 7. 验收测量

除全量测试外，P3 交付必须报告：

1. 同一确定性信号在 reward/backtest 的最大权重差、最大成本差、订单数差；
2. 每种频率的调仓索引和所有有效标签区间的最大重叠数；
3. 10 万默认配置的首次订单数、后续平均订单数、被阈值/金额/预算抑制数；
4. 一个固定裸因子的四象限净收益、成本、平均换手和订单数。
