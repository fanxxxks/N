# P61 换手预算诊断测量

用户2026-10-08单独授权。branch=codex/task61-turnover-budget-diagnostic，
base=ed5924b8d761259332518d3981863fafb4f6ee58，prereg=8afb38d，精确实测实现
08773ef885eedf1b8231215178b0457ee37ee2a7。仅本地，未push/PR/merge。
本次延续任务三已提交基础，不改原用户AlphaGPT工作区，不重载或写数据库/cache。
既有reports/task1_precommit_20261007/没有纳入提交。

## 实际命令与输入

```
python scripts/task61_turnover_budget_diagnostic.py --output reports/task61_20261008_exact
```

退出0，账户诊断实际3.78秒（不含解释器/torch导入）。2个正常账户各一次本金
连续1210期，0个压力账户，没有搜索或其它预算。ExperimentLedger包括整体/
输入/2账户，均成功关闭；独立重开hash-chain验证，无失败或遗漏trial。
Windows/Python3.13.12/numpy2.5.2/torch2.11.0+cu128，完整环境见provenance.json。

直接消费P60同一约53.1MB物化NPZ，文件/信号/节奏sha一致；snapshot
d36024cd1b197a8edcf976b2d950b3506bb0fae022f121d6aef06f2a281cde55、冻结年度方向/
标签10日、频率10日、raw交易限制/PIT股票池均保留。没有读新的2026收益。
原0.20正常账户的终值、费用、1210期逐日权益/净回报及577笔真实成交所有字段
精确等于P60。两臂完整config逐字段比较，唯一差异turnover_budget=.20 vs None。
None显式字段通过已有工厂；未省略字段以免域默认值补回预算。

## 结果

|指标|0.20|None|
|---|---:|---:|
|累计净赚人民币|10,755.76|8,634.87|
|扣费年化|2.15%|1.74%|
|最大回撤|11.20%|32.29%|
|平均实际仓位|16.22%|66.67%|
|平均实际持股|5.74|20.25|
|总费用|4,312.75|21,261.45|
|卖出无买入交易日|16|0|
|相对原被动净赚差额|−28,235.83|−30,356.72|

关闭后的年度净收益2021–2025分别1.30%、−9.38%、0.97%、0.11%、17.07%；
同本金原被动净赚38991.59元/年化7.10%/回撤38.83%。完整年度双臂及被动表
见RESULTS.md，实际累计双边换手18.60→95.68倍，实际成交577→2874笔。
佣金2885→14370元、印花税468.91→2260.66元、过户费18.80→90.80元、
滑点940.04→4540.00元。费用分项来自实际成交，不调低正常费率。

诊断符合预注册分支restored_without_passive_edge：预算确实显著压低仓位，
但关闭后净赚下降、费用和风险扩大，没有主动净增量；停止当前信号修复路线。
没有进入条件性压力，也不实现简单预算规则/任务四/trade_groups。P60所有
历史文件hash核对未变；本次为独立诊断，旧结论未撤销或覆盖。
仍有39.52%建仓后日期不足20只，不声称稳定达到既定分散目标或可部署。
项目复权财富近似、红利处理和财富重推股数限制不变，非券商精确结算。

## 软件及账本验收

RED：新增诊断模块缺失，1个预期collection error；旧测试零修改，新17个测试
服务唯一开关/明确关闭与省略差别/真实约束/实际日期计数/输入和基线拒绝/条件分支/
单年度依赖判据，未mock核心算法、未新增skip或放宽容差。

精确08773ef执行：

```
python -m pytest -q tests/test_p61_turnover_budget_diagnostic.py tests/test_p60_execution_experiment.py tests/test_p59_continuous_account.py tests/test_p58_small_capital_execution.py tests/test_backtest.py tests/test_p44_priority_turnover_allocation.py tests/test_golden_parity.py tests/test_ci_sharding.py tests/test_time_contract.py
python scripts/check_test_shards.py --check
python -m compileall -j 0 -q ashare_data ashare_model ashare_portfolio ashare_trading scripts webapi
git diff --check
```

172 passed/0 failed/0 skipped/0 warnings，28.74秒（exact_tests.txt）；127测试文件/
4分片完整覆盖，compile/diff通过。两账户费用、终值、年度复利链、现金非负、
实际买入整手及最低1000元均独立从CSV对账，期末缺价持仓为0；account_audit.json。

不变版本execution6/constructor2/rebalance2/protocol29/reward15/summary2；
既有支持的参数值对照不改变算法，新的诊断报告不用于promotion/resume。
全量pytest、其它Python/CPU-CUDA参考运行时、Web/完整CI-equivalent未运行，
因为仅本地目标范围交付且无这些模块修改或推送授权；不声称push-ready。
后置证据提交只保存本轮报告/原始逐日成交/配置/ledger/审计，不覆盖历史证据。
