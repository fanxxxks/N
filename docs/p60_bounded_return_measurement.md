# P60 任务三测量记录

2026-10-07。本地分支codex/task3-bounded-execution，从任务二
0de60820ccbecbbe1a9c08bbeb870c829cabb6e1延续；预注册fb9c7c9，实测实现
5b1efa97d8a93cfa1bac914dda134dd8bd026645；历史实现
dbaf7296a41ce9813ae736c0e4879d54d9488b0f。未push、PR、merge。
原工作区未改，原数据库/WAL与P59准备记录hash相同，独立私有primary副本也相同。
旧worktree保持干净、detached，只读重放。既有task1_precommit目录未纳入提交。

## 实际运行

解释器D:/minequant/.venv/Scripts/python.exe，Windows/Python3.13.12/
numpy2.5.2/torch2.11.0+cu128；完整环境及版本表见results.json。

```
python scripts/task3_execution_experiment.py --source-root D:/minequant/AlphaGPT --primary-snapshot D:/minequant/AlphaGPT-task1/data/task2_reference_20261007_clean/ashare.duckdb --historical-root D:/minequant/AlphaGPT-task3-before --replay-directory data/task3_20261007_replay --output reports/task3_20261007_exact
```

退出0，实际248.99秒。新主动3+新被动3+旧主动3，共9个账户、两个执行版本，
全部一次本金连续1210期计价。独立重开三个ledger核对hash chain和所有trial关闭；
root记录整轮及阶段，after记录输入+6账户，before记录3账户，阶段不重复计账户。
未失败、未丢弃trial。频率网格0轮，费用压力0次，策略晋级/未来观察0。
缓存hit，dataset=d36024cd1b197a8edcf976b2d950b3506bb0fae022f121d6aef06f2a281cde55；
三个完整因子字节与P59/P60冻结值相同，信号和10日节奏hash同P59，逐年方向截止
不变。九账户现金均非负，年度复利链/费用分项/终值全部对账，期末持仓缺价为0。
运行中观察到加载子进程PeakWorkingSet约4503MiB（OS计数采样，非全轮内存上界）。

输入NPZ约53.1MB保留在本地data/task3_20261007_replay/input.npz（ignored），路径
及sha见results.json；所有旧原生实际轨迹NPZ、观察重建CSV、新逐日/成交CSV、
配置/数据/版本/来源hash、ledger、审计留在reports/task3_20261007_exact。

## 结果与裁决

|本金|旧净赚|新净赚|增量人民币|新年化|新回撤|新平均实际持股|
|---:|---:|---:|---:|---:|---:|---:|
|1万|不适用|不适用|—|—|—|—|
|3万|3,478.12|57.80|−3,420.32|0.04%|8.45%|2.80|
|5万|−251.32|1,260.47|+1,511.79|0.52%|10.36%|4.98|
|10万|13,021.50|10,755.76|−2,265.74|2.15%|11.20%|5.74|

三档均失败进入条件：实际分散不足、比共同被动持有少赚6357/15804/28236元。
旧3/5/10万分别66/11/8笔买入整手后不到最低1000元，新实现全部0。
这轮整个执行修复未带来一致收益增益，资金利用未恢复到可用分散配置。
按契约停止，不搜频率、不开任务四，不自动扩大或转新路线。详见报告DECISION.md。
共同被动首期来自新路径；不声称旧首期匹配，不将不同仓位/回撤差全归因选股。
这些是既有引擎模型下金额、开发验证，非独立样本外、券商结算或盈利保证。

## 软件验收

RED：新测试因预期缺失模块collection error，见p60实施文档；独立手算现金/费用
fixture未借助被测实现生成oracle。实现中曾误引用不存在的execution_delay配置
字段和cost方法，测试分别暴露AttributeError，改为已有接口后通过；没有改旧测试、
expected、容差、skip或warning过滤。仅新增P60测试，符合新增行为白名单，已登记分片。

精确5b1efa9提交执行：

```
python -m pytest -q tests/test_p60_execution_experiment.py tests/test_p59_continuous_account.py tests/test_p58_small_capital_execution.py tests/test_backtest.py tests/test_p35_share_accounting.py tests/test_golden_parity.py tests/test_artifact_versions.py tests/test_ci_sharding.py tests/test_time_contract.py
python scripts/check_test_shards.py --check
python -m compileall -j 0 -q ashare_data ashare_model ashare_portfolio ashare_trading scripts webapi
git diff --check
```

170 passed/0 failed/0 skipped/0 warnings，33.19秒（原始exact_tests.txt）。
分片126文件/4片完整覆盖，compile/diff通过。当前交易/费用/预测语义无变化，
execution6/constructor2/rebalance2/protocol29/reward15/schema3/summary2未bump；
历史execution4明确单列，不迁移为current策略/模拟账户。

全量pytest、其它Python/CPU-CUDA运行时矩阵、Web及完整CI-equivalent未运行，原因是
本轮仅本地目标范围交付、没有相应模块修改或推送授权；不声称push-ready。
后置证据commit只保留本轮结果和测量，不改变实测算法或覆盖历史结果。
