# P60 实现与验收口径

契约fb9c7c9先独立提交。新增测试只因P60新增观察/拒绝接口而登记，无旧测试修改。
RED命令为`python -m pytest -q tests/test_p60_execution_experiment.py`，预期缺失
execution_experiment接口导致一个collection error，未执行研究账户。

现行引擎、费用、构造、频率和标签零改动。task2 CLI可选导出NPZ无pickle，
独占创建，输入加载与六账户默认行为不变。task3驱动复用它，严格核对任务二
所有冻结输入和六账户终值/费用；旧引擎仅在隔离进程内从精确干净历史checkout
加载全套执行依赖，记录实际路径和文件hash。所有算法直接调用已有权威。

旧引擎原生NPZ完整保留实际buy/sell/cost/wealth/positions，报告适配器只读重建
现金及费用，核对每期费用和期末现金，然后复用P59人民币汇总。旧拒绝记录为
null不可用。最大单股权重只能读取6位旧记录（误差界5e-7），不用于收益筛选。

新增门禁仅是本轮频率实验进入条件；所有分散度和正被动增量必要条件未同时
通过即停止该档。它不接promotion、策略账户resume或自动执行。不提供频率
配置绕过/标签改义，也不为无候选的后续阶段增加生产代码。

目标回归包括新P60、P59、P58、backtest、P35、golden parity、artifact versions、
sharding、time contract；compile及diff。全量/跨运行时/Web未运行时明确报告。
本地历史checkout可保留作精确重放，未改变原工作区及生产DB。
