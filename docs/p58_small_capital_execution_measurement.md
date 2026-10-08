# P58 任务一交付测量

本轮仅完成执行，不判断盈利，不运行任务二至四。授权/验收见
[p58_small_capital_execution.md](p58_small_capital_execution.md)。

## 身份与范围

- 分支：codex/task1-small-capital，独立工作区 D:/minequant/AlphaGPT-task1。
- base：dbaf7296a41ce9813ae736c0e4879d54d9488b0f。
- 前置契约：9dc79688a0b4b048113c9db5ed0eedd800b84e41。
- 实现：5275ab1；精确被测代码：1d276b9fc5b2874bc7b973b81f79c9a2686b8c31。
- 执行版本6（基线4，原工作区已有用户未提交版本5）；构造器2、协议29、
  reward15、schema3不变。旧版本只读/拒绝复用，新配置要求新账户。
- 原工作区 codex/b04-load-cache-performance 未修改、未吸收、未提交；WP18
  原始证据保留在那里。本分支尚未整合用户工作区后续修改，没有push/PR/merge。
- 改动：共享股数差额取整、现金逐笔先卖后买、最低实际成交额、现金费用
  不透支、lot_size读取/provenance、三档配置与针对性测试。未改公式/方向/
  因子权重/标签/费用参数/预算分配，没有trade_groups接线。

## RED 与 GREEN

环境：Windows 11 10.0.22621，D:/minequant/.venv/Scripts/python.exe，
Python3.13.12（Anaconda），torch2.11.0+cu128/CUDA12.8；计算输入为合成
常量价格与明确股价漂移。无网络、无正式数据库读写，无随机搜索预算。

实现前：`python -m pytest -q tests/test_p58_small_capital_execution.py`，
4 failed / 5 passed，7.43秒。四个预期失败为费用导致全体少一手、5000元
浮点边界全过滤、priority与proportional两种换股的账户不一致。随后新增边界
测试。测试中费用人民币转换曾漏乘初始资金，按归一净值定义修正；未扩大
任何数值容差以掩盖不一致。

最终精确1d276b9：

```text
python -m pytest -q tests/test_p58_small_capital_execution.py tests/test_execution.py
  tests/test_portfolio_constructor.py tests/test_p3_portfolio_parity.py
  tests/test_p35_share_accounting.py tests/test_run_sim.py tests/test_golden_parity.py
  tests/test_artifact_versions.py tests/test_backtest.py tests/test_price_semantics.py
  tests/test_ci_sharding.py tests/test_config.py --tb=short
```

实际作为单行命令执行，**245 passed，114.80秒**，0失败、0skip、无pytest
warning汇总。包括正常与资金档位、手续费、最低额、不同整手、代码置换、
普通换股、价格漂移减仓、停牌/涨跌停/缺失限制、无交易日、卖出不足付费、
旧版本拒绝、交付配置与两条普通路径股数/费用/现金对账。

既有测试修改的独立手算与白名单依据见契约末节。未放宽parity容差、删
场景或新增skip。P3小本金最低额反例明确区分计划权重奖励和实际零成交；
旧turnover/order_count指标仍是计划，本轮没有把它们冒称实际。

其他精确提交验证：

- `python scripts/check_test_shards.py --check`：124文件、4分片，全集完全覆盖。
- `python -m compileall -j 0 -q ashare_data ashare_model ashare_portfolio ashare_trading scripts webapi ashare_execution.py`：通过。
- `git diff --check`：通过。
- `python -m scripts.task1_execution_check --output reports/task1_20261007_exact`：通过，
  工程结果kind/provenance/git_commit齐全，tracked_code_dirty=False。
  未跟踪状态只来自新测量输出，原先precommit证据目录保留不覆盖。

未运行：全仓Python测试、最低Python/CI参考环境和Web构建；范围为局部执行
交付、无依赖或Web变更、无push/合并。不得因此宣称全量CI通过或push-ready。

## 结果与边界

完整人民币表：[RESULTS.md](../reports/task1_20261007_exact/RESULTS.md)，
机器原始记录：[results.json](../reports/task1_20261007_exact/results.json)，
表格：[profiles.csv](../reports/task1_20261007_exact/profiles.csv)。

候选每档仅一套，25只/4%/最低1000元/100股；分散度提高，不用集中度制造
可交易。10元定价下：1万不适用；3万买25000、费137.75、余4862.25；
5万买49000、费149.99、余850.01；10万买99000、费175.49、余824.51。
正常费用逐笔扣现金，没有降低佣金或滑点。4%按交易前权益限制，费用与
随后价格漂移会改变交易后占比，这是既有构造语义。

用户指定20只例子：原RED全体400股、买80000元；修正后最低1000元时
19笔500股、1笔400股，买99000元、费150.49、余849.51；若坚持最低5000，
末笔跳过，买95000元、费143.45、余4856.55。两条路径对账通过。

同输入0.20预算、10日原调仓，普通全换股会话：priority卖19000元/买0元/
现金19810.32；proportional卖0元/买0元/现金849.51。前者卖出耗尽计划预算，
后者把各腿稀释到最低额以下；本轮不改变其规则，不把任何一项宣称收益赢家。

单手价格理论上限4/12/20/40元只是机械边界，整户费用也必须付得起。
真实入选价格可能让剩余档位也不适用，不能通过换信号、选低价股或放宽
单票上限解决。尚未运行连续历史账户，故没有年化/超额/回撤或收益改善结论。
下一步决策须用后续任务的实际成交人民币与扣费复利证据。
