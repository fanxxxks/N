# P4 四搜索器统一与 Transformer 改造测量报告

状态：工程验收完成；**不是 alpha 发现，也不是生产 RL 晋级证据**。
契约来源：[p4_search_transformer_contract.md](p4_search_transformer_contract.md)。
原始产物：

- [p4_admission_engineering.json](../experiments/p4_admission_engineering.json)：
  修复后、带 tier gate 的最终 25 臂结果；
- [p4_admission_engineering_pre_cash_fix.json](../experiments/p4_admission_engineering_pre_cash_fix.json)：
  首轮保留的失败证据，未覆盖或删行。

## 1. 结论

1. GP、TPE、Random、随机初始化 RL、imitation RL 已在同一真实 DuckDB、同一
   fold/window、同一候选评分器和同一请求预算下完成五组独立 seed pair；修复后
   25/25 臂成功，未重复固定 baseline 结果。
2. Imitation 在五组都降低 teacher-forcing loss 并提高 token accuracy，但没有通过
   预注册的搜索有效性规则：相对随机 RL，area 胜 5/5、OOS IR 胜 3/5；相对 GP，
   area 胜 0/5、OOS IR 胜 2/5，门槛是每项至少 4/5 且中位数严格更高。
3. 本次使用 32 预算、100×120 窗口，低于预注册生产层 1024 预算、300×400 窗口，
   因而 `registered_tier=false`。最终同时记录
   `metric_rule_passed=false`、`rl_admitted=false`、
   `advanced_rl_allowed=false`，默认搜索器保持 `gp`。
4. P4 不实现或启用 PPO、辅助价值预测、AST-aware embedding。Transformer 仍是
   实验分支；GP/TPE/Random 主链路不读取 archive、不预训练 policy。

## 2. 版本、分支与旧产物策略

开发分支为 `codex/p4-search-transformer`，改动前快照提交为 `8b59592`。P4 串行提交：

| 提交 | 单一职责 |
|---|---|
| `8e2cc0e` | 预注册 P4 契约与失败测试 |
| `a4572c2` | 统一 `SearchBackend` / `SearchResult`，正式注册 TPE |
| `5f89a30` | benchmark v2 暴露统一预算与曲线 |
| `aa7076a` | GP/TPE/Random elite archive |
| `b310c26` | RL 完整诊断 |
| `9d55917` | elite 监督 imitation，再进入 RL |
| `2ef757d` | 五组独立配对 seed 与 v2 Admission |
| `1ae0e41` | 恢复交易过滤后的现金上限不变量 |
| `bdc3561` | 禁止工程层结果冒充生产晋级 |

最终版本：`SEARCH_CONTRACT_VERSION=1`、`ELITE_ARCHIVE_VERSION=1`、
`RL_DIAGNOSTICS_VERSION=1`、`IMITATION_VERSION=1`、
`ADMISSION_RULE_VERSION=2`、`SEARCHER_BENCH_VERSION=2`、
`PROTOCOL_VERSION=23`、`MODEL_VERSION=3`。

迁移/拒绝策略：

- v22/v2 strategy/checkpoint 可保留公式文本供人工审计，但不能冒充 v23/v3
  champion；checkpoint 不自动迁移，必须在 v3 下重训；
- T2 admission 使用固定 baseline seed，不满足 P4 pairing，只能作为历史证据；
- benchmark v1 缺少统一曲线/停滞字段，保留只读，不补造字段、不用于准入；
- elite archive 未知版本明确拒绝；缺失或空 archive 的 imitation 明确失败，禁止
  静默退化成随机初始化；
- `PORTFOLIO_CONSTRUCTOR_VERSION` 保持 1：`1ae0e41` 恢复既有
  `sum(weights) <= 1` 和“只缩新买入”契约，没有重新解释任何合法 v1 输出；受影响
  的旧运行会报错，迁移策略是用修复后代码重跑，不接受旧失败产物。

## 3. 改动前后软件测量

命令均为仓库根目录下 `python -m pytest -q`，没有新增/修改 skip、xfail、retry，
没有弱化断言。

| 状态 | commit/tree | passed | skipped | warnings | wall time |
|---|---:|---:|---:|---:|---:|
| 改动前 | `f4b8773` / `8b59592` 同树 | 1055 | 5 | 616 | 579.72s |
| 最终 | `bdc3561` + 最终未改代码的测量文档 | 1086 | 5 | 618 | 608.90s |
| 差值 | — | +31 | 0 | +2 | +29.18s |

补充验证：

- `python -m compileall -q ashare_model ashare_portfolio scripts`：通过；
- `git diff --check`：通过；
- cash-bound 相关回归：84 passed / 4 warnings；golden/optimizer/parity：
  86 passed / 51 warnings；
- RL/imitation 故障隔离三用例：3 passed / 2 warnings / 8.57s；注入 imitation
  构造失败后 GP 仍返回正式 `SearchResult`，Admission 保留失败行并保持 GP 默认。

pytest 只证明软件行为；研究结论来自下面的真实数据配对测量，不用测试通过替代。

## 4. 数据与门禁

`python scripts/check_production_gates.py` 的 G1–G7 全部通过：历史成分区间、上市日、
交易日历、区间非重叠、strict PIT universe、各年不少于 100 个合格股票、成分区间
日线覆盖均通过；G7 为 2426 个已观察区间、0 个零 bar 区间。

最终工程测量命令：

```text
python scripts/admission_experiment.py --steps 2 --batch-size 16 \
  --window 100,120 --output experiments/p4_admission_engineering.json
```

共同 provenance：

| 字段 | 值 |
|---|---|
| dataset id | `b927074a455a25c65698b61dbee9da48097d3121fc759585e73543c8d56d4318` |
| fold | train end `2020-12-31`；test end `2021-12-31` |
| pair seeds | `42, 7, 2024, 1337, 999`；pair 内五臂使用同一个 seed |
| 请求预算 | 每臂 32 个唯一语义公式评价（2 steps × 16） |
| window | 100 stocks × 120 dates |
| protocol / reward / grammar | 23 / 14 / 2 |
| execution / portfolio | 2 / 1；完整 resolved portfolio config 在 JSON 中 |
| max formula length | 12 |

这是真实生产数据库上的工程层，不是预注册生产准入层。脚本将注册层固定为
8×128=1024、300×400；任何覆盖值都写入
`non_registered_admission_tier` blocker。

## 5. 逐 pair 原始汇总

`area` 是按共同请求预算 32 积分的 best-so-far 均值；提前结束时持有最后 best 到
预算末端。JSON 中曲线 reward 的 `null` 只表示尚无有限 best 的 `-∞` sentinel，
不是缺失 pair；共 9 个这样的早期点。结构审计确认每条曲线从实际预算 1 覆盖到
`consumed`，预算坐标严格递增且 reward 单调不减。

| seed | arm | requested | consumed | termination | stagnation | area | OOS active IR |
|---:|---|---:|---:|---|---|---:|---:|
| 42 | GP | 32 | 32 | budget_exhausted | — | 0.937803 | -0.116817 |
| 42 | TPE | 32 | 32 | budget_exhausted | — | 0.810273 | -0.275134 |
| 42 | Random | 32 | 32 | budget_exhausted | — | 0.575515 | -0.084742 |
| 42 | RL random | 32 | 30 | steps_exhausted | — | 0.769307 | 0.858429 |
| 42 | RL imitation | 32 | 30 | steps_exhausted | — | 0.903000 | 1.549862 |
| 7 | GP | 32 | 32 | budget_exhausted | — | 0.968917 | -0.018913 |
| 7 | TPE | 32 | 32 | budget_exhausted | — | 0.895927 | 0.412147 |
| 7 | Random | 32 | 32 | budget_exhausted | — | 0.935826 | 1.417611 |
| 7 | RL random | 32 | 30 | steps_exhausted | — | 0.875309 | 0.831245 |
| 7 | RL imitation | 32 | 30 | steps_exhausted | — | 0.951844 | 0.596573 |
| 2024 | GP | 32 | 32 | budget_exhausted | — | 0.966486 | -0.247079 |
| 2024 | TPE | 32 | 32 | budget_exhausted | — | 0.900089 | -0.564746 |
| 2024 | Random | 32 | 32 | budget_exhausted | — | 0.902167 | -0.176279 |
| 2024 | RL random | 32 | 28 | steps_exhausted | — | 0.902052 | 0.166672 |
| 2024 | RL imitation | 32 | 31 | steps_exhausted | — | 0.914047 | -0.614373 |
| 1337 | GP | 32 | 32 | budget_exhausted | — | 0.965651 | 0.803682 |
| 1337 | TPE | 32 | 32 | budget_exhausted | — | 0.835974 | 0.000000 |
| 1337 | Random | 32 | 32 | budget_exhausted | — | 0.858021 | -1.090511 |
| 1337 | RL random | 32 | 29 | steps_exhausted | — | 0.841994 | -1.075336 |
| 1337 | RL imitation | 32 | 32 | budget_exhausted | — | 0.900177 | 0.205593 |
| 999 | GP | 32 | 32 | budget_exhausted | — | 0.970731 | 0.744121 |
| 999 | TPE | 32 | 32 | budget_exhausted | — | 0.844372 | 0.443682 |
| 999 | Random | 32 | 32 | budget_exhausted | — | 0.607846 | 0.500869 |
| 999 | RL random | 32 | 29 | steps_exhausted | — | 0.754514 | -0.708620 |
| 999 | RL imitation | 32 | 32 | budget_exhausted | — | 0.810688 | -0.492130 |

所有 baseline 实耗满 32；RL 实耗 28–32，差额来自被准确计数的 semantic duplicate，
没有伪造满预算。25 臂 `error` 全为 null。

## 6. Imitation 与 RL 诊断

| seed | elite samples | tokens | loss before | loss after | accuracy before | accuracy after |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 12 | 144 | 4.699794 | 2.045286 | 0.020833 | 0.750000 |
| 7 | 8 | 96 | 4.500729 | 1.362473 | 0.010417 | 0.822917 |
| 2024 | 7 | 84 | 4.471143 | 1.572077 | 0.261905 | 0.869048 |
| 1337 | 8 | 96 | 4.805254 | 1.633927 | 0.000000 | 0.864583 |
| 999 | 8 | 96 | 4.904519 | 1.890581 | 0.010417 | 0.791667 |

五组都证明了监督目标被学到，但这不等价于搜索/OOS 有效。十个 RL 臂的 run-level
诊断均含 reward 分布、拒绝原因、entropy、semantic duplicate、advantage 方差、
gradient norm、公式长度和算子覆盖；观测范围如下：

- reward count 每臂均为 32，mean 为 -1.434699 到 -1.082648；
- mean entropy 为 2.850138 到 3.376644；
- semantic duplicates 为 0–4，比例 0–0.125；
- mean advantage variance 为 1.184540–3.375378；
- mean gradient norm 为 23.874175–57.422553，单步最大 64.303515；
- mean formula length 为 10.935484–12.000000，算子覆盖 34–38/39；
- 拒绝原因实际出现：`constant_or_near_constant_signal`、`sign_instability`、
  `signal_activity_below_minimum`、`val_icir_below_minimum`、
  `val_reward_below_minimum`、`val_window_q25_below_minimum`、
  `valid_ic_days_below_minimum`。逐臂精确计数和每步分布在原始 JSON 中。

## 7. 预注册裁决

| 对手 | imitation median area | 对手 median area | area wins | imitation median OOS IR | 对手 median OOS IR | OOS wins | dominates |
|---|---:|---:|---:|---:|---:|---:|---|
| random RL | 0.903000 | 0.841994 | 5/5 | 0.205593 | 0.166672 | 3/5 | false |
| GP | 0.903000 | 0.966486 | 0/5 | 0.205593 | -0.018913 | 2/5 | false |

规则要求相对两个对手的两个指标都满足中位数严格更高且至少 4/5 胜。指标规则失败，
并且本次又是非注册工程层，因此最终 blockers 为：

```text
non_registered_admission_tier
metric_rule_failed
```

最终：`default_searcher=gp`、`rl_admitted=false`、
`advanced_rl_allowed=false`。这也直接执行 P4-09：不继续添加高级 RL 机制。

## 8. 测量发现与修复前后不变量

首轮工程测量在 `2ef757d` 上保留了 25 行，其中 TPE/seed 42 触发
`ValueError: prev_weights must not exceed full investment`，其余臂继续完成，证明 RL
或单一 backend 失败不会抹掉 GP 证据。根因是最小交易额/阈值把计划减仓恢复为旧
权重后，没有再次压缩已经计划的新买入。

先增加的失败测试按契约计算：旧仓 `[0.40, 0.35, 0.25, 0]` 中两笔小减仓被抑制，
25k 卖出只能为新标的提供 25k；旧实现却给出 33.33k 新买入，总权重 1.083333。
`1ae0e41` 复用同一个 cash-cap helper，只同比缩小 fresh increases；修复后输出
`[0.40, 0.35, 0, 0.25]`，总权重 1.0。修复后同参数重跑为 25/25 成功，TPE/seed 42
实耗 32 并正常产出 4 条 elite。

最终 JSON 结构审计：pair 内 seed 一致、0 error、每条曲线完整单调、五组 imitation
均改善、tier gate 生效。SHA-256：

```text
1974FADD0D7C678AC9B695E484DA0BFD048AEA67FFD949C03CF530CA7F12B7B9  p4_admission_engineering_pre_cash_fix.json
BE1D82C5B7F800E462A37F3ECB521E9F4F00C62CA962C6E67EA399E38DCE3181  p4_admission_engineering.json
```

上述工程层只验证实现、可观测性、公平配对和否决逻辑；它不支持任何收益、泛化或
alpha 声明，也不能替代未来精确 1024/300×400 注册层的生产准入测量。
