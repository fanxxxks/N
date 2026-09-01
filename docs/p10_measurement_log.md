# P10 四搜索器公平对比测量日志（Stage A 搜索运行）

- 任务：t7（AgentTeams alpha-orth-research，search-runner）
- 契约：docs/p10_searcher_fairness_contract.md（APPROVED 2026-09-01，captain 批准记录见 §11）
- 性质：**研究测量**（fold 0 上的搜索器对比；非 promotion/非 paper；不构成 alpha 或晋级证据）
- 本文只记录实际发生的事实；预注册内容一律见契约，二者不混写。

## 1. 运行身份

| 项 | 值 |
|---|---|
| run_id | `a45764aadaab444b9b0fa4a1c60b35a1` |
| campaign_status | **completed**（12/12 行 completed，not_run=0） |
| 时间窗 | 2026-09-01 11:51:02 → 15:45:25（**14062.8 s ≈ 3.91 h**，7h 上限的 56%） |
| 被测实现 commit | `d692b0517de4e841f4bc1f94c4de9b6dc038c4ab`（campaign.campaign.git_commit 与 git log 一致） |
| dataset_id | `a839ecf2284b354a5ab6ed3228d13fc5d7f3d93a2fadba0b08d8c909edf194fd`（与 t2/t5 基线一致；data_end=20260821，未重同步） |
| config hash | 产物内 `config_hash=None`（见 §6 缺口说明）；离线重算 **`cb83f092a5992429fd6733b9b18f0153629d3c792d9511b3a5684f3b3329bb00`**（config/ashare_config.yaml + runtime_overrides.yaml；两文件在运行窗口内未被修改，tracked 树干净） |
| 版本集合 | protocol 25 / reward 14 / grammar 5 / feature_registry 5 / factor_compute 1 / research_domain 2 / model 3 / **search_contract 3** / **searcher_bench 3** / semantic_cache 1 / data_tier 1 / execution_spec 2 / portfolio_constructor 1 / admission_rule 2（不变）/ ledger_schema 1 |
| 匹配设计 | B=2000 唯一语义评价/行；seeds [42,7,2024]；fold 0（train_end 2020-12-31）；window_cap (300,400)；RL split 8×250（random-init）；非 RL 2000×1 |
| 设备/环境 | CPU（torch.cuda.is_available=False）；Python 3.13.12；Windows 11（10.0.22621）；torch 2.11.0+cpu；6 torch threads |
| 命令 | `python -m ashare_model.searcher_bench --run-dir data/p10_searcher_comparison --budget 2000 --seeds 42,7,2024 --rl-steps 8 --fold 0 --window-cap 300x400 --wall-cap-hours 7` |

## 2. 预检（engineering，均独立目录，不覆盖任何既有产物）

1. **CPU 校准**（t6 契约附录 A）：budget 128，data/p10_calibration_engineering.json——B=2000 的标定依据。
2. **真实数据 campaign 冒烟**：budget 64 × seed 42，`data/p10_smoke_engineering/`，run_id `5d78d1b0650141f6a546b4321937107b`，4/4 行 completed（gp 64/64 budget_exhausted、tpe 64/64、random 64/64、rl 54/64 steps_exhausted @8×8），用时 ~12 min。发现 GP 在新长度上限下不再在 64 预算内停滞——长度对齐如实改变了 GP 的有效空间（契约 §4.3 的预期效果）。

## 3. 逐行结果（12 行）

| seed:searcher | consumed/req | proposals | invalid | sem.dups | 终止原因 | 停滞原因 | wall_s | s/eval | peak RSS MB | 选拔 val_reward |
|---|---|---|---|---|---|---|---|---|---|---|
| 42:gp | 199/2000 | 4600 | 3962 | 85 | proposal_stagnation | three_generations_without_new_semantic_class | 107.4 | 0.54 | 3095 | 0.9188 |
| 42:tpe | 2000/2000 | 5255 | 38 | 3155 | budget_exhausted | — | 2574.4 | 1.287 | 3551 | 0.8917 |
| 42:random | 2000/2000 | 3870 | 0 | 1870 | budget_exhausted | — | 1054.0 | 0.527 | 4338 | 0.9050 |
| 42:rl | 1287/2000 | 2000 | 11 | 702 | steps_exhausted | — | 684.4 | 0.532 | 5085 | 0.8950 |
| 7:gp | 99/2000 | 2160 | 1781 | 21 | proposal_stagnation | three_generations_without_new_semantic_class | 54.0 | 0.545 | 4323 | 0.9310 |
| 7:tpe | 2000/2000 | 4705 | 28 | 2623 | budget_exhausted | — | 2390.8 | 1.195 | 4810 | 0.8930 |
| 7:random | 2000/2000 | 3859 | 0 | 1859 | budget_exhausted | — | 1093.6 | 0.547 | 5585 | 0.8957 |
| 7:rl | 1289/2000 | 2000 | 17 | 694 | steps_exhausted | — | 809.5 | 0.628 | 6257 | 0.9030 |
| 2024:gp | 279/2000 | 5600 | 541 | 4134 | proposal_stagnation | three_generations_without_new_semantic_class | 224.5 | 0.805 | 5438 | 0.9570 |
| 2024:tpe | 2000/2000 | 4780 | 33 | 2725 | budget_exhausted | — | 2892.9 | 1.446 | 4848 | 0.8970 |
| 2024:random | 2000/2000 | 3993 | 0 | 1993 | budget_exhausted | — | 1280.7 | 0.640 | 4041 | 0.8970 |
| 2024:rl | 1280/2000 | 2000 | 11 | 709 | steps_exhausted | — | 892.7 | 0.697 | 4742 | 0.8950 |

汇总事实（按契约 §4.4 口径，不做裁决性解读）：

- **best-so-far 曲线终点 reward = 0.98（精确值，12/12 行一致）**：四个臂的搜索级最高验证奖励都到达同一 0.98 平台；因此臂间差异体现在**到达平台的预算坐标与曲线形状**（area 裁决见 t8），而不是终点值。选拔 val_reward（经 eligibility gates + 4 窗口中位数 + tie-break 选出，见上表"选拔 val_reward"列）低于曲线终点，属正常选拔语义（gp 0.9188/0.931/0.957、tpe 0.8917/0.893/0.897、random 0.905/0.8957/0.897、rl 0.895/0.903/0.895）。
- **GP 三种子全部提前停滞**（199/99/279 unique = 请求预算的 10.0%/5.0%/14.0%），停滞原因如实记录；消耗与请求的差额完整可见，未伪装成满预算。GP 的语义重复率波动大（85/21/4134 dup），无效提案率高（3962/1781/541——超尺寸/不合法树被拒）。
- **TPE 三种子全部耗满预算**，是唯一 wall > 2000 s/行 的臂；单评价耗时随 trial 数上升（1.20→1.45 s/eval，Optuna 历史增长）。
- **Random 三种子全部耗满预算**，~0.53–0.64 s/eval，零无效提案。
- **RL 三种子全部 steps_exhausted**（8×250 提案），unique 消耗 1280–1289（~64–65% unique 率），~0.53–0.70 s/eval。
- 12 行 peak RSS（进程级累计轮询，保守上界）最高 6257 MB；机器 16 GB，无资源事件。
- **统一 token 上限实测成立**：全部 12 行的被评公式 canonical 长度（content+EOS）最大值 = 12 = max_formula_len（摘要 per-row histogram）；契约 §4.3 的对齐在生产运行中得到直接验证。
- 12 行 dataset_id 一致（行级字段 + campaign identity 双重记录）；行间除 seed 外身份字段逐项一致。
- 每行 selected 公式（12 条，含 tokens/text/direction/val_reward/val_icir）完整保存在 campaign.json 与摘要中，供 t8 Stage B/C 使用；本文不预设任何臂的优劣结论。

## 4. 资源上限与熔断（契约 §4.5/§6 执行情况）

- **墙钟上限**：Stage A 7h 未触发（实际 3.91h）；行间核算在每行启动前执行。
- **校准偏差熔断**：三个 seed 块的块均墙钟 / 校准投影块均墙钟 = 0.760 / 0.748 / 0.910（seed42/7/2024；块均 = 1105.1s / 1087.0s / 1322.7s，投影块均 = 1453.5s），全部 < 2.0，未触发。逐行实际 vs 校准投影（rate×B）：gp 107/1314、54/1314、225/1314（远低——停滞使 wall 小于投影）；tpe 2574/2024、2391/2024、2893/2024（1.18–1.43×，符合 Optuna 开销趋势）；random 1054/1124、1094/1124、1281/1124（0.94–1.14×）；rl 684/1352、810/1352、893/1352（0.51–0.66×，steps_exhausted 使实际消耗 < B）。
- **fail-closed 触发表**：dataset_id 漂移 0 次；production gates 于入口 require_production 通过（G1–G7，与 t1 状态一致）；backend_error 0 行。
- **断点恢复**：本次为单次连续运行未触发 resume；resume/identity-drift/重试语义由单元测试覆盖（test_p10_campaign.py）。

## 5. Ledger（契约 §7）

- `data/p10_searcher_comparison/ledger.jsonl`：**24 entries = 12 trials × (running + terminal)**，12 trials 全部 `succeeded`，hash 链重新加载校验通过（ExperimentLedger 构造即验链）。
- trial 粒度 = 行尝试（algorithm=`searcher:<backend>`，candidate=`<backend>:<seed>`，payload 记录 requested/steps/batch）；metrics 记录 consumed/wall/termination/stagnation。
- 无覆盖、无删除、无重写；本次无 failed/retry 条目（如发生，按设计以新 trial 追加）。
- 说明：ledger 为 **unbound（v0 条目）**运行——P8 RunSpec/spec_id 绑定属 lifecycle 正式训练路径，本对比 campaign 的完整身份由 campaign.json 的 identity/provenance 承载（契约 §7 未要求 RunSpec 绑定；如 reviewer 要求升级绑定，另立修订）。

## 6. 已知缺口与偏差（如实记录）

1. **config_hash=None**：`_config_hash` 在默认配置路径（CLI 未传 `--config`）下 `Path(None)` 抛 TypeError 被吞，产物记录 None。已修复（commit `8c23e86`，RED：stash 隔离 FAILED→恢复 PASSED；`tests/test_p10_campaign.py::test_p10_config_hash_resolves_default_config_path`）。运行窗口内 config 文件未修改（tracked 干净），离线重算哈希 `cb83f092…` 见 §1；该缺口不影响 12 行搜索结果本身（config 内容与被测行为无关的纯 provenance 字段）。
2. **直方图口径**：行级 `formula_len_histogram` 统计被评提案的**去重 canonical AST** 长度；billing 去重（校准指纹）比 AST 去重更粗（数值等价不同 AST 合并计费一次），故 RL 行直方图总数（1371–1389）略大于 billed 消耗（1280–1289）。两套口径均为真实记录，公式长度上限断言在两套口径下都成立。
3. **peak RSS 为进程级累计轮询上界**（后行继承前行分配），非逐行独立值——searcher_bench 模块文档既有约定。
4. **数据截止 20260821**（bars 落后 6 开市日，t1 MEDIUM 项）：对 fold 0（2021 年测试窗）无影响；dataset_id 与 t2/t5 一致。
5. 运行期日志出现 `RuntimeWarning: All-NaN slice / Mean of empty slice`（reward.py capacity 路径）——**既有 warning 类别**（t2/t5/校准运行均已出现），非 p10 改动新增。

## 7. 验证矩阵（被测实现 commit d692b05 的精确候选树上）

| 验证 | 结果 |
|---|---|
| `python -m pytest -q tests -n auto` | **1374 passed / 5 skipped / 0 failed**（586.20s） |
| `python -m pytest -q tests`（串行，parity 对账） | **1374 passed / 5 skipped / 0 failed**（1017.01s）——并行/串行计数一致（AGENTS §10.2(3) 记录于本条；warnings 630 vs 629 为 xdist 实例倍数差，无净新增类别） |
| `python -m compileall -j 0 -q ashare_data ashare_model ashare_portfolio ashare_trading scripts webapi` | 0 错误 |
| `git diff --check` | 干净 |
| `python scripts/check_test_shards.py` | union == full set（89 文件 / 4 分片；test_p10_campaign.py 已登记 shard-model-1） |
| RED 证据 | test_p10_campaign.py collection ImportError（实现缺失）；test_p4 pin FAILED(2≠3)；config-hash 修复 stash-FAILED→restore-PASSED |
| 新增测试 | tests/test_p10_campaign.py 12 项（版本 pin、GP 长度对齐+满长表达+旧上限 parity、四后端统一上限、campaign 结构/身份/ledger、resume 重试、墙钟/偏差熔断、身份漂移拒绝、JSON 可序列化、config hash） |

## 8. 提交与产物

- `ce98004` 契约（t6，含 captain 批准编辑）；`d692b05` 实现（8 文件，+1104/−132）；`8c23e86` config-hash 修复。分支 `codex/p10-searcher-comparison`（base 403ee15）；**未 push、未合并**。
- 原始产物（git 未跟踪，路径如实登记）：`data/p10_searcher_comparison/{campaign.json, ledger.jsonl}`（campaign.json 含每行完整 SearchResult：requested/consumed、proposals、dups、termination/stagnation、best-so-far 曲线、formula 长度直方图、selected 候选）；`data/p10_calibration_engineering.json`；`data/p10_smoke_engineering/`；`logs/searcher_bench_20260901_114559.log` + `logs/searcher_bench_20260901_154525.txt`。
- 提交产物：`docs/p10_searcher_comparison_20260901/summary.json`（自 campaign.json 派生的逐行紧凑摘要，含 selected 公式 tokens/text/direction，供 t8 使用）+ 本日志。

## 9. 研究声明边界

- 本日志与全部产物只支持**搜索器行为测量结论**（预算消耗结构、停滞、长度分布、墙钟）；臂间优劣裁决（best-so-far area、OOS、回测硬门槛、"下一步改进方向"）属 t8 按契约 §5/§8 预注册规则执行，禁止由 Stage A 数据提前下结论。
- 未验证/未运行项：无（本任务范围内全部完成）；paper/sim、lifecycle、生产默认搜索器变更均明确不在范围。
