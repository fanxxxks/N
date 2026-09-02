# P11 Reward v15 四搜索器复刻测量日志（A 线正式测量，Stage A/B/C）

- 任务：t32（AgentTeams alphagpt-p0-p1，measurer）
- 契约：docs/p11_reward_v15_contract.md（APPROVED，t8/t40 双轮门禁）——本文只记录实际发生的事实；预注册内容一律见契约，二者不混写（§11.1）
- 性质：**研究测量**（fold 0 上的 P10 四搜索器复刻 + v15 裁决；非 promotion/非 paper；不构成 alpha 或晋级证据）
- 被测 merge SHA：**a9b2111883d19d1fcc99728c5bd86fb05fd94830**（A 线 reward v15 实现/集成 commit，t21 审核 pass + t26 全量 1419 passed 验证所在树；campaign.git_commit 与 git log 双重一致）
- dataset_id：**b7b4dd4b03fef19755814530dfbead040d6fb88137a3fc5f3dc4992526b64377**（t12 窗口① + t20 窗口②后的 manifest 现值，运行时实测与 identity/行级三重一致；data 截止 daily_bar 20260901）
- 运行结果 headline：**S1 pass / S2 成立（12/12 终点不同值 + 3 行首达 >35%×B）/ 四臂 area 极差 15.29%（P10 <0.4%）→ 按 §9.2 判定 v15 区分度修复有效；Stage B/C §8 映射 headline = `negative_no_admissible_formula`（合法负结果，与 P10 同一映射行）**

## 1. 运行身份

| 项 | 值 |
|---|---|
| run_id | `8a607af854bc42f3a11da4424445d806`（campaign_status=**completed**，12/12 行 succeeded，not_run=0，finished 18:18:36） |
| 命令 | `D:\minequant\.venv\Scripts\python.exe -m ashare_model.searcher_bench --run-dir data/p11_searcher_comparison --budget 2000 --seeds 42,7,2024 --rl-steps 8 --fold 0 --window-cap 300x400 --wall-cap-hours 7` |
| config hash | `9273ad43fa1c320684492e3a4cc2a0224f9dd637472c096604d4ef428394773e`（SHA-256 over config/ashare_config.yaml @ a9b2111；无 runtime_overrides.yaml、无 .env——p0 树与 p11meas 树同内容同哈希实测核对） |
| 版本集合 | protocol 25 / **reward 15** / grammar 5 / feature_registry 5 / factor_compute 1 / research_domain 2 / model 3 / search_contract 3 / searcher_bench 3 / semantic_cache 1 / data_tier 1 / execution_spec 2 / portfolio_constructor 1 / admission_rule 2 / ledger_schema 1（仅 REWARD_VERSION 14→15 为研究语义变更，其余与 P10 相同） |
| 匹配设计 | B=2000 唯一语义评价/行；paired seeds [42,7,2024]；fold 0（train_end 2020-12-31 → test_end 2021-12-31）；window cap (300,400)；RL split 8×250（random-init）；非 RL 2000×1；GP node cap 11（SEARCH_CONTRACT 3）；行序 seed 升序 × (gp,tpe,random,rl) |
| 设备/环境 | CPU 强制（CUDA_VISIBLE_DEVICES=-1 → torch.cuda.is_available=False，行级 device=cpu 12/12）；Python 3.13.12（sys.executable=D:\minequant\.venv\Scripts\python.exe）；torch 2.11.0+cu128 构建（运行时隐藏 GPU，与 P10 的 +cpu 运行时差异如实披露）；6 torch threads；Windows，16 GB RAM |
| 数据库 | D:\minequant\AlphaGPT\data\ashare.duckdb（经 ASHARE_DUCKDB_PATH 绝对路径绑定；只读连接；G1–G7 formal 数据资格门于 bench 入口 require_production() 独立重跑通过——t20 窗口②后数据双保险核验） |
| 研究域 | protocol.domain="unified"（配置默认，与 P10 同一配置源；§6 兼容语义） |
| 输出目录 | `data/p11_searcher_comparison/`（campaign.json 2.8 MB + ledger.jsonl 26 条 + adjudication.json；全新目录，未触碰任何 P10 产物） |
| 墙钟 | 12 行搜索墙钟合计 **13,946.8 s ≈ 3.87 h**（7h 上限的 55%；各段口径见 §7） |
| Stage B/C | `python -m ashare_model.searcher_adjudication --run-dir data/p11_searcher_comparison --fold 0`，脱离式进程运行，wall 418.6 s（≤1h 上限），P10_ADJUDICATION_VERSION=1 |

## 2. §7/§8.2 公平要素核对表（逐项）

| 要素 | 执行证据 |
|---|---|
| 同一数据 | dataset_id=b7b4dd4b 行级 12/12 一致（analysis `dataset_id_uniform_across_rows=true`）；campaign identity + 行级双记录；行间漂移 fail-closed 检查在位（未触发）；入口 require_production() G1–G7 formal 全过 |
| fold/window | fold 0、train_end 2020-12-31、window_cap (300,400)（identity 字段逐行一致） |
| research domain | 同一配置源 protocol.domain="unified"（P10 同源） |
| 候选评分器 | 每 row 单一 trainer → 单一 SemanticBudgetEvaluator + CandidateScorer（v15 两段惩罚唯一路径 candidates.py）；四臂共享，唯一自由变量=searcher+seed |
| 词表/语义规范化 | a9b2111 树实测：vocab size 114、EOS 113、特征名 73（合法 61=73−12 deprecated）、算子 39、feature_version ce3cf72b4af8、GRAMMAR 5、FEATURE_REGISTRY 5、SEMANTIC_CACHE 1——与 P10 契约 §1 记载逐项一致（只读导入探针实测） |
| 最大公式长度 | max_formula_len=12、GP node cap 11（SEARCH_CONTRACT 3）；token 上限不变量：12 行 histogram max = 12 ≤ 12 ✓ |
| 唯一预算计数器 | 唯一语义评价计数（invalid/dup 不计费）：12 行 requested/consumed/invalid/dup 全量入 campaign.json（§3 表） |
| paired seeds | [42,7,2024]（identity.seeds），行序固定 |
| execution/portfolio/Reward 配置 | 同一 config hash 单源；top_n=20、single_weight_cap=0.05、initial_capital=100000、日调仓、horizon=1、validation_fraction=0.35/splits=4；v15 reward（clip ±10、bad −20、penalty 0.05、free_bill 3.0）四臂共享 |
| 同一设备 | 行级 device=cpu 12/12（CUDA_VISIBLE_DEVICES=-1 强制；与 P10 校准口径一致） |
| 转述证据包（reviewer-lead，archive 口径 e5f63ff/a6fd653） | (1) build_action_mask 探针：feature_ids=[74,75,76,77] → step0 合法集恰为该四元；全词表 mask 65=77−12 deprecated；(2) GP build_pset 终端集补测：族⑤ 4 名全部在位、单一注册终端枚举无第二路径。**归属标注**：t32 未独立重跑，按 reviewer-lead 转述录入；两 tag 在 vocab.py 的 41 行差异属 legacy-decode 版本派发路径（e5f63ff plan-A 终态），探针所涉采样/掩码路径生产布局等同。该证据包验证的是**后修复（family-⑤ 解锁后）布局**，适用于后修复树的正式运行 |
| F2-(a) 条件义务处置 | t46→e5f63ff 修复链实证走了 (a)（触 vocab.py，plan-A 冻结 grammar-5 布局+版本派发，取代 plan-B d0d41d8）。按 reviewer-lead 条件，新布局重探针已由该证据包完成（上条）；**t32 测量树（a9b2111）为 family-⑤ 解锁前的 P10 v5 布局**（61 合法特征，ids 74-77 在本树不存在），本树四臂一致性为结构性保证（每行单一 FORMULA_VOCAB 实例）+ token 上限不变量实测。条件性重探针义务对最终 main 树（≠e5f63ff）经 captain 裁定转移至 t31/t59 台账（收官后事项），t32 不承担 |

## 3. Stage A 逐行结果（12 行）

| seed:searcher | consumed/req | proposals | invalid | sem.dups | 终止 | 停滞 | wall_s | s/eval | RSS MB | 选拔 val_reward | 曲线终点 | 首达最大值 (预算, %B) | area |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 42:gp | 111/2000 | 2120 | 1158 | 5 | proposal_stagnation | three_generations_without_new_semantic_class | 92.3 | 0.831 | 2208 | 4.6456 | 4.6456 | 38, 1.9% | 9229.9 |
| 42:tpe | 2000/2000 | 4822 | 36 | 2773 | budget_exhausted | — | 2632.7 | 1.316 | 2409 | 4.8253 | 5.2201 | 1313, 65.7% | 9260.4 |
| 42:random | 2000/2000 | 3868 | 0 | 1868 | budget_exhausted | — | 1075.9 | 0.538 | 2192 | 4.1616 | 5.0548 | 302, 15.1% | 9736.6 |
| 42:rl | 1176/2000 | 2000 | 7 | 817 | steps_exhausted | — | 639.7 | 0.544 | 2921 | 4.7525 | 4.7525 | 467, 23.4% | 8906.3 |
| 7:gp | 108/2000 | 1880 | 1613 | 36 | proposal_stagnation | three_generations_without_new_semantic_class | 57.0 | 0.528 | 2175 | 4.4824 | 4.5253 | 56, 2.8% | 8965.0 |
| 7:tpe | 2000/2000 | 4791 | 33 | 2750 | budget_exhausted | — | 2693.8 | 1.347 | 2468 | 4.7426 | 5.1363 | 299, 15.0% | 10110.8 |
| 7:random | 2000/2000 | 3854 | 0 | 1854 | budget_exhausted | — | 1201.7 | 0.601 | 3127 | 4.5484 | 4.8339 | 177, 8.9% | 9507.3 |
| 7:rl | 1215/2000 | 2000 | 12 | 773 | steps_exhausted | — | 780.9 | 0.642 | 3863 | 4.1655 | 5.3032 | 460, 23.0% | 10335.8 |
| 2024:gp | 315/2000 | 6760 | 262 | 5720 | proposal_stagnation | three_generations_without_new_semantic_class | 831.7 | 2.640 | 3125 | 3.8623 | 4.5910 | 237, 11.9% | 8875.1 |
| 2024:tpe | 1364/2000 | 3737 | 13 | 2357 | **proposal_stagnation** | 50_trials_without_new_semantic_class | 2007.3 | 1.471 | 3157 | 5.4868 | **7.2118** | 1358, 67.9% | 11247.7 |
| 2024:random | 2000/2000 | 3979 | 0 | 1979 | budget_exhausted | — | 1169.3 | 0.585 | 4013 | 4.7878 | 4.9405 | 1638, 81.9% | 9573.2 |
| 2024:rl | 1216/2000 | 2000 | 13 | 771 | steps_exhausted | — | 764.5 | 0.629 | 4801 | 6.0255 | 6.0255 | 402, 20.1% | 11724.6 |

（2024:tpe 的停滞原因串 `50_trials_without_new_semantic_class`——tpe 臂 50 连续无新语义类规则在 P10 从未触发，本次为该规则首次实测触发，作为观察事实记录。）

汇总事实（契约 §4.4/§8.3 口径，不做裁决性解读；裁决见 §6/§8）：

- **曲线终点 12/12 互不相同**（4.5253–7.2118，12 个不同值）：P10 基线为 12/12 同 = 0.98 精确平台。0.98 结构封顶在 v15 下消失；选拔 val_reward（经 eligibility gates + 4 窗口中位数 + tie-break）3.8623–6.0255，同样全部越过 v14 结构上限。
- **首达最大值坐标**：1.9%–81.9%×B，其中 3 行 >35%×B（42:tpe 65.7%、2024:tpe 67.9%、2024:random 81.9%）；P10 基线全部 ≤35%（最大恰 35.0%）。
- **GP 三种子仍全部 proposal_stagnation**（111/108/315 = 5.6%/5.4%/15.8% of B；P10 为 199/99/279 = 10.0%/5.0%/14.0%）——停滞是提案多样性属性，与 reward 量纲解耦，行为结构与 P10 一致；停滞行 wall 57–832 s。
- **TPE 两种子耗满预算**（2632.7/2693.8 s），一种子（2024）在 1364 停滞；dup 率 57.4–63.1%（P10 55.8–60.0%）。
- **Random 三种子耗满预算**（零 invalid，dup 率 48.1–49.7%）；**RL 三种子 steps_exhausted**（unique 58.8–60.8%，P10 64.0–64.5%）。
- 12 行 peak RSS 2175–4801 MB，机器无资源事件。
- 12 条 selected 公式（tokens/text/direction/val_reward/val_icir）完整保存于 campaign.json 供 Stage B/C；选拔集 12/12 条均含算子（11 条多算子组合 + 1 条单算子包装 STD60(MOMENTUM_20)），无一为裸因子——与 P10 精英档案被裸因子（bill=1.0）触顶占据形成对照：bill>1 组合在 v15 下可表达其奖励优势（§1.1 缺陷修复的直接行为学证据）。

## 4. Stage B/C：OOS 评价、硬门槛与活动门逐行披露

评价路径：`evaluation.evaluate_formula` 单一路径（全历史执行 → fold-0 测试窗 (2020-12-31, 2021-12-31] 切片 → evaluate_signal 全成本引擎），direction = 搜索时 selected.direction（12/12 捕获，无重拟合）。等权基准行（同窗）：年化 13.78% / 回撤 7.58% / Sharpe 0.784（与 P10 基准 13.77%/7.58%/0.784 一致——同一 fold 窗口的基准复算一致性检查通过）。

| seed:searcher | 年化 | 回撤 | Sharpe | Calmar | 测试窗 icir | 超额 | 换手 | 调仓/单/抑制 | 方向 | 门槛 |
|---|---|---|---|---|---|---|---|---|---|---|
| 42:gp | 4.47% | 19.98% | 0.105 | 0.224 | 0.128 | −7.84% | 0.50% | 4/24/6725 | −1 | ✗ |
| 42:tpe | **22.20%** | **11.41%** | **1.043** | **1.947** | 0.133 | **+7.07%** | 0.54% | 3/26/8504 | −1 | **✓ 全过** |
| 42:random | 11.99% | 20.80% | 0.491 | 0.577 | 0.010 | −1.50% | 0.46% | 3/22/8211 | −1 | ✗ |
| 42:rl | 3.68% | 10.00% | 0.114 | 0.368 | 0.137 | −8.50% | 0.50% | 5/24/6195 | +1 | ✗ |
| 7:gp | 7.85% | 11.68% | 0.323 | 0.672 | 0.106 | −4.99% | 0.50% | 4/24/7758 | −1 | ✗ |
| 7:tpe | 1.18% | 11.02% | −0.048 | 0.107 | 0.140 | −10.62% | 0.44% | 2/21/8795 | +1 | ✗ |
| 7:random | 1.67% | 7.61% | −0.023 | 0.220 | −0.002 | −10.20% | 0.54% | 4/26/8878 | −1 | ✗ |
| 7:rl | 7.89% | 9.73% | 0.348 | 0.811 | −0.078 | −4.96% | 0.46% | 3/22/8784 | +1 | ✗ |
| 2024:gp | 10.45% | 9.48% | 0.673 | 1.102 | 0.088 | −2.80% | 0.41% | 1/20/6270 | −1 | ✗ |
| 2024:tpe | 0.41% | 12.64% | −0.092 | 0.032 | −0.105 | −11.27% | 0.54% | 5/26/7547 | −1 | ✗ |
| 2024:random | −3.04% | 16.81% | −0.254 | −0.181 | 0.080 | −14.18% | 0.54% | 4/26/8596 | −1 | ✗ |
| 2024:rl | 4.12% | 16.11% | 0.118 | 0.255 | 0.102 | −8.14% | 0.41% | 1/20/8906 | −1 | ✗ |

臂级聚合（契约 §5.3，n=3 仅方向性描述，禁止显著性声明）：

| 臂 | area 中位数 | admissible（≥2/3 全过） | oos_positive（≥2/3 icir>0） | stagnant |
|---|---|---|---|---|
| gp | 8,964.96 | ✗（0/3） | ✓ | ✓（唯一停滞臂） |
| tpe | 10,110.82 | ✗（1/3） | ✓ | ✗ |
| random | 9,573.16 | ✗（0/3） | ✓ | ✗ |
| rl | 10,335.83 | ✗（0/3） | ✓ | ✗ |

- **§8 预注册裁决 headline = `negative_no_admissible_formula`**（matched row：no_arm_admissible_negative_result；无 additional findings）：唯一全过四门槛的行是 42:tpe（1/3，不足臂级多数）。与 P10 裁决（同 headline）对照：v15 恢复了搜索期区分度（S2），但**未**改变"当前预算/窗口/成本下无臂产出可过门槛公式族"的 OOS 结构——P10 §8 预注册的改进方向（评价器/Reward 与成本-换手结构）仍然成立。
- **组合活动门逐行披露**：12 行全部为 1–5 次实际调仓/年、20–26 笔订单、6270–8906 笔被抑制、日均换手 0.41–0.54%——与 P10 基线（2–6 次、21–33 笔、5940–9064、0.44–0.68%）结构一致的近静态书。42:tpe 虽单行过门槛，但其 3 次调仓/年、0.54% 换手的近静态剖面按 p11 契约 §5.3 与 AGENTS §8.5 不得单独支撑"有效"声明（臂级裁决亦为负）。
- seed 间方差依旧极大（同一臂年化 −3.04% → +22.20%），n=3 无显著性声明（契约 §5.3）。

## 5. 预注册判据裁决（契约 §9，运行前固定，逐条执行）

| 判据 | 结果 | 证据 |
|---|---|---|
| **S1**（结构性，测试级） | **pass** | RED-0 在 REWARD_VERSION="15" 下通过（bill≤3 组合与裸因子同顶）——t21 在被测树 a9b2111 上亲验（git archive 实树 RED 4 failed+1 passed → 实现 5/5 绿）；本 campaign 选拔 val_reward 3.86–6.03 全部越过 v14 的 0.98 结构上限，为生产级行为佐证 |
| **S2**（平台化消失） | **成立（两个析取支同时成立）** | (a) 12 行终点不同值数 = **12**（≥2）；(b) 3 行首达最大值坐标 >35%×B（65.7%/67.9%/81.9%；P10 基线全部 ≤35%） |
| **S3**（四臂 area 可分辨，辅助记录） | **描述性成立** | 四臂 area 中位数 8,965.0 / 10,110.8 / 9,573.2 / 10,335.8，极差 **15.29%**（P10 基线 <0.4%）；n=3，跨版本不可比（reward 量纲已变），禁止显著性声明 |
| **§9.2 裁决映射** | **第一行命中**：S1 通过 且 S2 成立 → **v15 区分度修复按预注册判据判定有效；P11 记录为正面结果**；四臂 matched 结论仅限本次运行（run_id 8a607af8）；映射原文提及的 t34 已随用户裁定取消（任务规模收敛），A 线测量的完整收口即本 campaign |
| 负结果合法性 | 不适用（S2 成立为正面结果）；但 Stage B/C §8 映射的 `negative_no_admissible_formula` 为**独立合法负结果**，两者并行不悖：搜索期区分度修复 ≠ OOS 可过门槛 |

## 6. P1-3 活动门收敛核查（结论：**收敛关闭**）

**存在性（代码级，a9b2111 实测）**：

1. 组合活动门于引擎单一路径：`ashare_model/backtest.py` L294–309（average_turnover / rebalance_count / **rebalance_due_count** / order_count / suppressed_trade_count 记入 engine metrics）；评价端四字段披露面：`ashare_model/eval_metrics.py` L182–184（P10 §5.2 additive observability，v15 零改动继承）。
2. 成本-换手在评价内：reward 三条公式化路径精确扣除年化执行成本（`ashare_model/reward.py` L805/L910/L954，cost_weight=1.0 诚实成本语义，v15 未触碰）。
3. 晋级端：换手硬门 max_average_turnover=0.15 与回撤/容量门为既有 promotion 门禁；B 线 G7（registry status fail-closed）已落地 main @ 466bcd5（**后继集成，不在被测树**）；活动门向晋级硬门槛的进一步扩展属 p12/后继契约辖区。

**生效证据（本 campaign 12 行实测）**：

- 四字段逐行落 payload 并入 adjudication.json（§4 表"调仓/单/抑制/换手"列，12/12 行全量真实值非占位）。
- 披露机制照常工作：12 行近静态剖面（1–5 调仓/年、换手 0.41–0.54%）被完整暴露，42:tpe 的单行过门槛被正确按"近静态不得单独支撑有效"处置——对照 P10 基线的近静态问题（2–6 调仓/年、5940–9064 抑制），该机制正是 p11 §5.3 预注册的披露与裁决落点。
- 唯一未单独聚合的 AGENTS §8.5 维度是"持仓暴露"——原始逐日 positions/target_weights 在 BacktestResult 完整可观测（单一语义路径产物），聚合暴露指标属后继活动门执法设计观察项（已随收官后事项条目入 t31/t59 台账记录）。

**结论**：活动门已纳入 v15 评价路径且生效证据完整（契约 §5.3 评价端落点按预注册执行），**P1-3 收敛关闭**。

## 7. 事故时间线、资源上限与墙钟核算（§10 执行情况）

**三次外部击杀 + 三次恢复（DSH 运行时回收后台作业树；已由 captain 定性为平台缺陷，非测量代码缺陷）**：

| 时刻（+08） | 事件 |
|---|---|
| 05:29:19 | 启动 #1 fail-closed：production gates 处 DuckDB 拒绝（错误原文指向 miniconda python PID 16404 持有读写锁）——t20 窗口②（04:03–05:31）尾段在位。该 PID 于 05:30:12 前退出、命令行未捕获，精确归属不可定论（同期存在运行时 miniconda 泵进程模式）；t20 写入命令链经 data-ops 确认为 venv 解释器 |
| 05:30:55–05:41 | 启动 #2（run_id 89e2521f）：gates 05:31:03 通过（写者已释放的机器证明），gp 行完成（121.4s），tpe 行进行中 |
| ~05:55 | 按 captain 指令受控停止（"等窗口②关闭"relay 为迟到通报——窗口②实际已于 05:31 关闭；见 §8 处置披露）。attempt 目录原样保留于 `data/p11_aborted_attempt_run89e2521f_b7b4dd4b/`（campaign status=running + ledger 2 条目），定性：中止的工程尝试，不入研究证据 |
| 05:56–06:05 | 启动 #3（**正式 run 8a607af8**）：gates 05:56:13、campaign 创建 06:04:06、gp 行 06:04:10–06:05:42 完成（92.3s，与 attempt #2 同坐标行 best 4.6456 逐位一致——同树同数据同 seed 可复现性直接实证） |
| 06:05:42 后 ~06:30–10:51 间 | **外部击杀 #2**（运行时回收）：无 WER 崩溃记录、无 python 层异常、job 注册表丢失、python 进程全消失；ledger seq=3 tpe:42 running 孤儿条目原状保留 |
| 10:55–11:02 | 恢复 #1（进程内 resume）：身份校验 rows-done=1 of 12，tpe:42 新 trial seq=4 started 11:02:05 |
| ~11:1x | **外部击杀 #3**（同机制）：ledger seq=4 孤儿条目原状保留 |
| 14:21:52 | 恢复 #2（**脱离式进程**：WMI Win32_Process.Create，进程树根 WmiPrvSE.exe PID 21776，与 agent 会话零关联；启动批处理 data/launch_resume3.cmd）：身份校验通过，tpe:42 新 trial **seq=5 started 14:27:36** |
| 18:18:36 | **campaign completed**：12/12 行 succeeded，bench exit 0，adjudication 随后 22:38 完成（脱离式，wall 418.6s） |

**墙钟核算（契约 §10.1"仅搜索墙钟计入"）**：

- 12 行 row wall_seconds 合计 = **13,946.8 s ≈ 3.87 h**（7h 上限的 55%）——搜索墙钟口径；两次击杀空窗（~06:30–10:55、~11:1x–14:21）为空闲，不计入搜索墙钟，时间戳如上完整披露。
- resume 段 campaign 内部墙钟（payload wall_seconds_total）= 13,859.8 s（post-load 起算至 18:18:36）。
- 墙钟上限：未触发（行间核算持续执行）；**校准偏差熔断：未触发**（三块 块均/投影 = 0.764× / 0.814× / 0.821×，全部 <2.0）。
- fail-closed 触发表：dataset_id 漂移 0 次；production gates 拒绝 1 次（05:29:19，写窗口在位——正确行为）；backend_error 行 0（击杀为进程级，行未产生错误行记录，中断 trial 以 running 原状保留）。
- 重跑规则：resume 为设计内机制（同 run_id、append-only ledger、中断 trial 不改写）；被中止 attempt（89e2521f）独立目录原样保留，未覆盖未拼接。

## 8. 处置披露（诚实记录）

1. **t34/t35 取消**：用户裁定任务规模收敛，7h 大预算运行及分析取消（t34/t35 attempt 0 cancelled，从未启动、零产出）；A 线收口即本 campaign。本取消不构成对任何契约判据的修改。
2. **受控停止一役**：05:55 的停止系 captain"等关窗通知"relay 迟到所致（窗口②已于 05:31 关闭，被中止 attempt 实际绑定最终数据集 b7b4dd4b）；处置本身按 §4.3 纪律执行（进程树确认退出、零 DB 连接残留、字节级保留、append-only）。常驻纪律已立：冲突指令先确认后动作。
3. **dataset 时序**：e15b4fc4（t12 窗口①，02:15）→ **b7b4dd4b（t20 窗口②，05:29:56 重建，最终值）**；本测量绑定 b7b4dd4b（identity/行级/manifest 三重一致）。t12 日志"应绑定 e15b4fc4"的预期被 t20 回填合法取代（fundamental_pit 205,589→215,951 行）；a9b2111 树不消费新增两列（family-⑤ 未解锁），搜索空间与 P10 逐项等同。
4. **torch 构建差异**：P10 运行时 torch 2.11.0+cpu，本次 venv 为 2.11.0+cu128（t1 环境修复后基线）但以 CUDA_VISIBLE_DEVICES=-1 强制 CPU（cuda_available=False 实测）——设备语义（cpu、6 threads、确定性）与 P10 校准口径一致，构建串差异如实披露。
5. **2024:tpe proposal_stagnation**：tpe 臂 50 连续无新语义类规则首次实测触发（P10 未触发）——作为观察事实记录，不构成臂间不公平（预算/计数器/词表全要素同构）。
6. **gp:42 行跨进程归属**：该行在启动 #3 进程完成，其余 11 行在恢复 #2 进程完成——两进程同树（a9b2111）、同 config hash、同 dataset（resume 身份校验强制 dataset_id 一致）、同 device，行级语义同构；campaign identity 机制保证无行间漂移。

## 9. 验证与工具边界

- t32 零代码/零 tracked 文件改动；被测树 a9b2111 的全部验证矩阵由 t21/t26 承担（RED 4+1、矩阵 98 passed、串行全量 1419 passed、compileall、diff --check、分片 91/91）。
- 本次使用工具：searcher_bench v3（被测树代码）、searcher_adjudication v1（同）；只读分析脚本 `data/p11_stage_a_analysis.py`（复用 `searcher_adjudication.best_so_far_area` 单一权威，无第二套 area 语义；输出 `data/p11_stage_a_analysis.json`）；只读 DB 审计脚本 `data/t32_db_audit.py`。以上脚本位于 gitignored data/，属工程分析工具，非仓库代码。
- 分析输出关键断言自检：dataset_id_uniform_across_rows=true；row_devices={cpu}；token_cap_invariant_max_hist_len=12。

## 10. 研究声明边界

1. 本日志与全部产物只支持：**v15 区分度修复的预注册判据裁决（S1/S2/S3 = 有效）**、四搜索器行为测量（预算消耗结构、停滞、长度分布、墙钟、area 可分辨性）、OOS 回测硬门槛与活动门披露、P1-3 收敛结论。
2. `negative_no_admissible_formula`（Stage B/C）是合法负结果：当前 B=2000 预算、fold-0 窗口与全成本日调仓协议下，无搜索器产出臂级可过门槛公式族；**不证明**任何搜索器"更差"，不构成对生产默认搜索器（gp）的变更依据，不构成 alpha/晋级/lifecycle/production 结论。
3. 42:tpe 单行四门槛全过 + 超额 +7.07% 为单 seed 观察（臂级 1/3 未过多数），且为近静态书（3 调仓/年、0.54% 换手）——按 AGENTS §8.5 与 p11 §5.3 不得单独解读为有效。
4. v14（P10）↔ v15（本测量）对照为**描述性**（不同 reward 量纲、不同 dataset_id），禁止宣称跨版本 matched；本 campaign 内四臂对照为 matched（§7 全要素，唯一自由变量=searcher+seed）。
5. fold 0 测试窗为协议常规使用窗口，非新锁定 holdout；paper/sim、lifecycle 状态转换、生产默认搜索器变更均不在本任务范围。

## 11. 原始产物与命令索引

| 产物 | 路径（gitignored，未跟踪） |
|---|---|
| campaign.json（SEARCHER_BENCH v3 完整 payload） | D:\minequant\AlphaGPT-p11meas\data\p11_searcher_comparison\campaign.json |
| ledger.jsonl（26 条 append-only：12 succeeded×2 + 2 孤儿 running 击杀痕迹） | 同目录 ledger.jsonl |
| adjudication.json（P10_ADJUDICATION_VERSION=1） | 同目录 adjudication.json |
| 只读分析输出 | D:\minequant\AlphaGPT-p11meas\data\p11_stage_a_analysis.json |
| 脱离式启动器与控制台捕获 | data\launch_resume3.cmd、data\launch_adjudication.cmd、data\p11_resume3_stdout/stderr.log、data\p11_adjudication_stdout/stderr.log、data\p11_resume3_exit.txt |
| 中止 attempt（不入研究证据） | data\p11_aborted_attempt_run89e2521f_b7b4dd4b\{campaign.json, ledger.jsonl} |
| bench loguru 日志 | logs\searcher_bench_20260902_{055613,105556,142157}.log（+导出 txt） |
| 测量 worktree | D:\minequant\AlphaGPT-p11meas（detached @ a9b2111；任务完成后由 captain/集成决定处置） |
