# WP18 侦察图（只读侦察，2026-09-27）

本文件是 WP18 冻结计划前的只读侦察产物。所有路径相对 `D:/minequant/AlphaGPT/`（p46/wp16c 测量脚本在 `D:/minequant/reports/` 下，只读）。未运行任何回测/测试/搜索器；全部结论来自本次会话内读取的代码、历史产物与日志（关键命令见 §8）。

---

## 1. medium/slow 运行入口与惯例

### 1.1 入口

| 层 | 位置 | 说明 |
|---|---|---|
| r01 驱动 | `ashare_model/r01_baseline.py:942` `run_r01_baseline(...)` | 全生命周期链（idea→SPEC_LOCKED→DATA_QUALIFIED→报告），臂由 `run_baseline_arms`（:259）执行 |
| r02 驱动 | `ashare_model/r02_matrix.py:683` `run_r02_matrix(...)`；round2 `:1196` `run_r02_periods(...)` | 均要求 `contract_sha256` 与 docs 契约文件匹配（fail-closed，:740-748） |
| 面板核心 | `r02_matrix.py:185` `run_r02_panel(loader, proto_cfg, folds, backtest_config, ...)` | 格×臂×折×成本档的单一实现；返回 `(rows, directions_by_fold)` |
| CLI 包装 | `scripts/r02_matrix.py`（头部 docstring 引用 run_r02_matrix）；`scripts/r01_baseline.py` | 退出码 0/1/2 |
| 行引擎 | `ashare_model/eval_metrics.py:143` `evaluate_signal(signal, loader, fold, bt_cfg)` | 唯一引擎行路径：单因子/组合/现金控制全走它（`r01_baseline.py:15-19` 模块契约） |

### 1.2 格/臂/折机制（r02_matrix）

- 六格 `R02_GRID`：`r02_matrix.py:85-92`，耦合参数冻结 `buy_rank=top_n`、`single_weight_cap=1/top_n`、`sell_rank=⌈1.5·top_n⌉`。
- **参考格 `R02_REFERENCE_CELL = "n20_m5000"`（`r02_matrix.py:96`）**——注意任务简报写"n10_m1000 参考格惯例"，与代码不符：n10_m1000 只是格表第一格；r1_medium_a.yaml 的 backtest 块（top_n 20 / min_trade 5000，`configs/r1_medium_a.yaml:59-63`）与 n20_m5000 逐字一致。p46/wp16c 测量脚本用的是**第三个**格 n10_m2500（`D:/minequant/reports/p46/run_measurement.py:61-64`）。三个"参考"并存，冻结计划必须点名用哪一个（见 risks）。
- 每折行集合（`run_r02_panel`，`r02_matrix.py:219-367`）：
  1. `benchmark:equal_weight`（cost_level="ref"，格无关，:256-261）；
  2. **参考格 1x 单因子行**经 `eval_search.baseline_candidates`（:268-294）——这一步同时产出该折的方向与方向证据（`directions_by_fold`，:295）；
  3. 每成本档 × 每格：单因子（参考格 1x 跳过防重复，:317-318）+ `R01_COMBO_ARMS` 组合（:335-357，`combo_signal` 等权平均秩）；
  4. `control:cash` 全 NaN 信号仅在参考格测一次（:358-367）。
- 方向冻结惯例：**每折在参考格学一次，跨格/跨成本档复用**（:263-295；reporting_notes `direction_reuse`，:995-1001）。`r02_matrix` 成本档只有 `("0x","1x")`（`R02_COST_LEVELS`，:99）——**2x 只在 r01 家族**（§3）。
- 折机制：`configs/*.yaml` `protocol.folds` 五折（2021..2025 测试年），经 `r01_baseline.resolve_folds_for`（:1457）→ `eval_folds.resolve_folds`（`eval_folds.py:114-147`）解析为列索引契约 `FoldTimeContract`。

### 1.3 slow 域机制

- slow 运行 = **同一个 r01 驱动** + `configs/r1_slow_ab.yaml`：domain `slow_fundamental`、every_20_days/horizon 20、`baseline_signals: [ROE, PE_TTM, REVENUE_YOY, DIVIDEND_YIELD, MARKET_CAP]`（`configs/r1_slow_ab.yaml:38-50`）；八单因子全部在 `FEATURE_NAMES`（`ashare_model/vocab.py:52-111` 逐名核实）。
- 历史产物 `data/r1/r01_slow_baseline/`（RETIRED，worth_search=false，run_meta 亲读）。**slow 历史组合臂全部失败**：报告行 `failure_reason: "KeyError: 'MOMENTUM_20'"`（亲读 report JSON，30 行 combo 全 failed）——`R01_COMBO_ARMS`（`r01_baseline.py:88-91`）成员是 medium 因子，slow 运行的 directions 字典里没有它们；该历史缺陷正是 WP14 precheck 的 documented case（`experiment_integrity/precheck.py:18-25` 引 `r01_baseline.py:380-391`）。
- **WP18 的 slow 五因子等权组合是真正的新实现面**：按 `r02_matrix.py:335-357` 的模式直接用 `combo_signal([(dir_member, factor), ...], mask)` 构造，成员用 slow 五因子与各自方向；不能走 `R01_COMBO_ARMS`。

### 1.4 factor_tensor_cache 复用

- 接线：`ashare_model/data_loader.py:746-770`——`factor_cache_for(self.config.data_dir)`，身份 = v2 duckdb 路径 + dataset_id + dates + ts_codes（`load_cache.py:79` `loader_factor_cache_identity_with_reason`）。命中时跳过 PIT/资金面重建（:776-790），日志打印 `factor cache outcome=hit|miss|invalidated`。
- 既有缓存目录：`data/r1/medium_a/factor_tensor_cache/`、`data/r1/slow_ab/factor_tensor_cache/`（ls 亲证）。**WP18 复用既有 config 的 data_dir 即命中**；新 data_dir 或改 dates/ts_codes ⇒ 冷重建（medium 实测 ~45 分钟，见 §1.5）。注意 RunStore 把 run 目录写进 data_dir（`run_store.py:357-361` `<root>/runs`），r0x 驱动另有 `out_root` 参数把测量副本写到新目录（`r02_matrix.py:1126-1140`）——WP18 产物照此写 `data/r1/wp18_*/` 或 `reports/wp18/`。

### 1.5 单行回测近似运行成本（实测，来自 run_meta + 日志时间戳分析）

| 运行 | 行数 | 墙钟 | 分解 |
|---|---|---|---|
| r02_matrix（medium，6 格面板） | 310 | 3369.1 s（run_meta 亲读） | 日志 `logs/r02_matrix_20260919_170253.log`：总 56.1 min；**因子缓存 miss 建 tensor 2730 s（45.5 min）**；305 条引擎行 span 2.68 min ⇒ **≈0.53 s/引擎行（medium）**；每折 baseline_candidates 方向腿 ≈12-14 s（3 单因子，± 双向批） |
| r01_slow_baseline（slow） | 125 | 1374.3 s（run_meta 亲读） | 日志 `logs/r01_baseline_20260921_013821.log`：span 22.7 min，**缓存 hit**，首条引擎行在 8.4 min 后（duckdb 装载≈8 min），190 条引擎行 ⇒ **≈4.5 s/引擎行（slow）** |

推论：缓存命中时，medium 域 WP18 正式评价 ≤300 行的引擎工作为分钟级；固定开销是数据装载（hit 下 ~8-10 min）。slow 域单行约贵 8.5 倍。**缓存 miss 是最大的单点成本**（45 min），配 data_dir 必须沿用。

---

## 2. 方向规则 B 的现有形态（A/B 对照）

权威模块 `ashare_model/direction.py`（`DIRECTION_RULE_VERSION = 1`，:43）：

| | Rule A `reference_return` | Rule B `label_ic` |
|---|---|---|
| 比较键 | `(val_reward, val_icir)` 字典序（`_rule_a_key`，:159-168；决策委托 `CandidateScorer.score_many`，:392-418） | 训练窗内信号 vs **固定因果标签** 的逐日截面秩 IC 均值的符号（`label_ic_direction`，:109-156） |
| 取证窗口 | 训练窗尾部的验证子窗 `val_windows`（`train_windows.py:65-104`：validation_fraction 尾巴切成 validation_splits 个不相交窗） | `train_signal_range` 切片；p46 实测形态 = **`(train_signal_start, val_start)`，即验证段之前的训练切片**（`D:/minequant/reports/p46/run_measurement.py:103`；`eval_search.py:147-150` 同一约定） |
| 参考配置 | 记录 `portfolio_config_provenance`（:353）——资金/最低订单/格可达决策 | **接口无任何配置参数**（:109-118 签名；p46 测试 N2 钉住该签名性质） |
| 证据字段 | 单批评分批双侧原值：`val_reward± / val_icir± / train_reward± / train_icir± / objectives± / reward_clip / tie_break_reason / canonical_orientation_locus`（`reference_return_provenance`，:249-355） | `{"ic_mean", "n_dates"}`（:155） |
| 正式命名 | 有：`REFERENCE_RETURN_RULE_NAME = "参考配置下的验证 reward 择向（Rule A）"`（:51） | **无正式中文名**，只有 id `label_ic`（:46）——冻结计划需铸造 |
| 兜底 | 平局走镜像稳定典范定向（首个有限非零观测为正，:90-106） | IC 均值 0/非有限走同一典范定向（:141-146） |

**B-fulltrain 是否存在**：全仓 grep `fulltrain|full_train|全训练窗`（代码+docs）零命中（唯一无关命中是 research_doctor 的训练时长估计）。**B 的全训练窗（含验证尾）变体不存在于任何代码或文档**；若 WP18 保留它，是全新政策身份，不是既有物。B 的既有唯一形态就是"验证段之前的训练切片 IC"。

p46 测量落盘 `D:/minequant/reports/p46/direction_measurement.json`（亲读）：顶层仅 `folds + config`；`folds[i].directions = {rule_a, rule_b, ic_mean_train, agreement}`，无 B 的窗口/命名登记。实测 A/B 分歧大（首折 4 成员仅 RSQ_60 一致）。wp16c 复测（`reports/wp16c/run_measurement.py` + `combo_pct_measurement.json`）只走 Rule A 单批证据（`member_direction_evidence`），未涉及 B。

---

## 3. 成本档 0x/1x/2x 的配置面

- 成本键在 **BacktestConfig**（来自 config 的 `backtest:` 节）：`commission_rate, min_commission, stamp_tax_rate, transfer_fee_rate, slippage_rate`（`configs/r1_medium_a.yaml:66-70`；slow 同构 `r1_slow_ab.yaml:65-69`）。
- **逐行切换的唯一权威是 `r01_baseline.cost_variant(backtest_config, level)`（`r01_baseline.py:113-138`）**：
  - `0x`：五个成本参数全部置 0（:121-129）；
  - `1x`：config 原样（:119-120）；
  - `2x`：`commission_rate/stamp_tax_rate/transfer_fee_rate/slippage_rate ×2`，**min_commission 不变**（:130-137）；
  - 未知档 ValueError（:138）；调用方 config 永不被原地修改。
- 惯用法：先格变体再成本变体，`cost_variant(cell_variant(backtest_config, cell), level)`（`r02_matrix.py:312-314` 现成范式）。
- 注意：`R02_COST_LEVELS = ("0x","1x")`（`r02_matrix.py:99`）——**r02 面板从不跑 2x**；三档全集 `COST_LEVELS = ("0x","1x","2x")` 只在 r01 家族（`r01_baseline.py:93`）。WP18 的 2x 压力行须直接用 `cost_variant`，不能指望 r02 常量。

---

## 4. hold 对照缺口确认与最小实现面

**行号证据（亲读）**：

- `ashare_portfolio/constructor.py:333-335`：
  ```python
  if not rebalance_due:
      selected = tuple(np.flatnonzero(prev > _WEIGHT_EPSILON).astype(int))
      return self._output(prev, prev, selected, "not_due", diagnostics)
  ```
  该早退发生在 `:357-359` 的 `forced_exit = held & ~eligible & ~sell_blocked` 计算之前 ⇒ **非到期会话上强制风险退出从未执行**，失资格持仓要等下一个到期日。
- 引擎侧同构门：`ashare_model/backtest.py:272` 读 `rebalance_mask[t]`，`:275-285` 把 `rebalance_due` 传入 construct，**:291 `if rebalance_due:` 把整个执行腿（share 重导出、卖、买、费用入账）全部关在门内**；else 分支零成本（:387-394）。

**最小实现面建议（须 p56 测试先行；默认关闭、逐字节保持既有基线）**：

1. `constructor.construct`：把 `forced_exit` 计算上提到 not_due 分支之前；新增 opt-in 关键字（如 `not_due_forced_exit: bool = False`）——为 True 且非到期且 `forced_exit.any()` 时返回 `target = prev.copy()`、`target[forced_exit]=0`、reason 如 `"not_due_forced_exit"`；默认 False 路径字节不变（红线：既有 p35+ 数字不可动）。
2. `backtest.py:291` 的门放宽为 `if rebalance_due or output.reason == "not_due_forced_exit":`，使卖腿+费用核算在这些会话执行（构造上买单为空）。
3. hold 模式政策建议：**保留正常 rebalance_mask**，hold 模式下构造器在首建仓后的到期会话只返回 `prev` 加 forced_exit 清零（跳过 survivors/entrants 再选择）——动态/持有于是只在"主动换股腿"上不同，共享同一会话网格与费用口径；备选是首建后抑制到期并用 (1)+(2) 补强制退出腿。
4. 观测面：`evaluate_signal` 的 funnel 键（`eval_metrics.py:295-336`）把这类会话计入 `funnel_not_due_sessions`——WP18 报告若需区分，加**additive** 新计数键，不改既有键名。

---

## 5. IC / 分位收益 / 有效覆盖摘要：现有工具与缺口

现有（均为单一权威）：

- **IC**：`diagnostics.rank_ic_stats`（`diagnostics.py:100-143`）——逐特征 `ic_mean / ic_abs_mean / icir / n_dates`（n_dates 即"有效日期"）；`evaluate_signal` 每行同时给 universe-masked IC 与 tradable IC（`eval_metrics.py:194-218, 352-357`）。
- **覆盖**：`diagnostics.factor_coverage`（`diagnostics.py:73-97`）——PIT mask 门控的非中性格比例；p46 脚本另有逐日 N_eff 统计范式（`run_measurement.py:110-122`）。
- **信号质量**：`signal_quality.py:38-158` HAC 方差 / effective_n / robust_icir / block bootstrap / effective_stock_count。
- **行内全套**：`evaluate_signal`（`eval_metrics.py:143-398`）一行同时含引擎指标 + 双 IC + funnel + lot gap（`lot_gap_fields`，:82）+ `daily_returns/daily_dates`（拼接与年化分解的原料）。
- `bare_factor_backtest.py`（:69/:268/:343 main）是独立的裸因子回测路径——**WP18 账户评价不得用它**（必须走 `evaluate_signal` 引擎同构路径），仅可参考。
- `aggregate_results`（`eval_metrics.py:501+`）的 quantile 是**行分布分位**，不是截面分位收益。

**缺口：分位收益（quantile spread）工具不存在**。全仓 grep `quantile|quintile|decile|fractile|spread` 仅命中 winsorize 诊断与无关统计（亲证）。WP18"纯信号摘要含分位收益"需要一个**新增 additive 助手**：按日把带向信号分位分桶、对固定因果标签求桶均值、产出 Q 高低差序列与均值（消费 `reward.rank_ic_series`/`diagnostics` 同一秩权威与 `causal_target_returns` 同一标签权威），不触引擎。

---

## 6. 试错记录 API 与 WP16-C oriented_evidence 落盘

**三选一结论：`ledger.py` 的 `ExperimentLedger`（经 `RunHandle.record_trial`）是登记方向/规则/格点/组合选择的正确入口。**

- `experiment_tracking.py` = 可选 MLflow 桥（`log_run`，:55-115；结构化拒绝，不装不炸）——**不是**选择登记处。
- `elite_archive.py` = 搜索器精英公式档案（`EliteFormula/EliteArchive`，:21-116）——只收公式候选，**不是**。
- `ledger.py` = 追加式、哈希链 JSONL 试错账（模块头 :1-16 明言"every trial is recorded — no manual omission of failed trials"）。`LedgerEntry` 自带 `algorithm / candidate / seed / fold_train_end / fold_test_end / metrics / payload`（:74-94）——`algorithm="direction_rule:label_ic"`、`payload={规则、格、组合、决策与证据}` 即可登记**每次**选择尝试（含失败/落选者，满足"不能只登记最后留下的候选"）。API：`record_trial`（:269）与 `trial()` 上下文管理器（:353-361，异常路径自动记 failed）。
- 绑定方式：`RunHandle` 为每个 run 挂账本在 `<data_dir>/runs/<spec_id>/<run_id>/trials.jsonl`（`run_store.py:221-222`，`TRIALS_FILENAME` :53）；协议运行经 `evaluation.py:909` 等消费。WP18 驱动开 run 后直接 `handle.record_trial(...)`。

**oriented_evidence 落盘物（WP16-C 双向证据链）**：

1. 产出：`candidates.py` `score_many(..., oriented_evidence_out=[])`（:390、:419、:664-667）——单批交织 ±signal，双侧原值；被拒候选有 `_rejected_oriented_row`（:670+）。
2. 行级：`eval_search.baseline_candidates` 把完整 `DirectionDecision` 序列化进**每条 baseline 行的 `direction_provenance` 键**（`eval_search.py:173-212`）——r01/r02/r03 报告行因此自动携带（p46 B3 接线）。
3. 独立产物：`AlphaGPT/reports/wp16c/combo_pct_measurement.json` 的 `member_direction_evidence`（`reports/wp16c/run_measurement.py` §member directions）。
4. **Rule B 的证据目前只落在 p46 测量 JSON**（`direction_measurement.json` 的 `ic_mean_train`）——没有任何 r0x 报告行携带 rule-B provenance；precheck 认得规则 id（见 §7）但无驱动落盘。WP18 若按 B 择向，需把 `label_ic_direction(...).to_json()` 写进行/账本。

---

## 7. experiment_integrity CLI 对新计划的使用方式

CLI：`scripts/wp14_integrity.py`（docstring :7-19；parser :122-190）：

```
wp14_integrity.py materialize --experiment <id> [--config PATH] [--repo-root PATH] [--out PATH]
wp14_integrity.py precheck    --plan PATH
wp14_integrity.py reconcile   --plan PATH --report PATH [--run-meta PATH] [--out PATH]
wp14_integrity.py status      --plan PATH --report PATH [--run-meta PATH] [--out PATH]
wp14_integrity.py errata      --plan PATH --report PATH --out-dir data/r1/r01_slow_baseline
```

退出码：0=产出（reconcile/status 的 findings 是合法产出）；1=校验拒绝/预检问题；2=用法错。写面卫兵：除 errata 外**永不写 `AlphaGPT/data/` 下**（:58-73）——WP18 的 plan/out 都放 `reports/wp18/` 或新数据目录。

各阶段对新计划的含义：

- **materialize**：`materialize_plan(experiment_id, config_path)`（`plan.py:649-708`）从 L1 契约/L2 config/L3 代码常量重建 RunPlan（臂已按格/期展开、各带 direction_rule 与 cost ladder），`expected_keys` = cross(arms, cost_levels, folds)（:710）。
- **precheck**：`run_precheck`（`precheck.py:589`）查臂形/域成员/折/成员词表/**方向证据覆盖**（`_check_direction_evidence` :415——组合成员必须是某单因子臂）/ **方向规则枚举**（`_check_direction_rules` :443-471，权威 = `direction.py` 的 `{reference_return, label_ic}` + `DIRECTION_RULE_VERSION`，:187-197）/组合引用。**运行计划必须过此关**（仓库事实）。
- **reconcile**：`reconcile_report(plan, report, run_meta)`（`reconcile.py:653+`）按唯一键对账 missing/duplicate/unexpected/failed 行 + 契约/config sha 溯源锚定。**运行后必须过此关**。
- **status**：`classify_status`（`status.py:180`）分类报告。
- 预算核对点：`RunPlan.expected_row_count = Σ len(arm.cost_levels)×len(folds)`（`types.py` RunPlan property）——**≤300 正式评价行上限可在物化计划上机械核对**。

**新计划来源声明方式**：`EXPERIMENT_IDS` 是封闭四元组 `("r01_slow_baseline","r02_matrix","r02_periods","r03_combos")`（`types.py:161`）；`plan_from_json` 拒绝未知 id（`plan.py:868-873`）；`materialize_plan` 查 `_EXPERIMENT_SOURCES`（:95 起，含 contract/config/code_constants/product/adjudication_path/summary_layer_path/key_axes）、`_ARM_EXTRACTORS`（:501）、`VERDICT_MAPS`（:45）。**WP18 必须 additive 地注册一个新 experiment_id：改 `types.EXPERIMENT_IDS` + `plan._EXPERIMENT_SOURCES` 新条目（指向 WP18 契约/新 config/wp18 驱动常量/新产物路径）+ 新 `_ARM_EXTRACTORS` + 新 `VERDICT_MAPS`**，随后 `materialize --experiment wp18_*` 出计划 JSON 供 precheck/reconcile/status 消费。臂的 `direction_rule` 字段必须取 `direction.py` 枚举值（precheck 强制）。

---

## 8. 本次侦察执行的命令（可复核）

1. `ls` / `find` 定位 r01/r02 模块、脚本、data/r1 产物、logs、reports/{p46,wp16c}。
2. 亲读代码：`r02_matrix.py`、`r01_baseline.py`(1-430+函数索引)、`direction.py` 全文、`eval_metrics.py:143-398`、`eval_folds.py` 全文、`constructor.py:280-429`、`backtest.py:240-419`、`experiment_tracking.py` 全文、`ledger.py:1-360`、`elite_archive.py` 头部、`eval_search.py:72-217`、`candidates.py:640-690`、`precheck/plan/types/reconcile/status` 相关段、`train_windows.py:65-104`、`data_loader.py:740-790`、`diagnostics.py:73-143`。
3. 亲读文档/契约：`docs/p46_direction_provenance_draft.md` 全文、`docs/p46_direction_provenance_measurement_log.md`（grep 命中段）。
4. 亲读历史产物（只读 JSON）：`data/r1/r02_matrix/r02_matrix_run_meta.json`（310 行/3369.1s）、`data/r1/r01_slow_baseline/r01_baseline_run_meta.json`（125 行/1374.3s；combo 行 `KeyError: 'MOMENTUM_20'`）、`D:/minequant/reports/p46/direction_measurement.json`（结构+首折方向）。
5. 日志时间戳分析（python 脚本，只读解析）：`logs/r02_matrix_20260919_170253.log`（56.1 min 总量、45.5 min 缓存 miss、305 引擎行 2.68 min ⇒ 0.53 s/行、每折方向腿 12-14 s）、`logs/r01_baseline_20260921_013821.log`（22.7 min、hit、8.4 min 装载、190 行 ⇒ 4.5 s/行）。
6. grep 验证：`label_ic*` 全仓调用点（仅 direction.py / precheck.py / test_p46）；`fulltrain|full_train|全训练窗` 零命中；`quantile|quintile|decile|spread` 无截面分位收益工具；vocab 八因子名逐一对上。

**未运行**：任何 pytest、任何回测/驱动、任何搜索器（只读侦察边界）。
