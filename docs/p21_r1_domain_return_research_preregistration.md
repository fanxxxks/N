# P21 R1 分域收益研究预注册契约（五配置统一协议）

- 状态：**DRAFT — 待用户批准**（起草：research-analyst，AgentTeams tushare-pit-migration t25，2026-09-06）。批准后本文保持前瞻性，禁止以实现或测量输出反向修改；测量只写入测量日志（`docs/p21_r1_measurement_log.md`，运行后建立），不回写本文。
- 被执行任务：AgentTeams t25（R1：分域收益研究，独立配置，五协议统一）。
- 仲裁顺序：本文/用户批准 → 契约测试 → 实现 → 测量日志。测量结果只能执行 §10 预注册裁决。
- 本文使用 AGENTS.md 的"必须/禁止/应当"语义。工作分支 `codex/r1-domain-research`（base = `codex/factor-semantic-fixes@1ad7b73`，cherry-pick 前置 `02e361d` = D7 数据资格门 4a88d84、`512a551` = E2 因子缓存 8411811；本地提交，不 push/不合并——push/merge 另需授权）。

## 0. 问题陈述与目标

P6 已把研究域按预测周期拆分（short/medium/slow，`RESEARCH_DOMAIN_VERSION=3`），但生产默认仍是 `unified/daily/h1` 兼容语义；77 因子在 v2 PIT 数据底座上完成重测与语义修复（t18/t22，dataset `157e0cf5…`）之后，**尚无任何按域预注册的 walk-forward 收益测量**。R1 目标：在不改动任何生产默认的前提下，以五个独立预注册配置，对三个研究域 + 一个兼容基线执行统一协议的收益研究，回答：

- **Q1（域内）**：方向对齐基线与公式搜索在各自域的合法执行点上，能否产出 OOS 为正、经成本与多重检验校正后仍然成立的收益信号？
- **Q2（medium 域内）**：在 medium 域内加入 Tier B 两融特征（A+B 对照 A），是否改善域内 OOS 表现？（唯一一次合法的域内配对比较：两配置共享除特征集外的一切协议要素。）
- **Q3（跨配置）**：各域结论只在本域协议内解释；不同 horizon/Reward/执行纪律的 reward 不可横向比较（AGENTS §6.2），跨配置只做预注册的描述性陈述。

**允许且预期可能的结论**：任一域都可以得到"未发现可晋级 alpha"（AGENTS §14；负结果是合法结果，禁止为得出正结论放宽任何预注册要素）。

## 1. 范围与非目标

**范围**：五个预注册配置（§4）的执行与裁决；D7 数据资格门（p17 §12）作为每配置硬前置；tier 纯度审计；DSR/max-t 校正与 append-only trial ledger。

**非目标**：

- 不做任何 promotion/lifecycle/paper/sim 操作；R1 是 research 级证据（AGENTS §4.1），**不构成 alpha 或晋级结论**。
- 不切换生产默认搜索器/协议/域；生产配置文件零改动（验收以 diff 为证）。
- 不新增/修改特征、算子、Reward、执行、费用、组合构造语义；`REWARD_VERSION`、`EXECUTION_SPEC_VERSION`、`PORTFOLIO_CONSTRUCTOR_VERSION`、`GRAMMAR_VERSION` 等零 bump。
- 不做跨域特征迁移：`TURNOVER_STD20`、`LIST_AGE` 维持原属 short 域（§7）。
- 不做 searcher 优劣排序（四个协议搜索臂共享预注册预算，random/gp/tpe 以协议固定 seed 运行，trained 臂使用 [42,7,2024]；臂间比较仅描述性）。
- 不修复数据缺口（§6 的 D7 FAIL 项只能由独立数据任务回填后解除，本文不得为放行运行而放宽 p17 §12 阈值）。

## 2. 不变量（实施与运行全程成立）

1. **生产默认零改动**：本分支相对 base 的 diff 只包含 `configs/r1_*.yaml`、`docs/p21_*`、`scripts/r1_*`、测试文件与 §8 授权的最小协议接线；`config/ashare_config.yaml` 与一切生产默认值零字节改动。
2. **统一协议**：五配置共享同一组五个 walk-forward folds、seeds [42,7,2024]、同一 screening 预算 S（§5）、同一 Reward v15、同一执行/资金/费用（100k，top20，`buy_rank=20/sell_rank=30`，佣金 0.025% 下限 5 元、印花税 0.05%、过户费 0.001%、滑点 0.05%）、同一数据与 universe 契约。
3. **域纪律**：域模式下执行点必须合法（p6 §1.2）；基线信号 ⊆ 域特征；域外特征恒中性；跨配置不比较 reward。
4. **顺序纪律**：每 fold 内先执行并记账方向对齐基线（`baseline_candidates`），后进公式搜索行（random/gp/tpe/trained）；该行序是 `run_protocol` 既有实现行为，运行后以 ledger 行序断言复核（§9）。基线集合与方向在本文预注册（§4 表），禁止消费搜索结果后再补/改基线。
5. **预算记账**：唯一语义评价为唯一预算单位；无效、退化、canonical duplicate、semantic duplicate 不计费（T2-01 语义）；每行记录 requested/consumed、终止/停滞原因、best-so-far、公式长度分布；提前停滞不得伪装成耗满预算。
6. **append-only**：全部 trial（成功/失败/崩溃/提前停止）进入共享 campaign ledger；重跑用新 run ID/新输出目录；禁止覆盖或删除不利行。
7. **fail-closed**：dataset_id 在任一配置运行时与 campaign 首行记录不一致 → 停止 campaign；G1–G8 或 D7 FAIL → 该配置 not_run（记录原因），不静默跳过；degraded 数据显式记录（p19 §2.4 价格约束降级逐股入日志）。
8. **方向权威**：全部基线与被评公式的方向由 `feature_registry` 的 `expected_direction` 与搜索期学习方向决定；DRAFT p20（MARGIN_BALANCE_CHG +1→−1）未获批准，**不适用**——若批准先于运行落地，则以运行 commit 的注册表为准并记录。

## 3. 假设与前提

- 前置任务已完成：t22（因子语义修复裁决）、t19（D7 资格门可用）、t20（E2 缓存可用）。
- 数据状态：v2 store dataset_id `157e0cf5…`（t68 收口后），legacy 生产库（universe/calendar/gates 源）daily 止 2026-09-01。
- E2 因子缓存已在本分支可用但**不接入**协议运行路径（协议每行新建 trainer，tensor 由 loader 单次构建；缓存消费留待工程任务接线）——如实披露，不宣称使用。
- 机器：6C/12T CPU、16GB RAM，串行运行；与 t62（P14 测量）并发时的墙钟按含竞争的保守投影解读（§5）。

## 4. 五个预注册配置

统一协议块（五配置逐字节相同，除声明维度外）：folds = 生产配置五折（2021..2025 五个测试年）；seeds = [42, 7, 2024]；random_match_budget=true（random/gp/trained 臂各自获得 S=floor(§5) 唯一语义评价/fold，gp_seed=1235、tpe_seed=1236、random_seed=1234 为协议既有固定 seed）； Reward v15（`cost_weight` 缺省 → 域默认）；`backtest.turnover_budget` 缺省 → 域默认（p6 §3）；组合/执行/费用 = 生产值（§2.2）。

| # | config_id | 域 | 预注册特征集（可采样口径：域特征 ∩ tier − deprecated） | 执行点 | 基线信号（预注册，方向=registry） | tier_set | 状态（2026-09-06 D7 探针） |
|---|---|---|---|---|---|---|---|
| 1 | r1_medium_a | medium_cross_section | 17（域 31 ∩ A 22 − deprecated 5） | every_10_days / h=10 | MOMENTUM_20, RSQ_60, REVERSAL_60（域默认去掉 Tier C 的 IND_REL_RET_20） | A | D7 PASS，可运行 |
| 2 | r1_medium_ab | medium_cross_section | 19（17 A + 2 B） | every_10_days / h=10 | 同上 | AB | D7 PASS，可运行 |
| 3 | r1_slow_ab | slow_fundamental | 16（1×A + 15×B，无 C，域全集） | every_20_days / h=20 | ROE, PE_TTM, REVENUE_YOY, DIVIDEND_YIELD, MARKET_CAP（域默认） | AB | **D7 FAIL**（income_statement_pit 期间覆盖 86.53% < 90%）→ not_run 待数据任务 |
| 4 | r1_short_a | short_price_volume | 23（域 29 ∩ A 29 − deprecated 6，域全集） | every_5_days / h=5 | REVERSAL_5, TURNOVER, ILLIQ_20, OVERNIGHT_RET, LIMIT_UP_CNT_20（域默认） | A | D7 PASS，可运行 |
| 5 | r1_unified_base | unified（兼容基线） | 65（全部可采样特征） | daily / h=1 | 生产默认 7 信号（REVERSAL_5, RSQ_60, ILLIQ_20, OVERNIGHT_RET, MOMENTUM_20, ROE, TURNOVER） | 无 | **D7 FAIL**（同 #3 income 项 + industry_index lag 15 > 3）→ not_run 待数据任务 |

配置 1/2 是唯一预注册的域内配对对照（差异 = Tier B 两融对 `MARGIN_BALANCE_CHG`+`MARGIN_CROWD_60` 的纳入）；配置 3/4 的 tier 集合是其域全集的自然构成（声明性标注）。运行特征集的单一解析规则（域 ∩ tier − deprecated）由 §8.2 的单一实现承载，`scripts/r1_data_eligibility_check.py` 消费同一函数。**预算计数器唯一**：S 对五配置取同一值（§5），全部行进同一 campaign ledger，DSR/max-t 同时输出配置级与 campaign 级（§9）。

## 5. 唯一语义预算与资源上限

（数值由 2026-09-06 fold-0 工程校准推导，校准属 engineering 证据，仅用于预算标定，禁止升格为研究结论。）

- 每配置每 fold 预算：random S + gp S + tpe S + trained 3×S（S = screening `steps × batch_size`，`batch_size ≥ 256` 为 trained 臂稳定性硬约束）；每配置合计 30S 请求评价；五配置合计 150S（+基线行，不计预算）。
- 校准结果与 S 取值：**TBD（校准运行完成后填入本节并冻结）**。
- 资源上限：campaign 总墙钟 ≤ TBD h；任一配置首折实测行均墙钟 > 投影 2× → 停止后续 seed/配置，campaign 标 FAILED（fail-closed，p10 §4.5 先例）；单行墙钟 > 投影 3× → 测量日志强制解释。
- 验证子窗口（validation_fraction 0.35 / 4 splits）消耗一并计入 consumed，如实在 ledger 与测量日志披露。

## 6. 门禁（每配置运行前置，全部硬性）

1. **G1–G8 生产门**（`ProductionGateRunner.require_production`，formal 模式；G8 新鲜度 lag ≤ 3 开市会话——legacy 库止 2026-09-01，评估日推进将触发 FAIL，如实记录）。
2. **D7 数据资格门**（p17 §12，`check_runspec_data_eligibility`）：以该配置预注册特征集 + 生产 universe（constituents 全集 1630 码）为因子集，PASS 才可运行；探针结果（2026-09-06）见 §4 表；复检在每个配置启动时执行并留档。
3. **身份一致性**：dataset_id、Git commit、版本集合（protocol 26→27 见 §8 / reward 15 / model 3 / grammar 6 / factor_compute 4 / feature_registry 7 / data_tier 2 / research_domain 3 / search_contract 3 / searcher_bench 3 / semantic_cache 1 / execution_spec 2 / portfolio_constructor 1 / rebalance_policy 2）逐行入产物；跨行 dataset_id 不一致 → fail-closed。
4. **tier 纯度**：tier_set 限定的配置，其全部 selected 公式经 `formula_data_tier_report`（单一权威）审计 max_tier ∈ 允许集合；违例 = 运行无效并记录（p14 §5.4.4 模式）。
5. **组合活动门**（AGENTS §8.5）：逐行记录 rebalance_count/order_count/suppressed/average_turnover；接近空仓或仅建仓一次的低换手结果不得宣称"有效"。

## 7. 跨域纪律（TURNOVER_STD20 / LIST_AGE）

两特征原属 short 域（p6 §1.1，Tier A）。R1 不做跨域迁移；若未来需要迁入 medium/slow，必须先由 governance-auditor 起草域版本契约（`RESEARCH_DOMAIN_VERSION` bump + 特征归属修订）并获用户批准——**截至本文批准日无此批准**，故不迁移、不偷用。

## 8. 版本影响表与最小实现（条件前置）

| 版本 | 旧→新 | 理由 | 迁移/拒绝 |
|---|---|---|---|
| PROTOCOL_VERSION | "26"→"27" | 协议获得 additive 的 `protocol.tier_set` 语义与产物字段；v26 产物不携带该字段，按既有规则 legacy（不转换不重写） | v26 及更早产物只读 |
| 其余全部 | 不变 | 配置维度（域/horizon/换手/成本/基线）均为既有已版本化机制的取值，无新语义 | — |

最小实现（先 RED 后 GREEN，同原子提交）：

1. `ProtocolConfig` 增 `tier_set: str | None = None`（合法值 None/"A"/"AB"，其他值配置期拒绝）。
2. `research_domain.restrict_tensor` / `feature_token_ids` 增可选 `tier_set`：域特征 ∩ tier 集合（单一权威 `data_tier.feature_tier` + `vocab.deprecated_names` 排除），单一实现、无第二套映射；`unified` 行为零变化，`tier_set=None` 行为逐字节不变。
3. `run_protocol` 域模式接线：tensor 行置零 + 采样 mask + 基线 ⊆ (域 ∩ tier) 校验 + 产物记录 `tier_set` 与解析后特征清单。
4. 预期 RED 测试：`test_research_domain.py` 增 tier 交集属性测试（可采样集合大小 medium A/AB = 17/19、short = 23，deprecated 与域外行置零、非法 tier_set 值拒绝、unified+None 恒等、artifact 字段断言）；`test_artifact_versions.py` 增 PROTOCOL 钉值 27。
5. `scripts/r1_data_eligibility_check.py`：D7 门只读 CLI（--config、解析 §4 特征集、输出 verdict JSON），消费 §8.2/§8.3 同一函数，不复制规则。

## 9. 测量方案（运行后写入 `docs/p21_r1_measurement_log.md`）

命令、commit、环境、dataset_id、effective config hash、版本集合、S、逐行 requested/consumed/终止/停滞/墙钟/RSS、方向对齐基线先于搜索的 ledger 行序断言、tier 纯度审计、DSR/max-t（配置级 + campaign 级 join via `--trials`）、G1-G8/D7 复检留档、失败与 not_run 原因。产物布局 `data/r1/<config_id>/`；campaign ledger `data/r1/ledger.jsonl`（append-only）。

## 10. 预注册裁决规则（运行前固定，禁止事后改写）

对每个**实际运行**的配置（not_run 的配置记录"未运行+原因"，不参与裁决）：

| 观测 | 裁决 |
|---|---|
| 某域 stitched 最优候选在 ≥2/3 seed 上 OOS active IR > 0，且 campaign 级 DSR > 0.95 与 max-t 95% 显著至少一项成立，且通过组合活动门 | 记"该域存在候选 alpha 信号（research 级）"；下一步 = R2/R3 组合矩阵（引用本文，不宣称 production 就绪） |
| DSR ≤ 0.95 且 max-t 不显著，或 OOS 不为正 | **"未发现可晋级 alpha"（该域）**——合法结论 |
| 全臂 consumed ≪ requested 且停滞 | 首要发现记为"预算瓶颈在提案生成"，方向 = 采样多样性改进 |
| medium A vs AB：AB 在 ≥2/3 seed 上 stitched 指标更优 | 记"Tier B 纳入在该域有正向增量（描述性，n=3 不宣称显著性）"；否则记"无正向增量/不确定" |
| 配置 not_run（门禁 FAIL） | 记录原因与解除条件（数据任务），不填补测 |
| 任何 fail-closed 触发 | campaign FAILED，保留全部已有行，修订本文后方可重跑 |

## 11. 停止条件

1. campaign 总墙钟超 §5 上限或校准偏差熔断（§5）；
2. dataset_id / 版本集合不一致；
3. 任一行 backend_error（行入 ledger，不重试覆盖）；
4. 内存不足（机器 16GB，与并发任务共享）——停止而非降配偷跑；
5. 用户或 captain 指示停止。

## 12. 批准请求清单（提请用户裁决）

1. 批准本契约全文（五配置、预算 S、门禁、裁决、停止条件）；
2. 批准 §8 最小实现（PROTOCOL_VERSION 26→27 + tier_set 接线）作为运行前置；
3. 知悉并接受：配置 3/5 在当前数据状态下的 D7 FAIL → not_run（除非先完成数据回填任务使 income 覆盖 ≥90% 且 SW 指数滞后 ≤3，之后复检 PASS 即可运行）；
4. 知悉：R1 全部结论为 research 级，不构成晋级/production 证据。
