# 预注册契约索引（单一权威清单）

本文是 `AGENTS.md`「单一权威指针」段所指的契约索引：登记 `docs/` 下全部
预注册契约、治理计划与预注册草案的**存在性与主题**。本文只做指针，不复述
契约内容；各契约的批准状态以文件自身标注为准，本文不作状态裁决。

**登记规则**：新增契约/计划/草案文件（`docs/pN_*.md`）必须在同一 commit
登记进本索引；删除或更名时同 commit 更新本表。测量日志
（`docs/pN_*_measurement_log.md`、`docs/phase*_measurement_log.md`）与审查
报告不进本索引——计划与证据分离（AGENTS §11.1），以 `docs/` 目录为准。

**勘察基线惯例**（[02-INT-09]，IP-16 采纳）：每份契约（含对既有契约的修订）
在问题陈述/证据节必须写明起草时的**勘察基线 commit**（survey baseline，
如「证据基线：main @ `<sha>`」），使契约的可核查范围显式化；后续修订换基线
时在修订记录中写新基线。P8-08/09 契约修订时按本惯例补齐并推广。

## 契约与计划清单（p2 起编号，按编号排列）

| 文件 | 主题 |
|---|---|
| [p2_data_tier_contract.md](p2_data_tier_contract.md) | P2 免费数据可信度分层（Tier A/B/C）契约 |
| [p3_portfolio_contract.md](p3_portfolio_contract.md) | P3 组合、调仓与多周期标签契约 |
| [p4_search_transformer_contract.md](p4_search_transformer_contract.md) | P4 四搜索器统一与 Transformer 改造契约（§3 = searcher_bench） |
| [p5_fullmarket_champion_contract.md](p5_fullmarket_champion_contract.md) | P5 全市场扩池与 Champion 双验证契约 |
| [p6_research_domain_contract.md](p6_research_domain_contract.md) | P6 按预测周期拆分研究域契约 |
| [p7_semantic_types_contract.md](p7_semantic_types_contract.md) | P7-E GP/搜索器语义类型契约 |
| [p7_artifact_schema_contract.md](p7_artifact_schema_contract.md) | P7-C 类型化产物契约（strategy / protocol / backtest / paper state） |
| [p7_maintainability_plan.md](p7_maintainability_plan.md) | P7 可维护性治理主计划（巨型模块拆分 + 类型化产物 + 单一注册表 + 语义类型） |
| [p8_research_lifecycle_contract.md](p8_research_lifecycle_contract.md) | P8 研究生命周期统一契约（预注册） |
| [p9_factor_family_contract.md](p9_factor_family_contract.md) | P9 因子族精简与正交因子族预注册契约 |
| [p10_searcher_fairness_contract.md](p10_searcher_fairness_contract.md) | P10 四搜索器公平对比预注册契约 |
| [p11_reward_v15_contract.md](p11_reward_v15_contract.md) | P11 Reward v15 区分度重构预注册契约（A 线：reward v14→v15） |
| [p12_promotion_enforcement_contract.md](p12_promotion_enforcement_contract.md) | P12 晋级执法预注册契约（B 线：promotion 门禁消费） |
| [p13_fundamental_fields_contract.md](p13_fundamental_fields_contract.md) | P13 fundamental PIT 字段补齐与第⑤族解锁预注册契约（C 线） |
| [p14_search_digest_preregistration.md](p14_search_digest_preregistration.md) | P14 搜索层消化率预注册契约（P1-1 + P1-2，S 线需求基线） |
| [p16_data_freshness_gate_contract.md](p16_data_freshness_gate_contract.md) | P16 数据新鲜度门禁（G8）+ 交易日历回退 fail-closed 预注册契约（IP-05a） |
