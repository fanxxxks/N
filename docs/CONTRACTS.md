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
| [p17_tushare_pit_data_contract.md](p17_tushare_pit_data_contract.md) | P17 Tushare/PIT 数据底座预注册契约（四层架构 + D4/D5/D6 语义基座） |
| [p18_captain_authority_clause.md](p18_captain_authority_clause.md) | P18 Captain 调度权与批准记录回写义务条款（治理授权） |
| [p19_market_price_semantics_contract.md](p19_market_price_semantics_contract.md) | P19（暂定编号）市场价格语义切换预注册契约（D4：raw 成交 / 复权因子序列） |
| [p20_margin_balance_direction_repreregistration_draft.md](p20_margin_balance_direction_repreregistration_draft.md) | P20 MARGIN_BALANCE_CHG 方向重预注册（authored +1 四轮证伪 → −1；t70 已实施，原草案文件名保留） |
| [p22_industry_momentum_direction_repreregistration_draft.md](p22_industry_momentum_direction_repreregistration_draft.md) | P22 INDUSTRY_MOMENTUM 方向重预注册（authored +1 修复后证伪；t24 路由负结果处置 + p22 注记 REJECTED 维持——12a7ed7 的 −1 翻转经 captain 终裁回退 969a126，方向证实 ≠ 冗余预检通过；IND_REL_* 冗余 caveat 在案） |
| [p23_f1_slow_fundamental_quality_draft.md](p23_f1_slow_fundamental_quality_draft.md) | P23 F1 批次预注册（v5 = 用户亲批四因子 NET_MARGIN_TREND_4Q/LEVERAGE_TREND_4Q/CAPEX_INTENSITY/CFO_TO_DEBT，用户「批准」+ captain 纠正确认；族门通过，候选池准入） |
| [p24_f2_margin_crowding_draft.md](p24_f2_margin_crowding_draft.md) | P24 F2 批次预注册（v2 = 路线图两因子 MARGIN_BALANCE_TO_CIRC_MV/MARGIN_BUY_PRESSURE_5，用户「批准」+ captain 代行签署；族门通过——MARGIN_BALANCE_TO_CIRC_MV 门承载 ΔIC +0.0065 且残差同向，同源共线风险实证排除；候选池准入。索引行补登：F2 实施 commit 0c1f0bb/d1c651d 遗漏 same-commit 登记，governance cc55e71/ec67997 发现后本行补记） |
| [p25_f3_expectation_event_draft.md](p25_f3_expectation_event_draft.md) | P25 F3 批次预注册（v1 = 预期与事件四因子 FORECAST_GROWTH_MIDPOINT/FORECAST_UNCERTAINTY/EXPRESS_VS_FORECAST/BUYBACK_INTENSITY，用户「批准」+ captain 代行签署；event 域首成员 F3-3） |
| [p26_a02_dataset_snapshot_identity_draft.md](p26_a02_dataset_snapshot_identity_draft.md) | P26 A02 可信数据身份与发布边界预注册（**批准状态以契约文件自身标注为准**；批准决策记录见 `reports/alphagpt-ab/00_authority_exception_record.md` §11/§4ter——关闭 F02：查询最新保存 id 与内容确认拆分、双库复合内容身份、发布清单与缓存键/RunSpec 同一权威；governance-author 起草，main @ 1fb1b16） |
| [p27_a03_lifecycle_runcontext_draft.md](p27_a03_lifecycle_runcontext_draft.md) | P27 A03 生命周期强制接线与 RunContext 预注册（APPROVED，captain 依转授批准、用户 §4ter 确认；批准记录见 `reports/alphagpt-ab/00_authority_exception_record.md` §11/§4ter——关闭 F03：正式入口身份必需、五类动作模式与允许状态表、身份先于昂贵计算、受保护 OOS 与 paper 账户绑定；governance-author 起草，main @ 1fb1b16） |
| [p28_a04_feature_metadata_semantic_type_draft.md](p28_a04_feature_metadata_semantic_type_draft.md) | P28 A04 因子元数据与语义类型纠偏预注册（**APPROVED**，captain 依转授批准（授权记录 §11 裁决 2/8 + §4ter 确认）+ 用户 2026-09-14 直接指令「完成A04」；批准记录回写见契约文件头——关闭 F04：F2 融资因子联合可见性与语义类型判定（CROSS_SECTIONAL_SIGNAL）、depends_on 口径统一、FO-01 存量因子/算子契约清单（70 可采样特征 + 39 算子）；governance-author 起草，main @ 1fb1b16） |
| [p29_b01_pit_tradable_data_quality_draft.md](p29_b01_pit_tradable_data_quality_draft.md) | P29 B01 PIT 与可交易数据质量预注册（**APPROVED**，captain 依转授批准（授权记录 §11 裁决 5/7 + 版本影响总表 B01 行）+ 用户 §4ter 条目 4 确认 + 用户 2026-09-15 直接指令「完成B01」；批准记录回写见契约文件头——时间格式导入边界规范化与同构断言（主库 118 行 ISO 历史缺陷已由 T46 修复，B01 只报告不修复）、四时间语义分离、月末快照 as-of、覆盖率分母、逐码逐日可交易缺口、零事件 vs 漏采、分域资格报告四态；governance-author 起草，main @ 1fb1b16） |
| [p26_a02_composite_coverage_erratum.md](p26_a02_composite_coverage_erratum.md) | P26 勘误 1：A02 复合身份覆盖面声明与实现的归一（APPROVED，captain 依转授批准——身份覆盖（完整 v2 manifest 超集）与研究准入（事件域待 F3/B01）显式分离；`NOT_INCLUDED_DOMAINS` → `PENDING_ADMISSION_DOMAINS`；R15 约束身份表范围变更、准入决定不触发；A02 合规双审 B2 的处置） |
| [p30_a05_doctor_eligibility_draft.md](p30_a05_doctor_eligibility_draft.md) | P30 A05 可信 doctor 与运行资格报告预注册（APPROVED，用户 2026-09-14 会话直接指令「完成A05」——六维×四态分层就绪报告（runtime/snapshot_integrity/data_eligibility/research_ready/promotion_ready/paper_ready）、移除 `assert_paper_account_writable("current")` 恒真假阳性、报告绑定配置/快照/模式/时点、O5 get_run_detail provenance 并入；执行模型起草，分支 codex/a05-doctor-eligibility @ a18cac8） |
| [p31_b02_entry_canary_ci_draft.md](p31_b02_entry_canary_ci_draft.md) | P31 B02 真实入口 canary 与 CI 质量预注册（APPROVED，用户 2026-09-15 会话直接指令「完成B02」+ captain 代行签署——小型确定性双库 canary（12 股×40 会话、PIT 成分、公告延迟、复权、交易阻断）经真实配置解析→RunSpec/资格链→最小搜索→回测→产物登记→API 读取全闭环、数据身份失效入口级阻断、未来扰动不影响过去、手工推导金额/日期对账、CI web job 增 npm test/lint 并与 onboarding §9.2 镜像同步；执行模型起草，分支 codex/b01-pit-data-quality @ 88a784e） |
