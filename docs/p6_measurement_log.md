# P6 测量日志：按预测周期拆分研究域

状态：实施完成。本日志只记录事实（命令、计数、决策点证据、样例），不
放宽 `docs/p6_research_domain_contract.md` 的任何裁决规则。

## 1. 改动前基线（main @ 02abf08 之后，feat 分支建立前）

- 命令：`python -m pytest -q`（无 pytest-timeout 插件）
- 结果：**1086 passed, 5 skipped**（5 个 skip 全部为
  `torch.cuda.is_available()` 环境条件，本机无 CUDA，改动前后一致），
  618 warnings，墙钟 625.70s（10:25）
- 原始输出：`logs/pytest_p6_baseline.txt`（logs/ 已 gitignore，不提交）

## 2. 中间回归（P6 主接线完成后、收尾修正前）

- 命令：`python -m pytest -q`
- 结果：1100 passed, 3 failed, 5 skipped（672.33s）
- 3 个失败及处置（全部为契约修订驱动的测试同步，非实现缺陷）：
  1. `test_baseline_harness.py::test_semantic_budget_skips_degenerate_and_canonical_duplicates`
     / `test_semantic_budget_dedups_equivalent_classes` —— monkeypatch 的
     `sample_random_formulas` 假实现未接收新增关键字参数 `feature_ids`
     （P6 §4.2）；按"不改变语义的测试重构"白名单补 `**kwargs`，断言不变。
  2. `test_p4_search_contract.py::test_p4_semantic_versions_are_pinned_by_contract`
     —— 版本钉定 `PROTOCOL_VERSION == "23"`；P6 §5 提升为 24（需求变更
     路径：先改契约、bump 版本、再同步改测试），断言强度不变。

## 3. 最终全量回归（最终代码状态）

- 命令：`python -m pytest -q`
- 结果：**1107 passed, 5 skipped**（墙钟 612.19s，10:12）；基线 1086 →
  净增 21 个测试（test_research_domain.py 14 个、rebalance 频率新增 3 个、
  artifact_versions 域字段 1 个、参数化扩展与白名单修订），0 新增
  skip/xfail；5 个 skip 均为 CUDA 环境条件，与基线一致
- 附加验证：`python -m compileall -q ashare_data ashare_model
  ashare_portfolio ashare_trading scripts webapi`（exit 0）；
  `git diff --check`（exit 0）
- 原始输出：`logs/pytest_p6_final.txt`

## 4. 域划分统计（契约 §1.1）

- `FEATURE_NAMES` 共 62 个成员；活跃特征 61 个全量划分：
  - short_price_volume：24（价格 4、成交 9、涨跌停 5、日内/隔夜 2、
    流动性 2、微结构 2）
  - medium_cross_section：25（动量/反转/锚定 6、波动/分布 5、风险 3、
    技术 5、行业相对 4、外部横截面 2）
  - slow_fundamental：12（估值 3、质量 5、增长 2、规模 1、股息 1）
- 废弃特征 NORTHBOUND_CHG 不归属任何域（域模式搜索空间天然不含它）。

## 5. 频率日历样例（契约 §2，固定 fixture）

`tests/test_rebalance_policy.py` 的 25 会话 fixture
（2024-01-02 .. 2024-02-08）：

| frequency | horizon | 信号索引 |
|---|---|---|
| daily | 1 | 0..24 |
| weekly | 1 | 2, 5, 10, 15, 20, 24 |
| every_5_days | 5 | 0, 5, 10, 15, 20 |
| every_10_days | 10 | 0, 10, 20 |
| every_20_days | 20 | 0, 20 |
| monthly | 1 | 18（20240131）, 24（20240208） |

- `(monthly, 20)` 在该 fixture 上抛 `ValueError`（相邻月间隔 6 < 20）；
  2024 全年工作日轴上 `(monthly, 20)` 可解析且最小信号间隔 = 20。
- 真实 A 股日历的春节二月仅约 15 个交易日，故慢域默认执行点为日历无关的
  `(every_20_days, 20)`（用户裁决，见 §6）。

## 6. 决策点记录

- **2026-08-30**：实现中发现契约初稿"慢域默认 (monthly, 20)"与 P3 非重叠
  标签约束在真实 A 股日历上冲突（春节二月约 15 个交易日 < horizon 20）。
  按 P5 惯例回报用户；用户裁决：新增 `every_20_days` 频率，慢域默认
  `(every_20_days, 20)`，`monthly` 保留为日历支持时的可选执行点（运行时
  fail-fast，不静默抽稀）。契约 §1/§2/§3/§6 与测试同步修订，修订过程在
  契约文档中留痕。

## 7. 版本与 artifact 样例

- 提升：`PROTOCOL_VERSION` 23→24、`REBALANCE_POLICY_VERSION` 1→2；
  新增 `RESEARCH_DOMAIN_VERSION = 1`；不 bump：REWARD_VERSION=14、
  EXECUTION_SPEC_VERSION=2、DATA_TIER_VERSION=1、GRAMMAR_VERSION=2、
  MODEL_VERSION=3。
- protocol artifact 新增字段样例：
  `"research_domain": "short_price_volume"`、
  `"research_domain_version": 1`（unified 运行记录 `"unified"`）。
- 域模式 window_id 样例：`fold:2020-12-31:2021-12-31:frequency:every_20_days:horizon:20:seed:42:domain:slow_fundamental`；
  unified 保持 pre-P6 字节不变。

## 8. 命令与 commit

- 分支：`feat/horizon-split-research-domains`（基于 main）
- 提交序列（一个 commit 一件事，全部本地测试通过后合并）：
  - `20565d5`：feat：新增 every_20_days 与 monthly 调仓频率（P6 §2），
    含 P3 契约频率表修订、p3_measurement 审计扩展与对应测试；
  - `7e9d827`：feat：按预测周期拆分研究域（P6），含研究域注册表、
    配置默认值、搜索空间限制、协议/artifact 接线、P6 契约与全部测试；
  - `<commit3>`：docs：P6 测量日志（本文）。
- 合并：全部测试本地通过后 fast-forward 合并回 main 并推送 origin。
