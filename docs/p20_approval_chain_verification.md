# P20 登记核验记录（MARGIN_BALANCE_CHG 方向重预注册——批准链 governance 核验）

- 性质：治理登记核验（evidence-only）。p20 契约文件本体在
  codex/factor-semantic-fixes @ `30092eb`（factor-engineer 辖区分支）；
  本记录把其批准链在 governance 侧入册，并固定登记安排。
- 核验人：governance-auditor（AgentTeams tushare-pit-migration t71），
  2026-09-06。
- 关联裁决：p9 §4.1 standing exception（t71，同 commit）——两事项同出
  captain 对 t22 路由 item 3 的最终处置。

## 1. 契约与证据链核验（全部通过）

| 核验项 | 结果 |
|---|---|
| 草案完整性 | ✓ 四轮证伪证据（t2/v4/t18/t22：IC10 −0.0263…−0.0335，ICIR −0.237…−0.414，年度同向 ≤0.08）跨 legacy/v2 两代底座与 t22 修复边界一致；t22 残差 OOS −0.0308 同号 |
| 新方向假设 | ✓ −1（杠杆拥挤/后期追涨经济学），与同族 MARGIN_CROWD_60（authored −1，四轮符号一致、族门 PASS）同向；p9 §5.4 预注册"方向按测量裁决"惯例的落盘 |
| 范围最小性 | ✓ 仅 feature_metadata expected_direction（+1→−1）+ FEATURE_REGISTRY_VERSION 7→8 + docs/feature_registry.md 同 commit 再生；FACTOR_COMPUTE 不变（方向元数据不改变因子值）；公式/数据源/PIT 可见性零改动（t22 已落地的 v2 margin_detail_pit T+1 语义） |
| 防再翻转纪律 | ✓ 批准后禁止按未来测量单方改向（再改须新预注册）；−1 方向若再被证伪 → 负结果/弃用流程（不得二次翻转） |
| 旧 artifact 兼容 | ✓ 不迁移不拒绝（artifact 携带自身 direction 载荷，元数据变更只影响未来治理/搜索解读） |

## 2. 批准链（governance 核验结论：链完整、闭合）

1. **授权基础**：用户 2026-09-04 授权扩展记录（`89dd121` 附录 C——勘误与
   流程批准点由 captain 代行；push/merge/外部状态变更/预算规模变更除外）
   + p18 §5.2。
2. **代行批准**：captain（2026-09-06，p18 §4 回写义务模式）——p20 草案的
   方向重预注册在代行范围内（p9 §5.4 预注册"方向按测量裁决"惯例 + 零
   计算面 + captain t22 路由 item 3 明示）；**实施启动以本核验 + 代行
   批准为据**。
3. **治理审查**：t69/t71 两轮（本文件为登记核验结论；草案四轮证据表已对
   `docs/t22_semantic_fix_measurement_log.md` §5 与
   `docs/t18_factor_retest_measurement_log.md` 抽查核实——MARGIN_BALANCE_
   CHG IC −0.0306/ICIR −0.366"方向仍被证伪 → 方向重预注册候选"逐字在案）。
4. **编号**：p20（p19 之后下一空闲号，p15 空缺不回填惯例）。

## 3. 登记与实施安排

- 契约文件本体（`docs/p20_margin_balance_direction_repreregistration_draft.md`
  @ `30092eb`）在 **factor-engineer 的实施 commit** 中随更名/定稿 +
  **CONTRACTS.md 索引行同 commit 登记**（草案阶段不索引——t42/t49 先例；
  same-commit 规则在批准/实施边界落实）。
- 实施人 = factor-engineer（codex/factor-semantic-fixes，单 commit：
  元数据一行 + FEATURE_REGISTRY 7→8 + RED 断言 + 文档再生 + 索引行）。
- 实施后 RED 证据（expected_direction == −1 断言先红后绿）与
  FEATURE_REGISTRY_VERSION == 8 钉值进 commit 证据。
- 本核验记录不替代 p20 文件头的批准行——批准后由实施 commit 按 p16
  `2f5e18b` 先例在 p20 文件头补批准记录（引用本核验 + 代行批准声明）。

## 4. 未生效声明

p20 在 factor-engineer 实施 commit 落地前为 DRAFT 草案（不索引、不生效、
不授权任何 registry 变更）；本记录仅为治理侧批准链核验与登记安排，未修改
任何契约正文。
