# P57 规则：审核 findings 必须逐条落盘（流程固位）

- 生效：2026-09-27（p57 收口包）。适用对象：**今后每一个工作包的审核员**
  （三路审核 / post-run audit / 复核员，任何以"审核"为名产出 finding 的角色）。
- 背景（两轮同型缺口）：
  - WP17：审核 findings 的逐条明细未落盘（报告 §6 自认，只在正文汇总）；
  - WP18：三路审核 17 条 finding 的逐条明细与 must-fix 计数未落盘
    （WP18_REPORT §9.5 自认；工作流运行日志在本会话不可达，在盘的只有处置
    结果 errata E4–E9 与质量门复跑）。
  两轮的处置结果都可查，但**发现本身**（发现了什么、为什么是问题、当时证据
  链接）没有独立载体——复核与回溯只能重信结论，不能重查发现。

## 规则（机械、可执行）

1. 审核员在完成审核的**同一会话**内，把全部 findings 逐条写入
   `reports/wpNN/audit_findings_pNN.md`（NN = 该工作包的测试前缀；一个工作包
   多轮审核则追加小节，不覆盖）。
2. 每条 finding 必须含：编号、严重度、位置（文件:行 或 产物路径）、发现
   描述、证据（命令/读取的产物）、处置（处置人/处置方式/状态）。
3. **未处置条目不得留白**：确不处置的写明理由（standing / deferred +
   归属包）。文件末尾维护汇总计数表。
4. 与 errata 的关系：涉及测量完整性/契约的处置**另按 errata 纪律追加**
   `errata/errata.jsonl`；本文件是 finding 明细载体，不替代 errata。
5. 质量门：工作包报告须在"遗留/审核"节链接本文件；报告只写汇总与链接，
   逐条明细以本文件为准。

## 模板

```markdown
# WPNN 审核 findings 逐条明细（pNN）

- 审核角色：<三路审核 / post-run audit / 复核员>
- 审核日期：YYYY-MM-DD
- 审核对象：<包名/产出范围>

## Findings

| # | 严重度 | 位置 | 发现 | 证据 | 处置 | 状态 |
|---|--------|------|------|------|------|------|
| F1 | blocker/major/minor | path:line 或 产物 | <问题描述> | <命令/产物引用> | <处置方式；不处置则理由> | resolved/standing/deferred(pXX) |

## 汇总

- 总数 N：resolved X / standing Y / deferred Z（deferred 须写归属包）
```

## 本包执行（吃自己的狗粮）

p57 自身按本规则执行：`reports/wp18/audit_findings_p57.md`（本包审核
findings 逐条明细，含声明审计 2 条 needs_annotation 的复核处置）。
