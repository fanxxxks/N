# 基本面低覆盖字段披露与回填触发条件台账（IP-13）

- 性质：**治理台账**——记录披露机制、覆盖快照出处与回填触发条件（"计划/触发"
  内容）；实际覆盖率是数据事实，以 `research_doctor` 的
  `fundamental_coverage` 输出为准，本台账不固化 current 数字（§11.3）。
- 依据：[03-F-09]（中·决策建议）、[05-④]；决策项非爬取项。
- 裁决状态：触发条件最终认可属**用户/族⑤裁决（U7）**，本台账按方案建议口径
  记账；**U7 裁决前不发生任何补爬/数据写**。

## 1. 披露机制（已实现，报告字段非门禁）

`ashare_model/research_doctor.py` 的 `fundamental_coverage` 报告段：

- 字段：`roa`、`debt_ratio`、`dividend_yield` 三项的 finite 计数与覆盖率；
  分母定义内联在输出的 `definition` 字段（有限值=非 NULL 且非 NaN，除以
  `fundamental_pit` 全部行数，镜像 P13 测量日志 §4.1 口径），另附
  `total_rows`、`distinct_ts_code`。
- 边界：`report_only: true`——不产生 finding、不参与门禁、不参与
  healthy 判定；DB 缺失/被锁时输出 `error` 字段而不是让 doctor 失败。
- 只读：与 doctor 其余 gather 一致，`read_only=True` 打开数据库；
  定期披露 = 每次运行 doctor 即刷新（无需 `--output` 落盘）。

## 2. 覆盖率快照出处（引用，不复制为本台账的 current 值）

| 来源 | 时点 | 数值 | 口径 |
| --- | --- | --- | --- |
| 方案 [03-F-09]/[05-④] 审计 | 2026-09-02 树（28bfefb 代） | ~10.1% / 10.1% / 1.4% | 方案审计口径（未随方案给出精确分母定义） |
| P13 回填测量日志（`docs/p13_fundamental_backfill_measurement_log.md`） | P13 收官 | "维持回填前状态" | 同上引用 |
| `research_doctor` 首次披露运行 | 2026-09-03，dataset `b7b4dd4b`，215,951 行 | roa 32.55% / debt_ratio 32.55% / dividend_yield 3.62% | `definition` 内联：有限值/总行 |

**开放标注（交 data lane 澄清）**：方案审计快照与当前 doctor 实测存在明显
差异（10.1%→32.55% 量级）。可能为口径不同（方案分母定义未随方案记录）或
P13 批量端点行携带 `roa/debt_ratio/dividend_yield` 列
（`ashare_data/fundamentals.py` "bulk earnings endpoint carry these columns"）
导致快照后覆盖自然上升。在口径澄清前，以 doctor 输出的内联定义为准；
本差异不构成任何回填授权。

## 3. 回填触发条件（U7，待用户/族⑤最终认可）

**触发条件**：未来某个预注册契约需要 `roa`/`debt_ratio`（族⑤或后续基本面
契约）且其验收要求目标覆盖率 ≥0.9 时，才启动补爬流程；届时以该新契约为唯一
授权来源。

**前置清单（按序，全部满足才可动工）**：

1. **批量端点可行性验证**：先确认批量端点能否供给 `roa`/`debt_ratio`——
   可行则走分钟级批量路径；**禁止直接上逐股 Sina 路径**。
2. **逐股路径根因修复**：若需逐股路径，必须先修复 `'NoneType'` 解析根因
   （既有实测缺陷），再谈覆盖率目标。
3. **新契约必须显式写明**：小时级资源上限；独占 DB 窗口（不与正式运行并发
   写入）；迁移前备份（按 onboarding §6.6 备份策略：`<tag>bak` 命名 + 世代
   记录）；COALESCE 更新语义不清空既有有限值；目标覆盖率 ≥0.9；失败中断
   阈值（如连续 200 次失败自动停，保留失败证据）。

## 4. 非目标

- 本台账不构成爬取/数据写授权；U7 裁决前无任何补爬发生。
- 不修改契约正文（P13 契约保持原样）；不修改 `fundamental_scope.json`
  生成链（IP-16 阶段已裁定 t17 仅做 research_doctor 输出字段披露）。
