# 精简后特征全集最终审计报告（v4 词表，任务 t5）

- 运行类型：research diagnostic（非晋级证据）
- 日期：2026-09-01；被测实现：commit **781f5e1**（P9 实现）+ **d6a034d**（§7 裁决执行）；分支 codex/p9-factor-families
- 数据：dataset_id `a839ecf2…`（与 t2 一致，未重同步）；窗口 2015-01-05..20260821；1630 股 × 2828 日
- 口径：与 t2 逐项一致（脚本 `audit_run_v4.py`，runtime 1325.8s，一次成功）；基准 = t2 `docs/factor_inventory_audit_20260831/metrics.json`
- 版本组：protocol 25 / reward 14 / model 3 / grammar 4→5（裁决后）/ **factor_compute 1** / feature_registry 4→5（裁决后）/ research_domain 2 / 其余不变
- 原始数值：本目录 `metrics.json`（73 因子全量）；裁决逐项记录：`docs/p9_measurement_log.md`

## 1. 无回归校验（PASS——合并族无信息丢失的实证）

57 个非稀疏特征（t2 全集减 5 个稀疏事件特征）的 IC(h=1/10/20) 与覆盖率与 t2 基线**逐位一致**（容差 1e-9，零失配）。结论：(a) v4 实现对非稀疏路径零影响；(b) 8 个 deprecated 稠密变体（RET_5、VOL_60 等）信息完整保留——legacy 公式语义不受精简影响；(c) 族合并没有丢失任何原有信息。

## 2. F1 修复验证（族③前置覆盖率硬门槛：PASS）

稀疏安全标准化将事件族从"休眠"中恢复：LIMIT_UP_EVENT 非退化天数 **19 → 719**（38×，门槛 ≥400）且每个日历年 ≥20 天；LIMIT_UP_CNT_20 537→2566、LIMIT_BREAK 122→2003；新特征 LIMIT_UP_CNT_5 1683、LIMIT_BREAK_5 2797。LIMIT_DOWN_EVENT/LIMIT_DOWN_STREAK 392 天、最少年度 4 天（跌停事件集中于暴跌年份）——年度覆盖不足如实记录为特征级 caveat。

## 3. 契约 §7 族级裁决（预注册规则，逐项证据见测量日志）

| 族 | 裁决 | 关键数值（ΔIC_OOS@h=10 / 残差 IC / 方向） |
|---|---|---|
| ① 行业残差化动量/反转 | **PASS** | IND_REL_RET_60 +0.0271 / -0.0201 ✓；IND_REL_RET_120 +0.0091 / -0.0108 ✓ |
| ② 流动性冲击/萎缩/量价背离 | **PASS** | PV_DIV_20 +0.0160 / -0.0268 ✓；LIQ_SHOCK_20 +0.0054 ✓（边际）；VOLUME_SHRINK_5_20 +0.0001 → 特征级负结果 |
| ③ 涨跌停事件条件 | **负结果**（合法） | 三成员 ΔIC 均 < +0.005；预注册延续方向被证伪（LIMIT_UP_CNT_5 实测 IC -0.0151，呈反转）→ 三特征 promotion_allowed=False（保留计算与采样） |
| ④ 横截面拥挤度 | **PASS** | MARGIN_CROWD_60 +0.0395 / -0.0287 ✓；CROWD_AMOUNT_60 +0.0293 ✓；CROWD_TURNOVER_60 +0.0202 ✓ |

## 4. 相关性与二次精简（0.9 阈值，全部按预注册执行）

- **LIMIT_STREAK 条件弃用触发**：修复后 |ρ(LIMIT_UP_EVENT, LIMIT_STREAK)| = 0.980 → 弃用（LIMIT_UP_EVENT 为保留原子）。
- **LIQ_SHOCK_20 弃用**：与 VOLUME_IMPACT |ρ|=0.968（其族门槛边际仅 +0.0054）。
- **CROWD_TURNOVER_60 弃用**：与 CROWD_AMOUNT_60 |ρ|=0.971（后者 ΔIC/ICIR 更强）。
- LIMIT_DOWN_STREAK ~ LIMIT_DOWN_EVENT 0.985：同属负结果族③，保留计算不再精简。

## 5. 裁决后的有效词表（baseline）

- 词表 73 名；deprecated **12**（NORTHBOUND_CHG + 8 窗口变体 + LIMIT_STREAK + LIQ_SHOCK_20 + CROWD_TURNOVER_60）；可采样 61；可晋级 **58**。
- 族③三名成员（LIMIT_UP_CNT_5、LIMIT_DOWN_STREAK、LIMIT_BREAK_5）可采样但 promotion_allowed=False（负结果，保留供后续研究观察）。
- 本报告与 `metrics.json` 取代 t2 基线（docs/factor_inventory_audit_20260831/）成为 **current baseline**；t6 四搜索器对比的词表与候选过滤应以本基线为准。

## 6. 边界

全部测量为 dev/validation 数据上的研究诊断（无已声明 regime/锁定 holdout），不构成任何 alpha 或晋级结论；负结果（族③、VOLUME_SHRINK_5_20）为合法结论并已按预注册规则处置。未 push/未合并；仅本地分支提交（781f5e1、d6a034d）。
